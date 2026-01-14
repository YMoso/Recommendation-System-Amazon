from typing import List
from files.models import RecommendationItem, HistoryItem
from files.data_loader import data_store
import numpy as np

class RecommenderEngine:
    @staticmethod
    def get_user_history(user_id: str, limit= 100):
        df = data_store.interactions_df

        user_df = df[df["user_id"] == user_id]

        if user_df.empty:
            return []

        for col in ["timestamp", "unix_review_time", "review_time"]:
            if col in user_df.columns:
                user_df = user_df.sort_values(col, ascending=False)
                break

        user_df = user_df.head(limit)

        history = []

        for _, row in user_df.iterrows():
            asin = row["parent_asin"]

            if asin in data_store.meta_lookup.index:
                meta = data_store.meta_lookup.loc[asin]
                title = meta.get("title", "Unknown")
                category = meta.get("category")
            else:
                title = "Unknown"
                category = None

            history.append(
                HistoryItem(
                    asin=asin,
                    title=title,
                    rating=float(row.get("rating", 0)),
                    category=category
                )
            )

        return history

    @staticmethod
    def recommend_cf(user_id: str, k: int) -> List[RecommendationItem]:
        rated = set(
            data_store.interactions_df[
                data_store.interactions_df["user_id"] == user_id
            ]["parent_asin"]
        )

        scores = []
        for asin in data_store.items_meta_df["parent_asin"].unique():
            if asin in rated:
                continue

            score = (
                data_store.best_svd.predict(user_id, asin).est +
                data_store.svdpp.predict(user_id, asin).est +
                data_store.knn.predict(user_id, asin).est
            ) / 3.0

            scores.append((asin, score))

        top = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        return [
            RecommendationItem(
                asin=a,
                title=data_store.meta_lookup.loc[a]["title"],
                score=float(s),
                model="hybrid_cf",
                category=data_store.meta_lookup.loc[a].get("category")
            )
            for a, s in top
        ]

    @staticmethod
    def recommend_cold_start(k: int = 5):
        nn = data_store.nn
        embeddings = data_store.embeddings
        idx_to_item = data_store.idx_to_item_id
        meta = data_store.meta_lookup

        if nn is None or embeddings is None:
            return []

        seed_idx = np.random.randint(0, embeddings.shape[0])
        seed_vec = embeddings[seed_idx].reshape(1, -1)

        distances, indices = nn.kneighbors(
            seed_vec,
            n_neighbors=min(k + 1, embeddings.shape[0])
        )

        recommendations = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == seed_idx:
                continue

            asin = idx_to_item[idx]
            similarity = round(1.0 - float(dist), 3)

            if asin in meta.index:
                row = meta.loc[asin]
                title = row.get("title", "Unknown")
                category = row.get("category")
            else:
                title = "Unknown"
                category = None

            recommendations.append(
                RecommendationItem(
                    asin=asin,
                    title=title,
                    score=similarity,
                    model="cold_start",
                    category=category
                )
            )

            if len(recommendations) >= k:
                break

        return recommendations

    @staticmethod
    def get_recommendations(user_id: str, k: int):
        if data_store.is_cf_eligible(user_id):
            return RecommenderEngine.recommend_cf(user_id, k)
        else:
            return RecommenderEngine.recommend_cold_start(k)