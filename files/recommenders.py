from functools import lru_cache
import pandas as pd

from models import RecommendationItem, HistoryItem
from files.data_loader import data_store
from CONFIG.config import Config


class RecommenderEngine:
    @staticmethod
    def get_user_history(user_id, limit= 100):
        _user = data_store.interactions_df[data_store.interactions_df["user_id"] == user_id]
        rows = (_user.head(limit))
        if rows.empty:
            return []
        merged = rows.merge(data_store.items_meta_df, on="parent_asin", how="left")

        return [HistoryItem(asin=row["parent_asin"],title=row["title"] if pd.notna(row["title"]) else "Unknown",
                            rating=row["rating"], category=row.get("main_category") if pd.notna(row.get("main_category"))
            else None) for _, row in merged.iterrows()]

    @staticmethod
    @lru_cache(maxsize=Config.CACHE_SIZE)
    def recommend_hybrid_cf(user_id, k):
        rated = set(data_store.interactions_df[data_store.interactions_df["user_id"] == user_id]["parent_asin"])
        weights = Config.HYBRID_WEIGHTS
        scores = []
        meta = data_store.items_meta_df.set_index("parent_asin")
        for asin in meta.index:
            if asin in rated:
                continue
            try:
                s1 = data_store.best_svd.predict(user_id, asin).est
                s2 = data_store.svdpp.predict(user_id, asin).est
                s3 = data_store.knn.predict(user_id, asin).est
            except Exception:
                continue
            score = (weights["svd"] * s1+ weights["svdpp"] * s2+ weights["knn"] * s3)
            scores.append((asin, score))
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        results = []
        for asin, score in top:
            row = meta.loc[asin]
            results.append(
                RecommendationItem(asin=asin, title=row["title"] if pd.notna(row["title"]) else "Unknown",category=row.get("main_category"),
                                   score=float(score), model="hybrid_cf"))
        return tuple(results)

    @staticmethod
    def recommend_embedding(user_id, k):
        #its just some example, you might to have to change it as you want to :)
        return [
            RecommendationItem(asin="1234",title="Embedding recommendation",category=None,
                               score=5,model="embedding",)]


    @staticmethod
    def recommend_popular(k):
        meta = data_store.items_meta_df.set_index("parent_asin")

        top = (data_store.interactions_df.groupby("parent_asin").agg(avg_rating=("rating", "mean"),
                cnt=("rating", "count")).query(f"cnt >= {Config.POPULAR_MIN_COUNT}").sort_values("avg_rating", ascending=False).head(k))

        results = []
        for asin, row in top.iterrows():
            meta_row = meta.loc[asin] if asin in meta.index else None

            results.append(
                RecommendationItem(asin=asin,title=meta_row["title"] if meta_row is not None else "Unknown",
                                   category=meta_row.get("main_category") if meta_row is not None
                    else None,score=float(row["avg_rating"]),model="popular",))

        return results

    @staticmethod
    def get_recommendations(user_id, k):
        if data_store.is_cf_eligible(user_id):
            return list(RecommenderEngine.recommend_hybrid_cf(user_id, k))
        else:
            recs = RecommenderEngine.recommend_embedding(user_id, k)
            if recs:
                return recs
            else:
                return RecommenderEngine.recommend_popular(k)