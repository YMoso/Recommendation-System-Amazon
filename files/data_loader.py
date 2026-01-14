import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from CONFIG.config import Config


class DataStore:
    def __init__(self):
        self.interactions_df = None
        self.user_stats_df = None
        self.items_meta_df = None
        self.best_svd = None
        self.svdpp = None
        self.knn = None
        self.embeddings = None
        self.nn = None
        self.idx_to_item_id = {}
        self.meta_lookup = None

    def load_all(self):
        self._load_core_data()
        self._build_meta_lookup()
        self._load_cf_models()
        self._load_cold_start_embeddings()

    def _load_core_data(self):
        self.interactions_df = pd.read_parquet(Config.INTERACTIONS_PATH)
        self.user_stats_df = pd.read_parquet(Config.USER_STATS_PATH)
        self.items_meta_df = pd.read_parquet(Config.ITEMS_META_PATH)


    def _build_meta_lookup(self):
        cat_col = None
        for c in ["category", "main_category", "categories"]:
            if c in self.items_meta_df.columns:
                cat_col = c
                break

        cols = ["parent_asin", "title"]
        if cat_col:
            cols.append(cat_col)

        tmp = self.items_meta_df[cols].copy()

        if cat_col and cat_col != "category":
            tmp = tmp.rename(columns={cat_col: "category"})
        elif not cat_col:
            tmp["category"] = None

        tmp["category"] = tmp["category"].apply(self._clean_category)

        self.meta_lookup = (
            tmp.drop_duplicates("parent_asin")
            .set_index("parent_asin")
        )


    @staticmethod
    def _clean_category(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, list):
            return x[0] if x else None

        s = str(x)
        if s.startswith("[") and s.endswith("]"):
            s = s.strip("[]").strip().strip("'").strip('"')
            return s.split(",")[0].strip().strip("'").strip('"')
        return s

    def _load_cf_models(self):
        self.best_svd = joblib.load(Config.SVD_PATH)
        self.svdpp = joblib.load(Config.SVDPP_PATH)
        self.knn = joblib.load(Config.KNN_PATH)

    def _load_cold_start_embeddings(self):

        with open(Config.ITEM_INDEX_PATH, "rb") as f:
            data = pickle.load(f)

        all_embeddings = data["embeddings"]
        total = all_embeddings.shape[0]

        sample_size = min(Config.COLD_START_SAMPLE, total)
        sample_idx = np.random.choice(total, size=sample_size, replace=False)

        self.embeddings = normalize(
            all_embeddings[sample_idx].astype("float32")
        )

        with open(Config.ITEM_MAPPING_PATH, "rb") as f:
            maps = pickle.load(f)

        self.idx_to_item_id = {
            i: maps["idx_to_product_id"][int(sample_idx[i])]
            for i in range(len(sample_idx))
        }

        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.embeddings)

    def is_cf_eligible(self, user_id: str) -> bool:
        row = self.user_stats_df[self.user_stats_df["user_id"] == user_id]
        return not row.empty and bool(row.iloc[0]["cf_eligible"])

    def get_random_user(self, cf_eligible: bool) -> str:
        df = self.user_stats_df[
            self.user_stats_df["cf_eligible"] == cf_eligible
        ]
        if df.empty:
            raise ValueError("No users found for given eligibility")
        return df.sample(1).iloc[0]["user_id"]

data_store = DataStore()