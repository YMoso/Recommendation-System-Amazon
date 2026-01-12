import pandas as pd
import joblib
import faiss
import pickle
from CONFIG.config import Config


class DataStore:
    def __init__(self):
        self.interactions_df = None
        self.user_stats_df = None
        self.items_meta_df = None
        self.best_svd = None
        self.svdpp = None
        self.knn = None

        self.product_index = None
        self.user_index = None
        self.item_mapping_path = None
        self.item_id_to_idx = None
        self.idx_to_item_id = None
        self.embedding_dim = None
        self.user_ids = None

    def load_all(self):
        self.interactions_df = pd.read_parquet(Config.INTERACTIONS_PATH)
        self.user_stats_df = pd.read_parquet(Config.USER_STATS_PATH)
        self.items_meta_df = pd.read_parquet(Config.ITEMS_META_PATH)
        self.best_svd = joblib.load(Config.SVD_PATH)
        self.svdpp = joblib.load(Config.SVDPP_PATH)
        self.knn = joblib.load(Config.KNN_PATH)

        self.product_index = faiss.read_index(Config.PRODUCT_INDEX)
        self.user_index = faiss.read_index(Config.USER_INDEX)
        self.item_mapping_path = Config.PRODUCT_MAPPING
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        self.embedding_dim = None
        self.user_ids = [p.stem for p in Config.USER_EMBEDDINGS_PATH.glob("*.npy")]

        with open(self.item_mapping_path, "rb") as f:
            data = pickle.load(f)
            self.item_id_to_idx = data["product_id_to_idx"]
            self.idx_to_item_id = data["idx_to_product_id"]
            self.embedding_dim = data["embedding_dim"]


    def get_user_stats(self, user_id):
        row = self.user_stats_df[self.user_stats_df["user_id"] == user_id]
        if not row.empty:
            return row.iloc[0]
        else:
            return None

    def is_cf_eligible(self, user_id):
        stats = self.get_user_stats(user_id)
        return stats is not None and bool(stats["cf_eligible"])

    def get_random_user(self, cf_eligible):
        _user = self.user_stats_df[self.user_stats_df["cf_eligible"] == cf_eligible]
        return _user.sample(1).iloc[0]["user_id"]


data_store = DataStore()