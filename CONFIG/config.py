import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(BASE_DIR, "../data")
    INTERACTIONS_PATH = os.path.join(DATA_DIR, "user_interactions_all.parquet")
    USER_STATS_PATH = os.path.join(DATA_DIR, "user_stats.parquet")
    ITEMS_META_PATH = os.path.join(DATA_DIR, "items_meta.parquet")

    MODELS_DIR = os.path.join(BASE_DIR, "../models")
    SVD_PATH = os.path.join(MODELS_DIR, "svd.pkl")
    SVDPP_PATH = os.path.join(MODELS_DIR, "svdpp.pkl")
    KNN_PATH = os.path.join(MODELS_DIR, "knn.pkl")
    ITEM_INDEX_PATH = os.path.join(MODELS_DIR, "item_index.pkl")
    ITEM_MAPPING_PATH = os.path.join(MODELS_DIR, "product_mappings.pkl")

    HYBRID_WEIGHTS = {"svd": 0.3, "svdpp": 0.5, "knn": 0.2 }

    CF_THRESHOLD = 20
    POPULAR_MIN_COUNT = 10
    MAX_WORKERS = 8
    CACHE_SIZE = 10_000
    COLD_START_SAMPLE = 200000