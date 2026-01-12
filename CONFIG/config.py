import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(BASE_DIR, "../data")
    INTERACTIONS_PATH = os.path.join(DATA_DIR, "user_interactions_all.parquet")
    USER_STATS_PATH = os.path.join(DATA_DIR, "user_stats.parquet")
    ITEMS_META_PATH = os.path.join(DATA_DIR, "items_meta.parquet")
    CD_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "cd_vinyl/embeddings_by_product_id")
    MOVIES_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "movies_tv/embeddings_by_product_id")
    SOFTWARE_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "software/embeddings_by_product_id")
    GAMES_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "video_games/embeddings_by_product_id")
    USER_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "user_embeddings_lt20")

    MODELS_DIR = os.path.join(BASE_DIR, "../models")
    SVD_PATH = os.path.join(MODELS_DIR, "svd.pkl")
    SVDPP_PATH = os.path.join(MODELS_DIR, "svdpp.pkl")
    KNN_PATH = os.path.join(MODELS_DIR, "knn.pkl")
    PRODUCT_INDEX = os.path.join(MODELS_DIR, "product_index.faiss")
    PRODUCT_MAPPING = os.path.join(MODELS_DIR, "product_mappings.pkl")
    USER_INDEX = os.path.join(MODELS_DIR, "user_index.faiss")

    HYBRID_WEIGHTS = {"svd": 0.3, "svdpp": 0.5, "knn": 0.2 }

    CF_THRESHOLD = 20
    POPULAR_MIN_COUNT = 10
    MAX_WORKERS = 8
    CACHE_SIZE = 10_000