import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from CONFIG.config import Config
from files.data_loader import data_store
from files.recommenders import RecommenderEngine

app = FastAPI(
    title="Hybrid Recommendation System",
    description="Amazon product recommendations",
    version="2.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],allow_headers=["*"],)
app.mount("/static", StaticFiles(directory="static"), name="static")
executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

@app.on_event("startup")
def startup_event():
    try:
        data_store.load_all()
    except Exception as e:
        raise RuntimeError("Failed")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": all([data_store.best_svd,data_store.svdpp,data_store.knn])
    }

@app.get("/api/history/{user_id}")
def get_history(user_id: str):
    return {
        "user_id": user_id,
        "history": RecommenderEngine.get_user_history(user_id),
    }


@app.get("/api/random/active")
def get_random_active():
    return {"user_id": data_store.get_random_user(cf_eligible=True)}

@app.get("/api/random/cold")
def get_random_cold():
    return {"user_id": data_store.get_random_user(cf_eligible=False)}

@app.get("/api/user/{user_id}/status", response_model=UserStatusResponse)
def get_user_status(user_id: str):
    stats = data_store.get_user_stats(user_id)
    if stats is None:
        return UserStatusResponse(exists=False)
    return UserStatusResponse(exists=True, interaction_count=int(stats["interaction_count"]),
                              cf_eligible=bool(stats["cf_eligible"]),)

@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    loop = asyncio.get_running_loop()
    history = RecommenderEngine.get_user_history(req.user_id)
    cf_eligible = data_store.is_cf_eligible(req.user_id)

    if cf_eligible:
        recs = await loop.run_in_executor(executor, RecommenderEngine.get_recommendations,req.user_id,
                                          req.k)
    else:
        recs = RecommenderEngine.get_recommendations(req.user_id, req.k)

    return RecommendResponse(user_id=req.user_id, cf_eligible=cf_eligible, history=history,
                             recommendations=recs)
