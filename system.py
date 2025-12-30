from fastapi import FastAPI
from pydantic import BaseModel
#import pandas as pd
#import faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor
#import myCode
#import YourCode

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=8)

users_df = None
items_df = None
svd_model = None
faiss_index = None
user_embeddings = None



# here is what the recommender needs to get from the frontend
class RecommendRequest(BaseModel):
    user_id: int
    num_user_products: int
    k: int = 10



@app.on_event("startup")
def load_models():
    global users_df, items_df, svd_model, faiss_index, user_embeddings

    #here i will load all pandas dataframes, my embedding indexes etc. so they are in memory and doesnt load everytime
    #someone wants to get a recommendation




def svd_recommend(user_id: int, k: int):
    #here i will import your code
    return 0 #top_items.tolist()


def embedding_recommend(user_id: int, k: int):
    # here i will import my code
    return 1 #indices[0].tolist()


def recommend_sync(user_id: int, num_user_products: int, k: int):
    # here it checks what to use
    if num_user_products >= 20:
        return svd_recommend(user_id, k)
    else:
        return embedding_recommend(user_id, k)


# here is your api endpoint to connect to
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        recommend_sync,
        req.user_id,
        req.num_user_products,
        req.k,
    )
    return {"user_id": req.user_id, "recommendations": result}

