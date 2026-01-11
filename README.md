# Hybrid Recommendation System 

Right now, the backend and models are working:
- Trained KNNBaseline, SVD, and SVD++ in a Jupyter notebook
- Combined them into a hybrid collaborative filtering model
- Exported trained models and cleaned datasets
- Built a FastAPI backend
- Added a simple frontend (HTML / CSS / JS) to visualize results


## Recommendation Logic

- Users with 20 or more interactions  
  → Hybrid collaborative filtering (SVD + SVD++ + KNN)

- Users with fewer than 20 interactions  
  → Embedding-based recommendations

- If embeddings are unavailable  
  → Popular-items fallback


## Project Structure
main.py              # FastAPI application 

In files:

recommenders.py       # Recommendation logic (hybrid CF, embeddings) 

data_loader.py       # Loads parquet data + trained models 

config.py            # Paths, thresholds, model weights  

models/              # Saved .pkl models (SVD, SVD++, KNN)  

data/                # Parquet files (interactions, stats, metadata)

static/         # Frontend (HTML, CSS, JS)  



## What You Should Focus On

- Plug in your embedding-based recommender  
  Replace recommend_embedding() in recommenders.py

- Improve frontend UI / UX  (if you want to)
  Layout, styling, interactions (no logic changes)

- Optional:
  - Optimize (caching, i added some basic cache but if you want to upgrade it)
  - If you have some ideas you are free to go

