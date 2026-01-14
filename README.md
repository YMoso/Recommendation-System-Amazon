# Hybrid Recommendation System

This project implements a hybrid recommendation system for product recommendation based on user interaction data.  
It combines collaborative filtering methods with an embedding-based approach to handle both active users and cold-start users.

## System Overview

The system is designed around two main recommendation strategies:

- **Collaborative Filtering (CF)** for users with sufficient interaction history
- **Embedding-based recommendations** for users with limited or no interaction history

A FastAPI backend exposes the recommendation logic via REST endpoints, and a lightweight frontend is provided to visualize results.

## Recommendation Strategy

User eligibility is determined by the number of historical interactions:

- **Users with 20 or more interactions**
  - Recommendations are generated using a hybrid collaborative filtering model
  - The hybrid score is computed as an average of:
    - SVD
    - SVD++
    - KNNBaseline

- **Users with fewer than 20 interactions**
  - Recommendations are generated using item embeddings
  - Similarity is computed using cosine similarity between item vectors

- **Fallback**
  - If embeddings are unavailable, popular items can be used as a fallback strategy

## Architecture

The system consists of three main layers:

1. **Data Layer**
   - Parquet files containing:
     - User–item interactions
     - User statistics (interaction counts, eligibility)
     - Item metadata (titles, categories)
   - Serialized machine learning models (`.pkl`)

2. **Model Layer**
   - Collaborative filtering models trained offline
   - Embedding-based item similarity model for cold-start handling

3. **API Layer**
   - FastAPI application serving recommendations and user data
   - JSON-based request and response schemas

## Project Structure

main.py  
FastAPI application entry point

files/  
- recommenders.py – recommendation logic (CF and cold-start)
- data_loader.py – loading datasets and trained models
- config.py – configuration and paths
- models.py – Pydantic schemas

models/  
Serialized collaborative filtering models

data/  
Parquet datasets (interactions, user stats, item metadata)

static/  
Frontend files (HTML, CSS, JavaScript)

## Key Characteristics

- Hybrid recommendation approach
- Automatic handling of cold-start users
- Separation between training (offline) and inference (online)
- Modular design allowing easy extension or replacement of models