from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import os

app = FastAPI(
    title="Urban Risk Intelligence API",
    description="Predicts crime risk for Chicago",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    print("Loading model...")
    app.state.model = pickle.load(open("models/crime_model_v3.pkl", "rb"))
    app.state.features = pickle.load(open("models/crime_features_v3.pkl", "rb"))
    app.state.cell_features = pd.read_csv("models/cell_features.csv")
    app.state.daily_features = pd.read_csv("models/daily_cell_features.csv")
    print("Model loaded successfully!")

@app.get("/")
def root():
    return {"message": "Urban Risk Intelligence API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "crime_v3"}

from api.routers import predict, heatmap, incidents
app.include_router(predict.router,   prefix="/api/v1")
app.include_router(heatmap.router,   prefix="/api/v1")
app.include_router(incidents.router, prefix="/api/v1")