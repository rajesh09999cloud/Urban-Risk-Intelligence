from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import os
import gdown

app = FastAPI(title="Urban Risk Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILES = {
    "models/cell_features.csv":       "1BmKFFzz5yE54bBO4nvE6_eOSTDPndQFo",
    "models/crime_features_v3.pkl":   "1lid29NBoOwFEehWyvK6epShprO-gjleo",
    "models/crime_model_v3.pkl":      "1upbWKTOf-zBxApa_-XP_6rPG1wRauwWJ",
    "models/daily_cell_features.csv": "157JrSlTvvqxEzxQa-0bmAbb9vE6_Jb1a",
}

def download_models():
    os.makedirs("models", exist_ok=True)
    for path, file_id in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
            print(f"Done: {path}")
        else:
            print(f"Already exists: {path}")

@app.on_event("startup")
async def load_models():
    download_models()
    print("Loading models into memory...")
    app.state.model          = pickle.load(open("models/crime_model_v3.pkl", "rb"))
    app.state.features       = pickle.load(open("models/crime_features_v3.pkl", "rb"))
    app.state.cell_features  = pd.read_csv("models/cell_features.csv")
    app.state.daily_features = pd.read_csv("models/daily_cell_features.csv")
    print("All models loaded!")

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