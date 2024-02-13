# fichier: feature_engineering_api.py
from fastapi import FastAPI, HTTPException
import requests
import uvicorn

from src.script_fengineering import extract_features
from src.clean_url import clean_url

app = FastAPI()

@app.post("/prepare/")
async def prepare_data(url: str):
    
    features = extract_features(url)

    # Ensuite, envoyez les caractéristiques à l'API du modèle pour la prédiction
    model_api_url = "http://0.0.0.0:8050/predict/"  # Changez l'URL selon votre configuration
    response = requests.post(model_api_url, json=features)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail="Model API request failed")
