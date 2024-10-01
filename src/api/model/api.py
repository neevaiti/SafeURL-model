from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from fastapi.encoders import jsonable_encoder

import pickle
import pandas as pd
import numpy as np
import os
from io import StringIO
from dotenv import load_dotenv

import asyncpg
import logging
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'application FastAPI
app = FastAPI()

API_MODEL_KEY = os.getenv("API_MODEL_KEY")

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"API_MODEL_KEY loaded: {'*' * len(API_MODEL_KEY)}")

# Définir le chemin du modèle relatif au fichier actuel
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Vérifier si le fichier model.pkl existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

# Charger le modèle
with open(model_path, 'rb') as file:
    model = pickle.load(file)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Connexion à la base de données
async def get_db_connection():
    return await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME")
    )

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_MODEL_KEY:
            raise HTTPException(status_code=401, detail="Accès non autorisé")
    return await call_next(request)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    logger.debug(f"Dependency - API Key: {'*' * len(api_key_header) if api_key_header else 'None'}")
    if api_key_header == API_MODEL_KEY:
        return api_key_header
    logger.warning("Dependency - Invalid API key")
    raise HTTPException(status_code=401, detail="Clé API invalide")

async def log_to_database(conn, level, message):
    try:
        query = "INSERT INTO Log (level, message) VALUES ($1, $2)"
        await conn.execute(query, level, message)
    except Exception as e:
        logger.error(f"Failed to log to database: {e}")

@app.post("/predict/")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    conn = await get_db_connection()
    try:
        csv_body = await request.body()
        csv_string = csv_body.decode('utf-8')

        features_df = pd.read_csv(StringIO(csv_string))
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        prediction = model.predict(features_df)
        
        # Forcer la prédiction à être une liste
        if isinstance(prediction, np.ndarray):  # Si c'est un tableau numpy
            prediction_list = prediction.tolist()
        else:
            prediction_list = [prediction]  # Sinon, forcer en liste

        logger.info(f"Prediction made: {prediction_list}")
        await log_to_database(conn, 'INFO', f"Prediction made: {prediction_list}")

        return {"prediction": prediction_list}
    except Exception as e:
        await conn.rollback()
        logger.error(f"Prediction error: {str(e)}")
        await log_to_database(conn, 'ERROR', f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()


@app.post("/train/")
async def train(api_key: str = Depends(get_api_key)):
    conn = await get_db_connection()
    try:
        logger.info("Starting training process")
        data = pd.read_sql_query("SELECT * FROM Model_training", conn)
        logger.info(f"Loaded training data shape: {data.shape}")

        new_model = RandomForestClassifier()
        X = data.drop(columns=['id', 'url', 'type'])
        y = data['type']
        new_model.fit(X, y)

        # Sauvegarder le modèle entraîné dans le chemin relatif
        with open(model_path, 'wb') as file:
            pickle.dump(new_model, file)

        logger.info("Model training completed successfully")
        await log_to_database(conn, 'INFO', "Model trained successfully")
        return {"detail": "Model trained successfully"}
    except Exception as e:
        await conn.rollback()
        logger.error(f"Training error: {str(e)}")
        await log_to_database(conn, 'ERROR', f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()

@app.get("/model-info/")
async def model_info(api_key: str = Depends(get_api_key)):
    if os.path.exists(model_path):
        mod_time = os.path.getmtime(model_path)
        file_size = os.path.getsize(model_path)
        return {
            "last_modified": datetime.fromtimestamp(mod_time).isoformat(),
            "file_size": file_size
        }
    else:
        raise HTTPException(status_code=404, detail="Model file not found")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API de Modèle",
        version="1.0.0",
        description="API pour prédire et entraîner un modèle",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    openapi_schema["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


