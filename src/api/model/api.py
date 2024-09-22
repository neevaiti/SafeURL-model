from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
import pickle
import pandas as pd
import os
from io import StringIO
from dotenv import load_dotenv
import psycopg2
import logging
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

from utils.read_result import read_result

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la connexion à la base de données
conn = psycopg2.connect(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME")
)
cursor = conn.cursor()

# Initialisation de l'application FastAPI
app = FastAPI()

API_MODEL_KEY = os.getenv("API_MODEL_KEY")

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"API_MODEL_KEY loaded: {'*' * len(API_MODEL_KEY)}")

# Chargement du modèle
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        logger.debug(f"Middleware - Path: {request.url.path}, API Key: {'*' * len(api_key) if api_key else 'None'}")
        if api_key != API_MODEL_KEY:
            logger.warning(f"Middleware - Invalid API key for path: {request.url.path}")
            raise HTTPException(status_code=401, detail="Accès non autorisé")
    response = await call_next(request)
    return response

async def get_api_key(api_key_header: str = Security(api_key_header)):
    logger.debug(f"Dependency - API Key: {'*' * len(api_key_header) if api_key_header else 'None'}")
    if api_key_header == API_MODEL_KEY:
        return api_key_header
    logger.warning("Dependency - Invalid API key")
    raise HTTPException(status_code=401, detail="Clé API invalide")

def log_to_database(level, message):
    try:
        cursor.execute("INSERT INTO Log (level, message) VALUES (%s, %s)", (level, message))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log to database: {str(e)}")

@app.post("/predict/")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    try:
        csv_body = await request.body()
        csv_string = csv_body.decode('utf-8')
        features_df = pd.read_csv(StringIO(csv_string))
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        prediction = model.predict(features_df)
        prediction = read_result(prediction)
        logger.info(f"Prediction made: {prediction}")
        log_to_database('INFO', f"Prediction made: {prediction}")

        return {"prediction": prediction}
    except Exception as e:
        conn.rollback()
        logger.error(f"Prediction error: {str(e)}")
        log_to_database('ERROR', f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def load_training_data():
    query = "SELECT * FROM Model_training"
    return pd.read_sql_query(query, conn)

def train_model(data):
    logger.info("Starting model training")
    model = RandomForestClassifier()
    X = data.drop(columns=['id', 'url', 'type'])
    y = data['type']
    
    logger.info(f"Training data shape: X: {X.shape}, y: {y.shape}")
    
    model.fit(X, y)
    
    logger.info("Model fitting completed")
    
    # Vérifier si le fichier existe et obtenir sa date de modification
    if os.path.exists(model_path):
        mod_time_before = os.path.getmtime(model_path)
        logger.info(f"Existing model last modified: {datetime.fromtimestamp(mod_time_before)}")
    else:
        logger.info("No existing model found")
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    # Vérifier la nouvelle date de modification
    mod_time_after = os.path.getmtime(model_path)
    logger.info(f"New model last modified: {datetime.fromtimestamp(mod_time_after)}")
    
    if os.path.exists(model_path):
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
    else:
        logger.error("Failed to create model file")
    
    return model

@app.post("/train/")
async def train(api_key: str = Depends(get_api_key)):
    try:
        logger.info("Starting training process")
        data = load_training_data()
        logger.info(f"Loaded training data shape: {data.shape}")
        
        new_model = train_model(data)
        
        # Vérifier si le modèle a été correctement mis à jour
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        
        if loaded_model == new_model:
            logger.info("Model successfully updated and loaded")
        else:
            logger.warning("Loaded model does not match newly trained model")
        
        logger.info("Model training completed successfully")
        log_to_database('INFO', "Model trained successfully")
        return {"detail": "Model trained successfully"}
    except Exception as e:
        conn.rollback()
        logger.error(f"Training error: {str(e)}")
        log_to_database('ERROR', f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
