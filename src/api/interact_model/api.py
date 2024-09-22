from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
import requests
import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2
import logging

from utils.script_fengineering import extract_features
from utils.clean_url import clean_url

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY")
API_MODEL_KEY = os.getenv("API_MODEL_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database connection
conn = psycopg2.connect(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME")
)
cursor = conn.cursor()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Accès non autorisé")
    response = await call_next(request)
    return response

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=401, detail="Clé API invalide")

@app.post("/prepare/")
async def prepare_data(url: str, api_key: str = Depends(get_api_key)):
    url = clean_url(url)
    features = extract_features(url)
    
    # Ensure 'url' column is present in features
    if 'url' not in features:
        features['url'] = url

    features_df = pd.DataFrame([features])

    # Log the columns of the DataFrame
    logger.debug(f"Features DataFrame columns: {features_df.columns.tolist()}")

    # Remove 'url' from features for prediction
    features_for_prediction = features_df.drop(columns=['url'])

    # Convert all columns to integers
    features_for_prediction = features_for_prediction.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    csv_string = features_for_prediction.to_csv(index=False)

    model_api_url = "http://api_model:12500/predict/"
    headers = {
        "X-API-Key": API_MODEL_KEY
    }

    response = requests.post(model_api_url, headers=headers, data=csv_string)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        logger.info(f"Prediction received: {prediction}")

        # Insert URL and prediction into the database
        try:
            for index, row in features_df.iterrows():
                # Log the columns of the DataFrame before dropping 'url'
                logger.debug(f"Features DataFrame columns before dropping 'url': {features_df.columns.tolist()}")

                # Remove 'url' from features_df to avoid duplication
                row = row.drop(labels=['url'])
                
                # Log the columns of the row after dropping 'url'
                logger.debug(f"Row columns after dropping 'url': {row.index.tolist()}")
                logger.debug(f"Row values after dropping 'url': {row.values.tolist()}")

                # Log the values to be inserted
                values_to_insert = (
                    url, prediction, int(row['use_of_ip']), int(row['abnormal_url']), int(row['count_www']), int(row['count_point']),
                    int(row['count_at']), int(row['count_https']), int(row['count_http']), int(row['count_percent']), int(row['count_question']),
                    int(row['count_dash']), int(row['count_equal']), int(row['count_dir']), int(row['count_embed_domain']), int(row['short_url']),
                    int(row['url_length']), int(row['hostname_length']), int(row['sus_url']), int(row['count_digits']), int(row['count_letters']),
                    int(row['fd_length']), int(row['tld_length'])
                )
                logger.debug(f"Inserting values: {values_to_insert}")
                logger.debug(f"Number of values to insert: {len(values_to_insert)}")

                # Log each value with its index
                for i, value in enumerate(values_to_insert):
                    logger.debug(f"Value {i}: {value}")

                # Ensure the number of values matches the number of columns (excluding the auto-incremented ID)
                if len(values_to_insert) != 23:
                    raise ValueError(f"Number of values to insert ({len(values_to_insert)}) does not match the number of columns (23)")

                cursor.execute("""
                    INSERT INTO Model_training (url, type, use_of_ip, abnormal_url, count_www, count_point, count_at, 
                                                count_https, count_http, count_percent, count_question, count_dash, 
                                                count_equal, count_dir, count_embed_domain, short_url, url_length, 
                                                hostname_length, sus_url, count_digits, count_letters, fd_length, tld_length)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, values_to_insert)

            # Vérifier si l'URL existe déjà
            cursor.execute("SELECT id FROM List_url WHERE url = %s", (url,))
            existing_url = cursor.fetchone()

            if existing_url is None:
                # Récupérer le dernier ID utilisé
                cursor.execute("SELECT MAX(id) FROM List_url")
                last_id = cursor.fetchone()[0]
                
                # Si la table est vide, commencer à 1, sinon incrémenter
                new_id = 1 if last_id is None else last_id + 1
                
                # L'URL n'existe pas, on l'insère avec le nouvel ID
                cursor.execute("""
                    INSERT INTO List_url (id, url, type) VALUES (%s, %s, %s)
                """, (new_id, url, prediction))
            else:
                logger.info(f"URL {url} already exists in List_url table. Skipping insertion.")

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to insert into database: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to insert into database")

        return {"prediction": prediction}
    else:
        raise HTTPException(status_code=500, detail="Model API request failed")

# Ajoutez cette fonction à la fin de votre fichier
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API d'interaction avec le modèle.",
        version="1.0.0",
        description="API pour préparer les données et interagir avec le modèle prédictif.",
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
