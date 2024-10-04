from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

import os
import sys
import pandas as pd
import asyncpg
import logging
import aiohttp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.script_fengineering import extract_features
from utils.clean_url import clean_url
from utils.read_result import read_result

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY")
API_MODEL_KEY = os.getenv("API_MODEL_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Asynchronous database connection
async def get_db_connection():
    """
    Initialise une connexion asynchrone à la base de données PostgreSQL.

    Returns:
        asyncpg.Connection: Connexion à la base de données.
    """
    return await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME")
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class UrlInput(BaseModel):
    """
    Modèle de données pour l'entrée d'URL.

    Attributes:
        url (HttpUrl): L'URL à traiter.
    """
    url: HttpUrl

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """
    Middleware pour vérifier la clé API dans les requêtes HTTP.

    Args:
        request (Request): La requête HTTP.
        call_next (Callable): La fonction suivante à appeler dans la chaîne de middleware.

    Returns:
        Response: La réponse HTTP après vérification de la clé API.

    Raises:
        HTTPException: Si la clé API est invalide.
    """
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Accès non autorisé")
    response = await call_next(request)
    return response

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Vérifie la clé API fournie dans l'en-tête.

    Args:
        api_key_header (str): La clé API fournie.

    Returns:
        str: La clé API validée.

    Raises:
        HTTPException: Si la clé API est invalide.
    """
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=401, detail="Clé API invalide")

# Retry mechanism for model API requests
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_prediction(session, model_api_url, headers, csv_string):
    """
    Envoie une requête à l'API du modèle pour obtenir une prédiction, avec un mécanisme de retry.

    Args:
        session (aiohttp.ClientSession): La session HTTP asynchrone.
        model_api_url (str): L'URL de l'API du modèle.
        headers (dict): Les en-têtes HTTP à inclure dans la requête.
        csv_string (str): Les données CSV à envoyer.

    Returns:
        dict: La réponse JSON de l'API du modèle.

    Raises:
        HTTPException: Si la requête échoue après plusieurs tentatives.
    """
    async with session.post(model_api_url, headers=headers, data=csv_string, timeout=10) as response:
        if response.status == 200:
            return await response.json()
        else:
            raise HTTPException(status_code=500, detail="Model API request failed")

@app.post("/prepare/")
async def prepare_data(data: UrlInput, api_key: str = Depends(get_api_key)):
    """
    Prépare les données à partir d'une URL et interagit avec le modèle pour obtenir une prédiction.

    Args:
        data (UrlInput): Les données d'entrée contenant l'URL.
        api_key (str): La clé API validée.

    Returns:
        dict: La prédiction du modèle sous forme de texte.

    Raises:
        HTTPException: Si une erreur survient lors de l'insertion dans la base de données.
    """
    url = clean_url(data.url)
    features = extract_features(url)
    
    if 'url' not in features:
        features['url'] = url

    features_df = pd.DataFrame([features])

    features_for_prediction = features_df.drop(columns=['url'])
    features_for_prediction = features_for_prediction.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    csv_string = features_for_prediction.to_csv(index=False)

    model_api_url = "http://api_model:12500/predict/"
    headers = {"X-API-Key": API_MODEL_KEY}

    async with aiohttp.ClientSession() as session:
        prediction_response = await fetch_prediction(session, model_api_url, headers, csv_string)

    # Extraire la prédiction en entier
    prediction_int = prediction_response["prediction"][0]  # Extraire l'entier de la liste [0] ou [1]

    # Convertir la prédiction entière en chaîne ('safe' ou 'phishing')
    prediction_str = read_result(prediction_int)

    logger.info(f"Prediction received: {prediction_str} (as int: {prediction_int})")

    conn = await get_db_connection()

    try:
        async with conn.transaction():
            # Insertion dans la table Model_training avec prédiction en entier
            for index, row in features_df.iterrows():
                row = row.drop(labels=['url'])

                values_to_insert = (
                    url, prediction_int,  # Insertion de l'entier dans la table Model_training
                    int(row['use_of_ip']), int(row['abnormal_url']), int(row['count_www']), 
                    int(row['count_point']), int(row['count_at']), int(row['count_https']), 
                    int(row['count_http']), int(row['count_percent']), int(row['count_question']), 
                    int(row['count_dash']), int(row['count_equal']), int(row['count_dir']), 
                    int(row['count_embed_domain']), int(row['short_url']), int(row['url_length']), 
                    int(row['hostname_length']), int(row['sus_url']), int(row['count_digits']), 
                    int(row['count_letters']), int(row['fd_length']), int(row['tld_length'])
                )

                await conn.execute("""
                    INSERT INTO model_training (url, type, use_of_ip, abnormal_url, count_www, count_point, count_at, 
                                                count_https, count_http, count_percent, count_question, count_dash, 
                                                count_equal, count_dir, count_embed_domain, short_url, url_length, 
                                                hostname_length, sus_url, count_digits, count_letters, fd_length, tld_length)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
                """, *values_to_insert)

            # Vérification de l'existence de l'URL dans List_url
            existing_url = await conn.fetchrow("SELECT id FROM list_url WHERE url = $1", url)
            if not existing_url:
                # Insertion dans la table List_url avec prédiction en tant que texte
                last_id = await conn.fetchval("SELECT MAX(id) FROM list_url")
                new_id = 1 if last_id is None else last_id + 1
                await conn.execute("""
                    INSERT INTO list_url (id, url, type) VALUES ($1, $2, $3)
                """, new_id, url, prediction_str)  # Insertion de la prédiction en tant que texte
            else:
                logger.info(f"URL {url} already exists in list_url table. Skipping insertion.")

    except Exception as e:
        logger.error(f"Failed to insert into database: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to insert into database")
    finally:
        await conn.close()

    return {"prediction": prediction_str}

def custom_openapi():
    """
    Personnalise le schéma OpenAPI pour inclure la sécurité de la clé API.

    Returns:
        dict: Le schéma OpenAPI personnalisé.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API d'interaction avec le modèle d'IA.",
        version="1.0.0",
        description="API pour préparer les données et interagir avec le modèle d'IA.",
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