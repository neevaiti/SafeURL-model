# from fastapi import FastAPI, HTTPException, Request, Depends, Security
# from fastapi.openapi.utils import get_openapi
# from fastapi.security import APIKeyHeader
# from fastapi.encoders import jsonable_encoder
# import mlflow


# import pickle
# import pandas as pd
# import numpy as np
# import os
# from io import StringIO
# from dotenv import load_dotenv

# import asyncpg
# import logging
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from datetime import datetime

# # Chargement des variables d'environnement
# load_dotenv()

# # Initialisation de l'application FastAPI
# app = FastAPI()

# API_MODEL_KEY = os.getenv("API_MODEL_KEY")
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("SafeURL Model Training")

# # Configuration du logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# logger.debug(f"API_MODEL_KEY loaded: {'*' * len(API_MODEL_KEY)}")

# # Définir le chemin du modèle relatif au fichier actuel
# model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# # Vérifier si le fichier model.pkl existe
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at path: {model_path}")

# # Charger le modèle
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# # Connexion à la base de données
# async def get_db_connection():
#     return await asyncpg.connect(
#         user=os.getenv("DB_USER"),
#         password=os.getenv("DB_PASSWORD"),
#         host=os.getenv("DB_HOST"),
#         port=os.getenv("DB_PORT"),
#         database=os.getenv("DB_NAME")
#     )

# @app.middleware("http")
# async def verify_api_key(request: Request, call_next):
#     excluded_paths = ["/docs", "/openapi.json", "/redoc"]
#     if request.url.path not in excluded_paths:
#         api_key = request.headers.get("X-API-Key")
#         if not api_key or api_key != API_MODEL_KEY:
#             raise HTTPException(status_code=401, detail="Accès non autorisé")
#     return await call_next(request)

# async def get_api_key(api_key_header: str = Security(api_key_header)):
#     logger.debug(f"Dependency - API Key: {'*' * len(api_key_header) if api_key_header else 'None'}")
#     if api_key_header == API_MODEL_KEY:
#         return api_key_header
#     logger.warning("Dependency - Invalid API key")
#     raise HTTPException(status_code=401, detail="Clé API invalide")

# async def log_to_database(conn, level, message):
#     try:
#         query = "INSERT INTO Log (level, message) VALUES ($1, $2)"
#         await conn.execute(query, level, message)
#     except Exception as e:
#         logger.error(f"Failed to log to database: {e}")

# @app.post("/predict/")
# async def predict(request: Request, api_key: str = Depends(get_api_key)):
#     conn = await get_db_connection()
#     try:
#         csv_body = await request.body()
#         csv_string = csv_body.decode('utf-8')

#         features_df = pd.read_csv(StringIO(csv_string))
#         features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

#         prediction = model.predict(features_df)
        
#         # Forcer la prédiction à être une liste
#         if isinstance(prediction, np.ndarray):  # Si c'est un tableau numpy
#             prediction_list = prediction.tolist()
#         else:
#             prediction_list = [prediction]  # Sinon, forcer en liste

#         logger.info(f"Prediction made: {prediction_list}")
#         await log_to_database(conn, 'INFO', f"Prediction made: {prediction_list}")

#         return {"prediction": prediction_list}
#     except Exception as e:
#         await conn.rollback()
#         logger.error(f"Prediction error: {str(e)}")
#         await log_to_database(conn, 'ERROR', f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         await conn.close()


# @app.post("/train/")
# async def train(api_key: str = Depends(get_api_key)):
#     conn = await get_db_connection()
#     try:
#         logger.info("Starting training process")
#         data = pd.read_sql_query("SELECT * FROM Model_training", conn)
#         logger.info(f"Loaded training data shape: {data.shape}")

#         new_model = RandomForestClassifier()
#         X = data.drop(columns=['id', 'url', 'type'])
#         y = data['type']
#         new_model.fit(X, y)

#         # Effectuer des prédictions sur les données d'entraînement pour évaluer les performances
#         y_pred = new_model.predict(X)
#         report = classification_report(y, y_pred, target_names=['safe', 'phishing'], output_dict=True)

#         # Sauvegarder le modèle entraîné avec MLflow
#         mlflow.set_tracking_uri("http://mlflow:5000")
#         mlflow.set_experiment("SafeURL Model Training")
#         with mlflow.start_run():
#             mlflow.log_metric("accuracy", report['accuracy'])
#             mlflow.log_metric("precision_safe", report['safe']['precision'])
#             mlflow.log_metric("recall_safe", report['safe']['recall'])
#             mlflow.log_metric("f1_safe", report['safe']['f1-score'])
#             mlflow.log_metric("precision_phishing", report['phishing']['precision'])
#             mlflow.log_metric("recall_phishing", report['phishing']['recall'])
#             mlflow.log_metric("f1_phishing", report['phishing']['f1-score'])
#             mlflow.sklearn.log_model(new_model, "model")

#         # Sauvegarder le modèle localement
#         with open(model_path, 'wb') as file:
#             pickle.dump(new_model, file)

#         logger.info("Model training completed successfully")
#         await log_to_database(conn, 'INFO', "Model trained successfully")
#         return {"detail": "Model trained successfully"}
#     except Exception as e:
#         logger.error(f"Training error: {str(e)}")
#         await log_to_database(conn, 'ERROR', f"Training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         await conn.close()



# @app.post("/train/")
# async def train(api_key: str = Depends(get_api_key)):
#     conn = await get_db_connection()
#     try:
#         logger.info("Starting training process")
#         data = pd.read_sql_query("SELECT * FROM Model_training", conn)
#         logger.info(f"Loaded training data shape: {data.shape}")

#         new_model = RandomForestClassifier()
#         X = data.drop(columns=['id', 'url', 'type'])
#         y = data['type']
#         new_model.fit(X, y)

#         # Sauvegarder le modèle entraîné dans le chemin relatif
#         with open(model_path, 'wb') as file:
#             pickle.dump(new_model, file)

#         logger.info("Model training completed successfully")
#         await log_to_database(conn, 'INFO', "Model trained successfully")
#         return {"detail": "Model trained successfully"}
#     except Exception as e:
#         await conn.rollback()
#         logger.error(f"Training error: {str(e)}")
#         await log_to_database(conn, 'ERROR', f"Training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         await conn.close()

# @app.get("/model-info/")
# async def model_info(api_key: str = Depends(get_api_key)):
#     if os.path.exists(model_path):
#         mod_time = os.path.getmtime(model_path)
#         file_size = os.path.getsize(model_path)
#         return {
#             "last_modified": datetime.fromtimestamp(mod_time).isoformat(),
#             "file_size": file_size
#         }
#     else:
#         raise HTTPException(status_code=404, detail="Model file not found")

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="API de Modèle",
#         version="1.0.0",
#         description="API pour prédire et entraîner un modèle",
#         routes=app.routes,
#     )
#     openapi_schema["components"]["securitySchemes"] = {
#         "APIKeyHeader": {
#             "type": "apiKey",
#             "in": "header",
#             "name": "X-API-Key"
#         }
#     }
#     openapi_schema["security"] = [{"APIKeyHeader": []}]
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi



# import os
# import pickle
# import logging
# import pandas as pd
# import numpy as np
# from fastapi import FastAPI, HTTPException, Request, Depends, Security
# from fastapi.security import APIKeyHeader
# from fastapi.openapi.utils import get_openapi
# from io import StringIO
# from dotenv import load_dotenv
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from datetime import datetime
# import asyncpg
# import sqlite3
# import mlflow

# # Chargement des variables d'environnement
# load_dotenv()

# # Initialisation de l'application FastAPI
# app = FastAPI()

# API_MODEL_KEY = os.getenv("API_MODEL_KEY")
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# IS_TEST = os.getenv("IS_TEST", "False").lower() == "true"

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("SafeURL Model Training")

# # Configuration du logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# logger.debug(f"API_MODEL_KEY loaded: {'*' * len(API_MODEL_KEY)}")

# # Définir le chemin du modèle relatif au fichier actuel
# model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# # Vérifier si le fichier model.pkl existe
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at path: {model_path}")

# # Charger le modèle
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# # Connexion à la base de données
# def get_db_connection():
#     if IS_TEST:
#         # SQLite en mode test
#         return sqlite3.connect(":memory:")  # Connexion SQLite synchronisée
#     else:
#         # PostgreSQL en production avec asyncpg
#         return asyncpg.connect(
#             user=os.getenv("DB_USER"),
#             password=os.getenv("DB_PASSWORD"),
#             host=os.getenv("DB_HOST"),
#             port=os.getenv("DB_PORT"),
#             database=os.getenv("DB_NAME")
#         )

# # Middleware pour vérifier la clé API
# @app.middleware("http")
# async def verify_api_key(request: Request, call_next):
#     excluded_paths = ["/docs", "/openapi.json", "/redoc"]
#     if request.url.path not in excluded_paths:
#         api_key = request.headers.get("X-API-Key")
#         if not api_key or api_key != API_MODEL_KEY:
#             raise HTTPException(status_code=401, detail="Accès non autorisé")
#     return await call_next(request)

# # Vérifier la clé API
# async def get_api_key(api_key_header: str = Security(api_key_header)):
#     logger.debug(f"Dependency - API Key: {'*' * len(api_key_header) if api_key_header else 'None'}")
#     if api_key_header == API_MODEL_KEY:
#         return api_key_header
#     logger.warning("Dependency - Invalid API key")
#     raise HTTPException(status_code=401, detail="Clé API invalide")

# # Fonction pour loguer dans la base de données
# def log_to_database(conn, level, message):
#     try:
#         query = "INSERT INTO Log (level, message) VALUES (?, ?)" if IS_TEST else "INSERT INTO Log (level, message) VALUES ($1, $2)"
#         conn.execute(query, (level, message) if IS_TEST else (level, message))
#         if IS_TEST:
#             conn.commit()
#     except Exception as e:
#         logger.error(f"Failed to log to database: {e}")

# # Fonction pour convertir les résultats en DataFrame pour SQLite
# def convert_sqlite_to_dataframe(cursor):
#     rows = cursor.fetchall()
#     columns = [description[0] for description in cursor.description]
#     return pd.DataFrame(rows, columns=columns)

# # Fonction pour convertir les résultats en DataFrame pour PostgreSQL
# def convert_pg_to_dataframe(rows):
#     columns = rows[0].keys() if rows else []
#     return pd.DataFrame(rows, columns=columns)

# # Endpoint pour prédire
# @app.post("/predict/")
# async def predict(request: Request, api_key: str = Depends(get_api_key)):
#     conn = get_db_connection() if IS_TEST else await get_db_connection()  # Gestion synchrone/asynchrone
#     try:
#         csv_body = await request.body()
#         csv_string = csv_body.decode('utf-8')

#         features_df = pd.read_csv(StringIO(csv_string))
#         features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

#         prediction = model.predict(features_df)
        
#         prediction_list = prediction.tolist() if isinstance(prediction, np.ndarray) else [prediction]

#         logger.info(f"Prediction made: {prediction_list}")
#         log_to_database(conn, 'INFO', f"Prediction made: {prediction_list}")

#         return {"prediction": prediction_list}
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         log_to_database(conn, 'ERROR', f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if IS_TEST:
#             conn.close()
#         else:
#             await conn.close()

# # Endpoint pour entraîner un nouveau modèle
# @app.post("/train/")
# async def train(api_key: str = Depends(get_api_key)):
#     conn = get_db_connection() if IS_TEST else await get_db_connection()  # Gestion synchrone/asynchrone
#     try:
#         logger.info("Starting training process")

#         if IS_TEST:
#             cursor = conn.cursor()
#             cursor.execute("SELECT * FROM Model_training")
#             data = convert_sqlite_to_dataframe(cursor)
#         else:
#             rows = await conn.fetch("SELECT * FROM Model_training")
#             data = convert_pg_to_dataframe(rows)

#         logger.info(f"Loaded training data shape: {data.shape}")

#         new_model = RandomForestClassifier()
#         X = data.drop(columns=['id', 'url', 'type'])
#         y = data['type']
#         new_model.fit(X, y)

#         y_pred = new_model.predict(X)
#         report = classification_report(y, y_pred, target_names=['safe', 'phishing'], labels=np.unique(y), output_dict=True)

#         with mlflow.start_run():
#             mlflow.log_metric("accuracy", report['accuracy'])
#             mlflow.log_metric("precision_safe", report['safe']['precision'])
#             mlflow.log_metric("recall_safe", report['safe']['recall'])
#             mlflow.log_metric("f1_safe", report['safe']['f1-score'])
#             mlflow.log_metric("precision_phishing", report['phishing']['precision'])
#             mlflow.log_metric("recall_phishing", report['phishing']['recall'])
#             mlflow.log_metric("f1_phishing", report['phishing']['f1-score'])
#             mlflow.sklearn.log_model(new_model, "model")

#         with open(model_path, 'wb') as file:
#             pickle.dump(new_model, file)

#         logger.info("Model training completed successfully")
#         log_to_database(conn, 'INFO', "Model trained successfully")
#         return {"detail": "Model trained successfully"}
#     except Exception as e:
#         logger.error(f"Training error: {str(e)}")
#         log_to_database(conn, 'ERROR', f"Training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if IS_TEST:
#             conn.close()
#         else:
#             await conn.close()

# # Endpoint pour obtenir les informations sur le modèle
# @app.get("/model-info/")
# async def model_info(api_key: str = Depends(get_api_key)):
#     if os.path.exists(model_path):
#         mod_time = os.path.getmtime(model_path)
#         file_size = os.path.getsize(model_path)
#         return {
#             "last_modified": datetime.fromtimestamp(mod_time).isoformat(),
#             "file_size": file_size
#         }
#     else:
#         raise HTTPException(status_code=404, detail="Model file not found")

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="API de Modèle",
#         version="1.0.0",
#         description="API pour prédire et entraîner un modèle",
#         routes=app.routes,
#     )
#     openapi_schema["components"]["securitySchemes"] = {
#         "APIKeyHeader": {
#             "type": "apiKey",
#             "in": "header",
#             "name": "X-API-Key"
#         }
#     }
#     openapi_schema["security"] = [{"APIKeyHeader": []}]
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi


import os
import pickle
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from io import StringIO
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
import asyncpg
import sqlite3
import json

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'application FastAPI
app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

API_MODEL_KEY = os.getenv("API_MODEL_KEY")
IS_TEST = os.getenv("IS_TEST", "False").lower() == "true"

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"API_MODEL_KEY loaded: {'*' * len(API_MODEL_KEY)}")

# Chemin de stockage des modèles
MODEL_DIR = "/app/models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model = None  # Initialise une variable globale pour stocker le modèle

def load_model():
    global model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No model versions found in storage.")
    
    latest_model = max(model_files, key=lambda f: int(f.split("_v")[1].split(".pkl")[0]))
    model_path = os.path.join(MODEL_DIR, latest_model)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logger.info(f"Loaded model from {model_path}")

# Charge un modèle existant au démarrage si disponible
try:
    load_model()
except FileNotFoundError as e:
    logger.warning(f"Model not found: {str(e)}. You need to train a model first.")

# Connexion à la base de données
def get_db_connection_sync():
    if IS_TEST:
        return sqlite3.connect(":memory:")
    raise Exception("get_db_connection_sync should not be called in production!")

async def get_db_connection():
    if IS_TEST:
        return get_db_connection_sync()
    else:
        return await asyncpg.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME")
        )

# Middleware pour vérifier la clé API
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_MODEL_KEY:
            raise HTTPException(status_code=401, detail="Accès non autorisé")
    return await call_next(request)

# Vérifier la clé API
async def get_api_key(api_key_header: str = Security(api_key_header)):
    logger.debug(f"Dependency - API Key: {'*' * len(api_key_header) if api_key_header else 'None'}")
    if api_key_header == API_MODEL_KEY:
        return api_key_header
    logger.warning("Dependency - Invalid API key")
    raise HTTPException(status_code=401, detail="Clé API invalide")


# Fonction pour sauvegarder le modèle avec versioning
async def save_model_version(model, metrics):
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"model_v{version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    # Sauvegarder le modèle en local
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # Sauvegarder les informations de version en base de données
    conn = await get_db_connection()
    try:
        query = """
        INSERT INTO model_versions (version, model_path, metrics, created_at)
        VALUES ($1, $2, $3, $4)
        """
        await conn.execute(query, version, model_path, json.dumps(metrics), datetime.now())
        logger.info(f"Model version {version} saved at {model_path}")
    finally:
        await conn.close()

# Fonction pour loguer dans la base de données
def log_to_database_sync(conn, level, message):
    try:
        query = "INSERT INTO logs (level, message) VALUES (?, ?)"
        conn.execute(query, (level, message))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log to database: {e}")

async def log_to_database_async(conn, level, message):
    try:
        query = "INSERT INTO logs (level, message) VALUES ($1, $2)"
        await conn.execute(query, level, message)
    except Exception as e:
        logger.error(f"Failed to log to database: {e}")
        
async def log_to_database(conn, level, message):
    if IS_TEST:
        log_to_database_sync(conn, level, message)
    else:
        await log_to_database_async(conn, level, message)
        

# Endpoint pour prédire
@app.post("/predict/")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="No model available. Please train a model first.")
    
    conn = get_db_connection() if IS_TEST else await get_db_connection()  # Gestion synchrone/asynchrone
    try:
        # Récupération des données CSV du corps de la requête
        csv_body = await request.body()
        csv_string = csv_body.decode('utf-8')

        # Conversion du CSV en DataFrame pandas
        features_df = pd.read_csv(StringIO(csv_string))
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        # Prédiction avec le modèle
        prediction = model.predict(features_df)
        
        # Conversion des prédictions en liste pour renvoi
        prediction_list = prediction.tolist() if isinstance(prediction, np.ndarray) else [prediction]

        logger.info(f"Prediction made: {prediction_list}")
        log_to_database(conn, 'INFO', f"Prediction made: {prediction_list}")

        return {"prediction": prediction_list}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        log_to_database(conn, 'ERROR', f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if IS_TEST:
            conn.close()
        else:
            await conn.close()



# Endpoint pour entraîner un nouveau modèle
@app.post("/train/")
async def train(api_key: str = Depends(get_api_key)):
    conn = get_db_connection_sync() if IS_TEST else await get_db_connection()  # Gestion synchrone/asynchrone
    try:
        logger.info("Starting training process")

        # Charger le modèle de base existant
        base_model_path = os.path.join(MODEL_DIR, "model_v2024.pkl")
        if not os.path.exists(base_model_path):
            raise HTTPException(status_code=404, detail="Base model not found.")

        with open(base_model_path, 'rb') as file:
            model = pickle.load(file)  # Charger le modèle de base

        logger.info(f"Loaded base model from {base_model_path}")

        # Charger les données d'entraînement
        if IS_TEST:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_training")
            data = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])
        else:
            rows = await conn.fetch("SELECT * FROM model_training")
            data = pd.DataFrame(rows, columns=rows[0].keys())

        logger.info(f"Loaded training data shape: {data.shape}")

        X = data.drop(columns=['id', 'url', 'type'])
        y = data['type']

        # Vérifier le nombre de classes uniques
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning(f"Training data contains only one class: {unique_classes[0]}. Adding synthetic data for the missing class.")
            # Ajouter des données factices pour l'autre classe (exemple pour une classification binaire)
            if unique_classes[0] == 0:
                synthetic_data = pd.DataFrame([[-1] * X.shape[1]], columns=X.columns)  # Ajouter des features factices
                X = pd.concat([X, synthetic_data])
                y = pd.concat([y, pd.Series([1])])  # Ajouter la classe factice
            else:
                synthetic_data = pd.DataFrame([[-1] * X.shape[1]], columns=X.columns)
                X = pd.concat([X, synthetic_data])
                y = pd.concat([y, pd.Series([0])])  # Ajouter la classe factice

            logger.info("Synthetic data added to balance classes.")

        # Réentraîner le modèle existant
        model.fit(X, y)  # Réentraînement du modèle de base sur les nouvelles données

        y_pred = model.predict(X)
        report = classification_report(
            y, 
            y_pred, 
            target_names=['safe', 'phishing'], 
            labels=unique_classes,  # Ajuster pour les classes présentes dans les données
            output_dict=True
        )

        metrics = {
            "accuracy": report['accuracy'],
            "precision_safe": report['safe']['precision'],
            "recall_safe": report['safe']['recall'],
            "f1_safe": report['safe']['f1-score'],
            "precision_phishing": report['phishing']['precision'],
            "recall_phishing": report['phishing']['recall'],
            "f1_phishing": report['phishing']['f1-score']
        }

        # Sauvegarder la nouvelle version du modèle avec versioning
        await save_model_version(model, metrics)

        logger.info("Model training completed successfully")
        if IS_TEST:
            log_to_database_sync(conn, 'INFO', "Model retrained successfully")
        else:
            await log_to_database_async(conn, 'INFO', "Model retrained successfully")

        return {"detail": "Model retrained successfully"}
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        if IS_TEST:
            log_to_database_sync(conn, 'ERROR', f"Training error: {str(e)}")
        else:
            await log_to_database_async(conn, 'ERROR', f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if IS_TEST:
            conn.close()
        else:
            await conn.close()



# Endpoint pour récupérer une version spécifique du modèle
@app.get("/load-model/{version}")
async def load_model(version: str):
    conn = await get_db_connection()
    try:
        query = "SELECT model_path, metrics FROM model_versions WHERE version = $1"
        result = await conn.fetchrow(query, version)
        if not result:
            raise HTTPException(status_code=404, detail="Model version not found")

        model_path = result['model_path']
        metrics = result['metrics']

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return {"message": "Model loaded", "metrics": metrics}
    finally:
        await conn.close()

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