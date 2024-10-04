import os
import asyncpg
import sqlite3
import json
import pickle
import logging
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from io import StringIO
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

API_MODEL_KEY = os.getenv("API_MODEL_KEY")
IS_TEST = os.getenv("IS_TEST", "false").lower() == "true"

# Chemin de stockage des modèles
MODEL_DIR = "/app/models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logger.debug(f"Model directory set to: {MODEL_DIR}")
    

# Classe pour la gestion des modèles
class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        os.makedirs(self.model_dir, exist_ok=True)

    def load_latest_model(self):
        """Charge le modèle le plus récent depuis le répertoire des modèles."""
        model_files = sorted(
            [f for f in os.listdir(self.model_dir) if f.startswith("model_v") and f.endswith(".pkl")],
            key=lambda f: int(f.split("_v")[1].split(".pkl")[0]),
            reverse=True
        )
        if not model_files:
            raise FileNotFoundError("Aucun modèle trouvé dans le répertoire de modèles.")

        latest_model = model_files[0]
        model_path = os.path.join(self.model_dir, latest_model)
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        logger.info(f"Modèle chargé depuis {model_path}")

    def load_model(self, version: str):
        """Charge un modèle spécifique basé sur la version donnée."""
        model_path = os.path.join(self.model_dir, version)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle {version} introuvable.")

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        logger.info(f"Modèle chargé depuis {model_path}")
        return model_path

    def get_model(self):
        """Retourne le modèle en mémoire ou charge le dernier modèle disponible."""
        if self.model is None:
            self.load_latest_model()
        return self.model

    def get_available_models(self):
        """Retourne la liste des modèles disponibles dans le répertoire."""
        return [f for f in os.listdir(self.model_dir) if f.startswith("model_v") and f.endswith(".pkl")]

# Fonction pour sauvegarder le modèle avec versioning
    async def save_model_version(self, model, metrics):
        """Sauvegarde le modèle avec versioning dans le répertoire des modèles et enregistre les informations de version dans la base de données."""
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        model_filename = f"model_v{version}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)

        try:
            logger.debug(f"Attempting to save model to {model_path}")
            # Sauvegarder le modèle en local
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            logger.info(f"Version du modèle {model_filename} sauvegardée dans {model_path}")

            # Sauvegarder les informations de version en base de données
            query = """
            INSERT INTO model_versions (version, model_path, metrics, created_at)
            VALUES ($1, $2, $3, $4)
            """
            await db_manager.conn.execute(query, model_filename, model_path, json.dumps(metrics), datetime.now())
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            raise


# Classe pour la gestion des logs et la connexion à la base de données
class DatabaseManager:
    def __init__(self):
        self.conn = None

    async def connect(self):
        """Initialise la connexion à la base de données."""
        if os.getenv("IS_TEST", "false").lower() == "true":
            self.conn = sqlite3.connect(":memory:")
        else:
            self.conn = await asyncpg.connect(
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                database=os.getenv("DB_NAME")
            )

    async def close(self):
        """Ferme la connexion à la base de données."""
        if self.conn:
            if IS_TEST:
                self.conn.close()
            else:
                await self.conn.close()

    async def log_to_database(self, level, message):
        """Enregistre les logs dans la base de données."""
        query = "INSERT INTO logs (level, message) VALUES ($1, $2)" if not IS_TEST else "INSERT INTO logs (level, message) VALUES (?, ?)"
        try:
            if IS_TEST:
                cursor = self.conn.cursor()
                cursor.execute(query, (level, message))
                self.conn.commit()
            else:
                await self.conn.execute(query, level, message)
        except Exception as e:
            logger.error(f"Échec de l'enregistrement du log dans la base de données: {e}")

    async def fetch(self, query, *args):
        """Récupère les résultats de la base de données en fonction d'une requête SQL."""
        try:
            if IS_TEST:
                cursor = self.conn.cursor()
                cursor.execute(query, args)
                return cursor.fetchall()  # Retourne toutes les lignes
            else:
                return await self.conn.fetch(query, *args)  # asyncpg retourne toutes les lignes
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            raise



# Initialisation des gestionnaires de modèles et de base de données
model_manager = ModelManager(MODEL_DIR)
db_manager = DatabaseManager()


# Middleware pour vérifier la clé API
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_MODEL_KEY:
            raise HTTPException(status_code=401, detail="Accès non autorisé")
    return await call_next(request)


# Fonction pour vérifier et obtenir la clé API
async def get_api_key(api_key_header: str = Security(APIKeyHeader(name="X-API-Key", auto_error=False))):
    if api_key_header == API_MODEL_KEY:
        return api_key_header
    raise HTTPException(status_code=401, detail="Clé API invalide")


# Endpoint pour prédire
@app.post("/predict/")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    await db_manager.connect()
    model = model_manager.get_model()

    try:
        csv_body = await request.body()
        csv_string = csv_body.decode('utf-8')
        features_df = pd.read_csv(StringIO(csv_string)).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        prediction = model.predict(features_df).tolist()

        logger.info(f"Prédiction effectuée: {prediction}")
        await db_manager.log_to_database('INFO', f"Prédiction effectuée: {prediction}")

        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Erreur de prédiction: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")
    finally:
        await db_manager.close()


# Endpoint pour entraîner un nouveau modèle avec noms de colonnes descriptifs

@app.post("/train/")
async def train(api_key: str = Depends(get_api_key)):
    await db_manager.connect()
    try:
        logger.info("Début de l'entraînement du modèle")

        # Charger les données d'entraînement depuis la base de données
        rows = await db_manager.fetch("SELECT * FROM model_training")

        # Les noms des colonnes descriptives
        columns = ['id', 'url', 'type', 'use_of_ip', 'abnormal_url', 'count_www', 'count_point', 'count_at', 
                   'count_https', 'count_http', 'count_percent', 'count_question', 'count_dash', 'count_equal', 
                   'count_dir', 'count_embed_domain', 'short_url', 'url_length', 'hostname_length', 'sus_url', 
                   'fd_length', 'tld_length', 'count_digits', 'count_letters']

        data = pd.DataFrame(rows, columns=columns)
        if data.empty:
            raise HTTPException(status_code=400, detail="Aucune donnée d'entraînement disponible.")

        # Extraire X et y avec les bonnes colonnes
        X = data[['use_of_ip', 'abnormal_url', 'count_point', 'count_www', 'count_at', 'count_dir',
                  'count_embed_domain', 'short_url', 'count_https', 'count_http', 'count_percent', 
                  'count_question', 'count_dash', 'count_equal', 'url_length', 'hostname_length',
                  'sus_url', 'fd_length', 'tld_length', 'count_digits', 'count_letters']]
        y = data['type']

        # Entraîner le modèle
        model = RandomForestClassifier()
        model.fit(X, y)

        # Prédire sur les données d'entraînement pour obtenir les métriques
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)

        # Sauvegarder le modèle et les métriques
        metrics_json = json.dumps({"metrics": report})
        await model_manager.save_model_version(model, metrics_json)
        logger.info(f"Métriques sauvegardées: {metrics_json}")

        logger.info("Entraînement du modèle terminé avec succès")
        return {"detail": "Entraînement du modèle terminé avec succès", "metrics": report}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await db_manager.close()





# Endpoint pour supprimer un modèle spécifique
@app.delete("/delete-model/{version}")
async def delete_model(version: str, api_key: str = Depends(get_api_key)):
    await db_manager.connect()

    try:
        # Rechercher le modèle dans la base de données
        query = "SELECT model_path FROM model_versions WHERE version = $1"
        result = await db_manager.conn.fetchrow(query, version)

        if not result:
            logger.error(f"Version de modèle {version} introuvable dans la base de données")
            await db_manager.log_to_database('ERROR', f"Version de modèle {version} introuvable dans la base de données")
            raise HTTPException(status_code=404, detail="Version de modèle introuvable")

        model_path = result['model_path']
        logger.debug(f"Version de modèle trouvée : {version}, chemin : {model_path}")

        # Vérifier si le fichier existe sur le disque et le supprimer
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Fichier modèle {model_path} supprimé avec succès")
            await db_manager.log_to_database('INFO', f"Fichier modèle {model_path} supprimé avec succès")
        else:
            logger.error(f"Fichier modèle {model_path} introuvable sur le disque")
            await db_manager.log_to_database('ERROR', f"Fichier modèle {model_path} introuvable sur le disque")
            raise HTTPException(status_code=404, detail="Fichier modèle introuvable")

        # Supprimer les informations du modèle de la base de données
        delete_query = "DELETE FROM model_versions WHERE version = $1"
        await db_manager.conn.execute(delete_query, version)

        logger.info(f"Version de modèle {version} supprimée de la base de données")
        await db_manager.log_to_database('INFO', f"Version de modèle {version} supprimée de la base de données")

        return {"message": f"Version de modèle {version} supprimée avec succès"}

    except Exception as e:
        logger.error(f"Erreur lors de la suppression: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Erreur lors de la suppression: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression: {str(e)}")

    finally:
        await db_manager.close()


# Endpoint pour récupérer les métriques d'un modèle spécifique
@app.get("/model-metrics/{version}")
async def get_model_metrics(version: str, api_key: str = Depends(get_api_key)):
    await db_manager.connect()
    try:
        query = "SELECT metrics FROM model_versions WHERE version = $1"
        result = await db_manager.conn.fetchrow(query, version)
        
        if not result:
            logger.error(f"Aucune métrique trouvée pour la version {version}")
            raise HTTPException(status_code=404, detail="Version de modèle introuvable")
        
        metrics = result['metrics']
        logger.info(f"Métriques récupérées pour la version {version}: {metrics}")
        return {"metrics": json.loads(metrics)}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des métriques")
    finally:
        await db_manager.close()

# Endpoint pour lister les modèles disponibles
@app.get("/models/")
async def list_models():
    """Retourne les modèles disponibles dans le répertoire des modèles"""
    model_files = model_manager.get_available_models()
    return {"available_models": model_files}


@app.get("/load-model/")
async def load_specific_model(version: str):
    """Charge un modèle spécifique en fonction de la version"""
    await db_manager.connect()

    try:
        logger.info(f"Tentative de chargement du modèle version {version}")
        await db_manager.log_to_database('INFO', f"Tentative de chargement du modèle version {version}")

        # Charger le modèle spécifique
        model_path = model_manager.load_model(version)

        # Si le modèle est chargé avec succès
        logger.info(f"Modèle {version} chargé avec succès depuis {model_path}")
        await db_manager.log_to_database('INFO', f"Modèle {version} chargé avec succès depuis {model_path}")

        return {"message": f"Modèle {version} chargé avec succès depuis {model_path}"}
    
    except FileNotFoundError as e:
        logger.error(f"Erreur: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Erreur lors du chargement du modèle {version}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle {version}: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Erreur inattendue lors du chargement du modèle {version}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle")

    finally:
        await db_manager.close()

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