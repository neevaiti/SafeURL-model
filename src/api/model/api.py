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
from enum import Enum
import asyncpg
import sqlite3
import json
import tempfile

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
            logger.info(f"Version du modèle {version} sauvegardée dans {model_path}")

            # Sauvegarder les informations de version en base de données
            query = """
            INSERT INTO model_versions (version, model_path, metrics, created_at)
            VALUES ($1, $2, $3, $4)
            """
            await db_manager.conn.execute(query, version, model_path, json.dumps(metrics), datetime.now())
            logger.info(f"Model version {version} saved to the database")

        except Exception as e:
            logger.error(f"Failed to save model version {version}: {e}")
            raise


# Classe pour la gestion des logs et la connexion à la base de données
class DatabaseManager:
    def __init__(self):
        self.conn = None

    async def connect(self):
        """Initialise la connexion à la base de données."""
        if IS_TEST:
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


# Endpoint pour entraîner un nouveau modèle
@app.post("/train/")
async def train(api_key: str = Depends(get_api_key)):
    await db_manager.connect()
    try:
        logger.info("Début de l'entraînement du modèle")
        await db_manager.log_to_database('INFO', "Début de l'entraînement du modèle")

        # Charger le modèle de base existant
        base_model_path = os.path.join(MODEL_DIR, "model_v2024.pkl")
        if not os.path.exists(base_model_path):
            raise HTTPException(status_code=404, detail="Modèle de base introuvable.")

        with open(base_model_path, 'rb') as file:
            model = pickle.load(file)  # Charger le modèle de base

        logger.info(f"Modèle de base chargé depuis {base_model_path}")
        await db_manager.log_to_database('INFO', f"Modèle de base chargé depuis {base_model_path}")

        # Charger les données d'entraînement depuis la base de données
        rows = await db_manager.fetch("SELECT * FROM model_training")

        # Conversion des données en DataFrame
        data = pd.DataFrame(rows, columns=['id', 'url', 'type'] + [f'feature_{i}' for i in range(1, 22)])
        if data.empty:
            raise HTTPException(status_code=400, detail="Aucune donnée d'entraînement disponible.")

        logger.info(f"Données d'entraînement chargées avec {data.shape[0]} lignes et {data.shape[1]} colonnes.")
        await db_manager.log_to_database('INFO', f"Données d'entraînement chargées avec {data.shape[0]} lignes et {data.shape[1]} colonnes.")

        # Préparation des données
        X = data.drop(columns=['id', 'url', 'type'])
        y = data['type']

        # Réentraîner le modèle
        model.fit(X, y)

        y_pred = model.predict(X)
        report = classification_report(y, y_pred, target_names=['safe', 'phishing'], output_dict=True)

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
        await model_manager.save_model_version(model, metrics)

        logger.info("Entraînement du modèle terminé avec succès")
        await db_manager.log_to_database('INFO', "Entraînement du modèle terminé avec succès")

        return {"detail": "Entraînement du modèle terminé avec succès"}
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Erreur lors de l'entraînement: {str(e)}")
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



# model = None  # Initialise une variable globale pour stocker le modèle

# def load_model():
#     global model
#     model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v") and f.endswith(".pkl")]
#     if not model_files:
#         raise FileNotFoundError("No model versions found in storage.")
    
#     latest_model = max(model_files, key=lambda f: int(f.split("_v")[1].split(".pkl")[0]))
#     model_path = os.path.join(MODEL_DIR, latest_model)
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     logger.info(f"Loaded model from {model_path}")

# # Charge un modèle existant au démarrage si disponible
# try:
#     load_model()
# except FileNotFoundError as e:
#     logger.warning(f"Model not found: {str(e)}. You need to train a model first.")

# # Connexion à la base de données
# def get_db_connection_sync():
#     if IS_TEST:
#         return sqlite3.connect(":memory:")
#     raise Exception("get_db_connection_sync should not be called in production!")

# async def get_db_connection():
#     if IS_TEST:
#         return get_db_connection_sync()
#     else:
#         return await asyncpg.connect(
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


# # Fonction pour sauvegarder le modèle avec versioning
# async def save_model_version(model, metrics):
#     version = datetime.now().strftime("%Y%m%d%H%M%S")
#     model_filename = f"model_v{version}.pkl"
#     model_path = os.path.join(MODEL_DIR, model_filename)
    
#     # Sauvegarder le modèle en local
#     with open(model_path, 'wb') as file:
#         pickle.dump(model, file)

#     # Sauvegarder les informations de version en base de données
#     conn = await get_db_connection()
#     try:
#         query = """
#         INSERT INTO model_versions (version, model_path, metrics, created_at)
#         VALUES ($1, $2, $3, $4)
#         """
#         await conn.execute(query, version, model_path, json.dumps(metrics), datetime.now())
#         logger.info(f"Model version {version} saved at {model_path}")
#     finally:
#         await conn.close()

# # Fonction pour loguer dans la base de données
# def log_to_database_sync(conn, level, message):
#     try:
#         query = "INSERT INTO logs (level, message) VALUES (?, ?)"
#         conn.execute(query, (level, message))
#         conn.commit()
#     except Exception as e:
#         logger.error(f"Failed to log to database: {e}")

# async def log_to_database_async(conn, level, message):
#     try:
#         query = "INSERT INTO logs (level, message) VALUES ($1, $2)"
#         await conn.execute(query, level, message)
#     except Exception as e:
#         logger.error(f"Failed to log to database: {e}")
        
# async def log_to_database(conn, level, message):
#     if IS_TEST:
#         log_to_database_sync(conn, level, message)
#     else:
#         await log_to_database_async(conn, level, message)
        

# # Endpoint pour prédire
# @app.post("/predict/")
# async def predict(request: Request, api_key: str = Depends(get_api_key)):
#     global model
#     if model is None:
#         raise HTTPException(status_code=500, detail="No model available. Please train a model first.")
    
#     conn = get_db_connection() if IS_TEST else await get_db_connection()  # Gestion synchrone/asynchrone
#     try:
#         # Récupération des données CSV du corps de la requête
#         csv_body = await request.body()
#         csv_string = csv_body.decode('utf-8')

#         # Conversion du CSV en DataFrame pandas
#         features_df = pd.read_csv(StringIO(csv_string))
#         features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

#         # Prédiction avec le modèle
#         prediction = model.predict(features_df)
        
#         # Conversion des prédictions en liste pour renvoi
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
#     conn = get_db_connection_sync() if IS_TEST else await get_db_connection()  # Gestion synchrone/asynchrone
#     try:
#         logger.info("Starting training process")

#         # Charger le modèle de base existant
#         base_model_path = os.path.join(MODEL_DIR, "model_v2024.pkl")
#         if not os.path.exists(base_model_path):
#             raise HTTPException(status_code=404, detail="Base model not found.")

#         with open(base_model_path, 'rb') as file:
#             model = pickle.load(file)  # Charger le modèle de base

#         logger.info(f"Loaded base model from {base_model_path}")

#         # Charger les données d'entraînement
#         if IS_TEST:
#             cursor = conn.cursor()
#             cursor.execute("SELECT * FROM model_training")
#             data = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])
#         else:
#             rows = await conn.fetch("SELECT * FROM model_training")
#             data = pd.DataFrame(rows, columns=rows[0].keys())

#         logger.info(f"Loaded training data shape: {data.shape}")

#         X = data.drop(columns=['id', 'url', 'type'])
#         y = data['type']

#         # Vérifier le nombre de classes uniques
#         unique_classes = np.unique(y)
#         if len(unique_classes) < 2:
#             logger.warning(f"Training data contains only one class: {unique_classes[0]}. Adding synthetic data for the missing class.")
#             # Ajouter des données factices pour l'autre classe (exemple pour une classification binaire)
#             if unique_classes[0] == 0:
#                 synthetic_data = pd.DataFrame([[-1] * X.shape[1]], columns=X.columns)  # Ajouter des features factices
#                 X = pd.concat([X, synthetic_data])
#                 y = pd.concat([y, pd.Series([1])])  # Ajouter la classe factice
#             else:
#                 synthetic_data = pd.DataFrame([[-1] * X.shape[1]], columns=X.columns)
#                 X = pd.concat([X, synthetic_data])
#                 y = pd.concat([y, pd.Series([0])])  # Ajouter la classe factice

#             logger.info("Synthetic data added to balance classes.")

#         # Réentraîner le modèle existant
#         model.fit(X, y)  # Réentraînement du modèle de base sur les nouvelles données

#         y_pred = model.predict(X)
#         report = classification_report(
#             y, 
#             y_pred, 
#             target_names=['safe', 'phishing'], 
#             labels=unique_classes,  # Ajuster pour les classes présentes dans les données
#             output_dict=True
#         )

#         metrics = {
#             "accuracy": report['accuracy'],
#             "precision_safe": report['safe']['precision'],
#             "recall_safe": report['safe']['recall'],
#             "f1_safe": report['safe']['f1-score'],
#             "precision_phishing": report['phishing']['precision'],
#             "recall_phishing": report['phishing']['recall'],
#             "f1_phishing": report['phishing']['f1-score']
#         }

#         # Sauvegarder la nouvelle version du modèle avec versioning
#         await save_model_version(model, metrics)

#         logger.info("Model training completed successfully")
#         if IS_TEST:
#             log_to_database_sync(conn, 'INFO', "Model retrained successfully")
#         else:
#             await log_to_database_async(conn, 'INFO', "Model retrained successfully")

#         return {"detail": "Model retrained successfully"}
#     except Exception as e:
#         logger.error(f"Training error: {str(e)}")
#         if IS_TEST:
#             log_to_database_sync(conn, 'ERROR', f"Training error: {str(e)}")
#         else:
#             await log_to_database_async(conn, 'ERROR', f"Training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if IS_TEST:
#             conn.close()
#         else:
#             await conn.close()


# # Fonction pour obtenir la liste des modèles disponibles
# def get_model_choices():
#     model_files = [file for file in os.listdir(MODEL_DIR) if file.endswith(".pkl")]
#     if not model_files:
#         raise HTTPException(status_code=404, detail="Aucun modèle trouvé")
#     return model_files

# # Endpoint pour lister les modèles disponibles
# @app.get("/models/")
# async def list_models():
#     """Retourne les modèles disponibles dans le dossier des modèles"""
#     model_files = get_model_choices()
#     return {"available_models": model_files}

# # Endpoint pour charger un modèle spécifique
# @app.get("/load-model/")
# async def load_model(version: str):
#     """Chargement d'un modèle spécifique"""
#     model_files = get_model_choices()
    
#     # Vérification si le modèle existe dans la liste
#     if version not in model_files:
#         raise HTTPException(status_code=404, detail=f"Modèle {version} introuvable")

#     # Construction du chemin vers le modèle
#     model_path = os.path.join(MODEL_DIR, version)
    
#     if not os.path.exists(model_path):
#         raise HTTPException(status_code=404, detail="Fichier du modèle introuvable")

#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)

#     return {"message": f"Modèle {version} chargé avec succès"}


# # Endpoint pour supprimer un modèle spécifique
# @app.delete("/delete-model/{version}")
# async def delete_model(version: str, api_key: str = Depends(get_api_key)):
#     conn = await get_db_connection()

#     try:
#         # Rechercher le modèle dans la base de données
#         query = "SELECT model_path FROM model_versions WHERE version = $1"
#         result = await conn.fetchrow(query, version)

#         if not result:
#             logger.error(f"Model version {version} not found in the database")
#             await log_to_database_async(conn, 'ERROR', f"Model version {version} not found in the database")
#             raise HTTPException(status_code=404, detail="Model version not found")

#         model_path = result['model_path']
#         logger.debug(f"Found model version {version}, path: {model_path}")

#         # Vérifier si le fichier existe et le supprimer
#         if os.path.exists(model_path):
#             os.remove(model_path)
#             logger.info(f"Model file {model_path} deleted successfully")
#             # Loguer la suppression réussie du fichier
#             await log_to_database_async(conn, 'INFO', f"Model file {model_path} deleted successfully")
#         else:
#             logger.error(f"Model file {model_path} not found on the disk")
#             await log_to_database_async(conn, 'ERROR', f"Model file {model_path} not found on the disk")
#             raise HTTPException(status_code=404, detail="Model file not found")

#         # Supprimer les informations du modèle de la base de données
#         delete_query = "DELETE FROM model_versions WHERE version = $1"
#         await conn.execute(delete_query, version)

#         logger.info(f"Model version {version} deleted from database")

#         return {"message": f"Model version {version} deleted successfully"}

#     except Exception as e:
#         logger.error(f"Error deleting model version {version}: {str(e)}")
#         # Loguer l'erreur de suppression
#         await log_to_database_async(conn, 'ERROR', f"Error deleting model version {version}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error deleting model version {version}")

#     finally:
#         await conn.close()


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