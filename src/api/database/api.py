import os
import asyncpg
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv
import json
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

# Chargement des variables d'environnement
load_dotenv()

DB_API_KEY = os.getenv("DB_API_KEY")  # Clé API chargée depuis le fichier .env
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialisation de l'application FastAPI
app = FastAPI()

# Classe pour la gestion de la base de données
class DatabaseManager:
    """
    Classe pour gérer la connexion à la base de données PostgreSQL.
    """

    def __init__(self):
        """Initialise le gestionnaire de base de données sans connexion active."""
        self.conn = None

    async def connect(self):
        """Initialise la connexion à la base de données."""
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
            await self.conn.close()

db_manager = DatabaseManager()

async def get_db():
    """
    Fournit une connexion à la base de données pour les opérations asynchrones.

    Yields:
        asyncpg.Connection: Connexion à la base de données.
    """
    await db_manager.connect()
    try:
        yield db_manager.conn
    finally:
        await db_manager.close()

# Modèles Pydantic
class ModelTraining(BaseModel):
    """
    Modèle de données pour l'entraînement du modèle.

    Attributes:
        url (str): L'URL à analyser.
        type (int): Le type de l'URL (e.g., phishing, safe).
        use_of_ip (int): Indicateur d'utilisation d'une adresse IP.
        abnormal_url (int): Indicateur d'URL anormale.
        count_www (int): Nombre de 'www' dans l'URL.
        count_point (int): Nombre de points dans l'URL.
        ...
    """
    url: str
    type: int
    use_of_ip: int
    abnormal_url: int
    count_www: int
    count_point: int
    count_at: int
    count_https: int
    count_http: int
    count_percent: int
    count_question: int
    count_dash: int
    count_equal: int
    count_dir: int
    count_embed_domain: int
    short_url: int
    url_length: int
    hostname_length: int
    sus_url: int
    count_digits: int
    count_letters: int
    fd_length: int
    tld_length: int

class ListUrl(BaseModel):
    """
    Modèle de données pour la liste des URLs.

    Attributes:
        url (str): L'URL à enregistrer.
        type (str): Le type de l'URL (e.g., phishing, safe).
    """
    url: str
    type: str

class Log(BaseModel):
    """
    Modèle de données pour les logs.

    Attributes:
        level (str): Le niveau du log (e.g., INFO, ERROR).
        message (str): Le message du log.
    """
    level: str
    message: str

class ModelVersion(BaseModel):
    """
    Modèle de données pour les versions de modèles.

    Attributes:
        version (str): La version du modèle.
        model_path (str): Le chemin du modèle.
        metrics (Optional[Dict]): Les métriques associées au modèle.
    """
    version: str
    model_path: str
    metrics: Optional[Dict]
    
    class Config:
        protected_namespaces = ()

# Vérification de la clé API
async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Vérifie la clé API fournie dans l'en-tête.

    Args:
        api_key_header (str): La clé API fournie.

    Returns:
        str: La clé API validée.

    Raises:
        HTTPException: Si la clé API est invalide ou absente.
    """
    if api_key_header == DB_API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=401, detail="Clé API invalide ou absente")

# Middleware pour sécuriser toutes les requêtes sauf la documentation
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """
    Middleware pour vérifier la clé API dans les requêtes HTTP.

    Args:
        request (Request): La requête HTTP.
        call_next (Callable): La fonction suivante à appeler dans la chaîne de middleware.

    Returns:
        Response: La réponse HTTP après vérification de la clé API.
    """
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if api_key != DB_API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Clé API invalide ou absente"})
    
    response = await call_next(request)
    return response

# Endpoints CRUD pour `model_training`
@app.post("/model_training/")
async def create_model_training(model: ModelTraining, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Crée un nouvel enregistrement d'entraînement de modèle.

    Args:
        model (ModelTraining): Les données d'entraînement du modèle.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: L'identifiant de l'enregistrement créé.
    """
    query = """
    INSERT INTO model_training (url, type, use_of_ip, abnormal_url, count_www, count_point)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id
    """
    try:
        result = await db.fetchval(query, model.url, model.type, model.use_of_ip, model.abnormal_url, model.count_www, model.count_point)
        return {"id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_training/{id}", response_model=ModelTraining)
async def get_model_training(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Récupère un enregistrement d'entraînement de modèle par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        ModelTraining: Les données de l'enregistrement.
    """
    query = "SELECT * FROM model_training WHERE id = $1"
    result = await db.fetchrow(query, id)
    if not result:
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
    return dict(result)

@app.put("/model_training/{id}")
async def update_model_training(id: int, model: ModelTraining, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Met à jour un enregistrement d'entraînement de modèle par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        model (ModelTraining): Les nouvelles données d'entraînement du modèle.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: Message de succès de la mise à jour.
    """
    query = """
    UPDATE model_training
    SET url = $1, type = $2, use_of_ip = $3, abnormal_url = $4, count_www = $5, count_point = $6
    WHERE id = $7
    """
    await db.execute(query, model.url, model.type, model.use_of_ip, model.abnormal_url, model.count_www, model.count_point, id)
    return {"message": "Mise à jour réussie"}

@app.delete("/model_training/{id}")
async def delete_model_training(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Supprime un enregistrement d'entraînement de modèle par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: Message de succès de la suppression.
    """
    query = "DELETE FROM model_training WHERE id = $1"
    result = await db.execute(query, id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
    return {"message": "Suppression réussie"}

# Endpoints CRUD pour `list_url`
@app.post("/list_url/")
async def create_list_url(list_url: ListUrl, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Crée un nouvel enregistrement dans la liste des URLs.

    Args:
        list_url (ListUrl): Les données de l'URL à enregistrer.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: L'identifiant de l'enregistrement créé.
    """
    query = "INSERT INTO list_url (url, type) VALUES ($1, $2) RETURNING id"
    try:
        result = await db.fetchval(query, list_url.url, list_url.type)
        return {"id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_url/{id}", response_model=ListUrl)
async def get_list_url(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Récupère un enregistrement de la liste des URLs par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        ListUrl: Les données de l'enregistrement.
    """
    query = "SELECT * FROM list_url WHERE id = $1"
    result = await db.fetchrow(query, id)
    if not result:
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
    return dict(result)

@app.put("/list_url/{id}")
async def update_list_url(id: int, list_url: ListUrl, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Met à jour un enregistrement de la liste des URLs par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        list_url (ListUrl): Les nouvelles données de l'URL.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: Message de succès de la mise à jour.
    """
    query = """
    UPDATE list_url
    SET url = $1, type = $2
    WHERE id = $3
    """
    await db.execute(query, list_url.url, list_url.type, id)
    return {"message": "Mise à jour réussie"}

@app.delete("/list_url/{id}")
async def delete_list_url(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Supprime un enregistrement de la liste des URLs par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: Message de succès de la suppression.
    """
    query = "DELETE FROM list_url WHERE id = $1"
    result = await db.execute(query, id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Enregistrement non trouvé")
    return {"message": "Suppression réussie"}

# Endpoints CRUD pour `logs`
@app.post("/logs/")
async def create_log(log: Log, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Crée un nouvel enregistrement de log.

    Args:
        log (Log): Les données du log à enregistrer.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: L'identifiant de l'enregistrement créé.
    """
    query = "INSERT INTO logs (level, message) VALUES ($1, $2) RETURNING id"
    try:
        result = await db.fetchval(query, log.level, log.message)
        return {"id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/")
async def get_logs(db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Récupère les 100 derniers enregistrements de logs.

    Args:
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        list: Liste des enregistrements de logs.
    """
    query = "SELECT * FROM logs ORDER BY date DESC LIMIT 100"
    logs = await db.fetch(query)
    return [dict(log) for log in logs]

# Endpoints CRUD pour `model_versions`
@app.post("/model_versions/")
async def create_model_version(model_version: ModelVersion, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Crée un nouvel enregistrement de version de modèle.

    Args:
        model_version (ModelVersion): Les données de la version du modèle.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: L'identifiant de l'enregistrement créé.
    """
    query = """
    INSERT INTO model_versions (version, model_path, metrics, created_at)
    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
    RETURNING id
    """
    try:
        result = await db.fetchval(query, model_version.version, model_version.model_path, json.dumps(model_version.metrics))
        return {"id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_versions/{id}", response_model=ModelVersion)
async def get_model_version(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Récupère un enregistrement de version de modèle par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        ModelVersion: Les données de l'enregistrement.
    """
    query = "SELECT * FROM model_versions WHERE id = $1"
    result = await db.fetchrow(query, id)
    if not result:
        raise HTTPException(status_code=404, detail="Version de modèle non trouvée")
    return dict(result)

@app.put("/model_versions/{id}")
async def update_model_version(id: int, model_version: ModelVersion, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Met à jour un enregistrement de version de modèle par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        model_version (ModelVersion): Les nouvelles données de la version du modèle.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: Message de succès de la mise à jour.
    """
    query = """
    UPDATE model_versions
    SET version = $1, model_path = $2, metrics = $3
    WHERE id = $4
    """
    await db.execute(query, model_version.version, model_version.model_path, json.dumps(model_version.metrics), id)
    return {"message": "Mise à jour réussie"}

@app.delete("/model_versions/{id}")
async def delete_model_version(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Supprime un enregistrement de version de modèle par ID.

    Args:
        id (int): L'identifiant de l'enregistrement.
        db (asyncpg.Connection): Connexion à la base de données.
        api_key (str): La clé API validée.

    Returns:
        dict: Message de succès de la suppression.
    """
    query = "DELETE FROM model_versions WHERE id = $1"
    result = await db.execute(query, id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Version de modèle non trouvée")
    return {"message": "Suppression réussie"}

# Personnalisation de la documentation pour inclure la clé API
def custom_openapi():
    """
    Personnalise le schéma OpenAPI pour inclure la sécurité de la clé API.

    Returns:
        dict: Le schéma OpenAPI personnalisé.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API Database",
        version="1.0.0",
        description="API Database CRUD pour l'application SafeURL",
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