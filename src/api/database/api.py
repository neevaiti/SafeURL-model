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

# Load environment variables
load_dotenv()

DB_API_KEY = os.getenv("DB_API_KEY")  # API key loaded from .env file
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize FastAPI application
app = FastAPI()

# Database management class
class DatabaseManager:
    """
    Class to manage PostgreSQL database connection.
    """

    def __init__(self):
        """Initialize the database manager without an active connection."""
        self.conn = None

    async def connect(self):
        """Establish a connection to the database."""
        self.conn = await asyncpg.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME")
        )

    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()

db_manager = DatabaseManager()

async def get_db():
    """
    Provide a database connection for asynchronous operations.

    Yields:
        asyncpg.Connection: Database connection.
    """
    await db_manager.connect()
    try:
        yield db_manager.conn
    finally:
        await db_manager.close()

# Pydantic models
class ModelTraining(BaseModel):
    """
    Data model for model training.

    Attributes:
        url (str): The URL to analyze.
        type (int): The type of the URL (e.g., phishing, safe).
        use_of_ip (int): Indicator of IP address usage.
        abnormal_url (int): Indicator of abnormal URL.
        count_www (int): Number of 'www' in the URL.
        count_point (int): Number of dots in the URL.
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
    Data model for URL list.

    Attributes:
        url (str): The URL to record.
        type (str): The type of the URL (e.g., phishing, safe).
    """
    url: str
    type: str

class Log(BaseModel):
    """
    Data model for logs.

    Attributes:
        level (str): The log level (e.g., INFO, ERROR).
        message (str): The log message.
    """
    level: str
    message: str

class ModelVersion(BaseModel):
    """
    Data model for model versions.

    Attributes:
        version (str): The model version.
        model_path (str): The model path.
        metrics (Optional[Dict]): Metrics associated with the model.
    """
    version: str
    model_path: str
    metrics: Optional[Dict]
    
    class Config:
        protected_namespaces = ()

# API key verification
async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Verify the API key provided in the header.

    Args:
        api_key_header (str): The provided API key.

    Returns:
        str: The validated API key.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """
    if api_key_header == DB_API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Middleware to secure all requests except documentation
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """
    Middleware to verify the API key in HTTP requests.

    Args:
        request (Request): The HTTP request.
        call_next (Callable): The next function to call in the middleware chain.

    Returns:
        Response: The HTTP response after API key verification.
    """
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if api_key != DB_API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    
    response = await call_next(request)
    return response

# CRUD endpoints for `model_training`
@app.post("/model_training/")
async def create_model_training(model: ModelTraining, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Create a new model training record.

    Args:
        model (ModelTraining): The model training data.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: The ID of the created record.
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
    Retrieve a model training record by ID.

    Args:
        id (int): The record ID.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        ModelTraining: The record data.
    """
    query = "SELECT * FROM model_training WHERE id = $1"
    result = await db.fetchrow(query, id)
    if not result:
        raise HTTPException(status_code=404, detail="Record not found")
    return dict(result)

@app.put("/model_training/{id}")
async def update_model_training(id: int, model: ModelTraining, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Update a model training record by ID.

    Args:
        id (int): The record ID.
        model (ModelTraining): The new model training data.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: Success message of the update.
    """
    query = """
    UPDATE model_training
    SET url = $1, type = $2, use_of_ip = $3, abnormal_url = $4, count_www = $5, count_point = $6
    WHERE id = $7
    """
    await db.execute(query, model.url, model.type, model.use_of_ip, model.abnormal_url, model.count_www, model.count_point, id)
    return {"message": "Update successful"}

@app.delete("/model_training/{id}")
async def delete_model_training(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Delete a model training record by ID.

    Args:
        id (int): The record ID.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: Success message of the deletion.
    """
    query = "DELETE FROM model_training WHERE id = $1"
    result = await db.execute(query, id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Record not found")
    return {"message": "Deletion successful"}

# CRUD endpoints for `list_url`
@app.post("/list_url/")
async def create_list_url(list_url: ListUrl, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Create a new record in the URL list.

    Args:
        list_url (ListUrl): The URL data to record.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: The ID of the created record.
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
    Retrieve a URL list record by ID.

    Args:
        id (int): The record ID.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        ListUrl: The record data.
    """
    query = "SELECT * FROM list_url WHERE id = $1"
    result = await db.fetchrow(query, id)
    if not result:
        raise HTTPException(status_code=404, detail="Record not found")
    return dict(result)

@app.put("/list_url/{id}")
async def update_list_url(id: int, list_url: ListUrl, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Update a URL list record by ID.

    Args:
        id (int): The record ID.
        list_url (ListUrl): The new URL data.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: Success message of the update.
    """
    query = """
    UPDATE list_url
    SET url = $1, type = $2
    WHERE id = $3
    """
    await db.execute(query, list_url.url, list_url.type, id)
    return {"message": "Update successful"}

@app.delete("/list_url/{id}")
async def delete_list_url(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Delete a URL list record by ID.

    Args:
        id (int): The record ID.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: Success message of the deletion.
    """
    query = "DELETE FROM list_url WHERE id = $1"
    result = await db.execute(query, id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Record not found")
    return {"message": "Deletion successful"}

# CRUD endpoints for `logs`
@app.post("/logs/")
async def create_log(log: Log, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Create a new log record.

    Args:
        log (Log): The log data to record.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: The ID of the created record.
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
    Retrieve the last 100 log records.

    Args:
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        list: List of log records.
    """
    query = "SELECT * FROM logs ORDER BY date DESC LIMIT 100"
    logs = await db.fetch(query)
    return [dict(log) for log in logs]

# CRUD endpoints for `model_versions`
@app.post("/model_versions/")
async def create_model_version(model_version: ModelVersion, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Create a new model version record.

    Args:
        model_version (ModelVersion): The model version data.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: The ID of the created record.
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
    Retrieve a model version record by ID.

    Args:
        id (int): The record ID.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        ModelVersion: The record data.
    """
    query = "SELECT * FROM model_versions WHERE id = $1"
    result = await db.fetchrow(query, id)
    if not result:
        raise HTTPException(status_code=404, detail="Model version not found")
    return dict(result)

@app.put("/model_versions/{id}")
async def update_model_version(id: int, model_version: ModelVersion, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Update a model version record by ID.

    Args:
        id (int): The record ID.
        model_version (ModelVersion): The new model version data.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: Success message of the update.
    """
    query = """
    UPDATE model_versions
    SET version = $1, model_path = $2, metrics = $3
    WHERE id = $4
    """
    await db.execute(query, model_version.version, model_version.model_path, json.dumps(model_version.metrics), id)
    return {"message": "Update successful"}

@app.delete("/model_versions/{id}")
async def delete_model_version(id: int, db = Depends(get_db), api_key: str = Depends(get_api_key)):
    """
    Delete a model version record by ID.

    Args:
        id (int): The record ID.
        db (asyncpg.Connection): Database connection.
        api_key (str): The validated API key.

    Returns:
        dict: Success message of the deletion.
    """
    query = "DELETE FROM model_versions WHERE id = $1"
    result = await db.execute(query, id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Model version not found")
    return {"message": "Deletion successful"}

# Customize documentation to include API key
def custom_openapi():
    """
    Customize the OpenAPI schema to include API key security.

    Returns:
        dict: The customized OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API Database",
        version="1.0.0",
        description="API Database CRUD for the SafeURL application",
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