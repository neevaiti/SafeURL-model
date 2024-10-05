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

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

API_MODEL_KEY = os.getenv("API_MODEL_KEY")
IS_TEST = os.getenv("IS_TEST", "false").lower() == "true"

# Model storage path
MODEL_DIR = "/app/models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logger.debug(f"Model directory set to: {MODEL_DIR}")

# Class for model management
class ModelManager:
    """
    Class for managing models.

    Attributes:
        model_dir (str): The directory where models are stored.
        model (Any): The currently loaded model in memory.
    """

    def __init__(self, model_dir):
        """
        Initialize the model manager with the specified directory.

        Args:
            model_dir (str): The directory where models are stored.
        """
        self.model_dir = model_dir
        self.model = None
        os.makedirs(self.model_dir, exist_ok=True)

    def load_latest_model(self):
        """
        Load the latest model from the model directory.

        Raises:
            FileNotFoundError: If no model is found in the directory.
        """
        model_files = sorted(
            [f for f in os.listdir(self.model_dir) if f.startswith("model_v") and f.endswith(".pkl")],
            key=lambda f: int(f.split("_v")[1].split(".pkl")[0]),
            reverse=True
        )
        if not model_files:
            raise FileNotFoundError("No model found in the model directory.")

        latest_model = model_files[0]
        model_path = os.path.join(self.model_dir, latest_model)
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        logger.info(f"Model loaded from {model_path}")

    def load_model(self, version: str):
        """
        Load a specific model based on the given version.

        Args:
            version (str): The version of the model to load.

        Returns:
            str: The path of the loaded model.

        Raises:
            FileNotFoundError: If the specified model is not found.
        """
        model_path = os.path.join(self.model_dir, version)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {version} not found.")

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        logger.info(f"Model loaded from {model_path}")
        return model_path

    def get_model(self):
        """
        Return the model in memory or load the latest available model.

        Returns:
            Any: The currently loaded model.
        """
        if self.model is None:
            self.load_latest_model()
        return self.model

    def get_available_models(self):
        """
        Return the list of available models in the directory.

        Returns:
            list: List of available model files.
        """
        return [f for f in os.listdir(self.model_dir) if f.startswith("model_v") and f.endswith(".pkl")]

    async def save_model_version(self, model, metrics):
        """
        Save the model with versioning in the model directory and store version information in the database.

        Args:
            model (Any): The model to save.
            metrics (dict): The metrics associated with the model.

        Raises:
            Exception: If an error occurs during saving.
        """
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        model_filename = f"model_v{version}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)

        try:
            logger.debug(f"Attempting to save model to {model_path}")
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            logger.info(f"Model version {model_filename} saved to {model_path}")

            query = """
            INSERT INTO model_versions (version, model_path, metrics, created_at)
            VALUES ($1, $2, $3, $4)
            """
            await db_manager.conn.execute(query, model_filename, model_path, json.dumps(metrics), datetime.now())
        except Exception as e:
            logger.error(f"Error while saving model: {str(e)}")
            raise

# Class for log management and database connection
class DatabaseManager:
    """
    Class for managing logs and database connection.
    """

    def __init__(self):
        """Initialize the database manager."""
        self.conn = None

    async def connect(self):
        """
        Initialize the database connection.

        Uses an in-memory database for testing, otherwise connects to a PostgreSQL database.
        """
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
        """Close the database connection."""
        if self.conn:
            if IS_TEST:
                self.conn.close()
            else:
                await self.conn.close()

    async def log_to_database(self, level, message):
        """
        Log messages to the database.

        Args:
            level (str): The log level (e.g., 'INFO', 'ERROR').
            message (str): The log message.
        """
        query = "INSERT INTO logs (level, message) VALUES ($1, $2)" if not IS_TEST else "INSERT INTO logs (level, message) VALUES (?, ?)"
        try:
            if IS_TEST:
                cursor = self.conn.cursor()
                cursor.execute(query, (level, message))
                self.conn.commit()
            else:
                await self.conn.execute(query, level, message)
        except Exception as e:
            logger.error(f"Failed to log message to the database: {e}")

    async def fetch(self, query, *args):
        """
        Fetch results from the database based on an SQL query.

        Args:
            query (str): The SQL query to execute.
            *args: The arguments for the SQL query.

        Returns:
            list: The query results.

        Raises:
            Exception: If an error occurs while fetching data.
        """
        try:
            if IS_TEST:
                cursor = self.conn.cursor()
                cursor.execute(query, args)
                return cursor.fetchall()
            else:
                return await self.conn.fetch(query, *args)
        except Exception as e:
            logger.error(f"Error while fetching data: {e}")
            raise

# Initialize model and database managers
model_manager = ModelManager(MODEL_DIR)
db_manager = DatabaseManager()

# Middleware to verify API key
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """
    Middleware to verify the API key in HTTP requests.

    Args:
        request (Request): The HTTP request object.
        call_next (Callable): The function to call the next middleware step.

    Returns:
        Response: The HTTP response after API key verification.
    """
    excluded_paths = ["/docs", "/openapi.json", "/redoc"]
    if request.url.path not in excluded_paths:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_MODEL_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized access")
    return await call_next(request)

# Function to verify and get the API key
async def get_api_key(api_key_header: str = Security(APIKeyHeader(name="X-API-Key", auto_error=False))):
    """
    Verify and get the API key from the request header.

    Args:
        api_key_header (str): The API key provided in the header.

    Returns:
        str: The API key if valid.

    Raises:
        HTTPException: If the API key is invalid.
    """
    if api_key_header == API_MODEL_KEY:
        return api_key_header
    raise HTTPException(status_code=401, detail="Invalid API key")

# Endpoint for prediction
@app.post("/predict/")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    """
    Endpoint to perform a prediction from the provided data.

    Args:
        request (Request): The HTTP request object containing CSV data.
        api_key (str): The validated API key.

    Returns:
        dict: The prediction results.

    Raises:
        HTTPException: If an error occurs during prediction.
    """
    await db_manager.connect()
    model = model_manager.get_model()

    try:
        csv_body = await request.body()
        csv_string = csv_body.decode('utf-8')
        features_df = pd.read_csv(StringIO(csv_string)).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        prediction = model.predict(features_df).tolist()

        logger.info(f"Prediction made: {prediction}")
        await db_manager.log_to_database('INFO', f"Prediction made: {prediction}")

        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during prediction")
    finally:
        await db_manager.close()

# Endpoint to train a new model with descriptive column names
@app.post("/train/")
async def train(api_key: str = Depends(get_api_key)):
    """
    Endpoint to train a new model with available data.

    Args:
        api_key (str): The validated API key.

    Returns:
        dict: Training details and model metrics.

    Raises:
        HTTPException: If an error occurs during training.
    """
    await db_manager.connect()
    try:
        logger.info("Starting model training")

        rows = await db_manager.fetch("SELECT * FROM model_training")

        columns = ['id', 'url', 'type', 'use_of_ip', 'abnormal_url', 'count_www', 'count_point', 'count_at', 
                   'count_https', 'count_http', 'count_percent', 'count_question', 'count_dash', 'count_equal', 
                   'count_dir', 'count_embed_domain', 'short_url', 'url_length', 'hostname_length', 'sus_url', 
                   'fd_length', 'tld_length', 'count_digits', 'count_letters']

        data = pd.DataFrame(rows, columns=columns)
        if data.empty:
            raise HTTPException(status_code=400, detail="No training data available.")

        X = data[['use_of_ip', 'abnormal_url', 'count_point', 'count_www', 'count_at', 'count_dir',
                  'count_embed_domain', 'short_url', 'count_https', 'count_http', 'count_percent', 
                  'count_question', 'count_dash', 'count_equal', 'url_length', 'hostname_length',
                  'sus_url', 'fd_length', 'tld_length', 'count_digits', 'count_letters']]
        y = data['type']

        model = RandomForestClassifier()
        model.fit(X, y)

        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)

        metrics_json = json.dumps({"metrics": report})
        await model_manager.save_model_version(model, metrics_json)
        logger.info(f"Metrics saved: {metrics_json}")

        logger.info("Model training completed successfully")
        return {"detail": "Model training completed successfully", "metrics": report}
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await db_manager.close()

# Endpoint to delete a specific model
@app.delete("/delete-model/{version}")
async def delete_model(version: str, api_key: str = Depends(get_api_key)):
    """
    Endpoint to delete a specific model.

    Args:
        version (str): The version of the model to delete.
        api_key (str): The validated API key.

    Returns:
        dict: Success message for deletion.

    Raises:
        HTTPException: If an error occurs during deletion.
    """
    await db_manager.connect()

    try:
        query = "SELECT model_path FROM model_versions WHERE version = $1"
        result = await db_manager.conn.fetchrow(query, version)

        if not result:
            logger.error(f"Model version {version} not found in the database")
            await db_manager.log_to_database('ERROR', f"Model version {version} not found in the database")
            raise HTTPException(status_code=404, detail="Model version not found")

        model_path = result['model_path']
        logger.debug(f"Model version found: {version}, path: {model_path}")

        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Model file {model_path} deleted successfully")
            await db_manager.log_to_database('INFO', f"Model file {model_path} deleted successfully")
        else:
            logger.error(f"Model file {model_path} not found on disk")
            await db_manager.log_to_database('ERROR', f"Model file {model_path} not found on disk")
            raise HTTPException(status_code=404, detail="Model file not found")

        delete_query = "DELETE FROM model_versions WHERE version = $1"
        await db_manager.conn.execute(delete_query, version)

        logger.info(f"Model version {version} deleted from the database")
        await db_manager.log_to_database('INFO', f"Model version {version} deleted from the database")

        return {"message": f"Model version {version} deleted successfully"}

    except Exception as e:
        logger.error(f"Error during deletion: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Error during deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during deletion: {str(e)}")

    finally:
        await db_manager.close()

# Endpoint to retrieve metrics of a specific model
@app.get("/model-metrics/{version}")
async def get_model_metrics(version: str, api_key: str = Depends(get_api_key)):
    """
    Endpoint to retrieve metrics of a specific model.

    Args:
        version (str): The version of the model to retrieve metrics for.
        api_key (str): The validated API key.

    Returns:
        dict: The model metrics.

    Raises:
        HTTPException: If an error occurs during metric retrieval.
    """
    await db_manager.connect()
    try:
        query = "SELECT metrics FROM model_versions WHERE version = $1"
        result = await db_manager.conn.fetchrow(query, version)
        
        if not result:
            logger.error(f"No metrics found for version {version}")
            raise HTTPException(status_code=404, detail="Model version not found")
        
        metrics = result['metrics']
        logger.info(f"Metrics retrieved for version {version}: {metrics}")
        return {"metrics": json.loads(metrics)}
    except Exception as e:
        logger.error(f"Error during metric retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during metric retrieval")
    finally:
        await db_manager.close()

# Endpoint to list available models
@app.get("/models/")
async def list_models():
    """
    Endpoint to list available models.

    Returns:
        dict: The available models in the model directory.
    """
    model_files = model_manager.get_available_models()
    return {"available_models": model_files}

@app.get("/load-model/")
async def load_specific_model(version: str):
    """
    Load a specific model based on the version.

    Args:
        version (str): The version of the model to load.

    Returns:
        dict: Success message for loading the model.

    Raises:
        HTTPException: If an error occurs during model loading.
    """
    await db_manager.connect()

    try:
        logger.info(f"Attempting to load model version {version}")
        await db_manager.log_to_database('INFO', f"Attempting to load model version {version}")

        model_path = model_manager.load_model(version)

        logger.info(f"Model {version} loaded successfully from {model_path}")
        await db_manager.log_to_database('INFO', f"Model {version} loaded successfully from {model_path}")

        return {"message": f"Model {version} loaded successfully from {model_path}"}
    
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Error while loading model {version}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error while loading model {version}: {str(e)}")
        await db_manager.log_to_database('ERROR', f"Unexpected error while loading model {version}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error while loading model")

    finally:
        await db_manager.close()

def custom_openapi():
    """
    Customize the OpenAPI schema to include API key security.

    Returns:
        dict: The customized OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Model API",
        version="1.0.0",
        description="API for predicting and training a model",
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