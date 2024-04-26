import requests
import os
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from scripts.script_fengineering import extract_features
from scripts.clean_url import clean_url

load_dotenv()

app = FastAPI()


API_KEY = os.getenv("API_KEY")
API_MODEL_KEY = os.getenv("API_MODEL_KEY")


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """
    A middleware function that verifies the API key in the request headers.
    
    Parameters:
        request (Request): The incoming HTTP request.
        call_next (callable): The callback function to proceed with the request handling.
    
    Returns:
        The HTTP response after verifying the API key.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    response = await call_next(request)
    return response


@app.post("/prepare/")
async def prepare_data(url: str):
    """
    A function to prepare and send data to a model API for prediction.
    
    Parameters:
    url (str): The URL to fetch data from.
    
    Returns:
    dict: The JSON response from the model API if successful, otherwise raises an HTTPException.
    """
    url = clean_url(url)
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    csv_string = features_df.to_csv(index=False)
    
    model_api_url = "http://api_model:8050/predict/"
    
    headers = {
        "X-API-Key": API_MODEL_KEY
    }

    response = requests.post(model_api_url, headers=headers, data=csv_string)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="Model API request failed")
