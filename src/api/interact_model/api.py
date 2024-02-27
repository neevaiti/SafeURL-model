from fastapi import FastAPI, HTTPException
import requests
import pandas as pd

from scripts.script_fengineering import extract_features
from scripts.clean_url import clean_url

app = FastAPI()

@app.post("/prepare/")
async def prepare_data(url: str):
    url = clean_url(url)
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    csv_string = features_df.to_csv(index=False)
    
    model_api_url = "http://api_model:8050/predict/"
    
    response = requests.post(model_api_url, data=csv_string)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail="Model API request failed")
