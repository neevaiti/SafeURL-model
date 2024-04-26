from fastapi import FastAPI, HTTPException, Request
import pickle
import pandas as pd
import os
from io import StringIO
from dotenv import load_dotenv

from scripts.read_result import read_result


load_dotenv()

app = FastAPI()


API_MODEL_KEY = os.getenv("API_MODEL_KEY")

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

    
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
    if api_key != API_MODEL_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    response = await call_next(request)
    return response


@app.post("/predict/")
async def predict(request: Request):
    """
    Asynchronous function to handle POST requests for prediction. 
    Takes a Request object as input. 
    Returns a dictionary containing the prediction result.
    """
    csv_body = await request.body()
    csv_string = csv_body.decode('utf-8')
    features_df = pd.read_csv(StringIO(csv_string))
    reading_columns = features_df.columns
    print(reading_columns)
    prediction = model.predict(features_df)
    prediction = read_result(prediction)
    
    return {"prediction": prediction}
