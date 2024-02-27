from fastapi import FastAPI, HTTPException, Request
import pickle
import pandas as pd
from io import StringIO


app = FastAPI()

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    

def read_result(prediction):
    """
    Function to interpret the prediction and return the corresponding response.
    Takes a prediction as input and returns a string response.
    """
    if int(prediction[0]) == 0:
        response="Bénin"
        return response
    elif int(prediction[0]) == 1:
        response="Scam"
        return response
    elif int(prediction[0]) == 2:
        response="Phishing"
        return response
    elif int(prediction[0]) == 3:
        response="Malware"
        return response

@app.post("/predict/")
async def predict(request: Request):
    csv_body = await request.body()
    csv_string = csv_body.decode('utf-8')
    features_df = pd.read_csv(StringIO(csv_string))
    prediction = model.predict(features_df)
    prediction = read_result(prediction)
    
    
    return {"prediction": prediction}
