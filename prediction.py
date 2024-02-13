import pickle
import pandas as pd
from src.script_fengineering import extract_features
from src.clean_url import clean_url
from src.read_result import read_result


model_path = './notebook/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

input_url = input("Entrez l'URL : ")

cleaned_url = clean_url(input_url)
print(cleaned_url)

# Extraire les caractéristiques de l'URL nettoyée
features = extract_features(input_url)

# Convertir les caractéristiques en DataFrame pour la prédiction
features_df = pd.DataFrame([features])
features_df.to_csv('features.csv', index=False)


prediction = model.predict(features_df)


result_prediction = read_result(prediction)


print(f"La prédiction pour l'URL '{input_url}' est : {result_prediction}")
