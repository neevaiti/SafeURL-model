import pytest
import sqlite3
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
from api.model.api import app, get_db_connection, MODEL_DIR
import os
import tempfile
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_MODEL_KEY = os.getenv("API_MODEL_KEY")

# Fonction pour exécuter le script SQL pour créer les tables
def run_sql_script(connection, script_path):
    with open(script_path, 'r') as sql_file:
        sql_script = sql_file.read()
    cursor = connection.cursor()
    cursor.executescript(sql_script)
    connection.commit()

# Fixture qui crée une base de données SQLite temporaire pour les tests
@pytest.fixture
def test_db():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    script_path = os.path.join(project_root, "setup", "azure_database", "db_test_create_tables.sql")
    
    connection = sqlite3.connect(":memory:")  # SQLite en mémoire
    run_sql_script(connection, script_path)  # Script SQL pour créer les tables
    yield connection
    connection.close()

# Surcharge la dépendance dans l'API avec la connexion à la base de tests
@pytest.fixture
def client(test_db):
    app.dependency_overrides[get_db_connection] = lambda: test_db

    # Utiliser un répertoire temporaire pour les modèles
    with tempfile.TemporaryDirectory() as temp_model_dir:
        with patch("api.model.api.MODEL_DIR", temp_model_dir):
            with TestClient(app) as client:
                yield client

# Mock pour simuler l'ouverture du fichier 'model.pkl' et le chargement du modèle
@pytest.fixture
def mock_model_loading():
    mock_open_file = patch("builtins.open", mock_open(read_data=b"dummy data"))  # Mock d'un fichier factice
    with mock_open_file as m_open:
        with patch("pickle.load", return_value=MagicMock()) as mock_load:  # Mock pour pickle.load
            yield mock_load  # Retourne le mock

# Test du endpoint d'entraînement avec toutes les colonnes
def test_train_model(client, mock_model_loading):
    headers = {"X-API-Key": API_MODEL_KEY}

    # DataFrame mocké avec toutes les colonnes nécessaires
    mock_data = pd.DataFrame({
        'id': [1, 2],
        'url': ['http://example.com', 'http://test.com'],
        'type': [0, 1],
        'use_of_ip': [0, 1],
        'abnormal_url': [1, 0],
        'count_www': [0, 1],
        'count_point': [2, 3],
        'count_at': [0, 0],
        'count_https': [1, 1],
        'count_http': [0, 1],
        'count_percent': [0, 0],
        'count_question': [1, 0],
        'count_dash': [0, 0],
        'count_equal': [0, 1],
        'count_dir': [1, 2],
        'count_embed_domain': [0, 0],
        'short_url': [0, 1],
        'url_length': [56, 78],
        'hostname_length': [35, 45],
        'sus_url': [0, 1],
        'count_digits': [10, 20],
        'count_letters': [46, 60],
        'fd_length': [15, 20],
        'tld_length': [3, 4]
    })

    # Patch pour simuler la lecture des données dans pd.read_sql_query
    with patch("api.model.api.pd.read_sql_query", return_value=mock_data):
        # Patch pour simuler l'enregistrement du modèle avec pickle.dump
        with patch("api.model.api.pickle.dump") as mock_dump:
            response = client.post("/train/", headers=headers)

            # Vérifie que le statut de la réponse est 200
            assert response.status_code == 200
            assert response.json() == {"detail": "Model trained successfully"}

            # Vérifie que le modèle a bien été sauvegardé (pickle.dump appelé)
            mock_dump.assert_called_once()

# Test du endpoint de prédiction
def test_predict_model(client, mock_model_loading):
    headers = {"X-API-Key": API_MODEL_KEY}

    # Simuler l'appel à l'API de prédiction avec un CSV en entrée
    test_csv_data = "use_of_ip,abnormal_url,count_www,count_point,count_at,count_https,count_http,count_percent,count_question,count_dash,count_equal,count_dir,count_embed_domain,short_url,url_length,hostname_length,sus_url,count_digits,count_letters,fd_length,tld_length\n0,1,0,2,0,1,0,0,1,0,0,1,0,0,56,35,0,10,46,15,3\n"
    
    # Patch pour simuler la prédiction du modèle
    with patch("api.model.api.model.predict", return_value=[0]):  # Simuler une prédiction 'safe' (0)
        response = client.post("/predict/", data=test_csv_data, headers=headers)

        # Vérifie que la réponse est correcte
        assert response.status_code == 200
        assert response.json() == {"prediction": [0]}
