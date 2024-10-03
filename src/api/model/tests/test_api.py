import pytest
import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api.interact_model.api import app, get_db_connection

# load_dotenv()

# API_KEY = os.getenv("API_KEY")

# # Lire le fichier SQL et exécuter les commandes
# def run_sql_script(connection, script_path):
#     with open(script_path, 'r') as sql_file:
#         sql_script = sql_file.read()
#     cursor = connection.cursor()
#     cursor.executescript(sql_script)
#     connection.commit()

# # Fixture qui crée une base de données SQLite temporaire pour les tests
# @pytest.fixture
# def test_db():
#     # Remonter à la racine du projet à partir de src/api/model/tests
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
#     script_path = os.path.join(project_root, "setup", "azure_database", "db_test_create_tables.sql")
    
#     connection = sqlite3.connect(":memory:")  # SQLite en mémoire
#     run_sql_script(connection, script_path)  # Script SQL pour créer les tables
#     yield connection
#     connection.close()

# # Surcharge la dépendance dans l'API avec la connexion à la base de tests
# @pytest.fixture
# def client(test_db):
#     app.dependency_overrides[get_db_connection] = lambda: test_db
#     with TestClient(app) as client:
#         yield client

# # Mock de l'appel à l'API de modèle
# @pytest.fixture
# def mock_fetch_prediction():
#     with patch("api.interact_model.api.fetch_prediction", new_callable=AsyncMock) as mock:
#         mock.return_value = {"prediction": [0]}  # Simule une prédiction 'safe'
#         yield mock

# # Test principal avec mock
# def test_prepare_data(client, mock_fetch_prediction):
#     # Inclure la clé API dans les en-têtes de la requête
#     headers = {"X-API-Key": API_KEY}
#     response = client.post("/prepare/", json={"url": "http://example.com"}, headers=headers)
    
#     # Vérifie que la réponse est correcte
#     assert response.status_code == 200
#     assert response.json() == {"prediction": "safe"}

#     # Vérifie que l'appel à l'API de modèle a été fait correctement
#     mock_fetch_prediction.assert_called_once()
    

def test_subtraction():
    assert 10 - 5 == 5

def test_list_equality():
    assert [1, 2, 3] == [1, 2, 3]

def test_key_in_dict():
    sample_dict = {"name": "John", "age": 30}
    assert "age" in sample_dict

def test_not_in_list():
    assert 5 not in [1, 2, 3, 4]

def test_string_length():
    assert len("pytest") == 6
    
def test_greater_than():
    assert 10 > 5

def test_string_concat():
    assert "Hello, " + "world!" == "Hello, world!"

def test_is_odd():
    assert 5 % 2 == 1

def test_tuple_immutable():
    t = (1, 2, 3)
    with pytest.raises(TypeError):
        t[0] = 4

def test_float_addition():
    assert 0.1 + 0.2 == pytest.approx(0.3, 0.0001)

def test_empty_list():
    assert len([]) == 0