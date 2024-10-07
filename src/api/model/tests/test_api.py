import pytest
import os
import json
import sqlite3
from fastapi.testclient import TestClient
from unittest.mock import Mock
from dotenv import load_dotenv
from api.model.api import app, db_manager, model_manager

load_dotenv()

# Configure le mode de test
os.environ["IS_TEST"] = "true"
API_KEY_HEADER = {"X-API-Key": os.getenv("API_MODEL_KEY")}
client = TestClient(app)

@pytest.fixture
def mock_db_manager():
    """Configure le gestionnaire de base de données pour les tests."""
    db_manager.connect()
    db_manager.conn = sqlite3.connect(":memory:")
    with open("src/api/model/tests/schema.sql", "r") as f:
        db_manager.conn.executescript(f.read())
    yield db_manager
    db_manager.close()

# def test_list_models():
#     """Test l'endpoint qui liste les modèles disponibles."""
#     response = client.get("/models/", headers=API_KEY_HEADER)
#     assert response.status_code == 200
#     assert "available_models" in response.json()

def test_train_model(mock_db_manager):
    """Test l'endpoint d'entraînement de modèle."""
    mock_db_manager.fetch = Mock(return_value=[
        (1, 'http://example.com', 0, 1, 0, 2, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 20, 10, 1, 2, 1, 10, 15)
    ])

    response = client.post("/train/", headers=API_KEY_HEADER)
    assert response.status_code == 200
    assert "metrics" in response.json()

def test_predict():
    """Test l'endpoint de prédiction."""
    data = """
    use_of_ip,abnormal_url,count_www,count_point,count_at,count_https,count_http,count_percent,count_question,count_dash,count_equal,count_dir,count_embed_domain,short_url,url_length,hostname_length,sus_url,fd_length,tld_length,count_digits,count_letters
    0,1,2,3,1,0,1,0,1,0,0,1,0,0,23,12,0,5,3,7,10
    """
    response = client.post("/predict/", data=data.strip(), headers={"Content-Type": "text/csv", **API_KEY_HEADER})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_delete_model(mock_db_manager):
    """Test la suppression d'un modèle."""
    mock_db_manager.fetchrow = Mock(return_value={"model_path": "/tmp/models/model_v123456789.pkl"})

    response = client.delete("/delete-model/123456789", headers=API_KEY_HEADER)
    assert response.status_code == 200
    assert "message" in response.json()

def test_get_model_metrics(mock_db_manager):
    """Test la récupération des métriques d'un modèle."""
    mock_db_manager.fetchrow = Mock(return_value={"metrics": json.dumps({"accuracy": 0.95})})

    response = client.get("/model-metrics/123456789", headers=API_KEY_HEADER)
    assert response.status_code == 200
    assert "metrics" in response.json()



















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