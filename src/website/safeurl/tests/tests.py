import pytest
from django.urls import reverse
from django.contrib.auth.models import User
from django.test import Client
from unittest.mock import patch
from safeurl.models import Prediction

@pytest.fixture
def user(db):
    """
    Crée un utilisateur pour les tests.
    """
    return User.objects.create_user(username='testuser', password='testpassword')

@pytest.fixture
def client_logged_in(user):
    """
    Retourne un client Django avec l'utilisateur connecté.
    """
    client = Client()
    client.login(username='testuser', password='testpassword')
    return client

def test_predict_view_get_not_logged_in(client):
    """
    Vérifie que l'accès à la vue 'predict' sans être authentifié redirige vers la page de connexion.
    """
    response = client.get(reverse('predict'))
    assert response.status_code == 302  
    assert '/login/' in response.url 

def test_predict_view_get_logged_in(client_logged_in):
    """
    Vérifie qu'une requête GET authentifiée à la vue 'predict' retourne un statut 200.
    """
    response = client_logged_in.get(reverse('predict'))
    assert response.status_code == 200
    assert 'predict.html' in [t.name for t in response.templates]
    

def test_predict_view_post_missing_url(client_logged_in):
    """
    Vérifie que la vue gère correctement une requête POST sans paramètre 'url'.
    """
    response = client_logged_in.post(reverse('predict'), {})
    assert response.status_code == 400 

def test_predict_view_post_success(client_logged_in, user, monkeypatch):
    """
    Vérifie qu'une requête POST valide crée une prédiction et rend le bon template.
    """
    
    def mock_post(url, json, headers):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            def json(self):
                return {'prediction': 'positive'}
        return MockResponse()

    monkeypatch.setattr('safeurl.views.requests.post', mock_post)
    monkeypatch.setattr('safeurl.views', 'API_KEY', 'test_api_key')

    url_to_predict = 'http://example.com'
    response = client_logged_in.post(reverse('predict'), {'url': url_to_predict})

    prediction = Prediction.objects.get(user=user, url=url_to_predict)
    assert prediction.prediction == 'positive'

    assert response.status_code == 200
    assert 'predict.html' in [t.name for t in response.templates]
    assert response.context['prediction'] == 'positive'

