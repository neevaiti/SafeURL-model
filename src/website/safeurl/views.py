from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.views import LogoutView
from django.contrib.admin.views.decorators import staff_member_required

from .models import Prediction
from dotenv import load_dotenv

import requests
import os
import matplotlib.pyplot as plt
import json
import io
import base64
from django.http import HttpResponse
import matplotlib



load_dotenv()

API_KEY = os.getenv('API_KEY')
API_MODEL_KEY = os.getenv('API_MODEL_KEY')

def home(request):
    """
    View for the home page.

    Redirects authenticated users to their respective home page
    (normal user or administrator).

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the home page or a redirect.
    """
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect('admin_home')
        else:
            return render(request, 'base.html')
    return redirect('login')

def user_register(request):
    """
    View for user registration.

    Allows users to register and automatically log in
    after a successful registration.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the registration form or a redirect.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('predict')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    """
    View for user login.

    Authenticates users and redirects them to their respective home page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the login form or a redirect.
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.is_staff:
                return redirect('admin_home') 
            return redirect('predict')
    return render(request, 'login.html')

@login_required
def predict(request):
    """
    View to make predictions on URLs.

    Sends the URL to the interaction API to get a prediction and saves
    the result in the database.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the prediction page.
    """
    if request.method == 'POST':
        url = request.POST['url']
        response = requests.post('http://api_interact:12600/prepare/', json={'url': url}, headers={'X-API-Key': API_KEY})
        prediction = response.json().get('prediction')
        Prediction.objects.create(user=request.user, url=url, prediction=prediction)
        return render(request, 'predict.html', {'prediction': prediction})
    return render(request, 'predict.html')

@login_required
def predictions_list(request):
    """
    View to list the user's predictions.

    Displays the predictions previously made by the logged-in user.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the list of predictions.
    """
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'predictions_list.html', {'predictions': predictions})

# ADMIN

@staff_member_required
def monitoring(request):
    """
    View for the admin monitoring page.

    Displays the monitoring page for administrators.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the monitoring page.
    """
    return render(request, 'admin/monitoring.html')

@staff_member_required
def admin_home(request):
    """
    View for the admin home page.

    Displays the home page for administrators.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the admin home page.
    """
    return render(request, 'admin/admin_home.html')

@staff_member_required
def train_model(request):
    """
    View to train the model.

    Sends a request to the model API to start training.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the training result.
    """
    headers = {'X-API-Key': API_MODEL_KEY}
    response = requests.post(f'http://api_model:12500/train/', headers=headers)
    
    if response.status_code == 200:
        message = response.json().get('message', 'Entraînement du modèle terminé avec succès.')
    else:
        message = response.json().get('message', 'Erreur lors de l\'entraînement du modèle.')
    
    return render(request, 'admin/train_model.html', {'message': message})

@staff_member_required
def list_models(request):
    """
    View to list available models.

    Retrieves the list of models from the model API.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response with the list of models.
    """
    headers = {'X-API-Key': API_MODEL_KEY}
    response = requests.get(f'http://api_model:12500/models/', headers=headers)
    models = response.json().get('available_models', [])
    return render(request, 'admin/list_models.html', {'models': models})

@staff_member_required
def inspect_model(request, version):
    """
    View to inspect a specific model.

    Retrieves and displays the metrics of a specific model from the model API.

    Args:
        request (HttpRequest): The HTTP request object.
        version (str): The version of the model to inspect.

    Returns:
        HttpResponse: The HTTP response with the model metrics or an error.
    """
    headers = {'X-API-Key': API_MODEL_KEY}
    error_message = None
    
    try:
        response = requests.get(f'http://api_model:12500/model-metrics/{version}', headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Erreur {response.status_code} lors de la récupération des métriques : {response.text}")

        metrics = response.json().get('metrics', {})

        if not metrics:
            raise Exception('Aucune métrique disponible pour ce modèle.')

        formatted_metrics = {
            'accuracy': metrics['metrics']['accuracy'],  
            'classes': {},
            'macro_avg': {k.replace('-', '_'): v for k, v in metrics['metrics']['macro avg'].items()}, 
            'weighted_avg': {k.replace('-', '_'): v for k, v in metrics['metrics']['weighted avg'].items()} 
        }

        
        for key, value in metrics['metrics'].items():
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                formatted_metrics['classes'][key] = {k.replace('-', '_'): v for k, v in value.items()}

    except Exception as e:
        error_message = str(e)

    context = {
        'version': version,
        'metrics': formatted_metrics if not error_message else None,
        'error_message': error_message
    }

    return render(request, 'admin/inspect_model.html', context)


# def inspect_model(request, version):
#     """
#     View to inspect a specific model.

#     Retrieves and displays the metrics of a specific model from the model API.

#     Args:
#         request (HttpRequest): The HTTP request object.
#         version (str): The version of the model to inspect.

#     Returns:
#         HttpResponse: The HTTP response with the model metrics or an error.
#     """
#     headers = {'X-API-Key': API_MODEL_KEY}
#     response = requests.get(f'http://api_model:12500/model-metrics/{version}', headers=headers)
    
#     if response.status_code != 200:
#         return render(request, 'admin/error.html', {'error': 'Impossible de récupérer les métriques du modèle'})

#     metrics = response.json().get('metrics', {})

#     if not metrics:
#         return render(request, 'admin/error.html', {'error': 'Aucune métrique disponible pour ce modèle'})

#     if metrics:
#         formatted_metrics = {
#             'accuracy': metrics['metrics']['accuracy'],
#             'classes': {},
#             'macro_avg': {k.replace('-', '_'): v for k, v in metrics['metrics']['macro avg'].items()},
#             'weighted_avg': {k.replace('-', '_'): v for k, v in metrics['metrics']['weighted avg'].items()}
#         }
        
#         for key, value in metrics['metrics'].items():
#             if key not in ['accuracy', 'macro avg', 'weighted avg']:
#                 formatted_metrics['classes'][key] = {k.replace('-', '_'): v for k, v in value.items()}

#     context = {
#         'version': version,
#         'metrics': formatted_metrics,
#     }

#     return render(request, 'admin/inspect_model.html', context)

@staff_member_required
def delete_model(request, version):
    """
    View to delete a specific model.

    Sends a request to the model API to delete a specific model.

    Args:
        request (HttpRequest): The HTTP request object.
        version (str): The version of the model to delete.

    Returns:
        HttpResponse: The HTTP response with the deletion result.
    """
    headers = {'X-API-Key': API_MODEL_KEY}
    response = requests.delete(f'http://api_model:12500/delete-model/{version}', headers=headers)
    message = response.json().get('message', 'Erreur lors de la suppression du modèle.')
    return render(request, 'admin/delete_model.html', {'message': message})

@staff_member_required
def load_model(request, version):
    """
    View to load a specific model.

    Sends a request to the model API to load a specific model.

    Args:
        request (HttpRequest): The HTTP request object.
        version (str): The version of the model to load.

    Returns:
        HttpResponse: The HTTP response with the loading result.
    """
    headers = {'X-API-Key': API_MODEL_KEY}
    response = requests.get(f'http://api_model:12500/load-model/', params={'version': version}, headers=headers)
    message = response.json().get('message', 'Erreur lors du chargement du modèle.')
    return render(request, 'admin/load_model.html', {'message': message})