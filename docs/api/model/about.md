# Documentation de l'API

## Introduction
Ce document décrit le fonctionnement de l'API développée avec FastAPI pour gérer les modèles de machine learning. L'API permet de prédire, entraîner, charger, supprimer et récupérer les métriques des modèles.

## Imports Principaux
Le fichier api.py commence par importer plusieurs bibliothèques essentielles :
os, asyncpg, sqlite3, json, pickle, logging : Utilisés pour la gestion des fichiers, la connexion à la base de données, la sérialisation des modèles, et la gestion des logs.
pandas, numpy : Utilisés pour la manipulation des données.
fastapi, HTTPException, Request, Depends, Security : Utilisés pour créer l'API et gérer les exceptions.
APIKeyHeader, get_openapi : Utilisés pour la gestion de la sécurité de l'API et la personnalisation du schéma OpenAPI.
dotenv : Pour charger les variables d'environnement.
sklearn : Pour les modèles de machine learning.

## Variables d'environnement
IS_TEST : Indique si l'API est en mode test.
API_MODEL_KEY : Clé API pour accéder aux fonctionnalités de l'API.
MODEL_DIR : Répertoire où sont stockés les modèles.

## Variables d'environnement
L'API utilise plusieurs variables d'environnement importantes :

- `IS_TEST` : Indique si l'API est en mode test.
- `API_MODEL_KEY` : Clé API pour sécuriser l'accès aux fonctionnalités de l'API.
- `MODEL_DIR` : Répertoire où sont stockés les modèles.
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT` : Informations de connexion à la base de données PostgreSQL.

## Configuration de la base de données
L'API peut être configurée pour utiliser :
- PostgreSQL en production
- SQLite pour les tests

La connexion à la base de données est gérée de manière asynchrone pour de meilleures performances.

## Endpoints principaux

### POST /predict
Effectue une prédiction à partir d'un modèle spécifié.

Paramètres :
- `url` : L'URL à analyser
- `model_version` (optionnel) : La version du modèle à utiliser

Réponse :
- `prediction` : Le résultat de la prédiction (malveillant ou non)
- `confidence` : Le niveau de confiance de la prédiction

### POST /train
Entraîne un nouveau modèle ou met à jour un modèle existant.

Paramètres :
- Données d'entraînement (format à spécifier)

Réponse :
- Informations sur le modèle entraîné (version, métriques, etc.)

### GET /load_model/{version}
Charge un modèle spécifique depuis le stockage.

Paramètres :
- `version` : La version du modèle à charger

Réponse :
- Confirmation du chargement du modèle

### DELETE /delete_model/{version}
Supprime un modèle spécifique du système.

Paramètres :
- `version` : La version du modèle à supprimer

Réponse :
- Confirmation de la suppression du modèle

### GET /model-metrics/{version}
Fournit des métriques sur un modèle spécifique.

Paramètres :
- `version` : La version du modèle

Réponse :
- Métriques détaillées du modèle (précision, rappel, etc.)

## Sécurité
L'API utilise une authentification par clé API. Chaque requête doit inclure une clé API valide dans l'en-tête `X-API-Key`.

## Gestion des erreurs
L'API utilise des exceptions personnalisées et des codes d'état HTTP appropriés pour communiquer clairement les erreurs.

## Logging
Un système de logging est implémenté pour enregistrer les activités importantes et les erreurs, facilitant le débogage et la maintenance.

## Documentation OpenAPI
L'API génère automatiquement une documentation OpenAPI, accessible via l'interface Swagger UI, détaillant tous les endpoints, leurs paramètres, et les réponses attendues.