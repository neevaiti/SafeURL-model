# SafeURL Model

This repository contains a solution for detecting phishing URLs using machine learning models, specifically a **RandomForestClassifier**. The project is structured around three main FastAPI-based APIs, a **Django** web application, and a **CI/CD** pipeline for continuous integration and deployment.

## Project Overview

### APIs
1. **Database Interaction API**:
   - This API interacts with a **PostgreSQL** database hosted on an Azure server.
   - It handles storing URL predictions, model performance metrics, and logs of model activities.

2. **URL Cleaning API**:
   - This API processes and cleans URLs to ensure they are in the correct format before being passed to the machine learning model.
   - It prepares URLs for prediction by sanitizing and transforming them into a structure that the model can easily interpret.

3. **Machine Learning API**:
   - This API contains the machine learning model, a **RandomForestClassifier**, which predicts whether a URL is a phishing link or safe.
   - The model is trained on historical URL data and exposed as an endpoint for real-time predictions.

### Django Application
- The **Django** application serves as an interface to manage the system, providing access to APIs and offering a user-friendly administration panel.
- It integrates with the machine learning API to make predictions and provides results to end-users.
  
### CI/CD Pipelines
- The project utilizes two pipelines for Continuous Integration (CI) and Continuous Deployment (CD), ensuring the smooth delivery and deployment of updates.
- **CI Pipeline**: Runs unit tests and validates the integrity of the codebase before merging.
- **CD Pipeline**: Deploys the services, including APIs and the Django app, to the server using Docker and **Uvicorn**.

### Docker
- All services are containerized using **Docker**, ensuring consistency across different environments and simplifying the deployment process.

### Unit Tests
- **Unit tests** are in place to verify the functionality of individual components, ensuring the reliability of APIs and the model.

### Server Setup
- The server is powered by **Uvicorn**, which handles the launch and management of the various services (APIs, Django app, etc.).

## Technologies Used
- **FastAPI**: For building high-performance APIs.
- **Django**: As the web framework for managing and serving the system.
- **PostgreSQL**: For database storage, hosted on Azure.
- **RandomForestClassifier**: As the machine learning model to detect phishing URLs.
- **Docker**: For containerizing services.
- **CI/CD Pipelines**: For automating testing, integration, and deployment.
- **Uvicorn**: As the ASGI server to run the FastAPI services.

## Getting Started

### Set Up Docker
Make sure you have Docker installed and set up the environment.

### Run the Services
Use Docker Compose to launch all services:
```bash
docker-compose up --build
```