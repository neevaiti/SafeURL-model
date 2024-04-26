# !/bin/bash


podman pod create --name safeurl-api -p 8050:8050 -p 8060:8060


podman run --pod safeurl-api --name api_interact -d \
           -v "$(pwd)/src/api/interact_model/:/app" \
           -w /app python:3.10-slim \
           /bin/bash -c "pip install --no-cache-dir -r requirements.txt && \
                         uvicorn api:app --host 0.0.0.0 --port 8060 --proxy-headers"

podman run --pod safeurl-api --name api_model -d \
           -v "$(pwd)/src/api/model/:/app" \
           -w /app python:3.10-slim \
           /bin/bash -c "pip install --no-cache-dir -r requirements.txt && \
                         uvicorn api:app --host 0.0.0.0 --port 8050 --proxy-headers"

