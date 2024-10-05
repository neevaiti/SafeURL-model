# API for model
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 12400

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "12400", "--proxy-headers"]
