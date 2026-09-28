FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_app.py .
COPY voice_pd_model_replicated_acoustic.joblib .
COPY feature_schema.json .

# Cloud Run injects $PORT; default to 8080 for local docker run
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn api_app:app --host 0.0.0.0 --port ${PORT}"]
