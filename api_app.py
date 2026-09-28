"""
NeuroInsight-PD backend API
Serves predictions from the trained model using pre-computed acoustic
feature values (pulled from the dataset).

Files required alongside this script:
  - voice_pd_model_replicated_acoustic.joblib  (trained sklearn pipeline)
  - feature_schema.json                        (exact feature names/order)

Run locally with:  uvicorn api_app:app --reload --port 8000
"""

import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="NeuroInsight-PD Prediction API", version="1.0")

# Allow the Flutter app / website to call this API from any origin.
# Tighten `allow_origins` to your actual domains before production launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "voice_pd_model_replicated_acoustic.joblib"
SCHEMA_PATH = "feature_schema.json"

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f:
    schema = json.load(f)
FEATURE_NAMES = schema["feature_names"]


class VoiceFeatures(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Map of feature name -> value, e.g. {'Jitter_rel': 0.003, ...}"
    )


class PredictionResponse(BaseModel):
    prediction: str          # "Healthy" or "PD"
    prediction_code: int     # 0 = Healthy, 1 = PD
    probability_pd: float    # model's confidence that this person has PD


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": schema.get("model_name"),
        "expected_features": len(FEATURE_NAMES),
    }


@app.get("/schema")
def get_schema():
    """Lets the frontend team see exactly which feature names/order are required."""
    return schema


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: VoiceFeatures):
    missing = [f for f in FEATURE_NAMES if f not in payload.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}"
        )

    x = np.array([[payload.features[f] for f in FEATURE_NAMES]])

    proba_pd = model.predict_proba(x)[0, 1]
    pred_code = int(proba_pd >= 0.5)
    pred_label = "PD" if pred_code == 1 else "Healthy"

    return PredictionResponse(
        prediction=pred_label,
        prediction_code=pred_code,
        probability_pd=round(float(proba_pd), 4),
    )
