"""
Bonsai Species Classification API.

Serves whichever version of the Classifier currently carries the @champion alias in the
MLflow Model Registry. Promoting a new version in the registry is enough to change what
this API answers — no redeploy, no code change.
"""

import os

import numpy as np
from fastapi import FastAPI, HTTPException, Request

import mlflow
import mlflow.pyfunc

app = FastAPI(title="Bonsai Species Classification API")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Bonsai-Species-Classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEATURE_NAMES = [
    "leaf_length_cm",
    "leaf_width_cm",
    "branch_thickness_mm",
    "height_cm",
]

SPECIES = {
    0: "Juniper",
    1: "Ficus",
    2: "Pine",
    3: "Maple",
}

CARE_RECOMMENDATIONS = {
    0: "Hardy evergreen, needs full sun, minimal watering, wire training in fall",
    1: "Prefers bright indirect light, consistent moisture, frequent pruning required",
    2: "Requires full sun, well-draining soil, candle pinching in spring",
    3: "Needs partial shade, consistent moisture, protection from wind",
}

# Loaded lazily. The API starts before MLflow is ready and long before any model has been
# trained, so loading at import time would permanently leave this at None and every
# prediction would answer 503 until someone restarted the container.
_model = None
_load_error = None


def get_model(force_reload: bool = False):
    """Return the champion model, loading it on first use."""
    global _model, _load_error

    if _model is not None and not force_reload:
        return _model

    try:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
        _load_error = None
    except Exception as exc:
        _model = None
        _load_error = str(exc)

    return _model


@app.get("/")
async def root():
    model = get_model()
    return {
        "message": "Bonsai Species Classification API",
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
        "load_error": _load_error,
        "species": list(SPECIES.values()),
        "features": FEATURE_NAMES,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/health")
async def health():
    """Liveness only. Reports model state without trying to load it."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_uri": MODEL_URI,
    }


@app.post("/reload")
async def reload_model():
    """
    Pick up a newly promoted champion without restarting the container.

    Call this after moving the @champion alias to a different version.
    """
    model = get_model(force_reload=True)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load {MODEL_URI}: {_load_error}",
        )
    return {"reloaded": True, "model_uri": MODEL_URI}


@app.post("/predict")
async def predict(request: Request):
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model at {MODEL_URI}. Train a Classifier in the notebook and give a "
                f"version the '{MODEL_ALIAS}' alias first. Underlying error: {_load_error}"
            ),
        )

    data = await request.json()

    if "features" not in data:
        raise HTTPException(status_code=400, detail="Missing 'features' field")

    features = data["features"]
    if len(features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features: {FEATURE_NAMES}",
        )

    try:
        prediction = model.predict(np.array(features, dtype=float).reshape(1, -1))
        species_id = int(prediction[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return {
        "prediction": species_id,
        "species": SPECIES.get(species_id, "Unknown"),
        "care_recommendations": CARE_RECOMMENDATIONS.get(
            species_id, "No care information available"
        ),
        "input_features": dict(zip(FEATURE_NAMES, features)),
    }


@app.get("/species")
async def get_species():
    """Every species this API can classify, with its care recommendation."""
    return {
        "species_info": {
            name: {"id": species_id, "name": name, "care": CARE_RECOMMENDATIONS[species_id]}
            for species_id, name in SPECIES.items()
        }
    }
