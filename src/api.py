"""FastAPI prediction service with health, schema, and monitoring endpoints."""
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import MODEL_PATH, REQUEST_LOG, SAMPLE_PATH
from src.monitoring import build_summary

app = FastAPI(title="EduPredict AI", version="1.0.0",
              description="Student academic outcome screening API. Not for sole high-stakes decisions.")

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

class PredictionRequest(BaseModel):
    student: dict

@app.get("/")
def root():
    return {"message": "EduPredict AI is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/schema")
def schema():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run python -m src.train")
    features = list(model.feature_names_in_)
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) if SAMPLE_PATH.exists() else {}
    return {"features": features, "sample_input": sample}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run python -m src.train")
    features = list(model.feature_names_in_)
    missing = [f for f in features if f not in request.student]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    row = {f: request.student[f] for f in features}
    frame = pd.DataFrame([row], columns=features)
    try:
        prediction = str(model.predict(frame)[0])
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probabilities = {str(label): round(float(prob), 4)
                             for label, prob in zip(model.classes_, model.predict_proba(frame)[0])}
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": row,
            "prediction": prediction,
            "probabilities": probabilities
        }
        with REQUEST_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return {"prediction": prediction, "probabilities": probabilities,
                "disclaimer": "Screening support only; review with qualified staff."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

@app.get("/monitoring/summary")
def monitoring_summary():
    return build_summary()
