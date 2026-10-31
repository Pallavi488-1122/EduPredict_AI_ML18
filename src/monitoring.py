"""Lightweight production monitoring: request volume and numeric PSI drift."""
import json
import numpy as np
import pandas as pd
from src.config import REFERENCE_PATH, REQUEST_LOG, ARTIFACTS

def psi(expected, actual, buckets=10):
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().to_numpy()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy()
    if len(expected) < 20 or len(actual) < 5:
        return None
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def build_summary():
    if not REFERENCE_PATH.exists():
        return {"status": "not_ready", "message": "Train the model first."}
    reference = pd.read_csv(REFERENCE_PATH)
    records = []
    if REQUEST_LOG.exists():
        with REQUEST_LOG.open(encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        return {"status": "waiting_for_predictions", "prediction_count": 0, "drift": {}}
    incoming = pd.DataFrame([r.get("input", {}) for r in records])
    drift = {}
    for col in reference.select_dtypes(include="number").columns:
        if col in incoming.columns:
            score = psi(reference[col], incoming[col])
            if score is not None:
                drift[col] = round(score, 4)
    max_psi = max(drift.values()) if drift else None
    status = "warning" if max_psi is not None and max_psi >= 0.20 else "healthy"
    return {
        "status": status,
        "prediction_count": len(records),
        "drift_threshold_psi": 0.20,
        "max_psi": max_psi,
        "drift": drift,
        "note": "PSI is a screening signal; investigate before taking action."
    }

if __name__ == "__main__":
    print(json.dumps(build_summary(), indent=2))
