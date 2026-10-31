from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
REFERENCE_PATH = ARTIFACTS / "reference_data.csv"
SAMPLE_PATH = ARTIFACTS / "sample_input.json"
REQUEST_LOG = ARTIFACTS / "prediction_log.jsonl"

for folder in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
    folder.mkdir(parents=True, exist_ok=True)

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
RANDOM_STATE = 42
TARGET = "Target"
