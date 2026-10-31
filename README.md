# EduPredict AI — ML-18 Capstone

An end-to-end machine-learning system that predicts **Dropout**, **Enrolled**, or **Graduate** outcomes using the UCI Predict Students' Dropout and Academic Success dataset.

> This is an educational risk-screening aid, not a tool for making high-stakes decisions about students. Predictions should be reviewed by qualified staff and must not be used as the sole basis for denying support or opportunities.

## Architecture

UCI dataset → ingestion & preprocessing → model comparison / validation → MLflow tracking → FastAPI + Docker → prediction logging & drift monitoring → Streamlit dashboard.

## Quick start (Windows / PyCharm)

1. Open this folder in PyCharm.
2. Create a Python 3.11 or 3.12 virtual environment.
3. In the PyCharm terminal:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_pipeline
python -m src.train
```

4. Start the API:

```bash
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

5. Start the dashboard in another terminal:

```bash
streamlit run dashboard/app.py
```

Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## Dataset

The pipeline downloads the public UCI dataset automatically on first run. The target column is `Target` and the three classes are Dropout, Enrolled, and Graduate.

## Main endpoints

- `GET /` — service information
- `GET /health` — readiness and model status
- `GET /schema` — feature names and sample input
- `POST /predict` — single-student prediction
- `GET /monitoring/summary` — basic request and drift summary

Use `/docs` to try the API. Copy the sample JSON from `artifacts/sample_input.json`.

## Retraining

Run `python -m src.train` to refresh the trained model. `python -m src.scheduler` starts a simple scheduled retraining process (daily by default; configure `RETRAIN_HOUR` and `RETRAIN_MINUTE` environment variables).

## Docker

```bash
docker build -t edupredict-ai .
docker run -p 8000:8000 edupredict-ai
```

## Project structure

```text
EduPredict_AI_ML18/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
├── dashboard/
│   └── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── train.py
│   ├── api.py
│   ├── monitoring.py
│   └── scheduler.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Evaluation checklist

- [ ] Dataset downloaded and validated
- [ ] Data pipeline runs end-to-end
- [ ] Models compared with stratified cross-validation
- [ ] Final model and metrics saved
- [ ] MLflow run recorded (when MLflow is available)
- [ ] API health and prediction endpoints work
- [ ] Docker image builds and runs
- [ ] Monitoring summary updates after predictions
- [ ] Dashboard displays live API and monitoring data
- [ ] README, architecture diagram, and demo recording completed
