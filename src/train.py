"""Train, compare, validate, and save a production-ready sklearn pipeline."""
import json
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.config import ARTIFACTS, MODEL_PATH, METRICS_PATH, REFERENCE_PATH, SAMPLE_PATH, RANDOM_STATE, TARGET
from src.data_pipeline import load_data

warnings.filterwarnings("ignore")

def build_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2500, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, random_state=RANDOM_STATE)
    }

def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_name, best_score, best_model = None, -1, None

    for name, estimator in build_candidates().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_score = float(scores.mean())
        results[name] = {"cv_f1_macro_mean": mean_score, "cv_f1_macro_std": float(scores.std())}
        print(f"{name}: CV macro-F1={mean_score:.4f} (+/- {scores.std():.4f})")
        if mean_score > best_score:
            best_name, best_score, best_model = name, mean_score, pipe

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, predictions, labels=labels).tolist()

    joblib.dump(best_model, MODEL_PATH)
    X_train.to_csv(REFERENCE_PATH, index=False)
    sample = X.iloc[[0]].to_dict(orient="records")[0]
    # Convert numpy scalars into JSON-safe Python values.
    sample = {k: (v.item() if hasattr(v, "item") else v) for k, v in sample.items()}
    SAMPLE_PATH.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    metrics = {
        "best_model": best_name,
        "best_cv_f1_macro": best_score,
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "class_labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
        "model_comparison": results,
        "rows": int(len(df)),
        "features": int(X.shape[1])
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import mlflow
        mlflow.set_experiment("EduPredict-AI-ML18")
        with mlflow.start_run(run_name=best_name):
            mlflow.log_param("best_model", best_name)
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("features", X.shape[1])
            mlflow.log_metric("cv_f1_macro", best_score)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_f1_macro", macro_f1)
            mlflow.sklearn.log_model(best_model, "model")
        print("MLflow run logged.")
    except Exception as exc:
        print(f"MLflow tracking skipped (model is still saved): {exc}")

    print("\nSelected:", best_name)
    print(f"Test accuracy: {accuracy:.4f} | Test macro-F1: {macro_f1:.4f}")
    print("Artifacts saved in:", ARTIFACTS)

if __name__ == "__main__":
    main()
