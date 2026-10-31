"""Download, validate, and prepare the UCI student success dataset."""
from zipfile import ZipFile
from io import BytesIO
import requests
import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED, UCI_DATA_URL, TARGET

def download_dataset():
    csv_path = DATA_RAW / "data.csv"
    if csv_path.exists():
        return csv_path
    print("Downloading UCI student dataset...")
    response = requests.get(UCI_DATA_URL, timeout=90)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        candidates = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError("No CSV file found in the UCI dataset archive.")
        # The archive contains the main semicolon-separated data.csv.
        preferred = next((n for n in candidates if n.lower().endswith("data.csv")), candidates[0])
        (DATA_RAW / "data.csv").write_bytes(archive.read(preferred))
    return csv_path

def load_data():
    path = download_dataset()
    df = pd.read_csv(path, sep=";")
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found: {list(df.columns)}")
    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    return df

def run_pipeline():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)
    clean = X.copy()
    clean[TARGET] = y
    output = DATA_PROCESSED / "students_clean.csv"
    clean.to_csv(output, index=False)
    print(f"Saved {len(clean):,} rows and {X.shape[1]} features to {output}")
    print("Target distribution:")
    print(y.value_counts().to_string())
    return X, y

if __name__ == "__main__":
    run_pipeline()
