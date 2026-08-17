"""
Full ML pipeline: load -> preprocess -> feature engineering -> train -> evaluate.

Run this to retrain the model from scratch:
    python pipeline.py
"""

from pathlib import Path

import pandas as pd

from src.offline.feature_pipeline.load import load_data
from src.offline.feature_pipeline.preprocessing import run_preprocess
from src.offline.feature_pipeline.feature_engineering import run_feature_engineering
from src.offline.training_pipeline.train import train
from src.offline.training_pipeline.evaluate import evaluate

PREPROCESSED_PARQUET = Path("data/processed/preprocessed_reduced.parquet")


def run_pipeline():
    print("=== Step 1: Load ===")
    df = load_data()  # loads data/raw/product_launch.parquet directly (SQLite DB is empty)

    print("\n=== Step 2: Preprocess ===")
    if PREPROCESSED_PARQUET.exists():
        print("Preprocessed parquet found — skipping preprocessing")
    else:
        run_preprocess(df)
    df = pd.read_parquet(PREPROCESSED_PARQUET)

    print("\n=== Step 3: Feature Engineering ===")
    X_train, y_train, X_test, y_test = run_feature_engineering(df)

    print("\n=== Step 4: Train ===")
    model = train(X_train, y_train)

    print("\n=== Step 5: Evaluate ===")
    metrics = evaluate(X_test, y_test, model=model)

    print("\n=== Pipeline complete ===")
    return metrics


if __name__ == "__main__":
    run_pipeline()
