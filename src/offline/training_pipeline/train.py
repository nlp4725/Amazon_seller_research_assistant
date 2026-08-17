"""
Train XGBClassifier on product titles and save the model to disk.

- Reads features_train.parquet and y_train.parquet from data/processed/
- Fits a Pipeline: TF-IDF (max_features=500, bigrams) + XGBClassifier
- Saves fitted Pipeline to models/model.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def build_pipeline() -> Pipeline:
    """
    Build the unfitted sklearn Pipeline: TF-IDF on title column + XGBClassifier.

    In: nothing
    Out: unfitted Pipeline
    """
    preprocessing = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2)), "title"),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("xgb", XGBClassifier(random_state=42, eval_metric="logloss")),
    ])


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: Path | str = MODELS_DIR,
) -> Pipeline:
    """
    Fit Pipeline and save to disk.

    In: X_train (DataFrame with title column), y_train (binary labels), models_dir for saving
    Out: fitted Pipeline saved to models/model.joblib
    """
    model = build_pipeline()
    model.fit(X_train, y_train)

    outdir = Path(models_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outdir / "model.joblib")
    print(f"Model saved to {outdir / 'model.joblib'}")

    return model


if __name__ == "__main__":
    X_train = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").squeeze()
    train(X_train, y_train)
