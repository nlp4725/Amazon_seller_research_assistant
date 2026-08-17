"""
Evaluate trained model on held-out test data.

- Loads model from models/ if not passed directly
- Computes ROC-AUC, PR-AUC, precision, recall, and F1 at threshold 0.4
- Saves metrics to data/processed/evaluation_results.parquet
- Prints classification report
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
THRESHOLD = 0.4


def evaluate(
    X_test: pd.DataFrame,
    y_test,
    model=None,
    models_dir: Path | str = MODELS_DIR,
    output_dir: Path | str = PROCESSED_DIR,
) -> dict:
    """
    Compute classification metrics on test data.

    In: X_test (DataFrame with title column), y_test (array-like), model (optional — loaded from models_dir if None)
    Out: dict with roc_auc, pr_auc, precision, recall, f1
    """
    if model is None:
        model = joblib.load(Path(models_dir) / "model.joblib")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "pr_auc":    average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    print(classification_report(y_test, y_pred))
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_parquet(outdir / "evaluation_results.parquet", index=False)

    return metrics


if __name__ == "__main__":
    X_test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    evaluate(X_test, y_test)
