"""
Inference: predict launch success for a list of product titles.

- predict(): takes a list of titles, returns predicted probability and label for each
- Loads model.joblib from models/ — a fitted Pipeline (TF-IDF + XGBClassifier)
- titles_from_csv(): helper to extract titles from a CSV file
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")
THRESHOLD = 0.4

_model = None


def _load_model(models_dir: Path | str = MODELS_DIR):
    global _model
    if _model is None:
        _model = joblib.load(Path(models_dir) / "model.joblib")
    return _model


def titles_from_csv(path: str, title_col: str | None = None) -> list[str]:
    """
    Extract titles from a CSV file. Auto-detects the title column if not specified.

    In: path to CSV, optional column name
    Out: list of title strings
    """
    df = pd.read_csv(path)
    if title_col:
        col = title_col
    else:
        col = next((c for c in df.columns if "title" in c.lower()), df.columns[0])
    return df[col].dropna().astype(str).tolist()


def predict(titles: list[str], models_dir: Path | str = MODELS_DIR) -> pd.DataFrame:
    """
    Predict launch success for a list of product titles.

    In: list of title strings
    Out: DataFrame with title, y_pred_prob, predicted_label columns
    """
    model = _load_model(models_dir)
    X = pd.DataFrame({"title": titles})
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred_label = (y_pred_prob >= THRESHOLD).astype(int)

    return pd.DataFrame({
        "title": titles,
        "y_pred_prob": y_pred_prob.round(4),
        "predicted_label": y_pred_label,
    })


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference.py 'Title One, Title Two'")
        print("  python inference.py --csv path/to/file.csv [--col title_column]")
        sys.exit(1)

    if sys.argv[1] == "--csv":
        csv_path = sys.argv[2]
        col = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--col" else None
        titles = titles_from_csv(csv_path, col)
    else:
        titles = [t.strip() for t in sys.argv[1].split(",") if t.strip()]

    results = predict(titles)
    print(results.to_string(index=False))
