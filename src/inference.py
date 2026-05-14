import sys
import joblib
import pandas as pd
from src.feature_engineering import build_features

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load("model/model.joblib")
    return _model


def titles_from_csv(path: str, title_col: str | None = None) -> list[str]:
    """Extract titles from a CSV file. Auto-detects the title column if not specified."""
    df = pd.read_csv(path)
    if title_col:
        col = title_col
    else:
        col = next((c for c in df.columns if "title" in c.lower()), df.columns[0])
    return df[col].dropna().astype(str).tolist()


def predict(titles: list[str]) -> pd.DataFrame:
    """
    Run inference on a list of product titles.
    Returns DataFrame with: title, y_pred_prob, predicted_label.
    """
    model = _load_model()
    X = build_features(titles)
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred_label = (y_pred_prob >= 0.5).astype(int)

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
