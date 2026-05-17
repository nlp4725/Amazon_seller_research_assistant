import pandas as pd
import pytest

from src.training_pipeline.train import train
from src.inference_pipeline.inference import predict, titles_from_csv


@pytest.fixture
def trained_model_dir(tmp_path):
    """Fit a minimal Pipeline and save model.joblib to tmp_path."""
    X_train = pd.DataFrame({"title": [
        "non-stick silicone spatula", "stainless steel mixing bowl",
        "bamboo cutting board", "cast iron skillet", "silicone baking mat",
        "ceramic knife set", "glass food containers", "copper measuring cups",
    ]})
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    train(X_train, y_train, models_dir=tmp_path)
    return tmp_path


# ---------- predict() ----------

def test_predict_returns_dataframe(trained_model_dir):
    result = predict(["non-stick silicone pan", "bamboo spoon"], models_dir=trained_model_dir)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_predict_has_expected_columns(trained_model_dir):
    result = predict(["silicone spatula"], models_dir=trained_model_dir)
    assert set(result.columns) == {"title", "y_pred_prob", "predicted_label"}


def test_predict_probability_in_range(trained_model_dir):
    result = predict(["cast iron skillet pre-seasoned"], models_dir=trained_model_dir)
    assert 0.0 <= result["y_pred_prob"].iloc[0] <= 1.0


def test_predict_label_is_binary(trained_model_dir):
    result = predict(["stainless steel mixing bowl large"], models_dir=trained_model_dir)
    assert result["predicted_label"].iloc[0] in {0, 1}


# ---------- titles_from_csv() ----------

def test_titles_from_csv_auto_detects_column(tmp_path):
    csv_path = tmp_path / "products.csv"
    pd.DataFrame({"title": ["spatula", "pan"], "price": [20, 30]}).to_csv(csv_path, index=False)
    titles = titles_from_csv(str(csv_path))
    assert titles == ["spatula", "pan"]


def test_titles_from_csv_uses_specified_column(tmp_path):
    csv_path = tmp_path / "products.csv"
    pd.DataFrame({"name": ["spatula", "pan"]}).to_csv(csv_path, index=False)
    titles = titles_from_csv(str(csv_path), title_col="name")
    assert titles == ["spatula", "pan"]
