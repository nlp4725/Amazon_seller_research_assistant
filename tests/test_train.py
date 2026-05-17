import pandas as pd
import pytest

from src.training_pipeline.train import build_pipeline, train


@pytest.fixture
def training_data():
    """Minimal title DataFrame and binary labels."""
    X_train = pd.DataFrame({"title": [
        "non-stick silicone spatula set",
        "stainless steel mixing bowls",
        "bamboo cutting board large",
        "cast iron skillet pre-seasoned",
        "silicone baking mat reusable",
    ]})
    y_train = pd.Series([1, 0, 1, 0, 1])
    return X_train, y_train


def test_build_pipeline_has_tfidf_and_xgb():
    model = build_pipeline()
    assert "preprocessing" in model.named_steps
    assert "xgb" in model.named_steps


def test_train_saves_model(tmp_path, training_data):
    X_train, y_train = training_data
    model = train(X_train, y_train, models_dir=tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert model is not None


def test_train_model_can_predict(tmp_path, training_data):
    X_train, y_train = training_data
    model = train(X_train, y_train, models_dir=tmp_path)
    preds = model.predict(X_train)
    assert len(preds) == len(X_train)
    assert set(preds).issubset({0, 1})
