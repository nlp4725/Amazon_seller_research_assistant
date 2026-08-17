import pandas as pd
import pytest

from src.offline.training_pipeline.train import train
from src.offline.training_pipeline.evaluate import evaluate

EXPECTED_KEYS = {"roc_auc", "pr_auc", "precision", "recall", "f1"}


@pytest.fixture
def trained_model(tmp_path):
    """Fit a small Pipeline on title strings — same input shape as real inference."""
    X_train = pd.DataFrame({"title": [
        "non-stick silicone spatula", "stainless steel mixing bowl",
        "bamboo cutting board", "cast iron skillet", "silicone baking mat",
        "ceramic knife set", "glass food containers", "copper measuring cups",
    ]})
    y_train = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    return train(X_train, y_train, models_dir=tmp_path)


@pytest.fixture
def test_data():
    X_test = pd.DataFrame({"title": [
        "non-stick pan set", "wooden spoon set", "silicone oven mitt", "steel colander",
    ]})
    y_test = pd.Series([1, 0, 1, 0])
    return X_test, y_test


def test_evaluate_returns_correct_keys(trained_model, test_data, tmp_path):
    X_test, y_test = test_data
    metrics = evaluate(X_test, y_test, model=trained_model, output_dir=tmp_path)
    assert set(metrics.keys()) == EXPECTED_KEYS


def test_evaluate_metrics_in_range(trained_model, test_data, tmp_path):
    X_test, y_test = test_data
    metrics = evaluate(X_test, y_test, model=trained_model, output_dir=tmp_path)
    for name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"


def test_evaluate_saves_parquet(trained_model, test_data, tmp_path):
    X_test, y_test = test_data
    evaluate(X_test, y_test, model=trained_model, output_dir=tmp_path)
    assert (tmp_path / "evaluation_results.parquet").exists()
