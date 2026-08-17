import pandas as pd
import numpy as np
import pytest

from src.offline.feature_pipeline.feature_engineering import (
    merge_scraper_data,
    filter_and_label,
    split,
    run_feature_engineering,
)


@pytest.fixture
def kitchen_df():
    """Minimal Home & Kitchen DataFrame already filtered to that category."""
    return pd.DataFrame({
        "asin":   ["B001", "B002", "B003", "B004"],
        "cat":    ["Home & Kitchen"] * 4,
        "month":  pd.to_datetime(["2024-03-01", "2024-06-01", "2025-07-01", "2025-11-01"]),
        "most_recent_review_time": [200, 50, 100, 200],
        "most_recent_review":      [20,   5,   9,   0],
        "title":  ["pot", "pan", "spatula", "ladle"],
        "price":  [20.0, 30.0, 25.0, 15.0],
        "seller": ["S1", "S2", "S3", "S4"],
    })


@pytest.fixture
def scraper_csv(tmp_path):
    """Small scraper CSV with one confirmed loser (review_count == 0)."""
    path = tmp_path / "scraper.csv"
    pd.DataFrame({
        "asin": ["B002", "B003"],
        "review_count": [0.0, 5.0],  # B002 is a confirmed loser, B003 is not
    }).to_csv(path, index=False)
    return path


# ---------- merge_scraper_data ----------

def test_merge_scraper_updates_loser(kitchen_df, scraper_csv):
    result = merge_scraper_data(kitchen_df, scraper_csv)
    loser = result[result["asin"] == "B002"].iloc[0]
    assert loser["most_recent_review"] == 0
    # most_recent_review_time should be days since launch (positive)
    assert loser["most_recent_review_time"] > 0


def test_merge_scraper_does_not_touch_non_loser(kitchen_df, scraper_csv):
    original = kitchen_df[kitchen_df["asin"] == "B003"].iloc[0]
    result = merge_scraper_data(kitchen_df, scraper_csv)
    updated = result[result["asin"] == "B003"].iloc[0]
    assert updated["most_recent_review"] == original["most_recent_review"]
    assert updated["most_recent_review_time"] == original["most_recent_review_time"]


# ---------- filter_and_label ----------

def test_filter_and_label_drops_under_90_days(kitchen_df):
    result = filter_and_label(kitchen_df)
    assert (result["most_recent_review_time"] >= 90).all()


def test_filter_and_label_computes_velocity(kitchen_df):
    result = filter_and_label(kitchen_df)
    expected = result["most_recent_review"] / result["most_recent_review_time"]
    pd.testing.assert_series_equal(result["review_velocity"], expected, check_names=False)


def test_filter_and_label_creates_binary_label(kitchen_df):
    result = filter_and_label(kitchen_df)
    assert set(result["label"].unique()).issubset({0, 1})
    # B001: velocity = 20/200 = 0.1 >= 0.056 → label 1
    assert result[result["asin"] == "B001"]["label"].iloc[0] == 1


# ---------- split ----------

def test_split_separates_by_date(kitchen_df):
    df = filter_and_label(kitchen_df)
    X_train, y_train, X_test, y_test = split(df)
    assert len(X_train) + len(X_test) == len(df)
    assert "label" not in X_train.columns
    assert "title" in X_train.columns


# ---------- run_feature_engineering ----------

def test_run_feature_engineering_saves_parquets(kitchen_df, scraper_csv, tmp_path):
    run_feature_engineering(kitchen_df, output_dir=tmp_path, scraper_path=scraper_csv)
    assert (tmp_path / "features_train.parquet").exists()
    assert (tmp_path / "features_test.parquet").exists()
    assert (tmp_path / "y_train.parquet").exists()
    assert (tmp_path / "y_test.parquet").exists()
