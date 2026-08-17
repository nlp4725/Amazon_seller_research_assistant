"""
Feature engineering: filter, label, and split products for model training.

- Filters to Home & Kitchen with >= 90 days of review data
- Merges scraper data to recover confirmed losers (review_count == 0 at 2026-05-12)
- Labels each product: review_velocity >= 0.056 -> 1 (success), else 0 (failure)
- Splits by launch date: train <= 2025-09-01, test > 2025-09-01
- Saves features_train.parquet, features_test.parquet, y_train.parquet, y_test.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
SCRAPER_PATH = Path("data/raw/scripted_review_counts_kitchen_only.csv")
SCRAPER_DATE = pd.Timestamp("2026-05-12")
VELOCITY_THRESHOLD = 0.056
MIN_REVIEW_DAYS = 90


def merge_scraper_data(df: pd.DataFrame, scraper_path: Path | str = SCRAPER_PATH) -> pd.DataFrame:
    """
    Add confirmed losers from scraper data.
    Products with review_count == 0 at SCRAPER_DATE are confirmed losers:
    their most_recent_review_time is set to days since launch and most_recent_review to 0.

    In: df with asin, month, most_recent_review, most_recent_review_time; path to scripted_review_counts.csv
    Out: df with updated most_recent_review and most_recent_review_time for confirmed losers
    """
    scraper = pd.read_csv(scraper_path)
    confirmed_losers = scraper[scraper["review_count"] == 0]["asin"].tolist()

    df = df.copy()
    mask = df["asin"].isin(confirmed_losers)
    df.loc[mask, "most_recent_review_time"] = (SCRAPER_DATE - df.loc[mask, "launch_year_month"]).dt.days
    df.loc[mask, "most_recent_review"] = 0
    return df


def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to products with >= 90 days of review data, compute velocity, and label.

    In: df already filtered to Home & Kitchen, with most_recent_review_time, most_recent_review, launch_year_month columns
    Out: filtered df with review_velocity and label columns
    """
    df = df[df["most_recent_review_time"] >= MIN_REVIEW_DAYS].copy()
    df["review_velocity"] = df["most_recent_review"] / df["most_recent_review_time"]
    df = df[df["launch_year_month"] <= pd.Timestamp("2025-12-01")]
    # drop Jan 2024 — no prior-month competitor baseline available
    df = df[df["launch_year_month"] > pd.Timestamp("2024-01-01")]
    df["label"] = np.where(df["review_velocity"] >= VELOCITY_THRESHOLD, 1, 0)
    return df


def split(df: pd.DataFrame):
    """
    Time-based train/test split. Train <= 2025-09-01, test > 2025-09-01.

    In: df with launch_year_month, title, price, launch_year, launch_month, and label columns
    Out: X_train, y_train, X_test, y_test
    """
    train = df[df["launch_year_month"] <= pd.Timestamp("2025-09-01")]
    test = df[df["launch_year_month"] > pd.Timestamp("2025-09-01")]

    X_train = train[["title"]]
    y_train = train["label"]
    X_test = test[["title"]]
    y_test = test["label"]

    return X_train, y_train, X_test, y_test


def run_feature_engineering(
    df: pd.DataFrame,
    output_dir: Path | str = PROCESSED_DIR,
    scraper_path: Path | str = SCRAPER_PATH,
) -> tuple:
    """
    Run full feature engineering pipeline and save outputs to disk.

    In: clean df from run_preprocess(), output_dir for saving parquets, scraper_path for review data
    Out: X_train, y_train, X_test, y_test
    """
    df = df[df["cat"] == "Home & Kitchen"].copy()
    df = merge_scraper_data(df, scraper_path)
    df = filter_and_label(df)
    X_train, y_train, X_test, y_test = split(df)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(outdir / "features_train.parquet", index=False)
    X_test.to_parquet(outdir / "features_test.parquet", index=False)
    y_train.to_frame().to_parquet(outdir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(outdir / "y_test.parquet", index=False)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Label distribution — train: {y_train.value_counts().to_dict()}, test: {y_test.value_counts().to_dict()}")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/preprocessed_reduced.parquet")
    run_feature_engineering(df)
