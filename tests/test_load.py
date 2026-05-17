import sqlite3
import pandas as pd
import pytest
from src.feature_pipeline.load import load_data


def _make_db(path, rows):
    """Create a minimal SQLite products table with given rows."""
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE products (
                asin TEXT, month TEXT, title TEXT, cat TEXT,
                raw_data TEXT, buybox TEXT, rating TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?)", rows
        )


def test_load_data_returns_dataframe(tmp_path):
    _make_db(tmp_path / "product_launch.db", [
        ("B001", "2024-01-01", "Product A", "Home & Kitchen", "{}", None, None),
    ])
    df = load_data(raw_path=tmp_path / "product_launch.db", output_dir=tmp_path)
    assert not df.empty
    assert "asin" in df.columns


def test_load_data_saves_parquet(tmp_path):
    _make_db(tmp_path / "product_launch.db", [
        ("B001", "2024-01-01", "Product A", "Home & Kitchen", "{}", None, None),
    ])
    load_data(raw_path=tmp_path / "product_launch.db", output_dir=tmp_path)
    assert (tmp_path / "product_launch.parquet").exists()


def test_load_data_skips_sqlite_on_rerun(tmp_path):
    # Write a parquet directly — no SQLite needed
    pd.DataFrame({"asin": ["B001"], "month": ["2024-01-01"]}).to_parquet(
        tmp_path / "product_launch.parquet", index=False
    )
    df = load_data(raw_path=tmp_path / "nonexistent.db", output_dir=tmp_path)
    assert len(df) == 1


def test_load_data_sorts_by_month(tmp_path):
    _make_db(tmp_path / "product_launch.db", [
        ("B002", "2024-03-01", "Product C", "Kitchen", "{}", None, None),
        ("B001", "2024-01-01", "Product A", "Kitchen", "{}", None, None),
    ])
    df = load_data(raw_path=tmp_path / "product_launch.db", output_dir=tmp_path)
    months = df["month"].tolist()
    assert months == sorted(months)
