import json
import pandas as pd
import pytest
from src.offline.preprocessing import (
    convert_history_to_dict,
    filter_price,
    add_review_velocity,
    drop_duplicates,
    add_price_seller_title,
    add_review_n_monthly_sold,
    run_preprocess,
)

KEEPA_EPOCH_MINS = 0  # list_time = 0 for simplicity in tests

RAW_DATA = json.dumps({
    'listedSince': 0,
    'title': 'Test Kitchen Product',
    'csv': [None, None, None, None, [None, 3000]],  # price = 3000 / 100 = 30.0
    'reviews': {'reviewCount': [0, 5, 60 * 24 * 100, 10]},  # day 100: 10 reviews
    'monthlySoldHistory': None,
    'rootCategory': 12345,
    'salesRanks': {},
})

BUYBOX = json.dumps({'buyBoxSellerIdHistory': ['SELLER123']})


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {'asin': 'B001', 'month': '2024-01-01', 'cat': 'Home & Kitchen', 'raw_data': RAW_DATA, 'buybox': BUYBOX},
        {'asin': 'B002', 'month': '2024-02-01', 'cat': 'Home & Kitchen', 'raw_data': RAW_DATA, 'buybox': BUYBOX},
    ])


# ---------- helpers ----------

def test_convert_history_to_dict_returns_none_for_none():
    assert convert_history_to_dict(None, 0) is None


def test_convert_history_to_dict_maps_days_correctly():
    # list_time=0, keepa timestamps in minutes
    result = convert_history_to_dict([0, 5, 60 * 24, 10], 0)
    assert result == {0: 5, 1: 10}  # day 0 and day 1


# ---------- main functions ----------

def test_filter_price_keeps_in_range():
    df = pd.DataFrame({'price': [10.0, 25.0, 50.0, 110.0]})
    result = filter_price(df)
    assert list(result['price']) == [25.0, 50.0]


def test_filter_price_excludes_boundaries():
    df = pd.DataFrame({'price': [14.99, 15.0, 100.0, 100.01]})
    result = filter_price(df)
    assert list(result['price']) == [15.0, 100.0]


def test_add_review_velocity_computed_for_90_plus_days():
    df = pd.DataFrame({
        'most_recent_review_time': [100, 50],
        'most_recent_review': [10, 5],
    })
    result = add_review_velocity(df)
    assert result.loc[0, 'review_velocity'] == pytest.approx(0.1)
    assert pd.isna(result.loc[1, 'review_velocity'])


def test_drop_duplicates_keeps_last():
    df = pd.DataFrame({'asin': ['B001', 'B001', 'B002'], 'val': [1, 2, 3]})
    result = drop_duplicates(df)
    assert len(result) == 2
    assert result[result['asin'] == 'B001']['val'].iloc[0] == 2


def test_add_price_seller_title_extracts_correctly(sample_df):
    result = add_price_seller_title(sample_df)
    assert result.loc[0, 'price'] == pytest.approx(30.0)
    assert result.loc[0, 'title'] == 'Test Kitchen Product'
    assert result.loc[0, 'seller'] == 'SELLER123'


def test_add_review_n_monthly_sold_adds_columns(sample_df):
    df = add_price_seller_title(sample_df)
    result = add_review_n_monthly_sold(df)
    assert 'most_recent_review' in result.columns
    assert 'most_recent_review_time' in result.columns


def test_run_preprocess_saves_parquet(sample_df, tmp_path):
    run_preprocess(sample_df, output_dir=tmp_path)
    assert (tmp_path / 'preprocessed.parquet').exists()
    assert (tmp_path / 'preprocessed_reduced.parquet').exists()


# ---------- data sanity check ----------

def test_cleaned_data_has_61635_rows():
    path = 'data/raw/product_launch_cleaned_61635.parquet'
    df = pd.read_parquet(path)
    assert len(df) == 61635
