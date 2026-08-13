"""
Preprocess raw product data into a clean, analysis-ready DataFrame.

- Extracts flat columns (price, seller, title) from raw_data and buybox JSON fields
- Extracts time-series signals (review history, review counts) from raw_data
- Extracts Amazon's real category_path from the raw_data categoryTree (for structured
  query-time filtering — see add_category_tree)
- Filters to products priced $15–$100
- Drops rows with null titles
- Computes review velocity (reviews/day) for products with >= 90 days of review data
- Removes duplicate ASINs, keeping the most recent entry
- Labeling and train/test split are handled in feature_engineering.py
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


# ---------- helpers ----------

def convert_history_to_dict(original_list, list_time):
    """
    Convert a Keepa time-series list to a {days_since_launch: value} dictionary.

    In: original_list (List[int]) alternating keepa timestamps and values, list_time (int) listing timestamp
    Out: dict {days_since_launch: value} or None if original_list is None
    """
    if original_list is None:
        return None
    times = original_list[0::2]
    counts = original_list[1::2]
    days = [int((t - list_time) / 60 / 24) for t in times]
    return dict(zip(days, counts))


def sales_rank_at_months(original_list, starting_days, ending_days, list_time):
    """
    Compute average sales rank within a post-launch time window.

    In: original_list (List[int]) raw Keepa sales rank list, starting_days / ending_days (int) window in days, list_time (int) listing timestamp
    Out: float average sales rank within the window, or None if no data
    """
    if original_list is None:
        return None

    starting_mins = starting_days * 24 * 60 + list_time
    ending_mins = ending_days * 24 * 60 + list_time
    times = original_list[0::2]
    counts = original_list[1::2]

    sum_rank = []
    for i, k in enumerate(times):
        if counts[i] and counts[i] != -1:
            if k >= starting_mins and k <= ending_mins:
                sum_rank.append(counts[i])

    return sum(sum_rank) / len(sum_rank) if len(sum_rank) != 0 else None


def extraction(row):
    """
    Extract time-series signals from a single raw_data JSON string.

    In: row (str) a single raw_data JSON string
    Out: pd.Series with monthly sold, review, and sales rank history columns — joined to df row-wise
    """
    data = json.loads(row)
    list_time = data.get('listedSince')

    time_n_monthly_sold = data.get('monthlySoldHistory')
    time_n_review_count = data.get('reviews', {}).get('reviewCount')
    root_cat = data.get('rootCategory')
    time_n_sales_rank = (data.get('salesRanks') or {}).get(str(root_cat))

    ms_history = convert_history_to_dict(time_n_monthly_sold, list_time)
    review_history = convert_history_to_dict(time_n_review_count, list_time)
    sales_rank_history = convert_history_to_dict(time_n_sales_rank, list_time)

    most_recent_review_time = max(review_history.keys()) if review_history else None
    most_recent_review = review_history[most_recent_review_time] if review_history else None

    if review_history:
        review_na_count = sum([1 for v in review_history.values() if v == -1])
    else:
        review_na_count = None

    return pd.Series({
        'most_recent_review_time': most_recent_review_time,
        'most_recent_review': most_recent_review,
    })


# ---------- main functions ----------

def filter_price(df: pd.DataFrame, min_price: float = 15.0, max_price: float = 100.0) -> pd.DataFrame:
    """
    Filter to products in the target price range.

    In: df with price column
    Out: df filtered to min_price <= price <= max_price
    """
    return df[(df['price'] >= min_price) & (df['price'] <= max_price)].copy()


def add_review_velocity(df: pd.DataFrame, min_days: int = 90) -> pd.DataFrame:
    """
    Compute review velocity (reviews/day) for products with >= min_days of review data.

    In: df with most_recent_review and most_recent_review_time columns
    Out: df with review_velocity column (None for products with < min_days data)
    """
    has_data = df['most_recent_review_time'] >= min_days
    df = df.copy()
    df.loc[has_data, 'review_velocity'] = (
        df.loc[has_data, 'most_recent_review'] / df.loc[has_data, 'most_recent_review_time']
    )
    df.loc[~has_data, 'review_velocity'] = None
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate ASINs, keeping the most recent entry.

    In: df with asin column
    Out: df with one row per asin
    """
    surviving_idx = df.drop_duplicates(subset='asin', keep='last').index.tolist()
    return df.loc[surviving_idx].reset_index(drop=True)


def add_price_seller_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract price, seller, and title from raw_data and buybox JSON fields.

    In: df with raw_data (JSON str) and buybox (JSON str) columns
    Out: df with new columns — listed_price, price, title, seller
    """
    df['listed_price'] = df['raw_data'].map(lambda x: (json.loads(x).get('csv') or [None])[4])
    df['price'] = df['listed_price'].map(
        lambda x: None if x is None else (None if x[1] is None else float(x[1] / 100))
    )
    df['title'] = df['raw_data'].map(lambda x: json.loads(x).get('title'))
    df['seller'] = df['buybox'].map(lambda x: (json.loads(x).get('buyBoxSellerIdHistory') or [None])[-1])
    return df


def add_category_tree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Amazon's real category tree from the raw_data JSON field.

    Keepa's categoryTree is Amazon's own assigned browse-node path for the product
    (e.g. "Pet Supplies > Dogs > Feeding & Watering Supplies > Fountains") — authoritative,
    not derived — so this replaces ad hoc hand-tagged taxonomies for query-time structured
    filtering: an LLM decomposing a query should score against the real distinct paths
    that exist in the data, not an invented category schema.

    In: df with raw_data (JSON str) column
    Out: df with new columns — category_path (str, "A > B > C"), category_levels (list[str] or None)
    """
    def _extract_tree(raw_json):
        tree = json.loads(raw_json).get('categoryTree')
        return [n['name'] for n in tree] if tree else None

    df['category_levels'] = df['raw_data'].map(_extract_tree)
    df['category_path'] = df['category_levels'].map(lambda levels: ' > '.join(levels) if levels else None)
    return df


def add_review_n_monthly_sold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract review history, monthly sold, and sales rank signals from raw_data JSON.

    In: df with raw_data (JSON str) column
    Out: df joined with time-series signal columns from extraction()
    """
    return df.join(df['raw_data'].apply(extraction))


def add_launch_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract launch year and month from the month column and rename it to launch_year_month.

    In: df with month column (datetime)
    Out: df with launch_year, launch_month, launch_year_month columns; month dropped
    """
    df = df.copy()
    df['launch_year'] = pd.to_datetime(df['month']).dt.year
    df['launch_month'] = pd.to_datetime(df['month']).dt.month
    df['launch_year_month'] = df['month']
    df.drop('month', axis=1, inplace=True)
    return df


def run_preprocess(df: pd.DataFrame, output_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """
    Run full preprocessing pipeline and save a parquet checkpoint.

    In: raw df from load_data(), output_dir for saving the parquet
    Out: clean df with price, seller, title, review history, monthly sold, and sales rank columns
    """
    df = add_price_seller_title(df)
    df = add_review_n_monthly_sold(df)
    df = add_category_tree(df)
    df = filter_price(df)
    df = df.dropna(subset=['title']).copy()
    df = add_review_velocity(df)
    df = drop_duplicates(df)
    df = add_launch_date_features(df)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "preprocessed.parquet", index=False)
    df[['asin','seller','cat','launch_year_month','launch_year','launch_month','price','title','most_recent_review','most_recent_review_time','category_path']].to_parquet(outdir / "preprocessed_reduced.parquet", index=False)

    print(f"Preprocessed {df.shape[0]} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = pd.read_parquet('data/raw/product_launch.parquet')
    df = run_preprocess(df)
    print(f"Preprocessed {df.shape[0]} rows, {df.shape[1]} columns")

    
