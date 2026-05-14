import numpy as np
import pandas as pd
from src.model_loader import get_embedder

_kw_stats = None


def _load_kw_stats() -> pd.DataFrame:
    global _kw_stats
    if _kw_stats is None:
        df = pd.read_parquet("data/keyword_stats_cleaned.parquet")
        hk = df[df["cat"] == "Home & Kitchen"]
        _kw_stats = (
            hk.groupby("keyword")
            .agg(
                avg_n_product=("n_product", "mean"),
                sum_n_product=("n_product", "sum"),
                max_n_product=("n_product", "max"),
                avg_price_median=("price_median", "mean"),
            )
            .reset_index()
        )
    return _kw_stats


def _keyword_stats_for_title(title: str, kw_stats: pd.DataFrame) -> dict:
    words = [w.lower() for w in title.split()]
    bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    candidates = set(words + bigrams)
    matched = kw_stats[kw_stats["keyword"].str.lower().isin(candidates)]
    cols = ["avg_n_product", "sum_n_product", "max_n_product", "avg_price_median"]
    if len(matched):
        return matched[cols].mean().to_dict()
    return kw_stats[cols].median().to_dict()


def build_features(titles: list[str]) -> pd.DataFrame:
    """
    Takes a list of product titles and returns a DataFrame ready for model.predict().
    Columns: title, emb_0..emb_383, avg_n_product, sum_n_product,
             max_n_product, avg_price_median, plus metadata defaults.
    """
    embedder = get_embedder()
    kw_stats = _load_kw_stats()

    embeddings = embedder.encode(titles, show_progress_bar=False)

    rows = []
    for title, emb in zip(titles, embeddings):
        stats = _keyword_stats_for_title(title, kw_stats)
        emb_dict = {f"emb_{i}": float(v) for i, v in enumerate(emb)}
        rows.append({
            "asin": "UNKNOWN",
            "seller": "-1",
            "cat": "Home & Kitchen",
            "price": stats["avg_price_median"],
            "title": title,
            "most_recent_review": 0,
            "most_recent_review_time": 0,
            "launch_year": 2025,
            "launch_month": 6,
            "launch_year_month": pd.Timestamp("2025-06-01"),
            "review_velocity": 0.0,
            **emb_dict,
            **stats,
        })

    return pd.DataFrame(rows)
