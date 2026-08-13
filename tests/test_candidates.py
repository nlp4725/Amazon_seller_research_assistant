import numpy as np
import pandas as pd

from src.retrieval_pipeline.candidates import hydrate_items

# Two real asins known to exist in the local ChromaDB fixture (data/raw/chroma_db) --
# same pattern as tests/test_build_chroma.py, which also hits the real persistent DB.
KNOWN_ASINS = ["B0CRHVWXYX", "B0CSHP2DQB"]


def test_hydrate_items_returns_full_metadata_row_per_asin():
    df, emb = hydrate_items(KNOWN_ASINS)

    assert len(df) == len(KNOWN_ASINS)
    assert set(df["asin"]) == set(KNOWN_ASINS)
    for col in ["asin", "title", "cat", "category_path", "price", "seller",
                "review_velocity", "launch_year", "launch_year_month"]:
        assert col in df.columns

    # chat_engine's downstream tools filter/group on launch_year_month as a Timestamp,
    # same as _get_product_subset's existing pd.to_datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(df["launch_year_month"])


def test_hydrate_items_embeddings_row_aligned_with_df():
    df, emb = hydrate_items(KNOWN_ASINS)

    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == len(df)
    assert emb.shape[1] == 384  # all-MiniLM-L6-v2 dimension


def test_hydrate_items_empty_input_returns_empty():
    df, emb = hydrate_items([])

    assert len(df) == 0
    assert emb.shape[0] == 0
