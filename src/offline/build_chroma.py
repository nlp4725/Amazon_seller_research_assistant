"""
Build ChromaDB from the cleaned product data.

- Loads preprocessed_reduced.parquet
- Merges losers_combined.csv to set most_recent_review=0 for confirmed losers
- Computes review_velocity = most_recent_review / most_recent_review_time (where >= 90 days, else -1)
- Connects to ChromaDB and checks if it is already up to date
- If not, encodes new titles and upserts them with metadata
"""

import pandas as pd
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.shared.paths import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    LOSERS_CSV,
    PREPROCESSED_PARQUET,
)

BATCH_SIZE = 1000
SCRAPER_DATE = pd.Timestamp("2026-05-12")


def _add_review_velocity(df: pd.DataFrame, losers_path: Path | str = LOSERS_CSV) -> pd.DataFrame:
    """
    Merge confirmed losers and compute review_velocity for all products.
    Losers (review_count == 0 at SCRAPER_DATE) get most_recent_review=0 and
    most_recent_review_time=days since launch.
    review_velocity == -1 means not enough data (< 90 days).

    In: df with most_recent_review, most_recent_review_time, launch_year_month, asin
    Out: df with review_velocity column
    """
    losers = pd.read_csv(losers_path)
    loser_asins = set(losers["asin"].tolist())

    mask = df["asin"].isin(loser_asins)
    df.loc[mask, "most_recent_review_time"] = (SCRAPER_DATE - df.loc[mask, "launch_year_month"]).dt.days
    df.loc[mask, "most_recent_review"] = 0

    has_data = df["most_recent_review_time"] >= 90
    df["review_velocity"] = -1.0  # -1 = unknown (< 90 days of data)
    df.loc[has_data, "review_velocity"] = (
        df.loc[has_data, "most_recent_review"] / df.loc[has_data, "most_recent_review_time"]
    ).round(4)
    return df


def build_chroma(
    parquet_path: Path | str = PREPROCESSED_PARQUET,
    chroma_path: Path | str = CHROMA_DIR,
    losers_path: Path | str = LOSERS_CSV,
) -> None:
    """
    Upsert new product embeddings into ChromaDB.
    Skips if ChromaDB is already up to date.

    In: path to preprocessed_reduced.parquet, path to chroma directory, path to losers_combined.csv
    Out: None (writes to chroma_path on disk)
    """
    df = pd.read_parquet(parquet_path)
    df = _add_review_velocity(df, losers_path)
    df["launch_year_month"] = df["launch_year_month"].astype(str)
    df["category_path"] = df["category_path"].fillna("")  # ChromaDB metadata rejects None/NaN

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    already_done = collection.count()
    print(f"ChromaDB: {already_done}/{len(df)} products indexed")

    if already_done == len(df):
        print("ChromaDB is up to date — nothing to do")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    new_df = df.iloc[already_done:]

    for i in range(0, len(new_df), BATCH_SIZE):
        batch = new_df.iloc[i : i + BATCH_SIZE]
        embeddings = model.encode(batch["title"].tolist(), show_progress_bar=False)
        collection.upsert(
            ids=batch["asin"].tolist(),
            embeddings=embeddings.tolist(),
            metadatas=batch[["asin", "seller", "cat", "launch_year_month", "launch_year", "price", "title", "review_velocity", "category_path"]].to_dict("records"),
        )
        print(f"  {already_done + i + len(batch)}/{len(df)}")

    print(f"Done: {collection.count()} products in ChromaDB")


if __name__ == "__main__":
    build_chroma()
