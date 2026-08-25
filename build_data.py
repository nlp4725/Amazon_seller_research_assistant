"""
Build every data artifact the assistant serves from.

    python build_data.py

Stages, in order:
  1. load          raw Keepa rows out of SQLite, parquet checkpoint in data/raw/
  2. preprocess    clean, filter to $15-$100, extract Amazon's real category_path
                     -> data/processed/preprocessed_reduced.parquet
  3. build_chroma  embed titles (all-MiniLM-L6-v2), upsert into ChromaDB
                     -> data/raw/chroma_db

Stage 2 and stage 3 produce the only two stores the live request path reads:
cat_selector/classify_agent read the category taxonomy out of the parquet, and
everything else queries ChromaDB. Nothing in here runs in production -- this is
the offline job that produces what production reads.

Both later stages are resumable: preprocessing is skipped when the parquet already
exists, and build_chroma no-ops when the collection is already up to date, so
re-running this is cheap.

Collecting the raw data in the first place is a separate, rate-limited job against
the Keepa API -- see src/offline/ingest.py. It is not part of this chain.
"""

from src.offline.build_chroma import build_chroma
from src.offline.load import load_data
from src.offline.preprocessing import run_preprocess
from src.shared.paths import PREPROCESSED_PARQUET


def build_all() -> None:
    print("=== Step 1: Load ===")
    df = load_data()  # reads data/raw/product_launch.parquet directly (SQLite DB is empty)

    print("\n=== Step 2: Preprocess ===")
    if PREPROCESSED_PARQUET.exists():
        print(f"{PREPROCESSED_PARQUET} found — skipping preprocessing")
    else:
        run_preprocess(df)

    print("\n=== Step 3: Build ChromaDB ===")
    build_chroma()

    print("\n=== Data build complete ===")
    print(f"  taxonomy + metadata: {PREPROCESSED_PARQUET}")
    print("  embeddings:          data/raw/chroma_db")


if __name__ == "__main__":
    build_all()
