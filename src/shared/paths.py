"""
Every filesystem location the project reads or writes, in one place.

Before this module the same paths were re-declared as private constants across
build_chroma.py, candidates.py, classify_agent.py and analysis_agent.py (which
hardcoded the ChromaDB directory inline), so "where does the data live?" had no
single answer. Import from here instead of writing a path literal.

Two stores back the whole system:
  - CHROMA_DIR      embeddings + product metadata, queried at request time
  - PREPROCESSED_PARQUET  the same catalog's category taxonomy, read at request
                    time by cat_selector/classify_agent and used offline to build
                    ChromaDB. Both are produced by src/offline/ (see build_data.py).
"""

from pathlib import Path

# ---------- roots ----------
DATA_DIR      = Path("data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------- raw inputs (offline only) ----------
PRODUCT_DB          = RAW_DIR / "product_launch.db"       # Keepa ingest lands here
PRODUCT_PARQUET     = RAW_DIR / "product_launch.parquet"  # load.py's checkpoint
LOSERS_CSV          = RAW_DIR / "losers_combined.csv"     # confirmed zero-review ASINs

# ---------- the two live stores ----------
CHROMA_DIR            = RAW_DIR / "chroma_db"
CHROMA_COLLECTION     = "title_embedding_db"
PREPROCESSED_PARQUET  = PROCESSED_DIR / "preprocessed_reduced.parquet"

# ---------- caches and run records (regenerable) ----------
CATEGORY_PATH_CACHE   = PROCESSED_DIR / "category_path_cache"
CATEGORY_CLASSIFY_DIR = PROCESSED_DIR / "category_classifications"
PIPELINE_RUNS_DIR     = PROCESSED_DIR / "pipeline_runs"
CAT_SELECTOR_RUNS_DIR = PROCESSED_DIR / "cat_selector_runs"
