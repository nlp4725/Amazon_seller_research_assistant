"""
Shared ChromaDB candidate-retrieval helpers -- used by both main_1.py (simple: vector-rank
everything in the picked root categories, then rerank) and main_2.py (structured:
exact-filter on classify_agent's confident/ambiguous category_paths, rerank only the
ambiguous residual).

Both retrieval shapes hit the same ChromaDB collection build_chroma.py already builds
(title_embedding_db), so they live in one module instead of being duplicated per main.

CLI:
    python -m src.retrieval_pipeline.candidates "dog drinking bowl" "Pet Supplies"
"""

import sys

import chromadb
import numpy as np
import pandas as pd

from src.shared.model_loader import get_embedder
from src.shared.paths import CHROMA_COLLECTION, CHROMA_DIR

_chroma_col = None


def _load_chroma():
    global _chroma_col
    if _chroma_col is None:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_col = chroma_client.get_collection(CHROMA_COLLECTION)
    return _chroma_col


def rank_category_items(query: str, categories: list[str]) -> list[dict]:
    """
    Vector-rank every item in the given root categories against the query -- the "rank"
    stage of main_1.py's two-stage rank+rerank approach. Uses ChromaDB's ANN query
    (ranked, with distances) instead of a flat fetch, but n_results is set to the
    category's full count, not an arbitrary cap -- capping top_n is exactly the counting
    bug this whole project exists to fix (see chat_engine.py's _get_product_subset,
    which caps at 200 regardless of the true match count). The point of this stage is
    ranking/scoring every item cheaply via cosine similarity, not truncating the pool;
    the reranker downstream does the actual score>0 filtering.

    In: query text, list of root category names
    Out: list of {asin, title, cat, category_path, vector_distance} dicts, sorted by
         ascending distance (most similar first); [] if the category is empty
    """
    col = _load_chroma()
    where = {"cat": {"$in": categories}} if len(categories) > 1 else {"cat": categories[0]}

    total = col.get(where=where, include=[])
    n = len(total["ids"])
    if n == 0:
        return []

    model = get_embedder()
    q_emb = model.encode([query]).tolist()
    results = col.query(
        query_embeddings=q_emb,
        n_results=n,
        where=where,
        include=["metadatas", "distances"],
    )
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        {
            "asin": m["asin"], "title": m["title"], "cat": m["cat"],
            "category_path": m.get("category_path", ""), "vector_distance": float(d),
        }
        for m, d in zip(metadatas, distances)
    ]


def fetch_items_for_paths(paths: list[str]) -> list[dict]:
    """
    Every item whose real category_path is exactly one of the given paths -- used for
    both classify_agent buckets in main_2.py: confident_match paths are trusted outright
    (no reranking), ambiguous_match paths' items are the pool handed to the reranker.

    In: list of exact category_path strings (e.g. "Pet Supplies > Dogs > Fountains")
    Out: list of {asin, title, cat, category_path} dicts; [] if paths is empty
    """
    if not paths:
        return []
    col = _load_chroma()
    results = col.get(
        where={"category_path": {"$in": paths}} if len(paths) > 1 else {"category_path": paths[0]},
        include=["metadatas"],
    )
    return [
        {"asin": m["asin"], "title": m["title"], "cat": m["cat"], "category_path": m.get("category_path", "")}
        for m in results["metadatas"]
    ]


def hydrate_items(asins: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full product rows + embeddings for a set of matched asins.

    main_1/main_2's `titles_found` schema only carries asin/title/cat/category_path
    (+ pipeline-specific fields like rerank_score) -- not price/seller/launch_year_month/
    review_velocity, which chat_engine's downstream tools (recent launches, theme-trend
    clustering, top sellers, velocity summary) need. This does one batched ChromaDB
    fetch by id to hydrate a matched-asin set with everything those tools require.

    In: list of asins (e.g. is_match=True items from a pipeline run)
    Out: (df, emb) -- df has one row per asin with full metadata (launch_year_month
         parsed to Timestamp, same as _get_product_subset), emb is the row-aligned
         (n, 384) embedding matrix; both empty if asins is empty
    """
    if not asins:
        return pd.DataFrame(), np.empty((0, 0))

    col = _load_chroma()
    results = col.get(ids=asins, include=["metadatas", "embeddings"])
    df = pd.DataFrame(results["metadatas"])
    df["launch_year_month"] = pd.to_datetime(df["launch_year_month"])
    emb = np.array(results["embeddings"])
    return df, emb


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m src.retrieval_pipeline.candidates "<query>" "<category>" ["<category2>" ...]')
        sys.exit(1)

    query, categories = sys.argv[1], sys.argv[2:]
    items = rank_category_items(query, categories)
    print(f"{len(items)} items in {categories}, ranked against {query!r}")
