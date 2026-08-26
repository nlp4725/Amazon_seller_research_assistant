"""
The embedding model, loaded once per process.

all-MiniLM-L6-v2 is ~80MB. Without this singleton, candidates.rank_category_items
would construct a fresh SentenceTransformer on every chat request.

EMBEDDING_MODEL is declared here and nowhere else on purpose. Two places embed
with it -- build_chroma.py encodes the catalog, candidates.py encodes the incoming
query -- and they must use the same model or the vectors are incomparable. That
failure is silent: no error, no crash, just meaningless similarity scores. One
constant means the two can't drift.

(The cross-encoder in reranker.py is a different model for a different job --
scoring (query, title) pairs, not producing vectors -- so it owns its own name.)
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output width

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
