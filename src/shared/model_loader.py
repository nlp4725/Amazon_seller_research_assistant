from sentence_transformers import SentenceTransformer

# chat_engine.py uses this to embed user queries (e.g. "dog grooming") before
# searching ChromaDB by cosine similarity — loading the model once avoids
# reloading 80MB on every chat message
_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
