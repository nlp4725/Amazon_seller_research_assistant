"""
Title reranker: cross-encoder relevance scoring for (query, product title) pairs.

Job: given a query and a list of candidate items (each with at least a "title"), score
every (query, title) pair with a cross-encoder and keep the ones that score > 0.

score > 0 is validated ONLY at this granularity -- (query, full product title), rich
sentence-length text -- not against short category-label text. An earlier attempt to
apply this same model/threshold to category-tree node names (e.g. "Fountains", "Pet
Supplies") instead of full titles was abandoned: the same model scores EVERY category
label negative, including correct ones, so 0 isn't a meaningful cutoff there. Here, on
real titles, it is: it's the threshold this whole pipeline settled on after comparing it
against top-K and absolute-vector-floor alternatives earlier in the project.

No LLM calls -- local model only, so this is cheap enough to run over thousands of
titles (e.g. every item in a broad root category for the "simple" approach in main_1.py).

CLI:
    python -m src.retrieval_pipeline.reranker "dog drinking bowl" "Fountain for Dogs" "Cat Litter Box"
"""

import sys

from langsmith import traceable
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SCORE_FLOOR = 0.0
BATCH_SIZE = 128

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)
    return _reranker


@traceable(run_type="tool", name="reranker.rerank_titles")
def rerank_titles(
    query: str,
    items: list[dict],
    title_key: str = "title",
    score_floor: float = SCORE_FLOOR,
) -> list[dict]:
    """
    Score every item's title against the query with the cross-encoder.

    In: query text, items (list of dicts, each with at least items[i][title_key]),
        title_key (which key holds the title text), score_floor (kept if score > this)
    Out: new list of dicts -- each input item plus "rerank_score" (float) and
         "is_match" (bool, score > score_floor) -- same order as input, NOT filtered or
         sorted, so callers can inspect every title considered, not just the survivors
    """
    if not items:
        return []

    model = get_reranker()
    pairs = [(query, item[title_key]) for item in items]
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

    scored = []
    for item, score in zip(items, scores):
        scored.append({**item, "rerank_score": float(score), "is_match": bool(score > score_floor)})
    return scored


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m src.retrieval_pipeline.reranker "<query>" "<title1>" "<title2>" ...')
        sys.exit(1)

    query, *titles = sys.argv[1:]
    results = rerank_titles(query, [{"title": t} for t in titles])
    for r in sorted(results, key=lambda r: -r["rerank_score"]):
        mark = "MATCH" if r["is_match"] else "     "
        print(f"  {mark}  {r['rerank_score']:7.3f}  {r['title']}")
