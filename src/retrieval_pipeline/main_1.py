"""
Approach 1 ("simple"): given 1-2 root categories (resolved by the caller via
cat_selector.select() -- see below), every item in those categories is vector-ranked and
reranked against the query -- no category-path filtering step. This is the standard
two-stage retrieve-then-rerank pattern (rank: cheap ANN cosine similarity over
everything; rerank: precise cross-encoder scoring), just applied to a whole root
category instead of a narrower structured-filter subset.

  candidates.rank_category_items(query, categories) -- ChromaDB ANN query (rank), no LLM
       -> reranker.rerank_titles(query, items)        -- local cross-encoder (rerank), no LLM
       -> keep items where rerank_score > 0

`categories` is always supplied by the caller -- resolved once via
src.retrieval_pipeline.cat_selector.select() and shared with main_2.py so both approaches
see identical input for a given query (evaluator/comparison.py does this once per query
instead of each main picking its own, independently).

The rank stage does NOT truncate to an arbitrary top_n -- it scores/orders every item in
the category via one batched ANN call (n_results = full category count). Capping top_n
is exactly the counting bug this whole project exists to fix (see analysis_agent.py's
_get_product_subset, which caps at 200 regardless of the true match count); the rank
stage here only orders the pool, the rerank stage's score>0 threshold does the filtering.

This is the cheap/fast approach compared against main_2.py's structured-filter approach
in evaluator/comparison.py. It should be lower cost and latency (no classify step) but is
expected to have worse precision on broad root categories where the query concept is a
small slice of a large catalog.

Every title considered is recorded (asin, title, rerank_score, is_match), not just the
survivors -- same "titles_found" schema as evaluator/golden_dataset.json, so
evaluator/metrics.py can score this output directly.

CLI:
    python -m src.retrieval_pipeline.main_1 "dog drinking bowl"
"""

import json
import sys
import time

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.retrieval_pipeline import llm_client
from src.retrieval_pipeline.candidates import rank_category_items
from src.retrieval_pipeline.reranker import rerank_titles
from src.shared.paths import PIPELINE_RUNS_DIR, safe_name



@traceable(run_type="chain", name="main_1.run")
def run(query: str, categories: list[dict]) -> dict:
    """
    Run the simple approach end-to-end for one query.

    In: query text; categories -- [{"category", "reason"}, ...], resolved by the caller
        via cat_selector.select() (required -- this main never picks its own categories)
    Out: dict with query, approach, categories, titles_found (every item considered,
         each tagged is_match/rerank_score/vector_distance), titles_found_count,
         match_count, latency_s (per stage + total), usage (LLM token/cost accounting
         for this run); also saved to OUTPUT_DIR/simple__{query}.json
    """
    llm_client.reset_usage()
    run_tree = get_current_run_tree()
    trace_id = str(run_tree.trace_id) if run_tree else None
    t0 = time.perf_counter()

    ranked_items = rank_category_items(query, [c["category"] for c in categories])
    t2 = time.perf_counter()

    titles_found = rerank_titles(query, ranked_items)
    t3 = time.perf_counter()

    match_count = sum(t["is_match"] for t in titles_found)

    output = {
        "query": query,
        "approach": "simple",
        "categories": categories,
        "titles_found_count": len(titles_found),
        "match_count": match_count,
        "titles_found": titles_found,
        "latency_s": {
            "rank": round(t2 - t0, 3),
            "rerank": round(t3 - t2, 3),
            "total": round(t3 - t0, 3),
        },
        "usage": llm_client.get_usage(),
        "langsmith_trace_id": trace_id,
    }

    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PIPELINE_RUNS_DIR / f"simple__{safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[simple] {query!r}: {match_count} matches / {len(titles_found)} considered "
          f"in {output['latency_s']['total']}s, ${output['usage']['estimated_cost_usd']}")
    print(f"Wrote {out_path}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.retrieval_pipeline.main_1 "<query>"')
        sys.exit(1)

    from src.retrieval_pipeline import cat_selector

    cli_query = sys.argv[1]
    resolved = cat_selector.select([{"role": "user", "content": cli_query}])
    if "clarify" in resolved:
        print(f"cat_selector wants clarification: {resolved['clarify']}")
        sys.exit(1)

    run(query=cli_query, categories=resolved["categories"])
