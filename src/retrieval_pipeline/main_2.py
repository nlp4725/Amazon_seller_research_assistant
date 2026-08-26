"""
Approach 2 ("structured"): given 1-2 root categories (resolved by the caller via
cat_selector.select() -- see below), classify_agent scores every real category_path under
each into confident/ambiguous/not_match, confident paths are trusted as an exact filter
(no title-reading), and only the ambiguous residual is handed to the reranker.

  for each category: classify_agent.classify_paths() -- 1+ LLM calls (batched)
       -> confident_match paths  -> candidates.fetch_items_for_paths()  -- trusted, no reranking
       -> ambiguous_match paths  -> candidates.fetch_items_for_paths()
            -> reranker.rerank_titles(query, items)      -- local cross-encoder, no LLM
                 -> keep items where rerank_score > 0

`categories` is always supplied by the caller -- resolved once via
src.retrieval_pipeline.cat_selector.select() and shared with main_1.py so both approaches
see identical input for a given query (evaluator/comparison.py does this once per query
instead of each main picking its own, independently).

This is the expensive/slow approach compared against main_1.py's simple approach in
evaluator/comparison.py -- one classify_paths call per candidate category, each itself a
batched multi-call LLM scoring pass over every real path in that category. Expected to
win on precision/recall on broad root categories where the query concept is a small,
identifiable slice (a real category_path exists for it) of a large catalog.

Every title considered is recorded -- both the confident-filter items (is_match=True,
rerank_score=None, since they were never scored by the reranker) and every ambiguous
item the reranker scored -- same "titles_found" schema as evaluator/golden_dataset.json.

CLI:
    python -m src.retrieval_pipeline.main_2 "dog drinking bowl"
"""

import json
import sys
import time

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.retrieval_pipeline import llm_client
from src.retrieval_pipeline.candidates import fetch_items_for_paths
from src.retrieval_pipeline.classify_agent import classify_paths
from src.retrieval_pipeline.reranker import rerank_titles
from src.shared.paths import PIPELINE_RUNS_DIR, safe_name



@traceable(run_type="chain", name="main_2.run")
def run(query: str, categories: list[dict]) -> dict:
    """
    Run the structured approach end-to-end for one query.

    In: query text; categories -- [{"category", "reason"}, ...], resolved by the caller
        via cat_selector.select() (required -- this main never picks its own categories)
    Out: dict with query, approach, categories, classifications (per-category
         confident/ambiguous/not_match path counts), titles_found (every item
         considered -- confident-filter items plus every reranked ambiguous item, each
         tagged is_match), titles_found_count, match_count, latency_s (per stage +
         total), usage (LLM token/cost accounting for this run); also saved to
         OUTPUT_DIR/structured__{query}.json
    """
    llm_client.reset_usage()
    run_tree = get_current_run_tree()
    trace_id = str(run_tree.trace_id) if run_tree else None
    t0 = time.perf_counter()

    classifications = [classify_paths(query, c["category"]) for c in categories]
    t2 = time.perf_counter()

    confident_paths, ambiguous_paths = [], []
    for c in classifications:
        confident_paths.extend(c["confident_match"].values())
        ambiguous_paths.extend(c["ambiguous_match"].values())

    confident_items = fetch_items_for_paths(confident_paths)
    ambiguous_items = fetch_items_for_paths(ambiguous_paths)
    t3 = time.perf_counter()

    confident_titles = [
        {**item, "rerank_score": None, "is_match": True, "source": "confident_filter"}
        for item in confident_items
    ]
    ambiguous_titles = [
        {**item, "source": "reranked_ambiguous"}
        for item in rerank_titles(query, ambiguous_items)
    ]
    t4 = time.perf_counter()

    titles_found = confident_titles + ambiguous_titles
    match_count = sum(t["is_match"] for t in titles_found)

    output = {
        "query": query,
        "approach": "structured",
        "categories": categories,
        "classifications": [
            {
                "category": c["category"],
                "confident_match_count": len(c["confident_match"]),
                "ambiguous_match_count": len(c["ambiguous_match"]),
                "not_match_count": len(c["not_match"]),
                "partial_failure": c["partial_failure"],
            }
            for c in classifications
        ],
        "titles_found_count": len(titles_found),
        "match_count": match_count,
        "titles_found": titles_found,
        "latency_s": {
            "classify": round(t2 - t0, 3),
            "retrieval": round(t3 - t2, 3),
            "rerank": round(t4 - t3, 3),
            "total": round(t4 - t0, 3),
        },
        "usage": llm_client.get_usage(),
        "langsmith_trace_id": trace_id,
    }

    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PIPELINE_RUNS_DIR / f"structured__{safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[structured] {query!r}: {match_count} matches / {len(titles_found)} considered "
          f"({len(confident_titles)} confident + {len(ambiguous_titles)} reranked-ambiguous) "
          f"in {output['latency_s']['total']}s, ${output['usage']['estimated_cost_usd']}")
    print(f"Wrote {out_path}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.retrieval_pipeline.main_2 "<query>"')
        sys.exit(1)

    from src.retrieval_pipeline import cat_selector

    cli_query = sys.argv[1]
    resolved = cat_selector.select([{"role": "user", "content": cli_query}])
    if "clarify" in resolved:
        print(f"cat_selector wants clarification: {resolved['clarify']}")
        sys.exit(1)

    run(query=cli_query, categories=resolved["categories"])
