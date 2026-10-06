"""
Approach 2b ("structured_b"): same classification step as main_2.py, but NO confident /
ambiguous distinction -- every path classify_agent scored at or above
AMBIGUOUS_SCORE_FLOOR becomes a candidate, and ALL of their items go to the reranker.

  for each category: classify_agent.classify_paths() -- 1+ LLM calls (batched), unchanged
       -> confident_match + ambiguous_match paths (i.e. score >= AMBIGUOUS_SCORE_FLOOR)
            -> candidates.fetch_items_for_paths()
                 -> reranker.rerank_titles(query, items)   -- local cross-encoder, no LLM
                      -> keep items where rerank_score > SCORE_FLOOR

WHY THIS EXISTS
main_2 admits confident_match items with is_match=True and never scores them, so nothing
downstream can remove them. On "dog drinking bowl" that cost most of the precision:
classify_agent marked BOTH "Pet Supplies > Cats > ... > Fountains" and the Dogs equivalent
as confident, so 21 items entered untouched, 12 of them cat products, against 11 true
matches -- precision 0.321 while main_1 managed 0.381 on the same query. One over-confident
path classification is unrecoverable by construction.

Reranking those same 31 candidates instead scored 0.500 precision / 0.636 recall
(F1 0.560 vs main_2's 0.462) at the current SCORE_FLOOR of 0. Note the trade: recall FELL
from 0.818, because the cross-encoder scored two genuine dog fountains negative
(-2.05, -2.35) for titles reading "Water Fountain for Dogs ... for Cats/Dogs". So this is
not free -- it swaps an unrecoverable classifier error for a recoverable reranker error.

The classification cost is identical to main_2 (same calls, same batches); reranking a few
more items is a local cross-encoder pass, so effectively free. The difference is purely
which model gets the final say.

Collapsing the split also makes CONFIDENT_SCORE_FLOOR unused in this path -- 2b needs only
"candidate vs not_match", not a three-way bucketing.

CLI:
    python -m src.retrieval_pipeline.main_2b "dog drinking bowl"
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


def candidate_paths(classification: dict) -> list[str]:
    """
    Every path worth fetching for 2b: the union of both match buckets.

    classify_paths() splits on two floors -- not_match is exactly "score below
    AMBIGUOUS_SCORE_FLOOR or unscored" -- so confident_match | ambiguous_match IS the
    "score >= AMBIGUOUS_SCORE_FLOOR" set. Taking the union here rather than re-deriving it
    from raw scores keeps 2b reading the same buckets main_2 does, so the two designs can
    share one classification pass in an evaluation without drifting apart.
    """
    return list(classification["confident_match"].values()) + list(
        classification["ambiguous_match"].values()
    )


@traceable(run_type="chain", name="main_2b.run")
def run(query: str, categories: list[dict], classifications: list[dict] | None = None) -> dict:
    """
    Run the structured_b approach end-to-end for one query.

    In: query text; categories -- [{"category", "reason"}, ...], resolved by the caller via
        cat_selector.select() (required -- this main never picks its own categories);
        classifications -- optional pre-computed classify_paths() output, one per category,
        so an evaluation can score 2 and 2b against an IDENTICAL classification instead of
        paying for two passes whose nondeterminism would confound the comparison (main_2 on
        "dog drinking bowl" measured precision 0.727 and 0.321 on two runs of the same
        input). Omit it and this computes its own, like main_2 does.
    Out: dict with query, approach, categories, classifications (per-category
         confident/ambiguous/not_match path counts), titles_found (every item the reranker
         scored, each tagged is_match), titles_found_count, match_count, latency_s, usage,
         langsmith_trace_id; also saved to PIPELINE_RUNS_DIR/structured_b__{query}.json
    """
    llm_client.reset_usage()
    run_tree = get_current_run_tree()
    trace_id = str(run_tree.trace_id) if run_tree else None
    t0 = time.perf_counter()

    if classifications is None:
        classifications = [classify_paths(query, c["category"]) for c in categories]
    t2 = time.perf_counter()

    paths: list[str] = []
    for c in classifications:
        paths.extend(candidate_paths(c))

    items = fetch_items_for_paths(paths)
    t3 = time.perf_counter()

    # The one behavioural difference from main_2: every candidate is scored, none trusted.
    titles_found = [
        {**item, "source": "reranked_candidate"} for item in rerank_titles(query, items)
    ]
    t4 = time.perf_counter()

    match_count = sum(t["is_match"] for t in titles_found)

    output = {
        "query": query,
        "approach": "structured_b",
        "categories": categories,
        "classifications": [
            {
                "category": c["category"],
                "confident_match_count": len(c["confident_match"]),
                "ambiguous_match_count": len(c["ambiguous_match"]),
                "not_match_count": len(c["not_match"]),
                "candidate_path_count": len(candidate_paths(c)),
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
    out_path = PIPELINE_RUNS_DIR / f"structured_b__{safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[structured_b] {query!r}: {match_count} matches / {len(titles_found)} considered "
          f"(all reranked, none trusted) in {output['latency_s']['total']}s, "
          f"${output['usage']['estimated_cost_usd']}", flush=True)
    print(f"Wrote {out_path}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.retrieval_pipeline.main_2b "<query>"')
        sys.exit(1)

    from src.retrieval_pipeline import cat_selector

    resolved = cat_selector.select([{"role": "user", "content": sys.argv[1]}])
    if "clarify" in resolved:
        print(f"cat_selector wants clarification: {resolved['clarify']}")
        sys.exit(0)
    run(sys.argv[1], categories=resolved["categories"])
