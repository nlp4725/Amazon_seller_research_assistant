"""
Approach 2 ("structured"): a three-stage funnel over 1-2 root categories (resolved by the
caller via cat_selector.select() -- see below).

  1. Path identification -- classify_agent scores every real category_path under each
     category into confident / ambiguous / not_match (Jev by default; DeepSeek with
     CLASSIFIER_BACKEND=deepseek). Judging distinct paths instead of items keeps this
     stage's cost tied to the taxonomy, not the catalog (Home & Kitchen: 629 paths vs.
     5,354 items).
  2. Rerank -- the local cross-encoder scores every title in the ambiguous paths;
     rerank_score > 0 is a candidate match. Confident paths skip this stage.
  3. LLM filtering -- title_filter asks Jev, per candidate title, whether the product
     itself is what the query asks for, and drops the substitutes and complements the
     first two stages cannot see (TITLE_FILTER=off to skip).

  for each category: classify_agent.classify_paths()
       -> confident_match paths -> candidates.fetch_items_for_paths()   -- candidates
       -> ambiguous_match paths -> candidates.fetch_items_for_paths()
            -> reranker.rerank_titles(query, items) -> rerank_score > 0 -- candidates
  all candidates -> title_filter.apply() -> p >= TITLE_FILTER_CUTOFF   -- matches

Measured against golden v2 (evaluator/golden_v2/README.md, evaluator/results/): Jev for
stage 1 cut latency ~100x vs. DeepSeek (345 s -> ~3 s per query), and stage 3 raised mean
precision 0.65 -> 0.89 and F1 0.60 -> 0.71 on 16 scored queries. Known limit, measured:
stage 1 judges path NAMES, so products sellers filed under an unexpected path (Bluetooth
speakers under "MP3 & MP4 Player Accessories", multi-pet fountains under Cats) never reach
stages 2-3 -- recall, not precision, is where this design loses.

`categories` is always supplied by the caller -- resolved once via
src.retrieval_pipeline.cat_selector.select() and shared with main_1.py so both approaches
see identical input for a given query (evaluator/comparison.py does this once per query
instead of each main picking its own, independently).

This is the expensive/slow approach compared against main_1.py's simple approach in
evaluator/comparison.py -- one classify_paths call per candidate category, each itself a
batched multi-call LLM scoring pass over every real path in that category. Expected to
win on precision/recall on broad root categories where the query concept is a small,
identifiable slice (a real category_path exists for it) of a large catalog.

Every title considered is recorded -- confident-path items (rerank_score=None, never
reranked) and every ambiguous item the reranker scored -- and every item that reached
stage 3 carries title_filter_p and matched_before_filter, so each stage's effect can be
read back from the saved run.

CLI:
    python -m src.retrieval_pipeline.main_2 "dog drinking bowl"
"""

import json
import sys
import time

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.retrieval_pipeline import llm_client, title_filter
from src.retrieval_pipeline.candidates import fetch_items_for_paths
from src.retrieval_pipeline.classify_agent import classify_paths
from src.retrieval_pipeline.reranker import rerank_titles
from src.shared.paths import PIPELINE_RUNS_DIR, safe_name


@traceable(run_type="chain", name="main_2.run")
def run(query: str, categories: list[dict], classifications: list[dict] | None = None) -> dict:
    """
    Run the structured approach end-to-end for one query.

    In: query text; categories -- [{"category", "reason"}, ...], resolved by the caller
        via cat_selector.select() (required -- this main never picks its own categories);
        classifications -- optional pre-computed classify_paths() output, one per category,
        for paired evaluation against main_2b (see the assignment below)
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

    # classifications may be supplied by an evaluation so this and main_2b score the SAME
    # classification pass -- classify_paths is nondeterministic enough to swamp the
    # difference between the two designs otherwise (precision 0.727 vs 0.321 measured on
    # two runs of "dog drinking bowl"). Omitted in normal use, where it computes its own.
    if classifications is None:
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
    candidate_count = sum(t["is_match"] for t in titles_found)
    if title_filter.TITLE_FILTER:
        titles_found = title_filter.apply(query, titles_found)
    t5 = time.perf_counter()
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
        "candidate_count": candidate_count,
        "title_filter": ({"cutoff": title_filter.TITLE_FILTER_CUTOFF} if title_filter.TITLE_FILTER else None),
        "match_count": match_count,
        "titles_found": titles_found,
        "latency_s": {
            "classify": round(t2 - t0, 3),
            "retrieval": round(t3 - t2, 3),
            "rerank": round(t4 - t3, 3),
            "filter": round(t5 - t4, 3),
            "total": round(t5 - t0, 3),
        },
        "usage": llm_client.get_usage(),
        "langsmith_trace_id": trace_id,
    }

    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PIPELINE_RUNS_DIR / f"structured__{safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[structured] {query!r}: {match_count} matches / {candidate_count} candidates / "
          f"{len(titles_found)} considered "
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
