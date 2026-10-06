"""
Per-stage recall ceilings for a pipeline run -- the no-LLM tier of the evaluation. No
API calls, no re-running the pipeline: every number here is computed from artifacts
main_1/main_2 already wrote to data/processed/pipeline_runs/.

evaluator/metrics.py answers "how good was the final answer". This module answers "which
stage lost it". Every stage before the reranker is a filter, and the costs are
asymmetric: whatever a filter drops is unrecoverable downstream, whatever it admits the
reranker can still remove. So the useful shape is a chain of recall ceilings, each
bounded by the one before it -- the biggest drop is the bottleneck.

  ground truth (golden_dataset.json)
    -> S1 cat_selector  : truth items whose root `cat` is in the categories it picked
    -> S2 candidate pool: truth items the run actually fetched into titles_found
    -> S3 rerank output : truth items it tagged is_match

Every number is ITEM-weighted, never path- or category-weighted. A category_path holding
800 products and one holding 1 are not worth the same. Accuracy over paths is useless
here for the same reason it's useless anywhere with this class balance: ~98.5% of paths
classify as not_match (kids costumes: 963 of 978), so a label-everything-not_match model
scores 98.5% accuracy and returns zero products.

Recall is always reported next to the pool size that bought it. Recall alone has a
trivial optimum -- admit everything -- which is exactly what main_1 does: ~100% ceiling,
27k titles reranked, precision 0.43, ~190s of cross-encoder time on a 2-vCPU box. The
pair is the metric; neither half means anything on its own.

S1's ground truth is DERIVED, not read from golden_dataset.json's "category_hint". The
hint records the naive guess and is wrong exactly where it matters: on "car phone mount"
it says Automotive, but 58 of 60 true items live in Cell Phones & Accessories, so
scoring against the hint would mark a correct selection wrong. The real label is the set
of root categories the true-match ASINs actually sit in, looked up in the preprocessed
parquet.

Queries whose ground truth is empty (4 of the 20 in golden_dataset.json) are not scored
as a recall chain -- there is nothing to recall. They are scored as a rejection task
instead: did the run correctly return nothing, and if not, how many false positives did
it invent. Reported separately, never averaged into the recall numbers, since a mean
over both would mix two different measurements (see metrics.compute_metrics's
zero-denominator conventions, which push precision and recall in opposite directions).

CLI:
    python -m evaluator.stage_metrics data/processed/pipeline_runs/structured__summer_dress.json
    python -m evaluator.stage_metrics          # score every saved run as one table
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

from evaluator.metrics import find_golden_entry, ground_truth_asins, load_golden_dataset
from src.shared.paths import PIPELINE_RUNS_DIR, PREPROCESSED_PARQUET

_catalog = None


def _load_catalog() -> pd.DataFrame:
    """asin-indexed catalog, loaded once per process -- the source of the derived S1/S2 labels."""
    global _catalog
    if _catalog is None:
        _catalog = pd.read_parquet(PREPROCESSED_PARQUET).set_index("asin")
    return _catalog


def truth_roots(truth: set[str]) -> dict[str, int]:
    """
    Root categories the ground-truth items actually live in, with per-root item counts.

    This is cat_selector's real label -- see module docstring on why category_hint isn't.

    In: set of ground-truth ASINs
    Out: {root category name: item count}, largest first; ASINs missing from the catalog
         are dropped (they can't be attributed to a category)
    """
    cats = _load_catalog().reindex(list(truth))["cat"].dropna()
    return cats.value_counts().to_dict()


def _reranked_asins(pipeline_output: dict) -> set[str]:
    """
    ASINs the cross-encoder actually scored -- main_2 reranks only the ambiguous residual
    (confident-filter items are trusted unscored), so scoring the reranker against the
    whole run would re-measure classify_agent's errors as if they were the reranker's.
    main_1 has no confident bucket and no "source" key: everything it considered was
    scored.
    """
    tf = pipeline_output.get("titles_found", [])
    if any("source" in t for t in tf):
        return {t["asin"] for t in tf if t.get("source") == "reranked_ambiguous"}
    return {t["asin"] for t in tf}


def stage_chain(pipeline_output: dict, dataset: list[dict] | None = None) -> dict:
    """
    Recall ceiling at each stage of one main_1/main_2 run, plus the reranker scored in
    isolation on the residual it was handed.

    In: pipeline_output (as saved by main_1.run / main_2.run), optional pre-loaded
        golden dataset
    Out: dict with query/approach/categories, truth_count, true_roots (derived S1 label),
         and per-stage {kept, recall, pool} for s1_cat_selector / s2_candidate_pool /
         s3_rerank_output, plus rerank_isolated (precision/recall over only the titles
         the cross-encoder actually scored) and count_error (|predicted - truth|, the
         product-level metric -- it stays defined when truth is 0).
         Empty-truth queries return {"mode": "rejection", ...} instead of a chain.
         Raises ValueError if no golden entry exists for this query.
    """
    query = pipeline_output["query"]
    entry = find_golden_entry(query, dataset)
    if entry is None:
        raise ValueError(f"No golden_dataset.json entry for query {query!r}")

    truth = ground_truth_asins(entry)
    predicted = {t["asin"] for t in pipeline_output.get("titles_found", []) if t["is_match"]}
    categories = {c["category"] for c in pipeline_output.get("categories", [])}

    # Negative controls: no recall chain exists, so score the rejection instead. Absolute
    # false-positive count, not precision -- reporting a niche with 50 competitors when it
    # has none is a categorically worse failure than reporting 2, and precision scores
    # both 0.00.
    if not truth:
        return {
            "query": query,
            "approach": pipeline_output.get("approach"),
            "mode": "rejection",
            "truth_count": 0,
            "returned": len(predicted),
            "correct_rejection": len(predicted) == 0,
            "false_positives": len(predicted),
        }

    fetched = {t["asin"] for t in pipeline_output.get("titles_found", [])}
    roots = _load_catalog().reindex(list(truth))["cat"]

    s1 = {a for a, c in roots.items() if c in categories}  # ceiling set by category selection
    s2 = truth & fetched      # survived classification/retrieval into the candidate pool
    s3 = truth & predicted    # survived the reranker's score_floor

    scored = _reranked_asins(pipeline_output)
    scored_truth = truth & scored
    scored_pred = predicted & scored
    scored_tp = scored_truth & scored_pred

    return {
        "query": query,
        "approach": pipeline_output.get("approach"),
        "categories": sorted(categories),
        "truth_count": len(truth),
        "true_roots": truth_roots(truth),
        "s1_cat_selector": {"kept": len(s1), "recall": round(len(s1) / len(truth), 4)},
        "s2_candidate_pool": {
            "kept": len(s2),
            "recall": round(len(s2) / len(truth), 4),
            "pool": len(fetched),  # recall is meaningless without the pool it cost
        },
        "s3_rerank_output": {
            "kept": len(s3),
            "recall": round(len(s3) / len(truth), 4),
            "pool": len(predicted),
        },
        "rerank_isolated": {
            "scored": len(scored),
            "precision": round(len(scored_tp) / len(scored_pred), 4) if scored_pred else None,
            "recall": round(len(scored_tp) / len(scored_truth), 4) if scored_truth else None,
        },
        "count_error": abs(len(predicted) - len(truth)),
    }


def score_all(paths: list[str] | None = None) -> list[dict]:
    """
    Every saved pipeline run, scored. Runs with no golden entry are skipped rather than
    raising -- the runs directory accumulates ad-hoc CLI queries alongside golden ones.

    In: list of run JSON paths (defaults to every *.json in PIPELINE_RUNS_DIR)
    Out: list of stage_chain dicts
    """
    if paths is None:
        paths = sorted(glob.glob(str(PIPELINE_RUNS_DIR / "*.json")))
    dataset = load_golden_dataset()
    rows = []
    for p in paths:
        with open(p) as f:
            output = json.load(f)
        try:
            rows.append({**stage_chain(output, dataset), "run": Path(p).name})
        except ValueError:
            continue
    return rows


def format_table(rows: list[dict]) -> str:
    """Chain rows as a fixed-width table; rejection rows listed separately underneath."""
    chains = [r for r in rows if r.get("mode") != "rejection"]
    rejects = [r for r in rows if r.get("mode") == "rejection"]

    out = [
        f"{'run':44s} {'truth':>6} | {'S1 cat_sel':>12} {'S2 pool':>12} {'S3 rerank':>12} | "
        f"{'pool':>7} {'cnt_err':>8}",
        "-" * 112,
    ]
    for r in sorted(chains, key=lambda r: (r["approach"] or "", -r["truth_count"])):
        s1, s2, s3 = r["s1_cat_selector"], r["s2_candidate_pool"], r["s3_rerank_output"]
        out.append(
            f"{r['run'][:44]:44s} {r['truth_count']:6d} | "
            f"{s1['kept']:5d} ({s1['recall']:4.0%}) {s2['kept']:5d} ({s2['recall']:4.0%}) "
            f"{s3['kept']:5d} ({s3['recall']:4.0%}) | {s2['pool']:7d} {r['count_error']:8d}"
        )
    if rejects:
        out += ["", "negative controls (ground truth empty -- scored as rejection, never averaged in):"]
        for r in sorted(rejects, key=lambda r: r["run"]):
            mark = "OK " if r["correct_rejection"] else "FP "
            out.append(f"  {mark} {r['run'][:44]:44s} returned {r['returned']:5d}  (correct answer: 0)")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            print(json.dumps(stage_chain(json.load(f)), indent=2))
    else:
        print(format_table(score_all()))
