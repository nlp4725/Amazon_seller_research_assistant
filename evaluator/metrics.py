"""
Precision/recall/F1 for a pipeline run (main_1.py or main_2.py output) against
evaluator/golden_dataset.json's ground truth for the same query.

Ground truth for a query is the set of ASINs golden_dataset.json tagged is_match=True
in its own titles_found list (built by hand/DeepSeek-judged earlier in the project, see
golden_dataset.json's per-entry "notes"). A pipeline run's predicted set is the ASINs it
tagged is_match=True in its own titles_found list. Comparing by ASIN set overlap works
even though the two lists don't come from the same candidate pool -- main_1/main_2 do
their own fresh retrieval, so a prediction can be a true/false positive regardless of
whether golden_dataset.json's own candidate sweep happened to consider that exact ASIN.

Caveat carried through from golden_dataset.json: entries with status "verified_partial"
(currently just "female fitness clothes") have a ground truth built from a sample, not
an exhaustive enumeration -- recall against them is only a lower bound, not exact. See
each entry's "status" field before trusting recall numbers at face value.

CLI:
    python -m evaluator.metrics data/processed/pipeline_runs/simple__dog_drinking_bowl.json
"""

import json
import sys
from pathlib import Path

GOLDEN_DATASET_PATH = Path("evaluator/golden_dataset.json")


def load_golden_dataset(path: Path | str = GOLDEN_DATASET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def find_golden_entry(query: str, dataset: list[dict] | None = None) -> dict | None:
    """Case-insensitive exact match on query text."""
    if dataset is None:
        dataset = load_golden_dataset()
    for entry in dataset:
        if entry["query"].strip().lower() == query.strip().lower():
            return entry
    return None


def ground_truth_asins(entry: dict) -> set[str]:
    """ASINs golden_dataset.json confirmed as genuine matches for this query."""
    return {t["asin"] for t in entry.get("titles_found", []) if t["is_match"] and t["asin"]}


def predicted_asins(pipeline_output: dict) -> set[str]:
    """ASINs a main_1/main_2 run predicted as matches for this query."""
    return {t["asin"] for t in pipeline_output.get("titles_found", []) if t["is_match"] and t["asin"]}


def compute_metrics(predicted: set[str], truth: set[str]) -> dict:
    """
    Precision/recall/F1 by ASIN set overlap.

    Zero-denominator conventions:
      - predicted empty, truth empty  -> precision=1.0, recall=1.0 (correctly found nothing)
      - predicted empty, truth non-empty -> precision=1.0 (vacuous, no false positives), recall=0.0
      - predicted non-empty, truth empty -> precision=0.0 (every prediction is a false positive), recall=1.0 (vacuous)
    """
    tp = predicted & truth
    fp = predicted - truth
    fn = truth - predicted

    precision = len(tp) / len(predicted) if predicted else 1.0  # vacuous -- nothing predicted, so no false positives
    recall = len(tp) / len(truth) if truth else 1.0  # vacuous -- nothing to miss

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "predicted_count": len(predicted),
        "truth_count": len(truth),
    }


def score_pipeline_run(pipeline_output: dict, dataset: list[dict] | None = None) -> dict:
    """
    Score one main_1/main_2 output dict against its matching golden_dataset.json entry.

    In: pipeline_output (as returned/saved by main_1.run or main_2.run),
        optional pre-loaded golden dataset (else loaded fresh)
    Out: dict with query, approach, ground_truth_status (golden_dataset.json's "status"
         field -- flags "verified_partial" recall as a lower bound, not exact), plus
         everything from compute_metrics(); raises ValueError if no golden entry exists
         for this query
    """
    query = pipeline_output["query"]
    golden_entry = find_golden_entry(query, dataset)
    if golden_entry is None:
        raise ValueError(f"No golden_dataset.json entry for query {query!r}")

    truth = ground_truth_asins(golden_entry)
    predicted = predicted_asins(pipeline_output)
    metrics = compute_metrics(predicted, truth)

    return {
        "query": query,
        "approach": pipeline_output.get("approach"),
        "ground_truth_status": golden_entry["status"],
        **metrics,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evaluator.metrics <pipeline_run_output.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        pipeline_output = json.load(f)

    result = score_pipeline_run(pipeline_output)
    print(json.dumps(result, indent=2))
