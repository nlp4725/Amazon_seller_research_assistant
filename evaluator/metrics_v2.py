"""
Scoring against golden dataset v2 (evaluator/golden_dataset_v2.json, method in
evaluator/golden_v2/README.md).

v2 entries carry a status, and each status is scored the only way that is meaningful for it:

  complete -- every pooled candidate was judged:
              precision, recall, F1 over the predicted set.
  partial  -- only a fixed random sample of the pool was judged (summer dress, female
              fitness clothes): precision over predicted items that fall inside the judged
              sample; recall as the share of the sample's true items that were predicted.
              Because the sample is random, both are unbiased estimates, not exact values.
  negative -- no exact match exists in the pool: scored by false positives only (anything
              predicted is wrong); P/R/F1 are None and never averaged.

Every score also counts how many predicted items v2 labels as substitutes (S) and
complements (C) -- e.g. mouse chargers predicted for "wireless gaming mouse" -- and how many
predictions were never judged at all (should be 0 for runs pooled into v2).
"""

import json
from pathlib import Path

GOLDEN_V2_PATH = Path("evaluator/golden_dataset_v2.json")


def load_v2(path: Path | str = GOLDEN_V2_PATH) -> dict[str, dict]:
    return {e["query"].lower(): e for e in json.loads(Path(path).read_text())["queries"]}


def predicted_asins(run: dict) -> set[str]:
    return {t["asin"] for t in run.get("titles_found", []) if t.get("is_match") and t.get("asin")}


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def score(predicted: set[str], entry: dict) -> dict:
    labels = {t["asin"]: t["label"] for t in entry["titles_found"]}
    truth = {a for a, lab in labels.items() if lab == "E"}
    judged_pred = predicted & labels.keys()
    tp = predicted & truth
    out = {
        "query": entry["query"], "status": entry["status"],
        "predicted": len(predicted), "truth": len(truth),
        "tp": len(tp), "fp": len(judged_pred - truth), "fn": len(truth - predicted),
        "substitutes": sum(labels[a] == "S" for a in judged_pred),
        "complements": sum(labels[a] == "C" for a in judged_pred),
        "unjudged": len(predicted - labels.keys()),
        "precision": None, "recall": None, "f1": None,
    }
    if entry["status"] == "negative":
        out["fp"] = len(predicted)
        return out
    base = judged_pred if entry["status"] == "partial" else predicted
    p = round(len(tp) / len(base), 4) if base else None
    r = round(len(tp) / len(truth), 4) if truth else None
    out.update(precision=p, recall=r, f1=_f1(p, r))
    return out


def summarize(scores: list[dict]) -> dict:
    """Macro means over scored (complete + partial) queries; negatives as false-positive counts."""
    scored = [s for s in scores if s["status"] != "negative" and s["precision"] is not None]
    mean = lambda k: round(sum(s[k] for s in scored) / len(scored), 4) if scored else None  # noqa: E731
    complete = [s for s in scored if s["status"] == "complete"]
    return {
        "queries": len(scores), "scored": len(scored),
        "mean_precision": mean("precision"), "mean_recall": mean("recall"), "mean_f1": mean("f1"),
        "mean_f1_complete_only": (round(sum(s["f1"] for s in complete) / len(complete), 4)
                                  if complete else None),
        "negative_false_positives": {s["query"]: s["fp"] for s in scores if s["status"] == "negative"},
        "complements_predicted": sum(s["complements"] for s in scores),
        "substitutes_predicted": sum(s["substitutes"] for s in scores),
    }
