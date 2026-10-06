"""
Jev vs. DeepSeek on the six August queries -- the structured approach (main_2) only.

Runs main_2 with CLASSIFIER_BACKEND=jev on each query, reusing the categories the August
run resolved (data/processed/pipeline_runs/structured__<query>.json), so no DeepSeek call
is made: no cat_selector, no fast-mode rerun. The DeepSeek side is the August report
(comparison_20260813T185733Z.json), not a fresh run, so the two arms differ in code
version as well as classifier.

Scored exactly as comparison.py scores: metrics.score_pipeline_run per query (same
zero-denominator conventions), comparison.summarize for the means, and latency/cost
from LangSmith's trace via langsmith_timing.summarize_trace.

main_2 overwrites structured__<query>.json and classify_agent overwrites the category
classification files, so both are copied aside first and restored afterwards; Jev's own
outputs are kept as jev__<query>.json.

CLI:
    python -m evaluator.jev_six                         # run (JEV_MODE=choice|noul)
    python -m evaluator.jev_six --rebuild <saved.json>  # re-score a saved run, no API calls
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["CLASSIFIER_BACKEND"] = "jev"

from evaluator.comparison import _zero_truth, summarize
from evaluator.langsmith_timing import summarize_trace
from evaluator.metrics import load_golden_dataset, score_pipeline_run
from src.retrieval_pipeline import jev_scorer, main_2
from src.shared.paths import (
    CATEGORY_CLASSIFY_DIR,
    PIPELINE_RUNS_DIR,
    safe_name,
)

QUERIES = ["dog drinking bowl", "ice tray", "car phone mount",
           "summer dress", "kids costumes", "dog Halloween costume"]
BASELINE = Path("evaluator/results/comparison_20260813T185733Z.json")
RESULTS_DIR = Path("evaluator/results")


def _label() -> str:
    if jev_scorer.JEV_MODE == "noul":
        return f"noul, cutoffs {jev_scorer.JEV_CONFIDENT_CUTOFF}/{jev_scorer.JEV_KEEP_CUTOFF}"
    return "choice"


def _report(rows: list[dict], base_rows: list[dict], jev: dict, deepseek: dict, stamp: str) -> str:
    lines = [f"# Jev ({_label()}) vs. DeepSeek, structured approach, six August queries -- {stamp}", "",
             (f"Jev model `{jev_scorer.JEV_MODEL}`. DeepSeek numbers are from `{BASELINE.name}` "
              "(an earlier code version), not a fresh run."), "",
             "P/R/F1 are means over queries with non-empty ground truth; zero-truth queries are "
             "scored by false positives instead (0 = correctly found nothing).", "",
             "| Classifier | Queries (scored) | Mean P | Mean R | Mean F1 | Zero-truth FPs | "
             "Total latency (s) | Mean latency (s) | Total cost ($) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("DeepSeek (Aug)", deepseek), (f"Jev {_label()}", jev)):
        lines.append(f"| {name} | {s['queries']} ({s['scored_queries']}) | {s['mean_precision']} | "
                     f"{s['mean_recall']} | {s['mean_f1']} | {_zero_truth(s)} | {s['total_latency_s']} | "
                     f"{s['mean_latency_s']} | {s['total_cost_usd']} |")
    lines += ["", "| Query | Classifier | P | R | F1 | TP | FP | FN | Latency (s) | Cost ($) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for base, row in zip(base_rows, rows):
        for name, r in (("DeepSeek (Aug)", base), (f"Jev {_label()}", row)):
            lines.append(f"| {r['query']} | {name} | {r['precision']} | {r['recall']} | {r['f1']} | "
                         f"{r['true_positives']} | {r['false_positives']} | {r['false_negatives']} | "
                         f"{r['latency_s']} | {r['cost_usd']} |")
    lines += ["", "Latency and cost are read from LangSmith's trace (evaluator/langsmith_timing.py)."]
    return "\n".join(lines)


def main() -> None:
    dataset = load_golden_dataset()
    backup = Path(tempfile.mkdtemp(prefix="jev_six_backup_"))
    for src_dir in (PIPELINE_RUNS_DIR, CATEGORY_CLASSIFY_DIR):
        shutil.copytree(src_dir, backup / src_dir.name)
    print(f"Backed up pipeline runs and classifications to {backup}")

    rows = []
    try:
        for query in QUERIES:
            saved = PIPELINE_RUNS_DIR / f"structured__{safe_name(query)}.json"
            categories = json.loads((backup / PIPELINE_RUNS_DIR.name / saved.name).read_text())["categories"]
            print(f"\n{query} -- categories {[c['category'] for c in categories]}")
            output = main_2.run(query, categories=categories)
            shutil.copy(saved, PIPELINE_RUNS_DIR / f"jev_{jev_scorer.JEV_MODE}__{safe_name(query)}.json")
            for c in categories:
                cls = CATEGORY_CLASSIFY_DIR / f"{safe_name(c['category'])}__{safe_name(query)}.json"
                shutil.copy(cls, CATEGORY_CLASSIFY_DIR / f"jev_{jev_scorer.JEV_MODE}__{cls.name}")

            metrics = score_pipeline_run(output, dataset)
            print("    fetching LangSmith trace...")
            trace = summarize_trace(output["langsmith_trace_id"]) if output.get("langsmith_trace_id") else {
                "root": None, "steps": {}, "total_cost_usd": 0.0, "warning": "no trace_id captured",
            }
            row = {
                **metrics,
                "approach": "structured",
                "latency_s": (trace["root"] or {}).get("latency_s") or output["latency_s"]["total"],
                "cost_usd": trace["total_cost_usd"],
                "langsmith_steps": trace["steps"],
                "langsmith_warning": trace.get("warning"),
                "self_measured_latency_s": output["latency_s"]["total"],
                "llm_calls": trace["steps"].get("jev.system_one", {}).get("calls", 0),
            }
            rows.append(row)
            print(f"    P={row['precision']:.2f}  R={row['recall']:.2f}  F1={row['f1']:.2f}  "
                  f"latency={row['latency_s']}s  cost=${row['cost_usd']}  jev calls={row['llm_calls']}")
    finally:
        for src_dir in (PIPELINE_RUNS_DIR, CATEGORY_CLASSIFY_DIR):
            for f in (backup / src_dir.name).iterdir():
                shutil.copy(f, src_dir / f.name)
        print(f"\nRestored the August pipeline runs and classifications from {backup}")

    base = {r["query"]: r for r in json.loads(BASELINE.read_text())["rows"] if r["approach"] == "structured"}
    base_rows = [base[r["query"]] for r in rows]
    jev = summarize(rows)["structured"]
    deepseek = summarize(base_rows)["structured"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"jev_six_{jev_scorer.JEV_MODE}_{stamp}.json"
    md_path = RESULTS_DIR / f"jev_six_{jev_scorer.JEV_MODE}_{stamp}.md"
    json_path.write_text(json.dumps({"timestamp": stamp, "jev_model": jev_scorer.JEV_MODEL,
                                     "jev_mode": _label(),
                                     "baseline": BASELINE.name, "summary": {"jev": jev, "deepseek_aug": deepseek},
                                     "rows": rows}, indent=2, ensure_ascii=False))
    md_path.write_text(_report(rows, base_rows, jev, deepseek, stamp))
    print(md_path.read_text())
    print(f"\nWrote {json_path}\nWrote {md_path}")


def rebuild_report(json_path: Path) -> None:
    """Re-summarize a saved run with the current scoring rules and rewrite its .md -- no API calls."""
    saved = json.loads(json_path.read_text())
    rows = saved["rows"]
    base = {r["query"]: r for r in json.loads(BASELINE.read_text())["rows"] if r["approach"] == "structured"}
    base_rows = [base[r["query"]] for r in rows]
    jev, deepseek = summarize(rows)["structured"], summarize(base_rows)["structured"]
    saved["summary"] = {"jev": jev, "deepseek_aug": deepseek}
    json_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    label = saved.get("jev_mode", "choice")
    md = _report(rows, base_rows, jev, deepseek, saved["timestamp"])
    md_path = json_path.with_suffix(".md")
    md_path.write_text(md.replace(f"Jev {_label()}", f"Jev {label}").replace(f"Jev ({_label()})", f"Jev ({label})"))
    print(md_path.read_text())


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--rebuild":
        rebuild_report(Path(sys.argv[2]))
    else:
        main()
