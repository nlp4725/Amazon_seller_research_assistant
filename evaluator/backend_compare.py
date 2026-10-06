"""
DeepSeek vs. Jev (noul) as main_2's path classifier, on the golden queries the six-query
August comparison did not use.

Per query: cat_selector.select() resolves the categories ONCE (one DeepSeek call, not
counted against either classifier), then main_2 runs twice on that identical input --
CLASSIFIER_BACKEND=jev with JEV_MODE=noul, then CLASSIFIER_BACKEND=deepseek -- on the same
code, same day. Scoring is comparison.py's: metrics.score_pipeline_run per run,
comparison.summarize for the means (zero-truth queries excluded from P/R/F1 and reported
as false positives), latency and cost from LangSmith via langsmith_timing.summarize_trace.

Each run's output is kept as {backend}__<query>.json in pipeline_runs (main_2's own
structured__<query>.json is overwritten by the next run), and results are written after
every query so a crash mid-run keeps what finished.

CLI:
    python -m evaluator.backend_compare            # every golden query not in SKIP
    python -m evaluator.backend_compare "yoga mat"  # just these
    BACKENDS=jev_noul python -m evaluator.backend_compare   # Jev only
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["JEV_MODE"] = "noul"

from evaluator.comparison import _zero_truth, summarize
from evaluator.langsmith_timing import summarize_trace
from evaluator.metrics import load_golden_dataset, score_pipeline_run
from src.retrieval_pipeline import cat_selector, jev_scorer, main_2
from src.shared.paths import CATEGORY_CLASSIFY_DIR, PIPELINE_RUNS_DIR, safe_name

SKIP = {"dog drinking bowl", "ice tray", "car phone mount",
        "summer dress", "kids costumes", "dog Halloween costume"}
# BACKENDS=jev_noul to run Jev alone (DeepSeek's classifier is ~100x slower).
BACKENDS = os.environ.get("BACKENDS", "jev_noul,deepseek").split(",")
RESULTS_DIR = Path("evaluator/results")


def _run(query: str, categories: list[dict], backend: str, dataset: list[dict]) -> dict:
    os.environ["CLASSIFIER_BACKEND"] = "jev" if backend.startswith("jev") else "deepseek"
    output = main_2.run(query, categories=categories)
    shutil.copy(PIPELINE_RUNS_DIR / f"structured__{safe_name(query)}.json",
                PIPELINE_RUNS_DIR / f"{backend}__{safe_name(query)}.json")
    for c in categories:
        cls = CATEGORY_CLASSIFY_DIR / f"{safe_name(c['category'])}__{safe_name(query)}.json"
        if cls.exists():
            shutil.copy(cls, CATEGORY_CLASSIFY_DIR / f"{backend}__{cls.name}")

    metrics = score_pipeline_run(output, dataset)
    trace = summarize_trace(output["langsmith_trace_id"]) if output.get("langsmith_trace_id") else {
        "root": None, "steps": {}, "total_cost_usd": 0.0, "warning": "no trace_id captured",
    }
    llm_steps = ("jev.system_one",) if backend.startswith("jev") else ("ChatOpenAI",)
    return {
        **metrics,
        "approach": "structured",
        "backend": backend,
        "categories": [c["category"] for c in categories],
        "latency_s": (trace["root"] or {}).get("latency_s") or output["latency_s"]["total"],
        "cost_usd": trace["total_cost_usd"],
        "langsmith_steps": trace["steps"],
        "langsmith_warning": trace.get("warning"),
        "self_measured_latency_s": output["latency_s"]["total"],
        "llm_calls": sum(trace["steps"].get(s, {}).get("calls", 0) for s in llm_steps),
    }


def _report(rows: list[dict], summaries: dict, stamp: str) -> str:
    lines = [f"# DeepSeek vs. Jev (noul) as the path classifier -- {stamp}", "",
             f"Structured approach (main_2), same code and categories for both. Jev `{jev_scorer.JEV_MODEL}`, "
             f"cutoffs {jev_scorer.JEV_CONFIDENT_CUTOFF}/{jev_scorer.JEV_KEEP_CUTOFF}. "
             "P/R/F1 are means over queries with non-empty ground truth; zero-truth queries are scored "
             "by false positives. Latency and cost come from LangSmith; cat_selector's category call is "
             "shared and not counted.", "",
             "| Classifier | Queries (scored) | Mean P | Mean R | Mean F1 | Zero-truth FPs | "
             "Total latency (s) | Mean latency (s) | Total cost ($) | LLM calls |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for backend, s in summaries.items():
        lines.append(f"| {backend} | {s['queries']} ({s['scored_queries']}) | {s['mean_precision']} | "
                     f"{s['mean_recall']} | {s['mean_f1']} | {_zero_truth(s)} | {s['total_latency_s']} | "
                     f"{s['mean_latency_s']} | {s['total_cost_usd']} | {s['total_llm_calls']} |")
    lines += ["", "| Query | Truth | Classifier | P | R | F1 | TP | FP | FN | Latency (s) | Cost ($) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['query']} | {r['truth_count']} | {r['backend']} | {r['precision']} | {r['recall']} | "
                     f"{r['f1']} | {r['true_positives']} | {r['false_positives']} | {r['false_negatives']} | "
                     f"{r['latency_s']} | {r['cost_usd']} |")
    lines += ["", "Note: \"female fitness clothes\" has partial ground truth (verified_partial), so its "
              "recall is a lower bound."]
    return "\n".join(lines)


def main(queries: list[str] | None = None) -> None:
    dataset = load_golden_dataset()
    if not queries:
        queries = [e["query"] for e in dataset if e["query"] not in SKIP]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"backend_compare_{stamp}.json"
    md_path = RESULTS_DIR / f"backend_compare_{stamp}.md"

    rows, skipped = [], {}
    for query in queries:
        print(f"\n=== {query!r} ===", flush=True)
        resolved = cat_selector.select([{"role": "user", "content": query}])
        if "clarify" in resolved:
            print(f"  cat_selector wants clarification, skipping: {resolved['clarify']!r}", flush=True)
            skipped[query] = resolved["clarify"]
            continue
        categories = resolved["categories"]
        print(f"  categories: {[c['category'] for c in categories]}", flush=True)

        for backend in BACKENDS:
            print(f"  running {backend}...", flush=True)
            try:
                row = _run(query, categories, backend, dataset)
            except Exception as e:  # one failed run should not lose the others
                print(f"  {backend} FAILED: {e!r}", flush=True)
                skipped[f"{query} [{backend}]"] = repr(e)
                continue
            rows.append(row)
            print(f"    P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f} "
                  f"latency={row['latency_s']}s cost=${row['cost_usd']} calls={row['llm_calls']}", flush=True)

        summaries = {b: summarize([r for r in rows if r["backend"] == b]).get("structured") for b in BACKENDS}
        summaries = {b: s for b, s in summaries.items() if s}
        json_path.write_text(json.dumps({"timestamp": stamp, "jev_model": jev_scorer.JEV_MODEL,
                                         "jev_cutoffs": [jev_scorer.JEV_CONFIDENT_CUTOFF, jev_scorer.JEV_KEEP_CUTOFF],
                                         "summary": summaries, "skipped": skipped, "rows": rows},
                                        indent=2, ensure_ascii=False))
        md_path.write_text(_report(rows, summaries, stamp))

    print("\n" + md_path.read_text())
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
