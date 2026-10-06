"""
Run main_1.py (simple: rank+rerank) and main_2.py (structured: classify+rerank-residual)
against every verified query in evaluator/golden_dataset.json, score each with
evaluator/metrics.py, and produce a side-by-side precision/recall/f1/latency/cost
comparison. Both mains and metrics.py are reused as-is -- this module just orchestrates
and aggregates, so any other script/agent can call run_comparison() directly instead of
going through the CLI.

Results are saved to evaluator/results/ as both a full JSON (every per-query row) and a
markdown summary table (per-approach aggregates), timestamped so past runs aren't
overwritten.

Latency and cost per query come from LangSmith's trace record (evaluator/langsmith_timing.py),
not from each main's own self-measured time.perf_counter() timestamps or in-process usage
accumulator -- LangSmith is an independent source that reports every LLM call and every
@traceable function's actual timing/tokens regardless of how it was invoked. Each main's
own self-measured latency_s/usage dict is kept alongside as self_measured_* fields for
cross-checking, since fetching the LangSmith trace requires a short poll (ingestion is
asynchronous -- see langsmith_timing.fetch_trace_runs) that adds real wall-clock time to
each run.

CLI:
    python -m evaluator.comparison
    python -m evaluator.comparison "dog drinking bowl" "kids costumes"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluator.langsmith_timing import summarize_trace
from evaluator.metrics import load_golden_dataset, score_pipeline_run
from src.retrieval_pipeline import cat_selector, main_1, main_2

RESULTS_DIR = Path("evaluator/results")
APPROACHES = [("simple", main_1), ("structured", main_2)]


def run_comparison(queries: list[str] | None = None, dataset: list[dict] | None = None) -> list[dict]:
    """
    Run both approaches on each query and score against golden_dataset.json.

    In: queries (defaults to every entry in golden_dataset.json with status starting
        "verified"), optional pre-loaded dataset
    Out: list of per-(query, approach) result dicts: metrics (precision/recall/f1/...)
         plus latency_s/cost_usd (LangSmith-sourced, see module docstring),
         langsmith_steps (per-step breakdown), self_measured_latency_s/cost_usd
         (each main's own timers, for cross-check), llm_calls, ground_truth_status
    """
    if dataset is None:
        dataset = load_golden_dataset()
    if queries is None:
        queries = [e["query"] for e in dataset if e["status"].startswith("verified")]

    rows = []
    for query in queries:
        print(f"\n=== {query!r} ===")

        # Resolve categories ONCE per query and share the identical list with both
        # approaches -- guarantees an apples-to-apples comparison instead of each main
        # independently picking its own (possibly different) categories.
        resolved = cat_selector.select([{"role": "user", "content": query}])
        if "clarify" in resolved:
            print(f"  cat_selector wants clarification, skipping: {resolved['clarify']!r}")
            continue
        categories = resolved["categories"]
        print(f"  categories: {[c['category'] for c in categories]}")

        for approach_name, module in APPROACHES:
            print(f"  running {approach_name}...")
            output = module.run(query, categories=categories)
            metrics = score_pipeline_run(output, dataset)

            print("    fetching LangSmith trace...")
            trace = summarize_trace(output["langsmith_trace_id"]) if output.get("langsmith_trace_id") else {
                "root": None, "steps": {}, "total_cost_usd": 0.0, "warning": "no trace_id captured",
            }

            row = {
                **metrics,
                "latency_s": (trace["root"] or {}).get("latency_s") or output["latency_s"]["total"],
                "cost_usd": trace["total_cost_usd"],
                "langsmith_steps": trace["steps"],
                "langsmith_warning": trace.get("warning"),
                "self_measured_latency_s": output["latency_s"]["total"],
                "self_measured_cost_usd": output["usage"]["estimated_cost_usd"],
                "llm_calls": output["usage"]["calls"],
            }
            rows.append(row)
            print(f"    P={row['precision']:.2f}  R={row['recall']:.2f}  F1={row['f1']:.2f}  "
                  f"latency={row['latency_s']}s  cost=${row['cost_usd']}  calls={row['llm_calls']}")

    return rows


def summarize(rows: list[dict]) -> dict:
    """
    Per-approach aggregates: mean precision/recall/f1 (query-level average, not
    micro-averaged over pooled ASINs), total latency, total cost, total LLM calls.

    Queries whose ground truth is empty are left out of the P/R/F1 means: their recall is
    vacuous (nothing to find) and their precision is 0 the moment anything is predicted,
    so averaging them in measures nothing about retrieval quality. They are reported as
    zero_truth_false_positives instead -- items predicted where none should be (0 = right).
    Latency, cost and LLM calls still cover every query, since that work was done.
    """
    summary = {}
    for approach_name, _ in APPROACHES:
        approach_rows = [r for r in rows if r["approach"] == approach_name]
        if not approach_rows:
            continue
        n = len(approach_rows)
        scored = [r for r in approach_rows if r["truth_count"] > 0]
        k = len(scored) or 1
        summary[approach_name] = {
            "queries": n,
            "scored_queries": len(scored),
            "mean_precision": round(sum(r["precision"] for r in scored) / k, 4),
            "mean_recall": round(sum(r["recall"] for r in scored) / k, 4),
            "mean_f1": round(sum(r["f1"] for r in scored) / k, 4),
            "zero_truth_false_positives": {
                r["query"]: r["false_positives"] for r in approach_rows if r["truth_count"] == 0
            },
            "total_latency_s": round(sum(r["latency_s"] for r in approach_rows), 2),
            "mean_latency_s": round(sum(r["latency_s"] for r in approach_rows) / n, 2),
            "total_cost_usd": round(sum(r["cost_usd"] for r in approach_rows), 6),
            "total_llm_calls": sum(r["llm_calls"] for r in approach_rows),
        }
    return summary


def _zero_truth(s: dict) -> str:
    """Zero-truth queries and the items wrongly predicted for each, e.g. "dog Halloween costume: 2"."""
    fps = s.get("zero_truth_false_positives") or {}
    return ", ".join(f"{q}: {n}" for q, n in fps.items()) or "-"


def _markdown_report(rows: list[dict], summary: dict, timestamp: str) -> str:
    lines = [f"# Retrieval pipeline comparison -- {timestamp}", ""]

    lines.append("## Summary (P/R/F1: mean across queries with non-empty ground truth)")
    lines.append("")
    lines.append("| Approach | Queries (scored) | Mean P | Mean R | Mean F1 | Zero-truth FPs | Total latency (s) | Mean latency (s) | Total cost ($) | LLM calls |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for approach_name, s in summary.items():
        lines.append(
            f"| {approach_name} | {s['queries']} ({s['scored_queries']}) | {s['mean_precision']} | "
            f"{s['mean_recall']} | {s['mean_f1']} | {_zero_truth(s)} | {s['total_latency_s']} | "
            f"{s['mean_latency_s']} | {s['total_cost_usd']} | {s['total_llm_calls']} |"
        )

    lines.append("")
    lines.append("## Per-query detail")
    lines.append("")
    lines.append("| Query | Approach | Truth status | P | R | F1 | TP | FP | FN | Latency (s) | Cost ($) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['query']} | {r['approach']} | {r['ground_truth_status']} | {r['precision']} | "
            f"{r['recall']} | {r['f1']} | {r['true_positives']} | {r['false_positives']} | "
            f"{r['false_negatives']} | {r['latency_s']} | {r['cost_usd']} |"
        )

    lines.append("")
    lines.append("## Per-step breakdown (LangSmith trace, per query/approach)")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['query']} -- {r['approach']}")
        if r.get("langsmith_warning"):
            lines.append(f"*{r['langsmith_warning']}*")
        lines.append("")
        lines.append("| Step | Calls | Latency (s) | Prompt tok | Completion tok | Cost ($) |")
        lines.append("|---|---|---|---|---|---|")
        for name, s in r["langsmith_steps"].items():
            lines.append(
                f"| {name} | {s['calls']} | {s['latency_s']} | {s['prompt_tokens']} | "
                f"{s['completion_tokens']} | {s['cost_usd']} |"
            )
        lines.append("")

    lines.append(
        "Note: queries with `ground_truth_status: verified_partial` (currently \"female "
        "fitness clothes\") have ground truth built from a sample, not an exhaustive "
        "enumeration -- their recall is a lower bound, not exact."
    )
    lines.append("")
    lines.append(
        "Note: latency/cost above are read from LangSmith's trace record, not each "
        "main's own self-measured timers (see evaluator/langsmith_timing.py) -- "
        "self_measured_latency_s/self_measured_cost_usd in the saved JSON are each "
        "main's own numbers, kept for cross-check."
    )

    return "\n".join(lines)


def save_results(rows: list[dict], summary: dict) -> Path:
    """Save the full JSON + a markdown report to evaluator/results/, timestamped."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = RESULTS_DIR / f"comparison_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"timestamp": timestamp, "summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)

    md_path = RESULTS_DIR / f"comparison_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(_markdown_report(rows, summary, timestamp))

    latest_json = RESULTS_DIR / "comparison_latest.json"
    with open(latest_json, "w") as f:
        json.dump({"timestamp": timestamp, "summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return md_path


if __name__ == "__main__":
    queries = sys.argv[1:] or None
    rows = run_comparison(queries)
    summary = summarize(rows)

    print("\n=== Summary ===")
    for approach_name, s in summary.items():
        print(f"{approach_name}: P={s['mean_precision']} R={s['mean_recall']} F1={s['mean_f1']} "
              f"total_latency={s['total_latency_s']}s total_cost=${s['total_cost_usd']}")

    save_results(rows, summary)
