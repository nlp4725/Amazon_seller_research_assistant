"""
Rescore every saved pipeline run against golden dataset v2 -- no API calls, no reruns.

Which saved file belongs to which system (data/processed/pipeline_runs/):
  simple__<q>     fast mode (main_1)
  structured__<q> DeepSeek structured (main_2) -- ONLY when the run made DeepSeek calls
                  (usage.calls > 0); several structured__ files were later overwritten by
                  Jev runs, which also write that name
  jev__<q>        Jev Choice structured
  jev_noul__<q>   Jev Noul structured (cutoffs 0.9 / 0.3)

Systems are compared only on queries every one of them has a run for ("common set"), so a
missing run never shifts a mean. Latency and cost do not depend on labels; they stay as
reported in the original result files.

CLI:
    python -m evaluator.rescore_v2
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluator.metrics_v2 import load_v2, predicted_asins, score, summarize
from src.shared.paths import PIPELINE_RUNS_DIR, safe_name

SYSTEMS = {"fast (main_1)": "simple", "DeepSeek structured": "structured",
           "Jev Choice": "jev", "Jev Noul": "jev_noul"}
RESULTS_DIR = Path("evaluator/results")


def load_run(prefix: str, query: str) -> dict | None:
    f = PIPELINE_RUNS_DIR / f"{prefix}__{safe_name(query)}.json"
    if not f.exists():
        return None
    run = json.loads(f.read_text())
    if prefix == "structured" and not (run.get("usage") or {}).get("calls"):
        return None  # overwritten by a Jev run -- not a DeepSeek result
    return run


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def main() -> None:
    v2 = load_v2()
    scores = {name: {} for name in SYSTEMS}
    for name, prefix in SYSTEMS.items():
        for q, entry in v2.items():
            run = load_run(prefix, entry["query"])
            if run is not None:
                scores[name][q] = score(predicted_asins(run), entry)

    groups = {
        "All four systems (common queries)": [n for n in SYSTEMS],
        "Jev Noul, every query it ran": ["Jev Noul"],
    }
    lines = [f"# Rescore against golden dataset v2 -- {datetime.now(timezone.utc):%Y-%m-%d}", "",
             "P/R/F1: macro means over complete + partial queries (partial = unbiased estimates from "
             "a random sample). Negative queries: false positives only. No reruns -- saved predictions "
             "rescored against v2 labels.", ""]
    summaries = {}
    for title, names in groups.items():
        common = sorted(set.intersection(*(set(scores[n]) for n in names)))
        lines += [f"## {title}", "", f"Queries ({len(common)}): {', '.join(v2[q]['query'] for q in common)}", "",
                  "| System | Scored | Mean P | Mean R | Mean F1 | F1 (complete only) | Negative-query FPs | Complements predicted | Substitutes predicted |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            s = summarize([scores[n][q] for q in common])
            summaries[f"{title} :: {n}"] = s
            neg = ", ".join(f"{q}: {c}" for q, c in s["negative_false_positives"].items()) or "-"
            lines.append(f"| {n} | {s['scored']} | {_fmt(s['mean_precision'])} | {_fmt(s['mean_recall'])} | "
                         f"{_fmt(s['mean_f1'])} | {_fmt(s['mean_f1_complete_only'])} | {neg} | "
                         f"{s['complements_predicted']} | {s['substitutes_predicted']} |")
        lines += ["", "| Query | Status | Truth | System | P | R | F1 | TP | FP | FN | S | C |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for q in common:
            for n in names:
                r = scores[n][q]
                lines.append(f"| {r['query']} | {r['status']} | {r['truth']} | {n} | {_fmt(r['precision'])} | "
                             f"{_fmt(r['recall'])} | {_fmt(r['f1'])} | {r['tp']} | {r['fp']} | {r['fn']} | "
                             f"{r['substitutes']} | {r['complements']} |")
        lines.append("")
    unjudged = sum(r["unjudged"] for n in scores for r in scores[n].values() if r["status"] != "partial")
    lines.append(f"Unjudged predictions on complete/negative queries: {unjudged} (should be ~0 -- every saved "
                 "run's predictions were pooled into v2). Predictions on partial queries that fall outside "
                 "the labeled sample are expected and excluded from precision by design.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = RESULTS_DIR / f"rescore_v2_{stamp}.md"
    md.write_text("\n".join(lines))
    (RESULTS_DIR / f"rescore_v2_{stamp}.json").write_text(json.dumps(
        {"summaries": summaries, "scores": scores}, indent=2, ensure_ascii=False))
    print(md.read_text())
    print(f"\nWrote {md}")


if __name__ == "__main__":
    main()
