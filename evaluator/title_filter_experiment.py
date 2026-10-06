"""
Experiment: a Jev title-level precision filter on top of main_2 + Jev Noul.

Rescored against golden v2, main_2's false positives are mostly substitutes and complements
that sit inside kept category paths (12 flat scratchers for "cat scratching post", balloon
accessories, non-Halloween dog costumes). Path classification cannot remove them -- it never
reads titles -- and the cross-encoder judges text similarity, not "is this the product".

This asks Jev one yes/no question per PREDICTED title: is this product itself what the
query asks for? It uses the query plus GENERIC rules only -- never the golden set's
per-query definitions, which a real system would not have. Items with p >= KEEP_CUTOFF stay.
The cutoff is fixed at 0.5 before looking at results; 0.3 and 0.7 are reported for context.

Input: the saved jev_noul__<query>.json runs (no pipeline rerun). Output: v2 scores before vs.
after, per cutoff, with filter latency and cost read from LangSmith.

CLI:
    python -m evaluator.title_filter_experiment
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from typesafe_sdk import AsyncTypeSafeClient, Noul

load_dotenv(".env")

from evaluator.langsmith_timing import summarize_trace  # noqa: E402
from evaluator.metrics_v2 import (  # noqa: E402
    load_v2,
    predicted_asins,
    score,
    summarize,
)
from evaluator.rescore_v2 import load_run  # noqa: E402
from src.retrieval_pipeline import title_filter  # noqa: E402
from src.retrieval_pipeline.jev_scorer import JEV_MODEL  # noqa: E402

KEEP_CUTOFF = 0.5
REPORTED_CUTOFFS = [0.3, 0.5, 0.7]
BATCH = 250
RESULTS_DIR = Path("evaluator/results")

# Same rules as the pipeline stage, so the experiment measures what main_2 runs.
GENERIC_RULES = title_filter.GENERIC_RULES


@traceable(run_type="llm", name="jev.system_one", metadata={"ls_provider": "typesafe", "ls_model_name": JEV_MODEL})
async def _ask(client: AsyncTypeSafeClient, query: str, titles: dict[str, str]) -> dict:
    response = await client.system_one(
        state={"query": query, "rules": GENERIC_RULES},
        questions={asin: Noul(instructions=(f'Product title: "{title}". Is this product itself what a shopper '
                                            "searching `query` wants, following `rules`?"))
                   for asin, title in titles.items()},
    )
    u = response.usage
    return {"p": {a: ans.noul for a, ans in response.answers.items()},
            "usage_metadata": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                               "total_tokens": u.input_tokens + u.output_tokens}}


@traceable(run_type="chain", name="title_filter.run")
async def filter_query(query: str, titles: dict[str, str]) -> dict:
    trace_id = str(get_current_run_tree().trace_id)
    items = list(titles.items())
    async with AsyncTypeSafeClient(model=JEV_MODEL) as client:
        parts = await asyncio.gather(*(_ask(client, query, dict(items[i:i + BATCH]))
                                       for i in range(0, len(items), BATCH)))
    p = {a: v for part in parts for a, v in part["p"].items()}
    return {"p": p, "trace_id": trace_id}


def main() -> None:
    v2 = load_v2()
    rows = []
    for entry in v2.values():
        run = load_run("jev_noul", entry["query"])
        if run is None:
            continue
        pred = predicted_asins(run)
        titles = {t["asin"]: t["title"] for t in run["titles_found"] if t["asin"] in pred}
        before = score(pred, entry)
        if titles:
            out = asyncio.run(filter_query(entry["query"], titles))
            trace = summarize_trace(out["trace_id"])
            p = out["p"]
            latency, cost = (trace["root"] or {}).get("latency_s"), trace["total_cost_usd"]
        else:
            p, latency, cost = {}, 0.0, 0.0
        after = {c: score({a for a in pred if p.get(a, 0) >= c}, entry) for c in REPORTED_CUTOFFS}
        rows.append({"query": entry["query"], "status": entry["status"], "before": before,
                     "after": after, "filter_latency_s": latency, "filter_cost_usd": cost,
                     "p": p})
        a = after[KEEP_CUTOFF]
        print(f"{entry['query']:27s} P {before['precision']} -> {a['precision']}   "
              f"R {before['recall']} -> {a['recall']}   kept {a['predicted']}/{before['predicted']}   "
              f"{latency}s ${cost}", flush=True)

    lines = [f"# Jev title filter on main_2 + Jev Noul -- {datetime.now(timezone.utc):%Y-%m-%d}", "",
             f"Generic rules only (no golden definitions). Headline cutoff p >= {KEEP_CUTOFF}, fixed before "
             "running. Scored against golden v2. Filter latency/cost from LangSmith.", "",
             "| Setting | Mean P | Mean R | Mean F1 | Negative-query FPs | Complements | Substitutes |",
             "|---|---|---|---|---|---|---|"]
    variants = [("main_2 + Jev Noul (before)", [r["before"] for r in rows])] + \
               [(f"+ title filter p >= {c}", [r["after"][c] for r in rows]) for c in REPORTED_CUTOFFS]
    for name, scores in variants:
        s = summarize(scores)
        neg = ", ".join(f"{q}: {n}" for q, n in s["negative_false_positives"].items())
        lines.append(f"| {name} | {s['mean_precision']} | {s['mean_recall']} | {s['mean_f1']} | {neg} | "
                     f"{s['complements_predicted']} | {s['substitutes_predicted']} |")
    lat = [r["filter_latency_s"] or 0 for r in rows]
    lines += ["", f"Filter latency: mean {sum(lat)/len(lat):.2f}s, max {max(lat):.2f}s per query; "
              f"total filter cost ${sum(r['filter_cost_usd'] for r in rows):.4f}.", "",
              "| Query | Status | P before | P after | R before | R after | F1 before | F1 after | Kept |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b, a = r["before"], r["after"][KEEP_CUTOFF]
        lines.append(f"| {r['query']} | {r['status']} | {b['precision']} | {a['precision']} | {b['recall']} | "
                     f"{a['recall']} | {b['f1']} | {a['f1']} | {a['predicted']}/{b['predicted']} |")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = RESULTS_DIR / f"title_filter_{stamp}.md"
    md.write_text("\n".join(lines))
    (RESULTS_DIR / f"title_filter_{stamp}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print("\n" + md.read_text() + f"\n\nWrote {md}")


if __name__ == "__main__":
    main()
