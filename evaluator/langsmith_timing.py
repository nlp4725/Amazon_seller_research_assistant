"""
Per-step latency and token-cost breakdown for one main_1/main_2 run, read directly from
LangSmith's trace record -- not from each agent's own time.perf_counter() timestamps or
llm_client's own usage accumulator. LangSmith is the independent source of truth: every
LLM call (via llm_client's wrap_openai-wrapped client) and every @traceable function in
this pipeline reports its own start/end time and token usage to LangSmith regardless of
how it was invoked, including from inside classify_agent's ThreadPoolExecutor batches
(see classify_agent.py's contextvars.copy_context() fix -- without it, those batch calls
show up as orphaned root traces instead of nesting under their parent run).

LangSmith does not have a registered price-per-token for a custom model like
"deepseek-v4-flash" -- confirmed empirically: every run's total_cost/prompt_cost/
completion_cost field comes back None. Cost here is computed by applying
llm_client.PRICE_PER_TOKEN to the token counts LangSmith DOES report (prompt_tokens,
completion_tokens), so the $ conversion is still our own price table, but the token
counts feeding it are LangSmith's authoritative trace record, not our own in-process
accumulator -- a real independent check on llm_client.get_usage(), not a duplicate of it.

The cache-hit/cache-miss split (DeepSeek prices a cache hit ~50x cheaper -- see
llm_client.PRICE_PER_TOKEN) isn't in the run's top-level prompt_tokens field, but it IS
preserved at run.extra["metadata"]["usage_metadata"]["input_token_details"]["cache_read"]
-- confirmed empirically by comparing against llm_client's own cache-aware accounting for
the same call, which otherwise disagreed with a naive all-cache-miss estimate by ~2x on a
repeat call. Falls back to treating all prompt tokens as cache-miss if that field is
absent (e.g. a model/provider that doesn't report prompt caching at all).

Ingestion into LangSmith is asynchronous (client-side batched upload), so a trace may
not be fully queryable via the API for a few seconds after the traced call returns --
fetch_trace_runs() polls with a timeout instead of assuming the trace is immediately
complete.

CLI:
    python -m evaluator.langsmith_timing <trace_id>
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from langsmith import Client

from src.retrieval_pipeline.llm_client import PRICE_PER_TOKEN

load_dotenv()

POLL_TIMEOUT_S = 30
POLL_INTERVAL_S = 2


def fetch_trace_runs(trace_id: str, project: str | None = None, timeout_s: float = POLL_TIMEOUT_S) -> list:
    """
    Every run belonging to one trace, polling until the root run shows an end_time (i.e.
    fully ingested) or the timeout elapses.

    In: trace_id (the root run's own id, e.g. main_1.run's run id -- see
        "langsmith_trace_id" in main_1/main_2's output dict), optional project name
        (defaults to LANGSMITH_PROJECT env var), timeout_s to wait for ingestion
    Out: list of langsmith Run objects belonging to this trace (possibly incomplete if
         the timeout was hit before ingestion finished)
    """
    project = project or os.environ["LANGSMITH_PROJECT"]
    client = Client()

    deadline = time.time() + timeout_s
    runs: list = []
    while time.time() < deadline:
        runs = list(client.list_runs(project_name=project, trace_id=trace_id))
        root = next((r for r in runs if str(r.id) == str(trace_id)), None)
        if root is not None and root.end_time is not None:
            break
        time.sleep(POLL_INTERVAL_S)
    return runs


def _cache_read_tokens(run) -> int:
    """Cache-hit portion of this run's prompt tokens, if the provider reported it (see module docstring)."""
    try:
        return run.extra["metadata"]["usage_metadata"]["input_token_details"]["cache_read"] or 0
    except (KeyError, TypeError, AttributeError):
        return 0


def _cost_for(run) -> float:
    """$ cost for one LLM run's token usage, via llm_client's price table (see module docstring)."""
    prompt = run.prompt_tokens or 0
    completion = run.completion_tokens or 0
    cache_hit = min(_cache_read_tokens(run), prompt)
    cache_miss = prompt - cache_hit
    return (
        cache_miss * PRICE_PER_TOKEN["input_cache_miss"]
        + cache_hit * PRICE_PER_TOKEN["input_cache_hit"]
        + completion * PRICE_PER_TOKEN["output"]
    )


def summarize_trace(trace_id: str, project: str | None = None, timeout_s: float = POLL_TIMEOUT_S) -> dict:
    """
    Per-step latency/token/cost breakdown for one traced main_1/main_2 run, grouped by
    run name -- e.g. "orchestrator_agent.pick_candidate_categories",
    "classify_agent.classify_paths", "classify_agent._score_batch",
    "reranker.rerank_titles".

    In: trace_id, optional project, optional ingestion-wait timeout
    Out: dict with root (name + total latency_s), steps ({name: {calls, latency_s,
         prompt_tokens, completion_tokens, cost_usd}}), total_cost_usd (llm runs only,
         so parent chain runs' rolled-up token totals aren't double-counted)
    """
    runs = fetch_trace_runs(trace_id, project, timeout_s)
    if not runs:
        return {"root": None, "steps": {}, "total_cost_usd": 0.0, "warning": "trace not found (ingestion timeout?)"}

    root = next((r for r in runs if str(r.id) == str(trace_id)), None)

    steps: dict[str, dict] = {}
    total_cost = 0.0
    for r in runs:
        step = steps.setdefault(r.name, {
            "calls": 0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
        })
        step["calls"] += 1
        if r.end_time is not None:
            step["latency_s"] += (r.end_time - r.start_time).total_seconds()
        if r.run_type == "llm":
            step["prompt_tokens"] += r.prompt_tokens or 0
            step["completion_tokens"] += r.completion_tokens or 0
            cost = _cost_for(r)
            step["cost_usd"] += cost
            total_cost += cost

    for step in steps.values():
        step["latency_s"] = round(step["latency_s"], 3)
        step["cost_usd"] = round(step["cost_usd"], 6)

    return {
        "root": {
            "name": root.name if root else None,
            "latency_s": round((root.end_time - root.start_time).total_seconds(), 3)
                if root and root.end_time else None,
        },
        "steps": steps,
        "total_cost_usd": round(total_cost, 6),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evaluator.langsmith_timing <trace_id>")
        sys.exit(1)

    print(json.dumps(summarize_trace(sys.argv[1]), indent=2))
