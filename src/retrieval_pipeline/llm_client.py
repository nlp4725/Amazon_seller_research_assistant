"""
Shared DeepSeek client for every LLM call in this pipeline (classify_agent,
orchestrator_agent, and anything built on top of them).

Wrapped once with LangSmith's wrap_openai so every chat.completions.create() call from
any agent lands in the same trace tree (LANGSMITH_PROJECT in .env), instead of each
agent module building its own untraced client. Also tracks token usage in a process-wide,
thread-safe accumulator so main_1/main_2/comparison.py can read exact $ cost for a run
without each agent computing it separately -- call reset_usage() before a run you want to
measure in isolation, get_usage() after.

Pricing is DeepSeek V4-Flash list price (input cache-miss / cache-hit, output), checked
2026-08-13 at https://deepseek.ai/pricing. Assumes cache-miss rate for any prompt tokens
the response doesn't report as cache hits.
"""

import os
import threading

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import OpenAI

load_dotenv()

MODEL = os.environ["DEEPSEEK_MODEL"]
REQUEST_TIMEOUT = 60
MAX_RETRIES = 2

PRICE_PER_TOKEN = {  # $ per token (list price / 1_000_000) -- see module docstring for source
    "input_cache_miss": 0.14 / 1_000_000,
    "input_cache_hit": 0.0028 / 1_000_000,
    "output": 0.28 / 1_000_000,
}

client = wrap_openai(OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    timeout=REQUEST_TIMEOUT,
    max_retries=MAX_RETRIES,
))

_lock = threading.Lock()
_usage = {"calls": 0, "input_cache_miss_tokens": 0, "input_cache_hit_tokens": 0, "output_tokens": 0}


def reset_usage() -> None:
    """Zero the usage accumulator -- call before a run you want to measure in isolation."""
    with _lock:
        for k in _usage:
            _usage[k] = 0


def record_usage(resp) -> None:
    """
    Add one chat.completions.create() response's token usage to the accumulator.
    Call this right after every LLM call made through `client` in this pipeline.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cache_hit = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    prompt_tokens = usage.prompt_tokens or 0
    with _lock:
        _usage["calls"] += 1
        _usage["input_cache_hit_tokens"] += cache_hit
        _usage["input_cache_miss_tokens"] += max(prompt_tokens - cache_hit, 0)
        _usage["output_tokens"] += usage.completion_tokens or 0


def get_usage() -> dict:
    """Snapshot of the accumulator plus a computed $ cost estimate."""
    with _lock:
        snap = dict(_usage)
    cost = (
        snap["input_cache_miss_tokens"] * PRICE_PER_TOKEN["input_cache_miss"]
        + snap["input_cache_hit_tokens"] * PRICE_PER_TOKEN["input_cache_hit"]
        + snap["output_tokens"] * PRICE_PER_TOKEN["output"]
    )
    snap["estimated_cost_usd"] = round(cost, 6)
    return snap
