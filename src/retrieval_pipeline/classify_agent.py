"""
Category classifier agent.

Job: given a query and a top-level category, score every REAL Amazon category path
under that category (from preprocessed_reduced.parquet's category_path column --
never invented) and split them into three buckets:
  - confident_match: path name unambiguously means the query concept (e.g. "Fountains"
    for "drinking") -- safe to use as an exact ChromaDB filter, no title-reading needed
  - ambiguous_match: path name is plausible but could hold both matching and
    non-matching items (e.g. "Bowls & Dishes" could be food or water) -- needs a
    downstream classifier (reranker) over the actual titles in this bucket
  - not_match: irrelevant to the query

Scoring is a single LLM call per batch against the raw query, guided by inline examples
in the prompt (not a separate rubric-decomposition step -- that was tried and reverted:
it added a second LLM call per query and, on "dog drinking bowl," produced a worse
result than this simpler version by reading "bowl" too literally and dropping the
"Fountains" path entirely). Keep this prompt's guidance concrete and example-based
rather than adding more structure back in.

Paths are assigned a stable id (hash of the path string) and cached locally so repeated
runs don't re-derive them -- see CATEGORY_PATH_CACHE. Each classification run writes a
JSON file of {bucket: {id: path}} to CATEGORY_CLASSIFY_DIR.

Concurrency: batches fan out with asyncio.gather over llm_client.aclient (this module was
a ThreadPoolExecutor before). The async rewrite buys two things a thread pool could not:
a real per-batch deadline (asyncio.timeout actually cancels the request and closes the
connection -- a worker thread can only be abandoned, and keeps generating and billing),
and correct LangSmith nesting for free, since asyncio tasks copy the parent context and
threads needed an explicit contextvars.copy_context() workaround.

Responses are streamed. In async, streaming is not what makes cancellation work
(asyncio.timeout would cancel a plain awaited call too) -- it is here so stream_options
include_usage can report per-call token/cache counts, and so the deadline is checked
between chunks rather than only at the end.

classify_paths() stays a normal sync function wrapping asyncio.run(); only this module's
internals are async, so main_2.py and everything above it are unchanged. Use
aclassify_paths() directly from an existing event loop -- asyncio.run() raises if a loop
is already running.

CLI:
    python -m src.retrieval_pipeline.classify_agent "Pet Supplies" "dog drinking bowl"
"""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from langsmith import traceable

from src.retrieval_pipeline import jev_scorer
from src.retrieval_pipeline.llm_client import MODEL, aclient, record_usage
from src.shared.paths import (
    CATEGORY_CLASSIFY_DIR,
    CATEGORY_PATH_CACHE,
    PREPROCESSED_PARQUET,
    safe_name,
)

BATCH_SIZE = 60

# Hard ceiling on one batch's LLM call. This is a SAFETY NET against a hung request, not a
# latency target -- deliberately generous. llm_client's REQUEST_TIMEOUT cannot do this job:
# DeepSeek sends blank keep-alive lines on non-streaming requests (documented at
# api-docs.deepseek.com/quick_start/rate_limit), which reset httpx's read timeout, so a
# 60s timeout was measured letting a call run 298s. Streaming + this deadline is what
# actually bounds a call, because breaking the chunk loop closes the connection and stops
# generation instead of merely abandoning a request that keeps billing.
#
# Do NOT lower this to trim p50. Measured on "dog water fountain": the slowest single-path
# calls were "Fountains" (47.7s) and "Bowls & Dishes" (87.5s) -- i.e. the paths that are
# actually relevant. The model thinks longest exactly where the decision is hardest, so an
# aggressive deadline preferentially deletes correct matches (a timed-out batch falls to
# not_match below). Slowest batch observed in production tracing: 298.7s.
BATCH_DEADLINE_S = 300

# Ceiling on batches in flight at once. DeepSeek's documented limit is 2500 concurrent for
# deepseek-v4-flash (no RPM/TPM limit published), so this is nowhere near the provider's
# cap -- it just stops a very large category from opening hundreds of sockets at once.
# At BATCH_SIZE=60 a category needs len(paths)/60 batches, so most categories never hit it.
MAX_CONCURRENT_BATCHES = 10
CONFIDENT_SCORE_FLOOR = 6
AMBIGUOUS_SCORE_FLOOR = 3

PROMPT_TEMPLATE = """A user asked: "{query}" against a product catalog.

Below are REAL, exact category paths (Amazon's own assigned categories) with their ids.
Use only these -- never invent a category name.

{items_block}

For each path that is at least plausibly relevant to the query, score it 0-10 for how
well it matches, and mark "confident":
  true  -- the path's name UNAMBIGUOUSLY means the query's underlying CONCEPT, even if
           the path uses different words than the query. Match the concept/function/
           purpose the user actually wants, not the literal noun they happened to type --
           a query's specific word is one example of the thing, not a strict requirement
           on shape, wording, or category label.
           BUT never relax WHO the product is for. Species, audience, age and gender are
           part of the request itself, not wording choices -- a "dog" query does not match
           a path under Cats, and a "kids" query does not match an adult path, however
           identical the product's function.
             e.g. a "dog drinking bowl" query still confidently means
             "Pet Supplies > Dogs > Feeding & Watering Supplies > Fountains" -- the intent
             is the drinking function, not literally "bowl"-shaped -- but NOT
             "Pet Supplies > Cats > Feeding & Watering Supplies > Fountains", which is the
             right function for the wrong animal.
             e.g. a "kids costumes" query still confidently means a path named
             "Dress Up & Pretend Play > Costumes" even though it does not say "kids"
             verbatim -- the intent is dress-up/pretend-play wear -- but NOT a general
             "Cosplay Apparel" path, which is not specific to children.
  false -- the path could plausibly hold BOTH matching and non-matching items
           (e.g. a generic "Bowls" path could hold food bowls or water bowls; a generic
           "Boys" clothing path could hold a costume or could hold anything else)
Omit paths that are clearly irrelevant -- do not force a low-relevance match to appear.

Return ONLY a JSON array: [{{"id": "...", "score": N, "confident": true/false, "note": "..."}}]
No other text, no markdown fences.
"""


def _path_id(path: str) -> str:
    """Stable id for a category path -- deterministic, order-independent."""
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def load_category_paths(category: str, parquet_path: Path | str = PREPROCESSED_PARQUET) -> dict[str, str]:
    """
    Load distinct real category paths for a top-level category, assign stable ids, cache.

    In: category (top-level name, e.g. "Pet Supplies"), parquet_path to preprocessed data
    Out: dict {path_id: category_path}; also written to CATEGORY_PATH_CACHE/{category}.json
    """
    CATEGORY_PATH_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CATEGORY_PATH_CACHE / f"{safe_name(category)}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    df = pd.read_parquet(parquet_path, columns=["cat", "category_path"])
    paths = df.loc[df["cat"] == category, "category_path"].dropna().unique()
    paths_by_id = {_path_id(p): p for p in sorted(paths)}

    with open(cache_path, "w") as f:
        json.dump(paths_by_id, f, indent=2)
    return paths_by_id


@traceable(run_type="chain", name="classify_agent._score_batch")
async def _score_batch(query: str, batch: dict[str, str]) -> list[dict]:
    """
    One streamed LLM call: score a batch of (id, path) pairs against the query.
    (run_type="chain", not "llm" -- the model call itself is separately traced by
    llm_client's wrap_openai-wrapped client; this wrapper also does JSON parsing.)

    Raises TimeoutError if the call exceeds BATCH_DEADLINE_S; the request is cancelled and
    the connection closed, so generation actually stops rather than billing on unread.
    classify_paths() catches it per batch.

    In: query text, batch {path_id: category_path}
    Out: list of {id, score, confident, note} dicts for paths judged at least plausible
    """
    items_block = "\n".join(f"{pid}: {path}" for pid, path in batch.items())
    prompt = PROMPT_TEMPLATE.format(query=query, items_block=items_block)

    parts: list[str] = []
    try:
        async with asyncio.timeout(BATCH_DEADLINE_S):
            stream = await aclient.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )
            try:
                async for chunk in stream:
                    if chunk.usage is not None:
                        # Usage rides on the final chunk only. A cancelled or malformed call
                        # never reaches it, so get_usage() undercounts a failed batch -- the
                        # tokens were still spent. LangSmith's trace is the accurate source
                        # (see evaluator/langsmith_timing.py).
                        record_usage(chunk)
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
            finally:
                await stream.close()
    except asyncio.TimeoutError as e:
        # Re-raised as plain TimeoutError so classify_paths' handler doesn't have to know
        # this module is async. (In 3.11 asyncio.TimeoutError IS TimeoutError, but being
        # explicit keeps the contract readable.)
        raise TimeoutError(
            f"batch of {len(batch)} paths exceeded BATCH_DEADLINE_S={BATCH_DEADLINE_S}s"
        ) from e

    text = "".join(parts).strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


@traceable(run_type="chain", name="classify_agent.classify_paths")
async def aclassify_paths(
    query: str,
    category: str,
    paths_by_id: dict[str, str] | None = None,
) -> dict:
    """
    Classify every real category path under `category` against `query`. Async form -- call
    this from inside an existing event loop; use classify_paths() from sync code.

    In: query text, top-level category name, optional pre-loaded paths_by_id (else cached/loaded)
    Out: dict with query, category, confident_match/ambiguous_match/not_match -> {id: path};
         also saved as JSON to CATEGORY_CLASSIFY_DIR/{category}__{query}.json
    """
    if paths_by_id is None:
        paths_by_id = load_category_paths(category)

    ids = list(paths_by_id.keys())
    batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    print(f"classify_agent: {category!r} -- {len(ids)} paths, {len(batches)} batches", flush=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
    # CLASSIFIER_BACKEND=jev swaps the DeepSeek call for TypeSafe's Jev (jev_scorer.py);
    # everything after scoring -- buckets, failure handling, the saved JSON -- is shared.
    jev = os.environ.get("CLASSIFIER_BACKEND", "jev") == "jev"
    jev_client = jev_scorer.new_client() if jev else None

    async def run_batch(batch_ids: list[str]) -> list[dict]:
        batch = {pid: paths_by_id[pid] for pid in batch_ids}
        async with sem:
            if jev:
                return await jev_scorer.score_batch(jev_client, query, batch)
            return await _score_batch(query, batch)

    # return_exceptions=True so one failed batch degrades that batch only, instead of
    # cancelling its siblings -- gather's default would propagate the first exception and
    # throw away work already done by the others.
    try:
        results = await asyncio.gather(
            *(run_batch(b) for b in batches), return_exceptions=True
        )
    finally:
        if jev_client is not None:
            await jev_client.aclose()

    scored: dict[str, dict] = {}
    failed_batches = 0
    for done, (batch_ids, result) in enumerate(zip(batches, results), start=1):
        if isinstance(result, BaseException):
            failed_batches += 1
            print(f"  batch {done}/{len(batches)} FAILED ({result!r}) -- its "
                  f"{len(batch_ids)} paths fall back to not_match, not silently dropped", flush=True)
            continue
        for item in result:
            scored[item["id"]] = item
        print(f"  batch {done}/{len(batches)} done", flush=True)

    if failed_batches:
        print(f"WARNING: {failed_batches}/{len(batches)} batches failed -- "
              f"result is a partial/degraded classification, not a complete one", flush=True)

    confident_match, ambiguous_match, not_match = {}, {}, {}
    for pid, path in paths_by_id.items():
        result = scored.get(pid)
        if result is None or result.get("score", 0) < AMBIGUOUS_SCORE_FLOOR:
            not_match[pid] = path
        elif result["confident"] and result["score"] >= CONFIDENT_SCORE_FLOOR:
            confident_match[pid] = path
        else:
            ambiguous_match[pid] = path

    output = {
        "query": query,
        "category": category,
        "partial_failure": failed_batches > 0,
        "failed_batches": failed_batches,
        "total_batches": len(batches),
        "confident_match": confident_match,
        "ambiguous_match": ambiguous_match,
        "not_match": not_match,
    }
    if jev:
        # Jev's raw probabilities per path, so cutoffs can be re-tuned without re-asking.
        output["jev_mode"] = jev_scorer.JEV_MODE
        output["jev_scores"] = {
            pid: r.get("p_match", r.get("probabilities")) for pid, r in scored.items()
        }

    CATEGORY_CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CATEGORY_CLASSIFY_DIR / f"{safe_name(category)}__{safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  confident_match: {len(confident_match)}")
    print(f"  ambiguous_match: {len(ambiguous_match)}")
    print(f"  not_match: {len(not_match)}")

    return output


def classify_paths(
    query: str,
    category: str,
    paths_by_id: dict[str, str] | None = None,
) -> dict:
    """
    Sync entry point -- unchanged signature, so main_2.py and the CLI need no edits.

    Raises RuntimeError if called from a thread that already has a running event loop
    (asyncio.run's rule). Nothing in this pipeline is async today; if that changes, call
    aclassify_paths() directly instead of reaching for asyncio.run() here.
    """
    return asyncio.run(aclassify_paths(query, category, paths_by_id))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m src.retrieval_pipeline.classify_agent "<category>" "<query>"')
        sys.exit(1)

    classify_paths(query=sys.argv[2], category=sys.argv[1])
