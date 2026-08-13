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
runs don't re-derive them -- see CACHE_DIR. Each classification run writes a JSON file
of {bucket: {id: path}} to OUTPUT_DIR.

CLI:
    python -m src.retrieval_pipeline.classify_agent "Pet Supplies" "dog drinking bowl"
"""

import contextvars
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from langsmith import traceable

from src.retrieval_pipeline.llm_client import MODEL, client, record_usage

PARQUET_PATH = Path("data/processed/preprocessed_reduced.parquet")
CACHE_DIR = Path("data/processed/category_path_cache")
OUTPUT_DIR = Path("data/processed/category_classifications")
BATCH_SIZE = 60
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
             e.g. a "dog drinking bowl" query still confidently means a path named
             "Fountains" -- the intent is the drinking function, not literally "bowl"-shaped.
             e.g. a "kids costumes" query still confidently means a path named
             "Dress Up & Pretend Play > Costumes" or "Cosplay Apparel" even though neither
             says "kids costumes" verbatim -- the intent is dress-up/pretend-play wear,
             not the exact phrase.
  false -- the path could plausibly hold BOTH matching and non-matching items
           (e.g. a generic "Bowls" path could hold food bowls or water bowls; a generic
           "Boys" clothing path could hold a costume or could hold anything else)
Omit paths that are clearly irrelevant -- do not force a low-relevance match to appear.

Return ONLY a JSON array: [{{"id": "...", "score": N, "confident": true/false, "note": "..."}}]
No other text, no markdown fences.
"""


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _path_id(path: str) -> str:
    """Stable id for a category path -- deterministic, order-independent."""
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def load_category_paths(category: str, parquet_path: Path | str = PARQUET_PATH) -> dict[str, str]:
    """
    Load distinct real category paths for a top-level category, assign stable ids, cache.

    In: category (top-level name, e.g. "Pet Supplies"), parquet_path to preprocessed data
    Out: dict {path_id: category_path}; also written to CACHE_DIR/{category}.json
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_safe_name(category)}.json"

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
def _score_batch(query: str, batch: dict[str, str]) -> list[dict]:
    """
    One LLM call: score a batch of (id, path) pairs against the query.
    (run_type="chain", not "llm" -- the actual model call is separately traced by
    llm_client's wrap_openai-wrapped client; this wrapper also does JSON parsing.)

    In: query text, batch {path_id: category_path}
    Out: list of {id, score, confident, note} dicts for paths judged at least plausible
    """
    items_block = "\n".join(f"{pid}: {path}" for pid, path in batch.items())
    prompt = PROMPT_TEMPLATE.format(query=query, items_block=items_block)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    record_usage(resp)
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


@traceable(run_type="chain", name="classify_agent.classify_paths")
def classify_paths(
    query: str,
    category: str,
    paths_by_id: dict[str, str] | None = None,
) -> dict:
    """
    Classify every real category path under `category` against `query`.

    In: query text, top-level category name, optional pre-loaded paths_by_id (else cached/loaded)
    Out: dict with query, category, confident_match/ambiguous_match/not_match -> {id: path};
         also saved as JSON to OUTPUT_DIR/{category}__{query}.json
    """
    if paths_by_id is None:
        paths_by_id = load_category_paths(category)

    ids = list(paths_by_id.keys())
    batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    print(f"classify_agent: {category!r} -- {len(ids)} paths, {len(batches)} batches", flush=True)
    scored: dict[str, dict] = {}
    failed_batches = 0
    with ThreadPoolExecutor(max_workers=min(len(batches), 10)) as ex:
        futures = {
            # contextvars (incl. LangSmith's current-run tracking) don't cross a thread
            # boundary on their own -- ex.submit(_score_batch, ...) directly would make
            # each batch call show up as its own orphaned root trace instead of nesting
            # under this classify_paths run. copy_context().run(...) carries the parent
            # run context into the worker thread so the trace tree stays correct.
            ex.submit(contextvars.copy_context().run, _score_batch, query, {pid: paths_by_id[pid] for pid in batch_ids}): batch_ids
            for batch_ids in batches
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                for result in fut.result():
                    scored[result["id"]] = result
                print(f"  batch {done}/{len(batches)} done", flush=True)
            except Exception as e:
                failed_batches += 1
                batch_ids = futures[fut]
                print(f"  batch {done}/{len(batches)} FAILED ({e!r}) -- its "
                      f"{len(batch_ids)} paths fall back to not_match, not silently dropped", flush=True)

    if failed_batches:
        print(f"WARNING: {failed_batches}/{len(batches)} batches failed -- "
              f"result is a partial/degraded classification, not a complete one", flush=True)

    confident_match, ambiguous_match, not_match = {}, {}, {}
    for pid, path in paths_by_id.items():
        result = scored.get(pid)
        if result is None or result["score"] < AMBIGUOUS_SCORE_FLOOR:
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{_safe_name(category)}__{_safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  confident_match: {len(confident_match)}")
    print(f"  ambiguous_match: {len(ambiguous_match)}")
    print(f"  not_match: {len(not_match)}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m src.retrieval_pipeline.classify_agent "<category>" "<query>"')
        sys.exit(1)

    classify_paths(query=sys.argv[2], category=sys.argv[1])
