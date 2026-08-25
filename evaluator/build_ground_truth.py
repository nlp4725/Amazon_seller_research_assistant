"""
Reusable ground-truth builder for evaluator/golden_dataset.json entries.

Job: given a query, a category_hint, and a recall-oriented keyword regex, pull every
candidate title from preprocessed_reduced.parquet matching the regex within those
categories (a broad net, not the pipeline being evaluated -- keyword matching here is
independent of main_1/main_2's retrieval, to avoid circularity), then have DeepSeek
judge each candidate as a genuine match or not, batched like classify_agent.py.

Same "every title considered, not just matches" schema as the rest of golden_dataset.json:
{asin, title, is_match} per candidate, plus match_count/titles_found_count. Each verdict
also carries a "reason" (one sentence, why it is/isn't a genuine match) and "confidence"
(0-10, how sure the judge is) -- both from the same LLM call, not a separate pass, so the
judgment is auditable instead of an opaque true/false.

For very broad queries where the keyword net returns far more candidates than is
practical to judge exhaustively (e.g. "summer dress"), pass sample_cap to randomly
sample -- the resulting entry should be marked status="verified_partial" in
golden_dataset.json, same convention as the existing "female fitness clothes" entry.

CLI:
    python -m evaluator.build_ground_truth "cat scratching post" "Pet Supplies" "scratch"
"""

import contextvars
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from langsmith import traceable

from src.retrieval_pipeline.llm_client import MODEL, client, record_usage
from src.shared.paths import PREPROCESSED_PARQUET

BATCH_SIZE = 40
RANDOM_SEED = 42

JUDGE_PROMPT = """A user asked: "{query}" against a product catalog.

Below are real product titles (with ids) surfaced by a recall-oriented keyword sweep --
some are genuine matches, many are false positives that just happen to share vocabulary.

{items_block}

For each title, judge whether it is a GENUINE match to the query's actual concept/
function/purpose -- not just keyword overlap. A genuine match means someone searching
"{query}" would consider this product a correct, relevant result. Reject items that
share a word but serve a clearly different purpose (e.g. a novelty print or a shared
adjective doesn't make it a match; the product's actual function has to match).

For each item also give a one-sentence reason (why it is/isn't a genuine match) and a
confidence 0-10 (how sure you are of this specific verdict).

Return ONLY a JSON array: [{{"id": "...", "is_match": true/false, "reason": "...", "confidence": N}}]
No other text, no markdown fences.
"""


def fetch_candidates(categories: list[str], keyword_pattern: str, parquet_path: Path | str = PREPROCESSED_PARQUET) -> pd.DataFrame:
    """
    Broad recall-oriented keyword sweep over titles within the given categories.

    In: list of root category names, case-insensitive regex to match against title,
        parquet_path
    Out: DataFrame with asin, title columns, one row per matching product
    """
    df = pd.read_parquet(parquet_path, columns=["cat", "asin", "title"])
    subset = df[df["cat"].isin(categories)]
    mask = subset["title"].str.contains(keyword_pattern, case=False, regex=True, na=False)
    return subset.loc[mask, ["asin", "title"]].reset_index(drop=True)


@traceable(run_type="chain", name="build_ground_truth._judge_batch")
def _judge_batch(query: str, batch: dict[str, str]) -> list[dict]:
    """One LLM call: judge a batch of {id: title} pairs against the query."""
    items_block = "\n".join(f"{i}: {title}" for i, title in batch.items())
    prompt = JUDGE_PROMPT.format(query=query, items_block=items_block)
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


@traceable(run_type="chain", name="build_ground_truth.judge_candidates")
def judge_candidates(query: str, candidates: pd.DataFrame) -> list[dict]:
    """
    Batch-judge every candidate with DeepSeek.

    In: query text, DataFrame with asin/title columns
    Out: list of {asin, title, is_match, reason, confidence} dicts, same order as
         candidates; a failed batch's items fall back to is_match=False with a reason
         noting the batch failure, not silently dropped
    """
    if candidates.empty:
        return []

    ids = [str(i) for i in candidates.index]
    id_to_title = {str(i): row.title for i, row in zip(candidates.index, candidates.itertuples())}
    batches = [ids[i:i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    print(f"build_ground_truth: {query!r} -- {len(ids)} candidates, {len(batches)} batches", flush=True)
    verdicts: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(batches), 10)) as ex:
        futures = {
            ex.submit(contextvars.copy_context().run, _judge_batch, query, {i: id_to_title[i] for i in batch_ids}): batch_ids
            for batch_ids in batches
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                for result in fut.result():
                    verdicts[result["id"]] = result
                print(f"  batch {done}/{len(batches)} done", flush=True)
            except Exception as e:
                batch_ids = futures[fut]
                print(f"  batch {done}/{len(batches)} FAILED ({e!r}) -- its {len(batch_ids)} "
                      f"candidates fall back to is_match=False, not silently dropped", flush=True)

    default = {"is_match": False, "reason": "batch judging failed", "confidence": 0}
    return [
        {
            "asin": row.asin,
            "title": row.title,
            "is_match": bool(verdicts.get(str(i), default)["is_match"]),
            "reason": verdicts.get(str(i), default)["reason"],
            "confidence": verdicts.get(str(i), default)["confidence"],
        }
        for i, row in zip(candidates.index, candidates.itertuples())
    ]


def build_ground_truth(
    query: str,
    categories: list[str],
    keyword_pattern: str,
    sample_cap: int | None = None,
    parquet_path: Path | str = PREPROCESSED_PARQUET,
) -> dict:
    """
    Full pipeline: keyword sweep -> optional sample -> DeepSeek judging.

    In: query, categories, keyword_pattern (regex), sample_cap (randomly downsample
        candidates to this many before judging, for very broad queries), parquet_path
    Out: dict {query, categories, keyword_pattern, sampled, match_count,
         titles_found_count, titles_found}
    """
    candidates = fetch_candidates(categories, keyword_pattern, parquet_path)
    sampled = False
    if sample_cap is not None and len(candidates) > sample_cap:
        candidates = candidates.sample(n=sample_cap, random_state=RANDOM_SEED).reset_index(drop=True)
        sampled = True

    titles_found = judge_candidates(query, candidates)
    match_count = sum(t["is_match"] for t in titles_found)

    print(f"  -> {match_count}/{len(titles_found)} genuine matches"
          f"{' (SAMPLED, not exhaustive)' if sampled else ''}", flush=True)

    return {
        "query": query,
        "categories": categories,
        "keyword_pattern": keyword_pattern,
        "sampled": sampled,
        "match_count": match_count,
        "titles_found_count": len(titles_found),
        "titles_found": titles_found,
    }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python -m evaluator.build_ground_truth "<query>" "<category>" "<keyword_regex>" [sample_cap]')
        sys.exit(1)

    query, category, pattern = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else None
    result = build_ground_truth(query, [category], pattern, sample_cap=cap)
    print(json.dumps(result, indent=2))
