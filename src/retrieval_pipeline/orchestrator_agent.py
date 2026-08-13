"""
Orchestrator agent.

Job: given a free-text query, decide WHICH top-level categories are worth searching --
a single cheap LLM call over root > level-2 category pairs, nothing downstream.

Deliberately just 1 LLM call per query, and its pick is trusted as-is: no
classify_agent.classify_paths() verification pass, no match-count threshold. That
verification step was tried and dropped -- it doubled latency (one classify_paths call
per candidate, each itself a batched multi-call LLM scoring pass over every leaf path)
for a check that rarely overturned the root-level pick.

This only decides WHICH root categories are relevant -- it does not run the downstream
exact-filter / reranker-on-residual / aggregate steps.

CLI:
    python -m src.retrieval_pipeline.orchestrator_agent "kids costumes"
"""

import json
import sys
from pathlib import Path

import pandas as pd
from langsmith import traceable

from src.retrieval_pipeline.classify_agent import PARQUET_PATH, _safe_name
from src.retrieval_pipeline.llm_client import MODEL, client, record_usage

OUTPUT_DIR = Path("data/processed/orchestrator_runs")
MAX_CANDIDATES = 2

CANDIDATE_PROMPT = """A user asked: "{query}" against a product catalog.

Below are ALL root category > level-2 branch pairs that exist in this catalog (Amazon's
own real category structure -- use only these, never invent one).

{pairs_block}

Pick up to {max_candidates} root categories that are plausible homes for this query.
A query can genuinely span more than one category (e.g. "kids costumes" belongs in both
"Clothing, Shoes & Jewelry" (Costumes & Accessories) and "Toys & Games" (Dress Up & Pretend
Play)) -- don't force a single pick if more than one is genuinely plausible, but don't pad
the list with categories that only weakly relate either.

Return ONLY a JSON array: [{{"category": "...", "reason": "one sentence"}}, ...]
No other text, no markdown fences.
"""


def load_root_level2_pairs(parquet_path: Path | str = PARQUET_PATH) -> list[str]:
    """
    Build the cheap root > level-2 candidate menu for orchestrator routing.

    In: parquet_path to preprocessed data
    Out: sorted list of "root > level2" strings, one per distinct real pair
    """
    df = pd.read_parquet(parquet_path, columns=["cat", "category_path"])
    df = df.dropna(subset=["category_path"])

    def _level2(path: str) -> str | None:
        parts = path.split(" > ")
        return parts[1] if len(parts) > 1 else None

    df["l2"] = df["category_path"].map(_level2)
    pairs = df[["cat", "l2"]].dropna().drop_duplicates()
    return sorted(f"{r.cat} > {r.l2}" for r in pairs.itertuples())


@traceable(run_type="chain", name="orchestrator_agent.pick_candidate_categories")
def pick_candidate_categories(query: str, pairs: list[str] | None = None) -> list[dict]:
    """
    LLM call: pick up to MAX_CANDIDATES root categories worth searching for this query.
    (run_type="chain", not "llm" -- the actual model call is separately traced by
    llm_client's wrap_openai-wrapped client; this wrapper also does JSON parsing.)

    In: query text, optional pre-loaded root>level2 pairs (else loaded fresh)
    Out: list of {category, reason} dicts, at most MAX_CANDIDATES long
    """
    if pairs is None:
        pairs = load_root_level2_pairs()

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": CANDIDATE_PROMPT.format(
                query=query, pairs_block="\n".join(pairs), max_candidates=MAX_CANDIDATES
            ),
        }],
    )
    record_usage(resp)
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    candidates = json.loads(text.strip())
    return candidates[:MAX_CANDIDATES]


@traceable(run_type="chain", name="orchestrator_agent.run")
def run(query: str) -> dict:
    """
    Pick up to MAX_CANDIDATES root categories for this query and save the decision.

    In: query text
    Out: dict with query, categories -> [{category, reason}, ...]; also saved to
         OUTPUT_DIR/{query}.json
    """
    candidates = pick_candidate_categories(query)
    print(f"Categories: {[c['category'] for c in candidates]}")

    output = {"query": query, "categories": candidates}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{_safe_name(query)}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.retrieval_pipeline.orchestrator_agent "<query>"')
        sys.exit(1)

    run(query=sys.argv[1])
