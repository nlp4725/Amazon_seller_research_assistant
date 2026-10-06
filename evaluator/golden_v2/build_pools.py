"""
Candidate pools for golden dataset v2 -- the "what gets looked at" step, before labeling.

A label can only be as complete as the set of products someone actually judged. v1 swept
one keyword over one hinted category, so true matches filed elsewhere were never seen
(baby toys, graduation gown, wireless gaming mouse). v2 pools catalog-wide from three
independent sources and labels the union (TREC-style pooling):

  keyword   -- every title matching ALL of the query's regexes in definitions.yaml
  embedding -- the top EMBED_K titles by vector similarity, no category filter
  systems   -- every ASIN any saved pipeline run predicted as a match for the query,
               plus every item v1 judged (so v1's labels are re-checked, not inherited)

Each candidate records which sources found it. Labelers never see that field -- it is for
auditing coverage afterwards (e.g. "how many true items did only the systems find?").

No API calls: the catalog parquet, the local ChromaDB collection, and saved run files.

CLI:
    python -m evaluator.golden_v2.build_pools            # every query in definitions.yaml
    python -m evaluator.golden_v2.build_pools "yoga mat"  # just these
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from evaluator.metrics import load_golden_dataset
from src.retrieval_pipeline.candidates import _load_chroma
from src.shared.model_loader import get_embedder
from src.shared.paths import PIPELINE_RUNS_DIR, PREPROCESSED_PARQUET, safe_name

HERE = Path(__file__).parent
DEFINITIONS = HERE / "definitions.yaml"
POOLS_DIR = HERE / "pools"
EMBED_K = 300


def keyword_hits(df: pd.DataFrame, patterns: list[str]) -> set[str]:
    mask = pd.Series(True, index=df.index)
    for p in patterns:
        if p:
            mask &= df["title"].str.contains(p, flags=re.IGNORECASE, regex=True, na=False)
    return set(df.loc[mask, "asin"])


def embedding_hits(query: str) -> set[str]:
    results = _load_chroma().query(query_embeddings=get_embedder().encode([query]).tolist(),
                                   n_results=EMBED_K, include=["metadatas"])
    return {m["asin"] for m in results["metadatas"][0]}


def system_hits(query: str, golden_v1: dict) -> set[str]:
    asins = {t["asin"] for t in golden_v1.get(query, {}).get("titles_found", []) if t.get("asin")}
    for f in PIPELINE_RUNS_DIR.glob(f"*__{safe_name(query)}.json"):
        run = json.loads(f.read_text())
        asins |= {t["asin"] for t in run.get("titles_found", []) if t.get("is_match") and t.get("asin")}
    return asins


def build(query: str, spec: dict, df: pd.DataFrame, golden_v1: dict) -> dict:
    sources = {
        "keyword": keyword_hits(df, spec.get("keywords", [])),
        "embedding": embedding_hits(query),
        "systems": system_hits(query, golden_v1),
    }
    by_asin = df.set_index("asin")
    candidates = []
    for asin in sorted(set().union(*sources.values())):
        if asin not in by_asin.index:
            continue
        row = by_asin.loc[asin]
        candidates.append({"asin": asin, "title": row["title"], "category_path": row["category_path"],
                           "found_by": [s for s, hits in sources.items() if asin in hits]})
    pool = {"query": query, "definition": spec, "sizes": {s: len(h) for s, h in sources.items()},
            "pool_size": len(candidates), "candidates": candidates}
    POOLS_DIR.mkdir(exist_ok=True)
    (POOLS_DIR / f"{safe_name(query)}.json").write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    return pool


def main(queries: list[str] | None = None) -> None:
    specs = yaml.safe_load(DEFINITIONS.read_text())
    df = pd.read_parquet(PREPROCESSED_PARQUET, columns=["asin", "title", "category_path"])
    df = df.drop_duplicates("asin")
    golden_v1 = {e["query"]: e for e in load_golden_dataset()}
    for query in queries or list(specs):
        pool = build(query, specs[query], df, golden_v1)
        s = pool["sizes"]
        print(f"{query:28s} pool={pool['pool_size']:5d}  keyword={s['keyword']:5d}  "
              f"embedding={s['embedding']:4d}  systems={s['systems']:4d}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
