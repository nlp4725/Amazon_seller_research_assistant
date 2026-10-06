# Retrieval pipeline: two approaches, benchmarked against each other

> **Update, October 2026:** `main_2` is now a three-stage funnel — path identification (Jev by
> default), rerank, then LLM title filtering (`title_filter.py`). Design rationale, results against
> the rebuilt golden set and known limits: [evaluation_2026_10.md](evaluation_2026_10.md).


Two competing implementations of the same job — given a free-text query (e.g. "dog
drinking bowl"), return every genuinely matching product in the catalog, not a
retrieval-size artifact capped at some arbitrary top_n. Both live in
`src/retrieval_pipeline/`, both are runnable and importable independently, and
`evaluator/comparison.py` runs them both against the same queries to measure which one
actually wins on precision/recall/f1/latency/cost.

`src/agent_pipeline/analysis_agent.py` (the live chat product) is wired into this
pipeline: a concept-narrowed query is routed through `main_1` ("simple") or `main_2`
("structured") based on a `mode` the user picks in the UI, never an LLM — see the
README's "Retrieval Mode Selection" section for the full wiring. Broad category browsing
(no concept) still bypasses this pipeline entirely with a direct ChromaDB fetch, since
there's no query to rank or classify against.

Both approaches take `categories` (up to 2, `[{"category", "reason"}, ...]`) as a
**required** argument from the caller — neither picks its own. `cat_selector.py` is the
single place that resolves `categories` (and, for the live app, a clarify-vs-proceed
decision and a concept), called once per query and shared identically with both mains —
see `cat_selector.py`'s own docstring for why this replaced two previously-separate,
occasionally-disagreeing category-picking calls.

## Approach 1 — `main_1.py`, "simple" (rank + rerank)

```mermaid
flowchart LR
    Q[query] --> R[candidates.rank_category_items\nChromaDB ANN query, no cap]
    Cat[categories\nfrom cat_selector] --> R
    R -->|every item, vector-ranked| X[reranker.rerank_titles\ncross-encoder, local]
    X -->|score > 0| M[matches]
```

Given `categories`, **every** item in them is vector-ranked (ChromaDB ANN query,
`n_results` = full category count — never an arbitrary cap) and cross-encoder reranked
directly against the query. `score > 0` is the final match/no-match line.

| File | Role |
|---|---|
| `candidates.py` (`rank_category_items`) | ChromaDB ANN query over the given categories, ranked by cosine similarity, no truncation |
| `reranker.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores every (query, title) pair; `score > 0` = match |
| `main_1.py` | Wires the two together, saves output, tracks latency/cost |

Cheap and fast (no LLM calls of its own), but expected to lose precision on broad root
categories where the query concept is a small slice of a large catalog — nothing narrows
the pool before the reranker sees it.

## Approach 2 — `main_2.py`, "structured" (classify + rerank residual)

```mermaid
flowchart LR
    Q[query] --> C[classify_agent.classify_paths\nbatched LLM calls, per category]
    Cat[categories\nfrom cat_selector] --> C
    C -->|confident_match paths| F1[candidates.fetch_items_for_paths\ntrusted, no reranking]
    C -->|ambiguous_match paths| F2[candidates.fetch_items_for_paths]
    F2 --> X[reranker.rerank_titles\ncross-encoder, local]
    X -->|score > 0| M2[matches]
    F1 --> M2
```

Given `categories`, instead of reranking everything, `classify_agent` scores every
**real** Amazon `category_path` under each into confident/ambiguous/not_match, and only
the ambiguous residual gets reranked:

- **`confident_match`**: paths specific enough that every item under them is trusted as a
  match outright, no title-reading needed. Example: for the query "dog water bowl", the
  real path `Pet Supplies > Dogs > Feeding & Watering Supplies > Bowls & Dishes` is a
  confident match — everything under it is, definitionally, a dog bowl or dish.
- **`ambiguous_match`** (the "candidate" paths): broader or adjacent paths that *could*
  hold matching items but aren't specific enough to trust blindly — handed to the
  reranker to score item-by-item. Example: `Pet Supplies > Dogs > Apparel & Accessories`
  is a candidate path for "dog water bowl" — it might contain a travel water bottle
  accessory, but most items under it (collars, coats) aren't matches, so each title needs
  individual reranking rather than a blanket trust.
- **`not_match`**: dropped outright.

| File | Role |
|---|---|
| `classify_agent.py` | 1+ batched LLM calls per category: scores every real `category_path` into confident/ambiguous/not_match |
| `candidates.py` (`fetch_items_for_paths`) | Exact ChromaDB filter by `category_path` — used for both confident items (trusted) and ambiguous items (sent to reranker) |
| `reranker.py` | Same cross-encoder, only run on the ambiguous residual |
| `main_2.py` | Wires it together, saves output, tracks latency/cost |

More expensive (one classify_paths call per candidate category, each itself a batched
multi-call LLM pass over every *distinct* `category_path` in that category — not every
item; e.g. Pet Supplies has 681 items but only 166 distinct paths, Home & Kitchen has
5,354 items but only 629 distinct paths) but expected to win on precision/recall when a
real `category_path` exists that cleanly identifies the query concept.

## Shared infrastructure

| File | Role |
|---|---|
| `llm_client.py` | Single DeepSeek client, wrapped once with LangSmith's `wrap_openai` so every LLM call from any agent lands in the same trace tree; also a thread-safe token-usage/cost accumulator (`reset_usage()` / `get_usage()`) that both mains read after a run |
| `cat_selector.py` | Single entry classifier (1 LLM call): clarify-vs-proceed, concept, and up to `MAX_CANDIDATES` root categories — see its own docstring. Called once per query by both mains' CLI blocks, `evaluator/comparison.py`, and `analysis_agent.py`, so every consumer shares the same resolved categories |
| `src/offline/build_chroma.py` | Builds/updates the ChromaDB collection (`title_embedding_db`) both approaches query against — lives in `src/offline/` precisely because it is not part of either approach's request-time path |

Every LLM call in this package goes through `llm_client.client` (wrapped for LangSmith
tracing) and calls `record_usage(resp)` right after, so `evaluator/comparison.py` can
read exact token counts and $ cost per run without computing it separately.

## Output schema

Both mains save their run to `data/processed/pipeline_runs/{simple,structured}__{query}.json`
and return the same dict shape:

```json
{
  "query": "...",
  "approach": "simple | structured",
  "categories": [...],
  "titles_found_count": 681,
  "match_count": 21,
  "titles_found": [
    {"asin": "...", "title": "...", "rerank_score": 4.2, "is_match": true, ...}
  ],
  "latency_s": {"...": "...", "total": 10.3},
  "usage": {"calls": 1, "estimated_cost_usd": 0.0001, ...}
}
```

`titles_found` records **every** title considered, not just matches — same schema as
`evaluator/golden_dataset.json`, so `evaluator/metrics.py` can score either output
directly by comparing `is_match=true` ASIN sets.

## Evaluation

| File | Role |
|---|---|
| `evaluator/golden_dataset.json` | Hand-verified ground truth for 20 queries (13 `verified`, 6 `reconciled`, 1 `verified_partial`) |
| `evaluator/metrics.py` | `score_pipeline_run(output)` — precision/recall/f1 by ASIN set overlap against the matching golden_dataset.json entry |
| `evaluator/comparison.py` | Runs both mains over every verified query, scores each with `metrics.py`, aggregates, saves JSON + markdown report to `evaluator/results/` |

```bash
python -m src.retrieval_pipeline.main_1 "dog drinking bowl"
python -m src.retrieval_pipeline.main_2 "dog drinking bowl"
python -m evaluator.metrics data/processed/pipeline_runs/simple__dog_drinking_bowl.json
python -m evaluator.comparison                                    # every verified query, both approaches
python -m evaluator.comparison "dog drinking bowl" "kids costumes"  # specific queries only
```

**Caveat:** `evaluator/golden_dataset.json` entries with `status: verified_partial`
(currently only "female fitness clothes") have ground truth built from a sample, not an
exhaustive enumeration — recall against them is a lower bound, not exact. `comparison.py`
carries this status through into its output so it isn't silently averaged in as if it
were as reliable as the fully verified queries.
