# Architecture context (for a fresh agent session)

This is a quick orientation doc — for full detail see `README.md` (deployment,
architecture overview) and `docs/retrieval_pipeline.md` (the two retrieval approaches,
diagrams, file-by-file responsibilities). See also `plan.md` for the reasoning behind the
current `cat_selector`/`analysis_agent` split, if it's not obvious from the code alone.

## What this project is

Amazon Seller Research Assistant: an AI chat tool that answers "how many X launched, and
what's trending" questions against a dataset of 61,635 real Amazon product launches
(2024–2026), stored with title embeddings in ChromaDB.

## The live request chain (one user message → one report)

```
main.py (Flask /api/chat)
  → src/agent_pipeline/analysis_agent.py: run_chat(messages, mode)
      → src/retrieval_pipeline/cat_selector.py: select(messages)      [DeepSeek, 1 call]
          -- decides clarify-vs-proceed; if proceeding, resolves concept (or null for a
             broad category question) + up to 2 categories from the real Amazon taxonomy
        clarify?  → return that reply directly, nothing else runs
        proceed   → niche_report(categories, concept, mode)            [plain function call]
                      → _get_product_subset()
                          concept given → src/retrieval_pipeline/main_1.py ("simple")
                                       or main_2.py ("structured"), chosen by `mode`
                              main_1: rank_category_items (ChromaDB ANN, no cap)
                                      → reranker.rerank_titles (cross-encoder, local)
                              main_2: classify_agent.classify_paths (DeepSeek, per
                                      category) → confident paths trusted outright,
                                      ambiguous paths → reranker.rerank_titles
                              either way: matched ASINs → candidates.hydrate_items()
                                      (fresh ChromaDB lookup for price/seller/velocity/
                                      embeddings -- main_1/main_2's own output doesn't
                                      carry those fields)
                          no concept  → direct ChromaDB category fetch, pipeline bypassed
                      → _get_recent_launches / _get_theme_trend (KMeans) / _get_top_sellers
                        / _get_velocity_summary, all reading the hydrated sub_df
                      → bounded JSON (capped summaries only, never the raw title list)
                → Claude Haiku narrates the JSON into the final report text
                    [the ONLY Claude call in the system -- everything upstream is DeepSeek]
```

## Who owns what

| File | Job |
|---|---|
| `src/agent_pipeline/analysis_agent.py` | `run_chat()` entry point; `niche_report()` and its stat helpers; Haiku narration |
| `src/retrieval_pipeline/cat_selector.py` | Single entry classifier: clarify, concept, categories — one LLM call, used by the live app **and** eval/CLI |
| `src/retrieval_pipeline/main_1.py` | "Simple" mode: vector-rank + rerank, no category-path filtering |
| `src/retrieval_pipeline/main_2.py` | "Structured" mode: `classify_agent` scores real category paths, only the ambiguous residual gets reranked |
| `src/retrieval_pipeline/classify_agent.py` | Per-category-path confident/ambiguous/not_match scoring (structured mode only) |
| `src/retrieval_pipeline/reranker.py` | Local cross-encoder, no LLM |
| `src/retrieval_pipeline/candidates.py` | ChromaDB retrieval helpers + `hydrate_items()` |
| `src/retrieval_pipeline/llm_client.py` | Shared DeepSeek client + token/cost accumulator used by every LLM call in `retrieval_pipeline/` |
| `evaluator/comparison.py` | Benchmarks `main_1` vs `main_2` on the same query with the same `cat_selector`-resolved categories |
| `main.py` | Flask API, the only caller of `analysis_agent.run_chat()`; `/api/chat` is its only real endpoint |
| `src/shared/paths.py` | Every data path in the project. Import from here, never write a path literal |
| `src/shared/naming.py` | `safe_name()` — the filename slug shared by every module that writes a run record |
| `src/offline/` | The batch jobs that BUILD the stores: `ingest` → `load` → `preprocessing` → `build_chroma`. Driven by `build_data.py`. Nothing here is imported by `main.py` |

## Two stores, both built offline

The live path reads exactly two things, and `src/offline/` produces both:

- **`data/raw/chroma_db`** — embeddings + product metadata. Every product query hits this.
- **`data/processed/preprocessed_reduced.parquet`** — the same catalog's category
  taxonomy. `cat_selector.select()` reads it on every request (for its root>level2 menu)
  and `classify_agent` reads it per category (for the full path list, cached to disk).

`python build_data.py` rebuilds both. Nothing in `src/offline/` runs in production.

## Key invariants worth knowing before changing things

- `main_1.run()`/`main_2.run()` **never** pick their own categories — `categories` is
  always a required argument, resolved upstream by `cat_selector.select()`. Don't
  reintroduce a self-pick fallback; that's exactly the "same input" bug this was fixed
  to prevent (see `plan.md`).
- `concept` can be `None`/`null` — that's a valid, common case (broad category browsing,
  no specific niche), not an error. It bypasses the retrieval pipeline entirely.
- The match count is never capped at an arbitrary `top_n` — `rank_category_items` always
  scores every item in the picked categories, and the reranker's `score > 0` threshold is
  what actually decides match/no-match. If you see a hardcoded cap reappear anywhere in
  this chain, that's a regression of the original bug this whole retrieval pipeline was
  built to fix.
- `evaluator/results/*.json` files predate the `cat_selector` unification — they were
  produced when `main_1`/`main_2` each independently picked their own categories. Don't
  treat them as validating today's shared-input behavior; a fresh `evaluator/comparison.py`
  run is needed for that (deferred, see `plan.md`).
