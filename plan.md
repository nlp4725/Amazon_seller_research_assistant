# Unify category selection into `cat_selector`, split `chat_engine` into `analysis_agent`

**Status: implemented.** Kept here as a durable record of why, in addition to the git
history of the individual file changes.

## Context

Before this change there were **two separate, overlapping systems** that each picked
"categories" for a query:

1. `chat_engine.py`'s first DeepSeek call — clarified ambiguous requests, then picked one
   `category` (from a fixed, hand-maintained 21-item list) + a `concept` (normalized
   phrasing). Ran on every live chat turn.
2. `src/retrieval_pipeline/orchestrator_agent.py`'s `pick_candidate_categories()` — a
   *separate* DeepSeek call that picked up to 2 categories from the real Amazon
   `root > level2` taxonomy. On the live path this never actually ran —
   `main_1.run()`/`main_2.run()` skipped it whenever a caller passed `category=`, which
   `chat_engine.py` always did. It only fired when `main_1`/`main_2` were called
   **without** a category — i.e. only in `evaluator/comparison.py` and standalone CLI use.

This caused two real bugs, confirmed against the code during investigation:

- **The system-design slides mislabeled the first call as "Claude"** — it was actually
  DeepSeek. Claude only appears once, at the very end, writing the narrative.
- **`evaluator/comparison.py` didn't give `main_1` and `main_2` the same input.**
  `run_comparison()` called `module.run(query)` for both approaches with no `category` —
  so each independently fired its own `pick_candidate_categories()` call. They happened
  to agree on the one case spot-checked (`ice_tray`: both picked
  `Home & Kitchen, Appliances`), but nothing guaranteed that — it was two independent LLM
  calls, not a shared input, undermining the apples-to-apples comparison the eval is
  supposed to be.

**Resolution:** merge both category-selection systems into one module, `cat_selector`,
used everywhere — live app and eval alike — so there's a single source of truth and a
guaranteed-shared input for comparisons.

## Target architecture (as built)

```
cat_selector.select(messages) -> {"clarify": "..."} | {"concept": "..." | null, "categories": [{"category","reason"}, ...]}
    - ONE DeepSeek call, absorbing:
        - chat_engine's clarify-or-proceed decision (ROLE/SECURITY/CONCEPT rules)
        - chat_engine's concept extraction (concept is null for broad-category questions)
        - orchestrator_agent's up-to-2-category picking, now from the real root>level2
          taxonomy instead of the fixed 21-item CATEGORIES list
    - used by BOTH analysis_agent (live, real multi-turn `messages`) and
      evaluator/comparison.py + main_1/main_2's own CLI (wrap the query as a single
      user message)

main_1.run(query, categories) / main_2.run(query, categories)
    - `categories` (plural, list[dict]) is REQUIRED -- no internal fallback, no
      self-picking. evaluator/comparison.py calls cat_selector.select() ONCE per query
      and passes the identical `categories` list into both -- the actual fix for the
      "same input" bug.

analysis_agent.run_chat(messages, mode)  [renamed from chat_engine.py]
    - calls cat_selector.select(messages); on "clarify", returns it directly
    - otherwise calls niche_report(categories, concept, mode) as a plain function call --
      no more TOOLS / tool-call round-trip, since there's nothing left to classify
    - Haiku still narrates from the JSON -- the only Claude call in the system
```

Net live LLM-call count is unchanged (still 1 classification-shaped call + 1 Haiku call)
— the live path never actually ran a second orchestrator call before this either. What
changed is that the logic is unified into one place, and eval now shares real input
between approaches.

## What changed, file by file

- **New `src/retrieval_pipeline/cat_selector.py`** replaces
  `orchestrator_agent.py`. `select(messages)` does clarify/concept/categories in one
  call, reusing `llm_client.py`'s shared DeepSeek client (previously `chat_engine.py`
  built its own separate client, so its classification call's cost/usage was untracked
  by `llm_client`'s accumulator — fixed as a side effect of the move).
- **`main_1.py` / `main_2.py`**: `categories` is now a required parameter, no fallback.
  Their own CLI blocks (`python -m src.retrieval_pipeline.main_1 "<query>"`) call
  `cat_selector.select()` first.
- **New `src/agent_pipeline/analysis_agent.py`** replaces `chat_engine.py`. No more
  `TOOLS`/`CATEGORIES`/`CATEGORY RULES`/`CONCEPT RULES` — that's `cat_selector`'s job now.
  `niche_report()`/`_get_product_subset()`/`_get_velocity_summary()` take `categories`
  (plural) and query ChromaDB with `$in` instead of `$eq`.
- **`main.py`**: import path updated to `analysis_agent`; `run_chat`'s external signature
  is unchanged.
- **`evaluator/comparison.py`**: calls `cat_selector.select()` once per query, passes the
  identical `categories` to both `main_1.run()` and `main_2.run()`.
- **Tests**: `tests/test_chat_engine.py` → `tests/test_analysis_agent.py` (signatures
  updated for `categories`). `tests/test_retrieval_pipeline_category_override.py` →
  `tests/test_retrieval_pipeline_categories_required.py` (rewritten — old premise no
  longer applies once the fallback is gone; now asserts `categories` is required and
  neither main imports any category-picking function).
- **Docs**: `README.md` and `docs/retrieval_pipeline.md` updated throughout (diagrams,
  file tables, prose) to describe `cat_selector`/`analysis_agent` instead of
  `orchestrator_agent`/`chat_engine`.

## Explicitly deferred (not part of this change)
- **Re-running the eval** with the new shared-categories `comparison.py` — old results in
  `evaluator/results/` predate this fix and were produced by each main picking its own
  categories independently. A fresh run is needed before trusting new precision/recall
  numbers, but that's a separate conversation.
- **`docs/counting_pipeline_slides.html`** system-design slide updates (correcting the
  Claude→DeepSeek mislabel, and the slide 3 + "One Figure, Whole Pipeline" merge) — being
  done as a follow-up pass after this plan.

## Verification performed
- `pytest tests/test_analysis_agent.py tests/test_retrieval_pipeline_categories_required.py tests/test_candidates.py -v` — 18 passed.
- `pytest --collect-only -q` across the whole `tests/` directory — all 52 tests collect
  cleanly, confirming no import breakage elsewhere in the suite.
- Manual smoke tests (CLI `main_1`/`main_2` end-to-end with live `cat_selector` call,
  live-app clarify-path) are still recommended before relying on this in production —
  not yet run in this session (would incur real DeepSeek/ChromaDB calls).
