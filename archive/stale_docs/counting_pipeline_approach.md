# Counting pipeline: structured filter + scoped reranker

**Problem:** "how many X" queries against the product catalog were reporting a bare `min(200, ...)` retrieval count regardless of the true answer — 200 for "summer female dresses" (true ~5,000) and 200 for "dog drinking bowls" (true ~10). The root cause: counting was implemented as `len(retrieved subset)`, which is a retrieval-size artifact, not a count. Vector similarity, BM25, and cross-encoder reranking were all tested as substitutes and all failed the same way — no stable, query-independent threshold exists on any of their score distributions ("the elbow method is unreliable" held up empirically, repeatedly).

**What works instead:** exact structured attribute filtering (`target=dog AND function=drinking_equipment`, evaluated via `ChromaDB.get(where=...)` — no top-K, no threshold) beat every retrieval-based approach tested. But it has its own gap: a taxonomy that assigns **one** function per product systematically drops dual-purpose products (a combo food+water bowl gets tagged `feeding_equipment`, never `drinking_equipment`). The fix that recovered the most accuracy: let an LLM widen the query's structured-filter scope to the 2-3 *adjacent* attribute values a product could plausibly have been mistagged into, then run a classifier (cross-encoder reranker) **only** on that narrow residual — never on the whole catalog, never skipped entirely.

Diagram of the mechanism: **https://claude.ai/code/artifact/9aac8dc0-a412-4dd7-b6ce-23c807a3d6c1**

---

## 1. How the LLM's scope expansion is kept to 2-3, not unbounded

The first version of this prompt just said *"pick as many function values as are genuinely justified, don't pad the list"* — free-form generation with no explicit ceiling. It happened to return exactly 2 for the first query tested, but nothing structurally prevented it from returning 6 on a different query. That's not a reliable engineering constraint, it's a hope.

**The fix: don't ask the LLM to decide how many. Ask it to score every candidate, then enforce the cap in code.**

```
Score EVERY value 0-10 for how likely a genuinely-matching product could carry
that tag, accounting for the fact that dual-purpose products only get ONE tag
at ingestion time (so a product might be genuinely relevant but filed under
an adjacent function).

Return ONLY a JSON object mapping every taxonomy key to an integer 0-10.
```

Then, deterministically in code (not left to the model):

```python
ranked = sorted(scores.items(), key=lambda kv: -kv[1])
primary = ranked[0][0]                                    # top score = the strict match
extras  = [k for k, s in ranked[1:] if s >= 6][:2]         # up to 2 more, only if score >= 6
expanded = [primary] + extras                               # hard cap: 3 total, always
```

This gives two independent guarantees a free-form list can't:
- **A hard ceiling** (`[:2]`) — even if five categories score above the threshold, only the top 2 extras are taken. The cap is a `list` slice, not a prompt instruction the model could ignore.
- **A relevance floor** (`score >= 6`) — a category that only plausibly applies (scored 5) is excluded even though it's not the ceiling that stopped it. This is what kept `carrier_travel` (scored 5) out of the "dog drinking bowl" expansion while `feeding_equipment` (scored 8) got in — a decision that's now auditable from the score, not just "the LLM felt like it."

Worth noting directly: **when nothing scores above the floor, the mechanism correctly expands to nothing.** For "ice tray" against the Home & Kitchen taxonomy, only `cookware` scored ≥6 (`small_appliance` scored 3, everything else 0) — so `expanded == [primary]`, no residual, no reranker step at all. The mechanism doesn't force a 2-3 expansion on every query; it produces one only when the score distribution actually supports it.

---

## 2. Results across three categories

Ground truth for all three was built the same way: keyword-swept candidate pool (checked with 2+ independent regexes for synonym coverage), then manually judged by reading each candidate title directly — no retrieval signal involved, to avoid circularity.

### 2a. Pet Supplies — "dog drinking bowl" (n=681, ground truth=11)

| Stage | Output |
|---|---|
| LLM decomposition | primary: `target=dog, function=drinking_equipment` · expanded residual: `target∈{dog,multi_pet} × function∈{drinking,feeding}` |
| Box 2: confident match | 7 |
| Box 2: residual | 11 |
| Box 2: confident non-match | 663 (audited: 0 false negatives) |
| Box 4: residual passing reranker (`score>0`) | 7 |
| **Box 5: final aggregate** | **14** |

**Precision 71.4%, Recall 90.9%, F1 80.0%** — best of every approach tested (beat plain attribute-filter-alone's F1 of 77.8%, and beat every retrieval-only combination by 20-40 points). This is the success case: the taxonomy's dual-purpose gap was real (4 of 11 true items mistagged as `feeding_equipment`/`carrier_travel`), and the scoped residual mechanism recovered 3 of those 4 without meaningfully hurting precision.

### 2b. Baby Products — "baby toys" (n=212, ground truth=8)

| Stage | Output |
|---|---|
| LLM decomposition | primary: `attr2=toys` · expanded residual: `attr2∈{toys,sleeping,bathing}` |
| Box 2: confident match | 10 |
| Box 2: residual | 58 |
| Box 4: residual passing reranker | 1 |
| **Box 5: final aggregate** | **11** |

**Precision 27.3%, Recall 37.5%, F1 31.6%** — *worse* than the plain attribute-filter baseline (F1 33.3%). Two independent failures, neither fixed by the scoped-residual mechanism:

1. **The confident-match bucket itself was already low-precision** (3 of 10 true) — the existing `toys` tag conflates actual toys with **toy storage** ("Toy Box, Toy Storage Organizer," "Stuffed Animal Storage Zoo Tower"). This is a tagging-granularity problem sitting entirely inside Box 2, which the residual/reranker mechanism never touches — it only ever adds to Box 2's output, never re-examines it.
2. **The reranker didn't recognize the true dual-purpose items either.** The 4 ground-truth crib mobiles were correctly identified by the LLM as residual candidates (all tagged `sleeping`, not `toys` — a real instance of the same taxonomy gap as Pet Supplies) — but the cross-encoder scored all 4 strongly negative (-5.7 to -9.5), because "Butterfly Mobile for Crib" has weak lexical/semantic overlap with "baby toys" even though it functions as one. Only 1 of 58 residual items passed. The right items were in the right pool; the classifier just didn't recognize them.

### 2c. Home & Kitchen — "ice tray" (n=5,354, ground truth=0)

[to be completed once the exhaustive tagging pass finishes]

---

## 3. What generalizes and what doesn't

- **Structured filtering beats every retrieval-only signal, every time it was tested.** Not once did BM25, vector similarity, a reranker, or any fusion of them beat a plain attribute filter on F1 — even at 100% recall, precision on unstructured signals was low enough (1.7%-38%) to tank the balanced score.
- **The scoped-residual pattern (expand → filter → classify only the delta) is not a universal win — it's conditional on the classifier actually recognizing the residual's true positives.** It added +2.2 F1 points in Pet Supplies and *cost* 1.7 F1 points in Baby Products. The difference wasn't the mechanism — it was whether the reranker's semantic judgment agreed with the ground truth on the specific items in the residual.
- **A precision problem inside the confident-match bucket is invisible to this whole architecture.** Box 2's output is trusted absolutely; nothing downstream ever reconsiders it. If the base tagging is systematically coarse (toy boxes tagged as toys) rather than just missing dual-purpose splits, the residual/reranker fix doesn't apply — that's a labeling-granularity fix, not a query-time fix.
- **A ground truth of 0 is a legitimate, useful test case**, not a degenerate one — it directly tests whether the pipeline reports "none found" or hallucinates matches from fuzzy retrieval when nothing genuinely exists.
