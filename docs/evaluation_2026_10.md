# Structured retrieval: Jev, a rebuilt golden set, and LLM filtering (October 2026)

How the structured pipeline (`main_2`) was made ~100x faster, how its evaluation was rebuilt
after the old labels proved unreliable, and what the new labels show about where the design
wins and loses. Every number here comes from a saved result file listed at the end.

## Summary

- **Stage 1 swapped from DeepSeek to TypeSafe's Jev decision model:** path identification
  went from ~345 s to ~3 s per query (~100x) and ~5x cheaper. On the queries where both
  have saved runs, F1 rose 0.60 → 0.68 against the rebuilt labels.
- **Golden set rebuilt (v2):** the old labels were swept over one category per query with no
  written definitions; v2 is pooled catalog-wide and double-blind labeled against written
  definitions (`evaluator/golden_v2/README.md`).
- **Stage 3 added, LLM title filtering:** a per-title Jev check removes substitutes and
  complements inside trusted paths. Mean precision 0.65 → 0.89, F1 0.60 → 0.71 on 16 scored
  queries, for ~0.27 s and well under $0.001 per query.
- **Measured limit:** path identification judges category *names*, so products sellers filed
  under an unexpected path never reach later stages. That is where recall is lost.

## The pipeline

```
query → cat_selector (1–2 root categories)
  1. Path identification   Jev scores every real category path: confident / ambiguous / drop
  2. Rerank                cross-encoder scores titles in ambiguous paths (confident paths skip)
  3. LLM filtering         Jev asks, per candidate title, "is this product what the query asks for?"
  → count of products that pass stage 3
```

The funnel judges distinct category paths first, which keeps stage 1's cost tied to the
taxonomy rather than the catalog (Home & Kitchen: 629 paths vs. 5,354 items), then spends
per-item judgment only on the candidates that survive.

## 1. Jev for path identification

DeepSeek classified paths by *writing* a JSON row (score, flag, note) per plausible path —
~15,000 output tokens per 60-path batch, the source of ~300 s batches. Jev returns calibrated
probabilities in one pass and bills input tokens only ($0.042 per million), so it needs no
generated text at all.

Two Jev question designs were compared on the six queries of the August benchmark:
a three-way **Choice** (confident / ambiguous / not_match) and a single yes/no **Noul**
probability bucketed by two cutoffs (0.9 / 0.3). They scored within 0.01 F1 of each other;
Noul is ~3x cheaper and is the default. Choice expresses "trust this path without reading
titles" more faithfully, which a single relevance probability cannot.

## 2. Rebuilding the golden set

Auditing the original labels against pipeline runs found:

| Problem | Example |
|---|---|
| Sweep limited to one hinted category | "baby toys" labeled 0 (real ones were in Toys & Games); kids' graduation gowns and a dual-mode gaming mouse in other categories never judged |
| No written definitions | one keyboard + mouse combo counted, another never considered; a solar umbrella counted as a "garden light" |
| Wrong single-item truths | the one "ice tray" was reusable plastic ice cubes; the catalog has no ice trays |

v2 method (`evaluator/golden_v2/`): definitions with include/exclude rules approved before
labeling; candidates pooled catalog-wide from keyword sweeps, embedding search and every saved
system prediction (8,306 candidates); blind batches labeled twice by different models with
Amazon's ESCI scheme (exact / substitute / complement / irrelevant); the 79 items (~1%) where
the passes disagreed on "exact" were settled by a third pass. Outcome: 14 complete queries,
2 sampled (`partial`), 4 with no true match (`negative`).

Scoring rules (`evaluator/metrics_v2.py`): P/R/F1 averaged over complete and partial queries
only; negative queries scored by false positives; substitutes and complements counted
separately so the *kind* of error is visible.

## 3. Results against v2

**Head-to-head** (saved runs exist for all systems on 5 queries; 4 have true matches):

| System | Mean P | Mean R | Mean F1 |
|---|---|---|---|
| Fast mode (vector + rerank) | 0.45 | 0.87 | 0.52 |
| Structured, DeepSeek | 0.61 | 0.60 | 0.60 |
| Structured, Jev Choice | 0.64 | 0.90 | 0.68 |
| Structured, Jev Noul | 0.64 | 0.89 | 0.67 |

**Against fast search on all 16 scored queries** (fast search rerun with the same
categories; `evaluator/results/fast_search_v2_16q.json`; figure
`docs/figures/pipeline_vs_fast_search.png`):

| | Mean P | Mean R | Mean F1 | Mean latency | Cost per query |
|---|---|---|---|---|---|
| Fast search (vector + cross-encoder) | 0.41 | 0.82 | 0.50 | 16.7 s | local only |
| Full pipeline (path id → rerank → LLM filter) | 0.89 | 0.67 | 0.71 | 2.7 s | ~$0.0015 |

Precision more than doubles and F1 rises 42%, but recall drops 15 points: fast search
ranks every title in the category, so it finds misfiled products the path stage drops. On
the three negative queries fast search returned 10 false positives, the pipeline 1.

**Structured + Jev Noul on all 16 scored queries**, before and after stage 3:

| | Mean P | Mean R | Mean F1 | Substitutes / complements counted | Latency |
|---|---|---|---|---|---|
| Stages 1–2 | 0.65 | 0.76 | 0.60 | 65 / 17 | ~2.75 s mean |
| **+ stage 3 (cutoff 0.5)** | **0.89** | 0.67 | **0.71** | 23 / 1 | +0.27 s |

Biggest precision gains from stage 3: dog Halloween costume 0.10 → 1.00, yoga mat
0.33 → 1.00, wireless gaming mouse 0.33 → 0.80, female fitness clothes 0.36 → 0.93.
The 0.5 cutoff was fixed before the run; 0.3 scored higher (F1 0.74) but choosing it
afterwards would be tuning on the test set.

**As a size estimate** (counts bucketed 0–20 / 20–100 / 100–300 / 300+): stages 1–2 put
14 of 19 queries in the right bucket and all 19 within one bucket.

## 4. What the errors show

**Recall: products filed under unexpected paths.** Stage 1 judges path names, so products
in a path whose name sounds wrong never reach stages 2–3:

- 11 Bluetooth speakers filed under "MP3 & MP4 Player Accessories" (path scored 0.24, dropped).
- 8 water fountains sold "for cats & dogs" but filed under *Cats > Fountains* — dropped by the
  matching rule that a dog query never matches a Cats path. The rule is right about the path
  name and wrong about multi-species products inside it.

**Precision: lookalikes inside the right path.** A relevant path can hold non-matches
(all dog costumes under *Dogs > Costumes*; flat scratchers under scratching posts). Raising
the path keep-cutoff did not help — at 0.8, precision on the seven weakest queries rose only
0.31 → 0.40 while recall fell 0.89 → 0.71 — because these items sit in high-scoring paths.
Only a per-title check (stage 3) separates them.

**Stage 3's own errors are query interpretation.** It applies the query literally: it dropped
"Water Fountain for Dogs" for "dog drinking bowl" and birthday decoration kits containing
balloons, which the golden definitions count. Most such drops sat just under the cutoff
(p 0.3–0.49).

## 5. Limitations

- Small sample: 16 scored queries, and the head-to-head rests on 4.
- The DeepSeek arm is from an earlier code version (before the async rewrite) and most of its
  per-item outputs were overwritten, which is why the head-to-head is limited to 4 queries.
- v2 recall is recall against the pooled truth: a true match none of the three pooling
  sources surfaced was never judged.
- Labels were produced by LLM labelers (two models plus a tie-break), not human annotators.
  One pass was found copied and redone; shortcuts some labelers disclosed are listed in the
  v2 changelog.

## 6. Next steps (not done)

1. Make stages 1–2 favor recall now that stage 3 removes false positives: treat wrong-species
   paths with the same function as ambiguous; lower the keep cutoff and rerank threshold.
2. Teach stage 3 bundles ("a kit or set that includes the product counts") and expand
   function-based queries ("drinking bowl" → water bowl, fountain, dispenser).
3. Choose cutoffs on new labeled queries (a tuning/test split), not on these 16.

## Reproduce

```bash
python -m evaluator.golden_v2.build_pools          # candidate pools (local)
python -m evaluator.golden_v2.reconcile --write    # merge label passes -> golden_dataset_v2.json
python -m evaluator.rescore_v2                     # rescore saved runs against v2 (no API calls)
python -m evaluator.title_filter_experiment        # stage 3 before/after (Jev, ~$0.005)
python -m src.retrieval_pipeline.main_2 "yoga mat" # the pipeline itself
```

Result files (`evaluator/results/`): `jev_six_20261005T235031Z.md` (Jev Choice),
`jev_six_noul_20261006T094211Z.md` (Jev Noul), `backend_compare_20261006T133638Z.md`
(Jev Noul, 13 more queries), `rescore_v2_20261006T145231Z.md` (all systems vs. v2),
`title_filter_20261006T163046Z.md` (stage 3).
