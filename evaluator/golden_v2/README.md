# Golden dataset v2

`evaluator/golden_dataset_v2.json` replaces `golden_dataset.json` (v1) for scoring. Same 20
queries, relabeled from scratch with a method built to fix what v1 got wrong.

## Why v1 was replaced

Auditing v1 against pipeline runs (October 2026) found its labels answered a narrower
question than the one the pipelines are scored on:

- **Category-limited sweeps.** Most v1 entries swept one keyword over one hinted category,
  so true matches filed elsewhere were never judged: wireless gaming mice in Video Games,
  kids' graduation gowns in Toys & Games, baby toys in Toys & Games (v1 labeled "baby
  toys" zero because it only looked in Baby Products).
- **No written definitions.** Edge cases were decided inconsistently: one keyboard + mouse
  combo labeled a match while another was never considered; a solar umbrella and a fountain
  pump labeled "solar garden lights" while solar pathway lights were not.
- **Wrong labels on tiny truths.** v1's single "ice tray" match is a pack of reusable
  plastic ice cubes, not a tray. The catalog contains no ice trays.
- **Rejected candidates not kept** for the reconciled entries, so labels could not be audited.

## Method

1. **Definitions first** — `definitions.yaml`: a definition, include/exclude rules and the
   judgment calls (with the default chosen) for every query, approved before any labeling.
   Labels use Amazon's ESCI scheme: **E**xact / **S**ubstitute / **C**omplement /
   **I**rrelevant. Only E is ground truth; S and C are kept so substitutes and complements
   (e.g. mouse chargers) can be measured separately.
2. **Catalog-wide pooling** — `build_pools.py`: the union of (a) titles matching the query's
   keyword regexes across the whole catalog, (b) the top 300 by embedding similarity with
   no category filter, (c) every item any saved pipeline run predicted, plus every item v1
   judged. No API calls.
3. **Blind batches** — `prepare_batches.py`: title and category path only, shuffled, with no
   record of which source or system found an item. `summer dress` and `female fitness
   clothes` pools (3,340 and 2,075) are labeled on a fixed random sample of 500
   (seed 42) and marked `partial`.
4. **Two independent passes** — pass A (Claude Opus sub-agents) and pass B (Claude Sonnet
   sub-agents), no paid API calls, labelers barred from each other's output.
5. **Tie-break** — `reconcile.py`: items where exactly one pass said E go to a third pass
   (`labels/T/`), whose label is final. S/C/I disagreements are recorded, not adjudicated.

## Status per query

| Status | Meaning | Scored on |
|---|---|---|
| `complete` | every pooled candidate judged | precision, recall, F1 |
| `partial` | a random sample judged | precision on judged items only; recall is a lower bound |
| `negative` | no exact match anywhere in the pool | false-positive count only |

Each entry also records `label_agreement` (share of items where passes A and B gave the
same ESCI label) and `e_disputes` (items sent to tie-break).

## Changelog and caveats

- **2026-10-06 — v2 created.** 8,306 candidates judged across 20 queries; passes agreed on
  the E/not-E decision for all but 79 items (about 1%), 41 of them in `female fitness
  clothes`, whose boundary (athleisure vs. workout wear) is the fuzziest.
- **ice tray** is `negative` in v2 (v1 had 1 match, which was mislabeled — see above).
- **car phone mount:** the first pass B file was an exact copy of pass A (the labeler read
  it despite instructions). It was discarded and relabeled in an isolated directory; the
  relabel matched pass A exactly as well. The labels were spot-checked by hand
  (golf-cart mounts S, adhesive pads C, Starlink/shower/airplane holders I) and accepted —
  car mounts are an unusually unambiguous category.
- **Labeler shortcuts disclosed by the labelers themselves:** pass B on `graduation_gown`
  filtered obvious dresses and stoles before reading part of its batch; pass B on
  `gluten_free_pasta__2` scanned items 313–441 by category. In both cases the E decisions
  agreed with pass A.
- **Pool ceiling:** an exact match that none of the three sources surfaced was never judged.
  Recall against v2 is therefore recall against the pooled truth, the standard limitation
  of pooled IR test collections.
- **Rescoring rule:** never edit labels after looking at a pipeline run being compared. If a
  label must change, bump the version and rescore every system against it.
