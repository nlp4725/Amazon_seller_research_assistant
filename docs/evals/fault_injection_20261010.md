# Fault injection, 2026-10-10

**13 of 13 simulated outages still gave the user an answer** (worst response 3.2 s on a laptop, at test-scale deadlines).

**Update 2026-11-09 (Cloud Build CPU):** the three Jev path-classification outages (rows 4-6) take ~13-15 s on a single-threaded cloud CPU, because their fallback is vector search + cross-encoder rerank over the catalog (the same cost as the 16.7 s vector-search baseline). They still return a labelled partial answer; the harness now allows 25 s for those three rows and 6 s for the rest. Production (Cloud Run, OMP_NUM_THREADS=1) should expect the same ~15 s for that fallback. Speeding up the fallback is separate work.

Query: "dog water fountain". Healthy run: 6 matches. Every provider is a fake; deadlines are 1 s at test scale, a hung fake waits 8 s, and the user must have an answer within 6 s. Pass rules are in tests/fault_injection/harness.py.

| # | Outage | Result | Outcome | Time (s) |
|---|---|---|---|---|
| 1 | DeepSeek (category selection) returns 5xx | PASS | full answer: 6 matches | 0.2 |
| 2 | DeepSeek (category selection) hangs | PASS | full answer: 6 matches | 1.2 |
| 3 | DeepSeek (category selection) returns invalid JSON | PASS | full answer: 6 matches | 0.2 |
| 4 | Jev (path classification) returns 5xx | PASS | partial results, labelled (category matching was unavailable, so this is a broader similarity search and may include related products): 26 matches | 2.0 |
| 5 | Jev (path classification) hangs | PASS | partial results, labelled (category matching was unavailable, so this is a broader similarity search and may include related products): 26 matches | 3.2 |
| 6 | Jev (path classification) drops half its answers | PASS | partial results, labelled (category matching was unavailable, so this is a broader similarity search and may include related products): 26 matches | 1.2 |
| 7 | Jev (title filter) returns 5xx | PASS | partial results, labelled (product titles were not checked, so lookalike products may be included): 10 matches | 0.3 |
| 8 | Jev (title filter) hangs | PASS | partial results, labelled (product titles were not checked, so lookalike products may be included): 10 matches | 2.3 |
| 9 | Jev (title filter) drops half its answers | PASS | partial results, labelled (product titles were not checked, so lookalike products may be included): 10 matches | 0.3 |
| 10 | Claude Haiku (summary writer) returns 5xx | PASS | dashboard shown; summary failed with HTTP 503 and a retry button | 0.2 |
| 11 | Claude Haiku (summary writer) hangs | PASS | dashboard shown; summary failed with HTTP 503 and a retry button | 1.2 |
| 12 | Firestore (cache) reads and writes fail | PASS | full answer: 6 matches | 0.2 |
| 13 | Firestore (cache) hangs | PASS | full answer: 6 matches | 2.2 |

## Before the layer H changes (same harness, same day)
1 of 13 passed. Only the cache outage was survived:
- DeepSeek failures and invalid JSON gave an HTTP 500, and a hang never answered
- Jev path-classification failures showed "No matching products found", a silent wrong answer
- Jev title-filter failures gave an HTTP 500
- Haiku failures gave a bare 500, and a hang never answered
- A hung Firestore cost about 30 s per search

## What changed
- Category selection: a hard deadline per attempt, then a fallback to Claude Haiku with the same prompt and the same validation
- Jev: one retry with backoff and jitter. A response missing answers counts as a failure, not as "no"
- Path classification failing outright falls back to vector search plus rerank, labelled as broader
- A title-filter failure returns the unfiltered matches, labelled
- Writers: async with a deadline. A failure or an empty reply returns 503 + Retry-After, and the page shows a retry button
- Cache: a 2 s deadline on Firestore calls, plus a circuit breaker that skips Firestore for 30 s after a failure
- Degraded results carry `degraded_reason`, the page shows a "Partial results" note, and they are never cached

## Caveats
- Deadlines here are at test scale. The production values (30 s category step, 45 s writers, 60 s per Jev batch, 2 s cache) are safety nets, not measured values; layer J replaces them with measured p99 × 2
- Both category providers down at once still returns a clear 503, not an answer. That case is not counted above
- One query, one fake per provider. These scenarios cover how failures are handled, not traffic
