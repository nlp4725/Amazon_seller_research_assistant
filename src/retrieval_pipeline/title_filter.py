"""
Stage 3 of main_2: LLM title filtering with TypeSafe's Jev.

main_2 is a funnel -- path identification (classify_agent), then rerank (cross-encoder on
the ambiguous paths), then this. The first two stages never ask "is this product itself what
the query asks for": path identification judges category NAMES, so every item in a trusted
path counts (all dog costumes for "dog Halloween costume"), and the cross-encoder judges
text SIMILARITY, so "Wireless Mouse Charger" passes for "wireless gaming mouse". Rescored
against golden v2, most remaining false positives were exactly these substitutes and
complements sitting inside kept paths.

This stage asks Jev one yes/no question (a Noul) per predicted title, with the query and
GENERIC rules only -- never per-query definitions -- and keeps items with
p >= TITLE_FILTER_CUTOFF. Measured on the 16 scored golden v2 queries (evaluator/results/
title_filter_*.md): precision 0.65 -> 0.89, recall 0.76 -> 0.67, F1 0.60 -> 0.71, for
~0.27 s and well under $0.001 per query. The cutoff (0.5) was fixed before that run; a lower
one scored better there but would be tuning on the test set.

It only removes items -- it cannot recover products earlier stages dropped.

Switch: TITLE_FILTER=on|off (default on). Needs TYPESAFE_API_KEY.
"""

import asyncio
import os

from langsmith import traceable
from typesafe_sdk import AsyncTypeSafeClient, Noul

from src.retrieval_pipeline.jev_scorer import JEV_MODEL

TITLE_FILTER = os.environ.get("TITLE_FILTER", "on") == "on"
TITLE_FILTER_CUTOFF = float(os.environ.get("TITLE_FILTER_CUTOFF", "0.5"))
BATCH = 250          # titles per Jev request -- well inside its 64k-token budget
DEADLINE_S = 60

GENERIC_RULES = """\
The product must BE the item the query names -- not an accessory, part, refill, charger,
case, mat, stand or add-on for it, and not a different product that is merely related.
Every qualifier in the query must hold for the product: audience or age (kids, baby),
species (dog, cat), occasion (Halloween, birthday), feature (wireless, solar, left-handed,
gluten-free), season or use (summer, fitness).
A product sold for several audiences or species counts if it includes the one in the query
(e.g. "for cats and dogs" satisfies a dog query).
Judge from the title. If the title does not state or clearly imply a required qualifier,
the answer is no."""


@traceable(run_type="llm", name="jev.system_one",
           metadata={"ls_provider": "typesafe", "ls_model_name": JEV_MODEL})
async def _ask(client: AsyncTypeSafeClient, query: str, titles: dict[str, str]) -> dict:
    """One Jev request: a Noul per title. Traced as an "llm" run with usage_metadata so
    evaluator/langsmith_timing.py prices it like every other model call."""
    async with asyncio.timeout(DEADLINE_S):
        response = await client.system_one(
            state={"query": query, "rules": GENERIC_RULES},
            questions={key: Noul(instructions=(f'Product title: "{title}". Is this product itself what a '
                                               "shopper searching `query` wants, following `rules`?"))
                       for key, title in titles.items()},
        )
    u = response.usage
    return {"p": {k: a.noul for k, a in response.answers.items()},
            "usage_metadata": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                               "total_tokens": u.input_tokens + u.output_tokens}}


@traceable(run_type="chain", name="title_filter.score_titles")
async def ascore_titles(query: str, titles: dict[str, str]) -> dict[str, float]:
    """
    In: query, {key: title} for the items to judge (keys are any unique ids, e.g. asins)
    Out: {key: p_match}. Batches run concurrently; a failed batch raises, so a caller never
         silently treats unjudged items as rejected.
    """
    items = list(titles.items())
    async with AsyncTypeSafeClient(model=JEV_MODEL) as client:
        parts = await asyncio.gather(*(_ask(client, query, dict(items[i:i + BATCH]))
                                       for i in range(0, len(items), BATCH)))
    return {k: p for part in parts for k, p in part["p"].items()}


def apply(query: str, titles_found: list[dict]) -> list[dict]:
    """
    Filter main_2's predicted matches in place of trusting them.

    In: query, titles_found (main_2 schema: dicts with asin, title, is_match)
    Out: the same list with, on every item that was a match before this stage:
         title_filter_p (Jev's probability), matched_before_filter=True, and is_match
         re-set to p >= TITLE_FILTER_CUTOFF. Non-matches pass through untouched.
    """
    matched = {i: t["title"] for i, t in enumerate(titles_found) if t.get("is_match")}
    if not matched:
        return titles_found
    p = asyncio.run(ascore_titles(query, {str(i): title for i, title in matched.items()}))
    for i in matched:
        t = titles_found[i]
        t["title_filter_p"] = round(p[str(i)], 4)
        t["matched_before_filter"] = True
        t["is_match"] = p[str(i)] >= TITLE_FILTER_CUTOFF
    return titles_found
