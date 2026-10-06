"""
Jev (TypeSafe System One) backend for classify_agent -- an experiment against DeepSeek.

classify_agent's DeepSeek prompt makes the model WRITE a JSON row per plausible path
(score, confident, note): ~15k completion tokens per 60-path batch, which is where the
~300 s batches come from. Jev never writes text. Each path becomes one Choice question
over three answers that mirror classify_agent's buckets, all asked in one request and
evaluated in parallel against the same state (the query + the matching rules).

Answer -> the {id, score, confident} shape classify_agent already buckets:
    confident  -> score 10, confident True   (confident_match)
    ambiguous  -> score 5,  confident False  (ambiguous_match)
    not_match  -> omitted                    (not_match)
The full probability distribution is kept on each row ("probabilities") so thresholds
can be tuned later instead of trusting the argmax.

JEV_MODE=noul asks a single yes/no question per path instead -- "does this path match
the query?" -- and buckets its one probability p with two cutoffs:
    p >= JEV_CONFIDENT_CUTOFF (0.9)  -> confident_match
    p >= JEV_KEEP_CUTOFF (0.3)       -> ambiguous_match
    otherwise                        -> not_match
The cutoffs are starting guesses, to be tuned on the saved probabilities.

Default backend for classify_agent (CLASSIFIER_BACKEND=deepseek for the old one), default
JEV_MODE=noul -- the configuration measured against golden v2. Needs TYPESAFE_API_KEY.
Price: input tokens only ($0.042 per Mtok for jev-1.13.0); output is free.
"""

import asyncio
import os

from langsmith import traceable
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

# Pinned rather than "jev-latest": an alias can move to a new model mid-experiment.
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")
JEV_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000
BATCH_DEADLINE_S = 60
JEV_MODE = os.environ.get("JEV_MODE", "noul")
JEV_CONFIDENT_CUTOFF = float(os.environ.get("JEV_CONFIDENT_CUTOFF", "0.9"))
JEV_KEEP_CUTOFF = float(os.environ.get("JEV_KEEP_CUTOFF", "0.3"))

# The same rules as classify_agent.PROMPT_TEMPLATE, moved into state so each question
# can reference them instead of repeating them 60 times.
MATCHING_RULES = """\
Match the concept, function or purpose the user actually wants, not the literal noun they
typed -- the query's specific word is one example of the thing, not a strict requirement
on shape, wording, or category label.
Never relax WHO the product is for. Species, audience, age and gender are part of the
request itself -- a "dog" query does not match a path under Cats, and a "kids" query does
not match an adult path, however identical the product's function.
Examples:
- "dog drinking bowl" confidently means "Pet Supplies > Dogs > Feeding & Watering Supplies >
  Fountains" (the intent is drinking, not literally bowl-shaped), but NOT "Pet Supplies >
  Cats > Feeding & Watering Supplies > Fountains" (right function, wrong animal).
- "kids costumes" confidently means "Dress Up & Pretend Play > Costumes" even without the
  word "kids", but NOT a general "Cosplay Apparel" path, which is not specific to children.
- A generic "Bowls" path could hold food bowls or water bowls, and a generic "Boys" clothing
  path could hold a costume or anything else: those are ambiguous, not confident."""

CRITERIA = {
    "confident": "The path's name unambiguously means the query's concept, for the right "
                 "audience; every product filed there should match.",
    "ambiguous": "The path is plausibly relevant but could hold both matching and "
                 "non-matching products, so the product titles would need checking.",
    "not_match": "The path is irrelevant to the query, or is for the wrong species, "
                 "audience, age or gender.",
}

_ROW = {"confident": (10, True), "ambiguous": (5, False)}

_usage = {"calls": 0, "input_tokens": 0}


def reset_usage() -> None:
    _usage.update(calls=0, input_tokens=0)


def get_usage() -> dict:
    return {**_usage, "cost_usd": round(_usage["input_tokens"] * JEV_PRICE_PER_INPUT_TOKEN, 6)}


def _question(path: str) -> Choice | Noul:
    if JEV_MODE == "noul":
        return Noul(instructions=(f'Category path: "{path}". Does this real Amazon category path '
                                  "match the user's `query`, following `matching_rules`?"))
    return Choice(
        instructions=(f'Category path: "{path}". Does this real Amazon category path match '
                      "the user's `query`, following `matching_rules`?"),
        criteria=CRITERIA,
    )


@traceable(run_type="llm", name="jev.system_one",
           metadata={"ls_provider": "typesafe", "ls_model_name": JEV_MODEL})
async def _system_one(client: AsyncTypeSafeClient, query: str, batch: dict[str, str]) -> dict:
    """
    The Jev call itself, traced as an "llm" run so LangSmith records its latency and token
    counts the same way it does for DeepSeek calls (the OpenAI wrapper does that for those;
    TypeSafe's SDK has no LangSmith wrapper, so usage_metadata is returned explicitly).
    evaluator/langsmith_timing.py prices runs whose ls_model_name starts with "jev".
    """
    async with asyncio.timeout(BATCH_DEADLINE_S):
        response = await client.system_one(
            state={"query": query, "matching_rules": MATCHING_RULES},
            questions={pid: _question(path) for pid, path in batch.items()},
        )
    usage = response.usage
    return {
        "answers": {pid: a.model_dump() for pid, a in response.answers.items()},
        "model": response.model,
        "usage_metadata": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                           "total_tokens": usage.input_tokens + usage.output_tokens},
    }


@traceable(run_type="chain", name="jev_scorer.score_batch")
async def score_batch(client: AsyncTypeSafeClient, query: str, batch: dict[str, str]) -> list[dict]:
    """
    One Jev request: a Choice question per path, answered in parallel.

    In: an open client, query text, batch {path_id: category_path}
    Out: choice mode -- {id, score, confident, note, probabilities} for paths not judged
         not_match; noul mode -- {id, jev_label, p_match} for EVERY path (so the
         probability is saved even below the keep cutoff), plus score/confident/note on kept ones
    """
    result = await _system_one(client, query, batch)
    _usage["calls"] += 1
    _usage["input_tokens"] += result["usage_metadata"]["input_tokens"]

    rows = []
    for pid, answer in result["answers"].items():
        if answer["type"] == "noul":
            p = answer["noul"]
            label = ("confident" if p >= JEV_CONFIDENT_CUTOFF
                     else "ambiguous" if p >= JEV_KEEP_CUTOFF else "not_match")
            rows.append({"id": pid, "jev_label": label, "p_match": p})
            if label != "not_match":
                score, confident = _ROW[label]
                rows[-1].update(score=score, confident=confident, note=f"jev noul p={p:.2f}")
            continue
        if answer["choice"] == "not_match":
            continue
        score, confident = _ROW[answer["choice"]]
        rows.append({
            "id": pid, "score": score, "confident": confident,
            "note": f"jev {answer['choice']} (confidence {answer['confidence']:.2f})",
            "probabilities": dict(answer["probabilities"]),
        })
    return rows


def new_client() -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(model=JEV_MODEL)
