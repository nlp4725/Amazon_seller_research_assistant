"""
Deterministic grounding audit for the report-writing step (analysis_agent.write_report).

evaluator/metrics.py asks "did retrieval find the right products". evaluator/stage_metrics.py
asks "which stage lost them". This module asks the last question in the chain: given a
correct niche_report() dict, did Haiku's prose stay faithful to it, or did it invent things.

WHY THIS IS NOT RAGAS
Ragas faithfulness decomposes an answer into atomic claims and has an LLM judge each one
against the context. That is the right shape when the context is retrieved prose. Here the
context is a JSON of figures this repo computed itself, so most "is this claim supported"
questions are exact lookups -- an LLM judge would add cost, latency and a nondeterministic
score to a check that can be decided with certainty. Worse, ragas would actively misfire:
the report legitimately restates precomputed fields, and a generic claim-decomposer flags
whatever it cannot re-derive from the context, so the score would be dominated by false
positives rather than by real hallucinations.

So this is the deterministic tier, and it is deliberately the whole of the numeric and
entity surface. What it CANNOT check is the genuinely semantic residue -- whether a cluster
label ("Electric/Automated Tools") is a fair reading of that cluster's representative_titles,
and whether a seller's inferred "niche focus" is supported by their recent_titles. Those are
the only two judgements the prompt actually delegates to the model, and they need a judge.
Treat a clean run here as "no fabricated figures", NOT as "the report is accurate".

WHAT COUNTS AS GROUNDED
Every number the report states must appear in the JSON. Numbers are harvested from the JSON
recursively, and from INSIDE its strings as well as from its numeric values -- product
titles carry figures ("3.4Gal Dog Water Fountain", "13L Stainless Steel") that the report
may legitimately quote, and treating those as ungrounded would bury real findings in noise.
CORPUS_CONSTANTS covers the handful of figures that come from the system prompt rather than
the data (the catalog size), which are otherwise unattributable.

This tier only became meaningful once the arithmetic moved out of the model: while the
prompt still asked Haiku to compute growth rates and projections itself, "every number must
appear in the JSON" was false by design and could not be asserted.

CLI:
    python -m evaluator.faithfulness                    # default query set
    python -m evaluator.faithfulness "dog water fountain" "Pet Supplies"
"""

import json
import re
import sys

from src.agent_pipeline.analysis_agent import niche_report, write_report

# Figures that legitimately appear in a report but originate in the SYSTEM prompt rather
# than the report JSON, so they can never be traced to the data.
CORPUS_CONSTANTS = {61635.0}

REQUIRED_SECTIONS = ["SEARCHED", "LAUNCH VOLUME", "THEME TRENDS", "TOP SELLERS", "BOTTOM LINE"]

# Claims the prompt explicitly forbids: this dataset covers new launches only, so it cannot
# support statements about the whole competitive field.
#
# This is the ONE heuristic check in this module -- everything else is an exact lookup. The
# prompt also tells the model to be honest about data limits, so the compliant report SAYS
# these words, in the negative: "we cannot assess how underserved these segments truly are"
# is correct behaviour, not a violation. A bare keyword scan flagged exactly those sentences
# and nothing else. NEGATION_CUES suppresses them; a claim that slips through is worth a
# human look rather than an automatic failure, so these are reported separately from the
# exact checks (see audit()'s "review" list).
BANNED_PHRASES = ["underserved", "least competition", "no competition", "least competitive"]
NEGATION_CUES = ["cannot", "can't", "unable", "not ", "without", "don't", "do not",
                 "no way to", "insufficient", "would need", "requires"]

VALID_TAGS = {"RISING", "DECLINING", "EMERGING", "STABLE", "INSUFFICIENT_DATA"}

# A number, optionally signed, comma-grouped, decimal. Leading $ and trailing % are left to
# the surrounding text -- only the value is compared. The sign MUST be part of the match:
# growth_pct_2024_2025 is stored negative (-14.3) and the prose renders it "-14.3%", so
# matching the magnitude alone reported every genuine decline as a fabrication.
# U+2212 MINUS SIGN is included because the model emits typographic minus, not hyphen.
_NUMBER = re.compile(r"[-\u2212]?\d[\d,]*(?:\.\d+)?")

# "3. **Tomxcute**" -- an ordered-list marker, not a claim about the data.
_ORDINAL = re.compile(r"(?m)^\s*[-\u2212]?\d+\.\s")
_SELLER_ID = re.compile(r"\bA[A-Z0-9]{9,13}\b")
_TAG = re.compile(r"\b(" + "|".join(VALID_TAGS) + r")\b")

_TOLERANCE = 0.01


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("\u2212", "-"))
    except ValueError:
        return None


def _numbers_in(obj) -> set[float]:
    """
    Every number reachable in the report JSON -- values, numeric-looking dict keys (by_year
    is keyed by year), and numbers embedded in strings (product titles).
    """
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(float(obj))
    elif isinstance(obj, str):
        found.update(v for v in (_to_float(m) for m in _NUMBER.findall(obj)) if v is not None)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found |= _numbers_in(k) | _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _numbers_in(v)
    return found


def _grounded(value: float, allowed: set[float]) -> bool:
    """True if `value` matches an allowed figure, or is its rounded form.

    The report rounds for readability ("$51.35" from 51.3548, "1,900%" from 1900.0), so an
    exact set membership test would flag correct prose. Comparison is absolute within
    _TOLERANCE, plus a check against each allowed value rounded to 0 and 1 decimals.
    """
    for a in allowed:
        if abs(a - value) <= _TOLERANCE:
            return True
        if round(a) == value or round(a, 1) == value:
            return True
    return False


def audit(report: dict, narrative: str) -> dict:
    """
    Check one narrative against the report dict it was written from.

    In: niche_report() output, the prose write_report() produced from it
    Out: {"findings": [{kind, detail}], "counts": {kind: n}, "checked": {...}, "passed": bool}
    """
    findings: list[dict] = []
    allowed = _numbers_in(report) | CORPUS_CONSTANTS

    # 1. every stated figure must trace to the JSON. Findings carry the surrounding clause:
    # a bare "14.3 is ungrounded" is not actionable, and most of the judgement about whether
    # a hit is a real fabrication or a harmless prose number lives in that context.
    ordinal_spans = {m.start() for m in _ORDINAL.finditer(narrative)}
    stated = []
    for m in _NUMBER.finditer(narrative):
        value = _to_float(m.group())
        if value is None:
            continue
        if any(abs(m.start() - s0) <= 4 for s0 in ordinal_spans):
            continue  # list numbering, not a figure
        stated.append(value)
        if not _grounded(value, allowed):
            lo, hi = max(0, m.start() - 60), min(len(narrative), m.end() + 60)
            context = " ".join(narrative[lo:hi].split())
            findings.append({"kind": "ungrounded_number",
                             "detail": f"{value:g} — ...{context}..."})

    # 2. no invented sellers
    real_sellers = {s["seller_id"] for s in report.get("top_sellers", [])}
    for sid in set(_SELLER_ID.findall(narrative)):
        if sid not in real_sellers:
            findings.append({"kind": "invented_seller", "detail": sid})

    # 3. trend tags must be ones the data actually assigned. A tag is precomputed, so a
    #    mismatch means the model overrode a computed verdict -- the failure precomputing
    #    was meant to eliminate, and the reason this check exists at all.
    real_tags = {t.get("trend_tag") for t in report.get("trend_by_theme", [])}
    for tag in set(_TAG.findall(narrative)):
        if tag not in real_tags:
            findings.append({"kind": "wrong_trend_tag",
                             "detail": f"{tag} stated; computed tags were {sorted(t for t in real_tags if t)}"})

    # 4. claims the dataset cannot support
    low = narrative.lower()
    review: list[dict] = []
    for phrase in BANNED_PHRASES:
        start = 0
        while (i := low.find(phrase, start)) != -1:
            start = i + len(phrase)
            window = low[max(0, i - 90):i]  # clause leading up to the phrase
            if any(cue in window for cue in NEGATION_CUES):
                continue  # disclaiming the assessment, which the prompt asks for
            lo, hi = max(0, i - 80), min(len(narrative), i + len(phrase) + 80)
            review.append({"kind": "banned_claim", "phrase": phrase,
                           "detail": f"{phrase!r} — ...{' '.join(narrative[lo:hi].split())}..."})

    # 5. structure
    for section in REQUIRED_SECTIONS:
        if section not in narrative.upper():
            findings.append({"kind": "missing_section", "detail": section})

    counts: dict[str, int] = {}
    for f in findings:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1

    return {
        "findings": findings,
        "review": review,   # heuristic hits -- read them, do not gate on them
        "counts": counts,
        "checked": {"numbers_stated": len(stated), "allowed_values": len(allowed),
                    "sellers_named": len(set(_SELLER_ID.findall(narrative)))},
        "passed": not findings,
    }


def run_case(query: str, category: str, mode: str = "simple") -> dict:
    """Build a real report for (query, category), narrate it, and audit the result."""
    report = niche_report(categories=[{"category": category}], concept=query, mode=mode)
    if report.get("error") or not report.get("trend_by_theme"):
        return {"query": query, "category": category, "skipped": report.get("error", "no data")}
    narrative = write_report([{"role": "user", "content": f"{query} niche in {category}"}], report)
    result = audit(report, narrative)
    return {"query": query, "category": category, "narrative": narrative, **result}


DEFAULT_CASES = [
    ("dog water fountain", "Pet Supplies"),
    ("dog grooming", "Pet Supplies"),
    ("ice cube tray", "Home & Kitchen"),
]


if __name__ == "__main__":
    cases = [(sys.argv[1], sys.argv[2])] if len(sys.argv) >= 3 else DEFAULT_CASES

    results = []
    for query, category in cases:
        r = run_case(query, category)
        results.append(r)
        if "skipped" in r:
            print(f"\n=== {query!r} / {category} -- SKIPPED ({r['skipped']})", flush=True)
            continue
        status = "PASS" if r["passed"] else f"FAIL ({len(r['findings'])} findings)"
        print(f"\n=== {query!r} / {category} -- {status}", flush=True)
        print(f"    checked: {r['checked']}")
        for f in r["findings"]:
            print(f"    [{f['kind']}] {f['detail']}")
        for f in r["review"]:
            print(f"    (review) [{f['kind']}] {f['detail']}")

    scored = [r for r in results if "skipped" not in r]
    print(f"\n{sum(r['passed'] for r in scored)}/{len(scored)} reports clean "
          f"(deterministic tier only -- cluster labels and seller-niche inferences are NOT checked)")
