"""
Fault injection: run the real API (main.py's Flask app, the real retrieval pipeline, the local
Chroma data and cross-encoder) with every EXTERNAL dependency replaced by a fake that can be
told to fail. Nothing here calls a paid API.

Faked boundaries:
  - DeepSeek   cat_selector's client (category selection)
  - Jev paths  TypeSafe client used by jev_scorer (path classification)
  - Jev titles TypeSafe client used by title_filter
  - Haiku      the report writers' client
  - Firestore  the result cache's backend

Failure modes: "error" (the provider returns a 5xx), "hang" (no answer until long past every
deadline), "bad" (a response the code can't use: invalid JSON, or answers missing).

A scenario PASSES when the user still gets something they can act on, quickly:
  - analyze-side failures: HTTP 200 with dashboard data within ANSWER_BOUND_S, and if the
    result differs from the healthy one, it is labelled degraded (no silent wrong answers);
    then, with the dependency healthy again, the same query returns the full result
    (proof the degraded answer was never cached)
  - writer-side failures: the dashboard still answers, and the writer endpoint responds
    (text, or a JSON error the page shows with a Retry button) within ANSWER_BOUND_S

Deadlines are shrunk to test scale (TEST_DEADLINE_S) so a "hang" takes seconds, not minutes;
the ratios are what matter. Used by tests/fault_injection/test_fault_injection.py and
evaluator/fault_injection.py (which writes docs/evals/fault_injection_YYYYMMDD.md).
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

os.environ["LANGSMITH_TRACING"] = "false"   # fakes must not be traced into the live project
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("CACHE_BACKEND", "memory")

QUERY = "dog water fountain"
CATEGORIES = [{"category": "Pet Supplies", "reason": "fault-injection fixture"}]

TEST_DEADLINE_S = 1.0   # every model-call deadline, at test scale
HANG_S = 8.0            # how long a hung fake waits before answering
ANSWER_BOUND_S = 6.0    # the user must have an answer by then


class FakeServerError(Exception):
    """Stands in for a provider 5xx / connection reset."""


# ---------- fake providers ----------

class Behaviour:
    """What each fake does right now: "ok", "error", "hang" or "bad"."""

    def __init__(self):
        self.mode = {"deepseek": "ok", "jev_paths": "ok", "jev_titles": "ok", "haiku": "ok", "firestore": "ok"}
        self.calls = {k: 0 for k in self.mode}

    def reset(self):
        for k in self.mode:
            self.mode[k] = "ok"
            self.calls[k] = 0


B = Behaviour()


def _sync_misbehave(dep: str):
    B.calls[dep] += 1
    mode = B.mode[dep]
    if mode == "error":
        raise FakeServerError(f"{dep}: 503 Service Unavailable")
    if mode == "hang":
        time.sleep(HANG_S)
    return mode


async def _async_misbehave(dep: str):
    B.calls[dep] += 1
    mode = B.mode[dep]
    if mode == "error":
        raise FakeServerError(f"{dep}: 503 Service Unavailable")
    if mode == "hang":
        await asyncio.sleep(HANG_S)
    return mode


SELECTOR_OK = json.dumps({"action": "proceed", "concept": QUERY, "categories": CATEGORIES})


def _chat_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None)


class FakeDeepSeek:
    """OpenAI-style client: .chat.completions.create(), sync or async."""

    def __init__(self, is_async: bool):
        create = self._acreate if is_async else self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def _create(self, **kw):
        return _chat_response("{not json" if _sync_misbehave("deepseek") == "bad" else SELECTOR_OK)

    async def _acreate(self, **kw):
        return _chat_response("{not json" if await _async_misbehave("deepseek") == "bad" else SELECTOR_OK)


HAIKU_TEXT = "Launch activity in dog water fountains is growing."


def _haiku_response(dep: str, mode: str):
    if mode == "bad":
        return SimpleNamespace(content=[])  # empty reply
    if "system" in _haiku_response.last_kwargs and "entry-point classifier" in _haiku_response.last_kwargs["system"]:
        return SimpleNamespace(content=[SimpleNamespace(text=SELECTOR_OK)])
    return SimpleNamespace(content=[SimpleNamespace(text=HAIKU_TEXT)])


_haiku_response.last_kwargs = {}


class FakeHaiku:
    """Anthropic-style client: .messages.create(), sync or async. Also serves as the
    category-selection fallback (same prompt, same JSON) when that exists."""

    def __init__(self, is_async: bool):
        self.messages = SimpleNamespace(create=self._acreate if is_async else self._create)

    def _create(self, **kw):
        _haiku_response.last_kwargs = kw
        return _haiku_response("haiku", _sync_misbehave("haiku"))

    async def _acreate(self, **kw):
        _haiku_response.last_kwargs = kw
        return _haiku_response("haiku", await _async_misbehave("haiku"))


def _path_p(path: str) -> float:
    if "Dogs" in path and "Fountain" in path:
        return 0.95
    if "Dogs" in path and "Feeding & Watering" in path:
        return 0.5
    return 0.02


def _title_p(title: str) -> float:
    t = title.lower()
    return 0.9 if "fountain" in t and ("dog" in t or "pet" in t) else 0.1


class _Answer:
    def __init__(self, p):
        self.noul = p

    def model_dump(self):
        return {"type": "noul", "noul": self.noul}


class FakeJev:
    """TypeSafe-style client. Path and title requests are told apart by their state keys."""

    def __init__(self, model=None, **kw):
        self.model = model

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        pass

    async def system_one(self, state, questions):
        dep = "jev_paths" if "matching_rules" in state else "jev_titles"
        mode = await _async_misbehave(dep)
        score = _path_p if dep == "jev_paths" else _title_p
        answers = {}
        for k, q in questions.items():
            subject = q.instructions.split('"')[1]   # the path or title quoted in the question
            answers[k] = _Answer(score(subject))
        if mode == "bad":   # a response that silently leaves out half the answers
            answers = dict(list(answers.items())[: len(answers) // 2])
        n = sum(len(q.instructions) for q in questions.values()) // 4
        return SimpleNamespace(answers=answers, model=self.model,
                               usage=SimpleNamespace(input_tokens=n, output_tokens=0))


class FakeFirestore:
    """Result-cache backend: a working dict, or one whose reads and writes fail."""

    def __init__(self):
        self.docs = {}

    def get(self, collection, key):
        _sync_misbehave("firestore")
        return self.docs.get((collection, key))

    def put(self, collection, key, value):
        _sync_misbehave("firestore")
        self.docs[(collection, key)] = value


# ---------- wiring ----------

def install():
    """Import the app and swap every external client for a fake. Returns the Flask module."""
    import main
    from src.agent_pipeline import analysis_agent
    from src.retrieval_pipeline import cat_selector, classify_agent, jev_scorer, main_2, title_filter

    tmp = Path(tempfile.mkdtemp(prefix="fault_injection_"))
    classify_agent.CATEGORY_CLASSIFY_DIR = tmp / "classify"   # don't overwrite real run artifacts
    main_2.PIPELINE_RUNS_DIR = tmp / "runs"

    for mod in (cat_selector,):
        if hasattr(mod, "client"):
            mod.client = FakeDeepSeek(is_async=False)
        if hasattr(mod, "aclient"):
            mod.aclient = FakeDeepSeek(is_async=True)
        if hasattr(mod, "fallback_client"):
            mod.fallback_client = FakeHaiku(is_async=True)
    analysis_agent.haiku_client = FakeHaiku(is_async=hasattr(analysis_agent, "WRITER_DEADLINE_S"))
    jev_scorer.AsyncTypeSafeClient = FakeJev
    title_filter.AsyncTypeSafeClient = FakeJev

    # test-scale deadlines (only the ones that exist in this version of the code)
    for mod, name in [(jev_scorer, "BATCH_DEADLINE_S"), (title_filter, "DEADLINE_S"),
                      (classify_agent, "BATCH_DEADLINE_S"), (cat_selector, "DEADLINE_S"),
                      (analysis_agent, "WRITER_DEADLINE_S")]:
        if hasattr(mod, name):
            setattr(mod, name, TEST_DEADLINE_S)
    for mod, name in [(jev_scorer, "RETRY_BACKOFF_S"), (title_filter, "RETRY_BACKOFF_S")]:
        if hasattr(mod, name):
            setattr(mod, name, 0.05)

    main.limiter.enabled = False
    return main


def fresh_cache(main):
    from src.shared import result_cache
    main.cache = result_cache.ResultCache(backend=FakeFirestore())
    return main.cache


def _post(main, path, body, bound=ANSWER_BOUND_S):
    """POST in a worker thread so a hung request can be timed out from here.
    Returns (status | None if no answer in time, json body, seconds)."""
    box = {}

    def go():
        t0 = time.perf_counter()
        r = main.app.test_client().post(path, json=body)
        box["r"] = (r.status_code, r.get_json(silent=True), time.perf_counter() - t0)

    th = threading.Thread(target=go, daemon=True)
    t0 = time.perf_counter()
    th.start()
    th.join(bound)
    if th.is_alive():
        th.join(HANG_S * 3)   # let it finish so it can't leak into the next scenario
        return None, None, time.perf_counter() - t0
    return box["r"]


def _degraded(body) -> bool:
    return bool(body and (body.get("degraded") or (body.get("report") or {}).get("degraded")))


# ---------- scenarios ----------

@dataclass
class Scenario:
    dep: str
    mode: str
    title: str


SCENARIOS = [
    Scenario("deepseek", "error", "DeepSeek (category selection) returns 5xx"),
    Scenario("deepseek", "hang", "DeepSeek (category selection) hangs"),
    Scenario("deepseek", "bad", "DeepSeek (category selection) returns invalid JSON"),
    Scenario("jev_paths", "error", "Jev (path classification) returns 5xx"),
    Scenario("jev_paths", "hang", "Jev (path classification) hangs"),
    Scenario("jev_paths", "bad", "Jev (path classification) drops half its answers"),
    Scenario("jev_titles", "error", "Jev (title filter) returns 5xx"),
    Scenario("jev_titles", "hang", "Jev (title filter) hangs"),
    Scenario("jev_titles", "bad", "Jev (title filter) drops half its answers"),
    Scenario("haiku", "error", "Claude Haiku (summary writer) returns 5xx"),
    Scenario("haiku", "hang", "Claude Haiku (summary writer) hangs"),
    Scenario("firestore", "error", "Firestore (cache) reads and writes fail"),
    Scenario("firestore", "hang", "Firestore (cache) hangs"),
]


@dataclass
class Result:
    scenario: Scenario
    passed: bool
    outcome: str
    seconds: float
    notes: list = field(default_factory=list)


def healthy_reference(main) -> dict:
    B.reset()
    fresh_cache(main)
    status, body, secs = _post(main, "/api/analyze", {"query": QUERY}, bound=120)
    assert status == 200 and body and body.get("data"), f"healthy run failed: {status} {body}"
    return {"match_count": body["report"]["match_count"], "seconds": secs}


def run_scenario(main, sc: Scenario, ref: dict) -> Result:
    B.reset()
    fresh_cache(main)
    B.mode[sc.dep] = sc.mode
    notes = []

    status, body, secs = _post(main, "/api/analyze", {"query": QUERY})

    if sc.dep == "haiku":
        if status != 200 or not (body or {}).get("data"):
            return Result(sc, False, f"dashboard failed ({status})", secs)
        w_status, w_body, w_secs = _post(main, "/api/bottom-line", {"query": QUERY, "report": body["report"]})
        if w_status is None:
            return Result(sc, False, f"summary request still waiting after {ANSWER_BOUND_S:.0f} s", secs + w_secs)
        clear = w_status == 200 or (w_body or {}).get("error")
        outcome = ("summary written" if w_status == 200
                   else f"dashboard shown; summary failed with HTTP {w_status} and a retry button")
        return Result(sc, bool(clear), outcome, secs + w_secs)

    if status is None:
        return Result(sc, False, f"no answer after {ANSWER_BOUND_S:.0f} s", secs)
    if status != 200:
        msg = (body or {}).get("error")
        return Result(sc, False, f"HTTP {status}" + (f": {msg}" if msg else " (server error)"), secs)
    if not body.get("data"):
        what = body.get("clarify") or (body.get("report") or {}).get("error")
        return Result(sc, False, f"no dashboard: {what!r}", secs)

    count = body["report"]["match_count"]
    degraded = _degraded(body)
    if count != ref["match_count"] and not degraded:
        return Result(sc, False, f"silent wrong answer: {count} matches vs {ref['match_count']}, "
                                 "not labelled degraded", secs)
    outcome = (f"partial results, labelled ({body['report'].get('degraded_reason') or 'degraded'}): {count} matches"
               if degraded else f"full answer: {count} matches")

    # the dependency recovers: the same query must now get the full result, not a cached degraded one
    B.mode[sc.dep] = "ok"
    s2, b2, _ = _post(main, "/api/analyze", {"query": QUERY}, bound=120)
    if s2 != 200 or not b2.get("data") or b2["report"]["match_count"] != ref["match_count"]:
        return Result(sc, False, outcome + "; after recovery the degraded answer was still served", secs)
    return Result(sc, True, outcome, secs, notes)


def run_all() -> tuple[dict, list[Result]]:
    main = install()
    ref = healthy_reference(main)
    results = [run_scenario(main, sc, ref) for sc in SCENARIOS]
    B.reset()
    return ref, results
