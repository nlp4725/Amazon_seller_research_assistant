import numpy as np
import pandas as pd
import pytest

from src.agent_pipeline import analysis_agent
from src.shared import result_cache
from src.shared.result_cache import FirestoreStore, MemoryStore, ResultCache

PET_SUPPLIES = [{"category": "Pet Supplies", "reason": "test"}]


# ---------- keys ----------

def test_normalize_query_ignores_case_punctuation_and_spacing():
    assert result_cache.normalize_query("  Silicone ICE-tray!  ") == "silicone ice tray"


def test_query_key_matches_normalized_wording_only():
    key = lambda q, mode="structured": result_cache.query_key([{"role": "user", "content": q}], mode)
    assert key("Dog bed") == key("  dog   BED. ")
    assert key("dog bed") != key("dog beds")
    assert key("dog bed") != key("dog bed", mode="simple")


def test_concept_key_ignores_category_order_and_reasons():
    a = [{"category": "Pet Supplies", "reason": "x"}, {"category": "Home & Kitchen", "reason": "y"}]
    b = [{"category": "Home & Kitchen", "reason": "other"}, {"category": "Pet Supplies", "reason": "z"}]
    assert result_cache.concept_key("Dog Bed", a, "structured", 6) == result_cache.concept_key("dog bed", b, "structured", 6)


def test_keys_change_with_dataset_and_cache_version(monkeypatch):
    msgs = [{"role": "user", "content": "dog bed"}]
    before = result_cache.query_key(msgs, "structured")

    monkeypatch.setattr(result_cache, "dataset_version", lambda: "rebuilt")
    after_rebuild = result_cache.query_key(msgs, "structured")
    monkeypatch.setattr(result_cache, "CACHE_VERSION", result_cache.CACHE_VERSION + 1)

    assert len({before, after_rebuild, result_cache.query_key(msgs, "structured")}) == 3


def test_prose_key_changes_with_prompt():
    msgs = [{"role": "user", "content": "dog bed"}]
    report = {"concept": "dog bed"}
    assert (result_cache.prose_key("report", msgs, report, "prompt v1", "haiku")
            != result_cache.prose_key("report", msgs, report, "prompt v2", "haiku"))


# ---------- stores ----------

def test_memory_store_evicts_least_recently_used():
    store = MemoryStore(max_items=2)
    store.put("c", "a", {"v": 1})
    store.put("c", "b", {"v": 2})
    store.get("c", "a")  # touch a, so b is now the oldest
    store.put("c", "c", {"v": 3})

    assert store.get("c", "b") is None
    assert store.get("c", "a") == {"v": 1} and store.get("c", "c") == {"v": 3}


def test_memory_store_hit_is_a_copy():
    store = MemoryStore()
    store.put("c", "k", {"items": [1]})
    store.get("c", "k")["items"].append(2)

    assert store.get("c", "k") == {"items": [1]}


class _BrokenStore:
    def get(self, collection, key):
        raise RuntimeError("firestore down")

    def put(self, collection, key, value):
        raise RuntimeError("firestore down")


def test_backend_failure_is_a_miss_not_an_error():
    cache = ResultCache(_BrokenStore())

    assert cache.get("c", "k") is None
    cache.put("c", "k", {"v": 1})            # write fails quietly...
    assert cache.get("c", "k") == {"v": 1}   # ...but the memory layer still serves it


class _FakeDoc:
    def __init__(self, docs, key):
        self.docs, self.key = docs, key

    def get(self):
        data = self.docs.get(self.key)
        return type("Snap", (), {"exists": data is not None, "get": lambda _, f: data[f]})()

    def set(self, data):
        self.docs[self.key] = data


class _FakeFirestore:
    def __init__(self):
        self.docs = {}

    def collection(self, name):
        return type("Col", (), {"document": lambda _, key: _FakeDoc(self.docs, (name, key))})()


def test_firestore_store_round_trips_nested_lists_and_sets_expiry():
    db = _FakeFirestore()
    store = FirestoreStore(db)
    store.put("analysis", "k", {"grid": [[1, 2], [3]]})  # nested arrays: stored as a JSON string

    assert store.get("analysis", "k") == {"grid": [[1, 2], [3]]}
    assert "expires_at" in db.docs[("analysis", "k")]
    assert store.get("analysis", "missing") is None


def test_firestore_store_skips_oversized_entries():
    db = _FakeFirestore()
    FirestoreStore(db).put("analysis", "k", {"blob": "x" * result_cache.MAX_DOC_BYTES})

    assert db.docs == {}


def test_from_env_defaults_to_memory_and_survives_missing_firestore(monkeypatch):
    monkeypatch.delenv("CACHE_BACKEND", raising=False)
    assert result_cache.from_env()._backend is None

    monkeypatch.setenv("CACHE_BACKEND", "firestore")
    monkeypatch.setattr(result_cache, "FirestoreStore", lambda: (_ for _ in ()).throw(RuntimeError("no credentials")))
    assert result_cache.from_env()._backend is None


# ---------- analyze() with a cache ----------

def _fake_sub(n=5, seed=0):
    rng = np.random.default_rng(seed)
    years = [2025, 2026]
    df = pd.DataFrame({
        "asin": [f"A{i}" for i in range(n)],
        "title": [f"fountain {i}" for i in range(n)],
        "price": rng.uniform(15, 100, size=n),
        "seller": [f"S{i % 2}" for i in range(n)],
        "cat": ["Pet Supplies"] * n,
        "category_path": ["Pet Supplies > Dogs > Fountains"] * n,
        "launch_year": [years[i % 2] for i in range(n)],
        "launch_year_month": pd.to_datetime([f"{years[i % 2]}-0{(i % 9) + 1}-01" for i in range(n)]),
        "review_velocity": rng.uniform(0, 0.1, size=n),
    })
    return df, rng.normal(size=(n, 4))


@pytest.fixture
def counted(monkeypatch):
    """Fake cat_selector + retrieval that count their calls. resolve maps query -> concept."""
    calls = {"select": 0, "retrieve": 0}
    meta = {"mode": "structured", "match_count": 5, "titles_found_count": 9, "partial_failure": False}

    def select(messages):
        calls["select"] += 1
        q = messages[-1]["content"].lower()
        if q == "hi":
            return {"clarify": "Which niche?"}
        return {"concept": "dog fountain", "categories": PET_SUPPLIES}

    def retrieve(categories, concept, mode):
        calls["retrieve"] += 1
        return (*_fake_sub(), dict(meta))

    monkeypatch.setattr(analysis_agent.cat_selector, "select", select)
    monkeypatch.setattr(analysis_agent, "_get_product_subset", retrieve)
    calls["meta"] = meta
    return calls


def _ask(q, cache):
    return analysis_agent.analyze([{"role": "user", "content": q}], cache=cache)


def test_same_query_skips_cat_selector_and_retrieval(counted):
    cache = ResultCache()
    first = _ask("dog fountain", cache)
    again = _ask("Dog Fountain!", cache)

    assert first == again
    assert counted["select"] == 1 and counted["retrieve"] == 1


def test_new_wording_same_concept_skips_retrieval_only(counted):
    cache = ResultCache()
    first = _ask("dog fountain", cache)
    reworded = _ask("water fountain for dogs", cache)

    assert first == reworded
    assert counted["select"] == 2 and counted["retrieve"] == 1

    _ask("water fountain for dogs", cache)  # that wording is now a query-key hit too
    assert counted["select"] == 2


def test_clarify_is_cached(counted):
    cache = ResultCache()
    assert _ask("hi", cache) == _ask("hi", cache) == {"clarify": "Which niche?"}
    assert counted["select"] == 1 and counted["retrieve"] == 0


def test_degraded_result_is_not_cached(counted):
    counted["meta"]["partial_failure"] = True
    cache = ResultCache()
    _ask("dog fountain", cache)
    _ask("dog fountain", cache)

    assert counted["retrieve"] == 2


def test_empty_chroma_error_is_not_cached(counted, monkeypatch):
    class EmptyCol:
        def count(self):
            return 0

    monkeypatch.setattr(analysis_agent, "_load_chroma", lambda: EmptyCol())
    cache = ResultCache()
    _ask("dog fountain", cache)
    _ask("dog fountain", cache)

    assert counted["select"] == 2


def test_no_cache_always_recomputes(counted):
    _ask("dog fountain", None)
    _ask("dog fountain", None)

    assert counted["select"] == 2 and counted["retrieve"] == 2
