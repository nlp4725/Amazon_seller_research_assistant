import numpy as np
import pandas as pd
import pytest

from src.agent_pipeline import chat_engine
from src.agent_pipeline.chat_engine import _get_theme_trend


def _make_sub(n: int, dim: int = 4, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """n synthetic matched products spread across a few launch years, with random embeddings."""
    rng = np.random.default_rng(seed)
    years = [2024, 2025, 2026]
    sub_df = pd.DataFrame({
        "title": [f"product {i}" for i in range(n)],
        "price": rng.uniform(15, 100, size=n),
        "launch_year": [years[i % len(years)] for i in range(n)],
    })
    sub_emb = rng.normal(size=(n, dim))
    return sub_df, sub_emb


# ---------- small match counts (the new pipeline can return 0, 1, or a handful) ----------

def test_theme_trend_zero_matches_returns_empty_list():
    sub_df, sub_emb = _make_sub(0)
    assert _get_theme_trend(sub_df, sub_emb, n_clusters=6) == []


def test_theme_trend_single_match_returns_empty_list():
    # KMeans(n_clusters=k) requires n_samples >= k -- a single matched product can't
    # form >1 cluster, so this must not attempt to call KMeans at all.
    sub_df, sub_emb = _make_sub(1)
    assert _get_theme_trend(sub_df, sub_emb, n_clusters=6) == []


def test_theme_trend_few_matches_does_not_request_more_clusters_than_samples():
    # 3 matched products, default n_clusters=6 -- KMeans would raise
    # "n_samples=3 should be >= n_clusters=6" unless k is capped by len(sub_df).
    sub_df, sub_emb = _make_sub(3)
    trend = _get_theme_trend(sub_df, sub_emb, n_clusters=6)
    assert 1 <= len(trend) <= 3
    assert sum(c["size"] for c in trend) == 3


# ---------- regression: normal-sized match set still clusters as before ----------

def test_theme_trend_normal_case_shapes_and_totals():
    sub_df, sub_emb = _make_sub(40)
    trend = _get_theme_trend(sub_df, sub_emb, n_clusters=6)
    assert len(trend) >= 2
    assert sum(c["size"] for c in trend) == 40
    for cluster in trend:
        assert cluster["representative_titles"]
        assert cluster["avg_price"] > 0
        assert cluster["by_year"]
    # largest clusters first
    sizes = [c["size"] for c in trend]
    assert sizes == sorted(sizes, reverse=True)


# ---------- _get_product_subset: routes to the user-chosen retrieval pipeline ----------

def _fake_pipeline_result():
    return {
        "titles_found": [
            {"asin": "B1", "is_match": True},
            {"asin": "B2", "is_match": False},  # considered but not a match -- must be filtered out
            {"asin": "B3", "is_match": True},
        ],
        "match_count": 2,
        "titles_found_count": 3,
    }


def test_get_product_subset_simple_mode_calls_main_1_with_category(monkeypatch):
    captured = {}

    def fake_run(query, category=None):
        captured["query"], captured["category"] = query, category
        return _fake_pipeline_result()

    def fake_hydrate(asins):
        captured["hydrated_asins"] = asins
        return pd.DataFrame({"asin": asins}), np.zeros((len(asins), 4))

    monkeypatch.setattr(chat_engine.main_1, "run", fake_run)
    monkeypatch.setattr(chat_engine.main_2, "run", lambda *a, **k: pytest.fail("wrong pipeline: structured called in simple mode"))
    monkeypatch.setattr(chat_engine, "hydrate_items", fake_hydrate)

    sub_df, sub_emb, meta = chat_engine._get_product_subset("Pet Supplies", "dog fountain", mode="simple")

    assert captured["query"] == "dog fountain"
    assert captured["category"] == "Pet Supplies"
    assert captured["hydrated_asins"] == ["B1", "B3"]  # only is_match=True items get hydrated
    assert meta == {"mode": "simple", "match_count": 2, "titles_found_count": 3}
    assert len(sub_df) == 2


def test_get_product_subset_structured_mode_calls_main_2(monkeypatch):
    monkeypatch.setattr(chat_engine.main_1, "run", lambda *a, **k: pytest.fail("wrong pipeline: simple called in structured mode"))
    monkeypatch.setattr(chat_engine.main_2, "run", lambda query, category=None: _fake_pipeline_result())
    monkeypatch.setattr(chat_engine, "hydrate_items", lambda asins: (pd.DataFrame({"asin": asins}), np.zeros((len(asins), 4))))

    sub_df, sub_emb, meta = chat_engine._get_product_subset("Pet Supplies", "dog fountain", mode="structured")

    assert meta["mode"] == "structured"


def test_get_product_subset_no_concept_bypasses_pipeline(monkeypatch):
    monkeypatch.setattr(chat_engine.main_1, "run", lambda *a, **k: pytest.fail("pipeline should not run without a concept"))
    monkeypatch.setattr(chat_engine.main_2, "run", lambda *a, **k: pytest.fail("pipeline should not run without a concept"))

    sub_df, sub_emb, meta = chat_engine._get_product_subset("Pet Supplies", None, mode="simple")

    assert meta is None
    assert len(sub_df) > 0  # real full-category fetch from local ChromaDB


# ---------- niche_report: reports the pipeline's real match_count/mode, not a cap ----------

def _fake_matched_sub(n: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2)
    years = [2024, 2025]
    df = pd.DataFrame({
        "asin": [f"B{i}" for i in range(n)],
        "title": [f"dog fountain {i}" for i in range(n)],
        "price": rng.uniform(15, 50, size=n),
        "seller": [f"S{i % 2}" for i in range(n)],
        "cat": ["Pet Supplies"] * n,
        "category_path": ["Pet Supplies > Dogs > Fountains"] * n,
        "launch_year": [years[i % 2] for i in range(n)],
        "launch_year_month": pd.to_datetime([f"{years[i % 2]}-0{(i % 9) + 1}-01" for i in range(n)]),
        "review_velocity": rng.uniform(0, 0.1, size=n),
    })
    emb = rng.normal(size=(n, 4))
    return df, emb


def test_niche_report_surfaces_pipeline_match_count_and_mode(monkeypatch):
    sub_df, sub_emb = _fake_matched_sub(5)
    pipeline_meta = {"mode": "structured", "match_count": 5, "titles_found_count": 166}
    monkeypatch.setattr(
        chat_engine, "_get_product_subset",
        lambda category, concept, mode: (sub_df, sub_emb, pipeline_meta),
    )

    result = chat_engine.niche_report("Pet Supplies", concept="dog fountain", mode="structured")

    assert result["mode"] == "structured"
    assert result["match_count"] == 5
    assert result["titles_found_count"] == 166
    assert result["category_total"] > 0
    assert result["pct_of_category_launches"] is not None
    # the old cosine-distance-threshold heuristic is fully retired
    assert "closely_related_count" not in result
    # context-size guardrail: never send the raw per-item list back to Claude
    assert "titles_found" not in result


def test_niche_report_no_concept_has_no_mode_or_match_count(monkeypatch):
    sub_df, sub_emb = _fake_matched_sub(5)
    monkeypatch.setattr(
        chat_engine, "_get_product_subset",
        lambda category, concept, mode: (sub_df, sub_emb, None),
    )

    result = chat_engine.niche_report("Pet Supplies", concept=None)

    assert result["mode"] is None
    assert result["match_count"] is None
    assert result["titles_found_count"] is None
