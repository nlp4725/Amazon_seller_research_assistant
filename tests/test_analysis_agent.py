import numpy as np
import pandas as pd
import pytest

from src.agent_pipeline import analysis_agent
from src.agent_pipeline.analysis_agent import _get_theme_trend

PET_SUPPLIES = [{"category": "Pet Supplies", "reason": "test"}]


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


def test_get_product_subset_simple_mode_calls_main_1_with_categories(monkeypatch):
    captured = {}

    def fake_run(query, categories):
        captured["query"], captured["categories"] = query, categories
        return _fake_pipeline_result()

    def fake_hydrate(asins):
        captured["hydrated_asins"] = asins
        return pd.DataFrame({"asin": asins}), np.zeros((len(asins), 4))

    monkeypatch.setattr(analysis_agent.main_1, "run", fake_run)
    monkeypatch.setattr(analysis_agent.main_2, "run", lambda *a, **k: pytest.fail("wrong pipeline: structured called in simple mode"))
    monkeypatch.setattr(analysis_agent, "hydrate_items", fake_hydrate)

    sub_df, sub_emb, meta = analysis_agent._get_product_subset(PET_SUPPLIES, "dog fountain", mode="simple")

    assert captured["query"] == "dog fountain"
    assert captured["categories"] == PET_SUPPLIES
    assert captured["hydrated_asins"] == ["B1", "B3"]  # only is_match=True items get hydrated
    assert meta == {"mode": "simple", "match_count": 2, "titles_found_count": 3}
    assert len(sub_df) == 2


def test_get_product_subset_structured_mode_calls_main_2(monkeypatch):
    monkeypatch.setattr(analysis_agent.main_1, "run", lambda *a, **k: pytest.fail("wrong pipeline: simple called in structured mode"))
    monkeypatch.setattr(analysis_agent.main_2, "run", lambda query, categories: _fake_pipeline_result())
    monkeypatch.setattr(analysis_agent, "hydrate_items", lambda asins: (pd.DataFrame({"asin": asins}), np.zeros((len(asins), 4))))

    sub_df, sub_emb, meta = analysis_agent._get_product_subset(PET_SUPPLIES, "dog fountain", mode="structured")

    assert meta["mode"] == "structured"


def test_get_product_subset_no_concept_bypasses_pipeline(monkeypatch):
    monkeypatch.setattr(analysis_agent.main_1, "run", lambda *a, **k: pytest.fail("pipeline should not run without a concept"))
    monkeypatch.setattr(analysis_agent.main_2, "run", lambda *a, **k: pytest.fail("pipeline should not run without a concept"))

    sub_df, sub_emb, meta = analysis_agent._get_product_subset(PET_SUPPLIES, None, mode="simple")

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
        analysis_agent, "_get_product_subset",
        lambda categories, concept, mode: (sub_df, sub_emb, pipeline_meta),
    )

    result = analysis_agent.niche_report(PET_SUPPLIES, concept="dog fountain", mode="structured")

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
        analysis_agent, "_get_product_subset",
        lambda categories, concept, mode: (sub_df, sub_emb, None),
    )

    result = analysis_agent.niche_report(PET_SUPPLIES, concept=None)

    assert result["mode"] is None
    assert result["match_count"] is None
    assert result["titles_found_count"] is None


# ---------- report_view: the frontend's chart payload, never sent to Haiku ----------

def _report_and_sub(monkeypatch, pipeline_meta):
    sub_df, sub_emb = _fake_matched_sub(5)
    monkeypatch.setattr(
        analysis_agent, "_get_product_subset",
        lambda categories, concept, mode: (sub_df, sub_emb, pipeline_meta),
    )
    report = analysis_agent.niche_report(PET_SUPPLIES, concept="dog fountain", mode="structured")
    return report, sub_df


def test_report_view_copies_report_numbers(monkeypatch):
    report, sub_df = _report_and_sub(monkeypatch, {"mode": "structured", "match_count": 5, "titles_found_count": 166})

    view = analysis_agent.report_view(report, sub_df)

    assert view["stats"]["products"] == 5
    assert view["stats"]["is_match_count"] is True
    assert view["stats"]["pct_of_category"] == report["pct_of_category_launches"]
    assert view["stats"]["scan_label"] == "Thorough scan"
    assert view["stats"]["candidates_considered"] == 166  # titles_found_count, shown as "Candidates checked"
    assert [t["trend_tag"] for t in view["themes"]] == [c["trend_tag"] for c in report["trend_by_theme"]]
    assert [s["pct"] for s in view["top_sellers"]] == [s["pct_of_total_launches"] for s in report["top_sellers"]]


def test_report_view_null_match_count_falls_back_to_total_analyzed(monkeypatch):
    report, sub_df = _report_and_sub(monkeypatch, None)

    view = analysis_agent.report_view(report, sub_df)

    assert view["stats"]["products"] == report["total_products_analyzed"] == 5
    assert view["stats"]["is_match_count"] is False


def test_report_view_months_are_zero_filled_and_sum_to_total(monkeypatch):
    report, sub_df = _report_and_sub(monkeypatch, None)

    months = analysis_agent.report_view(report, sub_df)["launches_by_month"]

    # fake launches: 2024-01, 2025-02, 2024-03, 2025-04, 2024-05 -> 16 consecutive months
    assert months[0]["month"] == "2024-01" and months[-1]["month"] == "2025-04"
    assert len(months) == 16
    assert sum(m["count"] for m in months) == 5
    assert any(m["count"] == 0 for m in months)


def test_report_view_recent_launches_newest_first():
    sub_df, _ = _fake_matched_sub(12)
    sub_df.loc[0, "seller"] = "-1"  # unknown seller must come through as None, not "-1"
    report = {"match_count": 12, "total_products_analyzed": 12, "pct_of_category_launches": 1.0,
              "category_total": 1200, "scan_label": "Fast scan", "titles_found_count": 681, "categories": ["Pet Supplies"],
              "concept": "dog fountain", "trend_by_theme": [], "top_sellers": [], "note": ""}

    recent = analysis_agent.report_view(report, sub_df)["recent_launches"]

    assert len(recent) == analysis_agent.RECENT_LAUNCHES_N
    assert [r["launched"] for r in recent] == sorted((r["launched"] for r in recent), reverse=True)
    assert all(r["seller"] != "-1" for r in recent)


def test_report_view_empty_subset_does_not_crash():
    report = {"match_count": 0, "total_products_analyzed": 0, "pct_of_category_launches": 0.0,
              "category_total": 100, "scan_label": "Fast scan", "titles_found_count": 40, "categories": ["Pet Supplies"],
              "concept": "x", "trend_by_theme": [], "top_sellers": [], "note": ""}

    view = analysis_agent.report_view(report, pd.DataFrame())

    assert view["launches_by_month"] == [] and view["recent_launches"] == []
    assert view["stats"]["median_price"] is None
    assert view["stats"]["products"] == 0


def test_run_chat_clarify_returns_no_data(monkeypatch):
    monkeypatch.setattr(analysis_agent.cat_selector, "select", lambda messages: {"clarify": "Which niche?"})

    assert analysis_agent.run_chat([{"role": "user", "content": "hi"}]) == {"reply": "Which niche?", "data": None}


def test_price_distribution_buckets_cover_every_product():
    prices = pd.Series([15.0, 19.99, 20.0, 29.0, 49.0, 50.0, 99.0, 100.0, 12.0])  # 12 falls in the open first bucket

    dist = analysis_agent._price_distribution(prices)

    assert [b["range"] for b in dist] == ["$15–20", "$20–30", "$30–50", "$50–75", "$75–100"]
    assert [b["count"] for b in dist] == [3, 2, 1, 1, 2]
    assert sum(b["count"] for b in dist) == len(prices)


def test_report_view_seller_stats_agree_with_report(monkeypatch):
    report, sub_df = _report_and_sub(monkeypatch, None)

    view = analysis_agent.report_view(report, sub_df)

    assert view["stats"]["unique_sellers"] == 2  # fake sub has sellers S0, S1
    assert view["stats"]["top_sellers_pct"] == round(sum(s["pct_of_total_launches"] for s in report["top_sellers"]), 1)
    assert view["price_peak"] in [b["range"] for b in view["price_distribution"]]
