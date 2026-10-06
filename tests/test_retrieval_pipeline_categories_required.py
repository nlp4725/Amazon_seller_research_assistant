"""
main_1/main_2 no longer pick their own categories at all -- cat_selector.select() is the
single source of truth, called once by the caller (analysis_agent.py live, or
evaluator/comparison.py / each main's own CLI block otherwise) and its `categories` output
passed straight into run(). There is no internal fallback to guard against anymore: neither
main imports or references any category-picking function.
"""

import pytest

from src.retrieval_pipeline import main_1, main_2

PET_SUPPLIES = [{"category": "Pet Supplies", "reason": "caller-specified"}]


def test_main_1_has_no_category_picking_import():
    assert not hasattr(main_1, "pick_candidate_categories")


def test_main_2_has_no_category_picking_import():
    assert not hasattr(main_2, "pick_candidate_categories")


def test_main_1_run_requires_categories():
    with pytest.raises(TypeError):
        main_1.run("dog drinking bowl")  # categories is a required positional arg


def test_main_2_run_requires_categories():
    with pytest.raises(TypeError):
        main_2.run("dog drinking bowl")


def test_main_1_run_uses_caller_supplied_categories(monkeypatch):
    monkeypatch.setattr(
        main_1, "rank_category_items",
        lambda query, categories: [
            {"asin": "B1", "title": "dog fountain", "cat": categories[0],
             "category_path": "", "vector_distance": 0.1},
        ],
    )
    monkeypatch.setattr(
        main_1, "rerank_titles",
        lambda query, items: [{**item, "rerank_score": 5.0, "is_match": True} for item in items],
    )

    output = main_1.run("dog drinking bowl", categories=PET_SUPPLIES)

    assert output["categories"] == PET_SUPPLIES
    assert output["match_count"] == 1


def test_main_2_run_uses_caller_supplied_categories(monkeypatch):
    monkeypatch.setattr(
        main_2, "classify_paths",
        lambda query, category: {
            "category": category,
            "confident_match": {"p1": "Pet Supplies > Dogs > Fountains"},
            "ambiguous_match": {},
            "not_match": {},
            "partial_failure": False,
        },
    )
    monkeypatch.setattr(
        main_2, "fetch_items_for_paths",
        lambda paths: (
            [{"asin": "B1", "title": "dog fountain", "cat": "Pet Supplies", "category_path": paths[0]}]
            if paths else []
        ),
    )
    # Stage 3 (title_filter) would call Jev; this test is about categories, not filtering.
    monkeypatch.setattr(main_2.title_filter, "TITLE_FILTER", False)

    output = main_2.run("dog drinking bowl", categories=PET_SUPPLIES)

    assert output["categories"] == PET_SUPPLIES
    assert output["match_count"] == 1
