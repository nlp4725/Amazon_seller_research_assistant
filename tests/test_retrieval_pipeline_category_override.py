"""
chat_engine already knows the user-selected category (validated against its own
CATEGORIES list) -- letting main_1/main_2 re-derive it via pick_candidate_categories
would be a redundant LLM call that could second-guess a category the caller has already
fixed. Both mains take an optional `category` override for exactly this: when given,
skip the orchestrator call entirely and search only that category.
"""

from src.retrieval_pipeline import main_1, main_2


def _boom(*a, **k):
    raise AssertionError("orchestrator should not be called when category is given")


def test_main_1_run_with_category_skips_orchestrator(monkeypatch):
    monkeypatch.setattr(main_1, "pick_candidate_categories", _boom)
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

    output = main_1.run("dog drinking bowl", category="Pet Supplies")

    assert output["categories"] == [{"category": "Pet Supplies", "reason": "caller-specified"}]
    assert output["match_count"] == 1


def test_main_2_run_with_category_skips_orchestrator(monkeypatch):
    monkeypatch.setattr(main_2, "pick_candidate_categories", _boom)
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

    output = main_2.run("dog drinking bowl", category="Pet Supplies")

    assert output["categories"] == [{"category": "Pet Supplies", "reason": "caller-specified"}]
    assert output["match_count"] == 1
