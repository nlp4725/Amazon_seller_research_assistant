"""title_filter.apply: re-decides only prior matches, records its evidence, no network."""

from src.retrieval_pipeline import title_filter


def test_apply_filters_only_prior_matches(monkeypatch):
    seen = {}

    async def fake_score(query, titles):
        seen.update(titles)
        return {k: (0.9 if "costume" in t else 0.1) for k, t in titles.items()}

    monkeypatch.setattr(title_filter, "ascore_titles", fake_score)
    items = [
        {"asin": "a", "title": "Pumpkin Halloween dog costume", "is_match": True},
        {"asin": "b", "title": "Dog Christmas sweater", "is_match": True},
        {"asin": "c", "title": "Dog leash", "is_match": False},
    ]
    out = title_filter.apply("dog Halloween costume", items)

    assert sorted(seen.values()) == ["Dog Christmas sweater", "Pumpkin Halloween dog costume"]  # non-match not sent
    assert [t["is_match"] for t in out] == [True, False, False]
    assert out[1]["matched_before_filter"] is True and out[1]["title_filter_p"] == 0.1
    assert "title_filter_p" not in out[2]


def test_apply_with_no_matches_makes_no_call(monkeypatch):
    async def boom(query, titles):
        raise AssertionError("should not be called")

    monkeypatch.setattr(title_filter, "ascore_titles", boom)
    items = [{"asin": "c", "title": "Dog leash", "is_match": False}]
    assert title_filter.apply("dog Halloween costume", items) == items
