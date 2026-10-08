import pytest

import main

REPORT = {"concept": "dog bed", "categories": ["Pet Supplies"], "trend_by_theme": [], "top_sellers": []}


@pytest.fixture
def client(monkeypatch):
    main.app.config["TESTING"] = True
    main.limiter.enabled = False  # rate limits aren't under test; keep tests order-independent
    monkeypatch.setattr(main, "analyze", lambda messages, mode: {"report": REPORT, "data": {"mode": mode, "q": messages[0]["content"]}})
    monkeypatch.setattr(main, "write_bottom_line", lambda messages, report: f"bottom line for {report['concept']}")
    monkeypatch.setattr(main, "write_report", lambda messages, report: f"full report for {report['concept']}")
    return main.app.test_client()


def test_analyze_always_uses_jev_pipeline_and_trims_query(client):
    res = client.post("/api/analyze", json={"query": "  dog bed  ", "mode": "simple"})

    assert res.status_code == 200
    assert res.get_json()["data"] == {"mode": "structured", "q": "dog bed"}


@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": "   "}, {"query": 5}, {"query": "x" * 1001}])
def test_analyze_rejects_bad_queries(client, body):
    res = client.post("/api/analyze", json=body)

    assert res.status_code == 400 and "error" in res.get_json()


@pytest.mark.parametrize("route, expected", [("/api/bottom-line", "bottom line for dog bed"), ("/api/report", "full report for dog bed")])
def test_writers_narrate_the_report_sent_back(client, route, expected):
    res = client.post(route, json={"query": "dog bed", "report": REPORT})

    assert res.status_code == 200 and res.get_json() == {"text": expected}


@pytest.mark.parametrize("report", [
    None,
    "not a dict",
    {"concept": "dog bed"},                                  # missing fields the prompt relies on
    {**REPORT, "error": "No matching products found."},      # nothing to narrate
    {**REPORT, "padding": "x" * (main.MAX_REPORT_BYTES + 1)},  # not a real report
])
@pytest.mark.parametrize("route", ["/api/bottom-line", "/api/report"])
def test_writers_reject_missing_or_invalid_report(client, route, report):
    res = client.post(route, json={"query": "dog bed", "report": report})

    assert res.status_code == 400
