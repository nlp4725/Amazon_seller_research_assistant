"""metrics_v2.score: each golden v2 status is scored its own way."""

import pytest

# evaluator/ is offline tooling, kept out of the API image (.dockerignore), so the
# in-image test step skips this module rather than failing on the import.
metrics_v2 = pytest.importorskip("evaluator.metrics_v2")
score, summarize = metrics_v2.score, metrics_v2.summarize


def _entry(status, labels):
    return {"query": "q", "status": status,
            "titles_found": [{"asin": a, "label": lab} for a, lab in labels.items()]}


def test_complete_counts_unjudged_predictions_against_precision():
    e = _entry("complete", {"a": "E", "b": "E", "c": "C", "d": "I"})
    s = score({"a", "c", "x"}, e)
    assert (s["tp"], s["fn"], s["complements"], s["unjudged"]) == (1, 1, 1, 1)
    assert s["precision"] == round(1 / 3, 4) and s["recall"] == 0.5


def test_partial_scores_precision_on_the_judged_sample_only():
    e = _entry("partial", {"a": "E", "b": "I"})
    s = score({"a", "b", "outside1", "outside2"}, e)
    assert s["precision"] == 0.5 and s["recall"] == 1.0


def test_negative_is_false_positives_only_and_never_averaged():
    neg = score({"a", "b"}, _entry("negative", {"a": "I"}))
    assert neg["fp"] == 2 and neg["precision"] is None
    ok = score({"a"}, _entry("complete", {"a": "E"}))
    s = summarize([neg, ok])
    assert s["scored"] == 1 and s["mean_f1"] == 1.0 and s["negative_false_positives"] == {"q": 2}
