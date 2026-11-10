"""Every simulated outage must still give the user an answer (see harness.py for the rules).
Runs the real pipeline on local data with fake providers: no paid calls, ~30 s."""

import pytest

from tests.fault_injection.harness import SCENARIOS, run_all


@pytest.fixture(scope="module")
def results():
    _, results = run_all()
    return {(r.scenario.dep, r.scenario.mode): r for r in results}


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda sc: f"{sc.dep}-{sc.mode}")
def test_outage_still_answers(results, sc):
    r = results[(sc.dep, sc.mode)]
    assert r.passed, f"{sc.title}: {r.outcome}"
