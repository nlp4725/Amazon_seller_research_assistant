"""
Run every fault-injection scenario (tests/fault_injection/harness.py) and write the result
to docs/evals/fault_injection_YYYYMMDD.md. No paid API calls: every provider is a fake.

    python -m evaluator.fault_injection            # write the report
    python -m evaluator.fault_injection --print    # print only
"""

import sys
from datetime import date
from pathlib import Path

from tests.fault_injection.harness import ANSWER_BOUND_S, HANG_S, QUERY, TEST_DEADLINE_S, run_all


def render(ref: dict, results: list) -> str:
    passed = sum(r.passed for r in results)
    worst = max(r.seconds for r in results)
    lines = [
        f"# Fault injection, {date.today().isoformat()}",
        "",
        f"**{passed} of {len(results)} simulated outages still gave the user an answer** "
        f"(worst response {worst:.1f} s, at test-scale deadlines).",
        "",
        f"Query: \"{QUERY}\". Healthy run: {ref['match_count']} matches. "
        f"Every provider is a fake; deadlines are {TEST_DEADLINE_S:.0f} s at test scale, a hung "
        f"fake waits {HANG_S:.0f} s, and the user must have an answer within {ANSWER_BOUND_S:.0f} s. "
        "Pass rules are in tests/fault_injection/harness.py.",
        "",
        "| # | Outage | Result | Outcome | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r.scenario.title} | {'PASS' if r.passed else 'FAIL'} | {r.outcome} | {r.seconds:.1f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ref, results = run_all()
    text = render(ref, results)
    print(text)
    if "--print" not in sys.argv:
        out = Path("docs/evals") / f"fault_injection_{date.today():%Y%m%d}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}")
