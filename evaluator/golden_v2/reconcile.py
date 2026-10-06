"""
Merge two blind labeling passes, send disagreements to a tie-break pass, write v2.

Only the E (exact) / not-E distinction decides ground truth, so that is what counts as a
disagreement: an item one pass called E and the other did not. S/C/I differences are kept
in the record but never adjudicated. Both passes agreeing on E -> E; both agreeing on
not-E -> pass A's label. Disputed items go to labels/T/ (a third, tie-break pass); its
label is final.

Steps:
    python -m evaluator.golden_v2.reconcile --disputes   # write tie-break batches, report agreement
    python -m evaluator.golden_v2.reconcile --write      # merge A, B, T -> golden_dataset_v2.json
"""

import csv
import json
import sys
from datetime import date

import yaml

from evaluator.golden_v2.build_pools import DEFINITIONS, HERE
from evaluator.golden_v2.prepare_batches import BATCHES_DIR, HEADER, SAMPLED

LABELS = HERE / "labels"
TIEBREAK_DIR = HERE / "tiebreak"
OUT = HERE.parent / "golden_dataset_v2.json"


def _labels(pass_name: str, batch: str) -> dict[str, str]:
    path = LABELS / pass_name / f"{batch}.csv"
    if not path.exists():
        return {}
    return {r["n"]: r["label"].strip().upper() for r in csv.DictReader(path.open())}


def _batch_lines(batch: str) -> dict[str, str]:
    lines = (BATCHES_DIR / f"{batch}.txt").read_text().split("PRODUCTS (number <TAB> title <TAB> category path):\n")[1]
    return {ln.split("\t", 1)[0]: ln for ln in lines.splitlines() if ln.strip()}


def disputes() -> None:
    specs = yaml.safe_load(DEFINITIONS.read_text())
    index = json.loads((BATCHES_DIR / "index.json").read_text())
    TIEBREAK_DIR.mkdir(exist_ok=True)
    total = 0
    for b in index:
        A, B = _labels("A", b["batch"]), _labels("B", b["batch"])
        assert len(A) == len(B) == b["n"], f"{b['batch']}: incomplete pass (A={len(A)}, B={len(B)}, n={b['n']})"
        disputed = [n for n in A if (A[n] == "E") != (B[n] == "E")]
        agree = sum(A[n] == B[n] for n in A) / b["n"]
        print(f"{b['batch']:38s} n={b['n']:4d} agreement={agree:.3f} E-disputes={len(disputed)}")
        if not disputed:
            continue
        total += len(disputed)
        spec = specs[b["query"]]
        lines = _batch_lines(b["batch"])
        header = HEADER.format(query=b["query"], definition=spec["definition"],
                               include="; ".join(spec.get("include", [])),
                               exclude="; ".join(spec.get("exclude", [])),
                               decide="; ".join(spec.get("decide", [])) or "-")
        (TIEBREAK_DIR / f"{b['batch']}.txt").write_text(header + "\n".join(lines[n] for n in disputed) + "\n")
    print(f"\n{total} disputed items written to {TIEBREAK_DIR}")


def write() -> None:
    specs = yaml.safe_load(DEFINITIONS.read_text())
    index = json.loads((BATCHES_DIR / "index.json").read_text())
    by_query: dict[str, dict] = {}
    for b in index:
        A, B, T = _labels("A", b["batch"]), _labels("B", b["batch"]), _labels("T", b["batch"])
        asin_of = json.loads((BATCHES_DIR / f"{b['batch']}.map.json").read_text())
        lines = _batch_lines(b["batch"])
        entry = by_query.setdefault(b["query"], {"items": [], "disputes": 0, "agree": 0, "n": 0})
        for n, a in A.items():
            disputed = (a == "E") != (B[n] == "E")
            if disputed:
                assert n in T, f"{b['batch']} #{n}: disputed but no tie-break label"
            final = T[n] if disputed else a
            _, title, path = lines[n].split("\t")
            entry["items"].append({"asin": asin_of[n], "title": title, "category_path": path,
                                   "label": final, "is_match": final == "E",
                                   "pass_a": a, "pass_b": B[n], "tiebreak": T.get(n)})
            entry["disputes"] += disputed
            entry["agree"] += a == B[n]
            entry["n"] += 1

    dataset = []
    for i, (query, e) in enumerate(by_query.items(), start=1):
        matches = sum(it["is_match"] for it in e["items"])
        status = "partial" if query in SAMPLED else ("negative" if matches == 0 else "complete")
        dataset.append({
            "id": i, "query": query, "status": status, "definition": specs[query],
            "match_count": matches, "titles_found_count": e["n"],
            "label_agreement": round(e["agree"] / e["n"], 4), "e_disputes": e["disputes"],
            "titles_found": e["items"],
        })
    OUT.write_text(json.dumps({"version": 2, "created": str(date.today()),
                               "method": "evaluator/golden_v2/README.md", "queries": dataset},
                              indent=1, ensure_ascii=False))
    for d in dataset:
        print(f"{d['query']:28s} {d['status']:9s} matches={d['match_count']:4d} "
              f"judged={d['titles_found_count']:5d} agreement={d['label_agreement']:.3f} disputes={d['e_disputes']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    {"--disputes": disputes, "--write": write}[sys.argv[1]]()
