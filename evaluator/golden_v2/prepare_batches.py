"""
Blind labeling batches for golden dataset v2.

Turns each pool (build_pools.py) into batch files a labeler can work through: the query's
definition, then one numbered line per candidate with ONLY its title and category path.
`found_by` is deliberately left out so no labeler can tell which system proposed an item.
Line order is shuffled with a fixed seed for the same reason (pools are sorted by ASIN).

Queries in SAMPLED are labeled on a fixed random sample of SAMPLE_N candidates and marked
status "partial" in v2: precision is scored on judged items only, recall is a lower bound.

Outputs, per batch id:
  batches/<id>.txt       what the labeler reads
  batches/<id>.map.json  line number -> asin (kept away from labelers)

CLI:
    python -m evaluator.golden_v2.prepare_batches
"""

import json
import random

import yaml

from evaluator.golden_v2.build_pools import DEFINITIONS, HERE, POOLS_DIR
from src.shared.paths import safe_name

BATCHES_DIR = HERE / "batches"
SAMPLED = {"summer dress", "female fitness clothes"}
SAMPLE_N = 500
MAX_BATCH = 500
SEED = 42

HEADER = """QUERY: {query}

DEFINITION: {definition}
INCLUDE: {include}
EXCLUDE: {exclude}
DECIDED: {decide}

Label every numbered product below against this definition with exactly one letter:
  E = exact: the product IS what the query asks for, per the definition
  S = substitute: a different product that could serve instead (e.g. wired instead of wireless)
  C = complement: an accessory, part or add-on FOR the product (e.g. a charger for a mouse)
  I = irrelevant
Judge only from the title and category given. When the title does not settle a detail the
definition requires, choose E only if the title makes it clearly likely.

PRODUCTS (number <TAB> title <TAB> category path):
"""


def main() -> None:
    specs = yaml.safe_load(DEFINITIONS.read_text())
    BATCHES_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    index = []
    for query, spec in specs.items():
        candidates = json.loads((POOLS_DIR / f"{safe_name(query)}.json").read_text())["candidates"]
        if query in SAMPLED and len(candidates) > SAMPLE_N:
            candidates = rng.sample(candidates, SAMPLE_N)
        else:
            candidates = candidates[:]
            rng.shuffle(candidates)
        chunks = [candidates[i:i + MAX_BATCH] for i in range(0, len(candidates), MAX_BATCH)]
        for k, chunk in enumerate(chunks):
            batch_id = f"{safe_name(query)}__{k}"
            header = HEADER.format(query=query, definition=spec["definition"],
                                   include="; ".join(spec.get("include", [])),
                                   exclude="; ".join(spec.get("exclude", [])),
                                   decide="; ".join(spec.get("decide", [])) or "-")
            lines = [f"{n}\t{c['title']}\t{c['category_path']}" for n, c in enumerate(chunk, start=1)]
            (BATCHES_DIR / f"{batch_id}.txt").write_text(header + "\n".join(lines) + "\n")
            (BATCHES_DIR / f"{batch_id}.map.json").write_text(
                json.dumps({str(n): c["asin"] for n, c in enumerate(chunk, start=1)}, indent=0))
            index.append({"batch": batch_id, "query": query, "n": len(chunk),
                          "sampled": query in SAMPLED})
    (BATCHES_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{len(index)} batches, {sum(b['n'] for b in index)} candidates")
    for b in index:
        print(f"  {b['batch']:40s} {b['n']:4d}{'  (sample)' if b['sampled'] else ''}")


if __name__ == "__main__":
    main()
