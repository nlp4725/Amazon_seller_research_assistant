"""
Where does reranker time go? Profiles the cross-encoder stage of the live pipeline on
whatever machine runs it -- meant to run inside the API image on Cloud Run (as a Job, same
CPU/memory as the service) so the numbers match production, not a laptop.

Measures, in one fresh process:
  1. startup: import torch / sentence-transformers / chromadb, load each model, open Chroma,
     plus a separate cold `import main` (what gunicorn does before serving)
  2. per search: candidate count, token lengths, padding waste per batch
  3. tokenization vs model forward time, per config (threads x batch size x sort x max_len)
  4. score drift vs the production config -- a faster config only counts if it flips no matches
  5. two searches reranking at the same time in one process

Input: a JSON file {query: [asin, ...]} of the ambiguous candidates each search reranks
(dumped from data/processed/pipeline_runs/structured__*.json, see --dump-candidates).

    python scripts/profile_reranker.py --dump-candidates candidates.json
    python scripts/profile_reranker.py candidates.json > profile.json
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time

T0 = time.perf_counter()
startup: dict[str, float] = {}


def mark(step: str, since: float) -> float:
    now = time.perf_counter()
    startup[step] = round(now - since, 3)
    return now


PROD = {"threads": 1, "batch": 128, "sort": False, "max_len": 512}  # what the service runs today
QUERIES = [  # small → huge; ambiguous-candidate counts from the saved structured runs
    "dog toy", "dog halloween costume", "solar powered garden light",
    "water bottle", "baby toys", "summer dress",
]


def dump_candidates(path: str) -> None:
    from src.shared.paths import PIPELINE_RUNS_DIR, safe_name
    out = {}
    for q in QUERIES:
        run = json.load(open(PIPELINE_RUNS_DIR / f"structured__{safe_name(q)}.json"))
        out[q] = [t["asin"] for t in run["titles_found"] if t.get("source") == "reranked_ambiguous"]
    json.dump(out, open(path, "w"))
    print({q: len(a) for q, a in out.items()})


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def batches_of(pairs, batch, sort):
    order = list(range(len(pairs)))
    if sort:  # group similar lengths so each batch pads less
        order.sort(key=lambda i: len(pairs[i][1]))
    return [[pairs[i] for i in order[s:s + batch]] for s in range(0, len(order), batch)], order


def run_config(model, torch, pairs, cfg):
    """One full rerank pass with tokenization and forward timed separately. Returns
    (tokenize_s, forward_s, scores in input order, padded_tokens, real_tokens)."""
    torch.set_num_threads(cfg["threads"])
    tok, net = model.tokenizer, model.model
    groups, order = batches_of(pairs, cfg["batch"], cfg["sort"])
    t_tok = t_fwd = 0.0
    padded = real = 0
    flat = []
    for g in groups:
        t = time.perf_counter()
        enc = tok([q for q, _ in g], [d for _, d in g], padding=True, truncation="longest_first",
                  max_length=cfg["max_len"], return_tensors="pt")
        t_tok += time.perf_counter() - t
        padded += enc["input_ids"].numel()
        real += int(enc["attention_mask"].sum())
        t = time.perf_counter()
        with torch.inference_mode():
            logits = net(**enc).logits
        t_fwd += time.perf_counter() - t
        flat.extend(logits.view(-1).tolist())
    scores = [0.0] * len(pairs)
    for pos, i in enumerate(order):
        scores[i] = flat[pos]
    return t_tok, t_fwd, scores, padded, real


def profile(candidates_path: str) -> dict:
    t = time.perf_counter()
    import torch
    t = mark("import_torch", t)
    from sentence_transformers import CrossEncoder, SentenceTransformer
    t = mark("import_sentence_transformers", t)
    import chromadb
    t = mark("import_chromadb", t)
    from src.shared.paths import CHROMA_COLLECTION, CHROMA_DIR
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")  # Macs default to MPS (GPU)
    t = mark("load_reranker", t)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    t = mark("load_embedder", t)
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(CHROMA_COLLECTION)
    col.count()
    t = mark("open_chroma", t)
    embedder.encode(["warm up"])
    t = mark("first_query_embedding", t)
    startup["total_in_process"] = round(time.perf_counter() - T0, 3)

    # A cold `import main` in a fresh interpreter: what gunicorn pays before it can serve.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "x"),
           "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "x"), "DEEPSEEK_MODEL": "x",
           "DEEPSEEK_BASE_URL": "http://localhost", "LANGSMITH_TRACING": "false"}
    out = subprocess.run([sys.executable, "-c", "import time; t=time.perf_counter(); import main; "
                          "print(round(time.perf_counter()-t, 3))"], capture_output=True, text=True, env=env)
    startup["app_boot_import_main"] = float(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else out.stderr[-300:]

    wanted = json.load(open(candidates_path))
    searches = {}
    for q, asins in wanted.items():
        got = col.get(ids=asins, include=["metadatas"])
        searches[q] = [(q, m["title"]) for m in got["metadatas"]]

    # First inference after load is slower (lazy init, allocator warm-up) -- the cold request pays it.
    torch.set_num_threads(1)
    t = time.perf_counter()
    model.predict(searches["water bottle"][:32], batch_size=32, show_progress_bar=False)
    first_call = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    model.predict(searches["water bottle"][:32], batch_size=32, show_progress_bar=False)
    second_call = round(time.perf_counter() - t, 3)

    configs = [PROD] + [
        {**PROD, "threads": 2},
        {**PROD, "batch": 32},
        {**PROD, "sort": True},
        {**PROD, "max_len": 128},
        {"threads": 2, "batch": 32, "sort": True, "max_len": 512},
        {"threads": 2, "batch": 32, "sort": True, "max_len": 128},
    ]
    per_search = {}
    for q, pairs in searches.items():
        lens = [len(model.tokenizer(a, b, truncation=True, max_length=512)["input_ids"]) for a, b in pairs]
        reps = 3 if len(pairs) <= 300 else 1
        rows = []
        base_scores = None
        for cfg in configs if len(pairs) <= 700 else [PROD, configs[-2], configs[-1]]:
            runs = [run_config(model, torch, pairs, cfg) for _ in range(reps)]
            t_tok = statistics.median(r[0] for r in runs)
            t_fwd = statistics.median(r[1] for r in runs)
            scores, padded, real = runs[0][2], runs[0][3], runs[0][4]
            if base_scores is None:
                base_scores = scores
            flips = sum((a > 0) != (b > 0) for a, b in zip(scores, base_scores))
            drift = max((abs(a - b) for a, b in zip(scores, base_scores)), default=0.0)
            rows.append({**cfg, "tokenize_s": round(t_tok, 3), "forward_s": round(t_fwd, 3),
                         "total_s": round(t_tok + t_fwd, 3), "padding_waste_pct": round(100 * (1 - real / padded), 1),
                         "match_flips_vs_prod": flips, "max_score_drift": round(drift, 4)})
        t = time.perf_counter()
        torch.set_num_threads(1)
        model.predict(pairs, batch_size=128, show_progress_bar=False)
        predict_s = round(time.perf_counter() - t, 3)
        per_search[q] = {"titles": len(pairs), "tokens_p50": pct(lens, 50), "tokens_p95": pct(lens, 95),
                         "tokens_max": max(lens), "titles_over_128_tokens": sum(n > 128 for n in lens),
                         "prod_predict_call_s": predict_s, "configs": rows}

    # Two searches reranking at once in one process (the service runs one gunicorn worker,
    # so this only happens if threads are added -- measures how they'd share 2 vCPUs).
    concurrency = {}
    pairs = searches["water bottle"]
    for threads in (1, 2):
        torch.set_num_threads(threads)
        t = time.perf_counter()
        model.predict(pairs, batch_size=128, show_progress_bar=False)
        solo = time.perf_counter() - t
        times = []

        def one():
            s = time.perf_counter()
            model.predict(pairs, batch_size=128, show_progress_bar=False)
            times.append(time.perf_counter() - s)
        ths = [threading.Thread(target=one) for _ in range(2)]
        [th.start() for th in ths]
        [th.join() for th in ths]
        concurrency[f"torch_threads_{threads}"] = {"solo_s": round(solo, 3), "two_at_once_each_s": [round(x, 3) for x in times]}

    return {"cpu_count": os.cpu_count(), "torch_version": torch.__version__,
            "startup_s": startup, "first_inference_32_titles_s": first_call,
            "second_inference_32_titles_s": second_call, "per_search": per_search, "concurrency": concurrency}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump-candidates":
        dump_candidates(sys.argv[2])
    else:
        print(json.dumps(profile(sys.argv[1]), indent=1))
