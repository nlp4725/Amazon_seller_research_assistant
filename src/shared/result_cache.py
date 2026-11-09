"""
Result cache for repeat searches, so the same question doesn't re-run retrieval or re-pay
for Haiku prose.

Three kinds of entry, all JSON:
  - analysis/{query_key}    -> {"concept_key": ...} or {"clarify": ...}
                               the query layer: a hit skips even cat_selector's DeepSeek call
  - analysis/{concept_key}  -> {"report": ..., "data": ...}, analyze()'s full output
                               the concept layer: different wordings that cat_selector
                               resolves to the same concept + categories share one entry
  - prose/{prose_key}       -> {"text": ...}, one Haiku bottom line or full report

Freshness: entries are never updated, they're made unreachable. Every key folds in what
the cached value depends on, so a change produces new keys and old entries are never
read again:
  - dataset_version()  hash of the preprocessed parquet -- changes on every data rebuild
  - CACHE_VERSION      bump by hand when retrieval/report logic changes results
  - prose keys         also hash the writer's system prompt and model, so a prompt edit
                       invalidates prose automatically
Unreachable Firestore documents are deleted by a TTL policy on expires_at (terraform/main.tf).

Prose keys include the query text, not just the report. The writer endpoints take the
query straight from the client without going through cat_selector, so keying prose on the
report alone would let one prompt-injected query write the prose every later searcher of
that report is served.

Two layers: a bounded in-process LRU in front (free, per instance, lost on restart), and
optionally Firestore behind it (CACHE_BACKEND=firestore) so entries survive restarts and
are shared across Cloud Run instances. The cache never raises: a failed read is a miss
and a failed write is skipped -- a cache outage costs a recompute, never a request.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from src.shared.paths import PREPROCESSED_PARQUET

log = logging.getLogger(__name__)

CACHE_VERSION = 1        # bump when retrieval or report logic changes what a query returns
TTL_DAYS = 90            # Firestore deletes entries this long after they were written
MEMORY_MAX_ITEMS = 512   # in-process LRU size, per instance
MAX_DOC_BYTES = 900_000  # Firestore caps a document at 1 MiB; anything near that is skipped
BACKEND_DEADLINE_S = 2.0 # a cache read slower than this is a miss: recomputing beats waiting
BREAKER_COOLDOWN_S = 30  # after a backend failure, skip it this long instead of waiting on each call


# ---------- keys ----------

def normalize_query(text: str) -> str:
    """"  Silicone ICE-tray! " -> "silicone ice tray": case, punctuation, spacing don't matter."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


@lru_cache(maxsize=1)
def dataset_version() -> str:
    """Short content hash of the preprocessed parquet. Hashed rather than read by mtime:
    a GCS restore into the build context doesn't reliably preserve mtimes. The ChromaDB
    store is left out because its sqlite file is rewritten on open, so it would hash
    differently on every instance; build_data.py rebuilds both stores together."""
    try:
        h = hashlib.sha256()
        with open(PREPROCESSED_PARQUET, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return "no-data"


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def query_key(messages: list[dict], mode: str) -> str:
    convo = [(m["role"], normalize_query(m["content"])) for m in messages]
    return _digest("query", CACHE_VERSION, dataset_version(), convo, mode)


def concept_key(concept: str | None, categories: list[dict], mode: str, n_clusters: int) -> str:
    names = sorted(c["category"] for c in categories)
    return _digest("concept", CACHE_VERSION, dataset_version(), normalize_query(concept or ""),
                   names, mode, n_clusters)


def prose_key(kind: str, messages: list[dict], report: dict, prompt: str, model: str) -> str:
    convo = [(m["role"], normalize_query(m["content"])) for m in messages]
    return _digest("prose", kind, convo, report, hashlib.sha256(prompt.encode()).hexdigest(), model)


# ---------- stores ----------

class MemoryStore:
    """Bounded LRU. Values are stored as JSON strings, so a hit is a fresh copy and
    behaves exactly like a Firestore round-trip (tuples come back as lists, etc.)."""

    def __init__(self, max_items: int = MEMORY_MAX_ITEMS):
        self._items: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def get(self, collection: str, key: str) -> dict | None:
        with self._lock:
            payload = self._items.get((collection, key))
            if payload is None:
                return None
            self._items.move_to_end((collection, key))
        return json.loads(payload)

    def put(self, collection: str, key: str, value: dict) -> None:
        payload = json.dumps(value)
        with self._lock:
            self._items[(collection, key)] = payload
            self._items.move_to_end((collection, key))
            while len(self._items) > self._max:
                self._items.popitem(last=False)


class FirestoreStore:
    """One document per entry: {payload: JSON string, expires_at}. The payload is a string
    rather than native fields because Firestore rejects nested arrays."""

    def __init__(self, client=None):
        if client is None:
            from google.cloud import firestore  # imported lazily: only needed when enabled
            client = firestore.Client()
        self._db = client

    def get(self, collection: str, key: str) -> dict | None:
        snap = self._db.collection(collection).document(key).get()
        return json.loads(snap.get("payload")) if snap.exists else None

    def put(self, collection: str, key: str, value: dict) -> None:
        payload = json.dumps(value)
        if len(payload.encode()) > MAX_DOC_BYTES:
            log.warning("cache: %s entry is %d bytes, too large for Firestore; not stored",
                        collection, len(payload))
            return
        self._db.collection(collection).document(key).set({
            "payload": payload,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=TTL_DAYS),
        })


class ResultCache:
    """Memory in front of an optional durable store. Never raises, and never waits on the
    store for more than BACKEND_DEADLINE_S.

    Circuit breaker: one backend failure or timeout opens the circuit, and for
    BREAKER_COOLDOWN_S every call skips the backend (memory only) instead of each request
    paying its own deadline -- without this, a hung Firestore cost every search one deadline
    per cache lookup (three on a miss). After the cool-down the next call tries again."""

    def __init__(self, backend=None, memory: MemoryStore | None = None):
        self._memory = memory or MemoryStore()
        self._backend = backend
        # A timed-out call can't be cancelled, only abandoned; a small pool bounds how many
        # can pile up while the circuit is closed.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache") if backend else None
        self._open_until = 0.0

    def _call(self, op: str, collection: str, *args):
        """Run one backend call under the deadline and the breaker. Raises on any failure."""
        if time.monotonic() < self._open_until:
            raise RuntimeError("circuit open")
        try:
            return self._pool.submit(getattr(self._backend, op), collection, *args).result(BACKEND_DEADLINE_S)
        except Exception:
            self._open_until = time.monotonic() + BREAKER_COOLDOWN_S
            raise

    def get(self, collection: str, key: str) -> dict | None:
        hit = self._memory.get(collection, key)
        if hit is not None or self._backend is None:
            return hit
        try:
            hit = self._call("get", collection, key)
        except Exception as e:
            log.warning("cache: read from %s failed (%r); treating as a miss", collection, e)
            return None
        if hit is not None:
            self._memory.put(collection, key, hit)
        return hit

    def put(self, collection: str, key: str, value: dict) -> None:
        self._memory.put(collection, key, value)
        if self._backend is None:
            return
        try:
            self._call("put", collection, key, value)
        except Exception as e:
            log.warning("cache: write to %s failed (%r); kept in memory only", collection, e)


def from_env() -> ResultCache:
    """CACHE_BACKEND=firestore on Cloud Run; anything else (the local default) is memory only.
    A Firestore client that can't be built falls back to memory rather than failing startup."""
    if os.environ.get("CACHE_BACKEND", "memory").lower() == "firestore":
        try:
            return ResultCache(FirestoreStore())
        except Exception as e:
            log.warning("cache: Firestore unavailable (%r); using memory only", e)
    return ResultCache()
