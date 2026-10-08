import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.agent_pipeline.analysis_agent import (
    BOTTOM_LINE_SYSTEM, HAIKU_MODEL, SYSTEM, analyze, write_bottom_line, write_report,
)
from src.shared import result_cache
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Repeat searches are served from here instead of re-running retrieval or Haiku.
# Firestore-backed on Cloud Run (CACHE_BACKEND=firestore), memory-only locally.
cache = result_cache.from_env()

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://",
)

MAX_QUERY_CHARS = 1000        # max characters per search -- prevents token bombs
MAX_REPORT_BYTES = 64 * 1024  # a real niche_report() is a few KB; anything larger isn't one
REPORT_KEYS = {"concept", "categories", "trend_by_theme", "top_sellers"}  # fields the prompts rely on

# The bottom line and the full report share one budget: each search triggers one bottom
# line automatically, and the full report only on request.
writer_limit = limiter.shared_limit("20 per minute; 120 per hour", scope="writers")


class BadRequest(Exception):
    pass


@app.errorhandler(BadRequest)
def bad_request(e):
    return jsonify({"error": str(e)}), 400


def _query_messages(data: dict) -> list[dict]:
    """The page sends one search query; the agent works on a one-message conversation."""
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise BadRequest("Please enter a product idea.")
    if len(query) > MAX_QUERY_CHARS:
        raise BadRequest(f"Please keep your search under {MAX_QUERY_CHARS} characters.")
    return [{"role": "user", "content": query.strip()}]


def _report(data: dict) -> dict:
    """
    The niche_report() dict the page got from /api/analyze, sent back so the writers narrate
    exactly the numbers on screen -- no re-run, and any Cloud Run instance can serve it.
    Editing it only changes the requester's own summary.
    """
    report = data.get("report")
    if not isinstance(report, dict) or not REPORT_KEYS <= report.keys() or "error" in report:
        raise BadRequest("Missing or invalid report. Run the search again.")
    if len(json.dumps(report)) > MAX_REPORT_BYTES:
        raise BadRequest("Report too large.")
    return report


@app.route("/health")
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("10 per minute; 30 per hour")
def analyze_route():
    """Fast path: dashboard data only. Out: {clarify} or {report, data}."""
    messages = _query_messages(request.get_json(force=True))
    # Always the structured pipeline (TypeSafe Jev path classification + title filter).
    return jsonify(analyze(messages, mode="structured", cache=cache))


def _cached_prose(kind: str, writer, prompt: str, data: dict) -> str:
    """Prose for this exact (query, report), written by Haiku once and then served from the
    cache. Keyed on the report the client sent, so an edited report only ever maps to
    its own entry -- it can't change what anyone else is served."""
    messages, report = _query_messages(data), _report(data)
    key = result_cache.prose_key(kind, messages, report, prompt, HAIKU_MODEL)
    hit = cache.get("prose", key)
    if hit is not None:
        return hit["text"]
    text = writer(messages, report)
    if text:  # an empty reply is a failure to retry next time, not an answer to keep
        cache.put("prose", key, {"text": text})
    return text


@app.route("/api/bottom-line", methods=["POST"])
@writer_limit
def bottom_line_route():
    data = request.get_json(force=True)
    return jsonify({"text": _cached_prose("bottom_line", write_bottom_line, BOTTOM_LINE_SYSTEM, data)})


@app.route("/api/report", methods=["POST"])
@writer_limit
def report_route():
    data = request.get_json(force=True)
    return jsonify({"text": _cached_prose("report", write_report, SYSTEM, data)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
