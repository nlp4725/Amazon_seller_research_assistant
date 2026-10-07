import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.agent_pipeline.analysis_agent import run_chat
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://",
)

MAX_MESSAGE_CHARS = 1000  # max characters per user message — prevents token bombs
MAX_HISTORY       = 10    # max messages sent to Claude — prevents chained manipulation


@app.route("/health")
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
@limiter.limit("10 per minute; 30 per hour")
def chat():
    data = request.get_json(force=True)
    messages = data.get("messages", [])

    if messages:
        last = messages[-1]
        if last.get("role") == "user" and len(last.get("content", "")) > MAX_MESSAGE_CHARS:
            return jsonify({"error": "Message too long. Please keep your query under 1000 characters."}), 400

    messages = messages[-MAX_HISTORY:]
    mode = data.get("mode", "structured")  # "structured" = Jev classify + filter; the UI always sends it, "simple" kept for the evaluator/API

    result = run_chat(messages, mode=mode)
    return jsonify({"reply": result["reply"], "data": result["data"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
