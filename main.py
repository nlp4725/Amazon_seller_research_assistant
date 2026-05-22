import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.agent_pipeline.chat_engine import run_chat
from src.inference_pipeline.inference import predict
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
MAX_TITLES        = 50    # max titles per predict request


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

    reply = run_chat(messages)
    return jsonify({"reply": reply})


@app.route("/api/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict_endpoint():
    data = request.get_json(force=True)
    titles = data.get("titles", [])

    if len(titles) > MAX_TITLES:
        return jsonify({"error": f"Too many titles. Please submit at most {MAX_TITLES} at a time."}), 400

    results = predict(titles)
    return jsonify({"results": results.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
