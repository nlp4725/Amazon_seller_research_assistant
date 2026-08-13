## Amazon Seller Research Assistant — ML End-to-End Project

## Project Overview

Amazon Seller Research Assistant is an AI-powered tool that helps third-party Amazon sellers research niches before launching a product. It's built as a **multi-agent system**: a Claude-powered market research agent (`chat_engine.py`) retrieves semantically relevant products from ChromaDB and synthesizes them into a structured report, and a second set of cooperating agents (`orchestrator_agent.py` + `classify_agent.py` in `src/retrieval_pipeline/`) handles the "how many X" / "find every X" counting logic the report agent depends on. The dataset covers 61,635 product launches from 2024–2026. The project follows ML engineering best practices with modular pipelines, containerization, Google Cloud deployment, and a two-service architecture separating the Flask API from the React frontend.

The retrieval-agent track is a second, in-progress part of this project: fixing and benchmarking the counting logic — see `src/retrieval_pipeline/` and `evaluator/` below.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Ingest → Preprocess → Retrieve/Serve (report agent) → Evaluate (retrieval benchmarking)`

### Core Modules

- **`src/agent_pipeline/`**: Claude-powered niche research agent
  - `chat_engine.py`: Sends conversation history + tool schema to Claude; on `tool_use`, runs `niche_report()` locally (ChromaDB semantic search + KMeans clustering + seller stats), sends result back to Claude for narrative generation. Its product-subset lookup is still capped at `top_n=200` (a retrieval-size artifact, not a real count) -- `src/retrieval_pipeline/` below is the uncapped replacement candidate, not yet wired in.

- **`src/retrieval_pipeline/`**: A multi-agent system with two competing pipeline configurations for "how many X" / "find every X" queries, benchmarked against each other -- see `retrieval_pipeline.md` for the full write-up (diagrams, file-by-file responsibilities)
  - `main_1.py` ("simple"): `orchestrator_agent` picks 1-2 root categories (1 LLM call) → vector-rank every item in them (no cap) → cross-encoder rerank, keep `score > 0`
  - `main_2.py` ("structured"): `orchestrator_agent` picks categories → `classify_agent` scores every real `category_path` under each into confident/ambiguous/not_match (LLM) → confident paths trusted outright, only the ambiguous residual gets reranked -- two cooperating agents in sequence, each with a distinct, narrow job
  - `orchestrator_agent.py` / `classify_agent.py`: the two LLM-calling agents, shared by both mains
  - `reranker.py`: cross-encoder (`ms-marco-MiniLM-L-6-v2`) title scorer, `score > 0` = match
  - `candidates.py`: shared ChromaDB retrieval helpers (vector-rank for main_1, exact category_path filter for main_2)
  - `build_chroma.py`: Embeds product titles with `all-MiniLM-L6-v2` and stores them in ChromaDB
  - `llm_client.py`: Shared DeepSeek client wrapped once for LangSmith tracing, plus a token-usage/cost accumulator both mains read after a run

- **`src/feature_pipeline/`**: Data loading and preprocessing (shared foundation for both the report agent and the retrieval pipeline)
  - `load.py`: Loads raw parquet or SQLite data
  - `preprocessing.py`: Extracts price, seller, title, review velocity, and real Amazon category paths from raw data -- produces `preprocessed_reduced.parquet`, the source both `build_chroma.py` and `classify_agent.py` read from

- **`src/data_collection_pipeline/`**: Raw data ingestion
  - `ingest.py`: Loads product listing data from SQLite into a flat DataFrame
  - `scraper.py`: Data collection utilities

- **`src/shared/`**: Shared utilities
  - `model_loader.py`: SentenceTransformer singleton — loads `all-MiniLM-L6-v2` once per session and reuses it across all chat requests

- **`evaluator/`**: Benchmarks `src/retrieval_pipeline/`'s two approaches against hand-verified ground truth -- see "Evaluation Methodology" below
  - `golden_dataset.json`: Ground truth for 20 queries
  - `build_ground_truth.py`: Reusable ground-truth builder -- keyword sweep over the parquet (independent of the pipeline being evaluated) + batched LLM judging
  - `metrics.py`: Precision/recall/f1 for a pipeline run, by ASIN-set overlap against `golden_dataset.json`
  - `langsmith_timing.py`: Per-step latency/cost read directly from LangSmith's trace record, not self-measured timers
  - `comparison.py`: Runs both approaches over the golden dataset, scores each, saves a JSON + markdown report to `evaluator/results/`

### Web Applications

- **`main.py`**: Flask backend
  - `GET /health` — health check
  - `POST /api/chat` — accepts `{"messages": [...]}`, runs the Claude niche research agent, returns `{"reply": "..."}`

- **`frontend/`**: React (Vite) frontend
  - Conversational niche research — ask about any Amazon category or subcategory; results displayed in a side panel with download and email options
  - Empty-state layout centers the greeting + input in the viewport; once a conversation starts it switches to a top-anchored scrolling message list with the input pinned to the bottom (ChatGPT-style)
  - Calls the backend via relative `/api/...` paths only — never a hardcoded URL. In dev, Vite's dev-server proxy forwards `/api` to `http://localhost:8080`; in prod, nginx (baked into the container) proxies `/api` to the backend Cloud Run service. This means **no CORS configuration exists anywhere** — the browser only ever talks to one origin.

### Cloud Infrastructure & Deployment

- **Google Cloud Run**: Two separate services — Flask backend (port 8080) and the React frontend, served by nginx (port 8080)
- **Google Secret Manager**: Stores `ANTHROPIC_API_KEY`; injected into the backend Cloud Run service at deploy time
- **Cloud Build**: CI/CD trigger on push to `main` — builds both Docker images, pushes to Artifact Registry, deploys API first, captures its URL, deploys frontend with `API_URL` wired automatically (nginx substitutes it into its reverse-proxy config at container startup via `envsubst`)
- **Terraform**: All infrastructure defined as code in `terraform/main.tf`

#### Cloud Run Services
- **seller-assistant-api**: Flask backend — (URL available on request)
- **seller-assistant**: React frontend (nginx) — (URL available on request)

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Local Development
```bash
# Add ANTHROPIC_API_KEY to .env
cp .env.example .env

# Terminal 1 — Flask backend
python main.py

# Terminal 2 — React frontend (proxies /api to localhost:8080 automatically)
cd frontend
npm install
npm run dev
```

### Retrieval Pipeline
```bash
# Build/update ChromaDB from preprocessed data
python src/retrieval_pipeline/build_chroma.py

# Run either approach on a query
python -m src.retrieval_pipeline.main_1 "dog drinking bowl"
python -m src.retrieval_pipeline.main_2 "dog drinking bowl"

# Benchmark both approaches against the golden dataset
python -m evaluator.comparison
```

### Testing
```bash
# Run all tests
pytest

# Run specific modules
pytest tests/test_chat_engine.py
pytest tests/test_build_chroma.py
pytest tests/test_preprocessing.py

# Verbose output
pytest -v
```

### Docker
```bash
# Build images
docker build -f Dockerfile_backend -t seller-assistant-api .
docker build -f Dockerfile_frontend -t seller-assistant-app .

# Run backend
docker run -p 8080:8080 --env-file .env seller-assistant-api

# Run frontend (API_URL is the backend's URL; nginx proxies /api to it)
docker run -p 8081:8080 -e API_URL=http://host.docker.internal:8080 seller-assistant-app
```

### Infrastructure
```bash
# Provision Google Cloud infrastructure
cd terraform
terraform init
terraform apply

# Trigger a manual Cloud Build deploy
gcloud builds submit --config cloudbuild.yaml
```

## Key Design Patterns

### Agentic RAG
The chat agent is an agentic RAG system. The **retrieval** step uses ChromaDB to fetch semantically relevant product launches at query time — not static context. The **agentic** layer is Claude autonomously deciding when to call `niche_report`, which category and concept arguments to pass, and how to synthesize the structured JSON result into a market research narrative. The agent can loop (while `stop_reason == "tool_use"`) if multiple tool calls are needed.

### Two-Step Claude Tool Use
The chat agent makes two Anthropic API calls per message. The first call returns `stop_reason="tool_use"` with structured arguments (category, concept). `niche_report()` runs locally — ChromaDB semantic search, KMeans clustering, seller stats — and the JSON result is sent back in a second API call so Claude can write the narrative report.

### Semantic Search with ChromaDB
Product titles are embedded with `all-MiniLM-L6-v2` (384-dim) and stored in ChromaDB. At query time, the user's concept (e.g. "dog grooming") is embedded and searched by cosine similarity, filtered by category. This lets Claude surface closely related products without exact keyword matching.

### KMeans Theme Clustering
Products in the result set are clustered by embedding similarity to identify distinct product themes. Each cluster gets year-over-year launch counts, representative titles, and an average price — giving sellers a signal on which themes are rising or declining.

### Two Retrieval Approaches: Rank+Rerank vs. Structured Filter+Rerank
The counting bug in `chat_engine.py` (`top_n` capped at 200 regardless of the true match count) motivated building two competing replacements in `src/retrieval_pipeline/`, benchmarked against each other rather than assumed correct:
- **Simple** (`main_1.py`): vector-rank every item in the picked root categories (no cap), then cross-encoder rerank, keep `score > 0`. Cheap (one LLM call total) but loses precision on broad categories.
- **Structured** (`main_2.py`): an LLM (`classify_agent`) scores every *real* Amazon `category_path` into confident/ambiguous/not_match; confident paths are trusted as an exact filter (no reranking), only the ambiguous residual gets reranked. More expensive (one LLM pass per candidate category) but expected to win precision/recall when a real category path cleanly identifies the query concept.

See `retrieval_pipeline.md` for diagrams and the full file-by-file breakdown.

### Self-Contained Docker Images
ChromaDB data is baked into the backend Docker image at build time — no GCS bucket or external storage needed at inference.

### CI/CD API URL Wiring
`cloudbuild.yaml` deploys the backend service first, captures its stable Cloud Run URL with `gcloud run services describe`, then injects it as `API_URL` into the frontend service. No manual URL updates are needed between deploys.

## Evaluation Methodology

`evaluator/golden_dataset.json` holds hand-verified ground truth for 20 queries spanning different categories and true-count magnitudes (0 up to several hundred), used to score `main_1.py` vs `main_2.py` on precision/recall/f1/latency/cost rather than trusting either implementation by inspection.

**How the ground truth is built** (`evaluator/build_ground_truth.py`):
1. A recall-oriented keyword regex sweep over `preprocessed_reduced.parquet`, scoped to the query's relevant categories — independent of the pipeline being evaluated, to avoid circularity (a keyword match isn't a model prediction).
2. Every candidate is judged genuine-match or not, with a **reason** and a **confidence** score per item, not just a bare true/false — so a judgment call is auditable, not opaque.
3. Each entry records **every title considered**, not just the confirmed matches (`titles_found`, each `{asin, title, is_match, reason, confidence}`), plus `titles_found_count` and `match_count` — the same schema a pipeline run itself produces, so `metrics.py` can score either one directly by ASIN-set overlap.

Query entries carry a `status`:
- `verified` — exhaustively judged (or a fully-covered small category)
- `verified_partial` — ground truth built from a sample, not an exhaustive enumeration (e.g. "summer dress", where the keyword sweep alone returns thousands of candidates); recall against these is a lower bound, not exact
- `reconciled` — ground truth cross-checked against an independent multi-pass judging run and reconciled by majority vote, for entries where a first single-pass build turned out to have a real gap (e.g. restricting the keyword sweep to one category when genuine matches existed in others)

**How a pipeline run is scored** (`evaluator/metrics.py`): precision/recall/f1 computed by comparing the set of ASINs a `main_1`/`main_2` run tagged `is_match=true` against the golden dataset's confirmed-match ASIN set for the same query — this works even though the two candidate pools come from independent retrieval, since only ASIN membership is compared, not list order or overlap in what was considered.

**How latency/cost are measured** (`evaluator/langsmith_timing.py`): read directly from each run's LangSmith trace record (every LLM call and every `@traceable` function reports its own timing/token-usage regardless of how it was invoked, including from inside threaded batch calls), not from self-measured `time.perf_counter()` timestamps — an independent source of truth, cross-checked against each main's own in-process usage accumulator.

`evaluator/comparison.py` ties it together: runs both approaches over every query in the golden dataset, scores each with `metrics.py`, pulls the LangSmith timing/cost breakdown, and saves a full JSON + markdown report to `evaluator/results/`.

## Dependencies

Key production dependencies (see `requirements.txt`):
- **AI/ML**: `anthropic>=0.40.0`, `openai>=1.50.0` (DeepSeek, OpenAI-compatible), `sentence-transformers>=2.7.0`, `scikit-learn`, `chromadb`, `langsmith`
- **Backend**: `flask>=3.0.0`, `gunicorn>=21.0.0`
- **Frontend** (see `frontend/package.json`): `react`, `react-dom`, `react-markdown`, `remark-gfm`, `vite`
- **Data**: `pandas`, `numpy`, `joblib`
- **Config**: `python-dotenv`, `requests`

## File Structure Notes

- **`data/raw/chroma_db/`**: ChromaDB vector store — gitignored, baked into backend Docker image at build time
- **`data/processed/`**: Pipeline checkpoint parquets — gitignored
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Unit and integration tests for each pipeline component
- **`terraform/main.tf`**: All Google Cloud infrastructure as code
- **`cloudbuild.yaml`**: CI/CD pipeline — builds both images, deploys both services
- **`Dockerfile_backend`**: Flask + gunicorn container
- **`Dockerfile_frontend`**: Multi-stage build — `node` builds the static Vite bundle, then `nginx:alpine` serves it and reverse-proxies `/api/*` to the backend (config templated from `frontend/nginx/default.conf.template` via `envsubst` at container startup, reading the `API_URL` env var)
