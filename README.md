## Amazon Seller Research Assistant — ML End-to-End Project

## Project Overview

Amazon Seller Research Assistant is an end-to-end AI-powered tool that helps third-party Amazon sellers research niches and evaluate product launch potential. It combines an **agentic RAG** system — a Claude-powered market research agent that retrieves semantically relevant products from ChromaDB and synthesizes them into a structured report — with an XGBoost classifier that scores product titles by predicted launch success. The dataset covers 61,635 product launches from 2024–2026. The project follows ML engineering best practices with modular pipelines, containerization, Google Cloud deployment, and a two-service architecture separating the Flask API from the React frontend.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Ingest → Preprocess → Feature Engineering → Train → Evaluate → Inference → Serve`

### Core Modules

- **`src/agent_pipeline/`**: Claude-powered niche research agent
  - `chat_engine.py`: Sends conversation history + tool schema to Claude; on `tool_use`, runs `niche_report()` locally (ChromaDB semantic search + KMeans clustering + seller stats), sends result back to Claude for narrative generation
  - `build_chroma.py`: Embeds product titles with `all-MiniLM-L6-v2` and stores them in ChromaDB

- **`src/inference_pipeline/`**: Launch success prediction
  - `inference.py`: `predict()` takes a list of product titles, runs them through a fitted TF-IDF + XGBoost pipeline, returns predicted probability and label per title

- **`src/feature_pipeline/`**: Data loading, preprocessing, and feature engineering
  - `load.py`: Loads raw parquet or SQLite data
  - `preprocessing.py`: Extracts price, seller, title, and review velocity from raw data
  - `feature_engineering.py`: TF-IDF vectorization, ColumnTransformer, train/test split, label generation (>10 reviews at 180 days)

- **`src/training_pipeline/`**: Model training and evaluation
  - `train.py`: XGBClassifier with threshold tuning; saves pipeline to `models/model.joblib`
  - `evaluate.py`: Computes ROC-AUC, PR-AUC, precision, recall, F1; saves results to parquet

- **`src/data_collection_pipeline/`**: Raw data ingestion
  - `ingest.py`: Loads product listing data from SQLite into a flat DataFrame
  - `scraper.py`: Data collection utilities

- **`src/shared/`**: Shared utilities
  - `model_loader.py`: SentenceTransformer singleton — loads `all-MiniLM-L6-v2` once per session and reuses it across all chat requests

### Web Applications

- **`main.py`**: Flask backend
  - `GET /health` — health check
  - `POST /api/chat` — accepts `{"messages": [...]}`, runs the Claude niche research agent, returns `{"reply": "..."}`
  - `POST /api/predict` — accepts `{"titles": [...]}`, runs XGBoost inference, returns `{"results": [...]}`

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

### Full ML Pipeline (retrain from scratch)
```bash
python pipeline.py
```

### Run Individual Pipeline Steps
```bash
# 1. Preprocess raw data
python src/feature_pipeline/preprocessing.py

# 2. Feature engineering + train/test split
python src/feature_pipeline/feature_engineering.py

# 3. Train model
python src/training_pipeline/train.py

# 4. Evaluate model
python src/training_pipeline/evaluate.py
```

### Inference
```bash
# Manual title input
python src/inference_pipeline/inference.py "Stainless Steel Air Fryer 5.8QT, Non-Stick Pan Set"

# Via API
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"titles": ["Stainless Steel Air Fryer 5.8QT"]}'

curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is trending in pet supplies?"}]}'
```

### Testing
```bash
# Run all tests
pytest

# Run specific modules
pytest tests/test_chat_engine.py
pytest tests/test_inference.py
pytest tests/test_feature_engineering.py
pytest tests/test_preprocessing.py
pytest tests/test_train.py
pytest tests/test_evaluate.py

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

### Class Imbalance Handling
Only ~12% of standalone listings accumulate >10 reviews in 180 days. The XGBoost model is trained at threshold 0.40 — tuned to balance precision and recall since both false positives (wasted launch investment) and false negatives (missed opportunities) matter.

### Self-Contained Docker Images
ChromaDB data and `model.joblib` are baked into the backend Docker image at build time — no GCS bucket or external storage needed at inference. Retraining runs `python pipeline.py` locally, then a push to `main` redeploys automatically via Cloud Build.

### CI/CD API URL Wiring
`cloudbuild.yaml` deploys the backend service first, captures its stable Cloud Run URL with `gcloud run services describe`, then injects it as `API_URL` into the frontend service. No manual URL updates are needed between deploys.

## Dependencies

Key production dependencies (see `requirements.txt`):
- **AI/ML**: `anthropic>=0.40.0`, `sentence-transformers>=2.7.0`, `scikit-learn`, `xgboost`, `chromadb`
- **Backend**: `flask>=3.0.0`, `gunicorn>=21.0.0`
- **Frontend** (see `frontend/package.json`): `react`, `react-dom`, `react-markdown`, `remark-gfm`, `vite`
- **Data**: `pandas`, `numpy`, `joblib`
- **Config**: `python-dotenv`, `requests`

## File Structure Notes

- **`data/raw/chroma_db/`**: ChromaDB vector store — gitignored, baked into backend Docker image at build time
- **`data/processed/`**: Pipeline checkpoint parquets — gitignored
- **`models/model.joblib`**: Trained XGBoost pipeline — committed (exception to gitignore)
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Unit and integration tests for each pipeline component
- **`terraform/main.tf`**: All Google Cloud infrastructure as code
- **`cloudbuild.yaml`**: CI/CD pipeline — builds both images, deploys both services
- **`Dockerfile_backend`**: Flask + gunicorn container
- **`Dockerfile_frontend`**: Multi-stage build — `node` builds the static Vite bundle, then `nginx:alpine` serves it and reverse-proxies `/api/*` to the backend (config templated from `frontend/nginx/default.conf.template` via `envsubst` at container startup, reading the `API_URL` env var)
