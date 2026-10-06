terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  default = "amazon-launch"
}

variable "region" {
  default = "us-central1"
}

# Look up project number dynamically — avoids hardcoding it
data "google_project" "project" {}

# Enable required APIs
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

# Artifact Registry repository
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "seller-assistant"
  format        = "DOCKER"
  depends_on    = [google_project_service.artifactregistry]
}

# ── Cloud Build IAM ───────────────────────────────────────────────────────────
# Cloud Build uses {project_number}@cloudbuild.gserviceaccount.com

resource "google_project_iam_member" "cloudbuild_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# Cloud Build needs to be able to act as the Compute SA to deploy Cloud Run
resource "google_service_account_iam_member" "cloudbuild_act_as_compute" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/${data.google_project.project.number}-compute@developer.gserviceaccount.com"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# ── Compute Engine (Cloud Run runtime) IAM ────────────────────────────────────
# Cloud Run uses {project_number}-compute@developer.gserviceaccount.com to pull images

resource "google_project_iam_member" "compute_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_service_account_iam_member" "compute_act_as_self" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/${data.google_project.project.number}-compute@developer.gserviceaccount.com"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# ── Secret Manager ────────────────────────────────────────────────────────────
# Shell created by Terraform; value added manually — never store secrets in code:
# gcloud secrets create anthropic-api-key --data-file=- <<< "$ANTHROPIC_API_KEY"

resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "anthropic-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.secretmanager]
}

# Grant Cloud Run runtime permission to read the secret
resource "google_secret_manager_secret_iam_member" "anthropic_secret_access" {
  secret_id = google_secret_manager_secret.anthropic_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# Grant Cloud Build permission to read the secret (needed for --set-secrets in cloudbuild.yaml)
resource "google_secret_manager_secret_iam_member" "cloudbuild_secret_access" {
  secret_id = google_secret_manager_secret.anthropic_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# DeepSeek powers the retrieval pipeline's LLM calls
# (src/retrieval_pipeline/llm_client.py reads DEEPSEEK_API_KEY at import time, so
# the API image will not start without it). Only the key is a secret -- the base
# URL and model name are plain env vars set in cloudbuild.yaml.
#
# Shell created here; value added manually:
#   gcloud secrets versions add deepseek-api-key --data-file=- <<< "$DEEPSEEK_API_KEY"

resource "google_secret_manager_secret" "deepseek_api_key" {
  secret_id = "deepseek-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_iam_member" "deepseek_secret_access" {
  secret_id = google_secret_manager_secret.deepseek_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_deepseek_access" {
  secret_id = google_secret_manager_secret.deepseek_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# LangSmith traces the agent pipeline (@traceable / wrap_anthropic in
# src/agent_pipeline/analysis_agent.py). Tracing is a no-op unless both
# LANGSMITH_TRACING=true and this key are present, so prod tracing is opt-in via
# cloudbuild.yaml's env vars.
#
# Shell created here; value added manually:
#   gcloud secrets versions add langsmith-api-key --data-file=- <<< "$LANGSMITH_API_KEY"

resource "google_secret_manager_secret" "langsmith_api_key" {
  secret_id = "langsmith-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_iam_member" "langsmith_secret_access" {
  secret_id = google_secret_manager_secret.langsmith_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_langsmith_access" {
  secret_id = google_secret_manager_secret.langsmith_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# TypeSafe's Jev is the retrieval pipeline's default judge: category classification
# (CLASSIFIER_BACKEND defaults to jev, src/retrieval_pipeline/jev_scorer.py) and the
# title filter (TITLE_FILTER defaults to on, src/retrieval_pipeline/title_filter.py).
# Both fail at request time without this key.
#
# Shell created here; value added manually:
#   gcloud secrets versions add typesafe-api-key --data-file=- <<< "$TYPESAFE_API_KEY"

resource "google_secret_manager_secret" "typesafe_api_key" {
  secret_id = "typesafe-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_iam_member" "typesafe_secret_access" {
  secret_id = google_secret_manager_secret.typesafe_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "cloudbuild_typesafe_access" {
  secret_id = google_secret_manager_secret.typesafe_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# ── Data bucket ───────────────────────────────────────────────────────────────
# The two stores the API reads at request time (src/shared/paths.py) are far too
# large for git and are gitignored, so a trigger-driven build -- which clones from
# GitHub -- has no data. They are staged here instead and pulled into the build
# context by cloudbuild.yaml's fetch-data step. Populate with:
#   scripts/sync_data_to_gcs.sh

resource "google_storage_bucket" "data" {
  name                        = "amazon-launch-seller-assistant-data"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  # A rebuilt ChromaDB replaces ~200MB; keep one generation back for rollback only.
  lifecycle_rule {
    condition {
      num_newer_versions = 2
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket_iam_member" "cloudbuild_data_reader" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
}

# Cloud Build's trigger runs as the Compute SA (see the trigger's serviceAccount),
# so that identity needs read access too.
resource "google_storage_bucket_iam_member" "compute_data_reader" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}

# Cloud Run services are created and updated by cloudbuild.yaml — not managed here.
# Terraform can't create them before images exist; Cloud Build deploys both services
# as part of every build: seller-assistant (frontend) and seller-assistant-api (backend).
