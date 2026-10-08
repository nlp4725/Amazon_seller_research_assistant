#!/usr/bin/env bash
#
# Copy API keys from the local .env into Secret Manager, so the Cloud Run deploy in
# cloudbuild.yaml (--set-secrets=...:latest) can read them. Each key is piped
# straight into gcloud -- never printed, never written to a temp file.
#
# Adds a new version only when the secret is empty or the .env value differs from
# the current one, so re-running is harmless.
#
#   ./scripts/sync_secrets_from_env.sh                      # deepseek + langsmith
#   ./scripts/sync_secrets_from_env.sh ANTHROPIC_API_KEY    # any .env var(s) listed
#
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=".env"

# .env variable -> Secret Manager secret, matching --set-secrets in cloudbuild.yaml.
secret_name() {
  case "$1" in
    ANTHROPIC_API_KEY) echo "anthropic-api-key" ;;
    DEEPSEEK_API_KEY)  echo "deepseek-api-key" ;;
    LANGSMITH_API_KEY) echo "langsmith-api-key" ;;
    TYPESAFE_API_KEY)  echo "typesafe-api-key" ;;
    *) return 1 ;;
  esac
}

# Value of VAR in .env: last assignment wins, optional "export ", surrounding
# quotes and trailing CR/whitespace stripped.
env_value() {
  grep -E "^(export )?$1=" "$ENV_FILE" | tail -n 1 \
    | sed -E "s/^(export )?$1=//; s/[[:space:]]+\$//; s/^\"(.*)\"\$/\\1/; s/^'(.*)'\$/\\1/"
}

[[ -f "$ENV_FILE" ]] || { echo "No $ENV_FILE found in $(pwd)" >&2; exit 1; }
VARS=("$@")
[[ ${#VARS[@]} -gt 0 ]] || VARS=(DEEPSEEK_API_KEY LANGSMITH_API_KEY)

for var in "${VARS[@]}"; do
  secret=$(secret_name "$var") || { echo "✗ $var: no matching secret (see secret_name)" >&2; exit 1; }
  value=$(env_value "$var")
  if [[ -z "$value" ]]; then
    echo "✗ $var is missing or empty in $ENV_FILE" >&2
    exit 1
  fi

  current=$(gcloud secrets versions access latest --secret="$secret" 2>/dev/null || true)
  if [[ "$current" == "$value" ]]; then
    echo "= $secret already matches $var — skipped"
    continue
  fi

  printf '%s' "$value" | gcloud secrets versions add "$secret" --data-file=- --quiet >/dev/null
  echo "✓ $secret updated from $var (${#value} chars)"
done
