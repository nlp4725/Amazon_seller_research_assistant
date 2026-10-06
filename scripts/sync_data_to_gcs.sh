#!/usr/bin/env bash
#
# Push the two live data stores to GCS so Cloud Build can bake them into the API
# image. data/ is gitignored, so a trigger-driven build (which clones from GitHub)
# has no data of its own -- this bucket is how it gets there.
#
# Run this after rebuilding either store with build_data.py, BEFORE pushing code
# that depends on the new data. The build always takes whatever is in the bucket.
#
#   ./scripts/sync_data_to_gcs.sh            # sync
#   ./scripts/sync_data_to_gcs.sh --dry-run  # show what would change
#
set -euo pipefail

BUCKET="${DATA_BUCKET:-gs://amazon-launch-seller-assistant-data}"
CHROMA_DIR="data/raw/chroma_db"
PARQUET="data/processed/preprocessed_reduced.parquet"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "DRY RUN — nothing will be uploaded"
fi

cd "$(dirname "$0")/.."

for path in "$CHROMA_DIR" "$PARQUET"; do
  if [[ ! -e "$path" ]]; then
    echo "error: $path is missing. Rebuild it with build_data.py first." >&2
    exit 1
  fi
done

echo "Source:"
du -sh "$CHROMA_DIR" "$PARQUET"
echo "Target: $BUCKET"
echo

# --delete-unmatched-destination-objects: ChromaDB rebuilds change which segment
# files exist; stale ones left behind would be baked into the image.
#
# The two branches are spelled out rather than collecting flags in an array:
# macOS ships bash 3.2, where expanding an empty array under `set -u` aborts.
# `gcloud storage cp` has no --dry-run at all, so it reports intent instead.
if [[ "$DRY_RUN" == true ]]; then
  gcloud storage rsync -r --delete-unmatched-destination-objects --dry-run \
    "$CHROMA_DIR" "$BUCKET/chroma_db"
  echo "Would copy $PARQUET to $BUCKET/preprocessed_reduced.parquet"
else
  gcloud storage rsync -r --delete-unmatched-destination-objects \
    "$CHROMA_DIR" "$BUCKET/chroma_db"
  gcloud storage cp "$PARQUET" "$BUCKET/preprocessed_reduced.parquet"
fi

echo
echo "Done. The next build to main will pick these up."
