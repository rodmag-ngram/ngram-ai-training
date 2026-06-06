#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INTERVAL="${INTERVAL:-30}"
LIMIT="${LIMIT:-10}"
CHUNK_SAMPLES="${CHUNK_SAMPLES:-40000}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" scripts/process_uploaded_edfs.py \
  --loop \
  --interval "$INTERVAL" \
  --limit "$LIMIT" \
  --chunk-samples "$CHUNK_SAMPLES"
