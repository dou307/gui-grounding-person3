#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/gui-project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/person3}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct}"
PY="${PY:-python}"
TEST_FILE="${TEST_FILE:-${PROJECT_ROOT}/data/final/screenspot_eval.jsonl}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/outputs/predictions/final/b0_screenspot.jsonl}"

cd "${REPO_DIR}"

"${PY}" -m src.person3.infer_qwen3vl \
  --model "${MODEL_DIR}" \
  --input "${TEST_FILE}" \
  --method direct \
  --output "${OUTPUT}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --max-pixels "${MAX_PIXELS:-401408}" \
  --resume \
  "$@"
