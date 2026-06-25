#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/gui-project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/person3}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct}"
PY="${PY:-python}"
TEST_FILE="${TEST_FILE:-${PROJECT_ROOT}/data/final/screenspot_eval.jsonl}"
ADAPTER="${ADAPTER:-${PROJECT_ROOT}/checkpoints/final/b1_hard_point}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/outputs/predictions/final/final_c2f_retry_screenspot.jsonl}"

cd "${REPO_DIR}"

"${PY}" -m src.final_experiments.infer_final_qwen3vl \
  --model "${MODEL_DIR}" \
  --adapter "${ADAPTER}" \
  --input "${TEST_FILE}" \
  --mode final \
  --output "${OUTPUT}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --max-pixels "${MAX_PIXELS:-401408}" \
  --crop-ratio "${CROP_RATIO:-0.3}" \
  --retry-samples "${RETRY_SAMPLES:-3}" \
  --retry-temperature "${RETRY_TEMPERATURE:-0.7}" \
  --max-retries "${MAX_RETRIES:-1}" \
  --resume \
  "$@"
