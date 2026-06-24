#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/gui-project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/person3}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct}"
PY="${PY:-python}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/final/D1-Hard_train_qwen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/final/b1_hard_point}"
LOGGING_DIR="${LOGGING_DIR:-${PROJECT_ROOT}/outputs/logs/final/b1_hard_point}"

EXTRA_ARGS=()
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--image-root "${IMAGE_ROOT}")
fi

cd "${REPO_DIR}"

"${PY}" -m src.person3.train_lora_qwen3vl \
  --model "${MODEL_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --logging-dir "${LOGGING_DIR}" \
  --max-steps "${MAX_STEPS:-500}" \
  --save-steps "${SAVE_STEPS:-250}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --max-pixels "${MAX_PIXELS:-401408}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
