#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ma-user/work/gui-project}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct}"
EXTRA_ARGS=()
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--image-root "${IMAGE_ROOT}")
fi

cd "${PROJECT_ROOT}/person3"

python -m src.person3.train_lora_qwen3vl \
  --model "${MODEL_DIR}" \
  --train-file "${PROJECT_ROOT}/data/person3/p3_direct_train.jsonl" \
  --output-dir "${PROJECT_ROOT}/checkpoints/person3/p3_direct" \
  --logging-dir "${PROJECT_ROOT}/outputs/logs/person3/p3_direct" \
  "${EXTRA_ARGS[@]}" \
  "$@"
