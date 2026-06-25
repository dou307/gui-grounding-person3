#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/gui-project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/person3}"
PY="${PY:-python}"
FINAL_DATA_DIR="${FINAL_DATA_DIR:-${PROJECT_ROOT}/data/final}"

cd "${REPO_DIR}"

echo "Checking training images with IMAGE_ROOT=${IMAGE_ROOT:-<unset>}"
"${PY}" -m src.final_experiments.check_image_paths \
  --input "${FINAL_DATA_DIR}/D1-Hard_train_qwen.jsonl" \
  --limit "${TRAIN_CHECK_LIMIT:-100}"

if [[ -n "${SCREENSPOT_IMAGE_ROOT:-}" ]]; then
  export IMAGE_ROOT="${SCREENSPOT_IMAGE_ROOT}"
fi

echo "Checking ScreenSpot images with IMAGE_ROOT=${IMAGE_ROOT:-<unset>}"
"${PY}" -m src.final_experiments.check_image_paths \
  --input "${FINAL_DATA_DIR}/screenspot_eval.jsonl" \
  --limit "${TEST_CHECK_LIMIT:-100}"
