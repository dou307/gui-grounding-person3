#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  export PROJECT_ROOT=/home/ma-user/work/gui-project
fi

PERSON3_DIR="${PROJECT_ROOT}/person3"
TRAIN_QWEN_JSONL="${PROJECT_ROOT}/data/splits/D1-Base_train_qwen.jsonl"
VAL_JSONL="${PROJECT_ROOT}/data/splits/D1-Base_val.jsonl"

cd "${PERSON3_DIR}"

mkdir -p "${PROJECT_ROOT}/data/person3" \
  "${PROJECT_ROOT}/outputs/predictions/person3" \
  "${PROJECT_ROOT}/outputs/metrics/person3" \
  "${PROJECT_ROOT}/outputs/cases/person3"

python -m src.person3.derive_person3_from_qwen \
  --input "${TRAIN_QWEN_JSONL}" \
  --output "${PROJECT_ROOT}/data/person3/p3_direct_train.jsonl" \
  --method direct

python -m src.person3.derive_person3_from_qwen \
  --input "${TRAIN_QWEN_JSONL}" \
  --output "${PROJECT_ROOT}/data/person3/p3_region_point_train.jsonl" \
  --method region_point

python -m src.person3.derive_person3_from_qwen \
  --input "${TRAIN_QWEN_JSONL}" \
  --output "${PROJECT_ROOT}/data/person3/p3_target_region_point_train.jsonl" \
  --method target_region_point

python -m src.person3.build_qwen_data \
  --input "${VAL_JSONL}" \
  --output "${PROJECT_ROOT}/data/person3/p3_direct_val.jsonl" \
  --method direct

python -m src.person3.build_qwen_data \
  --input "${VAL_JSONL}" \
  --output "${PROJECT_ROOT}/data/person3/p3_region_point_val.jsonl" \
  --method region_point

python -m src.person3.build_qwen_data \
  --input "${VAL_JSONL}" \
  --output "${PROJECT_ROOT}/data/person3/p3_target_region_point_val.jsonl" \
  --method target_region_point

echo "Training data generated. After inference, evaluate with:"
echo "python -m src.person3.evaluate --truth ${VAL_JSONL} --pred <predictions.jsonl> --out <metrics.json>"
