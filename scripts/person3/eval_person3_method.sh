#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/person3/eval_person3_method.sh <direct|region_point|target_region_point>" >&2
  exit 1
fi

METHOD="$1"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ma-user/work/gui-project}"
PY="${PY:-${PROJECT_ROOT}/envs/qwen3vl/bin/python}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3-VL-4B-Instruct}"
VAL_IMAGE_ROOT="${VAL_IMAGE_ROOT:-${PROJECT_ROOT}/data/rico_imgs/combined}"
SCREENSPOT_IMAGE_ROOT="${SCREENSPOT_IMAGE_ROOT:-${PROJECT_ROOT}/data/screenspot_hf/images}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/person3/D1-Base_val_1000.jsonl}"
SCREENSPOT_FILE="${SCREENSPOT_FILE:-${PROJECT_ROOT}/data/splits/screenspot_eval.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_PIXELS="${MAX_PIXELS:-401408}"

case "${METHOD}" in
  direct)
    ADAPTER_DIR="${PROJECT_ROOT}/checkpoints/person3/p3_direct"
    PREFIX="p3_direct"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
    ;;
  region_point)
    ADAPTER_DIR="${PROJECT_ROOT}/checkpoints/person3/p3_region_point"
    PREFIX="p3_region_point"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
    ;;
  target_region_point)
    ADAPTER_DIR="${PROJECT_ROOT}/checkpoints/person3/p3_target_region_point"
    PREFIX="p3_target_region_point"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 1
    ;;
esac

cd "${PROJECT_ROOT}/person3"
mkdir -p \
  "${PROJECT_ROOT}/data/person3" \
  "${PROJECT_ROOT}/outputs/predictions/person3" \
  "${PROJECT_ROOT}/outputs/metrics/person3" \
  "${PROJECT_ROOT}/outputs/cases/person3"

if [[ ! -f "${VAL_FILE}" ]]; then
  head -n 1000 "${PROJECT_ROOT}/data/splits/D1-Base_val.jsonl" > "${VAL_FILE}"
fi

IMAGE_ROOT="${VAL_IMAGE_ROOT}" "${PY}" -m src.person3.infer_qwen3vl \
  --model "${MODEL_DIR}" \
  --adapter "${ADAPTER_DIR}" \
  --input "${VAL_FILE}" \
  --method "${METHOD}" \
  --output "${PROJECT_ROOT}/outputs/predictions/person3/${PREFIX}_val_1000.jsonl" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-pixels "${MAX_PIXELS}" \
  --resume

"${PY}" -m src.person3.evaluate \
  --truth "${VAL_FILE}" \
  --pred "${PROJECT_ROOT}/outputs/predictions/person3/${PREFIX}_val_1000.jsonl" \
  --out "${PROJECT_ROOT}/outputs/metrics/person3/${PREFIX}_val_1000_metrics.json"

IMAGE_ROOT="${VAL_IMAGE_ROOT}" "${PY}" -m src.person3.visualize_cases \
  --truth "${VAL_FILE}" \
  --pred "${PROJECT_ROOT}/outputs/predictions/person3/${PREFIX}_val_1000.jsonl" \
  --out-dir "${PROJECT_ROOT}/outputs/cases/person3/${PREFIX}_val_1000" \
  --limit 50

IMAGE_ROOT="${SCREENSPOT_IMAGE_ROOT}" "${PY}" -m src.person3.infer_qwen3vl \
  --model "${MODEL_DIR}" \
  --adapter "${ADAPTER_DIR}" \
  --input "${SCREENSPOT_FILE}" \
  --method "${METHOD}" \
  --output "${PROJECT_ROOT}/outputs/predictions/person3/${PREFIX}_screenspot.jsonl" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-pixels "${MAX_PIXELS}" \
  --resume

"${PY}" -m src.person3.evaluate \
  --truth "${SCREENSPOT_FILE}" \
  --pred "${PROJECT_ROOT}/outputs/predictions/person3/${PREFIX}_screenspot.jsonl" \
  --out "${PROJECT_ROOT}/outputs/metrics/person3/${PREFIX}_screenspot_metrics.json"

IMAGE_ROOT="${SCREENSPOT_IMAGE_ROOT}" "${PY}" -m src.person3.visualize_cases \
  --truth "${SCREENSPOT_FILE}" \
  --pred "${PROJECT_ROOT}/outputs/predictions/person3/${PREFIX}_screenspot.jsonl" \
  --out-dir "${PROJECT_ROOT}/outputs/cases/person3/${PREFIX}_screenspot" \
  --limit 50

echo "Done: ${METHOD}"
echo "Validation metrics: ${PROJECT_ROOT}/outputs/metrics/person3/${PREFIX}_val_1000_metrics.json"
echo "ScreenSpot metrics: ${PROJECT_ROOT}/outputs/metrics/person3/${PREFIX}_screenspot_metrics.json"
