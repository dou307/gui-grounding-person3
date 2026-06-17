#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ma-user/work/gui-project}"
PY="${PY:-${PROJECT_ROOT}/envs/qwen3vl/bin/python}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_DIR="${PROJECT_ROOT}/outputs"
PACKAGE="${OUT_DIR}/person3_final_results_${STAMP}.tar.gz"
MANIFEST="${OUT_DIR}/person3_final_results_manifest_${STAMP}.txt"

cd "${PROJECT_ROOT}/person3"

"${PY}" -m src.person3.summarize_results \
  --metrics-dir "${PROJECT_ROOT}/outputs/metrics/person3" \
  --out-md "${PROJECT_ROOT}/outputs/metrics/person3/person3_results_summary.md" \
  --out-csv "${PROJECT_ROOT}/outputs/metrics/person3/person3_results_summary.csv"

if [[ -f "${PROJECT_ROOT}/outputs/predictions/person3/p3_direct_val_1000.jsonl" ]] \
  && [[ -f "${PROJECT_ROOT}/outputs/predictions/person3/p3_region_point_val_1000.jsonl" ]] \
  && [[ -f "${PROJECT_ROOT}/outputs/predictions/person3/p3_target_region_point_val_1000.jsonl" ]]; then
  "${PY}" -m src.person3.compare_predictions \
    --truth "${PROJECT_ROOT}/data/person3/D1-Base_val_1000.jsonl" \
    --pred-dir "${PROJECT_ROOT}/outputs/predictions/person3" \
    --split val_1000 \
    --out-dir "${PROJECT_ROOT}/outputs/analysis/person3"
fi

if [[ -f "${PROJECT_ROOT}/outputs/predictions/person3/p3_direct_screenspot.jsonl" ]] \
  && [[ -f "${PROJECT_ROOT}/outputs/predictions/person3/p3_region_point_screenspot.jsonl" ]] \
  && [[ -f "${PROJECT_ROOT}/outputs/predictions/person3/p3_target_region_point_screenspot.jsonl" ]]; then
  "${PY}" -m src.person3.compare_predictions \
    --truth "${PROJECT_ROOT}/data/splits/screenspot_eval.jsonl" \
    --pred-dir "${PROJECT_ROOT}/outputs/predictions/person3" \
    --split screenspot \
    --out-dir "${PROJECT_ROOT}/outputs/analysis/person3"
fi

{
  echo "Person3 final result package"
  echo "created_at=$(date)"
  echo "project_root=${PROJECT_ROOT}"
  echo
  echo "Git commit:"
  git rev-parse HEAD || true
  echo
  echo "Included files:"
  find \
    "${PROJECT_ROOT}/checkpoints/person3" \
    "${PROJECT_ROOT}/outputs/predictions/person3" \
    "${PROJECT_ROOT}/outputs/metrics/person3" \
    "${PROJECT_ROOT}/outputs/cases/person3" \
    "${PROJECT_ROOT}/outputs/analysis/person3" \
    "${PROJECT_ROOT}/outputs/run_notes/person3" \
    -type f 2>/dev/null | sort
} > "${MANIFEST}"

cd "${PROJECT_ROOT}"

tar -czf "${PACKAGE}" \
  checkpoints/person3 \
  outputs/predictions/person3 \
  outputs/metrics/person3 \
  outputs/cases/person3 \
  outputs/analysis/person3 \
  outputs/run_notes/person3 \
  outputs/"$(basename "${MANIFEST}")"

echo "Wrote ${PACKAGE}"
echo "Wrote ${MANIFEST}"
