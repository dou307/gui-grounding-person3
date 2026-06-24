#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/gui-project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/person3}"
PY="${PY:-python}"
TEST_FILE="${TEST_FILE:-${PROJECT_ROOT}/data/final/screenspot_eval.jsonl}"
PRED_DIR="${PRED_DIR:-${PROJECT_ROOT}/outputs/predictions/final}"
METRIC_DIR="${METRIC_DIR:-${PROJECT_ROOT}/outputs/metrics/final}"

cd "${REPO_DIR}"
mkdir -p "${METRIC_DIR}"

for name in b0_screenspot b1_screenspot final_c2f_retry_screenspot; do
  pred="${PRED_DIR}/${name}.jsonl"
  if [[ -f "${pred}" ]]; then
    "${PY}" -m src.person3.evaluate \
      --truth "${TEST_FILE}" \
      --pred "${pred}" \
      --out "${METRIC_DIR}/${name}_metrics.json"
  else
    echo "Skip missing prediction file: ${pred}"
  fi
done

"${PY}" -m src.person3.summarize_final_results \
  --metrics-dir "${METRIC_DIR}" \
  --out-md "${METRIC_DIR}/final_results_summary.md" \
  --out-csv "${METRIC_DIR}/final_results_summary.csv"
