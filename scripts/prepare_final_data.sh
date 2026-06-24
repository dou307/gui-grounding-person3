#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/gui-project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/person3}"
TEAM1_SPLITS="${TEAM1_SPLITS:-${PROJECT_ROOT}/team_repos/person1/data/splits}"
FINAL_DATA_DIR="${FINAL_DATA_DIR:-${PROJECT_ROOT}/data/final}"

mkdir -p "${FINAL_DATA_DIR}"

cp "${TEAM1_SPLITS}/D1-Hard_train_qwen.jsonl" "${FINAL_DATA_DIR}/D1-Hard_train_qwen.jsonl"
cp "${TEAM1_SPLITS}/screenspot_eval.jsonl" "${FINAL_DATA_DIR}/screenspot_eval.jsonl"

echo "Final data prepared:"
wc -l "${FINAL_DATA_DIR}/D1-Hard_train_qwen.jsonl" "${FINAL_DATA_DIR}/screenspot_eval.jsonl"
echo
echo "Next checks:"
echo "  train images root: ${IMAGE_ROOT:-<set IMAGE_ROOT to SeeClick/RICO image directory>}"
echo "  test images root:  ${SCREENSPOT_IMAGE_ROOT:-<set SCREENSPOT_IMAGE_ROOT to ScreenSpot image directory>}"
echo "  repo dir:          ${REPO_DIR}"
