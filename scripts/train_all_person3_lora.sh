#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/train_p3_direct_lora.sh" "$@"
bash "${SCRIPT_DIR}/train_p3_region_point_lora.sh" "$@"
bash "${SCRIPT_DIR}/train_p3_target_region_point_lora.sh" "$@"
