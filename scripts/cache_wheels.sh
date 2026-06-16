#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
WHEELHOUSE="${WHEELHOUSE:-${PROJECT_ROOT}/wheelhouse}"
export WHEELHOUSE

mkdir -p "${WHEELHOUSE}"

cd "${REPO_ROOT}"

echo "Caching wheels into: ${WHEELHOUSE}"

pip download -r requirements-lock.txt -d "${WHEELHOUSE}"

python - <<'PY'
from pathlib import Path

wheelhouse = Path(__import__("os").environ.get("WHEELHOUSE", ""))
if not wheelhouse:
    raise SystemExit("WHEELHOUSE is not set")

required_prefixes = [
    "torch-2.6.0+cu118",
    "torchvision-0.21.0+cu118",
    "nvidia_cudnn_cu11-9.1.0.70",
    "nvidia_cublas_cu11-11.11.3.6",
]

missing = []
for prefix in required_prefixes:
    if not list(wheelhouse.glob(prefix + "*.whl")):
        missing.append(prefix)

if missing:
    raise SystemExit("Missing expected wheels: " + ", ".join(missing))

print("Wheelhouse check passed.")
PY

echo "Done. Future installs can run scripts/install_env.sh and use the local wheelhouse."
