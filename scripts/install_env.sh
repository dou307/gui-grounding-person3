#!/usr/bin/env bash
set -euo pipefail

QWEN_ENV_NAME="${QWEN_ENV_NAME:-qwen3vl}"
QWEN_PYTHON_VERSION="${QWEN_PYTHON_VERSION:-3.10.20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
WHEELHOUSE="${WHEELHOUSE:-${PROJECT_ROOT}/wheelhouse}"
LOCAL_REQ="${REPO_ROOT}/.requirements-local.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found." >&2
  exit 1
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "${QWEN_ENV_NAME}"
}

env_python_version() {
  conda run -n "${QWEN_ENV_NAME}" python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

if env_exists; then
  EXISTING_PY_VERSION="$(env_python_version || true)"
  if [[ "${EXISTING_PY_VERSION}" == "3.10" ]]; then
    echo "Conda env ${QWEN_ENV_NAME} already exists with Python ${EXISTING_PY_VERSION}."
  else
    echo "Removing broken ${QWEN_ENV_NAME} env: expected Python 3.10, got ${EXISTING_PY_VERSION:-unknown}."
    conda env remove -n "${QWEN_ENV_NAME}" -y
  fi
fi

if ! env_exists; then
  conda create -n "${QWEN_ENV_NAME}" "python=${QWEN_PYTHON_VERSION}" -y
fi

set +u
conda activate "${QWEN_ENV_NAME}"
set -u

PY_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "${PY_VERSION}" != "3.10" ]]; then
  echo "Expected Python 3.10 in ${QWEN_ENV_NAME}, but got Python ${PY_VERSION}." >&2
  echo "Run: conda env remove -n ${QWEN_ENV_NAME} -y && QWEN_PYTHON_VERSION=${QWEN_PYTHON_VERSION} bash scripts/install_env.sh" >&2
  exit 1
fi

python -m pip install --upgrade pip

if [[ -f requirements-lock.txt ]]; then
  python - <<'PY' > "${LOCAL_REQ}"
from pathlib import Path

for line in Path("requirements-lock.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("torch @ "):
        print("torch==2.6.0+cu118")
    elif line.startswith("torchvision @ "):
        print("torchvision==0.21.0+cu118")
    elif line.strip():
        print(line)
PY

  if [[ -d "${WHEELHOUSE}" ]] && compgen -G "${WHEELHOUSE}/*.whl" >/dev/null; then
    echo "Installing from local wheelhouse: ${WHEELHOUSE}"
    pip install --no-index --find-links "${WHEELHOUSE}" -r "${LOCAL_REQ}"
  else
    echo "Local wheelhouse not found. Installing from network once."
    echo "After this succeeds, run: bash scripts/cache_wheels.sh"
    pip install -r requirements-lock.txt
  fi
else
  pip install \
    "https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.6.0%2Bcu118-cp310-cp310-linux_x86_64.whl" \
    "https://mirrors.aliyun.com/pytorch-wheels/cu118/torchvision-0.21.0%2Bcu118-cp310-cp310-linux_x86_64.whl"

  pip install transformers==4.57.0 accelerate==1.7.0 \
    peft==0.17.1 qwen-vl-utils==0.0.14 \
    datasets pillow scipy sentencepiece pyyaml \
    -i https://repo.huaweicloud.com/repository/pypi/simple
fi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
