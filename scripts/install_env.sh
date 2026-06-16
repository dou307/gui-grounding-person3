#!/usr/bin/env bash
set -euo pipefail

QWEN_ENV_NAME="${QWEN_ENV_NAME:-qwen3vl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${QWEN_ENV_NAME}"; then
  echo "Conda env ${QWEN_ENV_NAME} already exists."
else
  conda create -n "${QWEN_ENV_NAME}" python=3.10 -y
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME}"
set -u

PY_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "${PY_VERSION}" != "3.10" ]]; then
  echo "Expected Python 3.10 in ${QWEN_ENV_NAME}, but got Python ${PY_VERSION}." >&2
  echo "Run: conda env remove -n ${QWEN_ENV_NAME} -y && bash scripts/install_env.sh" >&2
  exit 1
fi

python -m pip install --upgrade pip

if [[ -f requirements-lock.txt ]]; then
  pip install -r requirements-lock.txt
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
