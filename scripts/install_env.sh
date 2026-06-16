#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-qwen3vl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists."
else
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

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

