#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dinov2_segmentation}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
DINOV2_ROOT="${2:-${WORKSPACE_ROOT}/external/dinov2}"
CHECKPOINT_DIR="${3:-${DINOV2_ROOT}/checkpoints}"
BASE_URL="https://dl.fbaipublicfiles.com/dinov2"

test -d "${DINOV2_ROOT}"
command -v conda >/dev/null 2>&1

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.9
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.0.0 torchvision==0.15.1 \
  --index-url https://download.pytorch.org/whl/cu117
conda run -n "${ENV_NAME}" python -m pip install -r "${DINOV2_ROOT}/requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install \
  mmsegmentation==0.27.0 \
  mmcv-full==1.5.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html

mkdir -p "${CHECKPOINT_DIR}"
download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ ! -s "${output}" ]]; then
    curl --fail --location --retry 3 --output "${output}.partial" "${url}"
    mv "${output}.partial" "${output}"
  fi
}

download_if_missing \
  "${BASE_URL}/dinov2_vitl14/dinov2_vitl14_pretrain.pth" \
  "${CHECKPOINT_DIR}/dinov2_vitl14_pretrain.pth"
download_if_missing \
  "${BASE_URL}/dinov2_vitl14/dinov2_vitl14_ade20k_linear_config.py" \
  "${CHECKPOINT_DIR}/dinov2_vitl14_ade20k_linear_config.py"
download_if_missing \
  "${BASE_URL}/dinov2_vitl14/dinov2_vitl14_ade20k_linear_head.pth" \
  "${CHECKPOINT_DIR}/dinov2_vitl14_ade20k_linear_head.pth"

conda run -n "${ENV_NAME}" python -c '
import mmcv
import mmseg
import torch
import torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("mmcv", mmcv.__version__)
print("mmseg", mmseg.__version__)
assert torch.cuda.is_available(), "CUDA is not available in the DINOv2 environment"
'

printf 'DINOv2 root: %s\ncheckpoint directory: %s\nconda environment: %s\n' \
  "${DINOV2_ROOT}" "${CHECKPOINT_DIR}" "${ENV_NAME}"
