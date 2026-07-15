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
  torch==2.0.0 torchvision==0.15.0 \
  --index-url https://download.pytorch.org/whl/cu117
conda run -n "${ENV_NAME}" python -m pip install -r "${DINOV2_ROOT}/requirements.txt"
# DINOv2 originally pinned mmcv-full 1.5.0 with mmsegmentation 0.27.0,
# but OpenMMLab does not publish a Python 3.9 / Torch 2.0 / CUDA 11.7 wheel
# for that MMCV version. Use the newest compatible MMSeg 0.x pair available
# as a prebuilt wheel, and forbid a silent fallback to a source build.
conda run -n "${ENV_NAME}" python -m pip install \
  mmsegmentation==0.30.0 \
  mmcv-full==1.7.2 \
  --only-binary=mmcv-full \
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

PYTHONPATH="${DINOV2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
conda run -n "${ENV_NAME}" python -c '
import mmcv
import mmseg
import torch
import torchvision
import mmcv._ext
import dinov2.eval.segmentation.models
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("mmcv", mmcv.__version__)
print("mmseg", mmseg.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available on this node", torch.cuda.is_available())
'

printf 'DINOv2 root: %s\ncheckpoint directory: %s\nconda environment: %s\n' \
  "${DINOV2_ROOT}" "${CHECKPOINT_DIR}" "${ENV_NAME}"
