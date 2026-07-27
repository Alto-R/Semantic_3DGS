#!/usr/bin/env bash
set -euo pipefail

# Installs code and runtime dependencies only. Model checkpoints are never
# downloaded by this script.

ENV_NAME="${1:-dinov3_semantic}"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
RECREATE_INCOMPATIBLE_ENV="${RECREATE_INCOMPATIBLE_ENV:-0}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
DINOV3_ROOT="${2:-${WORKSPACE_ROOT}/external/dinov3}"
CHECKPOINT_DIR="${3:-${WORKSPACE_ROOT}/data/models/dinov3}"
DINOV3_REPOSITORY="${DINOV3_REPOSITORY:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_COMMIT="${DINOV3_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}"

command -v conda >/dev/null 2>&1
command -v git >/dev/null 2>&1

if [[ -e "${DINOV3_ROOT}" && ! -d "${DINOV3_ROOT}/.git" ]]; then
  echo "Refusing to replace a non-Git path: ${DINOV3_ROOT}" >&2
  exit 2
fi
if [[ ! -e "${DINOV3_ROOT}" ]]; then
  mkdir -p "$(dirname -- "${DINOV3_ROOT}")"
  git clone "${DINOV3_REPOSITORY}" "${DINOV3_ROOT}"
fi
test -z "$(git -C "${DINOV3_ROOT}" status --porcelain --untracked-files=no)"
git -C "${DINOV3_ROOT}" fetch origin "${DINOV3_COMMIT}"
git -C "${DINOV3_ROOT}" checkout --detach "${DINOV3_COMMIT}"
test "$(git -C "${DINOV3_ROOT}" rev-parse HEAD)" = "${DINOV3_COMMIT}"

environment_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

if environment_exists; then
  CURRENT_PYTHON="$(
    conda run -n "${ENV_NAME}" python -c \
      'import platform; print(platform.python_version())'
  )"
  CURRENT_TORCH="$(
    conda run -n "${ENV_NAME}" python -c \
      'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || true
  )"
  if [[ "${CURRENT_PYTHON}" != "${PYTHON_VERSION}".* \
     || "${CURRENT_TORCH}" != "${TORCH_VERSION}" ]]; then
    if [[ "${RECREATE_INCOMPATIBLE_ENV}" != "1" ]]; then
      echo "${ENV_NAME} is incompatible:" >&2
      echo "  Python ${CURRENT_PYTHON:-<missing>} (required ${PYTHON_VERSION}.x)" >&2
      echo "  PyTorch ${CURRENT_TORCH:-<missing>} (required ${TORCH_VERSION})" >&2
      echo "Set RECREATE_INCOMPATIBLE_ENV=1 to rebuild this dedicated environment." >&2
      exit 2
    fi
    conda env remove -y -n "${ENV_NAME}"
  fi
fi

if ! environment_exists; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"
conda run -n "${ENV_NAME}" python -m pip install \
  ftfy \
  iopath \
  numpy \
  omegaconf \
  pandas \
  pillow \
  regex \
  scikit-learn \
  submitit \
  termcolor \
  torchmetrics

PYTHONPATH="${DINOV3_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
conda run -n "${ENV_NAME}" python -c '
import sys
import torch
import torchvision
from dinov3.eval.segmentation.inference import make_inference
from dinov3.hub.segmentors import dinov3_vit7b16_ms

assert sys.version_info[:2] == (3, 11)
assert tuple(int(value) for value in torch.__version__.split("+")[0].split(".")[:3]) >= (2, 7, 1)
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available on this node", torch.cuda.is_available())
print("DINOv3 imports", make_inference.__name__, dinov3_vit7b16_ms.__name__)
'

mkdir -p "${CHECKPOINT_DIR}"
BACKBONE="${CHECKPOINT_DIR}/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
SEGMENTOR="${CHECKPOINT_DIR}/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"

printf 'DINOv3 root: %s\nDINOv3 commit: %s\nconda environment: %s\nPyTorch index: %s\ncheckpoint directory: %s\n' \
  "${DINOV3_ROOT}" "${DINOV3_COMMIT}" "${ENV_NAME}" \
  "${PYTORCH_INDEX_URL}" "${CHECKPOINT_DIR}"
printf 'User-provided checkpoint required: %s\n' "${BACKBONE}"
printf 'User-provided checkpoint required: %s\n' "${SEGMENTOR}"
if [[ -f "${BACKBONE}" && -f "${SEGMENTOR}" ]]; then
  sha256sum "${BACKBONE}" "${SEGMENTOR}"
fi
