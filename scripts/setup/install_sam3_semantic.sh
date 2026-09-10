#!/usr/bin/env bash
# Keep the modern SAM3 environment separate from the existing CUDA 11.7 renderer.
set -euo pipefail
SAM3_ENV="${SAM3_ENV:-sam3_semantic}"
if ! conda run -n "${SAM3_ENV}" python --version >/dev/null 2>&1; then
  conda create -y -n "${SAM3_ENV}" python=3.11 pip
fi
conda run -n "${SAM3_ENV}" python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n "${SAM3_ENV}" python -m pip install \
  transformers==5.3.0 Pillow==12.2.0 modelscope==1.40.0 scipy numpy matplotlib pytest
conda run -n "${SAM3_ENV}" python -c \
  'from transformers import Sam3Model, Sam3Processor; import torch; print("SAM3 imports OK; CUDA:", torch.cuda.is_available())'
