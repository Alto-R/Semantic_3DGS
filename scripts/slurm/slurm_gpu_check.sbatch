#!/usr/bin/env bash
#SBATCH --job-name=eyenavgs-gpu-check
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/slurm_gpu_check_%j.out
#SBATCH --error=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/slurm_gpu_check_%j.err

set -euo pipefail

echo "date: $(date -Is)"
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

ENV_NAME="${EYENAVGS_CONDA_ENV:-gaussian_grouping_true}"

if command -v conda >/dev/null 2>&1; then
  echo
  echo "semantic env import check: $ENV_NAME"
  conda run -n "$ENV_NAME" python -c '
import torch
import flashsplat_rasterization
import diff_gaussian_rasterization
import diff_gaussian_rasterization_contrastive_f
import diff_gaussian_rasterization_depth

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("tensor", torch.ones(1, device="cuda"))
'
else
  echo "conda not found"
fi
