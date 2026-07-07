#!/usr/bin/env bash
#SBATCH --job-name=eyenavgs-gpu-check
#SBATCH --partition=gpulab01
#SBATCH --qos=gpulab01
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
