#!/usr/bin/env bash
#SBATCH --job-name=task1_bicycle_semantic
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
export SCENE="${SCENE:-bicycle}"
export CLASS_CONFIG="${CLASS_CONFIG:-${PROJECT_ROOT}/configs/task1_semantic_classes.example.json}"
export FOCUS_CLASSES="${FOCUS_CLASSES:-bicycle,bench}"
export FOCUS_NAME="${FOCUS_NAME:-bicycle_vs_bench}"

exec bash "${PROJECT_ROOT}/scripts/slurm/slurm_task1_semantic_scene.sbatch" "$@"
