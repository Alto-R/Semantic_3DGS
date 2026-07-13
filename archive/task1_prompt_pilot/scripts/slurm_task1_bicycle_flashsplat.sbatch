#!/usr/bin/env bash
#SBATCH --job-name=task1_bicycle_flashsplat
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --nodelist=l40gpu002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
FLASHSPLAT_ROOT="${ROOT}/external/FlashSplat"
MODEL_DIR="${ROOT}/data/3dgs_models/graphdeco/bicycle"
OUT_DIR="${ROOT}/outputs/eyenavgs_task1/bicycle"
PROMPT_DIR="${OUT_DIR}/prompt_masks_bicycle"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-320}"
RENDER_COUNT="${RENDER_COUNT:-20}"
RENDER_CAMERA_INDICES="${RENDER_CAMERA_INDICES:-41,43,148,70,115,68,64,87,69,39,85,120,42,153,123,88,37,40,7,111}"
SEED_CAMERA_INDEX="${SEED_CAMERA_INDEX:-70}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
SAM_ROOT="${SAM_ROOT:-${ROOT}/external/SegAnyGAussians/third_party/segment-anything}"
SLACKNESS="${SLACKNESS:-0.0}"

mkdir -p "${OUT_DIR}" "${PROMPT_DIR}" "${ROOT}/outputs/eyenavgs_task1/logs"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

conda activate gaussian_grouping_true

cd "${PROJECT_ROOT}"

echo "job=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "project=${PROJECT_ROOT}"
echo "model=${MODEL_DIR}"
nvidia-smi

python archive/task1_prompt_pilot/scripts/generate_flashsplat_prompt_masks.py \
  --model-path "${MODEL_DIR}" \
  --output-dir "${PROMPT_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --segment-anything-root "${SAM_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --camera-indices "${RENDER_CAMERA_INDICES}" \
  --count "${RENDER_COUNT}" \
  --seed-camera-index "${SEED_CAMERA_INDEX}" \
  --point 93,110 \
  --point 218,111 \
  --point 160,80 \
  --point 130,60 \
  --point 205,48 \
  --negative-point 45,80 \
  --negative-point 285,80 \
  --negative-point 286,132 \
  --negative-point 35,132 \
  --negative-point 160,30

python archive/task1_prompt_pilot/scripts/run_flashsplat_cameras.py \
  --model-path "${MODEL_DIR}" \
  --mask-dir "${PROMPT_DIR}/sam_masks" \
  --output-dir "${OUT_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --camera-indices "${RENDER_CAMERA_INDICES}" \
  --count "${RENDER_COUNT}" \
  --slackness "${SLACKNESS}" \
  --scene bicycle \
  --label-id 1 \
  --label-name bicycle \
  --label-class object \
  --overwrite

python scripts/task1/inspect_ply.py "${OUT_DIR}/semantic_point_cloud.ply" --json > "${OUT_DIR}/semantic_point_cloud_inspection.json"

python scripts/task1/make_contact_sheet.py \
  --image-dir "${OUT_DIR}/overlay_renders" \
  --output "${OUT_DIR}/overlay_contact_sheet.png" \
  --columns 5 \
  --thumb-width 220

find "${OUT_DIR}" -maxdepth 3 -type f | sort
