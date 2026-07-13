#!/usr/bin/env bash
#SBATCH --job-name=task1_bicycle_auto
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
FLASHSPLAT_ROOT="${ROOT}/external/FlashSplat"
MODEL_DIR="${ROOT}/data/3dgs_models/graphdeco/bicycle"
OUT_DIR="${ROOT}/outputs/eyenavgs_task1/bicycle_auto"
SAM_OUT_DIR="${OUT_DIR}/sam_auto"
PROPOSAL_DIR="${OUT_DIR}/flashsplat_proposals"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-320}"
AUTO_CAMERA_INDICES="${AUTO_CAMERA_INDICES:-70,115,68,64,87,120,42,153,7,111}"
AUTO_VIEW_COUNT="${AUTO_VIEW_COUNT:-10}"
MAX_MASKS_PER_VIEW="${MAX_MASKS_PER_VIEW:-16}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
SAM_ROOT="${SAM_ROOT:-${ROOT}/external/SegAnyGAussians/third_party/segment-anything}"

mkdir -p "${OUT_DIR}" "${SAM_OUT_DIR}" "${PROPOSAL_DIR}" "${ROOT}/outputs/eyenavgs_task1/logs"

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

python archive/task1_legacy/scripts/generate_sam_auto_masks.py \
  --model-path "${MODEL_DIR}" \
  --output-dir "${SAM_OUT_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --segment-anything-root "${SAM_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --camera-indices "${AUTO_CAMERA_INDICES}" \
  --count "${AUTO_VIEW_COUNT}" \
  --max-masks-per-view "${MAX_MASKS_PER_VIEW}" \
  --points-per-side 24 \
  --min-mask-area 100 \
  --max-mask-area-ratio 0.70

python scripts/task1/run_flashsplat_mask_proposals.py \
  --model-path "${MODEL_DIR}" \
  --sam-output-dir "${SAM_OUT_DIR}" \
  --output-dir "${PROPOSAL_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --mask-batch-size "${MASK_BATCH_SIZE}" \
  --max-masks-per-view "${MAX_MASKS_PER_VIEW}" \
  --min-support-gaussians 250

python archive/task1_legacy/scripts/cluster_flashsplat_proposals.py \
  --model-path "${MODEL_DIR}" \
  --proposal-dir "${PROPOSAL_DIR}" \
  --output-dir "${OUT_DIR}" \
  --iteration 30000 \
  --scene bicycle \
  --min-proposal-gaussians 500 \
  --min-group-gaussians 1500 \
  --min-group-proposals 2 \
  --merge-iou 0.35 \
  --containment-threshold 0.70 \
  --max-groups 64 \
  --overwrite

python scripts/task1/inspect_ply.py "${OUT_DIR}/semantic_point_cloud_auto.ply" --json > "${OUT_DIR}/semantic_point_cloud_auto_inspection.json"

python scripts/task1/render_auto_label_overlays.py \
  --model-path "${MODEL_DIR}" \
  --labels-npy "${OUT_DIR}/gaussian_labels_auto.npy" \
  --label-map "${OUT_DIR}/label_map_auto.json" \
  --output-dir "${OUT_DIR}/label_overlay_renders" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --camera-indices "${AUTO_CAMERA_INDICES}" \
  --count "${AUTO_VIEW_COUNT}"

python scripts/task1/make_contact_sheet.py \
  --image-dir "${SAM_OUT_DIR}/sam_auto_overlays" \
  --output "${OUT_DIR}/sam_auto_contact_sheet.png" \
  --columns 5 \
  --thumb-width 220

python scripts/task1/make_contact_sheet.py \
  --image-dir "${OUT_DIR}/label_overlay_renders" \
  --output "${OUT_DIR}/label_overlay_contact_sheet.png" \
  --columns 5 \
  --thumb-width 220

find "${OUT_DIR}" -maxdepth 3 -type f | sort
