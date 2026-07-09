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

ROOT=/lab/haoq_lab/cse12312032
PROJECT_ROOT="${ROOT}/projects/pku-3dgs-vr"
FLASHSPLAT_ROOT="${ROOT}/external/FlashSplat"
MODEL_DIR="${ROOT}/data/3dgs_models/graphdeco/bicycle"
OUT_DIR="${ROOT}/outputs/eyenavgs_task1/bicycle_semantic"
GROUND_OUT_DIR="${OUT_DIR}/grounded_sam"
PROPOSAL_DIR="${OUT_DIR}/flashsplat_proposals"
CLASS_CONFIG="${CLASS_CONFIG:-${PROJECT_ROOT}/configs/task1_semantic_classes.example.json}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-320}"
SEMANTIC_CAMERA_INDICES="${SEMANTIC_CAMERA_INDICES:-70,115,68,64,87,120,42,153,7,111}"
SEMANTIC_VIEW_COUNT="${SEMANTIC_VIEW_COUNT:-10}"
MAX_DETECTIONS_PER_VIEW="${MAX_DETECTIONS_PER_VIEW:-24}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
CONTACT_THUMB_WIDTH="${CONTACT_THUMB_WIDTH:-360}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.30}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"

mkdir -p "${OUT_DIR}" "${GROUND_OUT_DIR}" "${PROPOSAL_DIR}" "${ROOT}/outputs/eyenavgs_task1/logs"

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
echo "grounded_sam=${GROUNDED_SAM_ROOT}"
nvidia-smi

python scripts/generate_grounded_sam_masks.py \
  --model-path "${MODEL_DIR}" \
  --output-dir "${GROUND_OUT_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --groundingdino-root "${GROUNDED_SAM_ROOT}" \
  --groundingdino-config "${GROUNDINGDINO_CONFIG}" \
  --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}" \
  --segment-anything-root "${SAM_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --class-config "${CLASS_CONFIG}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
  --count "${SEMANTIC_VIEW_COUNT}" \
  --max-detections-per-view "${MAX_DETECTIONS_PER_VIEW}" \
  --box-threshold "${BOX_THRESHOLD}" \
  --text-threshold "${TEXT_THRESHOLD}" \
  --min-mask-area 100 \
  --max-mask-area-ratio 0.80

python scripts/run_flashsplat_mask_proposals.py \
  --model-path "${MODEL_DIR}" \
  --sam-output-dir "${GROUND_OUT_DIR}" \
  --output-dir "${PROPOSAL_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --mask-batch-size "${MASK_BATCH_SIZE}" \
  --max-masks-per-view "${MAX_DETECTIONS_PER_VIEW}" \
  --min-support-gaussians 250

python scripts/cluster_semantic_flashsplat_proposals.py \
  --model-path "${MODEL_DIR}" \
  --proposal-dir "${PROPOSAL_DIR}" \
  --output-dir "${OUT_DIR}" \
  --class-config "${CLASS_CONFIG}" \
  --iteration 30000 \
  --scene bicycle \
  --min-proposal-gaussians 500 \
  --min-group-gaussians 1500 \
  --min-group-proposals 1 \
  --merge-iou 0.35 \
  --containment-threshold 0.70 \
  --max-groups 96 \
  --overwrite

python scripts/inspect_ply.py "${OUT_DIR}/semantic_point_cloud.ply" --json > "${OUT_DIR}/semantic_point_cloud_inspection.json"

python scripts/render_auto_label_overlays.py \
  --model-path "${MODEL_DIR}" \
  --labels-npy "${OUT_DIR}/gaussian_labels.npy" \
  --label-map "${OUT_DIR}/label_map.json" \
  --output-dir "${OUT_DIR}/semantic_label_overlay_renders" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration 30000 \
  --max-width "${RENDER_MAX_WIDTH}" \
  --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
  --count "${SEMANTIC_VIEW_COUNT}"

python scripts/make_contact_sheet.py \
  --image-dir "${GROUND_OUT_DIR}/grounded_sam_overlays" \
  --output "${OUT_DIR}/grounded_sam_contact_sheet.png" \
  --columns 5 \
  --thumb-width "${CONTACT_THUMB_WIDTH}"

python scripts/make_contact_sheet.py \
  --image-dir "${OUT_DIR}/semantic_label_overlay_renders" \
  --output "${OUT_DIR}/semantic_label_overlay_contact_sheet.png" \
  --columns 5 \
  --thumb-width "${CONTACT_THUMB_WIDTH}"

python scripts/validate_task1_outputs.py \
  --labels-npy "${OUT_DIR}/gaussian_labels.npy" \
  --label-map "${OUT_DIR}/label_map.json" \
  --semantic-ply "${OUT_DIR}/semantic_point_cloud.ply" \
  > "${OUT_DIR}/task1_validation.json"

find "${OUT_DIR}" -maxdepth 3 -type f | sort
