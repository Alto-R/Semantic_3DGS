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
TASK1_ROOT="${ROOT}/outputs/eyenavgs_task1"
OUT_DIR="${TASK1_ROOT}/bicycle_semantic"
STAGE_ROOT="${OUT_DIR}/stages"
GROUND_OUT_DIR="${STAGE_ROOT}/01_grounded_sam"
PROPOSAL_DIR="${STAGE_ROOT}/02_flashsplat"
FUSION_DIR="${STAGE_ROOT}/03_semantic_fusion"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
PLY_VIS_DIR="${VISUALIZATION_DIR}/ply"
SEMANTIC_OVERLAY_DIR="${VISUALIZATION_DIR}/overlays/semantic_labels"
FOCUS_OVERLAY_DIR="${VISUALIZATION_DIR}/overlays/bicycle_vs_bench"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

CLASS_CONFIG="${CLASS_CONFIG:-${PROJECT_ROOT}/configs/task1_semantic_classes.example.json}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
# An empty camera list enables evenly spaced sampling by SEMANTIC_VIEW_COUNT.
SEMANTIC_CAMERA_INDICES="${SEMANTIC_CAMERA_INDICES:-}"
SEMANTIC_VIEW_COUNT="${SEMANTIC_VIEW_COUNT:-50}"
MAX_DETECTIONS_PER_VIEW="${MAX_DETECTIONS_PER_VIEW:-32}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
CONTACT_THUMB_WIDTH="${CONTACT_THUMB_WIDTH:-360}"
MIN_ASSIGNED_GAUSSIANS="${MIN_ASSIGNED_GAUSSIANS:-0}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
MIN_LABEL_SCORE="${MIN_LABEL_SCORE:-0.0}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.30}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TOKENIZERS_PARALLELISM

if [ "${RESET_OUTPUT}" = "1" ] && [ -d "${OUT_DIR}" ]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${GROUND_OUT_DIR}" \
  "${PROPOSAL_DIR}" \
  "${FUSION_DIR}" \
  "${DELIVERABLE_DIR}" \
  "${PLY_VIS_DIR}" \
  "${SEMANTIC_OVERLAY_DIR}" \
  "${FOCUS_OVERLAY_DIR}" \
  "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" \
  "${RUN_LOG_DIR}" \
  "${TASK1_ROOT}/logs"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

conda activate gaussian_grouping_true
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  echo "===== ${stage_name} ====="
  "$@" 2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

run_report() {
  local output_path="$1"
  local log_name="$2"
  shift 2
  "$@" | tee "${output_path}" "${RUN_LOG_DIR}/${log_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "semantic_camera_indices=${SEMANTIC_CAMERA_INDICES:-<evenly-spaced>}"
  echo "semantic_view_count=${SEMANTIC_VIEW_COUNT}"
  echo "max_detections_per_view=${MAX_DETECTIONS_PER_VIEW}"
  echo "mask_batch_size=${MASK_BATCH_SIZE}"
  echo "flashsplat_support_threshold=${FLASHSPLAT_SUPPORT_THRESHOLD}"
  echo "min_assigned_gaussians=${MIN_ASSIGNED_GAUSSIANS}"
  echo "min_assigned_thing_gaussians=${MIN_ASSIGNED_THING_GAUSSIANS}"
  echo "min_assigned_stuff_gaussians=${MIN_ASSIGNED_STUFF_GAUSSIANS}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_grounded_sam \
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

run_stage 02_flashsplat \
  python scripts/run_flashsplat_mask_proposals.py \
    --model-path "${MODEL_DIR}" \
    --sam-output-dir "${GROUND_OUT_DIR}" \
    --output-dir "${PROPOSAL_DIR}" \
    --manifest-name grounded_sam_manifest.json \
    --mask-dir-name mask_stacks \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --mask-batch-size "${MASK_BATCH_SIZE}" \
    --max-masks-per-view "${MAX_DETECTIONS_PER_VIEW}" \
    --support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --write-class-evidence \
    --class-evidence-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --min-support-gaussians 250

run_stage 03_semantic_fusion \
  python scripts/cluster_semantic_flashsplat_proposals.py \
    --model-path "${MODEL_DIR}" \
    --proposal-dir "${PROPOSAL_DIR}" \
    --output-dir "${FUSION_DIR}" \
    --labels-path "${FUSION_DIR}/gaussian_labels.npy" \
    --summary-path "${FUSION_DIR}/semantic_group_summary.json" \
    --label-map-path "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply-path "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --class-config "${CLASS_CONFIG}" \
    --iteration 30000 \
    --scene bicycle \
    --min-proposal-gaussians 500 \
    --min-group-gaussians 1500 \
    --min-group-proposals 2 \
    --merge-iou 0.35 \
    --containment-threshold 0.70 \
    --max-groups 96 \
    --assignment-reliability-views 2 \
    --assignment-min-quality 0.08 \
    --class-evidence-dir "${PROPOSAL_DIR}/class_evidence" \
    --class-evidence-min-positive-views 2 \
    --class-evidence-min-ratio 0.50 \
    --spatial-prune-thing-islands \
    --spatial-voxel-scale-multiplier 4 \
    --spatial-min-voxel-size 0.01 \
    --spatial-max-voxel-size 0.20 \
    --spatial-min-component-gaussians 500 \
    --spatial-min-component-ratio 0.01 \
    --min-assigned-gaussians "${MIN_ASSIGNED_GAUSSIANS}" \
    --min-assigned-thing-gaussians "${MIN_ASSIGNED_THING_GAUSSIANS}" \
    --min-assigned-stuff-gaussians "${MIN_ASSIGNED_STUFF_GAUSSIANS}" \
    --min-label-score "${MIN_LABEL_SCORE}" \
    --overwrite

run_report \
  "${VALIDATION_DIR}/semantic_point_cloud_inspection.json" \
  04_semantic_ply_inspection \
  python scripts/inspect_ply.py "${DELIVERABLE_DIR}/semantic_point_cloud.ply" --json

run_stage 05_rgb_debug_ply \
  python scripts/export_debug_label_colors.py \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${PLY_VIS_DIR}/semantic_point_cloud_rgb_debug.ply" \
    --metadata-json "${PLY_VIS_DIR}/semantic_point_cloud_rgb_debug.json" \
    --overwrite

run_report \
  "${VALIDATION_DIR}/semantic_point_cloud_rgb_debug_inspection.json" \
  05_rgb_debug_ply_inspection \
  python scripts/inspect_ply.py "${PLY_VIS_DIR}/semantic_point_cloud_rgb_debug.ply" --json

run_stage 06_supersplat_debug_ply \
  python scripts/export_supersplat_label_colors.py \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --overwrite

run_stage 07_bicycle_bench_debug_ply \
  python scripts/export_supersplat_label_colors.py \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${PLY_VIS_DIR}/bicycle_vs_bench_supersplat_debug.ply" \
    --metadata-json "${PLY_VIS_DIR}/bicycle_vs_bench_supersplat_debug.json" \
    --focus-classes bicycle,bench \
    --overwrite

run_report \
  "${VALIDATION_DIR}/semantic_point_cloud_supersplat_debug_inspection.json" \
  07_supersplat_debug_ply_inspection \
  python scripts/inspect_ply.py "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.ply" --json

run_stage 08_semantic_overlays \
  python scripts/render_auto_label_overlays.py \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${SEMANTIC_OVERLAY_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
    --count "${SEMANTIC_VIEW_COUNT}"

run_stage 09_bicycle_bench_overlays \
  python scripts/render_auto_label_overlays.py \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${FOCUS_OVERLAY_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
    --count "${SEMANTIC_VIEW_COUNT}" \
    --focus-classes bicycle,bench

run_stage 10_contact_sheets \
  python scripts/make_contact_sheet.py \
    --image-dir "${GROUND_OUT_DIR}/overlays" \
    --output "${CONTACT_DIR}/grounded_sam.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 11_semantic_contact_sheet \
  python scripts/make_contact_sheet.py \
    --image-dir "${SEMANTIC_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 12_bicycle_bench_contact_sheet \
  python scripts/make_contact_sheet.py \
    --image-dir "${FOCUS_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/bicycle_vs_bench.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_report \
  "${VALIDATION_DIR}/task1_validation.json" \
  13_task1_validation \
  python scripts/validate_task1_outputs.py \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply"

run_stage 14_pipeline_summary \
  python scripts/summarize_task1_semantic_run.py \
    --output-dir "${OUT_DIR}" \
    --summary-path "${VALIDATION_DIR}/pipeline_run_summary.json"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
