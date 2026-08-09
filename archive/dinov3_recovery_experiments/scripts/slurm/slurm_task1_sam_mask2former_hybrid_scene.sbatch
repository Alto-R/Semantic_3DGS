#!/usr/bin/env bash
#SBATCH --job-name=task1_sam_m2f_hybrid
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

# One-pass SAM-seeded, Mask2Former-region hybrid:
#   1. render views and export semantic + class-agnostic Mask2Former regions;
#   2. run GroundingDINO + SAM on the same RGB renders;
#   3. snap eligible SAM masks to class-agnostic regions with identity guards;
#   4. lift refined masks into initial FlashSplat proposal supports;
#   5. render multiview-confirmed 3D seeds and activate compatible regions once;
#   6. lift the final masks and run immutable-base overlap QA;
#   7. run multiview grouping and adaptive component pruning.
#
# REPORT_ONLY=1 is the safe default. It runs every computational stage but
# writes no Gaussian semantic labels, label map, or semantic PLY. Setting
# REPORT_ONLY=0 writes isolated candidate labels and a label map, still without
# merging into the immutable v5 base or writing a semantic PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
FLASHSPLAT_ROOT="${ROOT}/external/FlashSplat"
TASK1_ROOT="${ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_sam_mask2former_hybrid_audit_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
STAGE_ROOT="${OUT_DIR}/stages"
MASK2FORMER_DIR="${STAGE_ROOT}/01_mask2former_regions"
GROUNDED_SAM_DIR="${STAGE_ROOT}/02_grounded_sam"
HYBRID_MASK_DIR="${STAGE_ROOT}/03_hybrid_masks"
SEED_PROPOSAL_DIR="${STAGE_ROOT}/04_seed_flashsplat_proposals"
PROPAGATED_MASK_DIR="${STAGE_ROOT}/05_propagated_hybrid_masks"
PROPOSAL_DIR="${STAGE_ROOT}/06_final_flashsplat_proposals"
FUSION_DIR="${STAGE_ROOT}/08_3d_audit"
VALIDATION_DIR="${OUT_DIR}/validation"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
RUN_LOG_DIR="${OUT_DIR}/logs"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"

SCENE_CLASS_CONFIG="${PROJECT_ROOT}/configs/task1_semantic_classes.${SCENE}.json"
CLASS_CONFIG="${CLASS_CONFIG:-${SCENE_CLASS_CONFIG}}"
INCLUDE_CLASSES="${INCLUDE_CLASSES:?Set INCLUDE_CLASSES to a comma-separated target vocabulary}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.dense_backends.json}"

BASE_OUTPUT_NAME="${BASE_OUTPUT_NAME:-${SCENE}_dinov2_ade_refinement_adaptive_instance_guard_v5}"
BASE_DIR="${TASK1_ROOT}/${BASE_OUTPUT_NAME}"
BASE_LABELS="${BASE_LABELS:-${BASE_DIR}/deliverables/gaussian_labels.npy}"
BASE_LABEL_MAP="${BASE_LABEL_MAP:-${BASE_DIR}/deliverables/label_map.json}"

VIEW_COUNT="${VIEW_COUNT:-24}"
CAMERA_INDICES="${CAMERA_INDICES:-}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
MAX_DETECTIONS_PER_VIEW="${MAX_DETECTIONS_PER_VIEW:-32}"
FINAL_MAX_MASKS_PER_VIEW="${FINAL_MAX_MASKS_PER_VIEW:-0}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.30}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

MASK2FORMER_MODEL="${MASK2FORMER_MODEL:-facebook/mask2former-swin-large-ade-semantic}"
REGION_MIN_OBJECTNESS="${REGION_MIN_OBJECTNESS:-0.30}"
REGION_MASK_THRESHOLD="${REGION_MASK_THRESHOLD:-0.50}"
REGION_MIN_PIXEL_SCORE="${REGION_MIN_PIXEL_SCORE:-0.25}"
REGION_MIN_AREA="${REGION_MIN_AREA:-100}"
REGION_MAX_AREA_RATIO="${REGION_MAX_AREA_RATIO:-0.80}"
REGION_MAX_QUERIES="${REGION_MAX_QUERIES:-64}"

REFINE_MIN_IOU="${REFINE_MIN_IOU:-0.45}"
REFINE_MIN_CONTAINMENT="${REFINE_MIN_CONTAINMENT:-0.80}"
REFINE_MIN_SAM_COVERAGE="${REFINE_MIN_SAM_COVERAGE:-0.10}"
REFINE_MIN_REGION_CONFIDENCE="${REFINE_MIN_REGION_CONFIDENCE:-0.25}"
REFINE_MIN_MATCH_SCORE="${REFINE_MIN_MATCH_SCORE:-0.50}"
REFINE_MIN_AREA_RATIO="${REFINE_MIN_AREA_RATIO:-0.50}"
REFINE_MAX_AREA_RATIO="${REFINE_MAX_AREA_RATIO:-2.00}"
REFINE_IDENTITY_MARGIN="${REFINE_IDENTITY_MARGIN:-0.10}"
REFINE_MAX_CROSS_CLASS_OVERLAP="${REFINE_MAX_CROSS_CLASS_OVERLAP:-0.10}"
REFINE_MIN_OVERLAP_PIXELS="${REFINE_MIN_OVERLAP_PIXELS:-100}"

PROPAGATION_MIN_SEED_VIEWS="${PROPAGATION_MIN_SEED_VIEWS:-2}"
PROPAGATION_MIN_SEED_GAUSSIANS="${PROPAGATION_MIN_SEED_GAUSSIANS:-500}"
PROPAGATION_THRESHOLD="${PROPAGATION_THRESHOLD:-0.05}"
PROPAGATION_MIN_PIXELS="${PROPAGATION_MIN_PIXELS:-100}"
PROPAGATION_MIN_PROJECTION_COVERAGE="${PROPAGATION_MIN_PROJECTION_COVERAGE:-0.05}"
PROPAGATION_MIN_REGION_COVERAGE="${PROPAGATION_MIN_REGION_COVERAGE:-0.20}"
PROPAGATION_MIN_CONTAINMENT="${PROPAGATION_MIN_CONTAINMENT:-0.50}"
PROPAGATION_MIN_REGION_CONFIDENCE="${PROPAGATION_MIN_REGION_CONFIDENCE:-0.25}"
PROPAGATION_MIN_MATCH_SCORE="${PROPAGATION_MIN_MATCH_SCORE:-0.45}"
PROPAGATION_EXISTING_CONTAINMENT="${PROPAGATION_EXISTING_CONTAINMENT:-0.50}"
PROPAGATION_IDENTITY_MARGIN="${PROPAGATION_IDENTITY_MARGIN:-0.10}"
PROPAGATION_MAX_CROSS_CLASS_OVERLAP="${PROPAGATION_MAX_CROSS_CLASS_OVERLAP:-0.10}"
PROPAGATION_MAX_MASKS_PER_CLASS="${PROPAGATION_MAX_MASKS_PER_CLASS:-4}"

MIN_ASSIGNED_GAUSSIANS="${MIN_ASSIGNED_GAUSSIANS:-0}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
ADAPTIVE_STUFF_MIN_VIEW_RATIO="${ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
ADAPTIVE_STUFF_THRESHOLD_RATIO="${ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
MIN_LABEL_SCORE="${MIN_LABEL_SCORE:-0.0}"

SAM_CHECKPOINT="${SAM_CHECKPOINT:-${ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"

REPORT_ONLY="${REPORT_ONLY:-1}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
CONTACT_THUMB_WIDTH="${CONTACT_THUMB_WIDTH:-360}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TOKENIZERS_PARALLELISM

if [ "${REPORT_ONLY}" != "0" ] && [ "${REPORT_ONLY}" != "1" ]; then
  echo "REPORT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [ "${RESET_OUTPUT}" != "0" ] && [ "${RESET_OUTPUT}" != "1" ]; then
  echo "RESET_OUTPUT must be 0 or 1" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${CLASS_CONFIG}"
test -f "${ONTOLOGY}"
test -f "${BASE_LABELS}"
test -f "${BASE_LABEL_MAP}"
test -d "${FLASHSPLAT_ROOT}"
test -d "${GROUNDED_SAM_ROOT}"
test -f "${GROUNDINGDINO_CONFIG}"
test -f "${GROUNDINGDINO_CHECKPOINT}"
test -d "${SAM_ROOT}"
test -f "${SAM_CHECKPOINT}"

if [ "${CONFIG_ONLY}" = "1" ]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nclass_config=%s\ninclude_classes=%s\nbase=%s\nview_count=%s\ncamera_indices=%s\nmask2former_model=%s\nreport_only=%s\none_pass=1\npropagation_rounds=1\n' \
    "${SCENE}" \
    "${MODEL_DIR}" \
    "${OUT_DIR}" \
    "${CLASS_CONFIG}" \
    "${INCLUDE_CLASSES}" \
    "${BASE_DIR}" \
    "${VIEW_COUNT}" \
    "${CAMERA_INDICES:-<evenly-spaced>}" \
    "${MASK2FORMER_MODEL}" \
    "${REPORT_ONLY}"
  exit 0
fi

if [ -d "${OUT_DIR}" ]; then
  if [ "${RESET_OUTPUT}" = "1" ]; then
    case "${OUT_DIR}" in
      "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
      *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
    esac
  else
    echo "Output exists: ${OUT_DIR}; choose a fresh OUTPUT_NAME or set RESET_OUTPUT=1" >&2
    exit 2
  fi
fi

mkdir -p \
  "${MASK2FORMER_DIR}" \
  "${GROUNDED_SAM_DIR}" \
  "${HYBRID_MASK_DIR}" \
  "${SEED_PROPOSAL_DIR}" \
  "${PROPAGATED_MASK_DIR}" \
  "${PROPOSAL_DIR}" \
  "${FUSION_DIR}" \
  "${VALIDATION_DIR}" \
  "${CONTACT_DIR}" \
  "${RUN_LOG_DIR}" \
  "${DELIVERABLE_DIR}" \
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

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "class_config=${CLASS_CONFIG}"
  echo "include_classes=${INCLUDE_CLASSES}"
  echo "base_output=${BASE_DIR}"
  echo "view_count=${VIEW_COUNT}"
  echo "camera_indices=${CAMERA_INDICES:-<evenly-spaced>}"
  echo "mask2former_model=${MASK2FORMER_MODEL}"
  echo "report_only=${REPORT_ONLY}"
  echo "one_pass=1"
  echo "propagation_rounds=1"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_mask2former_regions \
  python -m scripts.task1.dense_seg.segment_views_semantic \
    --model-path "${MODEL_DIR}" \
    --output-dir "${MASK2FORMER_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CAMERA_INDICES}" \
    --count "${VIEW_COUNT}" \
    --backend mask2former \
    --mask2former-model "${MASK2FORMER_MODEL}" \
    --save-regions \
    --region-min-objectness "${REGION_MIN_OBJECTNESS}" \
    --region-mask-threshold "${REGION_MASK_THRESHOLD}" \
    --region-min-pixel-score "${REGION_MIN_PIXEL_SCORE}" \
    --region-min-area "${REGION_MIN_AREA}" \
    --region-max-area-ratio "${REGION_MAX_AREA_RATIO}" \
    --region-max-queries "${REGION_MAX_QUERIES}"

command -v jq >/dev/null 2>&1
RESOLVED_CAMERA_INDICES="$(jq -r '[.frames[].camera_index] | map(tostring) | join(",")' "${MASK2FORMER_DIR}/dense_seg_manifest.json")"
RESOLVED_VIEW_COUNT="$(jq -r '.frames | length' "${MASK2FORMER_DIR}/dense_seg_manifest.json")"
test -n "${RESOLVED_CAMERA_INDICES}"
test "${RESOLVED_VIEW_COUNT}" -gt 0

run_stage 02_grounded_sam \
  python -m scripts.task1.grounding.generate_grounded_sam_masks \
    --model-path "${MODEL_DIR}" \
    --output-dir "${GROUNDED_SAM_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --groundingdino-root "${GROUNDED_SAM_ROOT}" \
    --groundingdino-config "${GROUNDINGDINO_CONFIG}" \
    --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}" \
    --segment-anything-root "${SAM_ROOT}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --class-config "${CLASS_CONFIG}" \
    --include-classes "${INCLUDE_CLASSES}" \
    --source-view-manifest "${MASK2FORMER_DIR}/dense_seg_manifest.json" \
    --camera-indices "${RESOLVED_CAMERA_INDICES}" \
    --count "${RESOLVED_VIEW_COUNT}" \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-detections-per-view "${MAX_DETECTIONS_PER_VIEW}" \
    --box-threshold "${BOX_THRESHOLD}" \
    --text-threshold "${TEXT_THRESHOLD}"

run_stage 03_hybrid_mask_refinement \
  python -m scripts.task1.hybrid_refinement.refine_sam_with_mask2former_regions \
    --grounded-sam-manifest "${GROUNDED_SAM_DIR}/grounded_sam_manifest.json" \
    --dense-seg-manifest "${MASK2FORMER_DIR}/dense_seg_manifest.json" \
    --output-dir "${HYBRID_MASK_DIR}" \
    --min-iou "${REFINE_MIN_IOU}" \
    --min-containment "${REFINE_MIN_CONTAINMENT}" \
    --min-sam-coverage "${REFINE_MIN_SAM_COVERAGE}" \
    --min-region-confidence "${REFINE_MIN_REGION_CONFIDENCE}" \
    --min-match-score "${REFINE_MIN_MATCH_SCORE}" \
    --min-area-ratio "${REFINE_MIN_AREA_RATIO}" \
    --max-area-ratio "${REFINE_MAX_AREA_RATIO}" \
    --identity-margin "${REFINE_IDENTITY_MARGIN}" \
    --max-cross-class-overlap "${REFINE_MAX_CROSS_CLASS_OVERLAP}" \
    --min-overlap-pixels "${REFINE_MIN_OVERLAP_PIXELS}"

run_stage 04_seed_flashsplat_proposals \
  python -m scripts.task1.grounding.run_flashsplat_mask_proposals \
    --model-path "${MODEL_DIR}" \
    --sam-output-dir "${HYBRID_MASK_DIR}" \
    --output-dir "${SEED_PROPOSAL_DIR}" \
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

run_stage 05_multiview_seed_propagation \
  python -m scripts.task1.hybrid_refinement.propagate_regions_from_3d_seeds \
    --model-path "${MODEL_DIR}" \
    --hybrid-mask-manifest "${HYBRID_MASK_DIR}/grounded_sam_manifest.json" \
    --proposal-dir "${SEED_PROPOSAL_DIR}" \
    --dense-seg-manifest "${MASK2FORMER_DIR}/dense_seg_manifest.json" \
    --output-dir "${PROPAGATED_MASK_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --min-seed-views "${PROPAGATION_MIN_SEED_VIEWS}" \
    --min-seed-gaussians "${PROPAGATION_MIN_SEED_GAUSSIANS}" \
    --projection-threshold "${PROPAGATION_THRESHOLD}" \
    --min-projection-pixels "${PROPAGATION_MIN_PIXELS}" \
    --min-projection-coverage "${PROPAGATION_MIN_PROJECTION_COVERAGE}" \
    --min-region-coverage "${PROPAGATION_MIN_REGION_COVERAGE}" \
    --min-containment "${PROPAGATION_MIN_CONTAINMENT}" \
    --min-region-confidence "${PROPAGATION_MIN_REGION_CONFIDENCE}" \
    --min-match-score "${PROPAGATION_MIN_MATCH_SCORE}" \
    --existing-same-class-containment "${PROPAGATION_EXISTING_CONTAINMENT}" \
    --identity-margin "${PROPAGATION_IDENTITY_MARGIN}" \
    --max-cross-class-overlap "${PROPAGATION_MAX_CROSS_CLASS_OVERLAP}" \
    --max-masks-per-class-per-view "${PROPAGATION_MAX_MASKS_PER_CLASS}"

run_stage 06_final_flashsplat_proposals \
  python -m scripts.task1.grounding.run_flashsplat_mask_proposals \
    --model-path "${MODEL_DIR}" \
    --sam-output-dir "${PROPAGATED_MASK_DIR}" \
    --output-dir "${PROPOSAL_DIR}" \
    --manifest-name grounded_sam_manifest.json \
    --mask-dir-name mask_stacks \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --mask-batch-size "${MASK_BATCH_SIZE}" \
    --max-masks-per-view "${FINAL_MAX_MASKS_PER_VIEW}" \
    --support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --write-class-evidence \
    --class-evidence-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --min-support-gaussians 250

BASE_AUDIT_ARGS=()
if [ "${REPORT_ONLY}" = "0" ]; then
  BASE_AUDIT_ARGS+=(--fail-on-conflict)
fi

run_stage 07_base_overlap_audit \
  python -m scripts.task1.hybrid_refinement.audit_proposals_against_base \
    --proposal-dir "${PROPOSAL_DIR}" \
    --base-labels "${BASE_LABELS}" \
    --base-label-map "${BASE_LABEL_MAP}" \
    --output "${VALIDATION_DIR}/proposal_base_overlap.json" \
    --conflict-containment-threshold 0.10 \
    "${BASE_AUDIT_ARGS[@]}"

FUSION_OUTPUT_ARGS=(
  --no-semantic-ply
)
if [ "${REPORT_ONLY}" = "1" ]; then
  FUSION_OUTPUT_ARGS+=(--report-only)
else
  FUSION_OUTPUT_ARGS+=(
    --labels-path "${DELIVERABLE_DIR}/candidate_gaussian_labels.npy"
    --label-map-path "${DELIVERABLE_DIR}/candidate_label_map.json"
  )
fi

run_stage 08_multiview_3d_audit \
  python -m scripts.task1.grounding.cluster_semantic_flashsplat_proposals \
    --model-path "${MODEL_DIR}" \
    --proposal-dir "${PROPOSAL_DIR}" \
    --output-dir "${FUSION_DIR}" \
    --summary-path "${FUSION_DIR}/semantic_group_summary.json" \
    --class-config "${CLASS_CONFIG}" \
    --iteration 30000 \
    --scene "${SCENE}" \
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
    --spatial-prune-all-classes \
    --spatial-voxel-scale-multiplier 4 \
    --spatial-min-voxel-size 0.01 \
    --spatial-max-voxel-size 0.20 \
    --spatial-min-component-gaussians 500 \
    --spatial-min-component-ratio 0.01 \
    --consolidate-thing-instances \
    --instance-voxel-scale-multiplier 4 \
    --instance-min-voxel-size 0.01 \
    --instance-max-voxel-size 0.20 \
    --instance-min-component-gaussians 500 \
    --instance-min-component-ratio 0.01 \
    --min-assigned-gaussians "${MIN_ASSIGNED_GAUSSIANS}" \
    --min-assigned-thing-gaussians "${MIN_ASSIGNED_THING_GAUSSIANS}" \
    --min-assigned-stuff-gaussians "${MIN_ASSIGNED_STUFF_GAUSSIANS}" \
    --adaptive-stuff-min-view-ratio "${ADAPTIVE_STUFF_MIN_VIEW_RATIO}" \
    --adaptive-stuff-threshold-ratio "${ADAPTIVE_STUFF_THRESHOLD_RATIO}" \
    --adaptive-thing-threshold-floor-ratio "${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO}" \
    --min-label-score "${MIN_LABEL_SCORE}" \
    "${FUSION_OUTPUT_ARGS[@]}"

run_stage 09_mask2former_region_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${MASK2FORMER_DIR}/region_overlays" \
    --output "${CONTACT_DIR}/mask2former_regions.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 10_grounded_sam_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${GROUNDED_SAM_DIR}/overlays" \
    --output "${CONTACT_DIR}/grounded_sam.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 11_hybrid_refinement_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${HYBRID_MASK_DIR}/overlays" \
    --output "${CONTACT_DIR}/hybrid_refinement.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 12_propagated_hybrid_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${PROPAGATED_MASK_DIR}/overlays" \
    --output "${CONTACT_DIR}/propagated_hybrid.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

if [ "${REPORT_ONLY}" = "1" ]; then
  {
    echo "mode=one_pass_sam_mask2former_hybrid_report"
    echo "scene=${SCENE}"
    echo "include_classes=${INCLUDE_CLASSES}"
    echo "view_count=${RESOLVED_VIEW_COUNT}"
    echo "propagation_rounds=1"
    echo "base_modified=0"
    echo "semantic_labels_written=0"
    echo "label_map_written=0"
    echo "semantic_ply_written=0"
  } > "${OUT_DIR}/experiment_mode.txt"
else
  {
    echo "mode=one_pass_sam_mask2former_hybrid_candidates"
    echo "scene=${SCENE}"
    echo "include_classes=${INCLUDE_CLASSES}"
    echo "view_count=${RESOLVED_VIEW_COUNT}"
    echo "propagation_rounds=1"
    echo "base_modified=0"
    echo "candidate_semantic_labels_written=1"
    echo "semantic_ply_written=0"
  } > "${OUT_DIR}/experiment_mode.txt"
fi

find "${OUT_DIR}" -maxdepth 5 -type f | sort
