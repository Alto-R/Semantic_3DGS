#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_boundary_id
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Report-only identity-separated audit:
#   render -> DINOv3 semantic + class-agnostic queries
#   -> DINOv2 identity assignment -> FlashSplat proposal lift
#   -> immutable-v5 overlap report -> adaptive multiview 3D audit.
#
# DINOv3 semantic predictions never assign identity. This scheduler writes no
# Gaussian semantic labels, label map, merged output, or semantic PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${EXTERNAL_ROOT}/FlashSplat}"
DINOV3_ROOT="${DINOV3_ROOT:-${EXTERNAL_ROOT}/dinov3}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
INCLUDE_CLASSES="${INCLUDE_CLASSES:?Set INCLUDE_CLASSES to comma-separated ADE20K classes}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_boundary_dinov2_identity_audit_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_dinov3_query_regions"
IDENTITY_DIR="${OUT_DIR}/stages/02_dinov2_identity_masks"
PROPOSAL_DIR="${OUT_DIR}/stages/03_flashsplat_proposals"
AUDIT_3D_DIR="${OUT_DIR}/stages/04_multiview_3d_audit"
VALIDATION_DIR="${OUT_DIR}/validation"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
RUN_LOG_DIR="${OUT_DIR}/logs"

DINOV2_SOURCE_OUTPUT_NAME="${DINOV2_SOURCE_OUTPUT_NAME:-${SCENE}_dinov2_separate_abstain_allviews_v1}"
DINOV2_SOURCE_DIR="${TASK1_ROOT}/${DINOV2_SOURCE_OUTPUT_NAME}"
DINOV2_MANIFEST="${DINOV2_MANIFEST:-${DINOV2_SOURCE_DIR}/stages/01_real_camera_views/dinov2_manifest.json}"
BASE_OUTPUT_NAME="${BASE_OUTPUT_NAME:-${SCENE}_dinov2_ade_refinement_adaptive_instance_guard_v5}"
BASE_DIR="${TASK1_ROOT}/${BASE_OUTPUT_NAME}"
BASE_LABELS="${BASE_LABELS:-${BASE_DIR}/deliverables/gaussian_labels.npy}"
BASE_LABEL_MAP="${BASE_LABEL_MAP:-${BASE_DIR}/deliverables/label_map.json}"
CLASS_CONFIG="${CLASS_CONFIG:-${PROJECT_ROOT}/configs/task1_semantic_classes.${SCENE}.json}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
DINOV3_BACKBONE="${DINOV3_BACKBONE:-${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth}"
DINOV3_SEGMENTOR="${DINOV3_SEGMENTOR:-${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth}"
DINOV3_HUB_ENTRY="${DINOV3_HUB_ENTRY:-dinov3_vit7b16_ms}"
DINOV3_EXPECTED_REPO_COMMIT="${DINOV3_EXPECTED_REPO_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}"
DINOV3_EXPECTED_BACKBONE_SHA256="${DINOV3_EXPECTED_BACKBONE_SHA256:-a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7}"
DINOV3_EXPECTED_SEGMENTOR_SHA256="${DINOV3_EXPECTED_SEGMENTOR_SHA256:-bf307cb1c2fd95046feb1bf9a8a13dae60a746bddd8f5297134da95525dbcb42}"
DINOV3_PRECISION="${DINOV3_PRECISION:-float32}"
DINOV3_CROP_SIZE="${DINOV3_CROP_SIZE:-512}"
DINOV3_STRIDE="${DINOV3_STRIDE:-384}"

CAMERA_INDICES="${CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-24}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
REGION_MIN_OBJECTNESS="${REGION_MIN_OBJECTNESS:-0.30}"
REGION_MASK_THRESHOLD="${REGION_MASK_THRESHOLD:-0.50}"
REGION_MIN_PIXEL_SCORE="${REGION_MIN_PIXEL_SCORE:-0.25}"
REGION_MIN_AREA="${REGION_MIN_AREA:-100}"
REGION_MAX_AREA_RATIO="${REGION_MAX_AREA_RATIO:-0.80}"
REGION_MAX_QUERIES="${REGION_MAX_QUERIES:-64}"

IDENTITY_MIN_DINOV2_CONFIDENCE="${IDENTITY_MIN_DINOV2_CONFIDENCE:-0.50}"
IDENTITY_MIN_EVIDENCE_PIXELS="${IDENTITY_MIN_EVIDENCE_PIXELS:-100}"
IDENTITY_MIN_EVIDENCE_COVERAGE="${IDENTITY_MIN_EVIDENCE_COVERAGE:-0.25}"
IDENTITY_MIN_CLASS_SHARE="${IDENTITY_MIN_CLASS_SHARE:-0.60}"
IDENTITY_MIN_CLASS_MARGIN="${IDENTITY_MIN_CLASS_MARGIN:-0.20}"

FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
REPORT_ONLY="${REPORT_ONLY:-1}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${REPORT_ONLY}" != "1" ]]; then
  echo "This audit is report-only; REPORT_ONLY must remain 1" >&2
  exit 2
fi
if [[ "${RESET_OUTPUT}" != "0" && "${RESET_OUTPUT}" != "1" ]]; then
  echo "RESET_OUTPUT must be 0 or 1" >&2
  exit 2
fi
if [[ "${DINOV3_PRECISION}" != "bfloat16" && "${DINOV3_PRECISION}" != "float32" ]]; then
  echo "DINOV3_PRECISION must be bfloat16 or float32" >&2
  exit 2
fi
if ! [[ "${DINOV3_EXPECTED_REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "DINOV3_EXPECTED_REPO_COMMIT must be a full 40-character commit" >&2
  exit 2
fi
for expected_sha in \
  "${DINOV3_EXPECTED_BACKBONE_SHA256}" \
  "${DINOV3_EXPECTED_SEGMENTOR_SHA256}"; do
  if [[ -n "${expected_sha}" && ! "${expected_sha}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "Expected checkpoint SHA256 values must contain 64 hexadecimal characters" >&2
    exit 2
  fi
done

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -d "${FLASHSPLAT_ROOT}"
test -d "${DINOV3_ROOT}"
test -f "${DINOV3_BACKBONE}"
test -f "${DINOV3_SEGMENTOR}"
test -f "${DINOV2_MANIFEST}"
test -f "${BASE_LABELS}"
test -f "${BASE_LABEL_MAP}"
test -f "${CLASS_CONFIG}"
test -f "${ONTOLOGY}"
test "$(git -C "${DINOV3_ROOT}" rev-parse HEAD)" = "${DINOV3_EXPECTED_REPO_COMMIT}"
test -z "$(git -C "${DINOV3_ROOT}" status --porcelain --untracked-files=no)"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\ninclude_classes=%s\nview_count=%s\ncamera_indices=%s\ndinov2_manifest=%s\nbase=%s\nreport_only=1\nsemantic_labels_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${INCLUDE_CLASSES}" \
    "${VIEW_COUNT}" "${CAMERA_INDICES:-<evenly-spaced>}" \
    "${DINOV2_MANIFEST}" "${BASE_DIR}"
  exit 0
fi

if [[ -d "${OUT_DIR}" ]]; then
  if [[ "${RESET_OUTPUT}" = "1" ]]; then
    case "${OUT_DIR}" in
      "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
      *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
    esac
  else
    echo "Output exists; choose a fresh OUTPUT_NAME or explicitly set RESET_OUTPUT=1:" >&2
    echo "${OUT_DIR}" >&2
    exit 2
  fi
fi

mkdir -p \
  "${VIEW_DIR}" "${PROPOSAL_DIR}" "${AUDIT_3D_DIR}" \
  "${VALIDATION_DIR}" "${CONTACT_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" \
    2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "output=${OUT_DIR}"
  echo "include_classes=${INCLUDE_CLASSES}"
  echo "dinov2_identity_source=${DINOV2_MANIFEST}"
  echo "immutable_base=${BASE_DIR}"
  echo "view_count=${VIEW_COUNT}"
  echo "camera_indices=${CAMERA_INDICES:-<evenly-spaced>}"
  echo "dinov3_semantic_identity_used=0"
  echo "report_only=1"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_render_real_camera_views "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov2.render_task1_views \
    --model-path "${MODEL_DIR}" \
    --output-dir "${VIEW_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CAMERA_INDICES}" \
    --count "${VIEW_COUNT}" \
    --overwrite

run_stage 02_dinov3_semantic_and_query_regions "${DINOV3_ENV}" \
  python -m scripts.task1.dinov3.dinov3_segment_views \
    --input-dir "${VIEW_DIR}" \
    --dinov3-root "${DINOV3_ROOT}" \
    --backbone-checkpoint "${DINOV3_BACKBONE}" \
    --segmentor-checkpoint "${DINOV3_SEGMENTOR}" \
    --hub-entry "${DINOV3_HUB_ENTRY}" \
    --expected-repo-commit "${DINOV3_EXPECTED_REPO_COMMIT}" \
    --expected-backbone-sha256 "${DINOV3_EXPECTED_BACKBONE_SHA256}" \
    --expected-segmentor-sha256 "${DINOV3_EXPECTED_SEGMENTOR_SHA256}" \
    --ontology "${ONTOLOGY}" \
    --min-pixel-confidence 0.0 \
    --device cuda \
    --precision "${DINOV3_PRECISION}" \
    --crop-size "${DINOV3_CROP_SIZE}" \
    --stride "${DINOV3_STRIDE}" \
    --save-regions \
    --region-min-objectness "${REGION_MIN_OBJECTNESS}" \
    --region-mask-threshold "${REGION_MASK_THRESHOLD}" \
    --region-min-pixel-score "${REGION_MIN_PIXEL_SCORE}" \
    --region-min-area "${REGION_MIN_AREA}" \
    --region-max-area-ratio "${REGION_MAX_AREA_RATIO}" \
    --region-max-queries "${REGION_MAX_QUERIES}" \
    --overwrite

run_stage 03_assign_dinov2_identity "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.boundary_identity_audit \
    --dinov3-manifest "${VIEW_DIR}/dinov3_manifest.json" \
    --dinov2-manifest "${DINOV2_MANIFEST}" \
    --ontology "${ONTOLOGY}" \
    --include-classes "${INCLUDE_CLASSES}" \
    --output-dir "${IDENTITY_DIR}" \
    --min-dinov2-confidence "${IDENTITY_MIN_DINOV2_CONFIDENCE}" \
    --min-evidence-pixels "${IDENTITY_MIN_EVIDENCE_PIXELS}" \
    --min-evidence-coverage "${IDENTITY_MIN_EVIDENCE_COVERAGE}" \
    --min-class-share "${IDENTITY_MIN_CLASS_SHARE}" \
    --min-class-margin "${IDENTITY_MIN_CLASS_MARGIN}"

run_stage 04_lift_identity_regions "${GAUSSIAN_ENV}" \
  python -m scripts.task1.grounding.run_flashsplat_mask_proposals \
    --model-path "${MODEL_DIR}" \
    --sam-output-dir "${IDENTITY_DIR}" \
    --output-dir "${PROPOSAL_DIR}" \
    --manifest-name grounded_sam_manifest.json \
    --mask-dir-name mask_stacks \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --mask-batch-size "${MASK_BATCH_SIZE}" \
    --max-masks-per-view 0 \
    --support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --write-class-evidence \
    --class-evidence-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --min-support-gaussians 250

run_stage 05_immutable_base_overlap "${GAUSSIAN_ENV}" \
  python -m scripts.task1.hybrid_refinement.audit_proposals_against_base \
    --proposal-dir "${PROPOSAL_DIR}" \
    --base-labels "${BASE_LABELS}" \
    --base-label-map "${BASE_LABEL_MAP}" \
    --output "${VALIDATION_DIR}/proposal_base_overlap.json" \
    --conflict-containment-threshold 0.10

run_stage 06_adaptive_multiview_3d_audit "${GAUSSIAN_ENV}" \
  python -m scripts.task1.grounding.cluster_semantic_flashsplat_proposals \
    --model-path "${MODEL_DIR}" \
    --proposal-dir "${PROPOSAL_DIR}" \
    --output-dir "${AUDIT_3D_DIR}" \
    --summary-path "${AUDIT_3D_DIR}/semantic_group_summary.json" \
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
    --min-assigned-gaussians 0 \
    --min-assigned-thing-gaussians "${MIN_ASSIGNED_THING_GAUSSIANS}" \
    --min-assigned-stuff-gaussians "${MIN_ASSIGNED_STUFF_GAUSSIANS}" \
    --adaptive-stuff-min-view-ratio 0.50 \
    --adaptive-stuff-threshold-ratio 0.75 \
    --adaptive-thing-threshold-floor-ratio 0.50 \
    --min-label-score 0.0 \
    --report-only \
    --no-semantic-ply

run_stage 07_query_region_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${VIEW_DIR}/dinov3_region_overlays" \
    --output "${CONTACT_DIR}/dinov3_query_regions.png" \
    --columns 4 \
    --thumb-width 360

run_stage 08_identity_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${IDENTITY_DIR}/identity_overlays" \
    --output "${CONTACT_DIR}/dinov2_identity_on_dinov3_regions.png" \
    --columns 4 \
    --thumb-width 360

{
  echo "mode=report_only_dinov3_boundary_dinov2_identity"
  echo "boundary_source=dinov3_mask2former_queries"
  echo "identity_source=dinov2_vitl14_ade20k_linear"
  echo "immutable_base_overlap_audited=1"
  echo "adaptive_multiview_3d_audit_run=1"
  echo "semantic_labels_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

test ! -e "${OUT_DIR}/deliverables/gaussian_labels.npy"
test ! -e "${OUT_DIR}/deliverables/label_map.json"
test ! -e "${OUT_DIR}/deliverables/semantic_point_cloud.ply"
find "${OUT_DIR}" -maxdepth 4 -type f | sort
