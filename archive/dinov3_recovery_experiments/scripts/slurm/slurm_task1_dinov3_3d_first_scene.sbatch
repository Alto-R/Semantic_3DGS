#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_3d_first
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

# Independent report-only pipeline:
#   render -> DINOv3 query masks + full soft evidence
#   -> class-agnostic FlashSplat lift -> physical 3D association
#   -> component-level multiview DINOv3 identity.
#
# This scheduler never reads prior semantic labels and never writes Gaussian
# labels, a label map, a merged output, or a semantic PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${EXTERNAL_ROOT}/FlashSplat}"
DINOV3_ROOT="${DINOV3_ROOT:-${EXTERNAL_ROOT}/dinov3}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_dinov3_query_evidence"
REGION_DIR="${OUT_DIR}/stages/02_class_agnostic_regions"
LIFT_DIR="${OUT_DIR}/stages/03_flashsplat_region_supports"
ASSOCIATION_DIR="${OUT_DIR}/stages/04_3d_components"
OVERLAY_DIR="${OUT_DIR}/visualizations/component_overlays"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
RUN_LOG_DIR="${OUT_DIR}/logs"

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
DINOV3_CHECKPOINT_LOAD_MODE="${DINOV3_CHECKPOINT_LOAD_MODE:-standard}"
DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB:-}"

CAMERA_INDICES="${CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-24}"
CAMERA_SELECTION_POLICY="${CAMERA_SELECTION_POLICY:-evenly_spaced_or_explicit_indices}"
CAMERA_SELECTION_USED_PRIOR_LABELS="${CAMERA_SELECTION_USED_PRIOR_LABELS:-0}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
REGION_MIN_OBJECTNESS="${REGION_MIN_OBJECTNESS:-0.30}"
REGION_MASK_THRESHOLD="${REGION_MASK_THRESHOLD:-0.50}"
REGION_MIN_PIXEL_SCORE="${REGION_MIN_PIXEL_SCORE:-0.25}"
REGION_MIN_AREA="${REGION_MIN_AREA:-100}"
REGION_MAX_AREA_RATIO="${REGION_MAX_AREA_RATIO:-0.80}"
REGION_MAX_QUERIES="${REGION_MAX_QUERIES:-64}"

FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
MIN_SUPPORT_GAUSSIANS="${MIN_SUPPORT_GAUSSIANS:-250}"
ASSOCIATION_MIN_SHARED_GAUSSIANS="${ASSOCIATION_MIN_SHARED_GAUSSIANS:-250}"
ASSOCIATION_MIN_IOU="${ASSOCIATION_MIN_IOU:-0.05}"
ASSOCIATION_MIN_CONTAINMENT="${ASSOCIATION_MIN_CONTAINMENT:-0.25}"
ASSOCIATION_MIN_FEATURE_SIMILARITY="${ASSOCIATION_MIN_FEATURE_SIMILARITY:-0.0}"
COMPONENT_MIN_SOURCE_VIEWS="${COMPONENT_MIN_SOURCE_VIEWS:-2}"
COMPONENT_MIN_GAUSSIANS="${COMPONENT_MIN_GAUSSIANS:-500}"

REPORT_ONLY="${REPORT_ONLY:-1}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${REPORT_ONLY}" != "1" ]]; then
  echo "This pipeline is report-only; REPORT_ONLY must remain 1" >&2
  exit 2
fi
if [[ "${CAMERA_SELECTION_USED_PRIOR_LABELS}" != "0" && "${CAMERA_SELECTION_USED_PRIOR_LABELS}" != "1" ]]; then
  echo "CAMERA_SELECTION_USED_PRIOR_LABELS must be 0 or 1" >&2
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
if ! [[ "${DINOV3_CROP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DINOV3_CROP_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "${DINOV3_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DINOV3_STRIDE must be a positive integer" >&2
  exit 2
fi
if (( DINOV3_CROP_SIZE % 32 != 0 )); then
  echo "DINOV3_CROP_SIZE must be divisible by 32 for the DINOv3 spatial adapter" >&2
  exit 2
fi
if (( DINOV3_STRIDE > DINOV3_CROP_SIZE )); then
  echo "DINOV3_STRIDE must not exceed DINOV3_CROP_SIZE" >&2
  exit 2
fi
if [[ "${DINOV3_CHECKPOINT_LOAD_MODE}" != "standard" \
   && "${DINOV3_CHECKPOINT_LOAD_MODE}" != "local_mmap" ]]; then
  echo "DINOV3_CHECKPOINT_LOAD_MODE must be standard or local_mmap" >&2
  exit 2
fi
if [[ -n "${DINOV3_MAX_CUDA_MEMORY_GIB}" \
   && ! "${DINOV3_MAX_CUDA_MEMORY_GIB}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]]; then
  echo "DINOV3_MAX_CUDA_MEMORY_GIB must be empty or a positive GiB value" >&2
  exit 2
fi
if ! [[ "${DINOV3_EXPECTED_REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "DINOV3_EXPECTED_REPO_COMMIT must be a full commit" >&2
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
test -f "${ONTOLOGY}"
test "$(git -C "${DINOV3_ROOT}" rev-parse HEAD)" = "${DINOV3_EXPECTED_REPO_COMMIT}"
test -z "$(git -C "${DINOV3_ROOT}" status --porcelain --untracked-files=no)"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nview_count=%s\ncamera_indices=%s\ncamera_selection_policy=%s\ncamera_selection_used_prior_labels=%s\ndinov3_checkpoint_load_mode=%s\ndinov3_max_cuda_memory_gib=%s\nidentity_source=dinov3_mask2former_multiview\nprior_semantic_labels_used=0\nreport_only=1\nsemantic_labels_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${VIEW_COUNT}" \
    "${CAMERA_INDICES:-<evenly-spaced>}" "${CAMERA_SELECTION_POLICY}" \
    "${CAMERA_SELECTION_USED_PRIOR_LABELS}" \
    "${DINOV3_CHECKPOINT_LOAD_MODE}" \
    "${DINOV3_MAX_CUDA_MEMORY_GIB:-<unset>}"
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

mkdir -p "${VIEW_DIR}" "${ASSOCIATION_DIR}" "${CONTACT_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
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
  echo "camera_selection_policy=${CAMERA_SELECTION_POLICY}"
  echo "camera_selection_used_prior_labels=${CAMERA_SELECTION_USED_PRIOR_LABELS}"
  echo "identity_source=dinov3_mask2former_multiview"
  echo "prior_semantic_labels_used=0"
  echo "semantic_identity_hardened_before_3d=0"
  echo "report_only=1"
  echo "dinov3_checkpoint_load_mode=${DINOV3_CHECKPOINT_LOAD_MODE}"
  echo "dinov3_max_cuda_memory_gib=${DINOV3_MAX_CUDA_MEMORY_GIB:-<unset>}"
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

DINOV3_MEMORY_LIMIT_ARGS=()
if [[ -n "${DINOV3_MAX_CUDA_MEMORY_GIB}" ]]; then
  DINOV3_MEMORY_LIMIT_ARGS=(
    --max-cuda-memory-gib "${DINOV3_MAX_CUDA_MEMORY_GIB}"
  )
fi

run_stage 02_export_dinov3_query_evidence "${DINOV3_ENV}" \
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
    --checkpoint-load-mode "${DINOV3_CHECKPOINT_LOAD_MODE}" \
    "${DINOV3_MEMORY_LIMIT_ARGS[@]}" \
    --save-regions \
    --region-min-objectness "${REGION_MIN_OBJECTNESS}" \
    --region-mask-threshold "${REGION_MASK_THRESHOLD}" \
    --region-min-pixel-score "${REGION_MIN_PIXEL_SCORE}" \
    --region-min-area "${REGION_MIN_AREA}" \
    --region-max-area-ratio "${REGION_MAX_AREA_RATIO}" \
    --region-max-queries "${REGION_MAX_QUERIES}" \
    --overwrite

run_stage 03_prepare_class_agnostic_regions "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.prepare_3d_region_proposals \
    --dinov3-manifest "${VIEW_DIR}/dinov3_manifest.json" \
    --output-dir "${REGION_DIR}"

run_stage 04_lift_all_query_regions "${GAUSSIAN_ENV}" \
  python -m scripts.task1.grounding.run_flashsplat_mask_proposals \
    --model-path "${MODEL_DIR}" \
    --sam-output-dir "${REGION_DIR}" \
    --output-dir "${LIFT_DIR}" \
    --manifest-name query_region_manifest.json \
    --mask-dir-name mask_stacks \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --mask-batch-size "${MASK_BATCH_SIZE}" \
    --max-masks-per-view 0 \
    --support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --min-support-gaussians "${MIN_SUPPORT_GAUSSIANS}"

run_stage 05_associate_and_fuse_3d_components "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.associate_3d_query_regions \
    --proposal-manifest "${LIFT_DIR}/proposal_manifest.json" \
    --ontology "${ONTOLOGY}" \
    --output "${ASSOCIATION_DIR}/component_report.json" \
    --min-shared-gaussians "${ASSOCIATION_MIN_SHARED_GAUSSIANS}" \
    --min-iou "${ASSOCIATION_MIN_IOU}" \
    --min-containment "${ASSOCIATION_MIN_CONTAINMENT}" \
    --min-feature-similarity "${ASSOCIATION_MIN_FEATURE_SIMILARITY}" \
    --min-source-views "${COMPONENT_MIN_SOURCE_VIEWS}" \
    --min-component-gaussians "${COMPONENT_MIN_GAUSSIANS}" \
    --report-only

run_stage 06_render_component_overlays "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.render_3d_component_overlays \
    --association-report "${ASSOCIATION_DIR}/component_report.json" \
    --proposal-manifest "${LIFT_DIR}/proposal_manifest.json" \
    --output-dir "${OVERLAY_DIR}"

run_stage 07_query_region_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${VIEW_DIR}/dinov3_region_overlays" \
    --output "${CONTACT_DIR}/dinov3_query_regions.png" \
    --columns 4 \
    --thumb-width 360

run_stage 08_3d_component_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/dinov3_3d_components.png" \
    --columns 4 \
    --thumb-width 360

{
  echo "mode=report_only_dinov3_3d_first"
  echo "boundary_source=dinov3_mask2former_queries"
  echo "identity_source=dinov3_mask2former_multiview_soft_evidence"
  echo "prior_semantic_labels_used=0"
  echo "camera_selection_policy=${CAMERA_SELECTION_POLICY}"
  echo "camera_selection_used_prior_labels=${CAMERA_SELECTION_USED_PRIOR_LABELS}"
  echo "semantic_identity_hardened_before_3d=0"
  echo "semantic_labels_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

test ! -e "${OUT_DIR}/deliverables/gaussian_labels.npy"
test ! -e "${OUT_DIR}/deliverables/label_map.json"
test ! -e "${OUT_DIR}/deliverables/semantic_point_cloud.ply"
find "${OUT_DIR}" -maxdepth 4 -type f | sort
