#!/usr/bin/env bash
#SBATCH --job-name=task1_independent_evidence
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:-room}"
AUDIT_MODE="${AUDIT_MODE:-v9}"
SOURCE_NAME="${SOURCE_NAME:-${SCENE}_dinov2_separate_abstain_allviews_v1}"
CACHE_NAME="${CACHE_NAME:-${SCENE}_dinov2_ade_refinement_auto_spatial_thing_guard_v3}"
BASELINE_NAME="${BASELINE_NAME:-${SCENE}_dinov2_ade_refinement_adaptive_instance_guard_v5}"
if [[ "${AUDIT_MODE}" = "v10" ]]; then
  DEFAULT_OUTPUT_NAME="${SCENE}_identity_component_audit_v10"
  CONTACT_SHEET_NAME="identity_component_contact_sheet.png"
  EXPERIMENT_MODE="report_only_identity_consistent_adaptive_components_v10"
elif [[ "${AUDIT_MODE}" = "v9" ]]; then
  DEFAULT_OUTPUT_NAME="${SCENE}_independent_multiview_evidence_v9"
  CONTACT_SHEET_NAME="independent_evidence_contact_sheet.png"
  EXPERIMENT_MODE="report_only_independent_multiview_evidence"
else
  echo "AUDIT_MODE must be v9 or v10, got: ${AUDIT_MODE}" >&2
  exit 2
fi
OUTPUT_NAME="${OUTPUT_NAME:-${DEFAULT_OUTPUT_NAME}}"

SOURCE_DIR="${TASK1_ROOT}/${SOURCE_NAME}"
CACHE_DIR="${TASK1_ROOT}/${CACHE_NAME}"
BASELINE_DIR="${TASK1_ROOT}/${BASELINE_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_VIEW_DIR="${SOURCE_DIR}/stages/01_real_camera_views"
CACHE_CONFIG="${CACHE_DIR}/stages/04_grounding_camera_selection/automatic_ade_refinement_config.json"
CACHE_PROPOSALS="${CACHE_DIR}/stages/06_grounding_flashsplat"
CACHE_GROUNDED_SAM_MANIFEST="${CACHE_GROUNDED_SAM_MANIFEST:-}"
CACHE_GROUNDING_MASKS="${CACHE_GROUNDING_MASKS:-}"
BASELINE_LABELS="${BASELINE_DIR}/deliverables/gaussian_labels.npy"
BASELINE_LABEL_MAP="${BASELINE_DIR}/deliverables/label_map.json"
RUN_LOG_DIR="${OUT_DIR}/logs"

GROUNDING_ENV="${GROUNDING_ENV:-gaussian_grouping_true}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${WORKSPACE_ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"

MAX_SEEDS="${MAX_SEEDS:-48}"
MIN_SOURCE_ANGLE_DEGREES="${MIN_SOURCE_ANGLE_DEGREES:-8.0}"
MIN_PAIRWISE_ANGLE_DEGREES="${MIN_PAIRWISE_ANGLE_DEGREES:-8.0}"
MAX_TARGET_VIEWS="${MAX_TARGET_VIEWS:-6}"
MIN_SECOND_SEED_CONTRIBUTION_FRACTION="${MIN_SECOND_SEED_CONTRIBUTION_FRACTION:-0.20}"
MIN_SECOND_TO_BEST_CONTRIBUTION_RATIO="${MIN_SECOND_TO_BEST_CONTRIBUTION_RATIO:-0.50}"
COMPONENT_NEIGHBOR_K="${COMPONENT_NEIGHBOR_K:-4}"
COMPONENT_RADIUS_MULTIPLIER="${COMPONENT_RADIUS_MULTIPLIER:-2.0}"
COMPONENT_SPACING_MAX_SAMPLES="${COMPONENT_SPACING_MAX_SAMPLES:-512}"
DIAGNOSTIC_MAX_GROUNDING_SHORTFALL="${DIAGNOSTIC_MAX_GROUNDING_SHORTFALL:-0.05}"
DIAGNOSTIC_MAX_SAM_SHORTFALL="${DIAGNOSTIC_MAX_SAM_SHORTFALL:-0.025}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${CACHE_CONFIG}"
test -f "${CACHE_PROPOSALS}/proposal_manifest.json"
test -f "${BASELINE_LABELS}"
test -f "${BASELINE_LABEL_MAP}"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${ONTOLOGY}"
test -d "${FLASHSPLAT_ROOT}"
test -d "${GROUNDED_SAM_ROOT}"
test -f "${GROUNDINGDINO_CONFIG}"
test -f "${GROUNDINGDINO_CHECKPOINT}"
test -d "${SAM_ROOT}"
test -f "${SAM_CHECKPOINT}"
command -v jq >/dev/null 2>&1

if [[ -z "${CACHE_GROUNDED_SAM_MANIFEST}" ]]; then
  CACHE_GROUNDED_SAM_MANIFEST="$(jq -r '.source_manifest // empty' "${CACHE_PROPOSALS}/proposal_manifest.json")"
fi
test -f "${CACHE_GROUNDED_SAM_MANIFEST}"
if [[ -z "${CACHE_GROUNDING_MASKS}" ]]; then
  CACHE_GROUNDING_MASKS="$(dirname -- "${CACHE_GROUNDED_SAM_MANIFEST}")"
fi
test -d "${CACHE_GROUNDING_MASKS}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\naudit_mode=%s\nsource=%s\ncache=%s\nbaseline=%s\noutput=%s\ngrounded_sam_manifest=%s\ngrounded_sam_dir=%s\nmax_seeds=%s\nsource_angle=%s\npairwise_angle=%s\nmax_target_views=%s\nsecond_seed_fraction=%s\nsecond_to_best_ratio=%s\ncomponent_neighbor_k=%s\ncomponent_radius_multiplier=%s\ncomponent_spacing_samples=%s\ndiagnostic_grounding_shortfall=%s\ndiagnostic_sam_shortfall=%s\n' \
    "${SCENE}" "${AUDIT_MODE}" "${SOURCE_DIR}" "${CACHE_DIR}" "${BASELINE_DIR}" "${OUT_DIR}" \
    "${CACHE_GROUNDED_SAM_MANIFEST}" "${CACHE_GROUNDING_MASKS}" \
    "${MAX_SEEDS}" "${MIN_SOURCE_ANGLE_DEGREES}" \
    "${MIN_PAIRWISE_ANGLE_DEGREES}" "${MAX_TARGET_VIEWS}" \
    "${MIN_SECOND_SEED_CONTRIBUTION_FRACTION}" \
    "${MIN_SECOND_TO_BEST_CONTRIBUTION_RATIO}" "${COMPONENT_NEIGHBOR_K}" \
    "${COMPONENT_RADIUS_MULTIPLIER}" "${COMPONENT_SPACING_MAX_SAMPLES}" \
    "${DIAGNOSTIC_MAX_GROUNDING_SHORTFALL}" "${DIAGNOSTIC_MAX_SAM_SHORTFALL}"
  exit 0
fi

if [[ "${RESET_OUTPUT}" = "1" && -d "${OUT_DIR}" ]]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p "${OUT_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "===== independent_multiview_evidence ====="
AUDIT_EXTRA_ARGS=()
if [[ "${AUDIT_MODE}" = "v10" ]]; then
  AUDIT_EXTRA_ARGS+=(--diagnostic-lift-quality-failures)
fi
conda run --no-capture-output -n "${GROUNDING_ENV}" \
  python scripts/task1/audit_independent_multiview_evidence.py \
    --model-path "${MODEL_DIR}" \
    --ground-output-dir "${CACHE_GROUNDING_MASKS}" \
    --input-manifest-name "$(basename -- "${CACHE_GROUNDED_SAM_MANIFEST}")" \
    --proposal-dir "${CACHE_PROPOSALS}" \
    --class-config "${CACHE_CONFIG}" \
    --ontology "${ONTOLOGY}" \
    --baseline-labels "${BASELINE_LABELS}" \
    --baseline-label-map "${BASELINE_LABEL_MAP}" \
    --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
    --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --groundingdino-root "${GROUNDED_SAM_ROOT}" \
    --groundingdino-config "${GROUNDINGDINO_CONFIG}" \
    --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}" \
    --segment-anything-root "${SAM_ROOT}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --scene "${SCENE}" \
    --max-seeds "${MAX_SEEDS}" \
    --min-source-angle-degrees "${MIN_SOURCE_ANGLE_DEGREES}" \
    --min-pairwise-angle-degrees "${MIN_PAIRWISE_ANGLE_DEGREES}" \
    --max-target-views "${MAX_TARGET_VIEWS}" \
    --audit-mode "${AUDIT_MODE}" \
    --min-second-seed-contribution-fraction "${MIN_SECOND_SEED_CONTRIBUTION_FRACTION}" \
    --min-second-to-best-contribution-ratio "${MIN_SECOND_TO_BEST_CONTRIBUTION_RATIO}" \
    --component-neighbor-k "${COMPONENT_NEIGHBOR_K}" \
    --component-radius-multiplier "${COMPONENT_RADIUS_MULTIPLIER}" \
    --component-spacing-max-samples "${COMPONENT_SPACING_MAX_SAMPLES}" \
    --diagnostic-max-grounding-shortfall "${DIAGNOSTIC_MAX_GROUNDING_SHORTFALL}" \
    --diagnostic-max-sam-shortfall "${DIAGNOSTIC_MAX_SAM_SHORTFALL}" \
    "${AUDIT_EXTRA_ARGS[@]}" \
    --overwrite 2>&1 | tee "${RUN_LOG_DIR}/independent_multiview_evidence.log"

if find "${OUT_DIR}/overlays" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q .; then
  conda run --no-capture-output -n "${GROUNDING_ENV}" \
    python scripts/task1/make_contact_sheet.py \
      --image-dir "${OUT_DIR}/overlays" \
      --output "${OUT_DIR}/${CONTACT_SHEET_NAME}" \
      --columns 4 \
      --thumb-width 360
fi

printf 'mode=%s\nsemantic_labels_written=0\nsemantic_ply_written=0\n' "${EXPERIMENT_MODE}" \
  > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 3 -type f | sort
