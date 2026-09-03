#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_partial_audit
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Cached DINOv3-only report:
# immutable camera-owned component anchors + complete dense DINOv3 candidates
# -> conservative v5-style partial-anchor completion audit. No labels or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_camera_owned_dense_propagation_v2}"
DENSE_OUTPUT_NAME="${DENSE_OUTPUT_NAME:-${SCENE}_dinov3_dense_vote_complete_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_partial_anchor_dense_completion_audit_v1}"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
COMPONENT_DIR="${SOURCE_ROOT}/stages/01_plurality_component_seeds"
COMBINED_SEED_DIR="${SOURCE_ROOT}/stages/02_combined_seeds"
DENSE_DIR="${TASK1_ROOT}/${DENSE_OUTPUT_NAME}/deliverables"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
RUN_LOG_DIR="${OUT_DIR}/logs"

ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
MIN_ANCHOR_GAUSSIANS="${MIN_ANCHOR_GAUSSIANS:-500}"
MIN_ANCHOR_COVERAGE="${MIN_ANCHOR_COVERAGE:-0.80}"
MIN_ANCHOR_PRECISION="${MIN_ANCHOR_PRECISION:-0.10}"
MAX_ANCHOR_PRECISION="${MAX_ANCHOR_PRECISION:-0.25}"
MIN_ROBUST_FRACTION="${MIN_ROBUST_FRACTION:-0.90}"
MIN_SOURCE_VIEWS="${MIN_SOURCE_VIEWS:-5}"
MIN_CANDIDATE_SUPPORTING_VIEWS="${MIN_CANDIDATE_SUPPORTING_VIEWS:-5}"
MAX_COMPETING_THING_FRACTION="${MAX_COMPETING_THING_FRACTION:-0.05}"
MIN_SPATIAL_KEEP_FRACTION="${MIN_SPATIAL_KEEP_FRACTION:-0.95}"
MAX_PARTIAL_COMPETING_THING_RISK="${MAX_PARTIAL_COMPETING_THING_RISK:-0.40}"
VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"
MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"
MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

for value_name in RESET_OUTPUT CONFIG_ONLY; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1" >&2
    exit 2
  fi
done

test -d "${MODEL_DIR}"
test -f "${SOURCE_PLY}"
test -f "${ONTOLOGY}"
test -f "${COMBINED_SEED_DIR}/gaussian_project_class_ids.npy"
test -f "${COMBINED_SEED_DIR}/combined_seed_summary.json"
test -f "${COMPONENT_DIR}/gaussian_labels.npy"
test -f "${COMPONENT_DIR}/gaussian_project_class_ids.npy"
test -f "${COMPONENT_DIR}/label_map.json"
test -f "${COMPONENT_DIR}/camera_owned_component_summary.json"
test -f "${DENSE_DIR}/gaussian_project_class_ids.npy"
test -f "${DENSE_DIR}/supporting_views.npy"
test -f "${DENSE_DIR}/winner_share.npy"
test -f "${DENSE_DIR}/dense_vote_summary.json"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\ndense_candidates=%s\noutput=%s\nmode=dinov3_partial_anchor_dense_completion_audit\nreport_only=1\nanchor_policy=immutable_dinov3_component_instance\nmin_anchor_gaussians=%s\nmin_anchor_coverage=%s\nmin_anchor_precision=%s\nmax_anchor_precision=%s\nmin_robust_fraction=%s\nmin_source_views=%s\nmin_candidate_supporting_views=%s\nmax_competing_thing_fraction=%s\nmin_spatial_keep_fraction=%s\nmax_partial_competing_thing_risk=%s\nvoxel_scale_multiplier=%s\nmin_voxel_size=%s\nmax_voxel_size=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\noffline_threshold_sweep_used=0\nv5_used=0\nv5_labels_used=0\nv5_method_adapted=1\ndinov2_used=0\nsemantic_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${DENSE_DIR}" "${OUT_DIR}" \
    "${MIN_ANCHOR_GAUSSIANS}" "${MIN_ANCHOR_COVERAGE}" \
    "${MIN_ANCHOR_PRECISION}" "${MAX_ANCHOR_PRECISION}" \
    "${MIN_ROBUST_FRACTION}" "${MIN_SOURCE_VIEWS}" \
    "${MIN_CANDIDATE_SUPPORTING_VIEWS}" \
    "${MAX_COMPETING_THING_FRACTION}" "${MIN_SPATIAL_KEEP_FRACTION}" \
    "${MAX_PARTIAL_COMPETING_THING_RISK}" "${VOXEL_SCALE_MULTIPLIER}" \
    "${MIN_VOXEL_SIZE}" "${MAX_VOXEL_SIZE}"
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

mkdir -p "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" 2>&1 \
    | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "source=${SOURCE_ROOT}"
  echo "dense_candidates=${DENSE_DIR}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_partial_anchor_dense_completion_audit"
  echo "report_only=1"
  echo "v5_used=0"
  echo "v5_labels_used=0"
  echo "v5_method_adapted=1"
  echo "dinov2_used=0"
  echo "semantic_labels_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_partial_anchor_completion_audit \
  python -m scripts.task1.dinov3.audit_partial_anchor_dense_completion \
    --combined-seed-labels "${COMBINED_SEED_DIR}/gaussian_project_class_ids.npy" \
    --combined-seed-summary "${COMBINED_SEED_DIR}/combined_seed_summary.json" \
    --component-instance-labels "${COMPONENT_DIR}/gaussian_labels.npy" \
    --component-project-labels "${COMPONENT_DIR}/gaussian_project_class_ids.npy" \
    --component-label-map "${COMPONENT_DIR}/label_map.json" \
    --component-summary "${COMPONENT_DIR}/camera_owned_component_summary.json" \
    --dense-labels "${DENSE_DIR}/gaussian_project_class_ids.npy" \
    --dense-supporting-views "${DENSE_DIR}/supporting_views.npy" \
    --dense-winner-share "${DENSE_DIR}/winner_share.npy" \
    --dense-summary "${DENSE_DIR}/dense_vote_summary.json" \
    --ontology "${ONTOLOGY}" \
    --source-ply "${SOURCE_PLY}" \
    --output-dir "${AUDIT_DIR}" \
    --scene "${SCENE}" \
    --min-anchor-gaussians "${MIN_ANCHOR_GAUSSIANS}" \
    --min-anchor-coverage "${MIN_ANCHOR_COVERAGE}" \
    --min-anchor-precision "${MIN_ANCHOR_PRECISION}" \
    --max-anchor-precision "${MAX_ANCHOR_PRECISION}" \
    --min-robust-fraction "${MIN_ROBUST_FRACTION}" \
    --min-source-views "${MIN_SOURCE_VIEWS}" \
    --min-candidate-supporting-views "${MIN_CANDIDATE_SUPPORTING_VIEWS}" \
    --max-competing-thing-fraction "${MAX_COMPETING_THING_FRACTION}" \
    --min-spatial-keep-fraction "${MIN_SPATIAL_KEEP_FRACTION}" \
    --max-partial-competing-thing-risk "${MAX_PARTIAL_COMPETING_THING_RISK}" \
    --voxel-scale-multiplier "${VOXEL_SCALE_MULTIPLIER}" \
    --min-voxel-size "${MIN_VOXEL_SIZE}" \
    --max-voxel-size "${MAX_VOXEL_SIZE}"

test -f "${AUDIT_DIR}/partial_anchor_completion_audit.json"
test -f "${AUDIT_DIR}/proposed_fill_mask.npy"
test -f "${AUDIT_DIR}/proposed_fill_owner_count.npy"
test -f "${AUDIT_DIR}/proposed_fill_overlap_mask.npy"
test -f "${AUDIT_DIR}/proposed_fill_supports.npz"
test ! -e "${AUDIT_DIR}/gaussian_labels.npy"
test ! -e "${AUDIT_DIR}/gaussian_project_class_ids.npy"
test ! -e "${AUDIT_DIR}/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

{
  echo "mode=dinov3_partial_anchor_dense_completion_audit"
  echo "report_only=1"
  echo "source=${SOURCE_ROOT}"
  echo "dense_candidates=${DENSE_DIR}"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "offline_threshold_sweep_used=0"
  echo "v5_used=0"
  echo "v5_labels_used=0"
  echo "v5_method_adapted=1"
  echo "dinov2_used=0"
  echo "semantic_labels_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 3 -type f | sort
