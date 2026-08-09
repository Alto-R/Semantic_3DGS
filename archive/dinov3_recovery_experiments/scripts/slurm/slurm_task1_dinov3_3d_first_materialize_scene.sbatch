#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_materialize
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Materialize a completed, reviewed DINOv3 3D-first report without rerunning
# rendering, DINOv3, or FlashSplat. Cross-class Gaussian conflicts abstain.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_3d_first_conflict_abstain_v1}"
SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
RUN_LOG_DIR="${OUT_DIR}/logs"

PROPOSAL_MANIFEST="${SOURCE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
COMPONENT_REPORT="${SOURCE_ROOT}/stages/04_3d_components/component_report.json"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"

WRITE_SEMANTIC_PLY="${WRITE_SEMANTIC_PLY:-1}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

for value_name in WRITE_SEMANTIC_PLY RESET_OUTPUT CONFIG_ONLY; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1" >&2
    exit 2
  fi
done

test -f "${PROPOSAL_MANIFEST}"
test -f "${COMPONENT_REPORT}"
test -f "${SOURCE_PLY}"
test -f "${ONTOLOGY}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nassignment_policy=cross_class_conflict_abstain\nprior_semantic_labels_used=0\ndinov2_used=0\nsemantic_labels_written=1\nsemantic_ply_written=%s\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}" "${WRITE_SEMANTIC_PLY}"
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

command=(
  python -m scripts.task1.dinov3.materialize_3d_component_labels
  --proposal-manifest "${PROPOSAL_MANIFEST}"
  --component-report "${COMPONENT_REPORT}"
  --ontology "${ONTOLOGY}"
  --source-ply "${SOURCE_PLY}"
  --output-dir "${DELIVERABLE_DIR}"
  --scene "${SCENE}"
)
if [[ "${WRITE_SEMANTIC_PLY}" = "0" ]]; then
  command+=(--no-semantic-ply)
fi

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "source=${SOURCE_ROOT}"
  echo "output=${OUT_DIR}"
  echo "assignment_policy=cross_class_conflict_abstain"
  echo "prior_semantic_labels_used=0"
  echo "dinov2_used=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

conda run --no-capture-output -n "${GAUSSIAN_ENV}" "${command[@]}" \
  2>&1 | tee "${RUN_LOG_DIR}/01_materialize_conflict_abstaining_labels.log"

{
  echo "mode=dinov3_3d_first_conflict_abstain_materialization"
  echo "source_report=${SOURCE_ROOT}"
  echo "prior_semantic_labels_used=0"
  echo "dinov2_used=0"
  echo "semantic_identity_hardened_before_3d=0"
  echo "cross_class_conflicts_abstain=1"
  echo "semantic_labels_written=1"
  echo "semantic_ply_written=${WRITE_SEMANTIC_PLY}"
} > "${OUT_DIR}/experiment_mode.txt"

test -f "${DELIVERABLE_DIR}/gaussian_labels.npy"
test -f "${DELIVERABLE_DIR}/gaussian_project_class_ids.npy"
test -f "${DELIVERABLE_DIR}/cross_class_conflict_mask.npy"
test -f "${DELIVERABLE_DIR}/label_map.json"
test -f "${DELIVERABLE_DIR}/materialization_summary.json"
if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  test -f "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
else
  test ! -e "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
fi

find "${OUT_DIR}" -maxdepth 3 -type f | sort
