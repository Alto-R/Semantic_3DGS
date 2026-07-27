#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_region_vote
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Cached direct multiview voting over every lifted DINOv3 query region.
# No GPU, region-to-region association, v5, or DINOv2 input is used.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_region_vote_majority_v1}"
SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
RUN_LOG_DIR="${OUT_DIR}/logs"
VALIDATION_DIR="${OUT_DIR}/validation"

PROPOSAL_MANIFEST="${SOURCE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"

MIN_VIEWS="${MIN_VIEWS:-2}"
VOTE_BATCH_SIZE="${VOTE_BATCH_SIZE:-100000}"
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
if (( MIN_VIEWS < 2 )); then
  echo "MIN_VIEWS must be at least two" >&2
  exit 2
fi
if (( VOTE_BATCH_SIZE < 1 )); then
  echo "VOTE_BATCH_SIZE must be positive" >&2
  exit 2
fi

test -f "${PROPOSAL_MANIFEST}"
test -f "${SOURCE_PLY}"
test -f "${ONTOLOGY}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nmode=direct_per_gaussian_region_vote\nmin_views=%s\nregion_association_used=0\nprior_semantic_labels_used=0\ndinov2_used=0\nsemantic_labels_written=1\nsemantic_ply_written=%s\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}" "${MIN_VIEWS}" "${WRITE_SEMANTIC_PLY}"
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

mkdir -p "${RUN_LOG_DIR}" "${VALIDATION_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

command=(
  python -m scripts.task1.dinov3.fuse_query_region_votes
  --proposal-manifest "${PROPOSAL_MANIFEST}"
  --ontology "${ONTOLOGY}"
  --source-ply "${SOURCE_PLY}"
  --output-dir "${DELIVERABLE_DIR}"
  --scene "${SCENE}"
  --min-views "${MIN_VIEWS}"
  --batch-size "${VOTE_BATCH_SIZE}"
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
  echo "mode=direct_per_gaussian_region_vote"
  echo "region_association_used=0"
  echo "prior_semantic_labels_used=0"
  echo "dinov2_used=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

conda run --no-capture-output -n "${GAUSSIAN_ENV}" "${command[@]}" \
  2>&1 | tee "${RUN_LOG_DIR}/01_fuse_query_region_votes.log"

validation=(
  python -m scripts.task1.qa.validate_task1_outputs
  --labels "${DELIVERABLE_DIR}/gaussian_labels.npy"
  --label-map "${DELIVERABLE_DIR}/label_map.json"
)
if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  validation+=(--semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply")
fi
conda run --no-capture-output -n "${GAUSSIAN_ENV}" "${validation[@]}" \
  2>&1 | tee "${VALIDATION_DIR}/validate_task1_outputs.log"

{
  echo "mode=dinov3_query_regions_direct_multiview_majority"
  echo "source_report=${SOURCE_ROOT}"
  echo "region_association_used=0"
  echo "prior_semantic_labels_used=0"
  echo "dinov2_used=0"
  echo "one_vote_per_camera=1"
  echo "strict_majority_required=1"
  echo "semantic_labels_written=1"
  echo "semantic_ply_written=${WRITE_SEMANTIC_PLY}"
} > "${OUT_DIR}/experiment_mode.txt"

test -f "${DELIVERABLE_DIR}/gaussian_labels.npy"
test -f "${DELIVERABLE_DIR}/gaussian_project_class_ids.npy"
test -f "${DELIVERABLE_DIR}/winner_ade20k_ids.npy"
test -f "${DELIVERABLE_DIR}/supporting_views.npy"
test -f "${DELIVERABLE_DIR}/hard_winner_agreement.npy"
test -f "${DELIVERABLE_DIR}/soft_winner_confidence.npy"
test -f "${DELIVERABLE_DIR}/abstain_reason_codes.npy"
test -f "${DELIVERABLE_DIR}/label_map.json"
test -f "${DELIVERABLE_DIR}/region_vote_summary.json"
if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  test -f "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
else
  test ! -e "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
fi

find "${OUT_DIR}" -maxdepth 3 -type f | sort
