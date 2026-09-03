#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_anchor_assoc
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Cached DINOv3-only report:
# immutable strict 24-view components + proposals from a larger report
# -> direct anchored association, consensus audit, and review overlays.
# This scheduler writes no semantic labels, label map, or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
ANCHOR_OUTPUT_NAME="${ANCHOR_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_v1}"
CANDIDATE_OUTPUT_NAME="${CANDIDATE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_targeted_50plus20_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_anchored_association_audit_v1}"

ANCHOR_ROOT="${TASK1_ROOT}/${ANCHOR_OUTPUT_NAME}"
CANDIDATE_ROOT="${TASK1_ROOT}/${CANDIDATE_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
OVERLAY_DIR="${OUT_DIR}/visualizations/anchored_component_overlays"
CONTACT_SHEET="${OUT_DIR}/visualizations/anchored_component_extensions.png"
RUN_LOG_DIR="${OUT_DIR}/logs"

ANCHOR_COMPONENT_REPORT="${ANCHOR_ROOT}/stages/04_3d_components/component_report.json"
ANCHOR_PROPOSAL_MANIFEST="${ANCHOR_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
CANDIDATE_COMPONENT_REPORT="${CANDIDATE_ROOT}/stages/04_3d_components/component_report.json"
CANDIDATE_PROPOSAL_MANIFEST="${CANDIDATE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"

MIN_SHARED_GAUSSIANS="${MIN_SHARED_GAUSSIANS:-250}"
MIN_IOU="${MIN_IOU:-0.05}"
MIN_CONTAINMENT="${MIN_CONTAINMENT:-0.25}"
MIN_FEATURE_SIMILARITY="${MIN_FEATURE_SIMILARITY:-0.0}"
MIN_CANDIDATE_ANCHOR_CONTAINMENT="${MIN_CANDIDATE_ANCHOR_CONTAINMENT:-0.50}"
MIN_UNIQUE_ANCHOR_SCORE_RATIO="${MIN_UNIQUE_ANCHOR_SCORE_RATIO:-1.10}"
MIN_EXTENSION_VIEWS="${MIN_EXTENSION_VIEWS:-2}"
MIN_EXTENSION_GAUSSIANS="${MIN_EXTENSION_GAUSSIANS:-500}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"

for value_name in CONFIG_ONLY RESET_OUTPUT; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${RESET_OUTPUT}" != "0" ]]; then
  echo "This audit requires a fresh output and refuses RESET_OUTPUT=1" >&2
  exit 2
fi

test -f "${ONTOLOGY}"
test -f "${ANCHOR_COMPONENT_REPORT}"
test -f "${ANCHOR_PROPOSAL_MANIFEST}"
test -f "${CANDIDATE_COMPONENT_REPORT}"
test -f "${CANDIDATE_PROPOSAL_MANIFEST}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nanchor=%s\ncandidates=%s\noutput=%s\nmode=dinov3_anchored_component_association_audit\nreport_only=1\nassociation_policy=immutable_anchor_direct_unique_mutual_best_without_transitive_growth\nidentity_policy=strict_or_exactly_one_outlier_leave_one_out_stable\nmin_shared_gaussians=%s\nmin_iou=%s\nmin_containment=%s\nmin_feature_similarity=%s\nmin_candidate_anchor_containment=%s\nmin_unique_anchor_score_ratio=%s\nmin_extension_views=%s\nmin_extension_gaussians=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\noffline_threshold_sweep_used=0\nv5_used=0\nv5_labels_used=0\nv5_method_adapted=1\ndinov2_used=0\nsemantic_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${ANCHOR_ROOT}" "${CANDIDATE_ROOT}" "${OUT_DIR}" \
    "${MIN_SHARED_GAUSSIANS}" "${MIN_IOU}" "${MIN_CONTAINMENT}" \
    "${MIN_FEATURE_SIMILARITY}" "${MIN_CANDIDATE_ANCHOR_CONTAINMENT}" \
    "${MIN_UNIQUE_ANCHOR_SCORE_RATIO}" "${MIN_EXTENSION_VIEWS}" \
    "${MIN_EXTENSION_GAUSSIANS}"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists; choose a fresh OUTPUT_NAME:" >&2
  echo "${OUT_DIR}" >&2
  exit 2
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
  echo "anchor=${ANCHOR_ROOT}"
  echo "candidates=${CANDIDATE_ROOT}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_anchored_component_association_audit"
  echo "report_only=1"
  echo "v5_used=0"
  echo "v5_labels_used=0"
  echo "v5_method_adapted=1"
  echo "dinov2_used=0"
  echo "semantic_labels_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_anchored_component_association_audit \
  python -m scripts.task1.dinov3.audit_anchored_component_association \
    --anchor-component-report "${ANCHOR_COMPONENT_REPORT}" \
    --anchor-proposal-manifest "${ANCHOR_PROPOSAL_MANIFEST}" \
    --candidate-component-report "${CANDIDATE_COMPONENT_REPORT}" \
    --candidate-proposal-manifest "${CANDIDATE_PROPOSAL_MANIFEST}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --scene "${SCENE}" \
    --min-shared-gaussians "${MIN_SHARED_GAUSSIANS}" \
    --min-iou "${MIN_IOU}" \
    --min-containment "${MIN_CONTAINMENT}" \
    --min-feature-similarity "${MIN_FEATURE_SIMILARITY}" \
    --min-candidate-anchor-containment "${MIN_CANDIDATE_ANCHOR_CONTAINMENT}" \
    --min-unique-anchor-score-ratio "${MIN_UNIQUE_ANCHOR_SCORE_RATIO}" \
    --min-extension-views "${MIN_EXTENSION_VIEWS}" \
    --min-extension-gaussians "${MIN_EXTENSION_GAUSSIANS}"

run_stage 02_render_anchored_component_overlays \
  python -m scripts.task1.dinov3.render_3d_component_overlays \
    --association-report "${AUDIT_DIR}/anchored_component_association_audit.json" \
    --proposal-manifest "${CANDIDATE_PROPOSAL_MANIFEST}" \
    --output-dir "${OVERLAY_DIR}"

run_stage 03_anchored_component_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${CONTACT_SHEET}" \
    --columns 4 \
    --thumb-width 300

test -f "${AUDIT_DIR}/anchored_component_association_audit.json"
test -f "${AUDIT_DIR}/proposed_fill_mask.npy"
test -f "${AUDIT_DIR}/proposed_fill_owner_count.npy"
test -f "${AUDIT_DIR}/proposed_fill_overlap_mask.npy"
test -f "${AUDIT_DIR}/proposed_fill_supports.npz"
test -f "${OVERLAY_DIR}/component_overlay_manifest.json"
test -f "${CONTACT_SHEET}"
test ! -e "${AUDIT_DIR}/gaussian_labels.npy"
test ! -e "${AUDIT_DIR}/gaussian_project_class_ids.npy"
test ! -e "${AUDIT_DIR}/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

{
  echo "mode=dinov3_anchored_component_association_audit"
  echo "report_only=1"
  echo "anchor=${ANCHOR_ROOT}"
  echo "candidates=${CANDIDATE_ROOT}"
  echo "association_policy=immutable_anchor_direct_unique_mutual_best_without_transitive_growth"
  echo "identity_policy=strict_or_exactly_one_outlier_leave_one_out_stable"
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

find "${OUT_DIR}" -maxdepth 4 -type f | sort
