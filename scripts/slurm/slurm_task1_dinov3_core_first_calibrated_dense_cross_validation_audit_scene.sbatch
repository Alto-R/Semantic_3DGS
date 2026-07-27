#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_cal_xval
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Fully automatic report-only boundary diagnostic:
# two complete cached 70-view pixel sets -> one joint empirical calibration
# -> four nested profiles in one FlashSplat render per camera -> immutable-v2
# black intersection -> per-Gaussian camera identity -> spatial re-split.
# No scene/class/component decisions, labels, label maps, or PLY are written.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_targeted_50plus20_v1}"
CORE_FIRST_OUTPUT_NAME="${CORE_FIRST_OUTPUT_NAME:-${SCENE}_dinov3_core_first_semantic_identity_audit_v1}"
PREFERRED_OUTPUT_NAME="${PREFERRED_OUTPUT_NAME:-${SCENE}_dinov3_camera_owned_dense_propagation_v2}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_core_first_calibrated_dense_cross_validation_audit_v2}"

CALIBRATION_SOURCE_A_OUTPUT_NAME="${CALIBRATION_SOURCE_A_OUTPUT_NAME:-drjohnson_dinov3_3d_first_report_targeted_50plus20_v1}"
CALIBRATION_SOURCE_B_OUTPUT_NAME="${CALIBRATION_SOURCE_B_OUTPUT_NAME:-playroom_dinov3_3d_first_report_targeted_50plus20_v1}"
CALIBRATION_SOURCE_A_DIR="${TASK1_ROOT}/${CALIBRATION_SOURCE_A_OUTPUT_NAME}/stages/01_dinov3_query_evidence"
CALIBRATION_SOURCE_B_DIR="${TASK1_ROOT}/${CALIBRATION_SOURCE_B_OUTPUT_NAME}/stages/01_dinov3_query_evidence"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
CORE_FIRST_ROOT="${TASK1_ROOT}/${CORE_FIRST_OUTPUT_NAME}"
PREFERRED_ROOT="${TASK1_ROOT}/${PREFERRED_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
CALIBRATION_DIR="${OUT_DIR}/calibration"
EVIDENCE_DIR="${OUT_DIR}/evidence"
AUDIT_DIR="${OUT_DIR}/audit"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
RUN_LOG_DIR="${OUT_DIR}/logs"

SOURCE_VIEW_DIR="${SOURCE_ROOT}/stages/01_dinov3_query_evidence"
SOURCE_VIEW_MANIFEST="${SOURCE_VIEW_DIR}/view_manifest.json"
SEGMENTATION_MANIFEST="${SOURCE_VIEW_DIR}/dinov3_manifest.json"
CORE_FIRST_REPORT="${CORE_FIRST_ROOT}/audit/core_first_semantic_identity_audit.json"
CORE_FIRST_SUPPORTS="${CORE_FIRST_ROOT}/audit/core_first_supports.npz"
PREFERRED_LABELS="${PREFERRED_ROOT}/deliverables/gaussian_labels.npy"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
CALIBRATION_BIN_COUNT="${CALIBRATION_BIN_COUNT:-4096}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
MIN_RELIABLE_CAMERAS="${MIN_RELIABLE_CAMERAS:-3}"
MIN_CAMERA_WINNER_SHARE="${MIN_CAMERA_WINNER_SHARE:-0.50}"
MIN_CAMERA_WINNER_MARGIN="${MIN_CAMERA_WINNER_MARGIN:-0.10}"
MIN_GLOBAL_WINNER_SHARE="${MIN_GLOBAL_WINNER_SHARE:-0.55}"
MIN_GLOBAL_WINNER_MARGIN="${MIN_GLOBAL_WINNER_MARGIN:-0.10}"
MIN_BOUNDARY_MARGIN="${MIN_BOUNDARY_MARGIN:-0.10}"
MIN_SPATIAL_COMPONENT_GAUSSIANS="${MIN_SPATIAL_COMPONENT_GAUSSIANS:-500}"
MIN_SPATIAL_COMPONENT_CAMERAS="${MIN_SPATIAL_COMPONENT_CAMERAS:-3}"
VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"
MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"
MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"
FUSION_CHUNK_SIZE="${FUSION_CHUNK_SIZE:-100000}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
PROFILES=(baseline permissive balanced strict)

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
if (( CALIBRATION_BIN_COUNT < 256 )); then
  echo "CALIBRATION_BIN_COUNT must be at least 256" >&2
  exit 2
fi
if (( MIN_RELIABLE_CAMERAS < 2 )); then
  echo "MIN_RELIABLE_CAMERAS must be at least two" >&2
  exit 2
fi
if (( MIN_SPATIAL_COMPONENT_GAUSSIANS < 1 )); then
  echo "MIN_SPATIAL_COMPONENT_GAUSSIANS must be positive" >&2
  exit 2
fi
if (( MIN_SPATIAL_COMPONENT_CAMERAS < 2 )); then
  echo "MIN_SPATIAL_COMPONENT_CAMERAS must be at least two" >&2
  exit 2
fi
if (( FUSION_CHUNK_SIZE < 1 )); then
  echo "FUSION_CHUNK_SIZE must be positive" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -d "${FLASHSPLAT_ROOT}"
test -f "${SOURCE_PLY}"
test -f "${SEGMENTATION_MANIFEST}"
test -f "${SOURCE_VIEW_MANIFEST}"
test -f "${CORE_FIRST_REPORT}"
test -f "${CORE_FIRST_SUPPORTS}"
test -f "${PREFERRED_LABELS}"
test -f "${ONTOLOGY}"
test -f "${CALIBRATION_SOURCE_A_DIR}/dinov3_manifest.json"
test -f "${CALIBRATION_SOURCE_B_DIR}/dinov3_manifest.json"
if [[ "${CALIBRATION_SOURCE_A_DIR}" = "${CALIBRATION_SOURCE_B_DIR}" ]]; then
  echo "Joint calibration source directories must be different" >&2
  exit 2
fi

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\ncore_first=%s\npreferred_v2=%s\noutput=%s\nmode=dinov3_core_first_calibrated_dense_cross_validation_audit_v2\nreport_only=1\npreferred_labels_read_only=1\ncalibration_scope=joint_two_complete_cached_70_view_sources\ncalibration_source_a=%s\ncalibration_source_b=%s\ncalibration_metric=weakest_empirical_cdf_rank_of_relative_margin,max_probability,entropy_confidence\nprofiles=baseline,permissive,balanced,strict\nprofile_target_retention=1.00,0.50,0.25,0.10\nsingle_flashsplat_render_shared_across_profiles=1\nabstain_mass_preserved=1\nmin_reliable_cameras=%s\nmin_camera_accepted_fraction=automatic_half_joint_profile_retention\ncamera_winner_share_denominator=accepted_semantic_mass\nmin_camera_winner_share=%s\nmin_camera_winner_margin=%s\nmin_global_winner_share=%s\nmin_global_winner_margin=%s\nmin_boundary_margin=%s\nboundary_classes=wall,ceiling,floor\nmin_spatial_component_gaussians=%s\nmin_spatial_component_cameras=%s\nvoxel_scale_multiplier=%s\nmin_voxel_size=%s\nmax_voxel_size=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\nmanual_component_decisions=0\ninference_rerun=0\nflashsplat_rerun=1\nsemantic_labels_written=0\nsemantic_project_class_arrays_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${CORE_FIRST_ROOT}" "${PREFERRED_ROOT}" \
    "${OUT_DIR}" "${CALIBRATION_SOURCE_A_DIR}" \
    "${CALIBRATION_SOURCE_B_DIR}" "${MIN_RELIABLE_CAMERAS}" \
    "${MIN_CAMERA_WINNER_SHARE}" "${MIN_CAMERA_WINNER_MARGIN}" \
    "${MIN_GLOBAL_WINNER_SHARE}" "${MIN_GLOBAL_WINNER_MARGIN}" \
    "${MIN_BOUNDARY_MARGIN}" "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    "${MIN_SPATIAL_COMPONENT_CAMERAS}" "${VOXEL_SCALE_MULTIPLIER}" \
    "${MIN_VOXEL_SIZE}" "${MAX_VOXEL_SIZE}"
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
  echo "source=${SOURCE_ROOT}"
  echo "core_first=${CORE_FIRST_ROOT}"
  echo "preferred_v2=${PREFERRED_ROOT}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_core_first_calibrated_dense_cross_validation_audit_v2"
  echo "report_only=1"
  echo "preferred_labels_read_only=1"
  echo "calibration_scope=joint_two_complete_cached_70_view_sources"
  echo "profiles=baseline,permissive,balanced,strict"
  echo "profile_target_retention=1.00,0.50,0.25,0.10"
  echo "single_flashsplat_render_shared_across_profiles=1"
  echo "abstain_mass_preserved=1"
  echo "camera_winner_share_denominator=accepted_semantic_mass"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "manual_component_decisions=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=1"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_joint_confidence_calibration \
  python -m scripts.task1.dinov3.calibrate_dense_confidence \
    --input-dir "${CALIBRATION_SOURCE_A_DIR}" \
    --input-dir "${CALIBRATION_SOURCE_B_DIR}" \
    --output-dir "${CALIBRATION_DIR}" \
    --bin-count "${CALIBRATION_BIN_COUNT}"

run_stage 02_lift_all_calibrated_profiles \
  python -m scripts.task1.dinov3.lift_calibrated_dense_view_votes \
    --model-path "${MODEL_DIR}" \
    --input-dir "${SOURCE_VIEW_DIR}" \
    --segmentation-manifest "${SEGMENTATION_MANIFEST}" \
    --calibration-manifest "${CALIBRATION_DIR}/confidence_calibration.json" \
    --calibration-npz "${CALIBRATION_DIR}/confidence_calibration.npz" \
    --output-dir "${EVIDENCE_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}"

profile_reports=()
for profile in "${PROFILES[@]}"; do
  profile_evidence="${EVIDENCE_DIR}/${profile}"
  profile_audit="${AUDIT_DIR}/${profile}"
  profile_visuals="${VISUALIZATION_DIR}/${profile}"
  profile_overlays="${profile_visuals}/exact_overlays"
  profile_masks="${profile_visuals}/exact_masks"

  run_stage "03_audit_${profile}_dense_cross_validation" \
    python -m scripts.task1.dinov3.audit_core_first_dense_cross_validation \
      --core-first-report "${CORE_FIRST_REPORT}" \
      --core-first-supports "${CORE_FIRST_SUPPORTS}" \
      --vote-manifest "${profile_evidence}/vote_manifest.json" \
      --preferred-labels "${PREFERRED_LABELS}" \
      --source-ply "${SOURCE_PLY}" \
      --ontology "${ONTOLOGY}" \
      --output-dir "${profile_audit}" \
      --scene "${SCENE}" \
      --min-reliable-cameras "${MIN_RELIABLE_CAMERAS}" \
      --min-camera-winner-share "${MIN_CAMERA_WINNER_SHARE}" \
      --min-camera-winner-margin "${MIN_CAMERA_WINNER_MARGIN}" \
      --min-global-winner-share "${MIN_GLOBAL_WINNER_SHARE}" \
      --min-global-winner-margin "${MIN_GLOBAL_WINNER_MARGIN}" \
      --min-boundary-margin "${MIN_BOUNDARY_MARGIN}" \
      --min-spatial-component-gaussians \
        "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
      --min-spatial-component-cameras \
        "${MIN_SPATIAL_COMPONENT_CAMERAS}" \
      --voxel-scale-multiplier "${VOXEL_SCALE_MULTIPLIER}" \
      --min-voxel-size "${MIN_VOXEL_SIZE}" \
      --max-voxel-size "${MAX_VOXEL_SIZE}" \
      --chunk-size "${FUSION_CHUNK_SIZE}"

  run_stage "04_render_${profile}_exact_support" \
    python -m scripts.task1.dinov3.render_incremental_spatial_core_fill \
      --audit-report "${profile_audit}/dense_cross_validation_audit.json" \
      --support-npz "${profile_audit}/dense_validated_supports.npz" \
      --model-path "${MODEL_DIR}" \
      --source-view-manifest "${SOURCE_VIEW_MANIFEST}" \
      --overlay-output-dir "${profile_overlays}" \
      --mask-output-dir "${profile_masks}" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --iteration 30000 \
      --max-width "${RENDER_MAX_WIDTH}"

  run_stage "05_contact_sheet_${profile}_overlays" \
    python -m scripts.task1.qa.make_contact_sheet \
      --image-dir "${profile_overlays}" \
      --output "${profile_visuals}/exact_overlays.png" \
      --columns 4 \
      --thumb-width 300

  run_stage "06_contact_sheet_${profile}_masks" \
    python -m scripts.task1.qa.make_contact_sheet \
      --image-dir "${profile_masks}" \
      --output "${profile_visuals}/exact_masks.png" \
      --columns 4 \
      --thumb-width 300

  profile_reports+=(
    --report "${profile_audit}/dense_cross_validation_audit.json"
  )
done

run_stage 07_summarize_global_profile_sweep \
  python -m scripts.task1.dinov3.summarize_dense_cross_validation_sweep \
    "${profile_reports[@]}" \
    --output "${AUDIT_DIR}/dense_cross_validation_sweep.json"

test -f "${CALIBRATION_DIR}/confidence_calibration.json"
test -f "${CALIBRATION_DIR}/confidence_calibration.npz"
for profile in "${PROFILES[@]}"; do
  test -f "${EVIDENCE_DIR}/${profile}/vote_manifest.json"
  test -f "${AUDIT_DIR}/${profile}/dense_cross_validation_audit.json"
  test -f "${AUDIT_DIR}/${profile}/black_core_first_candidate_mask.npy"
  test -f "${AUDIT_DIR}/${profile}/excluded_preferred_label_mask.npy"
  test -f "${AUDIT_DIR}/${profile}/dense_survivor_before_spatial_mask.npy"
  test -f "${AUDIT_DIR}/${profile}/dense_validated_fill_mask.npy"
  test -f "${AUDIT_DIR}/${profile}/candidate_gate_reason_codes.npy"
  test -f "${AUDIT_DIR}/${profile}/dense_validated_supports.npz"
  test -f "${VISUALIZATION_DIR}/${profile}/exact_overlays.png"
  test -f "${VISUALIZATION_DIR}/${profile}/exact_masks.png"
done
test -f "${AUDIT_DIR}/dense_cross_validation_sweep.json"

if find "${OUT_DIR}" -type f \( \
  -name 'gaussian_labels.npy' -o \
  -name 'gaussian_project_class_ids.npy' -o \
  -name 'label_map.json' \
\) -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote semantic label artifacts" >&2
  exit 2
fi
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

{
  echo "mode=dinov3_core_first_calibrated_dense_cross_validation_audit_v2"
  echo "report_only=1"
  echo "preferred_labels_read_only=1"
  echo "calibration_scope=joint_two_complete_cached_70_view_sources"
  echo "calibration_metric=weakest_empirical_cdf_rank"
  echo "profiles=baseline,permissive,balanced,strict"
  echo "profile_target_retention=1.00,0.50,0.25,0.10"
  echo "single_flashsplat_render_shared_across_profiles=1"
  echo "abstain_mass_preserved=1"
  echo "camera_coverage_threshold=automatic_half_joint_profile_retention"
  echo "camera_winner_share_denominator=accepted_semantic_mass"
  echo "boundary_classes=wall,ceiling,floor"
  echo "geometry_policy=semantic_filter_then_spatial_resplit"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "manual_component_decisions=0"
  echo "automatic_profile_selection=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=1"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 6 -type f | sort
