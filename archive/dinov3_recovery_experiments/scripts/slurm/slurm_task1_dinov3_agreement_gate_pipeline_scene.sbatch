#!/usr/bin/env bash
# DINOv2 agreement-gate pipeline for one scene.
#
# Runs, in order, the gated component-graph audit, the held-out round-trip
# validation, the fill-precision audit, and the winner-disagreement
# diagnostic.  Every stage reuses cached votes and reruns no inference.
# When both scenes' round-trip output names are supplied, the paired
# agreement-gate comparator and the existing paired summaries also run.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_CACHE_OUTPUT_NAME="${SOURCE_CACHE_OUTPUT_NAME:?Set the selected DINOv3 cache output}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the completed hard-vote audit output}"
SOURCE_RECOVERY_OUTPUT_NAME="${SOURCE_RECOVERY_OUTPUT_NAME:?Set the completed recovery output}"
DINOV2_VOTE_MANIFEST="${DINOV2_VOTE_MANIFEST:-${TASK1_ROOT}/${SCENE}_dinov2_separate_abstain_allviews_v1/stages/02_flashsplat_votes/vote_manifest.json}"

COMPONENT_AUDIT_OUTPUT_NAME="${COMPONENT_AUDIT_OUTPUT_NAME:-${SCENE}_dinov3_observed_black_component_graph_audit_v3}"
ROUND_TRIP_OUTPUT_NAME="${ROUND_TRIP_OUTPUT_NAME:-${SCENE}_dinov3_observed_black_component_graph_round_trip_validation_v7}"
FILL_PRECISION_OUTPUT_NAME="${FILL_PRECISION_OUTPUT_NAME:-${SCENE}_dinov3_observed_black_fill_precision_audit_v4}"
WINNER_DISAGREEMENT_OUTPUT_NAME="${WINNER_DISAGREEMENT_OUTPUT_NAME:-${SCENE}_dinov3_observed_black_winner_disagreement_audit_v3}"
PAIRED_OUTPUT_PREFIX="${PAIRED_OUTPUT_PREFIX:-observed_black_agreement_gate_paired_v1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

SOURCE_CACHE_ROOT="${TASK1_ROOT}/${SOURCE_CACHE_OUTPUT_NAME}"
HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
RECOVERY_ROOT="${TASK1_ROOT}/${SOURCE_RECOVERY_OUTPUT_NAME}"

export DINOV2_VOTE_MANIFEST
export COMPONENT_AUDIT_OUTPUT_NAME ROUND_TRIP_OUTPUT_NAME
export FILL_PRECISION_OUTPUT_NAME WINNER_DISAGREEMENT_OUTPUT_NAME

if [[ "${SCENE}" != "playroom" && "${SCENE}" != "drjohnson" ]]; then
  echo "SCENE must be playroom or drjohnson" >&2
  exit 2
fi
if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi

for path in \
  "${SOURCE_CACHE_ROOT}/stages/01_real_camera_views/view_manifest.json" \
  "${SOURCE_CACHE_ROOT}/selection/selected_view_cache_report.json" \
  "${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json" \
  "${HARD_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json" \
  "${RECOVERY_ROOT}/stages/01_additional_flashsplat_votes/vote_manifest.json" \
  "${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/detected_abstention_recovery_report.json" \
  "${DINOV2_VOTE_MANIFEST}"; do
  test -f "${path}"
done

printf 'scene=%s\nsource_cache=%s\nhard_audit=%s\nrecovery=%s\ndinov2_vote_manifest=%s\ncomponent_audit_output=%s\nround_trip_output=%s\nfill_precision_output=%s\nwinner_disagreement_output=%s\nclass_agnostic=1\nreport_only=1\naccepted_gaussian_labels_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SOURCE_CACHE_ROOT}" "${HARD_ROOT}" "${RECOVERY_ROOT}" \
  "${DINOV2_VOTE_MANIFEST}" "${COMPONENT_AUDIT_OUTPUT_NAME}" \
  "${ROUND_TRIP_OUTPUT_NAME}" "${FILL_PRECISION_OUTPUT_NAME}" \
  "${WINNER_DISAGREEMENT_OUTPUT_NAME}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi

cd "${PROJECT_ROOT}"

run_pipeline_stage() {
  local stage_name="$1"
  shift
  echo "=== ${stage_name} ==="
  "$@"
}

run_gated_stage() {
  local stage_name="$1"
  local output_name="$2"
  shift 2
  echo "=== ${stage_name} ==="
  OUTPUT_NAME="${output_name}" "$@"
}

run_gated_stage "component graph audit (DINOv2 gated)" "${COMPONENT_AUDIT_OUTPUT_NAME}" \
  bash scripts/slurm/slurm_task1_dinov3_observed_black_component_graph_audit_scene.sbatch

run_gated_stage "component graph round-trip validation (DINOv2 gated)" "${ROUND_TRIP_OUTPUT_NAME}" \
  bash scripts/slurm/slurm_task1_dinov3_observed_black_component_graph_round_trip_validation_scene.sbatch

run_gated_stage "fill precision audit (DINOv2 gated)" "${FILL_PRECISION_OUTPUT_NAME}" \
  bash scripts/slurm/slurm_task1_dinov3_observed_black_fill_precision_audit_scene.sbatch

run_gated_stage "winner disagreement audit (DINOv2 gated)" "${WINNER_DISAGREEMENT_OUTPUT_NAME}" \
  bash scripts/slurm/slurm_task1_dinov3_observed_black_winner_disagreement_audit_scene.sbatch

if [[ -n "${PLAYROOM_ROUND_TRIP_OUTPUT_NAME:-}" && -n "${DRJOHNSON_ROUND_TRIP_OUTPUT_NAME:-}" ]]; then
  PLAYROOM_ROUND_TRIP_REPORT="${TASK1_ROOT}/${PLAYROOM_ROUND_TRIP_OUTPUT_NAME}/stages/01_component_graph_round_trip_validation/component_graph_round_trip_validation_report.json"
  DRJOHNSON_ROUND_TRIP_REPORT="${TASK1_ROOT}/${DRJOHNSON_ROUND_TRIP_OUTPUT_NAME}/stages/01_component_graph_round_trip_validation/component_graph_round_trip_validation_report.json"
  PLAYROOM_FILL_REPORT="${TASK1_ROOT}/${PLAYROOM_FILL_OUTPUT_NAME:-${PLAYROOM_ROUND_TRIP_OUTPUT_NAME}}/stages/01_fill_precision_audit/fill_precision_audit_report.json"
  DRJOHNSON_FILL_REPORT="${TASK1_ROOT}/${DRJOHNSON_FILL_OUTPUT_NAME:-${DRJOHNSON_ROUND_TRIP_OUTPUT_NAME}}/stages/01_fill_precision_audit/fill_precision_audit_report.json"
  PLAYROOM_WINNER_REPORT="${TASK1_ROOT}/${PLAYROOM_WINNER_OUTPUT_NAME:-${PLAYROOM_ROUND_TRIP_OUTPUT_NAME}}/stages/01_winner_disagreement_audit/winner_disagreement_audit_report.json"
  DRJOHNSON_WINNER_REPORT="${TASK1_ROOT}/${DRJOHNSON_WINNER_OUTPUT_NAME:-${DRJOHNSON_ROUND_TRIP_OUTPUT_NAME}}/stages/01_winner_disagreement_audit/winner_disagreement_audit_report.json"
  for path in \
    "${PLAYROOM_ROUND_TRIP_REPORT}" \
    "${DRJOHNSON_ROUND_TRIP_REPORT}" \
    "${PLAYROOM_FILL_REPORT}" \
    "${DRJOHNSON_FILL_REPORT}" \
    "${PLAYROOM_WINNER_REPORT}" \
    "${DRJOHNSON_WINNER_REPORT}"; do
    test -f "${path}"
  done

  run_pipeline_stage "paired agreement-gate decision" \
    conda run --no-capture-output -n "${GAUSSIAN_ENV:-semantic_3dgs_renderer}" \
    python -m scripts.task1.dinov3.compare_observed_black_agreement_gate_scenes \
      --playroom-round-trip-report "${PLAYROOM_ROUND_TRIP_REPORT}" \
      --drjohnson-round-trip-report "${DRJOHNSON_ROUND_TRIP_REPORT}" \
      --playroom-fill-report "${PLAYROOM_FILL_REPORT}" \
      --drjohnson-fill-report "${DRJOHNSON_FILL_REPORT}" \
      --output "${TASK1_ROOT}/${PAIRED_OUTPUT_PREFIX}_agreement_gate.json"

  run_pipeline_stage "paired round-trip comparison" \
    conda run --no-capture-output -n "${GAUSSIAN_ENV:-semantic_3dgs_renderer}" \
    python -m scripts.task1.dinov3.compare_observed_black_component_graph_round_trip_scenes \
      --playroom-report "${PLAYROOM_ROUND_TRIP_REPORT}" \
      --drjohnson-report "${DRJOHNSON_ROUND_TRIP_REPORT}" \
      --output "${TASK1_ROOT}/${PAIRED_OUTPUT_PREFIX}_round_trip.json"

  run_pipeline_stage "paired fill-precision summary" \
    conda run --no-capture-output -n "${GAUSSIAN_ENV:-semantic_3dgs_renderer}" \
    python -m scripts.task1.dinov3.compare_observed_black_fill_precision_scenes \
      --playroom-report "${PLAYROOM_FILL_REPORT}" \
      --drjohnson-report "${DRJOHNSON_FILL_REPORT}" \
      --output "${TASK1_ROOT}/${PAIRED_OUTPUT_PREFIX}_fill_precision.json"

  run_pipeline_stage "paired winner-disagreement summary" \
    conda run --no-capture-output -n "${GAUSSIAN_ENV:-semantic_3dgs_renderer}" \
    python -m scripts.task1.dinov3.compare_observed_black_winner_disagreement_scenes \
      --playroom-report "${PLAYROOM_WINNER_REPORT}" \
      --drjohnson-report "${DRJOHNSON_WINNER_REPORT}" \
      --output "${TASK1_ROOT}/${PAIRED_OUTPUT_PREFIX}_winner_disagreement.json"
else
  echo "Paired comparison skipped: set PLAYROOM_ROUND_TRIP_OUTPUT_NAME and DRJOHNSON_ROUND_TRIP_OUTPUT_NAME to run it"
fi
