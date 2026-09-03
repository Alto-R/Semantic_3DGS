#!/usr/bin/env python3
"""Materialize recovery labels only after the paired held-out gate passes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.materialize_hard_vote_consensus import (
    add_camera_winners,
    labels_from_statistics,
    sha256_file,
    status_counts,
    validate_inputs as validate_hard_inputs,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_ACCEPTED,
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    STATUS_UNOBSERVED,
    VOTE_CONTRACT,
    VOTE_SOURCE,
    collapse_camera_distribution,
    consensus_statistics,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


SOURCE = "dinov3_audited_abstention_recovery_materialization"
CONTRACT = "paired_two_scene_heldout_gate_recovery_materialization_v1"
RECOVERY_SOURCE_NAMES = {
    0: "unresolved",
    1: "locked_original_strict_majority",
    2: "additional_camera_strict_majority",
    3: "calibrated_weighted_strict_majority",
}
TARGET_STATUS_CODES = (
    STATUS_SINGLE_CAMERA,
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
)
GATE_METRICS = (
    "projected_ratio",
    "agreement_of_projected",
    "interior_agreement_of_projected",
    "boundary_agreement_of_projected",
)


def _require_report_flags(report: Mapping[str, Any], name: str) -> None:
    if report.get("report_only") is not True:
        raise ValueError(f"{name} is not report-only")
    for field in (
        "accepted_gaussian_labels_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if report.get(field) not in (0, False):
            raise ValueError(f"{name} violates {field}")


def validate_gate(
    gate: Mapping[str, Any], scene: str, validation: Mapping[str, Any]
) -> None:
    if gate.get("source") != "dinov3_abstention_round_trip_two_scene_gate":
        raise ValueError("paired gate has the wrong source")
    if gate.get("contract") != "paired_playroom_drjohnson_recovery_round_trip_gate_v1":
        raise ValueError("paired gate has the wrong contract")
    if gate.get("both_scenes_present") is not True or gate.get(
        "accepted_for_materialization"
    ) is not True:
        raise ValueError("paired held-out gate has not accepted materialization")
    scene_gate = gate.get("scenes", {}).get(scene)
    if not isinstance(scene_gate, dict) or scene_gate.get("passes") is not True:
        raise ValueError(f"{scene} does not pass the paired held-out gate")
    if int(scene_gate.get("candidate_recovered_count", 0)) <= 0:
        raise ValueError(f"{scene} has no recovered candidates in the gate")
    if validation.get("source") != "dinov3_detected_abstention_recovery_round_trip_validation":
        raise ValueError("validation report has the wrong source")
    if validation.get("contract") != "report_only_leave_one_camera_out_recovery_comparison_v1":
        raise ValueError("validation report has the wrong contract")
    if validation.get("scene") != scene:
        raise ValueError("validation report has the wrong scene")
    if validation.get("recovery_candidate_reproduced") is not True:
        raise ValueError("validation did not reproduce the recovery candidate")
    for field in (
        "coverage_non_regression",
        "overall_non_regression",
        "interior_non_regression",
        "boundary_non_regression",
    ):
        if validation.get(field) is not True:
            raise ValueError(f"validation violates {field}")
    if int(validation.get("candidate_recovered_count", 0)) != int(
        scene_gate["candidate_recovered_count"]
    ):
        raise ValueError("gate and validation recovered counts differ")
    for section in ("baseline_metrics", "candidate_metrics", "delta"):
        gate_values = scene_gate.get(section)
        validation_values = validation.get(section)
        if not isinstance(gate_values, dict) or not isinstance(validation_values, dict):
            raise ValueError(f"gate or validation lacks {section}")
        for metric in GATE_METRICS:
            if abs(float(gate_values[metric]) - float(validation_values[metric])) > 1e-12:
                raise ValueError(f"gate and validation differ on {section}.{metric}")
    _require_report_flags(validation, "validation report")


def validate_recovery_report(
    report: Mapping[str, Any], scene: str, gaussian_count: int
) -> None:
    if report.get("source") != "dinov3_detected_abstention_recovery_audit":
        raise ValueError("recovery report has the wrong source")
    if report.get("contract") != "immutable_hard_anchor_incremental_strict_then_calibrated_v1":
        raise ValueError("recovery report has the wrong contract")
    if report.get("scene") != scene or int(report.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("recovery report has the wrong scene or Gaussian count")
    _require_report_flags(report, "recovery report")


def build_label_map(
    scene: str, ontology_data: Any, labels: np.ndarray, **paths: Path
) -> dict[str, Any]:
    entries = [{"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}]
    for project_id in np.unique(labels[labels > 0]):
        item = ontology_data.by_project_id[int(project_id)]
        entries.append(
            {
                "id": int(project_id),
                "name": item.project_class,
                "class": item.project_class,
                "project_id": item.project_id,
                "ade_id": item.ade_id,
                "type": item.kind,
            }
        )
    return {
        "scene": scene,
        "source": SOURCE,
        "contract": CONTRACT,
        **{key: str(value) for key, value in paths.items()},
        "labels": entries,
    }


def validate_candidate_arrays(
    candidate_labels: np.ndarray,
    source_codes: np.ndarray,
    status: np.ndarray,
    hard_labels: np.ndarray,
    *,
    expected_recovered_count: int,
) -> int:
    """Validate every label/source transition against the original consensus."""

    anchor_mask = status == STATUS_ACCEPTED
    if np.any(candidate_labels[anchor_mask] != hard_labels[anchor_mask]):
        raise ValueError("recovery candidate changed an immutable hard anchor")
    valid_codes = np.isin(
        source_codes,
        np.asarray(list(RECOVERY_SOURCE_NAMES), dtype=source_codes.dtype),
    )
    if not np.all(valid_codes):
        raise ValueError("recovery source codes contain an unknown value")
    if np.any(source_codes[anchor_mask] != 1):
        raise ValueError("accepted hard anchors are not locked")
    if np.any((source_codes == 1) & ~anchor_mask):
        raise ValueError("locked-anchor source code appears outside accepted anchors")
    if np.any(source_codes[status == STATUS_UNOBSERVED] != 0):
        raise ValueError("zero-camera Gaussians were materialized")
    if np.any(
        (source_codes >= 2)
        & ~np.isin(status, np.asarray(TARGET_STATUS_CODES, dtype=np.uint8))
    ):
        raise ValueError("recovery labels were assigned outside detected abstentions")
    if np.any((source_codes == 0) & (candidate_labels != 0)):
        raise ValueError("unresolved recovery entries have nonzero labels")
    if np.any((source_codes > 0) & (candidate_labels == 0)):
        raise ValueError("materialized recovery source has a zero label")
    recovered_count = int(np.count_nonzero(source_codes >= 2))
    if recovered_count != expected_recovered_count:
        raise ValueError("materialized recovery count differs from audited reports")
    return recovered_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--hard-audit-report", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--candidate-labels", required=True, type=Path)
    parser.add_argument("--recovery-source-codes", required=True, type=Path)
    parser.add_argument("--two-scene-gate", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required = (
        args.source_ply,
        args.hard_audit_report,
        args.hard_diagnostics,
        args.baseline_vote_manifest,
        args.recovery_report,
        args.validation_report,
        args.candidate_labels,
        args.recovery_source_codes,
        args.two_scene_gate,
        args.ontology,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    audit = json.loads(args.hard_audit_report.read_text(encoding="utf-8"))
    with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
        expected_status = np.asarray(
            diagnostics["consensus_status"], dtype=np.uint8
        ).copy()
    vote_manifest = json.loads(
        args.baseline_vote_manifest.read_text(encoding="utf-8")
    )
    recovery = json.loads(args.recovery_report.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    gate = json.loads(args.two_scene_gate.read_text(encoding="utf-8"))
    validate_gate(gate, args.scene, validation)
    validate_recovery_report(
        recovery, args.scene, int(vote_manifest.get("gaussian_count", -1))
    )
    if Path(str(validation.get("baseline_audit_report", ""))).resolve() != (
        args.hard_audit_report.resolve()
    ):
        raise ValueError("validation belongs to a different hard audit")
    if Path(str(validation.get("recovery_report", ""))).resolve() != (
        args.recovery_report.resolve()
    ):
        raise ValueError("validation belongs to a different recovery report")

    candidate_labels = np.load(args.candidate_labels, allow_pickle=False)
    source_codes = np.load(args.recovery_source_codes, allow_pickle=False)
    gaussian_count = int(vote_manifest.get("gaussian_count", -1))
    if candidate_labels.shape != (gaussian_count,) or source_codes.shape != (
        gaussian_count,
    ):
        raise ValueError("recovery arrays have the wrong shape")
    if candidate_labels.dtype.kind not in "ui" or source_codes.dtype.kind not in "ui":
        raise ValueError("recovery arrays must be integer arrays")
    if sha256_file(args.candidate_labels) != validation.get("recovery_candidate_sha256"):
        raise ValueError("candidate labels changed after held-out validation")
    if sha256_file(args.recovery_source_codes) != validation.get(
        "recovery_source_codes_sha256"
    ):
        raise ValueError("recovery source codes changed after held-out validation")

    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    if np.any(candidate_labels > class_count):
        raise ValueError("candidate label exceeds ontology class count")
    header = read_ply_header(args.source_ply)
    vertex = header.element("vertex")
    if vertex is None or vertex.count != gaussian_count:
        raise ValueError("source PLY vertex count differs from recovery arrays")
    if int(audit.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("hard audit and vote manifest Gaussian counts differ")
    validate_hard_inputs(
        audit,
        vote_manifest,
        args.hard_audit_report,
        args.baseline_vote_manifest,
        args.ontology,
        args.source_ply,
    )
    # Older hard audits do not record diagnostics.  When an explicit hash is
    # present, enforce it; otherwise the diagnostics are checked by exact
    # status-array reproduction below.
    recorded = audit.get("input_sha256", {})
    diagnostic_hashes = [
        str(value)
        for path, value in recorded.items()
        if Path(str(path)).resolve() == args.hard_diagnostics.resolve()
    ]
    if diagnostic_hashes and diagnostic_hashes != [
        sha256_file(args.hard_diagnostics)
    ]:
        raise ValueError("hard diagnostics changed after audit")
    if vote_manifest.get("source") != VOTE_SOURCE or vote_manifest.get(
        "contract"
    ) != VOTE_CONTRACT:
        raise ValueError("baseline vote manifest has the wrong contract")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(
        prefix="recovery_materialize_", dir=args.output_dir
    ) as temporary:
        counts = np.memmap(
            Path(temporary) / "camera_counts.uint8",
            mode="w+",
            dtype=np.uint8,
            shape=(class_count + 1, gaussian_count),
        )
        counts[:] = 0
        for frame in vote_manifest["frames"]:
            vote_path = args.baseline_vote_manifest.parent / str(frame["vote_file"])
            if not vote_path.is_file():
                raise FileNotFoundError(vote_path)
            with np.load(vote_path, allow_pickle=False) as data:
                winners, _ = collapse_camera_distribution(
                    data["indices"],
                    data["class_ids"],
                    data["weights"],
                    gaussian_count=gaussian_count,
                    class_count=class_count,
                )
            add_camera_winners(counts, winners)
        counts.flush()
        statistics = consensus_statistics(counts, chunk_size=args.chunk_size)
        del counts

    if not np.array_equal(statistics["status"], expected_status):
        raise ValueError("hard consensus status differs from the audited diagnostics")
    if status_counts(statistics["status"]) != audit["gaussian_agreement"][
        "status_counts"
    ]:
        raise ValueError("hard consensus status counts differ from the audit")
    hard_labels = labels_from_statistics(statistics)
    expected_recovered_count = int(recovery["recovery"]["total_recovered_count"])
    if expected_recovered_count != int(validation["candidate_recovered_count"]):
        raise ValueError("recovery and validation recovered counts differ")
    recovered_count = validate_candidate_arrays(
        candidate_labels,
        source_codes,
        statistics["status"],
        hard_labels,
        expected_recovered_count=expected_recovered_count,
    )

    np.save(args.output_dir / "gaussian_labels.npy", candidate_labels.astype(np.int32))
    np.save(args.output_dir / "gaussian_project_class_ids.npy", candidate_labels.astype(np.int32))
    np.save(args.output_dir / "recovery_source_codes.npy", source_codes.astype(np.uint8))
    np.save(args.output_dir / "original_consensus_status.npy", statistics["status"])
    np.save(args.output_dir / "semantic_camera_count.npy", statistics["total"])
    np.save(args.output_dir / "winner_camera_count.npy", statistics["maximum"])
    np.save(args.output_dir / "winner_share.npy", statistics["winner_share"].astype(np.float32))
    np.save(args.output_dir / "abstain_reason_codes.npy", statistics["status"])

    label_map = build_label_map(
        args.scene,
        ontology,
        candidate_labels,
        hard_audit_report=args.hard_audit_report,
        recovery_report=args.recovery_report,
        validation_report=args.validation_report,
        two_scene_gate=args.two_scene_gate,
        ontology=args.ontology,
    )
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    partial_ply = semantic_ply.with_suffix(".ply.partial")
    write_ply_with_labels(
        args.source_ply, partial_ply, candidate_labels.astype(np.int32)
    )
    partial_ply.replace(semantic_ply)

    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(
            *np.unique(candidate_labels[candidate_labels > 0], return_counts=True)
        )
    }
    code_counts = {
        RECOVERY_SOURCE_NAMES[code]: int(np.count_nonzero(source_codes == code))
        for code in RECOVERY_SOURCE_NAMES
    }
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "hard_audit_report": str(args.hard_audit_report),
        "hard_audit_report_sha256": sha256_file(args.hard_audit_report),
        "recovery_report": str(args.recovery_report),
        "recovery_report_sha256": sha256_file(args.recovery_report),
        "validation_report": str(args.validation_report),
        "validation_report_sha256": sha256_file(args.validation_report),
        "two_scene_gate": str(args.two_scene_gate),
        "two_scene_gate_sha256": sha256_file(args.two_scene_gate),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "gaussian_count": gaussian_count,
        "hard_status_counts_reproduced": True,
        "recovery_candidate_reproduced_before_materialization": True,
        "recovery_source_counts": code_counts,
        "accepted_gaussian_count": int(np.count_nonzero(candidate_labels)),
        "accepted_gaussian_ratio": float(np.count_nonzero(candidate_labels) / gaussian_count),
        "unlabeled_gaussian_count": int(np.count_nonzero(candidate_labels == 0)),
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_rules": False,
        "zero_camera_fill_used": False,
        "propagation_used": False,
        "semantic_labels_written": True,
        "label_map_written": True,
        "semantic_ply_written": True,
    }
    (args.output_dir / "abstention_recovery_materialization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
