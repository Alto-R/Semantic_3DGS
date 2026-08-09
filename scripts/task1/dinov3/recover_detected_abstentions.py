#!/usr/bin/env python3
"""Report-only recovery of detected DINOv3 hard-vote abstentions.

Existing strict-majority labels are immutable.  Additional automatically
selected cameras first receive the exact original equal-camera hard-vote
consensus.  Remaining detected abstentions may receive a diagnostic candidate
only when reliability-weighted, within-camera-mass-weighted evidence forms a
unique strict majority.  Nothing in this module writes an accepted label map
or semantic PLY.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CONTRACT as AUDIT_CONTRACT,
    SOURCE as AUDIT_SOURCE,
    STATUS_ACCEPTED,
    STATUS_EXACT_TIE,
    STATUS_NAMES,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    VOTE_CONTRACT,
    VOTE_SOURCE,
    collapse_camera_distribution,
    consensus_statistics,
    leave_one_out_consensus,
)


SOURCE = "dinov3_detected_abstention_recovery_audit"
CONTRACT = "immutable_hard_anchor_incremental_strict_then_calibrated_v1"
RECOVERY_UNRESOLVED = 0
RECOVERY_LOCKED_ANCHOR = 1
RECOVERY_ADDITIONAL_STRICT_MAJORITY = 2
RECOVERY_CALIBRATED_WEIGHTED_MAJORITY = 3
RECOVERY_NAMES = {
    RECOVERY_UNRESOLVED: "unresolved",
    RECOVERY_LOCKED_ANCHOR: "locked_original_strict_majority",
    RECOVERY_ADDITIONAL_STRICT_MAJORITY: "additional_camera_strict_majority",
    RECOVERY_CALIBRATED_WEIGHTED_MAJORITY: "calibrated_weighted_strict_majority",
}
TARGET_STATUSES = (STATUS_SINGLE_CAMERA, STATUS_EXACT_TIE, STATUS_NO_STRICT_MAJORITY)


def collapse_camera_with_mass(
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the unique camera winner and its normalized FlashSplat mass."""

    winners, summary = collapse_camera_distribution(
        indices,
        class_ids,
        weights,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    idx = np.asarray(indices, dtype=np.int64)
    classes = np.asarray(class_ids, dtype=np.uint16)
    values = np.asarray(weights, dtype=np.float32)
    mass = np.zeros((gaussian_count,), dtype=np.float32)
    matched = winners[idx] == classes
    if np.any(matched):
        mass[idx[matched]] = values[matched]
    mass[winners == 0] = 0.0
    return winners, mass, summary


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Conservative 95% lower confidence bound for a camera agreement rate."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (centre - radius) / denominator)


def unique_weighted_strict_majority(scores: np.ndarray) -> dict[str, np.ndarray]:
    """Resolve only a unique winner with more than half of weighted evidence."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("weighted scores must have classes-plus-zero x items")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weighted scores must be finite and non-negative")
    class_scores = values[1:]
    maximum = class_scores.max(axis=0)
    winner = class_scores.argmax(axis=0).astype(np.uint16) + np.uint16(1)
    tied = (class_scores == maximum[None, :]).sum(axis=0) > 1
    total = class_scores.sum(axis=0, dtype=np.float32)
    accepted = (maximum > 0.0) & ~tied & (maximum * 2.0 > total)
    prediction = np.where(accepted, winner, 0).astype(np.uint16)
    share = np.divide(maximum, total, out=np.zeros_like(maximum), where=total > 0.0)
    return {
        "prediction": prediction,
        "accepted": accepted,
        "winner": winner,
        "winner_share": share,
        "maximum": maximum,
        "total": total,
        "tied": tied,
    }


def add_hard_camera_counts(counts: np.ndarray, winners: np.ndarray) -> None:
    supported = np.flatnonzero(winners).astype(np.int64)
    if supported.size:
        counts[winners[supported].astype(np.int64), supported] += np.uint8(1)


def validate_vote_manifest(manifest: dict[str, Any], *, name: str) -> None:
    if manifest.get("source") != VOTE_SOURCE or manifest.get("contract") != VOTE_CONTRACT:
        raise ValueError(f"{name} vote manifest has the wrong contract")
    for field, expected in (
        ("query_region_filtering_used", False),
        ("confidence_threshold_used", False),
        ("one_normalized_vote_per_camera", True),
        ("v5_used", False),
        ("dinov2_used", False),
    ):
        if manifest.get(field) is not expected:
            raise ValueError(f"{name} vote manifest violates {field}={expected}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{name} vote manifest has no frames")


def load_camera_evidence(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    gaussian_count: int,
    class_count: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for frame in manifest["frames"]:
        vote_path = manifest_path.parent / str(frame["vote_file"])
        if not vote_path.is_file():
            raise FileNotFoundError(vote_path)
        with np.load(vote_path, allow_pickle=False) as data:
            winners, mass, summary = collapse_camera_with_mass(
                data["indices"],
                data["class_ids"],
                data["weights"],
                gaussian_count=gaussian_count,
                class_count=class_count,
            )
        evidence.append(
            {
                "camera_index": int(frame["camera_index"]),
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                "winners": winners,
                "mass": mass,
                "collapse_summary": summary,
            }
        )
    return evidence


def camera_reliability_rows(
    baseline_evidence: list[dict[str, Any]],
    additional_evidence: list[dict[str, Any]],
    baseline_statistics: dict[str, np.ndarray],
    locked_labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence in baseline_evidence:
        winners = evidence["winners"]
        target, _ = leave_one_out_consensus(baseline_statistics, winners)
        eligible = (target > 0) & (winners > 0)
        trials = int(np.count_nonzero(eligible))
        successes = int(np.count_nonzero(eligible & (target == winners)))
        rows.append(
            {
                "camera_index": evidence["camera_index"],
                "camera_id": evidence["camera_id"],
                "source": "baseline_leave_one_camera_out",
                "calibration_trial_count": trials,
                "calibration_agreement_count": successes,
                "observed_agreement": successes / trials if trials else 0.0,
                "reliability_weight": wilson_lower_bound(successes, trials),
            }
        )
    locked = locked_labels > 0
    for evidence in additional_evidence:
        winners = evidence["winners"]
        eligible = locked & (winners > 0)
        trials = int(np.count_nonzero(eligible))
        successes = int(np.count_nonzero(eligible & (locked_labels == winners)))
        rows.append(
            {
                "camera_index": evidence["camera_index"],
                "camera_id": evidence["camera_id"],
                "source": "additional_camera_vs_locked_anchor",
                "calibration_trial_count": trials,
                "calibration_agreement_count": successes,
                "observed_agreement": successes / trials if trials else 0.0,
                "reliability_weight": wilson_lower_bound(successes, trials),
            }
        )
    return rows


def accumulate_weighted_scores(
    evidence: list[dict[str, Any]],
    reliabilities: dict[int, float],
    gaussian_indices: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(gaussian_indices, dtype=np.int64)
    scores = np.zeros((class_count + 1, target.size), dtype=np.float32)
    camera_count = np.zeros((target.size,), dtype=np.uint16)
    columns = np.arange(target.size, dtype=np.int64)
    for item in evidence:
        winner = item["winners"][target]
        mass = item["mass"][target]
        supported = winner > 0
        if not np.any(supported):
            continue
        contribution = (
            np.float32(reliabilities[item["camera_index"]]) * mass[supported]
        )
        np.add.at(
            scores,
            (winner[supported].astype(np.int64), columns[supported]),
            contribution,
        )
        camera_count[supported] += np.uint16(1)
    return scores, camera_count


def count_by_original_status(mask: np.ndarray, status: np.ndarray) -> dict[str, int]:
    selected = np.asarray(mask, dtype=bool)
    statuses = np.asarray(status, dtype=np.uint8)
    return {
        STATUS_NAMES[code]: int(np.count_nonzero(selected & (statuses == code)))
        for code in TARGET_STATUSES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--additional-vote-manifest", required=True, type=Path)
    parser.add_argument("--hard-audit-report", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (
        args.baseline_vote_manifest,
        args.additional_vote_manifest,
        args.hard_audit_report,
        args.hard_diagnostics,
        args.selection_report,
        args.ontology,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    baseline_manifest = json.loads(args.baseline_vote_manifest.read_text(encoding="utf-8"))
    additional_manifest = json.loads(args.additional_vote_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.hard_audit_report.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    validate_vote_manifest(baseline_manifest, name="baseline")
    validate_vote_manifest(additional_manifest, name="additional")
    gaussian_count = int(baseline_manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(additional_manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifests have different Gaussian counts")
    if Path(str(baseline_manifest.get("ply_path", ""))).resolve() != Path(
        str(additional_manifest.get("ply_path", ""))
    ).resolve():
        raise ValueError("vote manifests belong to different Gaussian PLYs")
    baseline_indices = [int(frame["camera_index"]) for frame in baseline_manifest["frames"]]
    additional_indices = [int(frame["camera_index"]) for frame in additional_manifest["frames"]]
    if len(set(baseline_indices)) != len(baseline_indices) or len(set(additional_indices)) != len(additional_indices):
        raise ValueError("a vote manifest repeats a camera")
    if set(baseline_indices) & set(additional_indices):
        raise ValueError("additional vote manifest repeats a baseline camera")
    if audit.get("source") != AUDIT_SOURCE or audit.get("contract") != AUDIT_CONTRACT:
        raise ValueError("hard audit report has the wrong contract")
    if Path(str(audit.get("vote_manifest", ""))).resolve() != args.baseline_vote_manifest.resolve():
        raise ValueError("hard audit report belongs to a different baseline vote manifest")
    if int(audit.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("hard audit report has a different Gaussian count")
    if [int(value) for value in audit.get("camera_indices", [])] != baseline_indices:
        raise ValueError("hard audit report has different baseline cameras")
    if selection.get("contract") != "automatic_visibility_and_pose_diverse_abstention_evidence_v1":
        raise ValueError("additional-camera selection report has the wrong contract")
    if [int(value) for value in selection.get("baseline_camera_indices", [])] != baseline_indices:
        raise ValueError("selection report has different baseline cameras")
    if [int(value) for value in selection.get("additional_camera_indices", [])] != additional_indices:
        raise ValueError("additional votes differ from the automatic camera selection")

    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    baseline_evidence = load_camera_evidence(
        args.baseline_vote_manifest,
        baseline_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    additional_evidence = load_camera_evidence(
        args.additional_vote_manifest,
        additional_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    args.output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="abstention_recovery_", dir=args.output_dir) as temporary:
        hard_counts = np.memmap(
            Path(temporary) / "hard_counts.uint8",
            mode="w+",
            dtype=np.uint8,
            shape=(class_count + 1, gaussian_count),
        )
        hard_counts[:] = 0
        for evidence in baseline_evidence:
            add_hard_camera_counts(hard_counts, evidence["winners"])
        baseline_statistics = consensus_statistics(hard_counts, chunk_size=args.chunk_size)
        with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
            expected_total = diagnostics["semantic_camera_count"]
            expected_maximum = diagnostics["winner_camera_count"]
            expected_status = diagnostics["consensus_status"]
        for key, expected in (
            ("total", expected_total),
            ("maximum", expected_maximum),
            ("status", expected_status),
        ):
            if not np.array_equal(baseline_statistics[key], expected):
                raise RuntimeError(f"baseline hard consensus does not reproduce {key}")
        expected_status_counts = audit.get("gaussian_agreement", {}).get("status_counts")
        reproduced_counts = {
            STATUS_NAMES[code]: int(np.count_nonzero(expected_status == code))
            for code in sorted(STATUS_NAMES)
        }
        if reproduced_counts != expected_status_counts:
            raise RuntimeError("hard diagnostics do not reproduce the audit status counts")

        locked_labels = np.where(
            expected_status == STATUS_ACCEPTED,
            baseline_statistics["winner"],
            0,
        ).astype(np.uint16)
        for evidence in additional_evidence:
            add_hard_camera_counts(hard_counts, evidence["winners"])
        combined_statistics = consensus_statistics(hard_counts, chunk_size=args.chunk_size)
        hard_counts.flush()
        del hard_counts

    target_mask = np.isin(expected_status, np.asarray(TARGET_STATUSES, dtype=np.uint8))
    strict_fill = target_mask & (combined_statistics["status"] == STATUS_ACCEPTED)
    diagnostic_labels = locked_labels.copy()
    diagnostic_labels[strict_fill] = combined_statistics["winner"][strict_fill]
    source_codes = np.zeros((gaussian_count,), dtype=np.uint8)
    source_codes[expected_status == STATUS_ACCEPTED] = RECOVERY_LOCKED_ANCHOR
    source_codes[strict_fill] = RECOVERY_ADDITIONAL_STRICT_MAJORITY

    reliability_rows = camera_reliability_rows(
        baseline_evidence,
        additional_evidence,
        baseline_statistics,
        locked_labels,
    )
    reliabilities = {
        int(row["camera_index"]): float(row["reliability_weight"])
        for row in reliability_rows
    }
    remaining_indices = np.flatnonzero(target_mask & ~strict_fill).astype(np.int64)
    scores, contributing = accumulate_weighted_scores(
        [*baseline_evidence, *additional_evidence],
        reliabilities,
        remaining_indices,
        class_count=class_count,
    )
    weighted = unique_weighted_strict_majority(scores)
    weighted_local_accept = weighted["accepted"] & (contributing >= 2)
    weighted_indices = remaining_indices[weighted_local_accept]
    diagnostic_labels[weighted_indices] = weighted["prediction"][weighted_local_accept]
    source_codes[weighted_indices] = RECOVERY_CALIBRATED_WEIGHTED_MAJORITY

    anchor_indices = np.flatnonzero(locked_labels).astype(np.int64)
    if anchor_indices.size > 250_000:
        positions = np.linspace(0, anchor_indices.size - 1, 250_000, dtype=np.int64)
        anchor_sample = anchor_indices[positions]
    else:
        anchor_sample = anchor_indices
    anchor_scores, anchor_contributing = accumulate_weighted_scores(
        [*baseline_evidence, *additional_evidence],
        reliabilities,
        anchor_sample,
        class_count=class_count,
    )
    anchor_weighted = unique_weighted_strict_majority(anchor_scores)
    anchor_valid = anchor_weighted["accepted"] & (anchor_contributing >= 2)
    anchor_correct = anchor_valid & (
        anchor_weighted["prediction"] == locked_labels[anchor_sample]
    )

    np.save(
        args.output_dir / "diagnostic_candidate_project_class_ids.npy",
        diagnostic_labels,
    )
    np.save(args.output_dir / "diagnostic_recovery_source_codes.npy", source_codes)
    np.save(args.output_dir / "original_consensus_status.npy", expected_status)
    recovered = source_codes >= RECOVERY_ADDITIONAL_STRICT_MAJORITY
    strict_count = int(np.count_nonzero(source_codes == RECOVERY_ADDITIONAL_STRICT_MAJORITY))
    weighted_count = int(np.count_nonzero(source_codes == RECOVERY_CALIBRATED_WEIGHTED_MAJORITY))
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": True,
        "gaussian_count": gaussian_count,
        "baseline_camera_indices": baseline_indices,
        "additional_camera_indices": additional_indices,
        "additional_camera_selection_report": str(args.selection_report),
        "baseline_status_counts": reproduced_counts,
        "immutable_anchor_count": int(np.count_nonzero(locked_labels)),
        "immutable_anchor_labels_changed": 0,
        "detected_abstention_count": int(np.count_nonzero(target_mask)),
        "recovery": {
            "additional_strict_majority_count": strict_count,
            "additional_strict_majority_by_original_status": count_by_original_status(
                source_codes == RECOVERY_ADDITIONAL_STRICT_MAJORITY, expected_status
            ),
            "calibrated_weighted_majority_count": weighted_count,
            "calibrated_weighted_majority_by_original_status": count_by_original_status(
                source_codes == RECOVERY_CALIBRATED_WEIGHTED_MAJORITY, expected_status
            ),
            "total_recovered_count": int(np.count_nonzero(recovered)),
            "total_recovered_ratio_of_detected_abstentions": float(
                np.count_nonzero(recovered) / np.count_nonzero(target_mask)
            ),
            "remaining_detected_abstention_count": int(
                np.count_nonzero(target_mask & ~recovered)
            ),
        },
        "calibration": {
            "camera_reliability_method": "95_percent_wilson_lower_bound",
            "baseline_camera_target": "strict_majority_after_excluding_that_camera",
            "additional_camera_target": "immutable_original_strict_majority_anchors",
            "fusion_contribution": "camera_reliability_times_within_camera_winning_mass",
            "acceptance": "at_least_two_camera_winners_and_unique_weighted_strict_majority",
            "anchor_evaluation_sample_count": int(anchor_sample.size),
            "anchor_candidate_count": int(np.count_nonzero(anchor_valid)),
            "anchor_candidate_coverage": float(np.mean(anchor_valid)) if anchor_valid.size else 0.0,
            "anchor_candidate_precision": float(
                np.count_nonzero(anchor_correct) / np.count_nonzero(anchor_valid)
            ) if np.any(anchor_valid) else 0.0,
            "per_camera": reliability_rows,
        },
        "recovery_source_codes": {
            str(code): name for code, name in RECOVERY_NAMES.items()
        },
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "diagnostic_candidate_array_written": True,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "detected_abstention_recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "calibration"}, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
