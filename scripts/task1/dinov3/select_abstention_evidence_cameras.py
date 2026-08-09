#!/usr/bin/env python3
"""Automatically select unused cameras for detected hard-vote abstentions.

The selector consumes the complete FlashSplat visibility matrix and the saved
hard-consensus diagnostics.  It targets only Gaussians that already received
at least one usable DINOv3 camera identity but were rejected as single-camera,
exact-tie, or no-strict-majority cases.  Existing semantic identities are not
read and no camera list can be supplied manually.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    STATUS_NAMES,
)
from scripts.task1.dinov3.select_visibility_cameras import (
    packed_intersection_counts,
    validate_visibility_matrix,
)


SOURCE = "dinov3_detected_abstention_additional_camera_selection"
CONTRACT = "automatic_visibility_and_pose_diverse_abstention_evidence_v1"
TARGET_COVERAGE = 0.99
TARGET_STATUSES = (STATUS_SINGLE_CAMERA, STATUS_EXACT_TIE, STATUS_NO_STRICT_MAJORITY)


def required_additional_evidence(
    total: np.ndarray,
    maximum: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    """Return the minimum extra agreeing identities needed for eligibility.

    The requirement is the larger of the two-camera minimum and the exact
    number of additional votes the current winner would need to become a
    strict majority.  It is calculated only for detected abstentions.
    """

    totals = np.asarray(total, dtype=np.int32)
    maxima = np.asarray(maximum, dtype=np.int32)
    statuses = np.asarray(status, dtype=np.uint8)
    if totals.ndim != 1 or maxima.shape != totals.shape or statuses.shape != totals.shape:
        raise ValueError("consensus diagnostic arrays must be aligned")
    if np.any(totals < 0) or np.any(maxima < 0) or np.any(maxima > totals):
        raise ValueError("consensus counts are invalid")

    target = np.isin(statuses, np.asarray(TARGET_STATUSES, dtype=np.uint8))
    two_camera_deficit = np.maximum(2 - totals, 0)
    strict_majority_deficit = np.maximum(totals - 2 * maxima + 1, 0)
    required = np.maximum(two_camera_deficit, strict_majority_deficit)
    required = np.where(target, np.maximum(required, 1), 0)
    return required.astype(np.uint16)


def camera_pose_features(cameras: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([camera["position"] for camera in cameras], dtype=np.float64)
    rotations = np.asarray([camera["rotation"] for camera in cameras], dtype=np.float64)
    if positions.shape != (len(cameras), 3) or rotations.shape != (len(cameras), 3, 3):
        raise ValueError("cameras.json has invalid pose arrays")
    directions = -rotations[:, :, 2]
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(positions).all():
        raise ValueError("cameras.json contains an invalid camera pose")
    directions /= norms[:, None]
    return positions, directions


def pose_novelty(
    candidate_rows: np.ndarray,
    reference_rows: np.ndarray,
    positions: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Measure global pose novelty for deterministic gain tie-breaking."""

    candidates = np.asarray(candidate_rows, dtype=np.int64)
    references = np.asarray(reference_rows, dtype=np.int64)
    if candidates.ndim != 1 or references.ndim != 1:
        raise ValueError("camera row arrays must be one-dimensional")
    if references.size == 0:
        return np.ones(candidates.shape, dtype=np.float64)
    extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    extent = max(extent, np.finfo(np.float64).eps)
    position_distance = np.linalg.norm(
        positions[candidates, None, :] - positions[references][None, :, :], axis=2
    ) / extent
    direction_distance = (
        1.0
        - np.clip(
            directions[candidates] @ directions[references].T,
            -1.0,
            1.0,
        )
    ) / 2.0
    return np.minimum(position_distance + direction_distance, 2.0).min(axis=1)


def greedy_abstention_camera_selection(
    packed_visibility: np.ndarray,
    *,
    gaussian_count: int,
    camera_indices: np.ndarray,
    baseline_camera_indices: np.ndarray,
    required: np.ndarray,
    cameras: list[dict[str, Any]],
    target_coverage: float = TARGET_COVERAGE,
) -> dict[str, Any]:
    matrix = np.asarray(packed_visibility, dtype=np.uint8)
    indices = np.asarray(camera_indices, dtype=np.int32)
    baseline = np.asarray(baseline_camera_indices, dtype=np.int32)
    requirement = np.asarray(required, dtype=np.uint16)
    if requirement.shape != (gaussian_count,):
        raise ValueError("required evidence must contain one value per Gaussian")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target coverage must be in (0, 1]")
    if len(cameras) != matrix.shape[0]:
        raise ValueError("cameras.json and visibility rows differ")
    global_counts = np.zeros((gaussian_count,), dtype=np.uint16)
    for row in matrix:
        global_counts += np.unpackbits(
            row, bitorder="little", count=gaussian_count
        ).astype(np.uint16, copy=False)
    validate_visibility_matrix(
        matrix,
        gaussian_count=gaussian_count,
        camera_indices=indices,
        global_view_count=global_counts,
    )
    index_to_row = {int(value): row for row, value in enumerate(indices)}
    if len(index_to_row) != indices.size or any(int(value) not in index_to_row for value in baseline):
        raise ValueError("baseline cameras do not map uniquely into visibility rows")
    baseline_rows = np.asarray([index_to_row[int(value)] for value in baseline], dtype=np.int32)
    remaining_rows = np.asarray(
        [row for row in range(matrix.shape[0]) if row not in set(baseline_rows.tolist())],
        dtype=np.int32,
    )
    if remaining_rows.size == 0:
        raise ValueError("no unused reconstruction cameras remain")

    unused_visible_count = np.zeros((gaussian_count,), dtype=np.uint16)
    for row in remaining_rows:
        unused_visible_count += np.unpackbits(
            matrix[row], bitorder="little", count=gaussian_count
        ).astype(np.uint16, copy=False)
    fulfillable = np.minimum(requirement, unused_visible_count)
    fulfillable_units = int(fulfillable.sum(dtype=np.uint64))
    if fulfillable_units == 0:
        raise ValueError("unused cameras cannot add evidence to detected abstentions")
    target_units = int(math.ceil(target_coverage * fulfillable_units))

    positions, directions = camera_pose_features(cameras)
    outstanding = fulfillable.copy()
    selected_rows: list[int] = []
    steps: list[dict[str, Any]] = []
    achieved = 0
    available = np.ones((matrix.shape[0],), dtype=bool)
    available[baseline_rows] = False

    while achieved < target_units:
        candidates = np.flatnonzero(available)
        if candidates.size == 0:
            break
        target_bits = np.packbits(outstanding > 0, bitorder="little")
        gains = packed_intersection_counts(matrix[candidates], target_bits)
        maximum_gain = int(gains.max(initial=0))
        if maximum_gain <= 0:
            break
        tied = candidates[gains == maximum_gain]
        references = np.asarray([*baseline_rows.tolist(), *selected_rows], dtype=np.int32)
        novelties = pose_novelty(tied, references, positions, directions)
        best_novelty = float(novelties.max())
        tied = tied[np.isclose(novelties, best_novelty, rtol=0.0, atol=1e-12)]
        chosen = int(tied[0])
        visible = np.unpackbits(
            matrix[chosen], bitorder="little", count=gaussian_count
        ).astype(bool, copy=False)
        newly_satisfied = visible & (outstanding > 0)
        gain = int(np.count_nonzero(newly_satisfied))
        outstanding[newly_satisfied] -= np.uint16(1)
        achieved += gain
        available[chosen] = False
        selected_rows.append(chosen)
        steps.append(
            {
                "selection_step": len(selected_rows),
                "visibility_row": chosen,
                "camera_index": int(indices[chosen]),
                "camera_id": int(cameras[chosen]["id"]),
                "new_evidence_units": gain,
                "pose_novelty_tie_break": float(
                    pose_novelty(
                        np.asarray([chosen]), references, positions, directions
                    )[0]
                ),
                "cumulative_evidence_units": achieved,
                "coverage_of_fulfillable_requirements": achieved / fulfillable_units,
            }
        )

    if achieved < target_units:
        raise RuntimeError("greedy selection could not reach its computed coverage target")
    return {
        "selected_camera_rows": np.asarray(selected_rows, dtype=np.int32),
        "selected_camera_indices": indices[np.asarray(selected_rows, dtype=np.int32)],
        "steps": steps,
        "required_units": int(requirement.sum(dtype=np.uint64)),
        "fulfillable_units": fulfillable_units,
        "target_units": target_units,
        "achieved_units": achieved,
        "target_coverage": target_coverage,
        "fulfillable_gaussian_count": int(np.count_nonzero(fulfillable)),
        "unfulfillable_gaussian_count": int(np.count_nonzero((requirement > 0) & (fulfillable == 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-dir", required=True, type=Path)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--cameras-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    visibility_report_path = args.visibility_dir / "all_camera_visibility_report.json"
    required_paths = (
        visibility_report_path,
        args.visibility_dir / "camera_visibility_bits.npy",
        args.visibility_dir / "camera_indices.npy",
        args.baseline_vote_manifest,
        args.hard_diagnostics,
        args.cameras_json,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(visibility_report_path.read_text(encoding="utf-8"))
    if report.get("contract") != "all_reconstruction_camera_nonzero_support_v1":
        raise ValueError("visibility report has the wrong contract")
    manifest = json.loads(args.baseline_vote_manifest.read_text(encoding="utf-8"))
    if manifest.get("contract") != "complete_dense_argmax_pixels_normalized_per_camera_v1":
        raise ValueError("baseline vote manifest has the wrong contract")
    cameras = json.loads(args.cameras_json.read_text(encoding="utf-8"))
    gaussian_count = int(manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(report.get("vertex_count", -1)) != gaussian_count:
        raise ValueError("visibility report and vote manifest Gaussian counts differ")
    if int(report.get("camera_count", -1)) != len(cameras):
        raise ValueError("visibility report and cameras.json camera counts differ")
    if int(report.get("max_width", -1)) != int(manifest.get("render_max_width", -2)):
        raise ValueError("visibility and semantic vote caches use different render widths")
    with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
        total = diagnostics["semantic_camera_count"]
        maximum = diagnostics["winner_camera_count"]
        status = diagnostics["consensus_status"]
    required = required_additional_evidence(total, maximum, status)
    packed = np.load(args.visibility_dir / "camera_visibility_bits.npy", mmap_mode="r")
    camera_indices = np.load(args.visibility_dir / "camera_indices.npy", allow_pickle=False)
    baseline_indices = np.asarray(
        [int(frame["camera_index"]) for frame in manifest["frames"]], dtype=np.int32
    )
    result = greedy_abstention_camera_selection(
        packed,
        gaussian_count=gaussian_count,
        camera_indices=camera_indices,
        baseline_camera_indices=baseline_indices,
        required=required,
        cameras=cameras,
    )

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "additional_camera_indices.npy", result["selected_camera_indices"])
    (args.output_dir / "additional_camera_indices.txt").write_text(
        ",".join(str(int(value)) for value in result["selected_camera_indices"]),
        encoding="utf-8",
    )
    status_requirements = {
        STATUS_NAMES[code]: {
            "gaussian_count": int(np.count_nonzero(status == code)),
            "required_evidence_units": int(required[status == code].sum(dtype=np.uint64)),
        }
        for code in TARGET_STATUSES
    }
    output = {
        "source": SOURCE,
        "contract": CONTRACT,
        "target_policy": "detected_abstentions_only",
        "evidence_requirement": "max(two_camera_deficit, strict_majority_deficit)",
        "coverage_target": TARGET_COVERAGE,
        "primary_selection_score": "new_fulfillable_evidence_units",
        "tie_break": "global_camera_pose_novelty_then_visibility_row_order",
        "baseline_camera_indices": baseline_indices.tolist(),
        "additional_camera_indices": result["selected_camera_indices"].tolist(),
        "additional_camera_count": int(result["selected_camera_indices"].size),
        "status_requirements": status_requirements,
        **{key: value for key, value in result.items() if key not in {"selected_camera_rows", "selected_camera_indices", "steps"}},
        "selection_steps": result["steps"],
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "additional_camera_selection_report.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "selection_steps"}, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
