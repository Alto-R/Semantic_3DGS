#!/usr/bin/env python3
"""Automatically select unused cameras that add evidence to black Gaussians.

The component-graph audits showed that most remaining black Gaussians are
camera-observed but ambiguous under the current view set.  This selector
targets every Gaussian that is still black in the recovery candidate and
chooses unused reconstruction cameras by how many new fulfillable evidence
units they add, with global pose novelty as the deterministic tie-break.  It
reuses the all-camera FlashSplat visibility matrix and the same greedy policy
as the detected-abstention selector, but excludes both baseline and already
used additional cameras.  Report-only; no labels or PLY are written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov3.recover_detected_abstentions import (
    load_camera_evidence,
    validate_vote_manifest,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    consensus_statistics,
)
from scripts.task1.dinov3.select_abstention_evidence_cameras import (
    camera_pose_features,
    pose_novelty,
)
from scripts.task1.dinov3.select_visibility_cameras import (
    packed_intersection_counts,
    validate_visibility_matrix,
)


SOURCE = "dinov3_black_evidence_additional_camera_selection"
CONTRACT = "automatic_visibility_black_evidence_selection_v1"
TARGET_COVERAGE = 0.99


def black_evidence_requirements(
    combined_total: np.ndarray,
    combined_maximum: np.ndarray,
    black_mask: np.ndarray,
) -> np.ndarray:
    """Return extra camera evidence units needed per black Gaussian.

    Each black Gaussian is required to reach at least two combined cameras and
    a strict majority for its current winner.  The requirement is the larger
    of the two-camera deficit and the strict-majority deficit, and is zero for
    non-black Gaussians.
    """

    totals = np.asarray(combined_total, dtype=np.int32)
    maxima = np.asarray(combined_maximum, dtype=np.int32)
    black = np.asarray(black_mask, dtype=bool)
    if totals.ndim != 1 or maxima.shape != totals.shape or black.shape != totals.shape:
        raise ValueError("combined counts and black mask must be aligned")
    if np.any(totals < 0) or np.any(maxima < 0) or np.any(maxima > totals):
        raise ValueError("combined counts are invalid")
    two_camera_deficit = np.maximum(2 - totals, 0)
    strict_majority_deficit = np.maximum(totals - 2 * maxima + 1, 0)
    required = np.maximum(two_camera_deficit, strict_majority_deficit)
    return np.where(black, required, 0).astype(np.uint16)


def greedy_black_evidence_selection(
    packed_visibility: np.ndarray,
    *,
    gaussian_count: int,
    camera_indices: np.ndarray,
    used_camera_indices: np.ndarray,
    required: np.ndarray,
    cameras: list[dict[str, Any]],
    target_coverage: float = TARGET_COVERAGE,
) -> dict[str, Any]:
    """Greedily select unused cameras by new fulfillable black-evidence units."""

    matrix = np.asarray(packed_visibility, dtype=np.uint8)
    indices = np.asarray(camera_indices, dtype=np.int32)
    used = np.asarray(used_camera_indices, dtype=np.int32)
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
    if len(index_to_row) != indices.size or any(
        int(value) not in index_to_row for value in used
    ):
        raise ValueError("used cameras do not map uniquely into visibility rows")
    used_rows = np.asarray([index_to_row[int(value)] for value in used], dtype=np.int32)
    if np.unique(used_rows).size != used_rows.size:
        raise ValueError("used camera rows are not unique")
    remaining_rows = np.asarray(
        [row for row in range(matrix.shape[0]) if row not in set(used_rows.tolist())],
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
        raise ValueError("unused cameras cannot add evidence to black Gaussians")
    target_units = int(math.ceil(target_coverage * fulfillable_units))

    positions, directions = camera_pose_features(cameras)
    outstanding = fulfillable.copy()
    selected_rows: list[int] = []
    steps: list[dict[str, Any]] = []
    achieved = 0
    available = np.ones((matrix.shape[0],), dtype=bool)
    available[used_rows] = False

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
        references = np.asarray([*used_rows.tolist(), *selected_rows], dtype=np.int32)
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
        "unfulfillable_gaussian_count": int(
            np.count_nonzero((requirement > 0) & (fulfillable == 0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-dir", required=True, type=Path)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--additional-vote-manifest", required=True, type=Path)
    parser.add_argument("--recovery-candidate-labels", required=True, type=Path)
    parser.add_argument("--cameras-json", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    visibility_report_path = args.visibility_dir / "all_camera_visibility_report.json"
    required_paths = (
        visibility_report_path,
        args.visibility_dir / "camera_visibility_bits.npy",
        args.visibility_dir / "camera_indices.npy",
        args.baseline_vote_manifest,
        args.additional_vote_manifest,
        args.recovery_candidate_labels,
        args.cameras_json,
        args.ontology,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    report = json.loads(visibility_report_path.read_text(encoding="utf-8"))
    if report.get("contract") != "all_reconstruction_camera_nonzero_support_v1":
        raise ValueError("visibility report has the wrong contract")
    baseline_manifest = json.loads(
        args.baseline_vote_manifest.read_text(encoding="utf-8")
    )
    additional_manifest = json.loads(
        args.additional_vote_manifest.read_text(encoding="utf-8")
    )
    validate_vote_manifest(baseline_manifest, name="baseline")
    validate_vote_manifest(additional_manifest, name="additional")
    cameras = json.loads(args.cameras_json.read_text(encoding="utf-8"))
    gaussian_count = int(baseline_manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(additional_manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifests have different Gaussian counts")
    if int(report.get("vertex_count", -1)) != gaussian_count:
        raise ValueError("visibility report and vote manifest Gaussian counts differ")
    if int(report.get("camera_count", -1)) != len(cameras):
        raise ValueError("visibility report and cameras.json camera counts differ")
    if int(report.get("max_width", -1)) != int(baseline_manifest.get("render_max_width", -2)):
        raise ValueError("visibility and semantic vote caches use different render widths")
    baseline_indices = np.asarray(
        [int(frame["camera_index"]) for frame in baseline_manifest["frames"]],
        dtype=np.int32,
    )
    additional_indices = np.asarray(
        [int(frame["camera_index"]) for frame in additional_manifest["frames"]],
        dtype=np.int32,
    )
    if len(set(baseline_indices.tolist())) != baseline_indices.size:
        raise ValueError("baseline vote manifest repeats a camera")
    if len(set(additional_indices.tolist())) != additional_indices.size:
        raise ValueError("additional vote manifest repeats a camera")
    if set(baseline_indices.tolist()) & set(additional_indices.tolist()):
        raise ValueError("additional vote manifest repeats a baseline camera")

    from scripts.task1.dinov2.dinov2_ontology import load_ontology

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
    combined_counts = np.zeros((class_count + 1, gaussian_count), dtype=np.uint8)
    for item in [*baseline_evidence, *additional_evidence]:
        supported = np.flatnonzero(item["winners"]).astype(np.int64)
        if supported.size:
            combined_counts[
                item["winners"][supported].astype(np.int64), supported
            ] += np.uint8(1)
    combined_statistics = consensus_statistics(
        combined_counts, chunk_size=args.chunk_size
    )
    candidate_labels = np.load(args.recovery_candidate_labels, allow_pickle=False)
    if candidate_labels.shape != (gaussian_count,):
        raise ValueError("recovery candidate labels have the wrong shape")
    black_mask = np.asarray(candidate_labels) == 0
    required = black_evidence_requirements(
        combined_statistics["total"],
        combined_statistics["maximum"],
        black_mask,
    )
    packed = np.load(args.visibility_dir / "camera_visibility_bits.npy", mmap_mode="r")
    camera_indices = np.load(
        args.visibility_dir / "camera_indices.npy", allow_pickle=False
    )
    used = np.concatenate([baseline_indices, additional_indices]).astype(np.int32)
    result = greedy_black_evidence_selection(
        packed,
        gaussian_count=gaussian_count,
        camera_indices=camera_indices,
        used_camera_indices=used,
        required=required,
        cameras=cameras,
    )

    args.output_dir.mkdir(parents=True)
    np.save(
        args.output_dir / "additional_camera_indices.npy",
        result["selected_camera_indices"],
    )
    (args.output_dir / "additional_camera_indices.txt").write_text(
        ",".join(str(int(value)) for value in result["selected_camera_indices"]),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "black_evidence_requirements.npz",
        black_mask=black_mask,
        combined_semantic_camera_count=combined_statistics["total"].astype(np.uint16),
        combined_winner_camera_count=combined_statistics["maximum"].astype(np.uint8),
        required_evidence_units=required,
    )
    output = {
        "source": SOURCE,
        "contract": CONTRACT,
        "target_policy": "current_black_gaussians_from_recovery_candidate",
        "evidence_requirement": "max(two_camera_deficit, strict_majority_deficit)",
        "coverage_target": TARGET_COVERAGE,
        "primary_selection_score": "new_fulfillable_evidence_units",
        "tie_break": "global_camera_pose_novelty_then_visibility_row_order",
        "baseline_camera_indices": baseline_indices.tolist(),
        "existing_additional_camera_indices": additional_indices.tolist(),
        "additional_camera_indices": result["selected_camera_indices"].tolist(),
        "additional_camera_count": int(result["selected_camera_indices"].size),
        "black_gaussian_count": int(np.count_nonzero(black_mask)),
        "black_camera_observed_count": int(
            np.count_nonzero(black_mask & (combined_statistics["total"] > 0))
        ),
        "black_zero_combined_camera_count": int(
            np.count_nonzero(black_mask & (combined_statistics["total"] == 0))
        ),
        **{
            key: value
            for key, value in result.items()
            if key not in {"selected_camera_rows", "selected_camera_indices", "steps"}
        },
        "selection_steps": result["steps"],
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "black_evidence_camera_selection_report.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "selection_steps"},
            indent=2,
        )
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
