#!/usr/bin/env python3
"""Rank reconstruction cameras from a measured Gaussian visibility matrix.

The selector is deterministic and report-only. It first maximizes coverage of
Gaussians that have no selected view. After all globally observable Gaussians
have one selected view, it maximizes second-view coverage for Gaussians that
are globally visible from at least two cameras. Ties preserve the source
camera-row order. No image, semantic model, prior label, or PLY is read or
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


POPCOUNT = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)
SINGLE_THRESHOLDS = (90, 95, 99, 100)
DOUBLE_THRESHOLDS = (90, 95, 99)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_popcount(values: np.ndarray) -> int:
    array = np.asarray(values, dtype=np.uint8)
    return int(POPCOUNT[array].sum(dtype=np.uint64))


def packed_intersection_popcount(left: np.ndarray, right: np.ndarray) -> int:
    left_array = np.asarray(left, dtype=np.uint8)
    right_array = np.asarray(right, dtype=np.uint8)
    if left_array.shape != right_array.shape:
        raise ValueError("packed visibility rows must have matching shapes")
    return packed_popcount(np.bitwise_and(left_array, right_array))


def packed_intersection_counts(rows: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Count target bits for every packed camera row in one vectorized pass."""

    matrix = np.asarray(rows, dtype=np.uint8)
    target_array = np.asarray(target, dtype=np.uint8)
    if matrix.ndim != 2 or target_array.shape != (matrix.shape[1],):
        raise ValueError("packed camera rows and target have incompatible shapes")
    intersections = np.bitwise_and(matrix, target_array[None, :])
    return POPCOUNT[intersections].sum(axis=1, dtype=np.uint64).astype(
        np.int64,
        copy=False,
    )


def validate_visibility_matrix(
    packed: np.ndarray,
    *,
    gaussian_count: int,
    camera_indices: np.ndarray,
    global_view_count: np.ndarray,
) -> None:
    """Validate shape, padding, row mapping, and exact per-Gaussian counts."""

    matrix = np.asarray(packed)
    indices = np.asarray(camera_indices)
    counts = np.asarray(global_view_count)
    if gaussian_count < 1:
        raise ValueError("gaussian_count must be positive")
    if matrix.ndim != 2 or matrix.dtype != np.uint8:
        raise ValueError("packed visibility must be a two-dimensional uint8 matrix")
    expected_width = (gaussian_count + 7) // 8
    if matrix.shape[1] != expected_width:
        raise ValueError("packed visibility has the wrong Gaussian width")
    if indices.shape != (matrix.shape[0],) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("camera indices must contain one integer per visibility row")
    if np.unique(indices).size != indices.size:
        raise ValueError("camera indices must be unique")
    if counts.shape != (gaussian_count,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("global view counts must contain one integer per Gaussian")
    if np.any(counts < 0) or np.any(counts > matrix.shape[0]):
        raise ValueError("global view counts are outside the valid camera range")

    remainder = gaussian_count % 8
    if remainder:
        padding_mask = np.uint8(0xFF ^ ((1 << remainder) - 1))
        if np.any(np.bitwise_and(matrix[:, -1], padding_mask)):
            raise ValueError("packed visibility has nonzero padding bits")

    derived = np.zeros((gaussian_count,), dtype=np.uint16)
    for row in matrix:
        derived += np.unpackbits(row, bitorder="little", count=gaussian_count).astype(
            np.uint16,
            copy=False,
        )
    if not np.array_equal(derived, counts.astype(np.uint16, copy=False)):
        raise ValueError("packed visibility does not match visibility_view_count.npy")


def greedy_multicover_order(
    packed: np.ndarray,
    *,
    gaussian_count: int,
    camera_indices: np.ndarray,
    global_view_count: np.ndarray,
) -> dict[str, Any]:
    """Return a deterministic full camera order and its prefix coverage curve."""

    matrix = np.asarray(packed, dtype=np.uint8)
    indices = np.asarray(camera_indices)
    global_counts = np.asarray(global_view_count)
    validate_visibility_matrix(
        matrix,
        gaussian_count=gaussian_count,
        camera_indices=indices,
        global_view_count=global_counts,
    )

    camera_count = matrix.shape[0]
    observed = global_counts > 0
    multiview_capable = global_counts >= 2
    observed_count = int(np.count_nonzero(observed))
    multiview_capable_count = int(np.count_nonzero(multiview_capable))
    selected_counts = np.zeros((gaussian_count,), dtype=np.uint16)
    uncovered_bits = np.bitwise_or.reduce(matrix, axis=0)
    remaining = np.ones((camera_count,), dtype=bool)

    order_rows: list[int] = []
    order_indices: list[int] = []
    phases: list[str] = []
    newly_single: list[int] = []
    newly_double: list[int] = []
    single_covered: list[int] = []
    double_covered: list[int] = []
    zero_remaining: list[int] = []
    exactly_one: list[int] = []

    for _ in range(camera_count):
        remaining_rows = np.flatnonzero(remaining)
        if remaining_rows.size == 0:
            break

        uncovered_count = packed_popcount(uncovered_bits)
        second_target = (selected_counts == 1) & multiview_capable
        second_target_bits = np.packbits(second_target, bitorder="little")
        second_target_count = int(np.count_nonzero(second_target))

        if uncovered_count:
            phase = "single_coverage"
        elif second_target_count:
            phase = "second_coverage"
        else:
            phase = "redundant"

        best_row = int(remaining_rows[0])
        if phase != "redundant":
            candidate_matrix = matrix[remaining_rows]
            if phase == "single_coverage":
                primary_gains = packed_intersection_counts(
                    candidate_matrix,
                    uncovered_bits,
                )
                secondary_gains = (
                    packed_intersection_counts(candidate_matrix, second_target_bits)
                    if second_target_count
                    else np.zeros_like(primary_gains)
                )
            else:
                primary_gains = packed_intersection_counts(
                    candidate_matrix,
                    second_target_bits,
                )
                secondary_gains = np.zeros_like(primary_gains)
            tied = np.flatnonzero(primary_gains == primary_gains.max())
            best_secondary = secondary_gains[tied].max()
            tied = tied[secondary_gains[tied] == best_secondary]
            best_row = int(remaining_rows[int(tied[0])])

        visible = np.unpackbits(
            matrix[best_row],
            bitorder="little",
            count=gaussian_count,
        ).astype(bool, copy=False)
        before = selected_counts[visible]
        new_single_count = int(np.count_nonzero(before == 0))
        new_double_count = int(np.count_nonzero(before == 1))
        selected_counts[visible] += np.uint16(1)
        np.bitwise_and(
            uncovered_bits,
            np.bitwise_not(matrix[best_row]),
            out=uncovered_bits,
        )
        remaining[best_row] = False

        once_count = int(np.count_nonzero(selected_counts >= 1))
        twice_count = int(np.count_nonzero(selected_counts >= 2))
        order_rows.append(best_row)
        order_indices.append(int(indices[best_row]))
        phases.append(phase)
        newly_single.append(new_single_count)
        newly_double.append(new_double_count)
        single_covered.append(once_count)
        double_covered.append(twice_count)
        zero_remaining.append(observed_count - once_count)
        exactly_one.append(int(np.count_nonzero(selected_counts == 1)))

    if not np.array_equal(selected_counts, global_counts.astype(np.uint16, copy=False)):
        raise RuntimeError("full greedy order did not reconstruct global view counts")

    return {
        "camera_rows": np.asarray(order_rows, dtype=np.int32),
        "camera_indices": np.asarray(order_indices, dtype=np.int32),
        "phases": phases,
        "newly_single_covered": np.asarray(newly_single, dtype=np.int64),
        "newly_double_covered": np.asarray(newly_double, dtype=np.int64),
        "single_covered": np.asarray(single_covered, dtype=np.int64),
        "double_covered": np.asarray(double_covered, dtype=np.int64),
        "zero_remaining": np.asarray(zero_remaining, dtype=np.int64),
        "exactly_one_selected_view": np.asarray(exactly_one, dtype=np.int64),
        "observed_gaussian_count": observed_count,
        "multiview_capable_gaussian_count": multiview_capable_count,
    }


def threshold_record(
    coverage: np.ndarray,
    *,
    denominator: int,
    percent: int,
    camera_order: np.ndarray,
) -> dict[str, Any]:
    if not 0 < percent <= 100:
        raise ValueError("percent must be between 1 and 100")
    if denominator < 1:
        return {
            "target_percent": percent,
            "required_gaussian_count": 0,
            "selected_camera_count": None,
            "camera_indices": [],
        }
    required = int(math.ceil((denominator * percent) / 100.0))
    matches = np.flatnonzero(np.asarray(coverage) >= required)
    if matches.size == 0:
        return {
            "target_percent": percent,
            "required_gaussian_count": required,
            "selected_camera_count": None,
            "camera_indices": [],
        }
    count = int(matches[0]) + 1
    return {
        "target_percent": percent,
        "required_gaussian_count": required,
        "selected_camera_count": count,
        "camera_indices": [int(value) for value in camera_order[:count]],
    }


def build_threshold_report(result: dict[str, Any]) -> dict[str, Any]:
    order = np.asarray(result["camera_indices"])
    single = np.asarray(result["single_covered"])
    double = np.asarray(result["double_covered"])
    observed_count = int(result["observed_gaussian_count"])
    capable_count = int(result["multiview_capable_gaussian_count"])
    return {
        "single_view_coverage_of_observable": {
            str(percent): threshold_record(
                single,
                denominator=observed_count,
                percent=percent,
                camera_order=order,
            )
            for percent in SINGLE_THRESHOLDS
        },
        "two_view_coverage_of_observable": {
            str(percent): threshold_record(
                double,
                denominator=observed_count,
                percent=percent,
                camera_order=order,
            )
            for percent in DOUBLE_THRESHOLDS
        },
        "two_view_coverage_of_multiview_capable": {
            str(percent): threshold_record(
                double,
                denominator=capable_count,
                percent=percent,
                camera_order=order,
            )
            for percent in DOUBLE_THRESHOLDS
        },
    }


def compact_threshold_report(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Remove long camera prefixes from the terminal-only threshold summary."""

    return {
        group_name: {
            percent: {
                "required_gaussian_count": record["required_gaussian_count"],
                "selected_camera_count": record["selected_camera_count"],
            }
            for percent, record in group.items()
        }
        for group_name, group in thresholds.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report_path = args.visibility_dir / "all_camera_visibility_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("contract") != "all_reconstruction_camera_nonzero_support_v1":
        raise ValueError("unsupported all-camera visibility contract")
    packed_meta = report.get("packed_visibility", {})
    required_paths = {
        "report": report_path,
        "packed_visibility": args.visibility_dir / str(packed_meta.get("file", "")),
        "camera_indices": args.visibility_dir / str(
            packed_meta.get("row_camera_indices_file", "")
        ),
        "view_count": args.visibility_dir / "visibility_view_count.npy",
    }
    for name, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    input_hashes_before = {
        name: sha256_file(path) for name, path in required_paths.items()
    }
    packed = np.load(required_paths["packed_visibility"], mmap_mode="r", allow_pickle=False)
    camera_indices = np.load(required_paths["camera_indices"], allow_pickle=False)
    global_view_count = np.load(required_paths["view_count"], mmap_mode="r", allow_pickle=False)
    gaussian_count = int(packed_meta.get("gaussian_count", -1))
    expected_shape = tuple(int(value) for value in packed_meta.get("shape", []))
    if packed_meta.get("dtype") != "uint8" or packed_meta.get("bit_order") != "little":
        raise ValueError("unsupported packed visibility encoding")
    if packed.shape != expected_shape:
        raise ValueError("packed visibility shape does not match its manifest")
    if int(report.get("camera_count", -1)) != packed.shape[0]:
        raise ValueError("camera count does not match packed visibility")
    if int(report.get("vertex_count", -1)) != gaussian_count:
        raise ValueError("Gaussian count does not match visibility report")

    started = time.time()
    result = greedy_multicover_order(
        packed,
        gaussian_count=gaussian_count,
        camera_indices=camera_indices,
        global_view_count=global_view_count,
    )
    thresholds = build_threshold_report(result)
    input_hashes_after = {
        name: sha256_file(path) for name, path in required_paths.items()
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("source visibility artifacts changed during selection")

    args.output_dir.mkdir(parents=True)
    order_rows = np.asarray(result["camera_rows"])
    order_indices = np.asarray(result["camera_indices"])
    np.save(args.output_dir / "greedy_camera_rows.npy", order_rows)
    np.save(args.output_dir / "greedy_camera_indices.npy", order_indices)
    np.savez_compressed(
        args.output_dir / "coverage_curve.npz",
        selected_camera_count=np.arange(1, order_indices.size + 1, dtype=np.int32),
        camera_rows=order_rows,
        camera_indices=order_indices,
        newly_single_covered=result["newly_single_covered"],
        newly_double_covered=result["newly_double_covered"],
        single_covered=result["single_covered"],
        double_covered=result["double_covered"],
        zero_remaining=result["zero_remaining"],
        exactly_one_selected_view=result["exactly_one_selected_view"],
    )

    source_frames = report.get("frames", [])
    selection_steps = []
    for step, (row, camera_index, phase) in enumerate(
        zip(order_rows, order_indices, result["phases"]),
        start=1,
    ):
        row_index = int(row)
        frame = source_frames[row_index] if row_index < len(source_frames) else {}
        single_count = int(result["single_covered"][step - 1])
        double_count = int(result["double_covered"][step - 1])
        observed_count = int(result["observed_gaussian_count"])
        capable_count = int(result["multiview_capable_gaussian_count"])
        selection_steps.append(
            {
                "selection_step": step,
                "phase": phase,
                "visibility_row": row_index,
                "camera_index": int(camera_index),
                "camera_id": frame.get("camera_id"),
                "file": frame.get("file"),
                "newly_single_covered": int(result["newly_single_covered"][step - 1]),
                "newly_double_covered": int(result["newly_double_covered"][step - 1]),
                "single_covered": single_count,
                "single_coverage_of_observable": single_count / observed_count,
                "double_covered": double_count,
                "two_view_coverage_of_observable": double_count / observed_count,
                "two_view_coverage_of_multiview_capable": (
                    double_count / capable_count if capable_count else 0.0
                ),
                "zero_remaining": int(result["zero_remaining"][step - 1]),
                "exactly_one_selected_view": int(
                    result["exactly_one_selected_view"][step - 1]
                ),
            }
        )

    selection_report = {
        "source": "flashsplat_visibility_camera_selection",
        "contract": "deterministic_greedy_visibility_multicover_v1",
        "source_visibility_dir": str(args.visibility_dir),
        "source_visibility_contract": report["contract"],
        "source_max_width": report.get("max_width"),
        "source_support_threshold": report.get("support_threshold"),
        "camera_count": int(packed.shape[0]),
        "gaussian_count": gaussian_count,
        "globally_observable_gaussian_count": int(result["observed_gaussian_count"]),
        "globally_unobservable_gaussian_count": int(
            gaussian_count - result["observed_gaussian_count"]
        ),
        "globally_single_view_gaussian_count": int(np.count_nonzero(global_view_count == 1)),
        "globally_multiview_capable_gaussian_count": int(
            result["multiview_capable_gaussian_count"]
        ),
        "selection_policy": (
            "maximize newly single-covered globally observable Gaussians; "
            "break gain ties by newly double-covered multiview-capable Gaussians; "
            "after full single coverage maximize newly double-covered Gaussians"
        ),
        "tie_break_policy": "source camera-row order",
        "semantic_inference_used": False,
        "prior_semantic_labels_used": False,
        "manual_camera_selection_used": False,
        "scene_specific_rules_used": False,
        "class_specific_rules_used": False,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "thresholds": thresholds,
        "input_sha256": input_hashes_before,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "greedy_camera_rows": "greedy_camera_rows.npy",
            "greedy_camera_indices": "greedy_camera_indices.npy",
            "coverage_curve": "coverage_curve.npz",
        },
        "selection_steps": selection_steps,
    }
    output_report = args.output_dir / "camera_selection_report.json"
    output_report.write_text(json.dumps(selection_report, indent=2), encoding="utf-8")

    compact = {
        "camera_count": selection_report["camera_count"],
        "gaussian_count": gaussian_count,
        "globally_observable_gaussian_count": selection_report[
            "globally_observable_gaussian_count"
        ],
        "globally_unobservable_gaussian_count": selection_report[
            "globally_unobservable_gaussian_count"
        ],
        "thresholds": compact_threshold_report(thresholds),
    }
    print(json.dumps(compact, indent=2))
    print(f"wrote {output_report}")


if __name__ == "__main__":
    main()
