#!/usr/bin/env python3
"""Automatically select projection-safe, low-coverage 3DGS camera views."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def unique_indices(frames: Iterable[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for frame in frames:
        index = int(frame["camera_index"])
        if index not in seen:
            indices.append(index)
            seen.add(index)
    return indices


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_cameras(model_path: Path) -> list[dict[str, Any]]:
    path = model_path / "cameras.json"
    if not path.exists():
        raise FileNotFoundError(path)
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def validate_indices(indices: Iterable[int], camera_count: int, label: str) -> list[int]:
    result = list(indices)
    for index in result:
        if index < 0 or index >= camera_count:
            raise IndexError(f"{label} camera index {index} is outside 0..{camera_count - 1}")
    return result


def camera_position(camera: dict[str, Any]) -> np.ndarray:
    return np.asarray(camera["position"], dtype=np.float64)


def camera_forward(camera: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    forward = rotation[:, 2]
    norm = float(np.linalg.norm(forward))
    return forward / max(norm, 1.0e-12)


def position_scale(cameras: list[dict[str, Any]]) -> float:
    positions = np.stack([camera_position(camera) for camera in cameras], axis=0)
    center = np.median(positions, axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    nonzero = radii[radii > 1.0e-9]
    if nonzero.size == 0:
        return 1.0
    return max(float(np.median(nonzero)), 1.0e-6)


def pose_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    scene_position_scale: float,
    orientation_weight: float = 0.5,
) -> float:
    position_delta = float(np.linalg.norm(camera_position(left) - camera_position(right)))
    position_term = position_delta / max(scene_position_scale, 1.0e-6)
    dot = float(np.clip(np.dot(camera_forward(left), camera_forward(right)), -1.0, 1.0))
    angle_term = math.acos(dot) / math.pi
    return math.sqrt(position_term * position_term + (orientation_weight * angle_term) ** 2)


def min_pose_distance(
    camera_index: int,
    reference_indices: list[int],
    cameras: list[dict[str, Any]],
    scene_position_scale: float,
) -> float:
    if not reference_indices:
        return 1.0
    return min(
        pose_distance(cameras[camera_index], cameras[index], scene_position_scale)
        for index in reference_indices
    )


def sample_scene_geometry(ply_path: Path, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    if sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    ply = PlyData.read(ply_path, mmap=True)
    vertex = ply["vertex"]
    count = int(vertex.count)
    if count <= 0:
        raise ValueError(f"No vertices found in {ply_path}")
    if sample_count >= count:
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.linspace(0, count - 1, sample_count, dtype=np.int64)
    points = np.stack(
        [np.asarray(vertex[name])[indices] for name in ("x", "y", "z")], axis=1
    ).astype(np.float64)
    log_scales = np.stack(
        [np.asarray(vertex[name])[indices] for name in ("scale_0", "scale_1", "scale_2")], axis=1
    ).astype(np.float64)
    return points, np.exp(log_scales).max(axis=1)


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def camera_to_world(camera: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(camera["rotation"], dtype=np.float64)
    transform[:3, 3] = camera_position(camera)
    return transform


def projection_metrics(
    camera: dict[str, Any],
    points: np.ndarray,
    max_scale: np.ndarray,
) -> dict[str, float]:
    world_to_camera = np.linalg.inv(camera_to_world(camera))
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera_points[:, 2]
    in_front = z > 0.01
    front_ratio = float(in_front.mean())
    if not np.any(in_front):
        return {
            "front_ratio": front_ratio,
            "inside_ratio": 0.0,
            "z_q01": 0.0,
            "scale_over_z_q99": 1.0e30,
            "scale_over_z_max": 1.0e30,
        }

    width = int(camera["width"])
    height = int(camera["height"])
    tan_fov_x = math.tan(focal2fov(float(camera["fx"]), width) * 0.5)
    tan_fov_y = math.tan(focal2fov(float(camera["fy"]), height) * 0.5)
    front_points = camera_points[in_front]
    front_z = front_points[:, 2]
    projected_x = front_points[:, 0] / front_z / tan_fov_x
    projected_y = front_points[:, 1] / front_z / tan_fov_y
    inside = (np.abs(projected_x) <= 1.0) & (np.abs(projected_y) <= 1.0)
    scale_over_z = max_scale[in_front] / np.maximum(front_z, 1.0e-6)
    return {
        "front_ratio": front_ratio,
        "inside_ratio": float(inside.mean()),
        "z_q01": float(np.quantile(front_z, 0.01)),
        "scale_over_z_q99": float(np.quantile(scale_over_z, 0.99)),
        "scale_over_z_max": float(np.max(scale_over_z)),
    }


def projection_thresholds(
    baseline_metrics: list[dict[str, float]],
    margin: float,
) -> dict[str, float]:
    if not baseline_metrics:
        raise ValueError("No baseline projection metrics")
    margin = max(float(margin), 1.0)
    return {
        "min_front_ratio": min(item["front_ratio"] for item in baseline_metrics) / margin,
        "min_inside_ratio": min(item["inside_ratio"] for item in baseline_metrics) / margin,
        "min_z_q01": min(item["z_q01"] for item in baseline_metrics) / margin,
        "max_scale_over_z_q99": max(item["scale_over_z_q99"] for item in baseline_metrics) * margin,
        "max_scale_over_z_max": max(item["scale_over_z_max"] for item in baseline_metrics) * margin,
    }


def projection_rejection_reasons(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if metrics["front_ratio"] < thresholds["min_front_ratio"]:
        reasons.append("front_ratio")
    if metrics["inside_ratio"] < thresholds["min_inside_ratio"]:
        reasons.append("inside_ratio")
    if metrics["z_q01"] < thresholds["min_z_q01"]:
        reasons.append("z_q01")
    if metrics["scale_over_z_q99"] > thresholds["max_scale_over_z_q99"]:
        reasons.append("scale_over_z_q99")
    if metrics["scale_over_z_max"] > thresholds["max_scale_over_z_max"]:
        reasons.append("scale_over_z_max")
    return reasons


def diverse_camera_pool(
    candidate_indices: list[int],
    baseline_indices: list[int],
    cameras: list[dict[str, Any]],
    count: int,
) -> list[int]:
    if count <= 0 or count >= len(candidate_indices):
        count = len(candidate_indices)
    scale = position_scale(cameras)
    selected: list[int] = []
    remaining = set(candidate_indices)
    references = list(baseline_indices)
    while remaining and len(selected) < count:
        ranked = [
            (min_pose_distance(index, references, cameras, scale), -index, index)
            for index in remaining
        ]
        _distance, _tie_break, chosen = max(ranked)
        selected.append(chosen)
        references.append(chosen)
        remaining.remove(chosen)
    return selected


def normalized(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def select_low_coverage_diverse_views(
    candidate_ratios: dict[int, float],
    baseline_indices: list[int],
    cameras: list[dict[str, Any]],
    additional_count: int,
    low_coverage_quantile: float,
    coverage_weight: float,
    novelty_weight: float,
) -> tuple[list[int], list[dict[str, float | int]]]:
    if additional_count <= 0:
        raise ValueError("--additional-count must be positive")
    if not candidate_ratios:
        raise ValueError("No candidate coverage ratios")
    if additional_count > len(candidate_ratios):
        raise ValueError(
            f"Requested {additional_count} additional cameras but only "
            f"{len(candidate_ratios)} candidates have coverage measurements"
        )

    count = additional_count
    ratio_values = np.asarray(list(candidate_ratios.values()), dtype=np.float64)
    quantile = float(np.clip(low_coverage_quantile, 0.0, 1.0))
    threshold = float(np.quantile(ratio_values, quantile))
    eligible = [index for index, ratio in candidate_ratios.items() if ratio <= threshold]
    if len(eligible) < count:
        remaining_by_coverage = sorted(
            (index for index in candidate_ratios if index not in eligible),
            key=lambda index: (candidate_ratios[index], index),
        )
        eligible.extend(remaining_by_coverage[: count - len(eligible)])

    coverage_weight = max(0.0, coverage_weight)
    novelty_weight = max(0.0, novelty_weight)
    weight_total = coverage_weight + novelty_weight
    if weight_total <= 0:
        raise ValueError("Coverage and novelty weights cannot both be zero")
    coverage_weight /= weight_total
    novelty_weight /= weight_total

    scale = position_scale(cameras)
    selected: list[int] = []
    records: list[dict[str, float | int]] = []
    remaining = set(eligible)
    references = list(baseline_indices)
    while remaining and len(selected) < count:
        needs = {index: 1.0 - candidate_ratios[index] for index in remaining}
        novelties = {
            index: min_pose_distance(index, references, cameras, scale)
            for index in remaining
        }
        normalized_needs = normalized(needs)
        normalized_novelties = normalized(novelties)
        scores = {
            index: coverage_weight * normalized_needs[index]
            + novelty_weight * normalized_novelties[index]
            for index in remaining
        }
        chosen = max(
            remaining,
            key=lambda index: (scores[index], -candidate_ratios[index], -index),
        )
        selected.append(chosen)
        references.append(chosen)
        remaining.remove(chosen)
        records.append(
            {
                "camera_index": chosen,
                "overlay_changed_pixel_ratio": candidate_ratios[chosen],
                "coverage_need": needs[chosen],
                "pose_novelty": novelties[chosen],
                "selection_score": scores[chosen],
            }
        )
    return selected, records


def write_indices(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(index) for index in indices) + "\n", encoding="utf-8")


def seed_command(args: argparse.Namespace) -> None:
    source_manifest = load_json(args.source_manifest)
    source_frames = source_manifest.get("frames", [])
    if not isinstance(source_frames, list) or not source_frames:
        raise ValueError(f"No frames found in {args.source_manifest}")
    unique_frames: list[dict[str, Any]] = []
    seen: set[int] = set()
    for frame in source_frames:
        camera_index = int(frame["camera_index"])
        if camera_index not in seen:
            unique_frames.append(frame)
            seen.add(camera_index)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.count >= len(unique_frames):
        selected_positions = list(range(len(unique_frames)))
    else:
        selected_positions = sorted(
            {int(round(value)) for value in np.linspace(0, len(unique_frames) - 1, args.count)}
        )
    selected_frames = [unique_frames[position] for position in selected_positions]
    selected_indices = [int(frame["camera_index"]) for frame in selected_frames]
    report = {
        "source": "evenly_spaced_manifest_seed",
        "source_manifest": str(args.source_manifest),
        "requested_count": args.count,
        "camera_count": len(selected_frames),
        "selected_camera_indices": selected_indices,
        "frames": selected_frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_indices(args.indices_output, selected_indices)
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))


def screen_command(args: argparse.Namespace) -> None:
    cameras = read_cameras(args.model_path)
    baseline_manifest = load_json(args.baseline_manifest)
    baseline_indices = validate_indices(
        unique_indices(baseline_manifest.get("frames", [])), len(cameras), "Baseline"
    )
    if not baseline_indices:
        raise ValueError(f"No baseline camera indices in {args.baseline_manifest}")
    if args.allowed_manifest is not None:
        allowed_manifest = load_json(args.allowed_manifest)
        allowed_indices = validate_indices(
            unique_indices(allowed_manifest.get("frames", [])),
            len(cameras),
            "Allowed",
        )
        if not allowed_indices:
            raise ValueError(f"No allowed camera indices in {args.allowed_manifest}")
    else:
        allowed_indices = list(range(len(cameras)))

    ply_path = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    points, max_scale = sample_scene_geometry(ply_path, args.sample_count)
    metrics_by_index = {
        index: projection_metrics(camera, points, max_scale)
        for index, camera in enumerate(cameras)
    }
    thresholds = projection_thresholds(
        [metrics_by_index[index] for index in baseline_indices], args.projection_margin
    )
    baseline_set = set(baseline_indices)
    candidate_records: list[dict[str, Any]] = []
    safe_indices: list[int] = []
    for index in allowed_indices:
        camera = cameras[index]
        if index in baseline_set:
            continue
        reasons = projection_rejection_reasons(metrics_by_index[index], thresholds)
        candidate_records.append(
            {
                "camera_index": index,
                "camera_id": int(camera["id"]),
                "image_name": camera.get("img_name", ""),
                "projection_metrics": metrics_by_index[index],
                "rejection_reasons": reasons,
            }
        )
        if not reasons:
            safe_indices.append(index)

    selected_pool = diverse_camera_pool(
        safe_indices, baseline_indices, cameras, args.candidate_count
    )
    report = {
        "method": "baseline_anchored_projection_screen_and_pose_diversity",
        "model_path": str(args.model_path),
        "baseline_manifest": str(args.baseline_manifest),
        "allowed_manifest": str(args.allowed_manifest) if args.allowed_manifest else None,
        "allowed_camera_count": len(allowed_indices),
        "point_cloud": str(ply_path),
        "sample_count": int(points.shape[0]),
        "projection_margin": max(float(args.projection_margin), 1.0),
        "projection_thresholds": thresholds,
        "baseline_camera_indices": baseline_indices,
        "safe_candidate_count": len(safe_indices),
        "selected_candidate_count": len(selected_pool),
        "selected_candidate_indices": selected_pool,
        "candidates": candidate_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_indices(args.indices_output, selected_pool)
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


def coverage_by_camera(
    candidate_manifest: dict[str, Any],
    coverage_report: dict[str, Any],
) -> dict[int, float]:
    camera_by_file = {
        str(frame["file"]): int(frame["camera_index"])
        for frame in candidate_manifest.get("frames", [])
    }
    ratios: dict[int, float] = {}
    for frame in coverage_report.get("frames", []):
        filename = str(frame["file"])
        if filename not in camera_by_file:
            raise ValueError(f"Coverage frame {filename} is absent from the candidate manifest")
        ratios[camera_by_file[filename]] = float(frame["overlay_changed_pixel_ratio"])
    if len(ratios) != len(camera_by_file):
        missing = sorted(set(camera_by_file.values()) - set(ratios))
        raise ValueError(f"Missing coverage for candidate camera indices: {missing}")
    return ratios


def select_command(args: argparse.Namespace) -> None:
    cameras = read_cameras(args.model_path)
    baseline_manifest = load_json(args.baseline_manifest)
    candidate_manifest = load_json(args.candidate_manifest)
    coverage_report = load_json(args.coverage_report)
    baseline_indices = validate_indices(
        unique_indices(baseline_manifest.get("frames", [])), len(cameras), "Baseline"
    )
    candidate_ratios = coverage_by_camera(candidate_manifest, coverage_report)
    validate_indices(candidate_ratios, len(cameras), "Candidate")
    for index in baseline_indices:
        candidate_ratios.pop(index, None)

    selected, records = select_low_coverage_diverse_views(
        candidate_ratios,
        baseline_indices,
        cameras,
        args.additional_count,
        args.low_coverage_quantile,
        args.coverage_weight,
        args.novelty_weight,
    )
    final_indices = [*baseline_indices, *selected]
    report = {
        "method": "low_overlay_coverage_with_iterative_pose_diversity",
        "model_path": str(args.model_path),
        "baseline_manifest": str(args.baseline_manifest),
        "candidate_manifest": str(args.candidate_manifest),
        "coverage_report": str(args.coverage_report),
        "low_coverage_quantile": float(np.clip(args.low_coverage_quantile, 0.0, 1.0)),
        "coverage_weight": args.coverage_weight,
        "novelty_weight": args.novelty_weight,
        "baseline_camera_count": len(baseline_indices),
        "candidate_camera_count": len(candidate_ratios),
        "additional_camera_count": len(selected),
        "final_camera_count": len(final_indices),
        "baseline_camera_indices": baseline_indices,
        "selected_additional_cameras": records,
        "final_camera_indices": final_indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_indices(args.indices_output, final_indices)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser(
        "seed",
        help="Select evenly spaced seed cameras from an existing rendered-view manifest",
    )
    seed.add_argument("--source-manifest", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)
    seed.add_argument("--indices-output", required=True, type=Path)
    seed.add_argument("--count", default=50, type=int)
    seed.set_defaults(handler=seed_command)

    screen = subparsers.add_parser("screen", help="Screen and diversify unused candidate cameras")
    screen.add_argument("--model-path", required=True, type=Path)
    screen.add_argument("--baseline-manifest", required=True, type=Path)
    screen.add_argument("--allowed-manifest", type=Path)
    screen.add_argument("--output", required=True, type=Path)
    screen.add_argument("--indices-output", required=True, type=Path)
    screen.add_argument("--iteration", default=30000, type=int)
    screen.add_argument("--sample-count", default=50_000, type=int)
    screen.add_argument("--candidate-count", default=100, type=int)
    screen.add_argument("--projection-margin", default=1.5, type=float)
    screen.set_defaults(handler=screen_command)

    select = subparsers.add_parser("select", help="Select low-coverage, pose-diverse additions")
    select.add_argument("--model-path", required=True, type=Path)
    select.add_argument("--baseline-manifest", required=True, type=Path)
    select.add_argument("--candidate-manifest", required=True, type=Path)
    select.add_argument("--coverage-report", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--indices-output", required=True, type=Path)
    select.add_argument("--additional-count", default=20, type=int)
    select.add_argument("--low-coverage-quantile", default=0.35, type=float)
    select.add_argument("--coverage-weight", default=0.8, type=float)
    select.add_argument("--novelty-weight", default=0.2, type=float)
    select.set_defaults(handler=select_command)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
