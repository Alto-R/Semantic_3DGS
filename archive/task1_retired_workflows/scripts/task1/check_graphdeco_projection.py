#!/usr/bin/env python3
"""Check camera projection sanity for a GraphDeco pretrained model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
from plyfile import PlyData


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def quantiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    qs = np.quantile(values, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "q01": float(qs[1]),
        "q05": float(qs[2]),
        "median": float(qs[3]),
        "q95": float(qs[4]),
        "q99": float(qs[5]),
        "max": float(qs[6]),
    }


def sample_indices(total: int, count: int) -> np.ndarray:
    if count >= total:
        return np.arange(total)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def transform_points(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def report_variant(name: str, points: np.ndarray, w2c: np.ndarray, tanfovx: float, tanfovy: float) -> None:
    camera_points = transform_points(points, w2c)
    z = camera_points[:, 2]
    in_front = z > 0.01
    projected_x = np.full_like(z, np.nan, dtype=np.float64)
    projected_y = np.full_like(z, np.nan, dtype=np.float64)
    projected_x[in_front] = camera_points[in_front, 0] / z[in_front] / tanfovx
    projected_y[in_front] = camera_points[in_front, 1] / z[in_front] / tanfovy
    inside = in_front & (np.abs(projected_x) <= 1.0) & (np.abs(projected_y) <= 1.0)

    print(f"variant={name}")
    print(f"  in_front={int(in_front.sum())}/{len(points)} ({in_front.mean():.4f})")
    print(f"  inside_frustum={int(inside.sum())}/{len(points)} ({inside.mean():.4f})")
    print(f"  z={json.dumps(quantiles(z), sort_keys=True)}")
    print(f"  ndc_x={json.dumps(quantiles(projected_x[in_front]), sort_keys=True)}")
    print(f"  ndc_y={json.dumps(quantiles(projected_y[in_front]), sort_keys=True)}")


def c2w_to_w2c(camera: Dict[str, object]) -> np.ndarray:
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    position = np.asarray(camera["position"], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position
    return np.linalg.inv(c2w)


def scan_cameras(cameras: list, points: np.ndarray, max_scale: np.ndarray, limit: int) -> None:
    rows = []
    for camera in cameras:
        fovx = focal2fov(float(camera["fx"]), int(camera["width"]))
        fovy = focal2fov(float(camera["fy"]), int(camera["height"]))
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        camera_points = transform_points(points, c2w_to_w2c(camera))
        z = camera_points[:, 2]
        in_front = z > 0.01
        if not np.any(in_front):
            continue
        projected_x = camera_points[in_front, 0] / z[in_front] / tanfovx
        projected_y = camera_points[in_front, 1] / z[in_front] / tanfovy
        inside = (np.abs(projected_x) <= 1.0) & (np.abs(projected_y) <= 1.0)
        z_front = z[in_front]
        scale_ratio = max_scale[in_front] / np.maximum(z_front, 1.0e-6)
        rows.append(
            {
                "id": int(camera["id"]),
                "name": camera.get("img_name", ""),
                "front": float(in_front.mean()),
                "inside": float(inside.mean()),
                "z_q01": float(np.quantile(z_front, 0.01)),
                "z_q05": float(np.quantile(z_front, 0.05)),
                "scale_over_z_q99": float(np.quantile(scale_ratio, 0.99)),
                "scale_over_z_max": float(np.max(scale_ratio)),
            }
        )

    rows.sort(key=lambda row: (row["scale_over_z_q99"], -row["inside"]))
    print("camera_scan=json_as_c2w")
    print("id name front inside z_q01 z_q05 scale_over_z_q99 scale_over_z_max")
    for row in rows[:limit]:
        print(
            f"{row['id']} {row['name']} {row['front']:.4f} {row['inside']:.4f} "
            f"{row['z_q01']:.4f} {row['z_q05']:.4f} "
            f"{row['scale_over_z_q99']:.6f} {row['scale_over_z_max']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--sample-count", default=200_000, type=int)
    parser.add_argument("--scan-cameras", action="store_true")
    parser.add_argument("--scan-limit", default=20, type=int)
    args = parser.parse_args()

    camera_path = args.model_path / "cameras.json"
    ply_path = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    cameras = json.loads(camera_path.read_text(encoding="utf-8"))
    camera = cameras[args.camera_index]

    ply = PlyData.read(ply_path, mmap=True)
    vertex = ply["vertex"]
    indices = sample_indices(vertex.count, args.sample_count)
    points = np.stack(
        (
            np.asarray(vertex["x"])[indices],
            np.asarray(vertex["y"])[indices],
            np.asarray(vertex["z"])[indices],
        ),
        axis=1,
    ).astype(np.float64)

    scales = np.stack(
        (
            np.asarray(vertex["scale_0"])[indices],
            np.asarray(vertex["scale_1"])[indices],
            np.asarray(vertex["scale_2"])[indices],
        ),
        axis=1,
    ).astype(np.float64)
    max_scale = np.exp(scales).max(axis=1)

    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    position = np.asarray(camera["position"], dtype=np.float64)
    fovx = focal2fov(float(camera["fx"]), int(camera["width"]))
    fovy = focal2fov(float(camera["fy"]), int(camera["height"]))
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    print(f"model={args.model_path}")
    print(f"camera_id={camera['id']} image_name={camera.get('img_name', '')}")
    print(f"sampled_points={len(points)} total_points={vertex.count}")
    print(f"max_exp_scale={json.dumps(quantiles(max_scale), sort_keys=True)}")

    if args.scan_cameras:
        scan_cameras(cameras, points, max_scale, args.scan_limit)
        return

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position
    report_variant("json_as_c2w", points, np.linalg.inv(c2w), tanfovx, tanfovy)

    w2c_json = np.eye(4, dtype=np.float64)
    w2c_json[:3, :3] = rotation
    w2c_json[:3, 3] = position
    report_variant("json_as_w2c", points, w2c_json, tanfovx, tanfovy)

    graphdeco_direct = np.eye(4, dtype=np.float64)
    graphdeco_direct[:3, :3] = rotation.T
    graphdeco_direct[:3, 3] = position
    report_variant("rotation_transposed_position_direct", points, graphdeco_direct, tanfovx, tanfovy)


if __name__ == "__main__":
    main()
