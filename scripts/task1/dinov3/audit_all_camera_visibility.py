#!/usr/bin/env python3
"""Measure DINO-liftable Gaussian visibility in every reconstruction camera.

The audit renders an all-zero mask through FlashSplat. Any Gaussian with
positive accumulated ``used_count`` in a camera can receive dense semantic
evidence from at least one pixel in that camera. No semantic model, prior
labels, label array, or PLY is read or written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"


def visibility_support(used_count: np.ndarray, gaussian_count: int) -> np.ndarray:
    """Collapse FlashSplat mask rows into one non-negative support vector."""

    values = np.asarray(used_count, dtype=np.float32)
    if gaussian_count < 1:
        raise ValueError("gaussian_count must be positive")
    if values.ndim == 1:
        if values.shape != (gaussian_count,):
            raise ValueError("used_count has the wrong Gaussian dimension")
        support = values
    elif values.ndim == 2:
        if values.shape[1] != gaussian_count:
            raise ValueError("used_count has the wrong Gaussian dimension")
        support = values.sum(axis=0, dtype=np.float32)
    else:
        raise ValueError("used_count must have shape gaussians or rows x gaussians")
    if not np.isfinite(support).all() or np.any(support < 0.0):
        raise ValueError("FlashSplat visibility support must be finite and non-negative")
    return support.astype(np.float32, copy=False)


def update_visibility_state(
    support: np.ndarray,
    *,
    camera_index: int,
    support_threshold: float,
    view_count: np.ndarray,
    total_support: np.ndarray,
    max_support: np.ndarray,
    first_camera_index: np.ndarray,
    best_camera_index: np.ndarray,
) -> np.ndarray:
    """Accumulate one camera and return its Boolean visibility mask."""

    values = np.asarray(support, dtype=np.float32)
    expected = view_count.shape
    arrays = (total_support, max_support, first_camera_index, best_camera_index)
    if values.shape != expected or any(array.shape != expected for array in arrays):
        raise ValueError("visibility state arrays must have one value per Gaussian")
    if support_threshold < 0.0:
        raise ValueError("support_threshold must be non-negative")
    if camera_index < 0:
        raise ValueError("camera_index must be non-negative")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("visibility support must be finite and non-negative")

    visible = values > np.float32(support_threshold)
    new = visible & (view_count == 0)
    first_camera_index[new] = np.int32(camera_index)
    stronger = visible & (values > max_support)
    max_support[stronger] = values[stronger]
    best_camera_index[stronger] = np.int32(camera_index)
    total_support[visible] += values[visible]
    view_count[visible] += np.uint16(1)
    return visible


def distribution(values: np.ndarray) -> dict[str, float]:
    """Return deterministic quantiles suitable for a JSON audit report."""

    array = np.asarray(values)
    if array.size == 0:
        return {key: 0.0 for key in ("min", "p25", "median", "p75", "p90", "p95", "p99", "max")}
    quantiles = np.quantile(array.astype(np.float64), [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p25", "median", "p75", "p90", "p95", "p99", "max"),
            quantiles,
        )
    }


def summarize_visibility(
    view_count: np.ndarray,
    total_support: np.ndarray,
    max_support: np.ndarray,
) -> dict[str, Any]:
    """Summarize zero-, one-, and multiview Gaussian coverage."""

    views = np.asarray(view_count)
    totals = np.asarray(total_support)
    maxima = np.asarray(max_support)
    if views.ndim != 1 or totals.shape != views.shape or maxima.shape != views.shape:
        raise ValueError("visibility arrays must be matching one-dimensional arrays")
    vertex_count = int(views.size)
    observed = views > 0
    observed_count = int(np.count_nonzero(observed))
    counts, frequencies = np.unique(views, return_counts=True)
    return {
        "vertex_count": vertex_count,
        "observed_gaussian_count": observed_count,
        "observed_ratio": float(observed_count / vertex_count) if vertex_count else 0.0,
        "unobserved_gaussian_count": int(np.count_nonzero(~observed)),
        "single_view_gaussian_count": int(np.count_nonzero(views == 1)),
        "multiview_gaussian_count": int(np.count_nonzero(views >= 2)),
        "view_count_distribution": distribution(views),
        "observed_total_support_distribution": distribution(totals[observed]),
        "observed_max_support_distribution": distribution(maxima[observed]),
        "view_count_histogram": {
            str(int(count)): int(frequency)
            for count, frequency in zip(counts, frequencies)
        },
    }


def main() -> None:
    import torch

    from scripts.task1.common.flashsplat_cameras import (
        background_tensor,
        camera_filename,
        default_pipeline,
        load_cameras,
        load_flashsplat,
        load_gaussians,
        make_camera,
        point_cloud_path,
        render_flashsplat,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--support-threshold", default=0.0, type=float)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_width < 1:
        raise ValueError("max_width must be positive")
    if args.support_threshold < 0.0:
        raise ValueError("support_threshold must be non-negative")

    cameras = load_cameras(args.model_path)
    if not cameras:
        raise ValueError("model contains no reconstruction cameras")
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    args.output_dir.mkdir(parents=True)
    camera_count = len(cameras)
    packed_width = (gaussian_count + 7) // 8
    packed_path = args.output_dir / "camera_visibility_bits.npy"
    packed = np.lib.format.open_memmap(
        packed_path,
        mode="w+",
        dtype=np.uint8,
        shape=(camera_count, packed_width),
    )
    view_count = np.zeros((gaussian_count,), dtype=np.uint16)
    total_support = np.zeros((gaussian_count,), dtype=np.float32)
    max_support = np.zeros((gaussian_count,), dtype=np.float32)
    first_camera_index = np.full((gaussian_count,), -1, dtype=np.int32)
    best_camera_index = np.full((gaussian_count,), -1, dtype=np.int32)
    camera_indices = np.arange(camera_count, dtype=np.int32)
    frame_records: list[dict[str, Any]] = []
    cumulative_observed = 0
    started = time.time()

    with torch.no_grad():
        for output_index, camera_json in enumerate(cameras):
            camera = make_camera(camera_json, modules, args.max_width)
            gt_mask = torch.zeros(
                (int(camera.image_height), int(camera.image_width)),
                dtype=torch.float32,
                device="cuda",
            )
            render_pkg = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                gt_mask=gt_mask,
                obj_num=1,
            )
            support = visibility_support(
                render_pkg["used_count"].detach().float().cpu().numpy(),
                gaussian_count,
            )
            visible = update_visibility_state(
                support,
                camera_index=output_index,
                support_threshold=args.support_threshold,
                view_count=view_count,
                total_support=total_support,
                max_support=max_support,
                first_camera_index=first_camera_index,
                best_camera_index=best_camera_index,
            )
            packed[output_index] = np.packbits(visible, bitorder="little")
            newly_observed = int(np.count_nonzero(visible & (view_count == 1)))
            cumulative_observed += newly_observed
            frame_records.append(
                {
                    "file": camera_filename(output_index, camera_json),
                    "camera_index": output_index,
                    "camera_id": int(camera_json["id"]),
                    "render_width": int(camera.image_width),
                    "render_height": int(camera.image_height),
                    "visible_gaussian_count": int(np.count_nonzero(visible)),
                    "newly_observed_gaussian_count": newly_observed,
                    "cumulative_observed_gaussian_count": cumulative_observed,
                }
            )
            print(
                f"camera {output_index + 1}/{camera_count}: "
                f"visible={int(np.count_nonzero(visible))} "
                f"new={newly_observed} cumulative={cumulative_observed}"
            )
            del render_pkg, gt_mask, support, visible
            torch.cuda.empty_cache()

    packed.flush()
    del packed
    outputs = {
        "visibility_view_count.npy": view_count,
        "visibility_total_support.npy": total_support,
        "visibility_max_support.npy": max_support,
        "first_camera_index.npy": first_camera_index,
        "best_camera_index.npy": best_camera_index,
        "camera_indices.npy": camera_indices,
        "unobserved_gaussian_indices.npy": np.flatnonzero(view_count == 0).astype(np.uint32),
        "single_view_gaussian_indices.npy": np.flatnonzero(view_count == 1).astype(np.uint32),
    }
    for filename, array in outputs.items():
        np.save(args.output_dir / filename, array)

    summary = summarize_visibility(view_count, total_support, max_support)
    report = {
        "source": "flashsplat_all_reconstruction_camera_visibility",
        "contract": "all_reconstruction_camera_nonzero_support_v1",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "support_threshold": args.support_threshold,
        "visibility_definition": "sum_flashsplat_used_count_over_all_mask_rows > support_threshold",
        "semantic_evidence_interpretation": "a visible Gaussian can receive dense pixel evidence from that camera",
        "camera_selection_policy": "all_reconstruction_cameras_in_cameras_json_order",
        "all_reconstruction_cameras_used": True,
        "camera_count": camera_count,
        **summary,
        "elapsed_seconds": time.time() - started,
        "semantic_inference_used": False,
        "prior_semantic_labels_used": False,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "packed_visibility": {
            "file": packed_path.name,
            "shape": [camera_count, packed_width],
            "dtype": "uint8",
            "bit_order": "little",
            "gaussian_count": gaussian_count,
            "row_camera_indices_file": "camera_indices.npy",
        },
        "frames": frame_records,
    }
    report_path = args.output_dir / "all_camera_visibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
