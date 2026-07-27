#!/usr/bin/env python3
"""Render exact accepted incremental-fill supports into their source views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
    tensor_to_rgb_array,
)
from scripts.task1.common.semantic_palette import rgb8_for_class, rgb_for_class


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
REPORT_CONTRACTS = {
    "report_only_dinov3_incremental_spatial_core_fill_v1",
    "report_only_dinov3_automatic_anchor_guard_fill_v1",
}


def build_exact_residual_colors(
    vertex_count: int,
    report: dict[str, Any],
    supports: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build in-memory class colors directly from accepted sparse supports."""

    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    if report.get("contract") not in REPORT_CONTRACTS:
        raise ValueError("incremental-fill report has the wrong contract")
    if int(report.get("vertex_count", -1)) != vertex_count:
        raise ValueError("incremental-fill report vertex count differs from model")

    colors = np.zeros((vertex_count, 3), dtype=np.float32)
    selected_mask = np.zeros((vertex_count,), dtype=bool)
    palette_by_class: dict[str, dict[str, Any]] = {}
    seen_component_ids: set[int] = set()
    for record in report.get("components", []):
        if not bool(record.get("accepted")):
            continue
        component_id = int(record["component_id"])
        if component_id < 1 or component_id in seen_component_ids:
            raise ValueError("accepted residual component IDs are invalid")
        seen_component_ids.add(component_id)
        index_key = f"component_{component_id:06d}_indices"
        if index_key not in supports:
            raise ValueError(
                f"missing sparse support for residual component {component_id}"
            )
        indices = np.asarray(supports[index_key], dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("sparse residual indices must be one-dimensional")
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= vertex_count
        ):
            raise ValueError("sparse residual index is outside the model")
        if indices.size and np.unique(indices).size != indices.size:
            raise ValueError("sparse residual indices contain duplicates")
        if np.any(selected_mask[indices]):
            raise ValueError("accepted residual sparse supports overlap")
        class_name = str(record["class"])
        colors[indices] = np.asarray(rgb_for_class(class_name), dtype=np.float32)
        selected_mask[indices] = True
        palette_by_class.setdefault(
            class_name,
            {
                "class": class_name,
                "rgb": rgb8_for_class(class_name),
            },
        )

    expected_count = int(report.get("incremental_fill_gaussian_count", -1))
    actual_count = int(np.count_nonzero(selected_mask))
    if expected_count != actual_count:
        raise ValueError("sparse support count differs from incremental-fill report")
    return colors, selected_mask, [
        palette_by_class[key] for key in sorted(palette_by_class)
    ]


def save_exact_overlay(
    base_rgb: np.ndarray,
    class_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    output_path: Path,
    alpha: float,
) -> int:
    if class_rgb.shape != base_rgb.shape or mask_rgb.shape != base_rgb.shape:
        raise ValueError("base, class, and mask renders must have matching shapes")
    visible = mask_rgb.max(axis=2) > 8
    overlay = base_rgb.copy()
    overlay[visible] = (
        (1.0 - alpha) * overlay[visible].astype(np.float32)
        + alpha * class_rgb[visible].astype(np.float32)
    ).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return int(np.count_nonzero(visible))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--support-npz", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-manifest", required=True, type=Path)
    parser.add_argument("--overlay-output-dir", required=True, type=Path)
    parser.add_argument("--mask-output-dir", required=True, type=Path)
    parser.add_argument(
        "--flashsplat-root",
        default=DEFAULT_FLASHSPLAT_ROOT,
        type=Path,
    )
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=0, type=int)
    parser.add_argument("--overlay-alpha", default=0.55, type=float)
    args = parser.parse_args()

    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between zero and one")
    for path in (args.audit_report, args.support_npz, args.source_view_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.overlay_output_dir.exists():
        raise FileExistsError(args.overlay_output_dir)
    if args.mask_output_dir.exists():
        raise FileExistsError(args.mask_output_dir)

    report = json.loads(args.audit_report.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        args.source_view_manifest.read_text(encoding="utf-8")
    )
    source_frames = source_manifest.get("frames", [])
    if not isinstance(source_frames, list) or not source_frames:
        raise ValueError("source view manifest has no frames")
    camera_indices = [int(frame["camera_index"]) for frame in source_frames]
    if len(set(camera_indices)) != len(camera_indices):
        raise ValueError("source view manifest contains duplicate cameras")
    output_files = [Path(str(frame["file"])).name for frame in source_frames]
    if len(set(output_files)) != len(output_files):
        raise ValueError("source view manifest contains duplicate output filenames")
    source_rgb_dir = args.source_view_manifest.parent / "rgb_renders"
    if not source_rgb_dir.is_dir():
        raise FileNotFoundError(source_rgb_dir)

    cameras = load_cameras(args.model_path)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    modules = load_flashsplat(args.flashsplat_root)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    vertex_count = int(gaussians.get_xyz.shape[0])
    with np.load(args.support_npz, allow_pickle=False) as supports:
        colors, selected_mask, palette = build_exact_residual_colors(
            vertex_count, report, supports
        )

    render_max_width = args.max_width
    if render_max_width <= 0:
        render_max_width = int(source_manifest.get("max_width", 0))
    if render_max_width <= 0:
        raise ValueError("max width is absent from both arguments and source manifest")

    args.overlay_output_dir.mkdir(parents=True, exist_ok=False)
    args.mask_output_dir.mkdir(parents=True, exist_ok=False)
    class_colors = torch.from_numpy(colors).to(device="cuda", dtype=torch.float32)
    binary_colors = torch.from_numpy(
        np.repeat(selected_mask[:, None], 3, axis=1).astype(np.float32)
    ).to(device="cuda", dtype=torch.float32)
    pipeline = default_pipeline()
    background = background_tensor(False)
    frames: list[dict[str, Any]] = []

    with torch.no_grad():
        for frame in source_frames:
            camera_index = int(frame["camera_index"])
            if camera_index < 0 or camera_index >= len(cameras):
                raise IndexError(f"camera index {camera_index} is outside the model")
            source_file = str(frame["file"])
            source_rgb_path = source_rgb_dir / source_file
            if not source_rgb_path.is_file():
                raise FileNotFoundError(source_rgb_path)
            camera = make_camera(
                cameras[camera_index],
                modules,
                render_max_width,
            )
            class_render = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                override_color=class_colors,
            )
            mask_render = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                override_color=binary_colors,
            )
            base_rgb = np.asarray(
                Image.open(source_rgb_path).convert("RGB"),
                dtype=np.uint8,
            )
            class_rgb = tensor_to_rgb_array(class_render["render"])
            mask_rgb = tensor_to_rgb_array(mask_render["render"])
            if base_rgb.shape != class_rgb.shape:
                raise ValueError(
                    f"source RGB shape {base_rgb.shape} does not match exact "
                    f"residual render shape {class_rgb.shape} for camera {camera_index}"
                )
            output_file = Path(source_file).name
            visible_pixel_count = save_exact_overlay(
                base_rgb,
                class_rgb,
                mask_rgb,
                args.overlay_output_dir / output_file,
                args.overlay_alpha,
            )
            Image.fromarray(mask_rgb.max(axis=2), mode="L").save(
                args.mask_output_dir / output_file
            )
            frames.append(
                {
                    "file": output_file,
                    "source_file": source_file,
                    "camera_index": camera_index,
                    "visible_residual_pixel_count": visible_pixel_count,
                }
            )
            print(f"wrote exact residual projection {output_file}")

    manifest = {
        "source": str(args.audit_report),
        "contract": str(report["contract"]),
        "visualization_scope": (
            "exact_projection_of_report_only_incremental_fill_3d_support"
        ),
        "support_npz": str(args.support_npz),
        "model_path": str(args.model_path),
        "source_view_manifest": str(args.source_view_manifest),
        "reused_matched_rgb_renders": True,
        "rendered_gaussian_count": int(np.count_nonzero(selected_mask)),
        "render_max_width": render_max_width,
        "palette": palette,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "frames": frames,
    }
    (args.overlay_output_dir / "exact_residual_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
