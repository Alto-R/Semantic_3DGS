#!/usr/bin/env python3
"""Render colored validation overlays from automatic Gaussian labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from PIL import Image, ImageDraw

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
    selected_camera_items,
    tensor_to_rgb_array,
)
from scripts.task1.common.semantic_palette import (
    load_label_items,
    normalize_classes,
    palette_records,
    rgb_for_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"


def label_ids_from_map(
    label_map: Dict[str, Any],
    max_labels: int,
    focus_classes: set[str],
) -> List[int]:
    ids = [
        int(item["id"])
        for item in label_map.get("labels", [])
        if int(item.get("id", 0)) > 0
        and (not focus_classes or str(item.get("class", "")).lower() in focus_classes)
    ]
    ids = sorted(ids)
    if max_labels > 0:
        ids = ids[:max_labels]
    return ids


def color_tensor_from_labels(
    labels: np.ndarray,
    label_ids: Iterable[int],
    label_items: Dict[int, Dict[str, Any]],
    color_mode: str,
) -> torch.Tensor:
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for label_id in label_ids:
        colors[labels == label_id] = np.asarray(
            rgb_for_label(label_id, label_items, color_mode=color_mode),
            dtype=np.float32,
        )
    return torch.from_numpy(colors).to(device="cuda", dtype=torch.float32)


def save_overlay(base_rgb: np.ndarray, label_rgb: np.ndarray, output_path: Path, alpha: float) -> None:
    mask = label_rgb.max(axis=2) > 8
    overlay = base_rgb.copy()
    overlay[mask] = ((1.0 - alpha) * overlay[mask] + alpha * label_rgb[mask]).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--labels-npy", required=True, type=Path)
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rgb-output-dir", type=Path)
    parser.add_argument("--source-view-manifest", type=Path)
    parser.add_argument(
        "--flashsplat-root",
        default=DEFAULT_FLASHSPLAT_ROOT,
        type=Path,
    )
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--count", default=10, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--max-labels", default=64, type=int)
    parser.add_argument("--focus-classes", default="")
    parser.add_argument("--color-mode", choices=("class", "instance"), default="class")
    parser.add_argument("--overlay-alpha", default=0.55, type=float)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    if not args.labels_npy.exists():
        raise FileNotFoundError(args.labels_npy)
    if not args.label_map.exists():
        raise FileNotFoundError(args.label_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rgb_output_dir is not None:
        args.rgb_output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.labels_npy).astype(np.int32)
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    label_items = load_label_items(args.label_map)
    focus_classes = normalize_classes(args.focus_classes)
    label_ids = label_ids_from_map(label_map, args.max_labels, focus_classes)
    if not label_ids:
        raise ValueError("No nonzero labels found in label map")

    cameras = load_cameras(args.model_path)
    selected_items = selected_camera_items(cameras, args.camera_indices, args.count)
    source_frames_by_camera: dict[int, dict[str, Any]] = {}
    source_rgb_dir: Path | None = None
    if args.source_view_manifest is not None:
        source_manifest = json.loads(args.source_view_manifest.read_text(encoding="utf-8"))
        source_frames_by_camera = {
            int(frame["camera_index"]): frame
            for frame in source_manifest.get("frames", [])
        }
        source_rgb_dir = args.source_view_manifest.parent / "rgb_renders"
        missing = [
            camera_index
            for camera_index, _camera in selected_items
            if camera_index not in source_frames_by_camera
        ]
        if missing:
            raise ValueError(f"Selected cameras are absent from source view manifest: {missing}")
        if not source_rgb_dir.is_dir():
            raise FileNotFoundError(source_rgb_dir)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    colors_precomp = color_tensor_from_labels(labels, label_ids, label_items, args.color_mode)

    manifest = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "labels_npy": str(args.labels_npy),
        "label_map": str(args.label_map),
        "rgb_output_dir": str(args.rgb_output_dir) if args.rgb_output_dir is not None else None,
        "source_view_manifest": (
            str(args.source_view_manifest) if args.source_view_manifest is not None else None
        ),
        "reused_rgb_renders": args.source_view_manifest is not None,
        "label_ids": label_ids,
        "focus_classes": sorted(focus_classes),
        "color_mode": args.color_mode,
        "palette": palette_records(label_ids, label_items, color_mode=args.color_mode),
        "frames": [],
    }

    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(selected_items):
            camera = make_camera(camera_json, modules, args.max_width)
            base = None
            if source_rgb_dir is None:
                base = render_flashsplat(camera, gaussians, modules, pipeline, background)
            labels_render = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                override_color=colors_precomp,
            )
            if source_rgb_dir is not None:
                source_frame = source_frames_by_camera[camera_index]
                source_rgb_path = source_rgb_dir / str(source_frame["file"])
                if not source_rgb_path.exists():
                    raise FileNotFoundError(source_rgb_path)
                base_rgb = np.asarray(Image.open(source_rgb_path).convert("RGB"), dtype=np.uint8)
                label_rgb_shape = tuple(int(value) for value in labels_render["render"].shape[1:])
                if base_rgb.shape[:2] != label_rgb_shape:
                    raise ValueError(
                        f"Source RGB shape {base_rgb.shape[:2]} does not match rendered label "
                        f"shape {label_rgb_shape} for camera {camera_index}"
                    )
            else:
                if base is None:
                    raise RuntimeError("Base RGB render was not initialized")
                base_rgb = tensor_to_rgb_array(base["render"])
            label_rgb = tensor_to_rgb_array(labels_render["render"])
            filename = camera_filename(output_index, camera_json)
            if args.rgb_output_dir is not None:
                Image.fromarray(base_rgb, mode="RGB").save(args.rgb_output_dir / filename)
            save_overlay(base_rgb, label_rgb, args.output_dir / filename, args.overlay_alpha)
            manifest["frames"].append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(camera_json["id"]),
                    "image_name": camera_json.get("img_name", ""),
                }
            )
            print(f"wrote {filename}")

    (args.output_dir / "label_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
