#!/usr/bin/env python3
"""Render real GraphDeco training cameras for DINOv2 semantic inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument(
        "--count",
        default=0,
        type=int,
        help="Evenly sample this many cameras; zero renders every real camera.",
    )
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cameras = load_cameras(args.model_path)
    if args.camera_indices:
        selected = selected_camera_items(cameras, args.camera_indices, max(args.count, 1))
    elif args.count <= 0:
        selected = list(enumerate(cameras))
    else:
        selected = selected_camera_items(cameras, "", args.count)
    if not selected:
        raise ValueError("No real cameras were selected")

    rgb_dir = args.output_dir / "rgb_renders"
    manifest_path = args.output_dir / "view_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to replace it")
    rgb_dir.mkdir(parents=True, exist_ok=True)

    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    frames: list[dict[str, object]] = []

    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(selected):
            camera = make_camera(camera_json, modules, args.max_width)
            output = render_flashsplat(camera, gaussians, modules, pipeline, background)
            filename = camera_filename(output_index, camera_json)
            output_path = rgb_dir / filename
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
            Image.fromarray(tensor_to_rgb_array(output["render"]), mode="RGB").save(output_path)
            frames.append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(camera_json["id"]),
                    "image_name": camera_json.get("img_name", ""),
                    "render_width": int(camera.image_width),
                    "render_height": int(camera.image_height),
                    "view_quality": 1.0,
                    "camera_source": "cameras.json",
                }
            )
            print(f"rendered {filename}")

    manifest = {
        "source": "graphdeco_real_cameras",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "sh_degree": args.sh_degree,
        "max_width": args.max_width,
        "white_background": args.white_background,
        "camera_count": len(frames),
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
