#!/usr/bin/env python3
"""Generate SAM masks by projecting seed-view prompts through FlashSplat cameras."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from flashsplat_cameras import (
    background_tensor,
    camera_filename,
    default_pipeline,
    ensure_camera_item,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
    selected_camera_items,
    tensor_to_rgb_array,
)


def parse_point(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("points must be formatted as x,y")
    return float(parts[0]), float(parts[1])


def load_sam(segment_anything_root: Path, checkpoint: Path, arch: str) -> Any:
    if segment_anything_root:
        sys.path.insert(0, str(segment_anything_root))
    from segment_anything import SamPredictor, sam_model_registry

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    sam = sam_model_registry[arch](checkpoint=str(checkpoint)).to("cuda")
    return SamPredictor(sam)


def nearest_gaussians(
    proj_xy: torch.Tensor,
    gs_depth: torch.Tensor,
    points: Sequence[tuple[float, float]],
    top_k: int,
) -> List[int]:
    indices: List[int] = []
    depth = gs_depth.detach().clone()
    depth[depth <= 0] = 1.0e9
    limit = min(top_k, proj_xy.shape[1])

    for x, y in points:
        target = torch.tensor([x, y], dtype=proj_xy.dtype, device=proj_xy.device)[:, None]
        distance = ((proj_xy.detach() - target) ** 2).sum(dim=0)
        near = torch.topk(distance, k=limit, largest=False).indices
        best = near[torch.argmin(depth[near])]
        indices.append(int(best.item()))

    return sorted(set(indices))


def projected_points_for_indices(
    proj_xy: torch.Tensor,
    gs_depth: torch.Tensor,
    gaussian_indices: Sequence[int],
    width: int,
    height: int,
) -> np.ndarray:
    points: List[List[float]] = []
    for gaussian_index in gaussian_indices:
        x = float(proj_xy[0, gaussian_index].detach().cpu().item())
        y = float(proj_xy[1, gaussian_index].detach().cpu().item())
        depth = float(gs_depth[gaussian_index].detach().cpu().item())
        if not math.isfinite(x) or not math.isfinite(y) or depth <= 0:
            continue
        if 0 <= x < width and 0 <= y < height:
            points.append([x, y])
    return np.asarray(points, dtype=np.float32)


def save_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    output_path: Path,
) -> None:
    overlay = rgb.copy()
    object_color = np.asarray([255, 64, 32], dtype=np.uint8)
    selected = mask.astype(bool)
    overlay[selected] = (0.55 * overlay[selected] + 0.45 * object_color).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    for x, y in positive_points:
        radius = 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 255), outline=(0, 0, 0))
    for x, y in negative_points:
        radius = 3
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 0), outline=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--flashsplat-root",
        default="/lab/haoq_lab/cse12312032/external/FlashSplat",
        type=Path,
    )
    parser.add_argument(
        "--segment-anything-root",
        default="/lab/haoq_lab/cse12312032/external/SegAnyGAussians/third_party/segment-anything",
        type=Path,
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="/lab/haoq_lab/cse12312032/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth",
        type=Path,
    )
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--count", default=20, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--seed-camera-index", required=True, type=int)
    parser.add_argument("--point", action="append", required=True, type=parse_point)
    parser.add_argument("--negative-point", action="append", default=[], type=parse_point)
    parser.add_argument("--top-k", default=100, type=int)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rgb_renders"
    mask_dir = args.output_dir / "sam_masks"
    overlay_dir = args.output_dir / "mask_overlays"
    for directory in (rgb_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    selected_items = selected_camera_items(cameras, args.camera_indices, args.count)
    selected_items = ensure_camera_item(selected_items, cameras, args.seed_camera_index)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    predictor = load_sam(args.segment_anything_root, args.sam_checkpoint, args.sam_arch)

    seed_camera = make_camera(cameras[args.seed_camera_index], modules, args.max_width)
    with torch.no_grad():
        seed_render = render_flashsplat(seed_camera, gaussians, modules, pipeline, background)
        positive_gaussian_indices = nearest_gaussians(
            seed_render["proj_xy"],
            seed_render["gs_depth"],
            args.point,
            args.top_k,
        )
        negative_gaussian_indices = nearest_gaussians(
            seed_render["proj_xy"],
            seed_render["gs_depth"],
            args.negative_point,
            args.top_k,
        )

    manifest: Dict[str, Any] = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "seed_camera_index": args.seed_camera_index,
        "seed_camera_id": int(cameras[args.seed_camera_index]["id"]),
        "seed_points": [{"x": x, "y": y} for x, y in args.point],
        "seed_negative_points": [{"x": x, "y": y} for x, y in args.negative_point],
        "nearest_positive_gaussian_indices": positive_gaussian_indices,
        "nearest_negative_gaussian_indices": negative_gaussian_indices,
        "frames": [],
    }

    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(selected_items):
            camera = make_camera(camera_json, modules, args.max_width)
            render_pkg = render_flashsplat(camera, gaussians, modules, pipeline, background)
            rgb = tensor_to_rgb_array(render_pkg["render"])
            positive_points = projected_points_for_indices(
                render_pkg["proj_xy"],
                render_pkg["gs_depth"],
                positive_gaussian_indices,
                camera.image_width,
                camera.image_height,
            )
            negative_points = projected_points_for_indices(
                render_pkg["proj_xy"],
                render_pkg["gs_depth"],
                negative_gaussian_indices,
                camera.image_width,
                camera.image_height,
            )

            if len(positive_points) > 0:
                point_coords = positive_points
                point_labels = np.ones((len(positive_points),), dtype=np.int32)
                if len(negative_points) > 0:
                    point_coords = np.concatenate([positive_points, negative_points], axis=0)
                    point_labels = np.concatenate(
                        [
                            point_labels,
                            np.zeros((len(negative_points),), dtype=np.int32),
                        ],
                        axis=0,
                    )
                predictor.set_image(rgb)
                masks, scores, _logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                best_index = int(np.argmax(scores))
                mask = masks[best_index].astype(np.uint8)
                score = float(scores[best_index])
            else:
                mask = np.zeros((camera.image_height, camera.image_width), dtype=np.uint8)
                score = 0.0

            filename = camera_filename(output_index, camera_json)
            Image.fromarray(rgb, mode="RGB").save(rgb_dir / filename)
            Image.fromarray(mask * 255, mode="L").save(mask_dir / filename)
            save_overlay(rgb, mask, positive_points, negative_points, overlay_dir / filename)

            manifest["frames"].append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(camera_json["id"]),
                    "image_name": camera_json.get("img_name", ""),
                    "positive_prompt_points": positive_points.tolist(),
                    "negative_prompt_points": negative_points.tolist(),
                    "valid_positive_prompt_count": int(len(positive_points)),
                    "valid_negative_prompt_count": int(len(negative_points)),
                    "mask_pixels": int(mask.sum()),
                    "sam_score": score,
                    "render_width": int(camera.image_width),
                    "render_height": int(camera.image_height),
                }
            )
            print(
                f"wrote {filename}: positive_prompts={len(positive_points)} "
                f"negative_prompts={len(negative_points)} "
                f"mask_pixels={int(mask.sum())} sam_score={score:.4f}"
            )

    (args.output_dir / "prompt_mask_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
