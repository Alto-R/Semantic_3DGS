#!/usr/bin/env python3
"""Generate automatic SAM mask proposals from GraphDeco 3DGS renders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_SEGMENT_ANYTHING_ROOT = (
    WORKSPACE_ROOT / "external" / "SegAnyGAussians" / "third_party" / "segment-anything"
)
DEFAULT_SAM_CHECKPOINT = WORKSPACE_ROOT / "InvRGBL_modif" / "pretrained" / "sam_vit_h_4b8939.pth"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "task1"))

from flashsplat_cameras import (
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


def load_sam_generator(
    segment_anything_root: Path,
    checkpoint: Path,
    arch: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    min_mask_region_area: int,
) -> Any:
    if segment_anything_root:
        sys.path.insert(0, str(segment_anything_root))
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    sam = sam_model_registry[arch](checkpoint=str(checkpoint)).to("cuda")
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=min_mask_region_area,
    )


def keep_mask(mask: Dict[str, Any], image_area: int, min_area: int, max_area_ratio: float) -> bool:
    area = int(mask.get("area", 0))
    if area < min_area:
        return False
    if max_area_ratio > 0 and area > image_area * max_area_ratio:
        return False
    return True


def mask_record(mask: Dict[str, Any], index: int) -> Dict[str, Any]:
    bbox = mask.get("bbox", [0, 0, 0, 0])
    return {
        "mask_index": index,
        "area": int(mask.get("area", 0)),
        "bbox": [int(value) for value in bbox],
        "predicted_iou": float(mask.get("predicted_iou", 0.0)),
        "stability_score": float(mask.get("stability_score", 0.0)),
        "crop_box": [int(value) for value in mask.get("crop_box", [])],
    }


def select_masks(
    masks: List[Dict[str, Any]],
    image_area: int,
    min_area: int,
    max_area_ratio: float,
    max_masks: int,
) -> List[Dict[str, Any]]:
    kept = [mask for mask in masks if keep_mask(mask, image_area, min_area, max_area_ratio)]
    kept.sort(
        key=lambda mask: (
            float(mask.get("predicted_iou", 0.0)) * float(mask.get("stability_score", 0.0)),
            int(mask.get("area", 0)),
        ),
        reverse=True,
    )
    if max_masks > 0:
        kept = kept[:max_masks]
    return kept


def save_mask_overlay(rgb: np.ndarray, masks: np.ndarray, output_path: Path) -> None:
    overlay = rgb.copy()
    for index, mask in enumerate(masks):
        color = np.asarray(
            [
                (37 * (index + 1)) % 255,
                (97 * (index + 3)) % 255,
                (173 * (index + 5)) % 255,
            ],
            dtype=np.uint8,
        )
        selected = mask.astype(bool)
        overlay[selected] = (0.55 * overlay[selected] + 0.45 * color).astype(np.uint8)

    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--flashsplat-root",
        default=DEFAULT_FLASHSPLAT_ROOT,
        type=Path,
    )
    parser.add_argument(
        "--segment-anything-root",
        default=DEFAULT_SEGMENT_ANYTHING_ROOT,
        type=Path,
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_SAM_CHECKPOINT,
        type=Path,
    )
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--count", default=20, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--max-masks-per-view", default=32, type=int)
    parser.add_argument("--min-mask-area", default=100, type=int)
    parser.add_argument("--max-mask-area-ratio", default=0.70, type=float)
    parser.add_argument("--points-per-side", default=24, type=int)
    parser.add_argument("--pred-iou-thresh", default=0.88, type=float)
    parser.add_argument("--stability-score-thresh", default=0.92, type=float)
    parser.add_argument("--box-nms-thresh", default=0.70, type=float)
    parser.add_argument("--crop-n-layers", default=0, type=int)
    parser.add_argument("--min-mask-region-area", default=100, type=int)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    rgb_dir = args.output_dir / "rgb_renders"
    mask_dir = args.output_dir / "sam_auto_masks"
    overlay_dir = args.output_dir / "sam_auto_overlays"
    for directory in (rgb_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    selected_items = selected_camera_items(cameras, args.camera_indices, args.count)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    mask_generator = load_sam_generator(
        args.segment_anything_root,
        args.sam_checkpoint,
        args.sam_arch,
        args.points_per_side,
        args.pred_iou_thresh,
        args.stability_score_thresh,
        args.box_nms_thresh,
        args.crop_n_layers,
        args.min_mask_region_area,
    )

    manifest: Dict[str, Any] = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "max_masks_per_view": args.max_masks_per_view,
        "min_mask_area": args.min_mask_area,
        "max_mask_area_ratio": args.max_mask_area_ratio,
        "sam": {
            "checkpoint": str(args.sam_checkpoint),
            "arch": args.sam_arch,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "box_nms_thresh": args.box_nms_thresh,
            "crop_n_layers": args.crop_n_layers,
            "min_mask_region_area": args.min_mask_region_area,
        },
        "frames": [],
    }

    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(selected_items):
            camera = make_camera(camera_json, modules, args.max_width)
            render_pkg = render_flashsplat(camera, gaussians, modules, pipeline, background)
            rgb = tensor_to_rgb_array(render_pkg["render"])
            filename = camera_filename(output_index, camera_json)
            stem = Path(filename).stem
            Image.fromarray(rgb, mode="RGB").save(rgb_dir / filename)

            raw_masks = mask_generator.generate(rgb)
            selected_masks = select_masks(
                raw_masks,
                image_area=int(camera.image_width * camera.image_height),
                min_area=args.min_mask_area,
                max_area_ratio=args.max_mask_area_ratio,
                max_masks=args.max_masks_per_view,
            )
            mask_stack = np.zeros((0, camera.image_height, camera.image_width), dtype=np.uint8)
            records: List[Dict[str, Any]] = []
            if selected_masks:
                mask_stack = np.stack(
                    [mask["segmentation"].astype(np.uint8) for mask in selected_masks],
                    axis=0,
                )
                records = [mask_record(mask, index) for index, mask in enumerate(selected_masks)]

            np.savez_compressed(mask_dir / f"{stem}.npz", masks=mask_stack)
            save_mask_overlay(rgb, mask_stack, overlay_dir / filename)

            frame_record = {
                "file": filename,
                "mask_file": f"{stem}.npz",
                "camera_index": camera_index,
                "camera_id": int(camera_json["id"]),
                "image_name": camera_json.get("img_name", ""),
                "render_width": int(camera.image_width),
                "render_height": int(camera.image_height),
                "raw_mask_count": int(len(raw_masks)),
                "kept_mask_count": int(mask_stack.shape[0]),
                "masks": records,
            }
            manifest["frames"].append(frame_record)
            print(
                f"wrote {filename}: raw_masks={len(raw_masks)} "
                f"kept_masks={mask_stack.shape[0]}"
            )

    (args.output_dir / "sam_auto_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
