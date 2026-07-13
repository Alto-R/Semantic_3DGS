#!/usr/bin/env python3
"""Assign Gaussian labels from 2D masks using FlashSplat used-counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
sys.path.insert(0, str(REPO_ROOT / "archive" / "task1_legacy" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "task1"))

from add_labels_from_npy import write_ply_with_labels
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


def load_mask(mask_path: Path, width: int, height: int) -> torch.Tensor:
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask = Image.open(mask_path).convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array).to(device="cuda", dtype=torch.float32)


def save_mask_overlay(rgb: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    overlay = rgb.copy()
    object_color = np.asarray([255, 64, 32], dtype=np.uint8)
    selected = mask.astype(bool)
    overlay[selected] = (0.55 * overlay[selected] + 0.45 * object_color).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def labels_from_counts(all_counts: torch.Tensor, slackness: float, label_id: int) -> np.ndarray:
    scores = F.normalize(all_counts.float(), dim=0)
    scores[0, :] += slackness
    binary_labels = scores.max(dim=0).indices.detach().cpu().numpy().astype(np.int32)
    labels = np.zeros_like(binary_labels, dtype=np.int32)
    labels[binary_labels == 1] = label_id
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--flashsplat-root",
        default=DEFAULT_FLASHSPLAT_ROOT,
        type=Path,
    )
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--count", default=20, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--slackness", default=0.0, type=float)
    parser.add_argument("--scene", default="")
    parser.add_argument("--label-id", default=1, type=int)
    parser.add_argument("--label-name", default="object")
    parser.add_argument("--label-class", default="object")
    parser.add_argument("--semantic-ply-name", default="semantic_point_cloud.ply")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts_dir = args.output_dir / "flashsplat"
    overlay_dir = args.output_dir / "overlay_renders"
    rgb_dir = args.output_dir / "flashsplat_rgb_renders"
    for directory in (counts_dir, overlay_dir, rgb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    selected_items = selected_camera_items(cameras, args.camera_indices, args.count)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    all_counts = None
    manifest: Dict[str, Any] = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "mask_dir": str(args.mask_dir),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "slackness": args.slackness,
        "frames": [],
    }

    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(selected_items):
            camera = make_camera(camera_json, modules, args.max_width)
            filename = camera_filename(output_index, camera_json)
            mask = load_mask(args.mask_dir / filename, camera.image_width, camera.image_height)
            render_pkg = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                gt_mask=mask,
                obj_num=1,
            )
            used_count = render_pkg["used_count"].detach()
            if all_counts is None:
                all_counts = torch.zeros_like(used_count)
            all_counts += used_count

            rgb = tensor_to_rgb_array(render_pkg["render"])
            mask_np = (mask.detach().cpu().numpy() >= 0.5).astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(rgb_dir / filename)
            save_mask_overlay(rgb, mask_np, overlay_dir / filename)

            manifest["frames"].append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(camera_json["id"]),
                    "image_name": camera_json.get("img_name", ""),
                    "mask_pixels": int(mask_np.sum()),
                    "used_count_shape": list(used_count.shape),
                }
            )
            print(f"processed {filename}: mask_pixels={int(mask_np.sum())}")

    if all_counts is None:
        raise RuntimeError("No views were processed")

    torch.save(all_counts.detach().cpu(), counts_dir / "flashsplat_counts.pt")
    labels = labels_from_counts(all_counts, args.slackness, args.label_id)
    labels_path = counts_dir / "gaussian_labels.npy"
    np.save(labels_path, labels)

    scene_name = args.scene or args.model_path.name
    label_map = {
        "scene": scene_name,
        "labels": [
            {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            {"id": args.label_id, "name": args.label_name, "class": args.label_class},
        ],
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2),
        encoding="utf-8",
    )

    manifest["counts_path"] = str(counts_dir / "flashsplat_counts.pt")
    manifest["labels_path"] = str(labels_path)
    manifest["label_histogram"] = {
        str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))
    }
    (counts_dir / "flashsplat_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    semantic_ply = args.output_dir / args.semantic_ply_name
    if semantic_ply.exists() and not args.overwrite:
        raise FileExistsError(f"{semantic_ply} exists; pass --overwrite to replace it")
    write_ply_with_labels(ply_path, semantic_ply, labels)
    print(f"wrote {semantic_ply}")


if __name__ == "__main__":
    main()
