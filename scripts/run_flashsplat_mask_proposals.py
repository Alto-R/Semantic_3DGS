#!/usr/bin/env python3
"""Lift automatic 2D mask proposals to sparse Gaussian support sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
)


def load_mask_stack(mask_path: Path, height: int, width: int) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    with np.load(mask_path) as data:
        masks = data["masks"].astype(bool)
    if masks.ndim != 3:
        raise ValueError(f"{mask_path} masks must have shape (K,H,W)")
    if masks.shape[1:] != (height, width):
        resized = []
        for mask in masks:
            image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            image = image.resize((width, height), Image.Resampling.NEAREST)
            resized.append(np.asarray(image, dtype=np.uint8) > 0)
        masks = np.stack(resized, axis=0) if resized else np.zeros((0, height, width), dtype=bool)
    return masks


def build_index_mask(masks: np.ndarray, start: int, end: int) -> torch.Tensor:
    height, width = masks.shape[1:]
    indexed = np.zeros((height, width), dtype=np.float32)
    for local_id, mask_index in enumerate(range(start, end), start=1):
        indexed[masks[mask_index].astype(bool)] = float(local_id)
    return torch.from_numpy(indexed).to(device="cuda", dtype=torch.float32)


def save_support(
    output_path: Path,
    support_indices: np.ndarray,
    support_counts: np.ndarray,
    proposal_id: int,
    frame_file: str,
    mask_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        indices=support_indices.astype(np.uint32),
        counts=support_counts.astype(np.float32),
        proposal_id=np.asarray([proposal_id], dtype=np.int32),
        frame_file=np.asarray([frame_file]),
        mask_index=np.asarray([mask_index], dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sam-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--flashsplat-root",
        default="/lab/haoq_lab/cse12312032/external/FlashSplat",
        type=Path,
    )
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--mask-batch-size", default=4, type=int)
    parser.add_argument("--max-masks-per-view", default=32, type=int)
    parser.add_argument("--support-threshold", default=0.0, type=float)
    parser.add_argument("--min-support-gaussians", default=100, type=int)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    if args.mask_batch_size <= 0:
        raise ValueError("--mask-batch-size must be positive")

    manifest_path = args.sam_output_dir / "sam_auto_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    sam_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    support_dir = args.output_dir / "proposal_supports"
    support_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    proposal_id = 1
    proposals: List[Dict[str, Any]] = []
    output_manifest: Dict[str, Any] = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "sam_output_dir": str(args.sam_output_dir),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "mask_batch_size": args.mask_batch_size,
        "support_threshold": args.support_threshold,
        "min_support_gaussians": args.min_support_gaussians,
        "proposals": proposals,
    }

    with torch.no_grad():
        for frame in sam_manifest["frames"]:
            camera_index = int(frame["camera_index"])
            camera_json = cameras[camera_index]
            camera = make_camera(camera_json, modules, args.max_width)
            mask_path = args.sam_output_dir / "sam_auto_masks" / frame["mask_file"]
            masks = load_mask_stack(mask_path, int(camera.image_height), int(camera.image_width))
            if args.max_masks_per_view > 0:
                masks = masks[: args.max_masks_per_view]
            if masks.shape[0] == 0:
                print(f"skipped {frame['file']}: no masks")
                continue

            for start in range(0, masks.shape[0], args.mask_batch_size):
                end = min(start + args.mask_batch_size, masks.shape[0])
                gt_mask = build_index_mask(masks, start, end)
                render_pkg = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    gt_mask=gt_mask,
                    obj_num=(end - start) + 1,
                )
                used_count = render_pkg["used_count"].detach().cpu()

                for local_id, mask_index in enumerate(range(start, end), start=1):
                    counts = used_count[local_id].numpy()
                    support = np.flatnonzero(counts > args.support_threshold)
                    if support.shape[0] < args.min_support_gaussians:
                        continue

                    support_file = f"proposal_{proposal_id:06d}.npz"
                    save_support(
                        support_dir / support_file,
                        support,
                        counts[support],
                        proposal_id,
                        frame["file"],
                        mask_index,
                    )
                    mask_meta = {}
                    if mask_index < len(frame.get("masks", [])):
                        mask_meta = frame["masks"][mask_index]
                    proposals.append(
                        {
                            "proposal_id": proposal_id,
                            "support_file": support_file,
                            "frame_file": frame["file"],
                            "camera_index": camera_index,
                            "camera_id": int(frame["camera_id"]),
                            "image_name": frame.get("image_name", ""),
                            "mask_index": int(mask_index),
                            "mask_area": int(mask_meta.get("area", int(masks[mask_index].sum()))),
                            "gaussian_count": int(support.shape[0]),
                            "predicted_iou": float(mask_meta.get("predicted_iou", 0.0)),
                            "stability_score": float(mask_meta.get("stability_score", 0.0)),
                        }
                    )
                    print(
                        f"proposal {proposal_id:06d}: frame={frame['file']} "
                        f"mask={mask_index} gaussians={support.shape[0]}"
                    )
                    proposal_id += 1

                del used_count
                del gt_mask
                torch.cuda.empty_cache()

    (args.output_dir / "proposal_manifest.json").write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'proposal_manifest.json'}")


if __name__ == "__main__":
    main()
