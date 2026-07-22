#!/usr/bin/env python3
"""Lift DINOv2 class maps into sparse per-view Gaussian votes with FlashSplat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
)
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov2.dinov2_voting import (
    flashsplat_class_rows,
    mean_class_confidences,
    sparse_view_votes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"


def local_index_map(project_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    present = np.unique(project_ids)
    present = present[present != 0]
    class_ids = np.concatenate(
        [np.zeros((1,), dtype=np.uint16), present.astype(np.uint16, copy=False)]
    )
    indexed = np.searchsorted(class_ids, project_ids).astype(np.float32)
    return indexed, class_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--min-pixel-confidence", default=0.5, type=float)
    parser.add_argument("--support-threshold", default=0.05, type=float)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.min_pixel_confidence <= 1.0:
        raise ValueError("min-pixel-confidence must be between zero and one")
    if args.support_threshold < 0.0:
        raise ValueError("support-threshold must be non-negative")

    output_dir = args.output_dir or args.input_dir
    vote_dir = output_dir / "view_votes"
    vote_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "vote_manifest.json"
    if output_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_manifest_path} exists; pass --overwrite to replace it"
        )

    segmentation_manifest_path = args.input_dir / "dinov2_manifest.json"
    segmentation_manifest = json.loads(
        segmentation_manifest_path.read_text(encoding="utf-8")
    )
    manifest_threshold = float(segmentation_manifest["min_pixel_confidence"])
    if abs(manifest_threshold - args.min_pixel_confidence) > 1e-8:
        raise ValueError(
            "min-pixel-confidence must match the DINOv2 manifest "
            f"({manifest_threshold})"
        )
    ontology = load_ontology(args.ontology)
    lookup = ontology.ade_to_project
    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    frames: list[dict[str, Any]] = []

    with torch.no_grad():
        for frame in segmentation_manifest["frames"]:
            filename = str(frame["file"])
            camera_index = int(frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            segment_path = args.input_dir / str(frame["segment_file"])
            with np.load(segment_path) as segment:
                raw_class = segment["class_id"].astype(np.uint8, copy=False)
                confidence = segment["confidence"].astype(np.float32)
            expected_shape = (int(camera.image_height), int(camera.image_width))
            if raw_class.shape != expected_shape or confidence.shape != expected_shape:
                raise ValueError(
                    f"{segment_path} has shape {raw_class.shape}; expected {expected_shape}"
                )
            if int(raw_class.max(initial=0)) >= ontology.class_count:
                raise ValueError(f"{segment_path} contains a class outside the ontology")
            if (
                not np.isfinite(confidence).all()
                or confidence.min() < 0.0
                or confidence.max() > 1.0
            ):
                raise ValueError(f"{segment_path} contains invalid confidence values")
            project_ids = lookup[raw_class]
            project_ids[confidence < args.min_pixel_confidence] = 0
            indexed, class_ids = local_index_map(project_ids)
            means = mean_class_confidences(project_ids, confidence, class_ids)
            gt_mask = torch.from_numpy(indexed).to(device="cuda", dtype=torch.float32)
            render_pkg = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                gt_mask=gt_mask,
                obj_num=int(class_ids.shape[0]),
            )
            used_count = flashsplat_class_rows(
                render_pkg["used_count"].detach().float().cpu().numpy(),
                int(class_ids.shape[0]),
                gaussian_count,
            )
            indices, vote_classes, weights = sparse_view_votes(
                used_count,
                class_ids,
                means,
                view_quality=float(frame.get("view_quality", 1.0)),
                support_threshold=args.support_threshold,
            )
            vote_path = vote_dir / f"{Path(filename).stem}.npz"
            if vote_path.exists() and not args.overwrite:
                raise FileExistsError(f"{vote_path} exists; pass --overwrite")
            np.savez_compressed(
                vote_path,
                indices=indices,
                class_ids=vote_classes,
                weights=weights,
            )
            frames.append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(frame["camera_id"]),
                    "vote_file": vote_path.relative_to(output_dir).as_posix(),
                    "view_quality": float(frame.get("view_quality", 1.0)),
                    "present_project_class_ids": [int(value) for value in class_ids[1:]],
                    "abstain_mean_confidence": float(means[0]),
                    "class_mean_confidences": [float(value) for value in means[1:]],
                    "sparse_vote_count": int(weights.shape[0]),
                    "abstain_pixel_ratio": float(np.mean(project_ids == 0)),
                }
            )
            print(f"lifted {filename}: {weights.shape[0]} sparse votes")
            del render_pkg, used_count, gt_mask
            torch.cuda.empty_cache()

    output_manifest = {
        "source": "dinov2_flashsplat_per_view_votes",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "segmentation_manifest": str(segmentation_manifest_path),
        "ontology": str(args.ontology),
        "iteration": args.iteration,
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "vote_formula": "view_quality * mean_class_confidence * used_count / visibility",
        "visibility_definition": "sum_used_count_over_all_local_rows_including_abstain",
        "abstain_confidence_weighting": (
            "mean_max_softmax_confidence_over_below_threshold_pixels"
        ),
        "parameters": {
            "min_pixel_confidence": args.min_pixel_confidence,
            "support_threshold": args.support_threshold,
            "max_width": args.max_width,
        },
        "frames": frames,
    }
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(f"wrote {output_manifest_path}")


if __name__ == "__main__":
    main()
