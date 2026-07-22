#!/usr/bin/env python3
"""Lift per-view dense semantic maps to per-Gaussian class votes.

For every segmented view, the compact project class map is used directly as a
FlashSplat multi-object index mask, so one rasterizer call per class batch
returns the per-Gaussian support of every class present in that view. Votes
are accumulated across views into:

    votes            (N, C) float32  weighted vote mass per project class
    views_supporting (N, C) uint16   distinct views voting that class
    visible_views    (N,)   uint16   views where the Gaussian contributed at all

Column c corresponds to compact project class id c+1 (id 0 = ignore casts no
vote but still counts toward visibility).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

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
from scripts.task1.dense_seg.ade20k_ontology import load_ontology


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.dense_backends.json"


def load_seg_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        project_class = data["project_class"].astype(np.int16)
        confidence = data["confidence"].astype(np.float32)
    return project_class, confidence


def resize_to(
    project_class: np.ndarray,
    confidence: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if project_class.shape == (height, width):
        return project_class, confidence
    class_image = Image.fromarray(project_class.astype(np.int32), mode="I")
    class_image = class_image.resize((width, height), Image.Resampling.NEAREST)
    confidence_image = Image.fromarray(confidence, mode="F")
    confidence_image = confidence_image.resize((width, height), Image.Resampling.BILINEAR)
    return (
        np.asarray(class_image, dtype=np.int16),
        np.asarray(confidence_image, dtype=np.float32),
    )


def class_mean_confidence(
    project_class: np.ndarray,
    confidence: np.ndarray,
    class_ids: np.ndarray,
) -> dict[int, float]:
    means: dict[int, float] = {}
    for class_id in class_ids:
        mask = project_class == class_id
        means[int(class_id)] = float(confidence[mask].mean()) if mask.any() else 0.0
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seg-dir", required=True, type=Path, help="Stage 01 output dir")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument(
        "--min-confidence",
        default=0.35,
        type=float,
        help="Pixels below this backend confidence cast no vote",
    )
    parser.add_argument(
        "--support-threshold",
        default=0.05,
        type=float,
        help="Minimum FlashSplat used_count for a Gaussian to receive a vote",
    )
    parser.add_argument(
        "--class-batch-size",
        default=32,
        type=int,
        help="Maximum classes per FlashSplat index-mask call",
    )
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    if args.class_batch_size <= 0:
        raise ValueError("--class-batch-size must be positive")

    manifest_path = args.seg_dir / "dense_seg_manifest.json"
    seg_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ontology = load_ontology(args.ontology)
    manifest_ontology = seg_manifest.get("ontology", {})
    if manifest_ontology.get("class_names") != list(
        ontology.class_names
    ) or manifest_ontology.get("class_types") != list(ontology.class_types):
        raise ValueError(
            "Ontology mismatch between stage 01 manifest and --ontology "
            "(class names or thing/stuff types differ); "
            "re-run segmentation or pass the matching config"
        )

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    vertex_count = int(gaussians.get_xyz.shape[0])
    num_classes = ontology.num_project_classes
    votes = np.zeros((vertex_count, num_classes), dtype=np.float32)
    views_supporting = np.zeros((vertex_count, num_classes), dtype=np.uint16)
    visible_views = np.zeros((vertex_count,), dtype=np.uint16)

    frame_records: list[dict[str, Any]] = []
    started = time.time()
    with torch.no_grad():
        for frame in seg_manifest["frames"]:
            camera_index = int(frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            height = int(camera.image_height)
            width = int(camera.image_width)

            project_class, confidence = load_seg_arrays(args.seg_dir / "seg" / frame["seg_file"])
            project_class, confidence = resize_to(project_class, confidence, height, width)
            project_class = np.where(
                confidence >= args.min_confidence, project_class, np.int16(0)
            )

            present = np.unique(project_class)
            present = present[present > 0]
            mean_confidence = class_mean_confidence(project_class, confidence, present)

            frame_visible = np.zeros((vertex_count,), dtype=bool)
            voted_classes: list[int] = []

            if present.shape[0] == 0:
                # No class survived confidence gating, but Gaussians rendered in
                # this view must still count as visible (they vote for nothing),
                # otherwise min-visible-ratio is biased permissive. A zero index
                # mask puts every contribution into used_count[0].
                gt_mask = torch.zeros((height, width), dtype=torch.float32, device="cuda")
                render_pkg = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    gt_mask=gt_mask,
                    obj_num=1,
                )
                used_count = render_pkg["used_count"].detach().cpu().numpy()
                frame_visible |= (used_count > args.support_threshold).any(axis=0)
                del render_pkg
                del gt_mask
                del used_count
                torch.cuda.empty_cache()

            for start in range(0, present.shape[0], args.class_batch_size):
                chunk = present[start : start + args.class_batch_size]
                # Local index mask: chunk class -> 1..k, everything else 0.
                local_map = np.zeros((height, width), dtype=np.float32)
                for local_id, class_id in enumerate(chunk, start=1):
                    local_map[project_class == class_id] = float(local_id)
                gt_mask = torch.from_numpy(local_map).to(device="cuda")

                render_pkg = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    gt_mask=gt_mask,
                    obj_num=int(chunk.shape[0]) + 1,
                )
                used_count = render_pkg["used_count"].detach().cpu().numpy()
                del render_pkg
                del gt_mask

                frame_visible |= (used_count > args.support_threshold).any(axis=0)
                for local_id, class_id in enumerate(chunk, start=1):
                    counts = used_count[local_id]
                    supported = counts > args.support_threshold
                    if not supported.any():
                        continue
                    column = int(class_id) - 1
                    weight = mean_confidence[int(class_id)]
                    votes[supported, column] += counts[supported] * weight
                    views_supporting[supported, column] += np.uint16(1)
                    voted_classes.append(int(class_id))
                del used_count
                torch.cuda.empty_cache()

            visible_views[frame_visible] += np.uint16(1)
            frame_records.append(
                {
                    "file": frame["file"],
                    "camera_index": camera_index,
                    "classes_present": [ontology.class_names[int(c)] for c in present],
                    "classes_voted": [ontology.class_names[c] for c in sorted(set(voted_classes))],
                    "visible_gaussians": int(frame_visible.sum()),
                }
            )
            print(
                f"lift {frame['file']}: classes={present.shape[0]} "
                f"visible={int(frame_visible.sum())}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    votes_path = args.output_dir / "votes.npz"
    np.savez_compressed(
        votes_path,
        votes=votes,
        views_supporting=views_supporting,
        visible_views=visible_views,
        class_names=np.asarray(ontology.class_names),
        class_types=np.asarray(ontology.class_types),
    )

    lift_manifest = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "seg_dir": str(args.seg_dir),
        "ontology": ontology.describe(),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "min_confidence": args.min_confidence,
        "support_threshold": args.support_threshold,
        "class_batch_size": args.class_batch_size,
        "view_count": len(frame_records),
        "vertex_count": vertex_count,
        "gaussians_with_any_vote": int((votes.sum(axis=1) > 0).sum()),
        "gaussians_visible_anywhere": int((visible_views > 0).sum()),
        "elapsed_seconds": time.time() - started,
        "frames": frame_records,
    }
    manifest_out = args.output_dir / "vote_lift_manifest.json"
    manifest_out.write_text(json.dumps(lift_manifest, indent=2), encoding="utf-8")
    print(f"wrote {votes_path}")
    print(f"wrote {manifest_out}")


if __name__ == "__main__":
    main()
