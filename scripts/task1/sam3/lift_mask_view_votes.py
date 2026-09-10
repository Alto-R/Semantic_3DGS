#!/usr/bin/env python3
"""S2 driver: lift SAM3 instance masks into per-view Gaussian votes.

One FlashSplat pass per (view, concept): instances of a concept share an
index map, while overlapping concepts get independent passes, preserving
multi-label structure. This module is importable without torch; the CUDA
work lives inside ``main``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov3.lift_dense_view_votes import flashsplat_class_rows
from scripts.task1.sam3.lift_mask_votes_core import (
    concat_or_empty,
    concept_index_map,
    mask_membership_votes,
    observed_gaussians,
)
from scripts.task1.sam3.segment_views_core import (
    stem_index,
    validate_masks_manifest,
)


VOTES_SOURCE = "sam3_mask_flashsplat_votes"
VOTES_CONTRACT = "per_concept_pass_membership_votes_v1"


def validate_votes_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("source") != VOTES_SOURCE:
        raise ValueError(f"unsupported votes source: {manifest.get('source')!r}")
    if manifest.get("contract") != VOTES_CONTRACT:
        raise ValueError(
            f"unsupported votes contract: {manifest.get('contract')!r}"
        )
    if int(manifest.get("gaussian_count", 0)) < 1:
        raise ValueError("votes manifest must record a positive gaussian count")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("votes manifest has no camera frames")
    if int(manifest.get("camera_count", -1)) != len(frames):
        raise ValueError("votes manifest camera count differs from its frames")


def group_masks_by_concept(
    frame_masks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Preserve mask order within each concept group."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for mask in frame_masks:
        groups.setdefault(str(mask["concept"]), []).append(mask)
    return groups


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--masks-manifest", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--render-manifest", required=True, type=Path,
                        help="render-stage manifest mapping view files to camera indices")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", type=Path, default=None)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    # Heavy cluster-only imports come after argument parsing so --help and
    # argument errors work in any environment.
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

    masks_manifest = json.loads(args.masks_manifest.read_text(encoding="utf-8"))
    validate_masks_manifest(masks_manifest)
    render_manifest = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    camera_index_by_stem = {
        stem: int(frame["camera_index"])
        for stem, frame in stem_index(
            render_manifest["frames"], "render manifest"
        ).items()
    }

    vote_dir = args.output_dir / "view_votes"
    manifest_path = args.output_dir / "sam3_vote_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(manifest_path)
    vote_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    frames: list[dict[str, Any]] = []

    with torch.no_grad():
        for frame in masks_manifest["frames"]:
            stem = Path(str(frame["file"])).stem
            if stem not in camera_index_by_stem:
                raise ValueError(f"render manifest does not know view {stem}")
            camera = make_camera(
                cameras[camera_index_by_stem[stem]], modules, args.max_width
            )
            with np.load(args.masks_dir / str(frame["mask_file"])) as data:
                mask_stack = data["mask_stack"]

            concept_groups = group_masks_by_concept(frame["masks"])
            visibility: np.ndarray | None = None
            all_indices: list[np.ndarray] = []
            all_mask_ids: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []
            for masks in concept_groups.values():
                rows = [int(mask["mask_index"]) for mask in masks]
                scores = np.array([float(mask["score"]) for mask in masks])
                index_map = concept_index_map(mask_stack[rows], scores)
                gt_mask = torch.from_numpy(index_map).to(
                    device="cuda", dtype=torch.float32
                )
                render_pkg = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    gt_mask=gt_mask,
                    obj_num=len(rows) + 1,
                )
                used_count = flashsplat_class_rows(
                    render_pkg["used_count"].detach().float().cpu().numpy(),
                    len(rows) + 1,
                    gaussian_count,
                )
                if visibility is None:
                    visibility = used_count.sum(axis=0, dtype=np.float32)
                indices, mask_ids, weights = mask_membership_votes(
                    used_count,
                    visibility,
                    np.array(rows, dtype=np.uint16),
                )
                all_indices.append(indices)
                all_mask_ids.append(mask_ids)
                all_weights.append(weights)
                del render_pkg, gt_mask
                torch.cuda.empty_cache()

            if visibility is None:
                visibility = np.zeros(gaussian_count, dtype=np.float32)
            observed = observed_gaussians(visibility)
            vote_path = vote_dir / f"{stem}.npz"
            if vote_path.exists() and not args.overwrite:
                raise FileExistsError(vote_path)
            np.savez_compressed(
                vote_path,
                indices=concat_or_empty(all_indices, np.uint32),
                mask_ids=concat_or_empty(all_mask_ids, np.uint16),
                weights=concat_or_empty(all_weights, np.float32),
                observed=observed,
            )
            frames.append(
                {
                    "file": str(frame["file"]),
                    "camera_index": camera_index_by_stem[stem],
                    "vote_file": vote_path.relative_to(args.output_dir).as_posix(),
                    "concept_passes": len(concept_groups),
                    "observed_gaussian_count": int(observed.shape[0]),
                }
            )
            print(f"lifted {stem}: {len(concept_groups)} concept passes")

    manifest = {
        "source": VOTES_SOURCE,
        "contract": VOTES_CONTRACT,
        "masks_manifest": str(args.masks_manifest),
        "masks_source": masks_manifest["source"],
        "masks_contract": masks_manifest["contract"],
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "render_max_width": args.max_width,
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "vote_formula": "used_count_for_mask/total_view_visibility",
        "sum_to_one_constraint": False,
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
