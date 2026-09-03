#!/usr/bin/env python3
"""Lift complete dense DINOv3 ADE20K maps into per-view Gaussian votes.

Every rendered pixel keeps its DINOv3 Mask2Former argmax class. FlashSplat
turns the dense class image into fractional class support at each Gaussian.
There is no query-region filtering, confidence threshold, or semantic prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scripts.task1.dinov2.dinov2_ontology import load_ontology


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"


def local_index_map(project_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map present project classes to compact FlashSplat row indices."""

    present = np.unique(np.asarray(project_ids))
    present = present[present != 0]
    class_ids = np.concatenate(
        [np.zeros((1,), dtype=np.uint16), present.astype(np.uint16, copy=False)]
    )
    indexed = np.searchsorted(class_ids, project_ids).astype(np.float32)
    return indexed, class_ids


def flashsplat_class_rows(
    used_count: np.ndarray,
    class_count: int,
    gaussian_count: int,
) -> np.ndarray:
    """Drop FlashSplat's optional final zero sentinel row."""

    used = np.asarray(used_count, dtype=np.float32)
    expected = (class_count, gaussian_count)
    if used.shape == expected:
        return used
    sentinel = (class_count + 1, gaussian_count)
    if used.shape == sentinel:
        if np.count_nonzero(used[-1]) != 0:
            raise ValueError("FlashSplat sentinel row contains nonzero support")
        return used[:-1]
    raise ValueError(
        f"Unexpected FlashSplat support shape {used.shape}; expected "
        f"{expected} or {sentinel}"
    )


def dense_sparse_view_votes(
    used_count: np.ndarray,
    project_class_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complete normalized class mass for every visible Gaussian."""

    used = np.asarray(used_count, dtype=np.float32)
    class_ids = np.asarray(project_class_ids, dtype=np.uint16)
    if used.ndim != 2:
        raise ValueError("used_count must have shape classes x gaussians")
    if class_ids.shape != (used.shape[0],):
        raise ValueError("project_class_ids must match the class axis")
    if class_ids.shape[0] == 0 or class_ids[0] != 0:
        raise ValueError("the compact class map must start with row zero")
    if not np.isfinite(used).all() or np.any(used < 0.0):
        raise ValueError("FlashSplat support must be finite and non-negative")

    visibility = used.sum(axis=0, dtype=np.float32)
    all_indices: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for local_id, project_id in enumerate(class_ids):
        if int(project_id) == 0:
            continue
        supported = (used[local_id] > 0.0) & (visibility > 0.0)
        indices = np.flatnonzero(supported).astype(np.uint32)
        if indices.size == 0:
            continue
        weights = used[local_id, indices] / visibility[indices]
        positive = weights > 0.0
        indices = indices[positive]
        weights = weights[positive]
        if indices.size == 0:
            continue
        all_indices.append(indices)
        all_classes.append(
            np.full(indices.shape, int(project_id), dtype=np.uint16)
        )
        all_weights.append(weights.astype(np.float32, copy=False))

    if not all_indices:
        return (
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.uint16),
            np.zeros((0,), dtype=np.float32),
        )
    indices = np.concatenate(all_indices)
    classes = np.concatenate(all_classes)
    weights = np.concatenate(all_weights)

    totals = np.zeros((used.shape[1],), dtype=np.float32)
    np.add.at(totals, indices.astype(np.int64, copy=False), weights)
    observed = visibility > 0.0
    if observed.any() and not np.allclose(
        totals[observed],
        np.ones(int(np.count_nonzero(observed)), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError("dense per-view class mass does not sum to one")
    return indices, classes, weights


def validate_dinov3_manifest(manifest: dict[str, Any]) -> None:
    source = str(manifest.get("source", ""))
    contract = str(manifest.get("contract", ""))
    if source != "dinov3_vit7b16_ade20k_mask2former":
        raise ValueError(f"unsupported dense semantic source: {source!r}")
    if contract != "raw_ade20k_class_and_relative_margin_confidence_v2":
        raise ValueError(f"unsupported dense semantic contract: {contract!r}")
    if float(manifest.get("min_pixel_confidence", -1.0)) != 0.0:
        raise ValueError("coverage-complete lifting requires min_pixel_confidence=0")


def main() -> None:
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--segmentation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.segmentation_manifest.is_file():
        raise FileNotFoundError(args.segmentation_manifest)
    manifest = json.loads(args.segmentation_manifest.read_text(encoding="utf-8"))
    validate_dinov3_manifest(manifest)
    ontology = load_ontology(args.ontology)
    lookup = ontology.ade_to_project

    vote_dir = args.output_dir / "view_votes"
    output_manifest_path = args.output_dir / "vote_manifest.json"
    if output_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(output_manifest_path)
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
        for frame in manifest["frames"]:
            filename = str(frame["file"])
            camera_index = int(frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            segment_path = args.input_dir / str(frame["segment_file"])
            with np.load(segment_path) as segment:
                raw_class = segment["class_id"].astype(np.uint8, copy=False)
            expected_shape = (int(camera.image_height), int(camera.image_width))
            if raw_class.shape != expected_shape:
                raise ValueError(
                    f"{segment_path} has shape {raw_class.shape}; expected {expected_shape}"
                )
            if int(raw_class.max(initial=0)) >= ontology.class_count:
                raise ValueError(f"{segment_path} contains an ADE20K class out of range")

            project_ids = lookup[raw_class]
            if np.any(project_ids == 0):
                raise ValueError(
                    "a dense DINOv3 class map unexpectedly contains unlabeled pixels"
                )
            indexed, class_ids = local_index_map(project_ids)
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
            indices, vote_classes, weights = dense_sparse_view_votes(
                used_count,
                class_ids,
            )
            vote_path = vote_dir / f"{Path(filename).stem}.npz"
            if vote_path.exists() and not args.overwrite:
                raise FileExistsError(vote_path)
            np.savez_compressed(
                vote_path,
                indices=indices,
                class_ids=vote_classes,
                weights=weights,
            )
            visible = np.unique(indices).shape[0]
            frames.append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(frame["camera_id"]),
                    "vote_file": vote_path.relative_to(args.output_dir).as_posix(),
                    "present_project_class_ids": [
                        int(value) for value in class_ids[1:]
                    ],
                    "visible_gaussian_count": int(visible),
                    "sparse_vote_count": int(weights.shape[0]),
                }
            )
            print(
                f"lifted {filename}: visible_gaussians={visible} "
                f"class_votes={weights.shape[0]}"
            )
            del render_pkg, used_count, gt_mask
            torch.cuda.empty_cache()

    output_manifest = {
        "source": "dinov3_dense_pixel_flashsplat_votes",
        "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
        "segmentation_source": manifest["source"],
        "segmentation_contract": manifest["contract"],
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "segmentation_manifest": str(args.segmentation_manifest),
        "ontology": str(args.ontology),
        "iteration": args.iteration,
        "render_max_width": args.max_width,
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "pixel_filtering": "none",
        "query_region_filtering_used": False,
        "confidence_threshold_used": False,
        "vote_formula": "used_count_for_class/sum_used_count_over_all_classes",
        "one_normalized_vote_per_camera": True,
        "v5_used": False,
        "dinov2_used": False,
        "frames": frames,
    }
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {output_manifest_path}")


if __name__ == "__main__":
    main()
