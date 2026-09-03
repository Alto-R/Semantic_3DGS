#!/usr/bin/env python3
"""Lift confidence-filtered dense DINOv3 pixels into Gaussian evidence.

The cached DINOv3 segmentation stores an ADE20K argmax and three calibration
maps for every pixel.  This report-only lifter applies one global conjunctive
confidence profile, maps rejected pixels to an explicit abstain row, and uses
FlashSplat to measure class and abstain mass at every visible Gaussian.

Unlike the coverage-complete dense lifter, semantic weights are normalized by
all visible mass, including abstentions.  A small confident pixel fragment
therefore cannot become a full camera vote after low-confidence pixels have
been removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.lift_dense_view_votes import (
    DEFAULT_FLASHSPLAT_ROOT,
    DEFAULT_ONTOLOGY,
    flashsplat_class_rows,
    local_index_map,
    validate_dinov3_manifest,
)


SOURCE = "dinov3_confident_dense_pixel_flashsplat_votes"
CONTRACT = "confident_dense_argmax_pixels_with_explicit_abstain_mass_v1"


def confident_sparse_view_votes(
    used_count: np.ndarray,
    project_class_ids: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return semantic mass plus total accepted fraction for each Gaussian."""

    used = np.asarray(used_count, dtype=np.float32)
    class_ids = np.asarray(project_class_ids, dtype=np.uint16)
    if used.ndim != 2:
        raise ValueError("used_count must have shape classes x gaussians")
    if class_ids.shape != (used.shape[0],):
        raise ValueError("project_class_ids must match the class axis")
    if class_ids.size == 0 or int(class_ids[0]) != 0:
        raise ValueError("the compact class map must start with abstain row zero")
    if not np.isfinite(used).all() or np.any(used < 0.0):
        raise ValueError("FlashSplat support must be finite and non-negative")

    visibility = used.sum(axis=0, dtype=np.float32)
    visible_indices = np.flatnonzero(visibility > 0.0).astype(np.uint32)
    semantic_mass = used[1:].sum(axis=0, dtype=np.float32)
    accepted_fraction = np.divide(
        semantic_mass[visible_indices],
        visibility[visible_indices],
        out=np.zeros(visible_indices.shape, dtype=np.float32),
        where=visibility[visible_indices] > 0.0,
    )

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

    if all_indices:
        indices = np.concatenate(all_indices)
        classes = np.concatenate(all_classes)
        weights = np.concatenate(all_weights)
    else:
        indices = np.zeros((0,), dtype=np.uint32)
        classes = np.zeros((0,), dtype=np.uint16)
        weights = np.zeros((0,), dtype=np.float32)

    totals = np.zeros((used.shape[1],), dtype=np.float32)
    np.add.at(totals, indices.astype(np.int64, copy=False), weights)
    if visible_indices.size and not np.allclose(
        totals[visible_indices],
        accepted_fraction,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError("semantic mass does not equal the accepted fraction")
    if np.any(accepted_fraction < 0.0) or np.any(accepted_fraction > 1.0 + 1e-5):
        raise RuntimeError("accepted fraction is outside [0, 1]")
    return (
        indices,
        classes,
        weights,
        visible_indices,
        accepted_fraction.astype(np.float32, copy=False),
    )


def confidence_keep_mask(
    segment: Any,
    *,
    min_relative_margin: float,
    min_entropy_confidence: float,
    min_max_probability: float,
) -> np.ndarray:
    """Apply one class-neutral conjunctive pixel-confidence profile."""

    required = (
        "class_id",
        "confidence",
        "max_softmax_probability",
        "normalized_entropy_confidence",
    )
    missing = [key for key in required if key not in segment]
    if missing:
        raise ValueError(f"cached DINOv3 segment lacks fields: {missing}")
    shape = np.asarray(segment["class_id"]).shape
    relative = np.asarray(segment["confidence"], dtype=np.float32)
    entropy_confidence = np.asarray(
        segment["normalized_entropy_confidence"],
        dtype=np.float32,
    )
    max_probability = np.asarray(
        segment["max_softmax_probability"],
        dtype=np.float32,
    )
    for name, values in (
        ("confidence", relative),
        ("normalized_entropy_confidence", entropy_confidence),
        ("max_softmax_probability", max_probability),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} shape differs from class_id")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{name} is outside [0, 1]")
    return (
        (relative >= min_relative_margin)
        & (entropy_confidence >= min_entropy_confidence)
        & (max_probability >= min_max_probability)
    )


def _validate_threshold(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


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
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--min-relative-margin", required=True, type=float)
    parser.add_argument("--min-entropy-confidence", required=True, type=float)
    parser.add_argument("--min-max-probability", required=True, type=float)
    parser.add_argument(
        "--flashsplat-root",
        default=DEFAULT_FLASHSPLAT_ROOT,
        type=Path,
    )
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    for value, name in (
        (args.min_relative_margin, "min_relative_margin"),
        (args.min_entropy_confidence, "min_entropy_confidence"),
        (args.min_max_probability, "min_max_probability"),
    ):
        _validate_threshold(value, name)
    if not args.profile_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in args.profile_name
    ):
        raise ValueError("profile_name must use lowercase letters, digits, or _")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.segmentation_manifest.is_file():
        raise FileNotFoundError(args.segmentation_manifest)

    manifest = json.loads(args.segmentation_manifest.read_text(encoding="utf-8"))
    validate_dinov3_manifest(manifest)
    ontology = load_ontology(args.ontology)
    lookup = ontology.ade_to_project

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    vote_dir = args.output_dir / "view_votes"
    vote_dir.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    total_pixels = 0
    total_kept_pixels = 0

    with torch.no_grad():
        for frame in manifest["frames"]:
            filename = str(frame["file"])
            camera_index = int(frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            segment_path = args.input_dir / str(frame["segment_file"])
            with np.load(segment_path, allow_pickle=False) as segment:
                raw_class = segment["class_id"].astype(np.uint8, copy=False)
                keep = confidence_keep_mask(
                    segment,
                    min_relative_margin=args.min_relative_margin,
                    min_entropy_confidence=args.min_entropy_confidence,
                    min_max_probability=args.min_max_probability,
                )
            expected_shape = (int(camera.image_height), int(camera.image_width))
            if raw_class.shape != expected_shape:
                raise ValueError(
                    f"{segment_path} has shape {raw_class.shape}; "
                    f"expected {expected_shape}"
                )
            if int(raw_class.max(initial=0)) >= ontology.class_count:
                raise ValueError(f"{segment_path} contains an ADE class out of range")

            project_ids = lookup[raw_class].copy()
            if np.any(project_ids == 0):
                raise ValueError("dense DINOv3 class map contains unmapped pixels")
            project_ids[~keep] = 0
            indexed, class_ids = local_index_map(project_ids)
            gt_mask = torch.from_numpy(indexed).to(
                device="cuda",
                dtype=torch.float32,
            )
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
            (
                indices,
                vote_classes,
                weights,
                visible_indices,
                accepted_fractions,
            ) = confident_sparse_view_votes(used_count, class_ids)

            vote_path = vote_dir / f"{Path(filename).stem}.npz"
            np.savez_compressed(
                vote_path,
                indices=indices,
                class_ids=vote_classes,
                weights=weights,
                visible_indices=visible_indices,
                accepted_fractions=accepted_fractions,
            )
            pixel_count = int(keep.size)
            kept_pixel_count = int(np.count_nonzero(keep))
            total_pixels += pixel_count
            total_kept_pixels += kept_pixel_count
            frames.append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(frame["camera_id"]),
                    "vote_file": vote_path.relative_to(args.output_dir).as_posix(),
                    "pixel_count": pixel_count,
                    "kept_pixel_count": kept_pixel_count,
                    "kept_pixel_ratio": kept_pixel_count / float(pixel_count),
                    "visible_gaussian_count": int(visible_indices.size),
                    "gaussian_with_semantic_mass_count": int(
                        np.count_nonzero(accepted_fractions > 0.0)
                    ),
                    "sparse_vote_count": int(weights.size),
                }
            )
            print(
                f"lifted {filename}: kept_pixels={kept_pixel_count}/{pixel_count} "
                f"visible_gaussians={visible_indices.size}"
            )
            del render_pkg, used_count, gt_mask
            torch.cuda.empty_cache()

    thresholds = {
        "min_relative_margin": args.min_relative_margin,
        "min_entropy_confidence": args.min_entropy_confidence,
        "min_max_probability": args.min_max_probability,
    }
    output_manifest = {
        "source": SOURCE,
        "contract": CONTRACT,
        "profile_name": args.profile_name,
        "thresholds": thresholds,
        "segmentation_source": manifest["source"],
        "segmentation_contract": manifest["contract"],
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "segmentation_manifest": str(args.segmentation_manifest),
        "ontology": str(args.ontology),
        "iteration": args.iteration,
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "pixel_filtering": (
            "relative_margin_and_entropy_confidence_and_max_probability"
        ),
        "query_region_filtering_used": False,
        "confidence_threshold_used": True,
        "inference_rerun": False,
        "flashsplat_rerun": True,
        "vote_formula": (
            "used_count_for_class/sum_used_count_including_abstain"
        ),
        "abstain_mass_preserved": True,
        "one_normalized_visibility_budget_per_camera": True,
        "total_pixel_count": total_pixels,
        "kept_pixel_count": total_kept_pixels,
        "kept_pixel_ratio": total_kept_pixels / float(max(total_pixels, 1)),
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "manual_component_decisions": False,
        "v5_used": False,
        "dinov2_used": False,
        "frames": frames,
    }
    (args.output_dir / "vote_manifest.json").write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "profile_name": args.profile_name,
        "camera_count": len(frames),
        "kept_pixel_ratio": output_manifest["kept_pixel_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
