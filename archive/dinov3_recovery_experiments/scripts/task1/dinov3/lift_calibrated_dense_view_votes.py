#!/usr/bin/env python3
"""Lift all joint-calibrated DINOv3 confidence profiles in one render.

Each pixel is assigned to one nested confidence level using the joint
empirical calibration report.  The FlashSplat object ID encodes both project
class and confidence level, allowing baseline, permissive, balanced, and
strict Gaussian evidence to be reconstructed from one rasterization per
camera.  Mass below a profile's level remains explicit abstention mass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.calibrate_dense_confidence import (
    CONTRACT as CALIBRATION_CONTRACT,
    METRICS,
    SOURCE as CALIBRATION_SOURCE,
    file_sha256,
    weakest_empirical_rank_bins,
)
from scripts.task1.dinov3.lift_confident_dense_view_votes import (
    SOURCE,
    confident_sparse_view_votes,
)
from scripts.task1.dinov3.lift_dense_view_votes import (
    DEFAULT_FLASHSPLAT_ROOT,
    DEFAULT_ONTOLOGY,
    flashsplat_class_rows,
    validate_dinov3_manifest,
)


CONTRACT = "joint_calibrated_nested_profiles_explicit_abstain_mass_v2"
EXPECTED_PROFILE_NAMES = ("baseline", "permissive", "balanced", "strict")


def pair_index_map(
    project_ids: np.ndarray,
    confidence_levels: np.ndarray,
    *,
    project_class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode every present confidence-level/project-class pair compactly."""

    projects = np.asarray(project_ids, dtype=np.uint16)
    levels = np.asarray(confidence_levels, dtype=np.uint8)
    if projects.shape != levels.shape:
        raise ValueError("project IDs and confidence levels must align")
    if np.any(projects == 0) or int(projects.max(initial=0)) > project_class_count:
        raise ValueError("project IDs must be mapped nonzero ontology IDs")
    if int(levels.max(initial=0)) >= len(EXPECTED_PROFILE_NAMES):
        raise ValueError("confidence level is outside the profile contract")

    stride = project_class_count + 1
    keys = levels.astype(np.uint32) * np.uint32(stride) + projects
    present_keys = np.unique(keys)
    indexed = (
        np.searchsorted(present_keys, keys).astype(np.float32)
        + np.float32(1.0)
    )
    row_project_ids = np.zeros((present_keys.size + 1,), dtype=np.uint16)
    row_levels = np.zeros((present_keys.size + 1,), dtype=np.uint8)
    row_project_ids[1:] = (present_keys % np.uint32(stride)).astype(np.uint16)
    row_levels[1:] = (present_keys // np.uint32(stride)).astype(np.uint8)
    return indexed, row_project_ids, row_levels


def confidence_levels_from_bins(
    score_bins: np.ndarray,
    profiles: list[dict[str, Any]],
) -> np.ndarray:
    if tuple(str(item["profile_name"]) for item in profiles) != (
        EXPECTED_PROFILE_NAMES
    ):
        raise ValueError("calibration profiles have unexpected names or order")
    minimum_bins = [
        int(item["minimum_weakest_rank_bin"]) for item in profiles
    ]
    if minimum_bins[0] != 0 or minimum_bins != sorted(minimum_bins):
        raise ValueError("calibration profile bins are not nested from baseline")
    levels = np.zeros(np.asarray(score_bins).shape, dtype=np.uint8)
    for level, minimum_bin in enumerate(minimum_bins[1:], start=1):
        levels[np.asarray(score_bins) >= minimum_bin] = np.uint8(level)
    return levels


def profile_used_count(
    used_count: np.ndarray,
    row_project_ids: np.ndarray,
    row_levels: np.ndarray,
    *,
    minimum_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate encoded pair rows into one class matrix plus abstention."""

    used = np.asarray(used_count, dtype=np.float32)
    projects = np.asarray(row_project_ids, dtype=np.uint16)
    levels = np.asarray(row_levels, dtype=np.uint8)
    if used.ndim != 2 or projects.shape != (used.shape[0],):
        raise ValueError("encoded FlashSplat rows do not match project IDs")
    if levels.shape != projects.shape:
        raise ValueError("encoded FlashSplat rows do not match confidence levels")
    if minimum_level < 0 or minimum_level >= len(EXPECTED_PROFILE_NAMES):
        raise ValueError("minimum profile level is invalid")

    semantic_rows = (projects > 0) & (levels >= minimum_level)
    present_classes = np.unique(projects[semantic_rows])
    class_ids = np.concatenate(
        [
            np.zeros((1,), dtype=np.uint16),
            present_classes.astype(np.uint16, copy=False),
        ]
    )
    aggregated = np.zeros(
        (class_ids.size, used.shape[1]),
        dtype=np.float32,
    )
    total_visibility = used.sum(axis=0, dtype=np.float32)
    for local_id, project_id in enumerate(class_ids[1:], start=1):
        rows = np.flatnonzero(
            (projects == project_id) & (levels >= minimum_level)
        )
        aggregated[local_id] = used[rows].sum(axis=0, dtype=np.float32)
    semantic_mass = aggregated[1:].sum(axis=0, dtype=np.float32)
    abstain = total_visibility - semantic_mass
    if np.any(abstain < -1e-4):
        raise RuntimeError("profile semantic mass exceeds total visibility")
    aggregated[0] = np.maximum(abstain, 0.0)
    return aggregated, class_ids


def load_calibration(
    manifest_path: Path,
    npz_path: Path,
    segmentation_manifest: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != CALIBRATION_SOURCE:
        raise ValueError("calibration manifest has the wrong source")
    if manifest.get("contract") != CALIBRATION_CONTRACT:
        raise ValueError("calibration manifest has the wrong contract")
    if manifest.get("report_only") is not True:
        raise ValueError("calibration manifest is not report-only")
    for field in (
        "inference_rerun",
        "flashsplat_rerun",
        "scene_specific_rules",
        "class_specific_thresholds",
        "manual_component_decisions",
        "semantic_labels_written",
        "semantic_project_class_arrays_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"calibration manifest violates {field}=false")
    calibrated_sources = {
        Path(str(item["segmentation_manifest"])).resolve()
        for item in manifest.get("sources", [])
    }
    if segmentation_manifest.resolve() not in calibrated_sources:
        raise ValueError("segmentation manifest was not part of joint calibration")
    if file_sha256(npz_path) != str(
        manifest.get("calibration_npz_sha256", "")
    ):
        raise ValueError("calibration NPZ hash differs from its manifest")
    bin_count = int(manifest["bin_count"])
    with np.load(npz_path, allow_pickle=False) as archive:
        cdfs = {
            metric_name: np.asarray(
                archive[f"{metric_name}_cdf"],
                dtype=np.float32,
            )
            for metric_name in METRICS
        }
    if any(values.shape != (bin_count,) for values in cdfs.values()):
        raise ValueError("calibration CDF arrays have the wrong shape")
    for metric_name, values in cdfs.items():
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or np.any(np.diff(values) < 0.0)
            or not np.isclose(values[-1], 1.0)
        ):
            raise ValueError(f"calibration CDF is invalid for {metric_name}")
    profiles = manifest.get("profiles", [])
    if tuple(str(item.get("profile_name", "")) for item in profiles) != (
        EXPECTED_PROFILE_NAMES
    ):
        raise ValueError("calibration profiles have unexpected names or order")
    minimum_bins = [
        int(item["minimum_weakest_rank_bin"]) for item in profiles
    ]
    if (
        minimum_bins[0] != 0
        or minimum_bins != sorted(minimum_bins)
        or minimum_bins[-1] >= bin_count
    ):
        raise ValueError("calibration profile bins are invalid")
    return manifest, cdfs


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
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--calibration-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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

    for path in (
        args.segmentation_manifest,
        args.calibration_manifest,
        args.calibration_npz,
        args.ontology,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    segmentation = json.loads(
        args.segmentation_manifest.read_text(encoding="utf-8")
    )
    validate_dinov3_manifest(segmentation)
    calibration, metric_cdfs = load_calibration(
        args.calibration_manifest,
        args.calibration_npz,
        args.segmentation_manifest,
    )
    profiles = list(calibration["profiles"])
    bin_count = int(calibration["bin_count"])
    ontology = load_ontology(args.ontology)
    lookup = ontology.ade_to_project

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    profile_roots: dict[str, Path] = {}
    profile_frames: dict[str, list[dict[str, Any]]] = {}
    profile_kept_pixels: dict[str, int] = {}
    for profile in profiles:
        profile_name = str(profile["profile_name"])
        profile_root = args.output_dir / profile_name
        (profile_root / "view_votes").mkdir(parents=True, exist_ok=False)
        profile_roots[profile_name] = profile_root
        profile_frames[profile_name] = []
        profile_kept_pixels[profile_name] = 0

    total_pixels = 0
    with torch.no_grad():
        for frame in segmentation["frames"]:
            filename = str(frame["file"])
            camera_index = int(frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            segment_path = args.input_dir / str(frame["segment_file"])
            with np.load(segment_path, allow_pickle=False) as segment:
                raw_class = segment["class_id"].astype(np.uint8, copy=False)
                score_bins = weakest_empirical_rank_bins(
                    segment,
                    metric_cdfs,
                    bin_count,
                )
            expected_shape = (int(camera.image_height), int(camera.image_width))
            if raw_class.shape != expected_shape:
                raise ValueError(
                    f"{segment_path} has shape {raw_class.shape}; "
                    f"expected {expected_shape}"
                )
            if int(raw_class.max(initial=0)) >= ontology.class_count:
                raise ValueError(f"{segment_path} contains an ADE class out of range")
            project_ids = lookup[raw_class]
            if np.any(project_ids == 0):
                raise ValueError("dense DINOv3 class map contains unmapped pixels")
            levels = confidence_levels_from_bins(score_bins, profiles)
            indexed, row_projects, row_levels = pair_index_map(
                project_ids,
                levels,
                project_class_count=ontology.class_count,
            )

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
                obj_num=int(row_projects.size),
            )
            used_count = flashsplat_class_rows(
                render_pkg["used_count"].detach().float().cpu().numpy(),
                int(row_projects.size),
                gaussian_count,
            )

            pixel_count = int(levels.size)
            total_pixels += pixel_count
            for minimum_level, profile in enumerate(profiles):
                profile_name = str(profile["profile_name"])
                aggregated, class_ids = profile_used_count(
                    used_count,
                    row_projects,
                    row_levels,
                    minimum_level=minimum_level,
                )
                (
                    indices,
                    vote_classes,
                    weights,
                    visible_indices,
                    accepted_fractions,
                ) = confident_sparse_view_votes(aggregated, class_ids)
                vote_path = (
                    profile_roots[profile_name]
                    / "view_votes"
                    / f"{Path(filename).stem}.npz"
                )
                np.savez_compressed(
                    vote_path,
                    indices=indices,
                    class_ids=vote_classes,
                    weights=weights,
                    visible_indices=visible_indices,
                    accepted_fractions=accepted_fractions,
                )
                minimum_bin = int(profile["minimum_weakest_rank_bin"])
                kept_pixel_count = int(np.count_nonzero(score_bins >= minimum_bin))
                profile_kept_pixels[profile_name] += kept_pixel_count
                profile_frames[profile_name].append(
                    {
                        "file": filename,
                        "camera_index": camera_index,
                        "camera_id": int(frame["camera_id"]),
                        "vote_file": vote_path.relative_to(
                            profile_roots[profile_name]
                        ).as_posix(),
                        "pixel_count": pixel_count,
                        "kept_pixel_count": kept_pixel_count,
                        "kept_pixel_ratio": (
                            kept_pixel_count / float(pixel_count)
                        ),
                        "visible_gaussian_count": int(visible_indices.size),
                        "gaussian_with_semantic_mass_count": int(
                            np.count_nonzero(accepted_fractions > 0.0)
                        ),
                        "sparse_vote_count": int(weights.size),
                    }
                )
                del aggregated
            print(
                f"lifted calibrated profiles for {filename}: "
                f"encoded_rows={row_projects.size}"
            )
            del render_pkg, used_count, gt_mask
            torch.cuda.empty_cache()

    for profile_level, profile in enumerate(profiles):
        profile_name = str(profile["profile_name"])
        profile_root = profile_roots[profile_name]
        kept_pixel_count = profile_kept_pixels[profile_name]
        actual_joint_ratio = float(profile["actual_joint_retained_ratio"])
        manifest = {
            "source": SOURCE,
            "contract": CONTRACT,
            "profile_name": profile_name,
            "profile_level": profile_level,
            "thresholds": {
                "minimum_weakest_rank_bin": int(
                    profile["minimum_weakest_rank_bin"]
                ),
                "minimum_weakest_rank": float(
                    profile["minimum_weakest_rank"]
                ),
                "target_joint_retained_ratio": float(
                    profile["target_joint_retained_ratio"]
                ),
                "actual_joint_retained_ratio": actual_joint_ratio,
            },
            "calibration_source": CALIBRATION_SOURCE,
            "calibration_contract": CALIBRATION_CONTRACT,
            "calibration_manifest": str(args.calibration_manifest),
            "calibration_npz": str(args.calibration_npz),
            "joint_calibration_used": True,
            "model_path": str(args.model_path),
            "ply_path": str(ply_path),
            "segmentation_manifest": str(args.segmentation_manifest),
            "ontology": str(args.ontology),
            "iteration": args.iteration,
            "gaussian_count": gaussian_count,
            "camera_count": len(profile_frames[profile_name]),
            "pixel_filtering": (
                "joint_empirical_weakest_metric_rank_nested_profile"
            ),
            "query_region_filtering_used": False,
            "confidence_threshold_used": profile_level > 0,
            "inference_rerun": False,
            "flashsplat_rerun": True,
            "single_flashsplat_render_shared_across_profiles": True,
            "vote_formula": (
                "used_count_for_profile_class/"
                "sum_used_count_over_all_confidence_levels"
            ),
            "abstain_mass_preserved": True,
            "one_normalized_visibility_budget_per_camera": True,
            "automatic_min_camera_accepted_fraction": (
                0.5 * actual_joint_ratio
            ),
            "total_pixel_count": total_pixels,
            "kept_pixel_count": kept_pixel_count,
            "kept_pixel_ratio": kept_pixel_count / float(max(total_pixels, 1)),
            "scene_specific_rules": False,
            "class_specific_thresholds": False,
            "manual_component_decisions": False,
            "v5_used": False,
            "dinov2_used": False,
            "frames": profile_frames[profile_name],
        }
        (profile_root / "vote_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({
            "profile_name": profile_name,
            "camera_count": manifest["camera_count"],
            "kept_pixel_ratio": manifest["kept_pixel_ratio"],
            "automatic_min_camera_accepted_fraction": manifest[
                "automatic_min_camera_accepted_fraction"
            ],
        }, indent=2))


if __name__ == "__main__":
    main()
