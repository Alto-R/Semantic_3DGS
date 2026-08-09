#!/usr/bin/env python3
"""Calibrate DINOv3 pixel confidence from cached target-scene distributions.

Absolute Mask2Former probability and entropy-confidence values are not
calibrated to intuitive 0-1 thresholds.  This report-only preflight streams
all requested cached segment maps, builds one joint empirical CDF for each
stored confidence metric, and ranks each pixel by the weakest of its three
metric percentiles.  Global nested profiles are then defined by retained
quantiles of that joint weakest-rank distribution.

The output contains only calibration histograms and reports.  It does not run
inference, lift pixels into 3D, or write semantic labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov3.lift_dense_view_votes import (
    validate_dinov3_manifest,
)


SOURCE = "dinov3_joint_empirical_pixel_confidence_calibration"
CONTRACT = "joint_metric_cdf_weakest_rank_nested_profiles_v1"
METRICS = {
    "relative_margin": "confidence",
    "max_softmax_probability": "max_softmax_probability",
    "normalized_entropy_confidence": "normalized_entropy_confidence",
}
PROFILE_TARGETS = (
    ("baseline", 1.00),
    ("permissive", 0.50),
    ("balanced", 0.25),
    ("strict", 0.10),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def histogram_indices(values: np.ndarray, bin_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("confidence values contain non-finite entries")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("confidence values are outside [0, 1]")
    return np.minimum(
        (array * np.float32(bin_count)).astype(np.int32),
        bin_count - 1,
    )


def empirical_cdf(histogram: np.ndarray) -> np.ndarray:
    counts = np.asarray(histogram, dtype=np.uint64)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("histogram must be one-dimensional with at least 2 bins")
    total = int(counts.sum())
    if total < 1:
        raise ValueError("histogram is empty")
    return (
        np.cumsum(counts, dtype=np.uint64).astype(np.float64)
        / float(total)
    ).astype(np.float32)


def weakest_empirical_rank_bins(
    segment: Any,
    metric_cdfs: dict[str, np.ndarray],
    bin_count: int,
) -> np.ndarray:
    weakest: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    for metric_name, archive_key in METRICS.items():
        if archive_key not in segment:
            raise ValueError(f"segment lacks calibration field {archive_key}")
        values = np.asarray(segment[archive_key], dtype=np.float32)
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError("segment confidence maps have different shapes")
        bins = histogram_indices(values, bin_count)
        ranks = metric_cdfs[metric_name][bins]
        weakest = ranks if weakest is None else np.minimum(weakest, ranks)
    if weakest is None:
        raise AssertionError("calibration metric set is empty")
    return histogram_indices(weakest, bin_count)


def profile_minimum_bins(
    composite_histogram: np.ndarray,
    profile_targets: tuple[tuple[str, float], ...] = PROFILE_TARGETS,
) -> list[dict[str, Any]]:
    histogram = np.asarray(composite_histogram, dtype=np.uint64)
    cdf = empirical_cdf(histogram)
    profiles: list[dict[str, Any]] = []
    previous_bin = -1
    for profile_name, target_ratio in profile_targets:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target retained ratios must be in (0, 1]")
        if target_ratio == 1.0:
            minimum_bin = 0
        else:
            minimum_bin = int(
                np.searchsorted(
                    cdf,
                    1.0 - target_ratio,
                    side="left",
                )
            )
        minimum_bin = min(minimum_bin, histogram.size - 1)
        if minimum_bin < previous_bin:
            raise RuntimeError("calibrated profile thresholds are not nested")
        previous_bin = minimum_bin
        retained = int(histogram[minimum_bin:].sum())
        profiles.append(
            {
                "profile_name": profile_name,
                "target_joint_retained_ratio": target_ratio,
                "minimum_weakest_rank_bin": minimum_bin,
                "minimum_weakest_rank": minimum_bin / float(histogram.size),
                "actual_joint_retained_ratio": (
                    retained / float(histogram.sum())
                ),
            }
        )
    return profiles


def _load_sources(input_dirs: list[Path]) -> list[dict[str, Any]]:
    if len(input_dirs) < 2:
        raise ValueError("joint calibration requires at least two input dirs")
    sources: list[dict[str, Any]] = []
    seen_manifests: set[Path] = set()
    for input_dir in input_dirs:
        manifest_path = input_dir / "dinov3_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        resolved = manifest_path.resolve()
        if resolved in seen_manifests:
            raise ValueError("joint calibration input manifests are duplicated")
        seen_manifests.add(resolved)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_dinov3_manifest(manifest)
        frames = manifest.get("frames", [])
        if not isinstance(frames, list) or not frames:
            raise ValueError("DINOv3 manifest has no frames")
        sources.append(
            {
                "input_dir": input_dir,
                "manifest_path": manifest_path,
                "manifest": manifest,
            }
        )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-count", default=4096, type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.bin_count < 256:
        raise ValueError("bin_count must be at least 256")

    sources = _load_sources(args.input_dir)
    metric_histograms = {
        metric_name: np.zeros((args.bin_count,), dtype=np.uint64)
        for metric_name in METRICS
    }
    total_pixels = 0
    total_frames = 0
    for source in sources:
        input_dir = source["input_dir"]
        for frame in source["manifest"]["frames"]:
            segment_path = input_dir / str(frame["segment_file"])
            with np.load(segment_path, allow_pickle=False) as segment:
                class_shape = np.asarray(segment["class_id"]).shape
                for metric_name, archive_key in METRICS.items():
                    values = np.asarray(segment[archive_key], dtype=np.float32)
                    if values.shape != class_shape:
                        raise ValueError(
                            f"{segment_path} {archive_key} shape differs "
                            "from class_id"
                        )
                    bins = histogram_indices(values, args.bin_count)
                    metric_histograms[metric_name] += np.bincount(
                        bins.ravel(),
                        minlength=args.bin_count,
                    ).astype(np.uint64)
            total_pixels += int(np.prod(class_shape))
            total_frames += 1
            print(f"calibration metric pass: {segment_path}")

    if any(int(hist.sum()) != total_pixels for hist in metric_histograms.values()):
        raise RuntimeError("metric histogram pixel counts do not agree")
    metric_cdfs = {
        metric_name: empirical_cdf(histogram)
        for metric_name, histogram in metric_histograms.items()
    }

    composite_histogram = np.zeros((args.bin_count,), dtype=np.uint64)
    source_score_histograms: list[np.ndarray] = []
    source_frame_histograms: list[list[np.ndarray]] = []
    for source in sources:
        input_dir = source["input_dir"]
        source_histogram = np.zeros((args.bin_count,), dtype=np.uint64)
        frame_histograms: list[np.ndarray] = []
        for frame in source["manifest"]["frames"]:
            segment_path = input_dir / str(frame["segment_file"])
            with np.load(segment_path, allow_pickle=False) as segment:
                score_bins = weakest_empirical_rank_bins(
                    segment,
                    metric_cdfs,
                    args.bin_count,
                )
            histogram = np.bincount(
                score_bins.ravel(),
                minlength=args.bin_count,
            ).astype(np.uint64)
            source_histogram += histogram
            composite_histogram += histogram
            frame_histograms.append(histogram)
            print(f"calibration composite pass: {segment_path}")
        source_score_histograms.append(source_histogram)
        source_frame_histograms.append(frame_histograms)

    if int(composite_histogram.sum()) != total_pixels:
        raise RuntimeError("composite histogram pixel count differs from metrics")
    profiles = profile_minimum_bins(composite_histogram)
    profile_by_name = {
        str(profile["profile_name"]): profile for profile in profiles
    }

    source_reports: list[dict[str, Any]] = []
    for source, source_histogram, frame_histograms in zip(
        sources,
        source_score_histograms,
        source_frame_histograms,
    ):
        source_pixel_count = int(source_histogram.sum())
        profile_retention: dict[str, Any] = {}
        for profile_name, _target in PROFILE_TARGETS:
            profile = profile_by_name[profile_name]
            minimum_bin = int(profile["minimum_weakest_rank_bin"])
            kept = int(source_histogram[minimum_bin:].sum())
            frame_ratios = np.asarray(
                [
                    int(frame_histogram[minimum_bin:].sum())
                    / float(frame_histogram.sum())
                    for frame_histogram in frame_histograms
                ],
                dtype=np.float64,
            )
            profile_retention[profile_name] = {
                "kept_pixel_count": kept,
                "retained_ratio": kept / float(source_pixel_count),
                "minimum_frame_retained_ratio": float(frame_ratios.min()),
                "median_frame_retained_ratio": float(
                    np.median(frame_ratios)
                ),
                "maximum_frame_retained_ratio": float(frame_ratios.max()),
                "zero_retention_frame_count": int(
                    np.count_nonzero(frame_ratios == 0.0)
                ),
            }
        source_reports.append(
            {
                "input_dir": str(source["input_dir"]),
                "segmentation_manifest": str(source["manifest_path"]),
                "segmentation_manifest_sha256": file_sha256(
                    source["manifest_path"]
                ),
                "frame_count": len(source["manifest"]["frames"]),
                "pixel_count": source_pixel_count,
                "profile_retention": profile_retention,
            }
        )

    for profile in profiles:
        if float(profile["actual_joint_retained_ratio"]) <= 0.0:
            raise RuntimeError("a calibrated profile retains no joint pixels")
    for source_report in source_reports:
        for profile_name, _target in PROFILE_TARGETS:
            retained = float(
                source_report["profile_retention"][profile_name][
                    "retained_ratio"
                ]
            )
            if retained <= 0.0:
                raise RuntimeError(
                    f"{profile_name} retains no pixels in a calibration source"
                )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration_npz = args.output_dir / "confidence_calibration.npz"
    np.savez_compressed(
        calibration_npz,
        **{
            f"{metric_name}_histogram": metric_histograms[metric_name]
            for metric_name in METRICS
        },
        **{
            f"{metric_name}_cdf": metric_cdfs[metric_name]
            for metric_name in METRICS
        },
        composite_histogram=composite_histogram,
    )
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "report_only": True,
        "calibration_scope": "joint_all_pixels_from_all_input_manifests",
        "confidence_score": (
            "minimum_empirical_cdf_rank_across_relative_margin_max_softmax_"
            "probability_and_normalized_entropy_confidence"
        ),
        "profile_policy": "fixed_joint_retained_quantiles",
        "profile_targets": {
            name: target for name, target in PROFILE_TARGETS
        },
        "profiles": profiles,
        "metric_archive_keys": METRICS,
        "bin_count": args.bin_count,
        "source_count": len(sources),
        "frame_count": total_frames,
        "pixel_count": total_pixels,
        "sources": source_reports,
        "calibration_npz": str(calibration_npz),
        "calibration_npz_sha256": file_sha256(calibration_npz),
        "inference_rerun": False,
        "flashsplat_rerun": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "manual_component_decisions": False,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "confidence_calibration.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_count": len(sources),
        "frame_count": total_frames,
        "pixel_count": total_pixels,
        "profiles": profiles,
    }, indent=2))


if __name__ == "__main__":
    main()
