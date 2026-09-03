#!/usr/bin/env python3
"""Resolve and validate an automatic selected-camera DINOv3 cache.

The selected cameras always come from a saved deterministic visibility ranking.
This module does not accept a manual camera list.  Before inference it validates
the complete ranking, saved coverage curve, threshold record, and reconstruction
camera count.  After inference it validates that the renderer and DINOv3
manifests contain the exact selected prefix and the expected report-only
settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SELECTION_CONTRACT = "deterministic_greedy_visibility_multicover_v1"
CACHE_CONTRACT = "threshold_selected_dinov3_view_cache_v1"
THRESHOLD_GROUP = "two_view_coverage_of_multiview_capable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _integer_vector(value: np.ndarray, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain exactly {length} integers")
    return array.astype(np.int64, copy=False)


def resolve_selected_prefix(
    selection_dir: Path,
    model_path: Path,
    *,
    target_percent: int = 99,
    source_max_width: int = 960,
) -> dict[str, Any]:
    """Return a fully validated threshold-derived camera prefix."""

    report_path = selection_dir / "camera_selection_report.json"
    order_path = selection_dir / "greedy_camera_indices.npy"
    curve_path = selection_dir / "coverage_curve.npz"
    cameras_path = model_path / "cameras.json"
    required = {
        "camera_selection_report": report_path,
        "greedy_camera_indices": order_path,
        "coverage_curve": curve_path,
        "model_cameras": cameras_path,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    hashes_before = {name: sha256_file(path) for name, path in required.items()}
    report = read_json(report_path)
    if report.get("contract") != SELECTION_CONTRACT:
        raise ValueError("unsupported camera-selection contract")
    if report.get("manual_camera_selection_used") is not False:
        raise ValueError("camera-selection report is not explicitly automatic")
    if int(report.get("source_max_width", -1)) != source_max_width:
        raise ValueError("camera selection was not derived from the expected render width")

    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("cameras.json must contain a nonempty camera list")
    camera_count = int(report.get("camera_count", -1))
    if camera_count != len(cameras):
        raise ValueError("selection and reconstruction camera counts do not match")

    order = _integer_vector(
        np.load(order_path, allow_pickle=False),
        name="greedy camera order",
        length=camera_count,
    )
    if np.unique(order).size != camera_count:
        raise ValueError("greedy camera order must contain unique indices")
    if np.any(order < 0) or np.any(order >= len(cameras)):
        raise ValueError("greedy camera order contains an out-of-range index")

    with np.load(curve_path, allow_pickle=False) as curve_file:
        required_curve_keys = {
            "selected_camera_count",
            "camera_rows",
            "camera_indices",
            "single_covered",
            "double_covered",
        }
        missing_keys = sorted(required_curve_keys.difference(curve_file.files))
        if missing_keys:
            raise ValueError(f"coverage curve is missing arrays: {missing_keys}")
        selected_count_curve = _integer_vector(
            curve_file["selected_camera_count"],
            name="selected-camera-count curve",
            length=camera_count,
        ).copy()
        curve_rows = _integer_vector(
            curve_file["camera_rows"],
            name="coverage-curve camera rows",
            length=camera_count,
        ).copy()
        curve_indices = _integer_vector(
            curve_file["camera_indices"],
            name="coverage-curve camera indices",
            length=camera_count,
        ).copy()
        single_curve = _integer_vector(
            curve_file["single_covered"],
            name="single-coverage curve",
            length=camera_count,
        ).copy()
        double_curve = _integer_vector(
            curve_file["double_covered"],
            name="double-coverage curve",
            length=camera_count,
        ).copy()

    if not np.array_equal(
        selected_count_curve,
        np.arange(1, camera_count + 1, dtype=np.int64),
    ):
        raise ValueError("selected-camera-count curve is not sequential")
    if np.unique(curve_rows).size != camera_count:
        raise ValueError("coverage-curve camera rows must be unique")
    if np.any(curve_rows < 0) or np.any(curve_rows >= camera_count):
        raise ValueError("coverage-curve camera rows are out of range")
    if not np.array_equal(curve_indices, order):
        raise ValueError("coverage-curve camera order does not match ranking")
    if np.any(np.diff(single_curve) < 0) or np.any(np.diff(double_curve) < 0):
        raise ValueError("coverage curves must be monotonic")

    steps = report.get("selection_steps")
    if not isinstance(steps, list) or len(steps) != camera_count:
        raise ValueError("selection report must contain one step per camera")
    for offset, (step, camera_index, camera_row) in enumerate(
        zip(steps, order, curve_rows),
        start=1,
    ):
        if not isinstance(step, dict):
            raise ValueError("selection steps must be JSON objects")
        if int(step.get("selection_step", -1)) != offset:
            raise ValueError("selection-step numbering is inconsistent")
        if int(step.get("camera_index", -1)) != int(camera_index):
            raise ValueError("selection-step camera order is inconsistent")
        expected_camera_id = int(cameras[int(camera_index)]["id"])
        if int(step.get("camera_id", -1)) != expected_camera_id:
            raise ValueError("selection-step camera ID does not match cameras.json")
        if int(step.get("visibility_row", -1)) != int(camera_row):
            raise ValueError("selection-step visibility rows are inconsistent")
        if int(step.get("single_covered", -1)) != int(single_curve[offset - 1]):
            raise ValueError("selection-step single coverage is inconsistent")
        if int(step.get("double_covered", -1)) != int(double_curve[offset - 1]):
            raise ValueError("selection-step double coverage is inconsistent")

    thresholds = report.get("thresholds", {})
    group = thresholds.get(THRESHOLD_GROUP, {}) if isinstance(thresholds, dict) else {}
    record = group.get(str(target_percent), {}) if isinstance(group, dict) else {}
    if not isinstance(record, dict):
        raise ValueError("requested threshold record is missing")
    if int(record.get("target_percent", -1)) != target_percent:
        raise ValueError("requested threshold percentage is inconsistent")
    selected_camera_count = record.get("selected_camera_count")
    if not isinstance(selected_camera_count, int):
        raise ValueError("requested threshold was not reached")
    if not 1 <= selected_camera_count <= camera_count:
        raise ValueError("threshold selected-camera count is invalid")
    prefix = order[:selected_camera_count]
    recorded_prefix = record.get("camera_indices")
    if recorded_prefix != [int(value) for value in prefix]:
        raise ValueError("threshold camera prefix does not match ranking")

    multiview_count = int(report.get("globally_multiview_capable_gaussian_count", -1))
    observable_count = int(report.get("globally_observable_gaussian_count", -1))
    if not 1 <= multiview_count <= observable_count:
        raise ValueError("invalid observable or multiview-capable Gaussian count")
    if int(single_curve[-1]) != observable_count:
        raise ValueError("single-coverage curve does not reach the observable count")
    if int(double_curve[-1]) != multiview_count:
        raise ValueError("double-coverage curve does not reach the multiview-capable count")

    required_gaussian_count = int(record.get("required_gaussian_count", -1))
    expected_required_count = (multiview_count * target_percent + 99) // 100
    if required_gaussian_count != expected_required_count:
        raise ValueError("threshold required-Gaussian count is inconsistent")
    achieved_double_count = int(double_curve[selected_camera_count - 1])
    if required_gaussian_count < 0 or achieved_double_count < required_gaussian_count:
        raise ValueError("selected prefix does not reach its recorded threshold")
    if (
        selected_camera_count > 1
        and int(double_curve[selected_camera_count - 2]) >= required_gaussian_count
    ):
        raise ValueError("selected prefix is not the first threshold-reaching prefix")

    if achieved_double_count > multiview_count:
        raise ValueError("invalid multiview-capable Gaussian count")
    hashes_after = {name: sha256_file(path) for name, path in required.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("camera-selection inputs changed during validation")

    return {
        "selection_contract": SELECTION_CONTRACT,
        "threshold_group": THRESHOLD_GROUP,
        "target_percent": target_percent,
        "selected_camera_count": selected_camera_count,
        "camera_indices": [int(value) for value in prefix],
        "camera_ids": [int(cameras[int(value)]["id"]) for value in prefix],
        "required_gaussian_count": required_gaussian_count,
        "achieved_single_covered": int(single_curve[selected_camera_count - 1]),
        "achieved_double_covered": achieved_double_count,
        "source_max_width": source_max_width,
        "globally_observable_gaussian_count": observable_count,
        "globally_multiview_capable_gaussian_count": multiview_count,
        "achieved_two_view_coverage_of_multiview_capable": (
            achieved_double_count / multiview_count
        ),
        "manual_camera_selection_used": False,
        "selection_input_sha256": hashes_before,
    }


def _validate_exact_frames(
    frames: Any,
    expected_indices: list[int],
    expected_ids: list[int],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(frames, list) or len(frames) != len(expected_indices):
        raise ValueError(f"{label} must contain exactly the selected cameras")
    if any(not isinstance(frame, dict) for frame in frames):
        raise ValueError(f"{label} frames must be JSON objects")
    actual = [int(frame.get("camera_index", -1)) for frame in frames]
    if actual != expected_indices:
        raise ValueError(f"{label} camera order does not match selected prefix")
    actual_ids = [int(frame.get("camera_id", -1)) for frame in frames]
    if actual_ids != expected_ids:
        raise ValueError(f"{label} camera IDs do not match selected prefix")
    filenames = [str(frame.get("file", "")) for frame in frames]
    if any(not filename for filename in filenames) or len(set(filenames)) != len(filenames):
        raise ValueError(f"{label} frame filenames must be nonempty and unique")
    return frames


def validate_segment_file(path: Path, *, height: int, width: int) -> None:
    expected_shape = (height, width)
    if height < 1 or width < 1:
        raise ValueError("render dimensions must be positive")
    required = {
        "class_id",
        "confidence",
        "max_softmax_probability",
        "top1_top2_margin",
        "normalized_entropy_confidence",
    }
    with np.load(path, allow_pickle=False) as values:
        missing = sorted(required.difference(values.files))
        if missing:
            raise ValueError(f"DINOv3 segment cache is missing arrays: {missing}")
        class_id = np.asarray(values["class_id"])
        if class_id.shape != expected_shape or not np.issubdtype(
            class_id.dtype, np.integer
        ):
            raise ValueError("DINOv3 class map has the wrong shape or dtype")
        if np.any(class_id < 0) or np.any(class_id >= 150):
            raise ValueError("DINOv3 class map contains an invalid ADE20K class")
        for key in required.difference({"class_id"}):
            metric = np.asarray(values[key])
            if metric.shape != expected_shape or not np.issubdtype(
                metric.dtype, np.floating
            ):
                raise ValueError(f"DINOv3 {key} map has the wrong shape or dtype")
            if not np.isfinite(metric).all():
                raise ValueError(f"DINOv3 {key} map contains non-finite values")
            if np.any(metric < 0.0) or np.any(metric > 1.0):
                raise ValueError(f"DINOv3 {key} map is outside [0, 1]")


def validate_probability_file(
    path: Path,
    segment_path: Path,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    probabilities = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = (150, height, width)
    if probabilities.shape != expected_shape or probabilities.dtype != np.float16:
        raise ValueError("DINOv3 probability tensor has the wrong shape or dtype")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("DINOv3 probability tensor is non-finite or negative")
    sums = probabilities.sum(axis=0, dtype=np.float32)
    if not np.allclose(sums, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError("stored DINOv3 probability mass does not sum to one")
    with np.load(segment_path, allow_pickle=False) as segment:
        hard = np.asarray(segment["class_id"], dtype=np.uint8)
    quantized = np.argmax(probabilities, axis=0).astype(np.uint8)
    disagreements = int(np.count_nonzero(quantized != hard))
    return {
        "size_bytes": path.stat().st_size,
        "float16_argmax_disagreement_count": disagreements,
        "float16_argmax_disagreement_ratio": disagreements / hard.size,
    }


def finalize_selected_cache(
    selection_dir: Path,
    model_path: Path,
    output_dir: Path,
    *,
    target_percent: int = 99,
    render_max_width: int = 960,
    crop_size: int = 512,
    stride: int = 384,
    precision: str = "bfloat16",
    checkpoint_load_mode: str = "local_mmap",
    require_probabilities: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    selected = resolve_selected_prefix(
        selection_dir,
        model_path,
        target_percent=target_percent,
        source_max_width=render_max_width,
    )
    expected_indices = selected["camera_indices"]
    expected_ids = selected["camera_ids"]
    view_dir = output_dir / "stages" / "01_real_camera_views"
    view_manifest_path = view_dir / "view_manifest.json"
    dinov3_manifest_path = view_dir / "dinov3_manifest.json"
    view_manifest = read_json(view_manifest_path)
    dinov3_manifest = read_json(dinov3_manifest_path)

    if int(view_manifest.get("camera_count", -1)) != len(expected_indices):
        raise ValueError("view-manifest camera count is incorrect")
    if int(view_manifest.get("max_width", -1)) != render_max_width:
        raise ValueError("view-manifest render width is incorrect")
    view_frames = _validate_exact_frames(
        view_manifest.get("frames"),
        expected_indices,
        expected_ids,
        label="view manifest",
    )
    for frame in view_frames:
        if not (view_dir / "rgb_renders" / str(frame["file"])).is_file():
            raise FileNotFoundError(f"missing RGB render for {frame['file']}")

    expected_contract = (
        "raw_ade20k_class_probabilities_and_relative_margin_confidence_v3"
        if require_probabilities
        else "raw_ade20k_class_and_relative_margin_confidence_v2"
    )
    if dinov3_manifest.get("contract") != expected_contract:
        raise ValueError("unsupported DINOv3 cache contract")
    dino_frames = _validate_exact_frames(
        dinov3_manifest.get("frames"),
        expected_indices,
        expected_ids,
        label="DINOv3 manifest",
    )
    if [str(frame["file"]) for frame in dino_frames] != [
        str(frame["file"]) for frame in view_frames
    ]:
        raise ValueError("DINOv3 filenames do not match rendered views")
    model = dinov3_manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("DINOv3 manifest is missing model settings")
    expected_settings = {
        "precision": precision,
        "crop_size": crop_size,
        "stride": stride,
    }
    for key, expected in expected_settings.items():
        if model.get(key) != expected:
            raise ValueError(f"DINOv3 {key} does not match selected-cache settings")
    checkpoint_loading = model.get("checkpoint_loading")
    if not isinstance(checkpoint_loading, dict) or checkpoint_loading.get("mode") != checkpoint_load_mode:
        raise ValueError("DINOv3 checkpoint-loading mode is incorrect")
    probability_summaries: list[dict[str, Any]] = []
    for frame in dino_frames:
        for key in ("segment_file", "overlay_file"):
            relative = str(frame.get(key, ""))
            if not relative or not (view_dir / relative).is_file():
                raise FileNotFoundError(f"missing DINOv3 {key} for {frame['file']}")
        validate_segment_file(
            view_dir / str(frame["segment_file"]),
            height=int(frame.get("render_height", -1)),
            width=int(frame.get("render_width", -1)),
        )
        if require_probabilities:
            probability_relative = str(frame.get("probability_file", ""))
            probability_path = view_dir / probability_relative
            if not probability_relative or not probability_path.is_file():
                raise FileNotFoundError("missing complete DINOv3 probability tensor")
            summary = validate_probability_file(
                probability_path,
                view_dir / str(frame["segment_file"]),
                height=int(frame.get("render_height", -1)),
                width=int(frame.get("render_width", -1)),
            )
            probability_summaries.append(
                {
                    "file": str(frame["file"]),
                    "probability_file": probability_relative,
                    **summary,
                }
            )

    for stage in (
        output_dir / "stages" / "02_flashsplat_votes",
        output_dir / "stages" / "03_exact_fusion",
        output_dir / "deliverables",
    ):
        if stage.exists() and any(path.is_file() for path in stage.rglob("*")):
            raise ValueError(f"report-only cache contains a populated later stage: {stage}")

    forbidden_names = {
        "gaussian_labels.npy",
        "project_class_id.npy",
        "label_map.json",
        "semantic_point_cloud.ply",
    }
    forbidden = [
        path for path in output_dir.rglob("*") if path.is_file() and (
            path.name in forbidden_names or path.suffix.lower() == ".ply"
        )
    ]
    if forbidden:
        raise ValueError(f"report-only cache contains forbidden semantic output: {forbidden[0]}")

    report = {
        "source": "automatic_visibility_threshold_selected_dinov3_cache",
        "contract": CACHE_CONTRACT,
        **selected,
        "model_path": str(model_path),
        "selection_dir": str(selection_dir),
        "output_dir": str(output_dir),
        "render": {"max_width": render_max_width},
        "dinov3": {
            "precision": precision,
            "crop_size": crop_size,
            "stride": stride,
            "checkpoint_load_mode": checkpoint_load_mode,
            "complete_probabilities": require_probabilities,
        },
        "dense_probability_cache": {
            "available": require_probabilities,
            "contract": expected_contract,
            "dtype": "float16" if require_probabilities else None,
            "layout": "ade20k_class_height_width" if require_probabilities else None,
            "class_count": 150 if require_probabilities else None,
            "total_size_bytes": sum(
                item["size_bytes"] for item in probability_summaries
            ),
            "float16_argmax_disagreement_count": sum(
                item["float16_argmax_disagreement_count"]
                for item in probability_summaries
            ),
            "frames": probability_summaries,
        },
        "semantic_inference_used": True,
        "semantic_vote_lifting_run": False,
        "semantic_fusion_run": False,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "output_manifest_sha256": {
            "view_manifest": sha256_file(view_manifest_path),
            "dinov3_manifest": sha256_file(dinov3_manifest_path),
        },
    }
    if write_report:
        report_dir = output_dir / "selection"
        report_dir.mkdir(parents=True, exist_ok=False)
        np.save(
            report_dir / "selected_camera_indices.npy",
            np.asarray(expected_indices, dtype=np.int32),
        )
        report_path = report_dir / "selected_view_cache_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_existing_selected_cache(
    selection_dir: Path,
    model_path: Path,
    output_dir: Path,
    *,
    target_percent: int = 99,
    render_max_width: int = 960,
    crop_size: int = 512,
    stride: int = 384,
    precision: str = "bfloat16",
    checkpoint_load_mode: str = "local_mmap",
) -> dict[str, Any]:
    """Revalidate a completed dense cache without rewriting any artifact."""

    report_dir = output_dir / "selection"
    report_path = report_dir / "selected_view_cache_report.json"
    indices_path = report_dir / "selected_camera_indices.npy"
    for path in (report_path, indices_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {
        "selected_view_cache_report": sha256_file(report_path),
        "selected_camera_indices": sha256_file(indices_path),
    }
    existing = read_json(report_path)
    expected = finalize_selected_cache(
        selection_dir,
        model_path,
        output_dir,
        target_percent=target_percent,
        render_max_width=render_max_width,
        crop_size=crop_size,
        stride=stride,
        precision=precision,
        checkpoint_load_mode=checkpoint_load_mode,
        require_probabilities=True,
        write_report=False,
    )
    if existing != expected:
        raise ValueError("existing selected-cache report does not match current artifacts")
    indices = np.load(indices_path, allow_pickle=False)
    expected_indices = np.asarray(expected["camera_indices"], dtype=np.int32)
    if indices.dtype != np.int32 or not np.array_equal(indices, expected_indices):
        raise ValueError("existing selected-camera index array is inconsistent")
    hashes_after = {
        "selected_view_cache_report": sha256_file(report_path),
        "selected_camera_indices": sha256_file(indices_path),
    }
    if hashes_before != hashes_after:
        raise RuntimeError("selected-cache resume artifacts changed during validation")
    return existing


def compare_hard_caches(
    reference_view_dir: Path,
    candidate_view_dir: Path,
    output_path: Path,
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    """Measure exact hard-map reproducibility without changing either cache."""

    reference_manifest_path = reference_view_dir / "dinov3_manifest.json"
    candidate_manifest_path = candidate_view_dir / "dinov3_manifest.json"
    required = (reference_manifest_path, candidate_manifest_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256_file(path) for path in required}
    reference = read_json(reference_manifest_path)
    candidate = read_json(candidate_manifest_path)
    reference_frames = reference.get("frames", [])
    candidate_frames = candidate.get("frames", [])
    for field in ("camera_index", "camera_id", "file"):
        if [frame.get(field) for frame in reference_frames] != [
            frame.get(field) for frame in candidate_frames
        ]:
            raise ValueError(f"candidate cache differs from reference {field} order")
    if not reference_frames:
        raise ValueError("reference cache has no frames")
    frames: list[dict[str, Any]] = []
    total_pixels = 0
    total_disagreements = 0
    for reference_frame, candidate_frame in zip(reference_frames, candidate_frames):
        reference_path = reference_view_dir / str(reference_frame["segment_file"])
        candidate_path = candidate_view_dir / str(candidate_frame["segment_file"])
        with np.load(reference_path, allow_pickle=False) as values:
            reference_class = np.asarray(values["class_id"], dtype=np.uint8)
        with np.load(candidate_path, allow_pickle=False) as values:
            candidate_class = np.asarray(values["class_id"], dtype=np.uint8)
        if reference_class.shape != candidate_class.shape:
            raise ValueError("candidate hard map shape differs from reference")
        disagreements = int(np.count_nonzero(reference_class != candidate_class))
        pixels = int(reference_class.size)
        frames.append({
            "file": str(reference_frame["file"]),
            "camera_index": int(reference_frame["camera_index"]),
            "pixel_count": pixels,
            "hard_class_disagreement_count": disagreements,
            "hard_class_disagreement_ratio": disagreements / pixels,
        })
        total_pixels += pixels
        total_disagreements += disagreements
    hashes_after = {str(path): sha256_file(path) for path in required}
    if hashes_before != hashes_after:
        raise RuntimeError("hard-cache manifests changed during comparison")
    report = {
        "source": "dinov3_selected_cache_hard_class_reproducibility",
        "contract": "pixel_exact_hard_class_cache_comparison_v1",
        "reference_view_dir": str(reference_view_dir),
        "candidate_view_dir": str(candidate_view_dir),
        "camera_count": len(frames),
        "pixel_count": total_pixels,
        "hard_class_disagreement_count": total_disagreements,
        "hard_class_disagreement_ratio": total_disagreements / total_pixels,
        "pixel_exact_match": total_disagreements == 0,
        "manual_correction_used": False,
        "manifest_sha256": hashes_before,
        "frames": frames,
    }
    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_probability_resume(
    selection_dir: Path,
    model_path: Path,
    output_dir: Path,
    reference_view_dir: Path,
) -> dict[str, Any]:
    """Validate completed stages 1-6 before resuming at the soft lift."""

    cache_report = validate_existing_selected_cache(
        selection_dir,
        model_path,
        output_dir,
    )
    comparison_path = output_dir / "selection" / "hard_cache_reproducibility.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    comparison_hash_before = sha256_file(comparison_path)
    existing_comparison = read_json(comparison_path)
    expected_comparison = compare_hard_caches(
        reference_view_dir,
        output_dir / "stages" / "01_real_camera_views",
        comparison_path,
        write_output=False,
    )
    if existing_comparison != expected_comparison:
        raise ValueError("existing hard-cache comparison does not match current caches")
    if sha256_file(comparison_path) != comparison_hash_before:
        raise RuntimeError("hard-cache comparison changed during resume validation")
    return {
        "source": "dinov3_probability_cache_resume_validation",
        "contract": "validated_existing_probability_cache_resume_v1",
        "camera_count": int(cache_report["selected_camera_count"]),
        "probability_size_bytes": int(
            cache_report["dense_probability_cache"]["total_size_bytes"]
        ),
        "hard_cache_pixel_exact_match": bool(
            existing_comparison["pixel_exact_match"]
        ),
        "selected_cache_report_sha256": sha256_file(
            output_dir / "selection" / "selected_view_cache_report.json"
        ),
        "hard_cache_comparison_sha256": comparison_hash_before,
        "artifacts_rewritten": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--selection-dir", required=True, type=Path)
    resolve.add_argument("--model-path", required=True, type=Path)
    resolve.add_argument("--target-percent", default=99, type=int)
    resolve.add_argument("--source-max-width", default=960, type=int)
    resolve.add_argument("--format", choices=("json", "lines"), default="json")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--selection-dir", required=True, type=Path)
    finalize.add_argument("--model-path", required=True, type=Path)
    finalize.add_argument("--output-dir", required=True, type=Path)
    finalize.add_argument("--target-percent", default=99, type=int)
    finalize.add_argument("--render-max-width", default=960, type=int)
    finalize.add_argument("--crop-size", default=512, type=int)
    finalize.add_argument("--stride", default=384, type=int)
    finalize.add_argument("--precision", default="bfloat16")
    finalize.add_argument("--checkpoint-load-mode", default="local_mmap")
    finalize.add_argument("--require-probabilities", action="store_true")
    compare = subparsers.add_parser("compare-hard-cache")
    compare.add_argument("--reference-view-dir", required=True, type=Path)
    compare.add_argument("--candidate-view-dir", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    resume = subparsers.add_parser("validate-resume")
    resume.add_argument("--selection-dir", required=True, type=Path)
    resume.add_argument("--model-path", required=True, type=Path)
    resume.add_argument("--output-dir", required=True, type=Path)
    resume.add_argument("--reference-view-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "resolve":
        result = resolve_selected_prefix(
            args.selection_dir,
            args.model_path,
            target_percent=args.target_percent,
            source_max_width=args.source_max_width,
        )
        if args.format == "lines":
            print(result["selected_camera_count"])
            print(",".join(str(value) for value in result["camera_indices"]))
        else:
            print(json.dumps(result, indent=2))
        return

    if args.command == "compare-hard-cache":
        result = compare_hard_caches(
            args.reference_view_dir,
            args.candidate_view_dir,
            args.output,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "validate-resume":
        result = validate_probability_resume(
            args.selection_dir,
            args.model_path,
            args.output_dir,
            args.reference_view_dir,
        )
        print(json.dumps(result, indent=2))
        return

    result = finalize_selected_cache(
        args.selection_dir,
        args.model_path,
        args.output_dir,
        target_percent=args.target_percent,
        render_max_width=args.render_max_width,
        crop_size=args.crop_size,
        stride=args.stride,
        precision=args.precision,
        checkpoint_load_mode=args.checkpoint_load_mode,
        require_probabilities=args.require_probabilities,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
