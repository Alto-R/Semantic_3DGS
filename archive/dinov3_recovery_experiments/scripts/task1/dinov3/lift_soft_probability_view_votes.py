#!/usr/bin/env python3
"""Estimate probability-preserving DINOv3-to-Gaussian evidence with FlashSplat.

FlashSplat's public mask input accepts one integer class per pixel, not a
class-probability vector.  This module therefore uses a fixed deterministic
eight-stratum categorical estimator.  Its expectation is the complete pixel
probability tensor, and no class, camera, scene, or confidence threshold is
selected manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.lift_dense_view_votes import flashsplat_class_rows


SOURCE = "dinov3_probability_flashsplat_votes"
CONTRACT = "deterministic_stratified_probability_mass_per_camera_v1"
PROBABILITY_CONTRACT = (
    "raw_ade20k_class_probabilities_and_relative_margin_confidence_v3"
)
FIXED_SAMPLE_COUNT = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_stratified_thresholds(
    height: int,
    width: int,
    sample_count: int,
    camera_index: int,
) -> np.ndarray:
    """Return reproducible circularly stratified thresholds in [0, 1)."""

    if height < 1 or width < 1 or sample_count < 2 or sample_count % 2:
        raise ValueError("dimensions must be positive and sample_count positive/even")
    values = np.arange(height * width, dtype=np.uint64)
    values += np.uint64((camera_index + 1) * 0x9E3779B1)
    values ^= values >> np.uint64(16)
    values = (values * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    values ^= values >> np.uint64(15)
    values = (values * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    values ^= values >> np.uint64(16)
    base = (values.astype(np.float64) + 0.5) / float(2**32)
    offsets = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    return np.mod(base[:, None] + offsets[None, :], 1.0).astype(np.float32)


def validate_probability_tensor(
    probabilities: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    if probabilities.shape != expected_shape or probabilities.dtype != np.float16:
        raise ValueError("probability tensor has the wrong shape or dtype")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probability tensor is non-finite or negative")
    sums = probabilities.sum(axis=0, dtype=np.float32)
    if not np.allclose(sums, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError("probability tensor does not sum to one")


def validate_one_to_one_ade20k_ontology(ontology: Any) -> np.ndarray:
    """Return the complete ADE lookup after validating the identity mapping."""

    lookup = np.asarray(ontology.ade_to_project)
    if int(ontology.class_count) != 150:
        raise ValueError("soft lift requires exactly 150 ADE20K classes")
    if lookup.shape != (256,) or lookup.dtype != np.uint16:
        raise ValueError("soft lift requires the complete uint16 ADE lookup")
    if not np.array_equal(lookup[:150], np.arange(1, 151, dtype=np.uint16)):
        raise ValueError("soft lift requires the validated one-to-one ADE20K ontology")
    if np.any(lookup[150:] != 0):
        raise ValueError("non-ADE and ignore lookup entries must map to zero")
    return lookup


def sample_probability_classes(
    probabilities: np.ndarray,
    *,
    sample_count: int,
    camera_index: int,
    torch: Any,
) -> np.ndarray:
    """Sample all strata on CUDA and return zero-based ADE classes on CPU."""

    class_count, height, width = probabilities.shape
    thresholds = deterministic_stratified_thresholds(
        height, width, sample_count, camera_index
    )
    probability_cuda = torch.as_tensor(
        np.asarray(probabilities), device="cuda", dtype=torch.float32
    )
    probability_cuda /= probability_cuda.sum(dim=0, keepdim=True).clamp_min(1e-12)
    cdf = torch.cumsum(probability_cuda, dim=0).permute(1, 2, 0).contiguous()
    threshold_cuda = torch.from_numpy(thresholds.reshape(height, width, sample_count)).to(
        device="cuda"
    )
    sampled = torch.searchsorted(cdf, threshold_cuda, right=False)
    sampled.clamp_(max=class_count - 1)
    output = sampled.permute(2, 0, 1).to(dtype=torch.uint8).cpu().numpy()
    del probability_cuda, cdf, threshold_cuda, sampled
    torch.cuda.empty_cache()
    return output


def normalize_soft_support(
    used_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visible indices and one normalized 150-class distribution each."""

    used = np.asarray(used_count, dtype=np.float32)
    if used.ndim != 2 or used.shape[0] != 151:
        raise ValueError("soft support must contain zero plus 150 project rows")
    if not np.isfinite(used).all() or np.any(used < 0.0):
        raise ValueError("soft support must be finite and non-negative")
    total = used.sum(axis=0, dtype=np.float32)
    indices = np.flatnonzero(total > 0.0).astype(np.uint32)
    if not indices.size:
        return indices, np.zeros((0, 150), dtype=np.float16), total
    probabilities = (used[1:, indices] / total[indices][None, :]).T
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise RuntimeError("per-camera Gaussian probability mass does not sum to one")
    return indices, probabilities.astype(np.float16), total


def winner_agreement(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    first_indices, first_probs, _ = normalize_soft_support(first)
    second_indices, second_probs, _ = normalize_soft_support(second)
    common, first_pos, second_pos = np.intersect1d(
        first_indices, second_indices, assume_unique=True, return_indices=True
    )
    if not common.size:
        return {"common_visible_gaussian_count": 0}
    first_winner = np.argmax(first_probs[first_pos], axis=1)
    second_winner = np.argmax(second_probs[second_pos], axis=1)
    return {
        "common_visible_gaussian_count": int(common.size),
        "winner_agreement_count": int(np.count_nonzero(first_winner == second_winner)),
        "winner_agreement_ratio": float(np.mean(first_winner == second_winner)),
    }


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
    parser.add_argument("--probability-manifest", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--sample-count", default=FIXED_SAMPLE_COUNT, type=int)
    args = parser.parse_args()

    if args.sample_count != FIXED_SAMPLE_COUNT:
        raise ValueError(f"the reviewed estimator requires {FIXED_SAMPLE_COUNT} samples")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.probability_manifest.read_text(encoding="utf-8"))
    cache_report = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    if manifest.get("contract") != PROBABILITY_CONTRACT:
        raise ValueError("input is not a complete DINOv3 probability cache")
    dense = cache_report.get("dense_probability_cache", {})
    if not isinstance(dense, dict) or dense.get("available") is not True:
        raise ValueError("selected cache report did not validate dense probabilities")
    if [int(f["camera_index"]) for f in manifest["frames"]] != [
        int(v) for v in cache_report["camera_indices"]
    ]:
        raise ValueError("probability cameras differ from automatic selected prefix")

    ontology = load_ontology(args.ontology)
    ade_to_project = validate_one_to_one_ade20k_ontology(ontology)
    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    pipeline = default_pipeline()
    background = background_tensor(False)
    vote_dir = args.output_dir / "view_probabilities"
    vote_dir.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []

    with torch.no_grad():
        for frame in manifest["frames"]:
            camera_index = int(frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            height, width = int(camera.image_height), int(camera.image_width)
            probability_path = args.input_dir / str(frame["probability_file"])
            probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
            validate_probability_tensor(probabilities, (150, height, width))
            samples = sample_probability_classes(
                probabilities,
                sample_count=args.sample_count,
                camera_index=camera_index,
                torch=torch,
            )
            accumulator = None
            half_accumulator = None
            for sample_index, raw_class in enumerate(samples):
                project_ids = ade_to_project[raw_class]
                gt_mask = torch.from_numpy(project_ids.astype(np.float32)).to("cuda")
                package = render_flashsplat(
                    camera, gaussians, modules, pipeline, background,
                    gt_mask=gt_mask, obj_num=151,
                )
                current = package["used_count"].detach().float()
                if current.shape[0] == 152:
                    if int(torch.count_nonzero(current[-1]).item()) != 0:
                        raise ValueError("FlashSplat sentinel row contains support")
                    current = current[:-1]
                if tuple(current.shape) != (151, gaussian_count):
                    raise ValueError("unexpected FlashSplat soft-support shape")
                accumulator = current.clone() if accumulator is None else accumulator.add_(current)
                if sample_index + 1 == args.sample_count // 2:
                    half_accumulator = accumulator.clone()
                del package, current, gt_mask
            assert accumulator is not None and half_accumulator is not None
            used = accumulator.cpu().numpy()
            half_used = half_accumulator.cpu().numpy()
            indices, distributions, visibility = normalize_soft_support(used)
            stem = Path(str(frame["file"])).stem
            index_path = vote_dir / f"{stem}_indices.npy"
            probability_output = vote_dir / f"{stem}_probabilities.npy"
            np.save(index_path, indices, allow_pickle=False)
            np.save(probability_output, distributions, allow_pickle=False)
            convergence = winner_agreement(half_used, used)
            frames.append({
                "file": str(frame["file"]),
                "camera_index": camera_index,
                "camera_id": int(frame["camera_id"]),
                "gaussian_indices_file": index_path.relative_to(args.output_dir).as_posix(),
                "class_probabilities_file": probability_output.relative_to(args.output_dir).as_posix(),
                "visible_gaussian_count": int(indices.size),
                "probability_shape": [int(indices.size), 150],
                "probability_dtype": "float16",
                "sample_convergence_first_half_vs_full": convergence,
                "total_flashsplat_support": float(visibility.sum(dtype=np.float64)),
            })
            print(
                f"soft-lifted {frame['file']}: visible={indices.size} "
                f"half/full={convergence.get('winner_agreement_ratio', 0.0):.6f}"
            )
            del accumulator, half_accumulator, used, half_used, samples, probabilities
            torch.cuda.empty_cache()

    output = {
        "source": SOURCE,
        "contract": CONTRACT,
        "report_only": True,
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "probability_manifest": str(args.probability_manifest),
        "selected_cache_report": str(args.selected_cache_report),
        "ontology": str(args.ontology),
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "camera_indices": [int(frame["camera_index"]) for frame in frames],
        "render_max_width": args.max_width,
        "estimator": {
            "method": "deterministic_circular_stratified_categorical_sampling",
            "sample_count": args.sample_count,
            "unbiased_per_pixel_expectation": True,
            "fixed_hash": "uint32_mix_7feb352d_846ca68b",
            "flashsplat_native_soft_vector_input_available": False,
        },
        "camera_evidence_scale": "each_camera_gaussian_distribution_sums_to_one",
        "probability_manifest_sha256": sha256_file(args.probability_manifest),
        "selected_cache_report_sha256": sha256_file(args.selected_cache_report),
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "class_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "semantic_ply_written": False,
        "frames": frames,
    }
    (args.output_dir / "soft_vote_manifest.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
