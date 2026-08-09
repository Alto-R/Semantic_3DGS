#!/usr/bin/env python3
"""Report-only leave-one-camera-out audit of soft DINOv3 evidence fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
)
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.calibrated_probability_fusion import (
    calibrate_probability_distribution,
    load_selected_policy,
)
from scripts.task1.dinov3.lift_soft_probability_view_votes import (
    CONTRACT as SOFT_VOTE_CONTRACT,
    SOURCE as SOFT_VOTE_SOURCE,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    boundary_mask,
    quantile_summary,
    render_binary_project_ids,
    save_visuals,
)


SOURCE = "dinov3_soft_probability_round_trip_fidelity_audit"
CONTRACT = "report_only_leave_one_camera_out_soft_probability_fusion_v1"
CALIBRATED_SOURCE = "dinov3_calibrated_probability_round_trip_fidelity_audit"
CALIBRATED_CONTRACT = (
    "report_only_leave_one_camera_out_calibrated_probability_fusion_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_camera_distribution(
    indices: np.ndarray,
    probabilities: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices)
    values = np.asarray(probabilities)
    if idx.ndim != 1 or values.shape != (idx.size, class_count):
        raise ValueError("soft camera evidence has an invalid shape")
    if idx.size and (np.any(idx < 0) or int(idx.max()) >= gaussian_count):
        raise ValueError("soft camera evidence references an invalid Gaussian")
    if idx.size and np.any(idx[1:] <= idx[:-1]):
        raise ValueError("soft camera Gaussian indices must be unique and sorted")
    values32 = values.astype(np.float32)
    if not np.isfinite(values32).all() or np.any(values32 < 0.0):
        raise ValueError("soft camera probabilities are non-finite or negative")
    sums = values32.sum(axis=1, dtype=np.float32)
    if idx.size and not np.allclose(sums, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError("soft camera Gaussian probabilities do not sum to one")
    values32 /= sums[:, None]
    return idx.astype(np.int64, copy=False), values32


def confusion_counts(
    source_project: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    """Return a uint64 source-by-prediction confusion matrix."""

    source = np.asarray(source_project)
    target = np.asarray(predicted)
    mask = np.asarray(valid, dtype=bool)
    if source.shape != target.shape or source.shape != mask.shape:
        raise ValueError("confusion inputs must have identical shapes")
    if class_count < 1:
        raise ValueError("class_count must be positive")
    if mask.any() and (
        np.any(source[mask] < 0)
        or np.any(source[mask] > class_count)
        or np.any(target[mask] < 0)
        or np.any(target[mask] > class_count)
    ):
        raise ValueError("confusion inputs contain an invalid project class")
    side = class_count + 1
    flat = source[mask].astype(np.int64) * side + target[mask].astype(np.int64)
    return np.bincount(flat, minlength=side**2).reshape(side, side).astype(
        np.uint64, copy=False
    )


def soft_consensus(
    evidence: np.ndarray,
    camera_count: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Choose a class only after equal-camera probability fusion."""

    values = np.asarray(evidence)
    counts = np.asarray(camera_count)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("evidence must have shape classes x gaussians")
    if counts.shape != (values.shape[1],) or chunk_size < 1:
        raise ValueError("camera_count shape or chunk_size is invalid")
    gaussian_count = values.shape[1]
    labels = np.zeros(gaussian_count, dtype=np.uint16)
    winning_probability = np.zeros(gaussian_count, dtype=np.float32)
    margin = np.zeros(gaussian_count, dtype=np.float32)
    normalized_entropy = np.zeros(gaussian_count, dtype=np.float32)
    tie = np.zeros(gaussian_count, dtype=bool)
    log_classes = np.float32(np.log(values.shape[0]))
    for start in range(0, gaussian_count, chunk_size):
        end = min(start + chunk_size, gaussian_count)
        local = np.array(values[:, start:end], dtype=np.float32, copy=True)
        np.maximum(local, 0.0, out=local)
        local_counts = counts[start:end]
        total = local.sum(axis=0, dtype=np.float32)
        valid = (local_counts >= 2) & (total > 0.0)
        if not valid.any():
            continue
        winner = np.argmax(local, axis=0)
        top_two = np.partition(local, -2, axis=0)[-2:]
        maximum = top_two[-1]
        second = top_two[-2]
        local_tie = maximum == second
        accepted = valid & ~local_tie
        labels[start:end][accepted] = winner[accepted].astype(np.uint16) + 1
        winning_probability[start:end][valid] = maximum[valid] / total[valid]
        margin[start:end][valid] = (maximum[valid] - second[valid]) / total[valid]
        probabilities = np.divide(
            local, total[None, :], out=np.zeros_like(local), where=total[None, :] > 0
        )
        logarithms = np.zeros_like(probabilities)
        np.log(probabilities, out=logarithms, where=probabilities > 0)
        entropy = -np.sum(
            probabilities * logarithms, axis=0, dtype=np.float32
        ) / log_classes
        normalized_entropy[start:end][valid] = entropy[valid]
        tie[start:end] = valid & local_tie
    return {
        "labels": labels,
        "winning_probability": winning_probability,
        "top1_top2_margin": margin,
        "normalized_entropy": normalized_entropy,
        "exact_tie": tie,
    }


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-dir", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--soft-vote-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--calibration-policy-report", type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required = [
        args.selected_cache_report,
        args.soft_vote_manifest,
        args.ontology,
        args.source_view_dir / "view_manifest.json",
        args.source_view_dir / "dinov3_manifest.json",
    ]
    if args.calibration_policy_report is not None:
        required.append(args.calibration_policy_report)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256_file(path) for path in required}
    cache = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    votes = json.loads(args.soft_vote_manifest.read_text(encoding="utf-8"))
    views = json.loads((args.source_view_dir / "view_manifest.json").read_text(encoding="utf-8"))
    dino = json.loads((args.source_view_dir / "dinov3_manifest.json").read_text(encoding="utf-8"))
    if votes.get("source") != SOFT_VOTE_SOURCE or votes.get("contract") != SOFT_VOTE_CONTRACT:
        raise ValueError("input is not the reviewed soft probability vote cache")
    if votes.get("accepted_gaussian_labels_written") is not False:
        raise ValueError("soft vote cache is not report-only")
    expected = [int(v) for v in cache.get("camera_indices", [])]
    for label, manifest in (("view", views), ("DINOv3", dino), ("soft vote", votes)):
        frames = manifest.get("frames", [])
        if [int(frame["camera_index"]) for frame in frames] != expected:
            raise ValueError(f"{label} cameras differ from the automatic selected prefix")
    gaussian_count = int(votes.get("gaussian_count", -1))
    if gaussian_count < 1:
        raise ValueError("soft vote manifest has an invalid Gaussian count")
    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    calibration_policy = None
    calibration_policy_report = None
    if args.calibration_policy_report is not None:
        calibration_policy, calibration_policy_report = load_selected_policy(
            args.calibration_policy_report,
            class_count=class_count,
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mask_dir = args.output_dir / "heldout_masks"
    overlay_dir = args.output_dir / "heldout_overlays"
    disagreement_dir = args.output_dir / "heldout_disagreement"
    for directory in (mask_dir, overlay_dir, disagreement_dir):
        directory.mkdir()

    with tempfile.TemporaryDirectory(prefix="soft_fusion_", dir=args.output_dir) as temporary:
        evidence = np.memmap(
            Path(temporary) / "evidence.float32", mode="w+", dtype=np.float32,
            shape=(class_count, gaussian_count),
        )
        evidence[:] = 0
        camera_counts = np.zeros(gaussian_count, dtype=np.uint16)
        vote_root = args.soft_vote_manifest.parent
        for frame in votes["frames"]:
            indices = np.load(vote_root / str(frame["gaussian_indices_file"]), mmap_mode="r")
            probabilities = np.load(vote_root / str(frame["class_probabilities_file"]), mmap_mode="r")
            idx, values = validate_camera_distribution(
                indices, probabilities,
                gaussian_count=gaussian_count, class_count=class_count,
            )
            contribution = (
                calibrate_probability_distribution(values, calibration_policy)
                if calibration_policy is not None
                else values
            )
            support = contribution.sum(axis=1, dtype=np.float32) > 0.0
            evidence[:, idx] += contribution.T
            camera_counts[idx[support]] += np.uint16(1)
        evidence.flush()

        cameras = load_cameras(args.model_path)
        modules = load_flashsplat(args.flashsplat_root)
        ply_path = point_cloud_path(args.model_path, args.iteration)
        if Path(str(votes.get("ply_path", ""))).resolve() != ply_path.resolve():
            raise ValueError("soft vote cache belongs to another Gaussian model")
        gaussians = load_gaussians(modules, ply_path, args.sh_degree)
        if int(gaussians.get_xyz.shape[0]) != gaussian_count:
            raise ValueError("Gaussian count differs from soft vote cache")
        pipeline = default_pipeline()
        background = background_tensor(False)
        lookup = ontology.ade_to_project
        render_max_width = int(votes["render_max_width"])
        rgb_dir = args.source_view_dir / "rgb_renders"
        aggregate = {key: 0 for key in (
            "pixels", "projected", "agreed", "boundary_pixels",
            "boundary_projected", "boundary_agreed", "interior_pixels",
            "interior_projected", "interior_agreed",
        )}
        confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.uint64)
        camera_summaries: list[dict[str, Any]] = []

        for vote_frame, dino_frame in zip(votes["frames"], dino["frames"]):
            indices = np.load(vote_root / str(vote_frame["gaussian_indices_file"]), mmap_mode="r")
            probabilities = np.load(vote_root / str(vote_frame["class_probabilities_file"]), mmap_mode="r")
            idx, values = validate_camera_distribution(
                indices, probabilities,
                gaussian_count=gaussian_count, class_count=class_count,
            )
            contribution = (
                calibrate_probability_distribution(values, calibration_policy)
                if calibration_policy is not None
                else values
            )
            contribution_support = contribution.sum(axis=1, dtype=np.float32) > 0.0
            original_evidence = np.asarray(evidence[:, idx], dtype=np.float32).copy()
            evidence[:, idx] = original_evidence - contribution.T
            remaining = camera_counts.copy()
            remaining[idx[contribution_support]] -= np.uint16(1)
            consensus = soft_consensus(evidence, remaining, chunk_size=args.chunk_size)
            evidence[:, idx] = original_evidence
            del original_evidence

            camera_index = int(vote_frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, render_max_width)
            predicted, valid, binary_margin = render_binary_project_ids(
                consensus["labels"], camera, gaussians, modules, pipeline, background,
                class_count=class_count,
            )
            with np.load(args.source_view_dir / str(dino_frame["segment_file"]), allow_pickle=False) as segment:
                source_project = lookup[np.asarray(segment["class_id"], dtype=np.uint8)]
            if source_project.shape != predicted.shape:
                raise ValueError("held-out source and projection shapes differ")
            boundary = boundary_mask(source_project)
            interior = ~boundary
            agreement = valid & (predicted == source_project)
            confusion += confusion_counts(
                source_project,
                predicted,
                valid,
                class_count=class_count,
            )
            pixels = int(source_project.size)
            projected = int(np.count_nonzero(valid))
            summary = {
                "file": str(vote_frame["file"]),
                "camera_index": camera_index,
                "camera_id": int(vote_frame["camera_id"]),
                "source_pixel_count": pixels,
                "projected_pixel_count": projected,
                "projected_pixel_ratio": projected / pixels,
                "heldout_agreement_count": int(np.count_nonzero(agreement)),
                "heldout_agreement_of_projected": float(np.mean(agreement[valid])) if projected else 0.0,
                "boundary_pixel_count": int(np.count_nonzero(boundary)),
                "boundary_projected_count": int(np.count_nonzero(valid & boundary)),
                "boundary_agreement_count": int(np.count_nonzero(agreement & boundary)),
                "interior_pixel_count": int(np.count_nonzero(interior)),
                "interior_projected_count": int(np.count_nonzero(valid & interior)),
                "interior_agreement_count": int(np.count_nonzero(agreement & interior)),
                "remaining_camera_count": quantile_summary(remaining[consensus["labels"] > 0]),
                "fused_winning_probability": quantile_summary(consensus["winning_probability"][consensus["labels"] > 0]),
                "fused_top1_top2_margin": quantile_summary(consensus["top1_top2_margin"][consensus["labels"] > 0]),
                "fused_normalized_entropy": quantile_summary(consensus["normalized_entropy"][consensus["labels"] > 0]),
                "exact_tie_gaussian_count": int(np.count_nonzero(consensus["exact_tie"])),
                "binary_projection_margin": quantile_summary(binary_margin[valid]),
            }
            camera_summaries.append(summary)
            for key, value in (
                ("pixels", pixels), ("projected", projected),
                ("agreed", summary["heldout_agreement_count"]),
                ("boundary_pixels", summary["boundary_pixel_count"]),
                ("boundary_projected", summary["boundary_projected_count"]),
                ("boundary_agreed", summary["boundary_agreement_count"]),
                ("interior_pixels", summary["interior_pixel_count"]),
                ("interior_projected", summary["interior_projected_count"]),
                ("interior_agreed", summary["interior_agreement_count"]),
            ):
                aggregate[key] += int(value)
            stem = Path(str(vote_frame["file"])).stem
            np.savez_compressed(
                mask_dir / f"{stem}.npz",
                projected_project_id=predicted.astype(np.uint8),
                projection_valid=valid,
                source_boundary=boundary,
                agreement=agreement,
                minimum_binary_margin=binary_margin.astype(np.float16),
            )
            base = np.asarray(Image.open(rgb_dir / str(vote_frame["file"])).convert("RGB"), dtype=np.uint8)
            save_visuals(
                base, predicted, valid, source_project, ontology,
                overlay_dir / str(vote_frame["file"]),
                disagreement_dir / str(vote_frame["file"]),
            )
            print(
                f"{'calibrated' if calibration_policy is not None else 'soft'} "
                f"held out camera {camera_index}: coverage={projected / pixels:.4f} "
                f"agreement={summary['heldout_agreement_of_projected']:.4f}"
            )

    np.savez_compressed(
        args.output_dir / "heldout_confusion_matrix.npz",
        counts=confusion,
        project_ids=np.arange(class_count + 1, dtype=np.uint16),
    )
    hashes_after = {str(path): sha256_file(path) for path in required}
    if hashes_before != hashes_after:
        raise RuntimeError("audit provenance inputs changed during execution")
    pixel_metrics = {
        **aggregate,
        "projected_ratio": aggregate["projected"] / aggregate["pixels"],
        "agreement_of_projected": aggregate["agreed"] / aggregate["projected"] if aggregate["projected"] else 0.0,
        "boundary_agreement_of_projected": aggregate["boundary_agreed"] / aggregate["boundary_projected"] if aggregate["boundary_projected"] else 0.0,
        "interior_agreement_of_projected": aggregate["interior_agreed"] / aggregate["interior_projected"] if aggregate["interior_projected"] else 0.0,
    }
    per_source_class: list[dict[str, Any]] = []
    for item in ontology.classes:
        row = confusion[item.project_id]
        total = int(row.sum())
        if total:
            per_source_class.append({
                "project_id": item.project_id,
                "class": item.project_class,
                "projected_source_pixel_count": total,
                "correct_pixel_count": int(row[item.project_id]),
                "agreement": int(row[item.project_id]) / total,
            })
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    largest_confusions: list[dict[str, Any]] = []
    for flat_index in np.argsort(off_diagonal.ravel())[::-1][:50]:
        count = int(off_diagonal.ravel()[flat_index])
        if count == 0:
            break
        source_id, predicted_id = np.unravel_index(flat_index, off_diagonal.shape)
        largest_confusions.append({
            "source_project_id": int(source_id),
            "source_class": ontology.by_project_id[int(source_id)].project_class,
            "predicted_project_id": int(predicted_id),
            "predicted_class": ontology.by_project_id[int(predicted_id)].project_class,
            "pixel_count": count,
        })
    is_calibrated = calibration_policy is not None
    report = {
        "source": CALIBRATED_SOURCE if is_calibrated else SOURCE,
        "contract": CALIBRATED_CONTRACT if is_calibrated else CONTRACT,
        "report_only": True,
        "model_path": str(args.model_path),
        "source_view_dir": str(args.source_view_dir),
        "selected_cache_report": str(args.selected_cache_report),
        "soft_vote_manifest": str(args.soft_vote_manifest),
        "camera_count": len(expected),
        "gaussian_count": gaussian_count,
        "camera_indices": expected,
        "fusion_policy": (
            "automatically_selected_global_calibrated_probability_sum_then_argmax"
            if is_calibrated
            else "equal_camera_full_probability_sum_then_argmax"
        ),
        "calibration_policy_report": (
            str(args.calibration_policy_report) if is_calibrated else None
        ),
        "calibration_policy": calibration_policy,
        "calibration_policy_selection": (
            {
                "source": calibration_policy_report["source"],
                "contract": calibration_policy_report["contract"],
                "selection_rule": calibration_policy_report["selection_rule"],
                "manual_candidate_selection_used": calibration_policy_report[
                    "manual_candidate_selection_used"
                ],
            }
            if calibration_policy_report is not None
            else None
        ),
        "heldout_policy": "exclude_target_camera_probability_distribution_before_fusion",
        "minimum_remaining_camera_count": 2,
        "confidence_or_majority_threshold_used": False,
        "exact_tie_policy": "abstain",
        "projection_policy": "eight_binary_project_id_feature_renders",
        "heldout_pixel_metrics": pixel_metrics,
        "per_camera": camera_summaries,
        "per_source_class": per_source_class,
        "largest_off_diagonal_confusions": largest_confusions,
        "input_sha256": hashes_before,
        "manual_candidate_selection_used": False,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_name = (
        "calibrated_probability_round_trip_report.json"
        if is_calibrated
        else "soft_probability_round_trip_report.json"
    )
    (args.output_dir / report_name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "contract": report["contract"],
                "calibration_policy": calibration_policy,
                "heldout_pixel_metrics": pixel_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
