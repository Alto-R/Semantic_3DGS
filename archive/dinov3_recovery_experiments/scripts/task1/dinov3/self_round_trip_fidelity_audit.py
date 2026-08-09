#!/usr/bin/env python3
"""Report-only same-camera FlashSplat self-round-trip fidelity audit.

The audit consumes the complete dense per-camera Gaussian class distributions
already produced for the leave-one-camera-out audit.  It reduces one camera's
distribution to one unique hard Gaussian identity (or abstention), immediately
renders those identities back into the same camera, and compares that result
with the camera's cached DINOv3 map.  No cross-camera fusion is performed.

This isolates the combined loss from dense hard-pixel FlashSplat lifting,
per-camera Gaussian hardening, and same-camera projection.  DINOv3 inference
and FlashSplat lifting are not rerun, and no semantic labels or PLY are written.
"""

from __future__ import annotations

import argparse
import json
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
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    boundary_mask,
    collapse_camera_distribution,
    quantile_summary,
    render_binary_project_ids,
    save_visuals,
    sha256_file,
    validate_provenance,
)


SOURCE = "dinov3_cached_flashsplat_same_camera_self_round_trip_fidelity_audit"
CONTRACT = "report_only_same_camera_flashsplat_self_round_trip_fidelity_v1"


def pixel_metric_counts(
    source: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Return exact same-camera pixel counts plus boundary and agreement masks."""

    source_ids = np.asarray(source)
    predicted_ids = np.asarray(predicted)
    projection_valid = np.asarray(valid, dtype=bool)
    if source_ids.ndim != 2:
        raise ValueError("source class map must be two-dimensional")
    if not (
        source_ids.shape == predicted_ids.shape == projection_valid.shape
    ):
        raise ValueError("source, predicted, and valid maps must have identical shapes")
    boundary = boundary_mask(source_ids)
    interior = ~boundary
    agreement = projection_valid & (predicted_ids == source_ids)
    return (
        {
            "pixels": int(source_ids.size),
            "projected": int(np.count_nonzero(projection_valid)),
            "agreed": int(np.count_nonzero(agreement)),
            "boundary_pixels": int(np.count_nonzero(boundary)),
            "boundary_projected": int(np.count_nonzero(projection_valid & boundary)),
            "boundary_agreed": int(np.count_nonzero(agreement & boundary)),
            "interior_pixels": int(np.count_nonzero(interior)),
            "interior_projected": int(np.count_nonzero(projection_valid & interior)),
            "interior_agreed": int(np.count_nonzero(agreement & interior)),
        },
        boundary,
        agreement,
    )


def metric_ratios(counts: dict[str, int]) -> dict[str, float]:
    """Return safe ratios for an exact pixel-count dictionary."""

    def ratio(numerator: str, denominator: str) -> float:
        total = int(counts[denominator])
        return float(counts[numerator]) / total if total else 0.0

    return {
        "projected_ratio": ratio("projected", "pixels"),
        "agreement_of_projected": ratio("agreed", "projected"),
        "boundary_projected_ratio": ratio("boundary_projected", "boundary_pixels"),
        "boundary_agreement_of_projected": ratio(
            "boundary_agreed", "boundary_projected"
        ),
        "interior_projected_ratio": ratio("interior_projected", "interior_pixels"),
        "interior_agreement_of_projected": ratio(
            "interior_agreed", "interior_projected"
        ),
    }


def aggregate_hardening_metrics(
    camera_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine exact per-camera hardening counts without averaging cameras."""

    count_fields = (
        "visible_gaussian_count",
        "unique_camera_winner_count",
        "camera_abstain_count",
        "exact_tie_count",
    )
    output: dict[str, Any] = {
        field: sum(int(summary[field]) for summary in camera_summaries)
        for field in count_fields
    }
    winning_count = sum(
        int(summary["winning_mass"].get("count", 0))
        for summary in camera_summaries
    )
    weighted_sum = sum(
        float(summary["winning_mass"].get("mean", 0.0))
        * int(summary["winning_mass"].get("count", 0))
        for summary in camera_summaries
    )
    visible = int(output["visible_gaussian_count"])
    output["unique_winner_ratio_of_visible"] = (
        int(output["unique_camera_winner_count"]) / visible if visible else 0.0
    )
    output["winning_mass_count"] = winning_count
    output["winning_mass_mean"] = weighted_sum / winning_count if winning_count else 0.0
    output["per_camera_winning_mass_mean"] = quantile_summary(
        np.asarray(
            [
                float(summary["winning_mass"].get("mean", 0.0))
                for summary in camera_summaries
                if int(summary["winning_mass"].get("count", 0)) > 0
            ],
            dtype=np.float64,
        )
    )
    return output


def confusion_rows(
    confusion: np.ndarray,
    ontology: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-source-class agreement and largest off-diagonal confusions."""

    class_rows: list[dict[str, Any]] = []
    for item in ontology.classes:
        row = confusion[item.project_id]
        total = int(row.sum())
        correct = int(row[item.project_id])
        if total:
            class_rows.append(
                {
                    "project_id": item.project_id,
                    "class": item.project_class,
                    "projected_source_pixel_count": total,
                    "correct_pixel_count": correct,
                    "agreement": correct / total,
                }
            )

    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    largest: list[dict[str, Any]] = []
    for flat_index in np.argsort(off_diagonal.ravel())[::-1][:50]:
        count = int(off_diagonal.ravel()[flat_index])
        if count == 0:
            break
        source_id, predicted_id = np.unravel_index(flat_index, off_diagonal.shape)
        largest.append(
            {
                "source_project_id": int(source_id),
                "source_class": ontology.by_project_id[int(source_id)].project_class,
                "predicted_project_id": int(predicted_id),
                "predicted_class": ontology.by_project_id[
                    int(predicted_id)
                ].project_class,
                "pixel_count": count,
            }
        )
    return class_rows, largest


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-dir", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--vote-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required_paths = (
        args.selected_cache_report,
        args.vote_manifest,
        args.ontology,
        args.source_view_dir / "view_manifest.json",
        args.source_view_dir / "dinov3_manifest.json",
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256_file(path) for path in required_paths}
    cache_report = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    view_manifest = json.loads(
        (args.source_view_dir / "view_manifest.json").read_text(encoding="utf-8")
    )
    dino_manifest = json.loads(
        (args.source_view_dir / "dinov3_manifest.json").read_text(encoding="utf-8")
    )
    vote_manifest = json.loads(args.vote_manifest.read_text(encoding="utf-8"))
    camera_indices = validate_provenance(
        cache_report, view_manifest, dino_manifest, vote_manifest
    )
    source_root = args.source_view_dir.parents[1]
    if Path(str(cache_report.get("output_dir", ""))).resolve() != source_root.resolve():
        raise ValueError("selected-cache report belongs to a different output root")
    if Path(str(vote_manifest.get("model_path", ""))).resolve() != args.model_path.resolve():
        raise ValueError("vote manifest belongs to a different Gaussian model")
    if Path(str(vote_manifest.get("segmentation_manifest", ""))).resolve() != (
        args.source_view_dir / "dinov3_manifest.json"
    ).resolve():
        raise ValueError("vote manifest belongs to a different DINOv3 cache")
    expected_manifest_hashes = cache_report.get("output_manifest_sha256", {})
    if not isinstance(expected_manifest_hashes, dict):
        raise ValueError("selected-cache report is missing manifest hashes")
    if expected_manifest_hashes.get("view_manifest") != hashes_before[
        str(args.source_view_dir / "view_manifest.json")
    ]:
        raise ValueError("view manifest hash differs from selected-cache provenance")
    if expected_manifest_hashes.get("dinov3_manifest") != hashes_before[
        str(args.source_view_dir / "dinov3_manifest.json")
    ]:
        raise ValueError("DINOv3 manifest hash differs from selected-cache provenance")

    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    gaussian_count = int(vote_manifest.get("gaussian_count", -1))
    if gaussian_count < 1:
        raise ValueError("vote manifest has an invalid Gaussian count")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mask_dir = args.output_dir / "self_masks"
    overlay_dir = args.output_dir / "self_overlays"
    disagreement_dir = args.output_dir / "self_disagreement"
    for directory in (mask_dir, overlay_dir, disagreement_dir):
        directory.mkdir()

    cameras = load_cameras(args.model_path)
    if any(index < 0 or index >= len(cameras) for index in camera_indices):
        raise ValueError("selected cache references a camera outside cameras.json")
    expected_ids = [int(value) for value in cache_report["camera_ids"]]
    if [int(cameras[index]["id"]) for index in camera_indices] != expected_ids:
        raise ValueError("selected-cache camera IDs differ from current cameras.json")
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    if Path(str(vote_manifest.get("ply_path", ""))).resolve() != ply_path.resolve():
        raise ValueError("vote manifest belongs to a different Gaussian PLY")
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    if int(gaussians.get_xyz.shape[0]) != gaussian_count:
        raise ValueError("model Gaussian count differs from vote manifest")
    pipeline = default_pipeline()
    background = background_tensor(False)
    render_max_width = int(cache_report["render"]["max_width"])
    lookup = ontology.ade_to_project
    rgb_dir = args.source_view_dir / "rgb_renders"
    confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.uint64)
    aggregate = {
        "pixels": 0,
        "projected": 0,
        "agreed": 0,
        "boundary_pixels": 0,
        "boundary_projected": 0,
        "boundary_agreed": 0,
        "interior_pixels": 0,
        "interior_projected": 0,
        "interior_agreed": 0,
    }
    camera_summaries: list[dict[str, Any]] = []

    for vote_frame, dino_frame in zip(
        vote_manifest["frames"], dino_manifest["frames"]
    ):
        vote_path = args.vote_manifest.parent / str(vote_frame["vote_file"])
        if not vote_path.is_file():
            raise FileNotFoundError(vote_path)
        with np.load(vote_path, allow_pickle=False) as data:
            self_labels, hardening_summary = collapse_camera_distribution(
                data["indices"],
                data["class_ids"],
                data["weights"],
                gaussian_count=gaussian_count,
                class_count=class_count,
            )
        camera_index = int(vote_frame["camera_index"])
        camera = make_camera(cameras[camera_index], modules, render_max_width)
        predicted, valid, binary_margin = render_binary_project_ids(
            self_labels,
            camera,
            gaussians,
            modules,
            pipeline,
            background,
            class_count=class_count,
        )
        segment_path = args.source_view_dir / str(dino_frame["segment_file"])
        with np.load(segment_path, allow_pickle=False) as segment:
            raw = np.asarray(segment["class_id"], dtype=np.uint8)
        source_project = lookup[raw]
        counts, boundary, agreement = pixel_metric_counts(
            source_project, predicted, valid
        )
        ratios = metric_ratios(counts)

        flat = source_project[valid].astype(np.int64) * (
            class_count + 1
        ) + predicted[valid].astype(np.int64)
        confusion += np.bincount(
            flat, minlength=(class_count + 1) ** 2
        ).reshape(class_count + 1, class_count + 1).astype(np.uint64)
        for key in aggregate:
            aggregate[key] += counts[key]

        frame_summary = {
            "file": str(vote_frame["file"]),
            "camera_index": camera_index,
            "camera_id": int(vote_frame["camera_id"]),
            "source_pixel_count": counts["pixels"],
            "projected_pixel_count": counts["projected"],
            "projected_pixel_ratio": ratios["projected_ratio"],
            "self_agreement_count": counts["agreed"],
            "self_agreement_of_projected": ratios["agreement_of_projected"],
            "boundary_pixel_count": counts["boundary_pixels"],
            "boundary_projected_count": counts["boundary_projected"],
            "boundary_agreement_count": counts["boundary_agreed"],
            "boundary_agreement_of_projected": ratios[
                "boundary_agreement_of_projected"
            ],
            "interior_pixel_count": counts["interior_pixels"],
            "interior_projected_count": counts["interior_projected"],
            "interior_agreement_count": counts["interior_agreed"],
            "interior_agreement_of_projected": ratios[
                "interior_agreement_of_projected"
            ],
            "binary_projection_margin": quantile_summary(binary_margin[valid]),
            **hardening_summary,
        }
        camera_summaries.append(frame_summary)

        stem = Path(str(vote_frame["file"])).stem
        np.savez_compressed(
            mask_dir / f"{stem}.npz",
            projected_project_id=predicted.astype(np.uint8),
            projection_valid=valid,
            source_boundary=boundary,
            agreement=agreement,
            minimum_binary_margin=binary_margin.astype(np.float16),
        )
        base_rgb = np.asarray(
            Image.open(rgb_dir / str(vote_frame["file"])).convert("RGB"),
            dtype=np.uint8,
        )
        save_visuals(
            base_rgb,
            predicted,
            valid,
            source_project,
            ontology,
            overlay_dir / str(vote_frame["file"]),
            disagreement_dir / str(vote_frame["file"]),
        )
        print(
            f"self camera {camera_index}: "
            f"coverage={ratios['projected_ratio']:.4f} "
            f"agreement={ratios['agreement_of_projected']:.4f}"
        )

    class_rows, largest_confusions = confusion_rows(confusion, ontology)
    np.savez_compressed(
        args.output_dir / "self_confusion_matrix.npz",
        counts=confusion,
        project_ids=np.arange(class_count + 1, dtype=np.uint16),
    )

    hashes_after = {str(path): sha256_file(path) for path in required_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("audit provenance inputs changed during execution")
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "report_only": True,
        "model_path": str(args.model_path),
        "source_view_dir": str(args.source_view_dir),
        "selected_cache_report": str(args.selected_cache_report),
        "vote_manifest": str(args.vote_manifest),
        "ontology": str(args.ontology),
        "camera_count": len(camera_indices),
        "gaussian_count": gaussian_count,
        "camera_indices": camera_indices,
        "measurement_scope": (
            "cached_DINOv3_hard_argmax_to_FlashSplat_dense_lift_to_unique_"
            "per_camera_Gaussian_winner_to_same_camera_binary_reprojection"
        ),
        "cross_camera_fusion_used": False,
        "camera_vote_policy": "one_unique_max_class_per_camera_and_gaussian_else_abstain",
        "self_round_trip_policy": "render_each_camera_only_from_its_own_cached_Gaussian_identities",
        "projection_policy": "eight_binary_project_id_feature_renders_decoded_at_each_pixel",
        "boundary_definition": "four_connected_change_in_same_camera_DINOv3_project_class",
        "per_camera_gaussian_hardening": aggregate_hardening_metrics(camera_summaries),
        "self_pixel_metrics": {**aggregate, **metric_ratios(aggregate)},
        "per_camera": camera_summaries,
        "per_source_class": class_rows,
        "largest_off_diagonal_confusions": largest_confusions,
        "input_sha256": hashes_before,
        "dinov3_inference_rerun": False,
        "flashsplat_lifting_rerun": False,
        "flashsplat_vote_cache_reused": True,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "self_round_trip_fidelity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "contract": CONTRACT,
                "camera_count": len(camera_indices),
                "self_pixel_metrics": report["self_pixel_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
