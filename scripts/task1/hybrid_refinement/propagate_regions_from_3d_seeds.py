#!/usr/bin/env python3
"""Activate Mask2Former regions from multiview SAM-confirmed 3D seeds.

Initial Grounded-SAM masks are lifted by FlashSplat before this stage. A class
seed contains only Gaussians supported from at least ``min_seed_views``
distinct RGB frames. The seed is rendered into the audited cameras and may
activate a class-agnostic Mask2Former region where the projection agrees.

This is one bounded propagation round. It never consumes its own output as a
new seed, preventing iterative identity drift.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from scripts.task1.hybrid_refinement.refine_sam_with_mask2former_regions import (
    load_mask_stack,
    match_metrics,
    save_comparison_overlay,
    semantic_class,
    write_binary_masks,
)


@dataclass(frozen=True)
class PropagationThresholds:
    projection_threshold: float = 0.05
    min_projection_pixels: int = 100
    min_projection_coverage: float = 0.05
    min_region_coverage: float = 0.20
    min_containment: float = 0.50
    min_region_confidence: float = 0.25
    min_match_score: float = 0.45
    existing_same_class_containment: float = 0.50
    identity_margin: float = 0.10
    max_cross_class_overlap: float = 0.10
    max_masks_per_class_per_view: int = 4

    def validate(self) -> None:
        unit_interval = (
            "projection_threshold",
            "min_projection_coverage",
            "min_region_coverage",
            "min_containment",
            "min_region_confidence",
            "min_match_score",
            "existing_same_class_containment",
            "identity_margin",
            "max_cross_class_overlap",
        )
        for field_name in unit_interval:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.min_projection_pixels < 1:
            raise ValueError("min_projection_pixels must be positive")
        if self.max_masks_per_class_per_view < 1:
            raise ValueError("max_masks_per_class_per_view must be positive")


@dataclass(frozen=True)
class PropagationCandidate:
    class_name: str
    region_id: int
    source_view_count: int
    source_gaussian_count: int
    projection_area: int
    region_area: int
    intersection: int
    iou: float
    containment: float
    projection_coverage: float
    region_coverage: float
    mean_region_confidence: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_support_indices(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        indices = data["indices"].astype(np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{path} indices must be one-dimensional")
    return np.unique(indices)


def build_multiview_seeds(
    proposal_manifest: dict[str, Any],
    support_dir: Path,
    gaussian_count: int,
    min_seed_views: int,
    min_seed_gaussians: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Return per-class Gaussians supported from independent source frames."""

    if min_seed_views < 1 or min_seed_gaussians < 1:
        raise ValueError("Seed view and Gaussian thresholds must be positive")
    supports_by_class_frame: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for proposal in proposal_manifest.get("proposals", []):
        class_name = semantic_class(proposal)
        frame_file = str(proposal.get("frame_file", "")).strip()
        if not frame_file:
            raise ValueError("Every proposal must record frame_file")
        indices = load_support_indices(support_dir / str(proposal["support_file"]))
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= gaussian_count
        ):
            raise IndexError("Proposal support index is outside the Gaussian array")
        supports_by_class_frame[class_name][frame_file].append(indices)

    seeds: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, int]] = {}
    for class_name, supports_by_frame in supports_by_class_frame.items():
        view_support = np.zeros(gaussian_count, dtype=np.uint16)
        for frame_supports in supports_by_frame.values():
            if not frame_supports:
                continue
            frame_union = np.unique(np.concatenate(frame_supports))
            view_support[frame_union] += 1
        seed_indices = np.flatnonzero(view_support >= min_seed_views).astype(np.int64)
        reports[class_name] = {
            "source_view_count": len(supports_by_frame),
            "seed_gaussian_count": int(seed_indices.size),
            "min_seed_views": min_seed_views,
        }
        if seed_indices.size >= min_seed_gaussians:
            seeds[class_name] = seed_indices
    return seeds, reports


def propagation_match_score(metrics: dict[str, float | int]) -> float:
    return (
        0.45 * float(metrics["region_coverage"])
        + 0.25 * float(metrics["containment"])
        + 0.15 * float(metrics["sam_coverage"])
        + 0.15 * float(metrics["mean_region_confidence"])
    )


def _same_class_already_covers_region(
    region_mask: np.ndarray,
    masks: np.ndarray,
    metadata: list[dict[str, Any]],
    class_name: str,
    threshold: float,
) -> bool:
    region_area = int(region_mask.sum())
    for mask, item in zip(masks, metadata):
        if semantic_class(item) != class_name:
            continue
        mask = np.asarray(mask, dtype=bool)
        intersection = int(np.logical_and(mask, region_mask).sum())
        containment = intersection / float(max(min(int(mask.sum()), region_area), 1))
        if containment >= threshold:
            return True
    return False


def _cross_class_overlap(
    region_mask: np.ndarray,
    masks: np.ndarray,
    metadata: list[dict[str, Any]],
    class_name: str,
) -> float:
    region_area = int(region_mask.sum())
    maximum = 0.0
    for mask, item in zip(masks, metadata):
        if semantic_class(item) == class_name:
            continue
        mask = np.asarray(mask, dtype=bool)
        intersection = int(np.logical_and(mask, region_mask).sum())
        containment = intersection / float(max(min(int(mask.sum()), region_area), 1))
        maximum = max(maximum, containment)
    return maximum


def select_propagated_regions(
    masks: np.ndarray,
    metadata: list[dict[str, Any]],
    region_id: np.ndarray,
    region_confidence: np.ndarray,
    projections: dict[str, np.ndarray],
    seed_reports: dict[str, dict[str, int]],
    thresholds: PropagationThresholds,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Append only regions activated by an independently supported 3D seed."""

    thresholds.validate()
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("masks must have shape (K,H,W)")
    if len(metadata) != masks.shape[0]:
        raise ValueError("mask metadata count must match mask stack")
    if region_id.shape != masks.shape[1:] or region_confidence.shape != region_id.shape:
        raise ValueError("Region arrays must match mask dimensions")

    candidates: list[PropagationCandidate] = []
    skipped_existing = 0
    for class_name, projection_score in projections.items():
        if projection_score.shape != region_id.shape:
            raise ValueError("Projection score map must match region arrays")
        projection_mask = projection_score >= thresholds.projection_threshold
        projection_area = int(projection_mask.sum())
        if projection_area < thresholds.min_projection_pixels:
            continue
        for compact_region_id in np.unique(region_id[projection_mask]):
            compact_region_id = int(compact_region_id)
            if compact_region_id == 0:
                continue
            region_mask = region_id == compact_region_id
            if _same_class_already_covers_region(
                region_mask,
                masks,
                metadata,
                class_name,
                thresholds.existing_same_class_containment,
            ):
                skipped_existing += 1
                continue
            metrics = match_metrics(
                projection_mask,
                region_mask,
                region_confidence,
            )
            score = propagation_match_score(metrics)
            if not (
                int(metrics["intersection"]) >= thresholds.min_projection_pixels
                and float(metrics["sam_coverage"])
                >= thresholds.min_projection_coverage
                and float(metrics["region_coverage"]) >= thresholds.min_region_coverage
                and float(metrics["containment"]) >= thresholds.min_containment
                and float(metrics["mean_region_confidence"])
                >= thresholds.min_region_confidence
                and score >= thresholds.min_match_score
            ):
                continue
            seed_report = seed_reports[class_name]
            candidates.append(
                PropagationCandidate(
                    class_name=class_name,
                    region_id=compact_region_id,
                    source_view_count=int(seed_report["source_view_count"]),
                    source_gaussian_count=int(seed_report["seed_gaussian_count"]),
                    projection_area=int(metrics["sam_area"]),
                    region_area=int(metrics["region_area"]),
                    intersection=int(metrics["intersection"]),
                    iou=float(metrics["iou"]),
                    containment=float(metrics["containment"]),
                    projection_coverage=float(metrics["sam_coverage"]),
                    region_coverage=float(metrics["region_coverage"]),
                    mean_region_confidence=float(metrics["mean_region_confidence"]),
                    score=score,
                )
            )

    candidates_by_region: dict[int, list[PropagationCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_region[candidate.region_id].append(candidate)

    accepted: list[PropagationCandidate] = []
    identity_conflicts: list[dict[str, Any]] = []
    per_class_counts: dict[str, int] = defaultdict(int)
    for compact_region_id, region_candidates in candidates_by_region.items():
        ordered = sorted(region_candidates, key=lambda item: item.score, reverse=True)
        best = ordered[0]
        competing = next(
            (item for item in ordered[1:] if item.class_name != best.class_name),
            None,
        )
        if competing is not None and best.score - competing.score < thresholds.identity_margin:
            identity_conflicts.append(
                {
                    "region_id": compact_region_id,
                    "best": best.to_dict(),
                    "competing": competing.to_dict(),
                }
            )
            continue
        if per_class_counts[best.class_name] >= thresholds.max_masks_per_class_per_view:
            continue
        region_mask = region_id == compact_region_id
        cross_class_overlap = _cross_class_overlap(
            region_mask,
            masks,
            metadata,
            best.class_name,
        )
        if cross_class_overlap > thresholds.max_cross_class_overlap:
            continue
        accepted.append(best)
        per_class_counts[best.class_name] += 1

    appended_masks: list[np.ndarray] = []
    appended_metadata: list[dict[str, Any]] = []
    for candidate in accepted:
        region_mask = region_id == candidate.region_id
        rows, columns = np.nonzero(region_mask)
        evidence = {
            "status": "propagated_from_multiview_3d_seed",
            "semantic_identity_source": "grounded_sam_multiview_3d_seed",
            "boundary_source": "mask2former_class_agnostic_region",
            "mask2former_semantic_class_used": False,
            "selected_region_id": candidate.region_id,
            "source_view_count": candidate.source_view_count,
            "source_gaussian_count": candidate.source_gaussian_count,
            "match": candidate.to_dict(),
        }
        appended_masks.append(region_mask)
        appended_metadata.append(
            {
                "source": "multiview_3d_seed_mask2former_region",
                "class": candidate.class_name,
                "class_name": candidate.class_name,
                "phrase": "multiview 3d reprojection",
                "confidence": candidate.score,
                "grounding_score": candidate.score,
                "sam_score": 0.0,
                "area": int(region_mask.sum()),
                "bbox_xyxy": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                ],
                "hybrid_refinement": evidence,
            }
        )

    if appended_masks:
        output_masks = np.concatenate(
            [masks, np.stack(appended_masks, axis=0)],
            axis=0,
        )
    else:
        output_masks = masks.copy()
    report = {
        "candidate_count": len(candidates),
        "accepted_count": len(appended_masks),
        "skipped_existing_same_class": skipped_existing,
        "identity_conflict_count": len(identity_conflicts),
        "identity_conflicts": identity_conflicts,
        "accepted": [candidate.to_dict() for candidate in accepted],
    }
    return output_masks, appended_metadata, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--hybrid-mask-manifest", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--dense-seg-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--min-seed-views", default=2, type=int)
    parser.add_argument("--min-seed-gaussians", default=500, type=int)
    parser.add_argument("--projection-threshold", default=0.05, type=float)
    parser.add_argument("--min-projection-pixels", default=100, type=int)
    parser.add_argument("--min-projection-coverage", default=0.05, type=float)
    parser.add_argument("--min-region-coverage", default=0.20, type=float)
    parser.add_argument("--min-containment", default=0.50, type=float)
    parser.add_argument("--min-region-confidence", default=0.25, type=float)
    parser.add_argument("--min-match-score", default=0.45, type=float)
    parser.add_argument("--existing-same-class-containment", default=0.50, type=float)
    parser.add_argument("--identity-margin", default=0.10, type=float)
    parser.add_argument("--max-cross-class-overlap", default=0.10, type=float)
    parser.add_argument("--max-masks-per-class-per-view", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    thresholds = PropagationThresholds(
        projection_threshold=args.projection_threshold,
        min_projection_pixels=args.min_projection_pixels,
        min_projection_coverage=args.min_projection_coverage,
        min_region_coverage=args.min_region_coverage,
        min_containment=args.min_containment,
        min_region_confidence=args.min_region_confidence,
        min_match_score=args.min_match_score,
        existing_same_class_containment=args.existing_same_class_containment,
        identity_margin=args.identity_margin,
        max_cross_class_overlap=args.max_cross_class_overlap,
        max_masks_per_class_per_view=args.max_masks_per_class_per_view,
    )
    thresholds.validate()

    hybrid_manifest = json.loads(
        args.hybrid_mask_manifest.read_text(encoding="utf-8")
    )
    proposal_manifest = json.loads(
        (args.proposal_dir / "proposal_manifest.json").read_text(encoding="utf-8")
    )
    dense_manifest = json.loads(args.dense_seg_manifest.read_text(encoding="utf-8"))
    dense_frames = {
        int(frame["camera_index"]): frame for frame in dense_manifest.get("frames", [])
    }

    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    seeds, seed_reports = build_multiview_seeds(
        proposal_manifest,
        args.proposal_dir / "proposal_supports",
        gaussian_count,
        args.min_seed_views,
        args.min_seed_gaussians,
    )
    seed_colors: dict[str, torch.Tensor] = {}
    for class_name, seed_indices in seeds.items():
        colors = torch.zeros(
            (gaussian_count, 3),
            dtype=torch.float32,
            device="cuda",
        )
        colors[torch.from_numpy(seed_indices).to(device="cuda", dtype=torch.long)] = 1.0
        seed_colors[class_name] = colors

    cameras = load_cameras(args.model_path)
    pipeline = default_pipeline()
    background = background_tensor(False)
    source_root = args.hybrid_mask_manifest.parent
    dense_root = args.dense_seg_manifest.parent
    mask_stack_dir = args.output_dir / "mask_stacks"
    binary_mask_dir = args.output_dir / "binary_masks"
    overlay_dir = args.output_dir / "overlays"
    for directory in (mask_stack_dir, binary_mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_manifest_path = args.output_dir / "grounded_sam_manifest.json"
    report_path = args.output_dir / "hybrid_propagation_report.json"
    for output_path in (output_manifest_path, report_path):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    output_frames: list[dict[str, Any]] = []
    total_added = 0
    total_conflicts = 0
    with torch.no_grad():
        for source_frame in hybrid_manifest.get("frames", []):
            camera_index = int(source_frame["camera_index"])
            dense_frame = dense_frames.get(camera_index)
            if dense_frame is None:
                raise ValueError(f"Camera {camera_index} is absent from dense segmentation")
            height = int(dense_frame["height"])
            width = int(dense_frame["width"])
            masks = load_mask_stack(
                source_root / "mask_stacks" / str(source_frame["mask_file"]),
                height,
                width,
            )
            metadata = list(source_frame.get("masks", []))
            with np.load(
                dense_root / "seg" / str(dense_frame["seg_file"]),
                allow_pickle=False,
            ) as dense_arrays:
                region_id = dense_arrays["region_id"].astype(np.uint16)
                region_confidence = dense_arrays["region_confidence"].astype(np.float32)

            camera = make_camera(cameras[camera_index], modules, args.max_width)
            projections: dict[str, np.ndarray] = {}
            for class_name, colors in seed_colors.items():
                render = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    override_color=colors,
                )
                projection = render["render"].detach().to(torch.float32).mean(dim=0).cpu().numpy()
                if projection.shape != (height, width):
                    raise ValueError(
                        f"Projection shape {projection.shape} does not match "
                        f"dense frame {(height, width)}"
                    )
                projections[class_name] = projection
                del render

            propagated_masks, appended_metadata, frame_report = select_propagated_regions(
                masks,
                metadata,
                region_id,
                region_confidence,
                projections,
                seed_reports,
                thresholds,
            )
            total_added += int(frame_report["accepted_count"])
            total_conflicts += int(frame_report["identity_conflict_count"])
            output_metadata = metadata + appended_metadata

            filename = str(source_frame["file"])
            stem = Path(filename).stem
            mask_file = f"{stem}.npz"
            np.savez_compressed(
                mask_stack_dir / mask_file,
                masks=propagated_masks.astype(np.uint8),
            )
            binary_mask_files = write_binary_masks(
                propagated_masks,
                output_metadata,
                stem,
                binary_mask_dir,
            )
            rgb = np.asarray(
                Image.open(dense_root / "rgb_renders" / filename).convert("RGB"),
                dtype=np.uint8,
            )
            original_for_overlay = np.concatenate(
                [
                    masks,
                    np.zeros(
                        (len(appended_metadata), height, width),
                        dtype=bool,
                    ),
                ],
                axis=0,
            )
            overlay_audits = [
                {
                    "status": "retained_seed_mask",
                    "mask2former_semantic_class_used": False,
                }
                for _ in metadata
            ] + [item["hybrid_refinement"] for item in appended_metadata]
            save_comparison_overlay(
                rgb,
                original_for_overlay,
                propagated_masks,
                output_metadata,
                overlay_audits,
                overlay_dir / filename,
            )

            output_mask_metadata = []
            for index, item in enumerate(output_metadata):
                output_mask_metadata.append(
                    {
                        **item,
                        "binary_mask_file": binary_mask_files[index],
                        "area": int(propagated_masks[index].sum()),
                    }
                )
            output_frames.append(
                {
                    **source_frame,
                    "mask_file": mask_file,
                    "binary_mask_files": binary_mask_files,
                    "kept_mask_count": int(propagated_masks.shape[0]),
                    "masks": output_mask_metadata,
                    "hybrid_propagation": frame_report,
                }
            )
            print(
                f"propagated {filename}: existing={masks.shape[0]} "
                f"added={len(appended_metadata)}"
            )
            torch.cuda.empty_cache()

    output_manifest = {
        **hybrid_manifest,
        "source": "groundingdino_sam_mask2former_regions_with_3d_propagation",
        "upstream_hybrid_mask_manifest": str(args.hybrid_mask_manifest),
        "seed_proposal_manifest": str(args.proposal_dir / "proposal_manifest.json"),
        "dense_seg_manifest": str(args.dense_seg_manifest),
        "mask2former_semantic_class_used": False,
        "propagation_rounds": 1,
        "propagation_parameters": asdict(thresholds),
        "seed_parameters": {
            "min_seed_views": args.min_seed_views,
            "min_seed_gaussians": args.min_seed_gaussians,
        },
        "seed_reports": seed_reports,
        "output_directories": {
            "mask_stacks": str(mask_stack_dir),
            "binary_masks": str(binary_mask_dir),
            "overlays": str(overlay_dir),
            "rgb_renders": str(dense_root / "rgb_renders"),
        },
        "frames": output_frames,
    }
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    report = {
        "source": output_manifest["source"],
        "upstream_manifest": str(args.hybrid_mask_manifest),
        "output_manifest": str(output_manifest_path),
        "propagation_rounds": 1,
        "seed_reports": seed_reports,
        "seeded_classes": sorted(seeds),
        "frame_count": len(output_frames),
        "propagated_mask_count": total_added,
        "identity_conflict_count": total_conflicts,
        "mask2former_semantic_class_used": False,
        "semantic_labels_written": False,
        "semantic_ply_written": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output_manifest_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
