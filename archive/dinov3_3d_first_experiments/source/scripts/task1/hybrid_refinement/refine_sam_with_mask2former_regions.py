#!/usr/bin/env python3
"""Snap Grounded-SAM masks to class-agnostic Mask2Former regions.

GroundingDINO supplies the semantic identity and SAM supplies the prompted
object mask. Mask2Former contributes only a region boundary: its semantic
class prediction is never consulted by this module.

The output keeps the Grounded-SAM manifest/mask-stack contract so the existing
FlashSplat proposal and 3D grouping stages can consume it unchanged. A
Mask2Former-only region can never create a semantic mask; every output mask
retains the identity of one input Grounded-SAM detection.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.task1.common.semantic_palette import rgb8_for_class


@dataclass(frozen=True)
class RefinementThresholds:
    min_iou: float = 0.45
    min_containment: float = 0.80
    min_sam_coverage: float = 0.10
    min_region_confidence: float = 0.25
    min_match_score: float = 0.50
    min_area_ratio: float = 0.50
    max_area_ratio: float = 2.00
    identity_margin: float = 0.10
    max_cross_class_overlap: float = 0.10
    min_overlap_pixels: int = 100

    def validate(self) -> None:
        unit_interval = (
            "min_iou",
            "min_containment",
            "min_sam_coverage",
            "min_region_confidence",
            "min_match_score",
            "identity_margin",
            "max_cross_class_overlap",
        )
        for field_name in unit_interval:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.min_area_ratio <= 0.0:
            raise ValueError("min_area_ratio must be positive")
        if self.max_area_ratio < self.min_area_ratio:
            raise ValueError("max_area_ratio must be at least min_area_ratio")
        if self.min_overlap_pixels < 1:
            raise ValueError("min_overlap_pixels must be positive")


@dataclass(frozen=True)
class RegionMatch:
    mask_index: int
    class_name: str
    region_id: int
    sam_area: int
    region_area: int
    intersection: int
    iou: float
    containment: float
    sam_coverage: float
    region_coverage: float
    area_ratio: float
    mean_region_confidence: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def semantic_class(mask_metadata: dict[str, Any]) -> str:
    return str(
        mask_metadata.get(
            "class_name",
            mask_metadata.get("class", "unknown"),
        )
    ).strip() or "unknown"


def match_metrics(
    sam_mask: np.ndarray,
    region_mask: np.ndarray,
    region_confidence: np.ndarray,
) -> dict[str, float | int]:
    sam_mask = np.asarray(sam_mask, dtype=bool)
    region_mask = np.asarray(region_mask, dtype=bool)
    if sam_mask.shape != region_mask.shape:
        raise ValueError("SAM and region masks must have identical shapes")
    if region_confidence.shape != sam_mask.shape:
        raise ValueError("region_confidence must match the mask shape")
    sam_area = int(sam_mask.sum())
    region_area = int(region_mask.sum())
    intersection = int(np.logical_and(sam_mask, region_mask).sum())
    union = sam_area + region_area - intersection
    minimum_area = min(sam_area, region_area)
    return {
        "sam_area": sam_area,
        "region_area": region_area,
        "intersection": intersection,
        "iou": intersection / float(max(union, 1)),
        "containment": intersection / float(max(minimum_area, 1)),
        "sam_coverage": intersection / float(max(sam_area, 1)),
        "region_coverage": intersection / float(max(region_area, 1)),
        "area_ratio": region_area / float(max(sam_area, 1)),
        "mean_region_confidence": (
            float(np.asarray(region_confidence, dtype=np.float32)[region_mask].mean())
            if region_area > 0
            else 0.0
        ),
    }


def region_match_score(metrics: dict[str, float | int]) -> float:
    """Global score balancing overlap, containment, and region confidence."""

    return (
        0.50 * float(metrics["iou"])
        + 0.30 * float(metrics["containment"])
        + 0.20 * float(metrics["mean_region_confidence"])
    )


def eligible_match(
    metrics: dict[str, float | int],
    thresholds: RefinementThresholds,
) -> bool:
    return (
        int(metrics["intersection"]) >= thresholds.min_overlap_pixels
        and float(metrics["sam_coverage"]) >= thresholds.min_sam_coverage
        and (
            float(metrics["iou"]) >= thresholds.min_iou
            or float(metrics["containment"]) >= thresholds.min_containment
        )
        and float(metrics["mean_region_confidence"])
        >= thresholds.min_region_confidence
        and thresholds.min_area_ratio
        <= float(metrics["area_ratio"])
        <= thresholds.max_area_ratio
        and region_match_score(metrics) >= thresholds.min_match_score
    )


def candidate_matches(
    masks: np.ndarray,
    mask_metadata: list[dict[str, Any]],
    region_id: np.ndarray,
    region_confidence: np.ndarray,
    thresholds: RefinementThresholds,
) -> list[RegionMatch]:
    matches: list[RegionMatch] = []
    for mask_index, sam_mask in enumerate(masks):
        class_name = semantic_class(mask_metadata[mask_index])
        overlapping_regions = np.unique(region_id[np.asarray(sam_mask, dtype=bool)])
        for compact_region_id in overlapping_regions:
            compact_region_id = int(compact_region_id)
            if compact_region_id == 0:
                continue
            metrics = match_metrics(
                sam_mask,
                region_id == compact_region_id,
                region_confidence,
            )
            if not eligible_match(metrics, thresholds):
                continue
            matches.append(
                RegionMatch(
                    mask_index=mask_index,
                    class_name=class_name,
                    region_id=compact_region_id,
                    score=region_match_score(metrics),
                    **metrics,
                )
            )
    return matches


def _region_owners(
    matches: list[RegionMatch],
    thresholds: RefinementThresholds,
) -> tuple[dict[int, int], set[int]]:
    """Choose at most one SAM mask owner for each region.

    Competing semantic identities inside the configured score margin make the
    region ambiguous. Same-class duplicate detections are not an identity
    conflict; the strongest detection owns the boundary.
    """

    matches_by_region: dict[int, list[RegionMatch]] = {}
    for match in matches:
        matches_by_region.setdefault(match.region_id, []).append(match)

    owners: dict[int, int] = {}
    ambiguous_regions: set[int] = set()
    for compact_region_id, region_matches in matches_by_region.items():
        ordered = sorted(region_matches, key=lambda item: item.score, reverse=True)
        best = ordered[0]
        competing_identity = next(
            (
                item
                for item in ordered[1:]
                if item.class_name != best.class_name
            ),
            None,
        )
        if (
            competing_identity is not None
            and best.score - competing_identity.score < thresholds.identity_margin
        ):
            ambiguous_regions.add(compact_region_id)
            continue
        owners[compact_region_id] = best.mask_index
    return owners, ambiguous_regions


def _cross_class_overlap(
    proposed_mask: np.ndarray,
    mask_index: int,
    masks: np.ndarray,
    mask_metadata: list[dict[str, Any]],
) -> float:
    class_name = semantic_class(mask_metadata[mask_index])
    proposed_area = int(proposed_mask.sum())
    maximum = 0.0
    for other_index, other_mask in enumerate(masks):
        if other_index == mask_index:
            continue
        if semantic_class(mask_metadata[other_index]) == class_name:
            continue
        other_mask = np.asarray(other_mask, dtype=bool)
        intersection = int(np.logical_and(proposed_mask, other_mask).sum())
        containment = intersection / float(max(min(proposed_area, int(other_mask.sum())), 1))
        maximum = max(maximum, containment)
    return maximum


def refine_frame_masks(
    masks: np.ndarray,
    mask_metadata: list[dict[str, Any]],
    region_id: np.ndarray,
    region_confidence: np.ndarray,
    thresholds: RefinementThresholds,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Return refined masks and per-mask evidence without adding identities."""

    thresholds.validate()
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("masks must have shape (K,H,W)")
    if len(mask_metadata) != masks.shape[0]:
        raise ValueError("mask metadata count must match mask stack")
    if region_id.shape != masks.shape[1:]:
        raise ValueError("region_id shape must match SAM masks")
    if region_confidence.shape != region_id.shape:
        raise ValueError("region_confidence shape must match region_id")

    matches = candidate_matches(
        masks,
        mask_metadata,
        region_id,
        region_confidence,
        thresholds,
    )
    owners, ambiguous_regions = _region_owners(matches, thresholds)
    matches_by_mask: dict[int, list[RegionMatch]] = {}
    for match in matches:
        matches_by_mask.setdefault(match.mask_index, []).append(match)

    refined = masks.copy()
    audits: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for mask_index, sam_mask in enumerate(masks):
        ordered = sorted(
            matches_by_mask.get(mask_index, []),
            key=lambda item: item.score,
            reverse=True,
        )
        audit: dict[str, Any] = {
            "status": "unchanged_no_region_match",
            "semantic_identity_source": "groundingdino_sam",
            "mask2former_semantic_class_used": False,
            "original_area": int(sam_mask.sum()),
            "refined_area": int(sam_mask.sum()),
            "selected_region_id": 0,
            "candidate_matches": [item.to_dict() for item in ordered],
        }
        for match in ordered:
            if match.region_id in ambiguous_regions:
                audit["status"] = "unchanged_identity_conflict"
                continue
            if owners.get(match.region_id) != mask_index:
                audit["status"] = "unchanged_region_claimed_by_stronger_mask"
                continue
            proposed_mask = region_id == match.region_id
            cross_class_overlap = _cross_class_overlap(
                proposed_mask,
                mask_index,
                masks,
                mask_metadata,
            )
            if cross_class_overlap > thresholds.max_cross_class_overlap:
                audit["status"] = "unchanged_cross_class_overlap"
                audit["cross_class_overlap"] = cross_class_overlap
                continue
            refined[mask_index] = proposed_mask
            audit.update(
                {
                    "status": "snapped_to_mask2former_region",
                    "refined_area": int(proposed_mask.sum()),
                    "selected_region_id": match.region_id,
                    "selected_match": match.to_dict(),
                    "cross_class_overlap": cross_class_overlap,
                }
            )
            break
        status = str(audit["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        audits.append(audit)
    return refined, audits, status_counts


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "unknown"


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def save_comparison_overlay(
    rgb: np.ndarray,
    original_masks: np.ndarray,
    refined_masks: np.ndarray,
    mask_metadata: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    output_path: Path,
) -> None:
    blended = np.asarray(rgb, dtype=np.float32).copy()
    for mask_index, refined_mask in enumerate(refined_masks):
        color = np.asarray(
            rgb8_for_class(semantic_class(mask_metadata[mask_index])),
            dtype=np.float32,
        )
        selected = np.asarray(refined_mask, dtype=bool)
        blended[selected] = 0.45 * blended[selected] + 0.55 * color
        boundary = _mask_boundary(original_masks[mask_index])
        blended[boundary] = np.asarray([255.0, 255.0, 255.0])

    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    y = 4
    for mask_index, audit in enumerate(audits[:8]):
        label = (
            f"{mask_index}:{semantic_class(mask_metadata[mask_index])} "
            f"{audit['status']}"
        )
        draw.text((4, y), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font=font)
        y += 12
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def load_mask_stack(path: Path, height: int, width: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        masks = data["masks"].astype(bool)
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(
            f"{path} masks have shape {masks.shape}; expected (K,{height},{width})"
        )
    return masks


def write_binary_masks(
    masks: np.ndarray,
    metadata: list[dict[str, Any]],
    stem: str,
    output_dir: Path,
) -> list[str]:
    filenames: list[str] = []
    for index, mask in enumerate(masks):
        class_name = semantic_class(metadata[index])
        filename = (
            f"{stem}__mask_{index:03d}__{_safe_filename_part(class_name)}.png"
        )
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_dir / filename)
        filenames.append(filename)
    return filenames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-sam-manifest", required=True, type=Path)
    parser.add_argument("--dense-seg-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-iou", default=0.45, type=float)
    parser.add_argument("--min-containment", default=0.80, type=float)
    parser.add_argument("--min-sam-coverage", default=0.10, type=float)
    parser.add_argument("--min-region-confidence", default=0.25, type=float)
    parser.add_argument("--min-match-score", default=0.50, type=float)
    parser.add_argument("--min-area-ratio", default=0.50, type=float)
    parser.add_argument("--max-area-ratio", default=2.00, type=float)
    parser.add_argument("--identity-margin", default=0.10, type=float)
    parser.add_argument("--max-cross-class-overlap", default=0.10, type=float)
    parser.add_argument("--min-overlap-pixels", default=100, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    thresholds = RefinementThresholds(
        min_iou=args.min_iou,
        min_containment=args.min_containment,
        min_sam_coverage=args.min_sam_coverage,
        min_region_confidence=args.min_region_confidence,
        min_match_score=args.min_match_score,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        identity_margin=args.identity_margin,
        max_cross_class_overlap=args.max_cross_class_overlap,
        min_overlap_pixels=args.min_overlap_pixels,
    )
    thresholds.validate()

    grounded_manifest = json.loads(
        args.grounded_sam_manifest.read_text(encoding="utf-8")
    )
    dense_manifest = json.loads(args.dense_seg_manifest.read_text(encoding="utf-8"))
    if not dense_manifest.get("save_regions", False):
        raise ValueError("Dense segmentation manifest does not contain region exports")

    dense_frames = {
        int(frame["camera_index"]): frame for frame in dense_manifest.get("frames", [])
    }
    grounded_root = args.grounded_sam_manifest.parent
    dense_root = args.dense_seg_manifest.parent
    mask_stack_dir = args.output_dir / "mask_stacks"
    binary_mask_dir = args.output_dir / "binary_masks"
    overlay_dir = args.output_dir / "overlays"
    for directory in (mask_stack_dir, binary_mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_manifest_path = args.output_dir / "grounded_sam_manifest.json"
    report_path = args.output_dir / "hybrid_refinement_report.json"
    for output_path in (output_manifest_path, report_path):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    output_frames: list[dict[str, Any]] = []
    total_status_counts: dict[str, int] = {}
    for grounded_frame in grounded_manifest.get("frames", []):
        camera_index = int(grounded_frame["camera_index"])
        if camera_index not in dense_frames:
            raise ValueError(f"Camera {camera_index} is absent from dense segmentation")
        dense_frame = dense_frames[camera_index]
        if str(dense_frame["file"]) != str(grounded_frame["file"]):
            raise ValueError(
                f"Camera {camera_index} filename mismatch: "
                f"{grounded_frame['file']} versus {dense_frame['file']}"
            )

        height = int(dense_frame["height"])
        width = int(dense_frame["width"])
        masks = load_mask_stack(
            grounded_root / "mask_stacks" / str(grounded_frame["mask_file"]),
            height,
            width,
        )
        metadata = list(grounded_frame.get("masks", []))
        with np.load(
            dense_root / "seg" / str(dense_frame["seg_file"]),
            allow_pickle=False,
        ) as dense_arrays:
            if "region_id" not in dense_arrays or "region_confidence" not in dense_arrays:
                raise ValueError(
                    f"{dense_frame['seg_file']} lacks class-agnostic region arrays"
                )
            region_id = dense_arrays["region_id"].astype(np.uint16)
            region_confidence = dense_arrays["region_confidence"].astype(np.float32)

        refined_masks, audits, status_counts = refine_frame_masks(
            masks,
            metadata,
            region_id,
            region_confidence,
            thresholds,
        )
        for status, count in status_counts.items():
            total_status_counts[status] = total_status_counts.get(status, 0) + count

        filename = str(grounded_frame["file"])
        stem = Path(filename).stem
        mask_file = f"{stem}.npz"
        np.savez_compressed(mask_stack_dir / mask_file, masks=refined_masks.astype(np.uint8))
        binary_mask_files = write_binary_masks(
            refined_masks,
            metadata,
            stem,
            binary_mask_dir,
        )
        rgb = np.asarray(
            Image.open(dense_root / "rgb_renders" / filename).convert("RGB"),
            dtype=np.uint8,
        )
        save_comparison_overlay(
            rgb,
            masks,
            refined_masks,
            metadata,
            audits,
            overlay_dir / filename,
        )

        output_masks: list[dict[str, Any]] = []
        for index, source_metadata in enumerate(metadata):
            output_masks.append(
                {
                    **source_metadata,
                    "binary_mask_file": binary_mask_files[index],
                    "area": int(refined_masks[index].sum()),
                    "hybrid_refinement": audits[index],
                }
            )
        output_frames.append(
            {
                **grounded_frame,
                "mask_file": mask_file,
                "binary_mask_files": binary_mask_files,
                "kept_mask_count": int(refined_masks.shape[0]),
                "masks": output_masks,
                "hybrid_refinement_status_counts": status_counts,
            }
        )
        print(
            f"refined {filename}: masks={refined_masks.shape[0]} "
            f"snapped={status_counts.get('snapped_to_mask2former_region', 0)}"
        )

    output_manifest = {
        **grounded_manifest,
        "source": "groundingdino_sam_mask2former_regions",
        "upstream_grounded_sam_manifest": str(args.grounded_sam_manifest),
        "dense_seg_manifest": str(args.dense_seg_manifest),
        "mask2former_semantic_class_used": False,
        "refinement_parameters": asdict(thresholds),
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
        "grounded_sam_manifest": str(args.grounded_sam_manifest),
        "dense_seg_manifest": str(args.dense_seg_manifest),
        "output_manifest": str(output_manifest_path),
        "mask2former_semantic_class_used": False,
        "frame_count": len(output_frames),
        "input_mask_count": sum(
            int(frame.get("kept_mask_count", 0))
            for frame in grounded_manifest.get("frames", [])
        ),
        "output_mask_count": sum(
            int(frame.get("kept_mask_count", 0)) for frame in output_frames
        ),
        "status_counts": total_status_counts,
        "parameters": asdict(thresholds),
        "semantic_labels_written": False,
        "semantic_ply_written": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output_manifest_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
