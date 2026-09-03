#!/usr/bin/env python3
"""Assign DINOv3 query regions using independent DINOv2 identity evidence.

DINOv3 contributes only class-agnostic Mask2Former query boundaries. DINOv2
supplies weighted ADE20K identity votes. Ambiguous, weakly covered, or
non-target regions abstain. The output follows the mask-stack contract used by
the existing FlashSplat proposal audit and never writes Gaussian labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.task1.common.semantic_palette import (
    normalize_class_name,
    rgb8_for_class,
)
from scripts.task1.dinov2.dinov2_ontology import load_ontology


@dataclass(frozen=True)
class IdentityThresholds:
    min_dinov2_confidence: float = 0.50
    min_evidence_pixels: int = 100
    min_evidence_coverage: float = 0.25
    min_class_share: float = 0.60
    min_class_margin: float = 0.20

    def validate(self) -> None:
        for name in (
            "min_dinov2_confidence",
            "min_evidence_coverage",
            "min_class_share",
            "min_class_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_evidence_pixels < 1:
            raise ValueError("min_evidence_pixels must be positive")


def _class_ranking(
    project_ids: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    project_names: dict[int, str],
) -> list[dict[str, Any]]:
    maximum_id = max(project_names, default=0)
    weights = np.bincount(
        project_ids[valid].astype(np.int64),
        weights=confidence[valid].astype(np.float64),
        minlength=maximum_id + 1,
    )
    counts = np.bincount(
        project_ids[valid].astype(np.int64),
        minlength=maximum_id + 1,
    )
    total_weight = float(weights.sum())
    ranking: list[dict[str, Any]] = []
    for project_id in np.argsort(weights)[::-1]:
        project_id = int(project_id)
        if project_id == 0 or weights[project_id] <= 0.0:
            continue
        ranking.append(
            {
                "project_id": project_id,
                "class": normalize_class_name(
                    project_names.get(project_id, f"project_class_{project_id}")
                ),
                "pixels": int(counts[project_id]),
                "weight": float(weights[project_id]),
                "weight_share": float(weights[project_id] / max(total_weight, 1e-12)),
            }
        )
    return ranking


def score_region_identity(
    region_mask: np.ndarray,
    dinov2_class_id: np.ndarray,
    dinov2_confidence: np.ndarray,
    ade_to_project: np.ndarray,
    project_names: dict[int, str],
    allowed_classes: set[str],
    thresholds: IdentityThresholds,
) -> dict[str, Any]:
    """Return an auditable accept/abstain decision for one region."""

    thresholds.validate()
    region_mask = np.asarray(region_mask, dtype=bool)
    dinov2_class_id = np.asarray(dinov2_class_id)
    dinov2_confidence = np.asarray(dinov2_confidence, dtype=np.float32)
    if region_mask.shape != dinov2_class_id.shape:
        raise ValueError("region and DINOv2 class maps must have identical shapes")
    if dinov2_confidence.shape != region_mask.shape:
        raise ValueError("DINOv2 confidence must match the region shape")
    if dinov2_class_id.size and int(dinov2_class_id.max()) >= len(ade_to_project):
        raise ValueError("DINOv2 class id is outside the ontology")

    region_area = int(region_mask.sum())
    project_ids = ade_to_project[dinov2_class_id]
    valid = (
        region_mask
        & (project_ids > 0)
        & np.isfinite(dinov2_confidence)
        & (dinov2_confidence >= thresholds.min_dinov2_confidence)
    )
    evidence_pixels = int(valid.sum())
    evidence_coverage = evidence_pixels / float(max(region_area, 1))
    ranking = _class_ranking(
        project_ids,
        dinov2_confidence,
        valid,
        project_names,
    )
    decision: dict[str, Any] = {
        "status": "abstained_no_identity_evidence",
        "accepted": False,
        "region_area": region_area,
        "evidence_pixels": evidence_pixels,
        "evidence_coverage": evidence_coverage,
        "class_ranking": ranking[:8],
        "identity_source": "dinov2_vitl14_ade20k_linear",
        "dinov3_semantic_class_used": False,
    }
    if not ranking:
        return decision

    best = ranking[0]
    runner_up_share = float(ranking[1]["weight_share"]) if len(ranking) > 1 else 0.0
    class_margin = float(best["weight_share"]) - runner_up_share
    decision.update(
        {
            "selected_class": best["class"],
            "selected_project_id": best["project_id"],
            "selected_class_share": best["weight_share"],
            "runner_up_class": ranking[1]["class"] if len(ranking) > 1 else None,
            "runner_up_class_share": runner_up_share,
            "class_margin": class_margin,
        }
    )
    if evidence_pixels < thresholds.min_evidence_pixels:
        decision["status"] = "abstained_insufficient_identity_pixels"
    elif evidence_coverage < thresholds.min_evidence_coverage:
        decision["status"] = "abstained_insufficient_identity_coverage"
    elif str(best["class"]) not in allowed_classes:
        decision["status"] = "abstained_stronger_non_target_identity"
    elif float(best["weight_share"]) < thresholds.min_class_share:
        decision["status"] = "abstained_low_identity_share"
    elif class_margin < thresholds.min_class_margin:
        decision["status"] = "abstained_identity_margin"
    else:
        decision["status"] = "accepted_dinov2_identity"
        decision["accepted"] = True
    return decision


def _class_distribution(
    raw_class: np.ndarray,
    region_mask: np.ndarray,
    ade_names: dict[int, str],
) -> list[dict[str, Any]]:
    class_ids, counts = np.unique(raw_class[region_mask], return_counts=True)
    total = int(counts.sum())
    order = np.argsort(counts)[::-1]
    return [
        {
            "ade_id": int(class_ids[index]),
            "class": ade_names.get(int(class_ids[index]), "unknown"),
            "pixels": int(counts[index]),
            "ratio": int(counts[index]) / float(max(total, 1)),
        }
        for index in order[:8]
    ]


def _load_npz(path: Path, *keys: str) -> tuple[np.ndarray, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return tuple(np.asarray(data[key]) for key in keys)


def _frames_by_camera(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for frame in manifest.get("frames", []):
        camera_index = int(frame["camera_index"])
        if camera_index in frames:
            raise ValueError(f"Duplicate camera index in manifest: {camera_index}")
        frames[camera_index] = frame
    return frames


def _save_overlay(
    rgb_path: Path,
    masks: np.ndarray,
    metadata: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if len(masks) != len(metadata):
        raise ValueError("mask and metadata counts differ")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    overlay = rgb.copy()
    for mask, item in zip(masks, metadata):
        color = np.asarray(rgb8_for_class(str(item["class_name"])), dtype=np.uint8)
        selected = np.asarray(mask, dtype=bool)
        overlay[selected] = (
            0.45 * rgb[selected].astype(np.float32)
            + 0.55 * color.astype(np.float32)
        ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3-manifest", required=True, type=Path)
    parser.add_argument("--dinov2-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--include-classes", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-dinov2-confidence", default=0.50, type=float)
    parser.add_argument("--min-evidence-pixels", default=100, type=int)
    parser.add_argument("--min-evidence-coverage", default=0.25, type=float)
    parser.add_argument("--min-class-share", default=0.60, type=float)
    parser.add_argument("--min-class-margin", default=0.20, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    thresholds = IdentityThresholds(
        min_dinov2_confidence=args.min_dinov2_confidence,
        min_evidence_pixels=args.min_evidence_pixels,
        min_evidence_coverage=args.min_evidence_coverage,
        min_class_share=args.min_class_share,
        min_class_margin=args.min_class_margin,
    )
    thresholds.validate()
    allowed_classes = {
        normalize_class_name(item)
        for item in args.include_classes.split(",")
        if item.strip()
    }
    if not allowed_classes:
        raise ValueError("include-classes must contain at least one class")

    ontology = load_ontology(args.ontology)
    project_names = {
        item.project_id: normalize_class_name(item.project_class)
        for item in ontology.classes
    }
    ade_names = {
        item.ade_id: normalize_class_name(item.ade_name)
        for item in ontology.classes
    }
    missing_classes = sorted(allowed_classes - set(project_names.values()))
    if missing_classes:
        raise ValueError(
            "Requested classes are absent from the maintained ontology: "
            + ", ".join(missing_classes)
        )

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} exists; pass --overwrite to replace audit files"
        )
    mask_dir = args.output_dir / "mask_stacks"
    overlay_dir = args.output_dir / "identity_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dinov3_manifest = json.loads(
        args.dinov3_manifest.read_text(encoding="utf-8")
    )
    region_description = dinov3_manifest.get("class_agnostic_query_regions", {})
    if not region_description.get("available", False):
        raise ValueError("DINOv3 manifest does not contain query-region exports")
    if region_description.get("semantic_class_used_for_identity", True):
        raise ValueError("DINOv3 query regions are not declared class-agnostic")
    dinov2_manifest = json.loads(
        args.dinov2_manifest.read_text(encoding="utf-8")
    )
    dinov2_by_camera = _frames_by_camera(dinov2_manifest)

    output_frames: list[dict[str, Any]] = []
    accepted_total = 0
    status_counts: dict[str, int] = {}
    for frame in dinov3_manifest.get("frames", []):
        camera_index = int(frame["camera_index"])
        if camera_index not in dinov2_by_camera:
            raise ValueError(
                f"DINOv2 manifest has no matching camera index {camera_index}"
            )
        dinov2_frame = dinov2_by_camera[camera_index]
        dinov3_root = args.dinov3_manifest.parent
        dinov2_root = args.dinov2_manifest.parent
        region_id, region_confidence = _load_npz(
            dinov3_root / str(frame["region_file"]),
            "region_id",
            "region_confidence",
        )
        dinov3_class, = _load_npz(
            dinov3_root / str(frame["segment_file"]),
            "class_id",
        )
        dinov2_class, dinov2_confidence = _load_npz(
            dinov2_root / str(dinov2_frame["segment_file"]),
            "class_id",
            "confidence",
        )
        expected_shape = region_id.shape
        if any(
            array.shape != expected_shape
            for array in (
                region_confidence,
                dinov3_class,
                dinov2_class,
                dinov2_confidence,
            )
        ):
            raise ValueError(
                f"Aligned segmentation shapes differ for camera {camera_index}"
            )

        accepted_masks: list[np.ndarray] = []
        accepted_metadata: list[dict[str, Any]] = []
        region_audits: list[dict[str, Any]] = []
        metadata_by_id = {
            int(item["region_id"]): item
            for item in frame.get("regions", [])
        }
        for compact_id in sorted(int(item) for item in np.unique(region_id) if item):
            region_mask = region_id == compact_id
            decision = score_region_identity(
                region_mask,
                dinov2_class,
                dinov2_confidence,
                ontology.ade_to_project,
                project_names,
                allowed_classes,
                thresholds,
            )
            query_metadata = metadata_by_id.get(compact_id, {})
            decision.update(
                {
                    "region_id": compact_id,
                    "mean_region_confidence": float(
                        np.asarray(region_confidence, dtype=np.float32)[
                            region_mask
                        ].mean()
                    ),
                    "dinov3_query_diagnostic": query_metadata,
                    "dinov3_semantic_distribution": _class_distribution(
                        dinov3_class,
                        region_mask,
                        ade_names,
                    ),
                }
            )
            status = str(decision["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            region_audits.append(decision)
            if not decision["accepted"]:
                continue
            rows, columns = np.nonzero(region_mask)
            confidence = float(
                decision["selected_class_share"]
                * decision["evidence_coverage"]
            )
            accepted_masks.append(region_mask)
            accepted_metadata.append(
                {
                    "source": "dinov3_query_boundary_dinov2_identity",
                    "class": decision["selected_class"],
                    "class_name": decision["selected_class"],
                    "phrase": decision["selected_class"],
                    "area": int(region_mask.sum()),
                    "bbox_xyxy": [
                        int(columns.min()),
                        int(rows.min()),
                        int(columns.max()) + 1,
                        int(rows.max()) + 1,
                    ],
                    "confidence": confidence,
                    "predicted_iou": confidence,
                    "stability_score": 1.0,
                    "identity_audit": decision,
                }
            )

        height, width = expected_shape
        masks = (
            np.stack(accepted_masks, axis=0).astype(bool)
            if accepted_masks
            else np.zeros((0, height, width), dtype=bool)
        )
        stem = Path(str(frame["file"])).stem
        mask_file = f"{stem}.npz"
        np.savez_compressed(mask_dir / mask_file, masks=masks)
        rgb_path = dinov3_root / "rgb_renders" / str(frame["file"])
        overlay_file = str(frame["file"])
        _save_overlay(
            rgb_path,
            masks,
            accepted_metadata,
            overlay_dir / overlay_file,
        )
        accepted_total += int(masks.shape[0])
        output_frames.append(
            {
                **frame,
                "mask_file": mask_file,
                "masks": accepted_metadata,
                "accepted_region_count": int(masks.shape[0]),
                "audited_region_count": len(region_audits),
                "identity_overlay_file": (
                    Path("identity_overlays") / overlay_file
                ).as_posix(),
                "region_audits": region_audits,
            }
        )
        print(
            f"camera {camera_index}: accepted {masks.shape[0]} of "
            f"{len(region_audits)} query regions"
        )

    output = {
        "source": "dinov3_query_boundary_dinov2_identity_audit_v1",
        "contract": "grounded_sam_compatible_mask_stacks_report_only",
        "dinov3_manifest": str(args.dinov3_manifest),
        "dinov2_manifest": str(args.dinov2_manifest),
        "ontology": str(args.ontology),
        "include_classes": sorted(allowed_classes),
        "identity_thresholds": asdict(thresholds),
        "boundary_source": "dinov3_mask2former_query_masks",
        "identity_source": "dinov2_vitl14_ade20k_linear",
        "dinov3_semantic_class_used": False,
        "immutable_base_used_during_2d_identity": False,
        "immutable_base_checked_after_flashsplat_lifting": True,
        "accepted_region_count": accepted_total,
        "status_counts": status_counts,
        "semantic_labels_written": False,
        "semantic_ply_written": False,
        "frames": output_frames,
    }
    manifest_text = json.dumps(output, indent=2)
    (args.output_dir / "grounded_sam_manifest.json").write_text(
        manifest_text,
        encoding="utf-8",
    )
    (args.output_dir / "boundary_identity_audit.json").write_text(
        manifest_text,
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'boundary_identity_audit.json'}")


if __name__ == "__main__":
    main()
