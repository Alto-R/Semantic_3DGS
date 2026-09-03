#!/usr/bin/env python3
"""Render per-view overlays for report-only DINOv3 3D components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.task1.common.semantic_palette import rgb8_for_class


AMBIGUOUS_COLOR = np.asarray([128, 128, 128], dtype=np.uint8)
REPORT_CONTRACTS = {
    "report_only_class_agnostic_3d_association_v1",
    "report_only_dinov3_anchored_component_association_v1",
    "report_only_dinov3_multiview_spatial_core_v1",
}


def render_overlay(
    rgb_path: Path,
    masks: np.ndarray,
    mask_records: list[dict[str, Any]],
    output_path: Path,
    alpha: float,
) -> None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
        raise ValueError(f"mask stack shape differs from RGB for {rgb_path}")
    if masks.shape[0] != len(mask_records):
        raise ValueError(f"mask metadata count differs for {rgb_path}")
    overlay = rgb.copy()
    for mask, record in zip(masks, mask_records):
        selected = np.asarray(mask, dtype=bool)
        if not selected.any():
            continue
        color = (
            np.asarray(rgb8_for_class(str(record["class"])), dtype=np.uint8)
            if bool(record["accepted"])
            else AMBIGUOUS_COLOR
        )
        overlay[selected] = (
            (1.0 - alpha) * rgb[selected].astype(np.float32)
            + alpha * color.astype(np.float32)
        ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--association-report", required=True, type=Path)
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", default=0.55, type=float)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    report = json.loads(args.association_report.read_text(encoding="utf-8"))
    if report.get("contract") not in REPORT_CONTRACTS:
        raise ValueError("association report has the wrong contract")
    proposal_manifest = json.loads(
        args.proposal_manifest.read_text(encoding="utf-8")
    )
    query_manifest_path = Path(str(proposal_manifest["source_manifest"]))
    query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
    dinov3_manifest_path = Path(str(query_manifest["dinov3_manifest"]))
    dinov3_root = dinov3_manifest_path.parent
    mask_dir = query_manifest_path.parent / "mask_stacks"

    component_by_proposal: dict[int, dict[str, Any]] = {}
    report_components = report.get(
        "visualization_components", report.get("components", [])
    )
    for component in report_components:
        for proposal_id in component.get("proposal_ids", []):
            component_by_proposal[int(proposal_id)] = component
    proposal_by_frame_mask = {
        (str(item["frame_file"]), int(item["mask_index"])): item
        for item in proposal_manifest.get("proposals", [])
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    for frame in query_manifest.get("frames", []):
        frame_file = str(frame["file"])
        mask_path = mask_dir / str(frame["mask_file"])
        with np.load(mask_path, allow_pickle=False) as data:
            masks = np.asarray(data["masks"], dtype=bool)
        records: list[dict[str, Any]] = []
        for mask_index in range(masks.shape[0]):
            proposal = proposal_by_frame_mask.get((frame_file, mask_index))
            component = (
                component_by_proposal.get(int(proposal["proposal_id"]))
                if proposal is not None
                else None
            )
            if component is None:
                records.append(
                    {
                        "component_id": 0,
                        "accepted": False,
                        "class": "unlifted_region",
                        "status": "not_lifted_or_filtered",
                    }
                )
            else:
                records.append(
                    {
                        "component_id": int(component["component_id"]),
                        "accepted": bool(component["accepted"]),
                        "class": str(component["class"]),
                        "status": str(component["status"]),
                    }
                )
        output_file = Path(frame_file).name
        render_overlay(
            dinov3_root / "rgb_renders" / frame_file,
            masks,
            records,
            args.output_dir / output_file,
            args.alpha,
        )
        frames.append(
            {
                "file": output_file,
                "camera_index": int(frame["camera_index"]),
                "components": records,
            }
        )
    manifest = {
        "source": str(args.association_report),
        "visualization_scope": report.get(
            "visualization_scope", "full_source_proposal_masks"
        ),
        "accepted_regions_use_semantic_palette": True,
        "ambiguous_or_unlifted_regions_are_gray": True,
        "semantic_labels_written": False,
        "semantic_ply_written": False,
        "frames": frames,
    }
    (args.output_dir / "component_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
