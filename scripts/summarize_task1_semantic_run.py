#!/usr/bin/env python3
"""Create one run-level summary for the automatic Task 1 semantic pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def grounded_sam_stage(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "grounded_sam" / "grounded_sam_manifest.json"
    manifest = read_json(manifest_path)
    frames = manifest.get("frames", []) if not manifest.get("missing") else []
    class_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    for frame in frames:
        for mask in frame.get("masks", []):
            class_counts[str(mask.get("class", mask.get("class_name", "unknown")))] += 1
            phrase = str(mask.get("phrase", "")).strip()
            if phrase:
                phrase_counts[phrase] += 1
    return {
        "name": "GroundingDINO + SAM masks",
        "manifest": file_record(manifest_path),
        "frame_count": len(frames),
        "raw_detection_count": sum(int(frame.get("raw_detection_count", 0)) for frame in frames),
        "kept_mask_count": sum(int(frame.get("kept_mask_count", 0)) for frame in frames),
        "class_counts": dict(sorted(class_counts.items())),
        "top_phrases": dict(phrase_counts.most_common(20)),
        "rgb_render_dir": file_record(output_dir / "grounded_sam" / "rgb_renders"),
        "mask_dir": file_record(output_dir / "grounded_sam" / "grounded_sam_masks"),
        "overlay_dir": file_record(output_dir / "grounded_sam" / "grounded_sam_overlays"),
        "contact_sheet": file_record(output_dir / "grounded_sam_contact_sheet.png"),
    }


def flashsplat_stage(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "flashsplat_proposals" / "proposal_manifest.json"
    manifest = read_json(manifest_path)
    proposals = manifest.get("proposals", []) if not manifest.get("missing") else []
    class_counts: Counter[str] = Counter()
    support_total = 0
    for proposal in proposals:
        class_name = str(proposal.get("class_name", proposal.get("class", "unknown")))
        class_counts[class_name] += 1
        support_total += int(proposal.get("gaussian_count", 0))
    return {
        "name": "FlashSplat mask-to-Gaussian proposals",
        "manifest": file_record(manifest_path),
        "proposal_count": len(proposals),
        "proposal_class_counts": dict(sorted(class_counts.items())),
        "proposal_support_gaussian_total": support_total,
        "proposal_support_dir": file_record(output_dir / "flashsplat_proposals" / "proposal_supports"),
    }


def semantic_fusion_stage(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "semantic_group_summary.json"
    summary = read_json(summary_path)
    groups = summary.get("groups", []) if not summary.get("missing") else []
    pruning = summary.get("pruning", {}) if not summary.get("missing") else {}
    return {
        "name": "3D semantic fusion and automatic pruning",
        "summary": file_record(summary_path),
        "proposal_count": summary.get("proposal_count", 0),
        "group_count": summary.get("group_count", 0),
        "label_histogram": summary.get("label_histogram", {}),
        "parameters": summary.get("parameters", {}),
        "pruning": {
            "pruned_group_count": pruning.get("pruned_group_count", 0),
            "pruned_groups": pruning.get("pruned_groups", []),
        },
        "groups": groups,
    }


def export_stage(output_dir: Path) -> dict[str, Any]:
    return {
        "name": "Semantic PLY and debug exports",
        "semantic_ply": file_record(output_dir / "semantic_point_cloud.ply"),
        "labels_npy": file_record(output_dir / "gaussian_labels.npy"),
        "label_map": file_record(output_dir / "label_map.json"),
        "semantic_ply_inspection": file_record(output_dir / "semantic_point_cloud_inspection.json"),
        "debug_color_ply": file_record(output_dir / "semantic_point_cloud_debug_colors.ply"),
        "debug_color_metadata": file_record(output_dir / "semantic_point_cloud_debug_colors.json"),
        "debug_color_inspection": file_record(output_dir / "semantic_point_cloud_debug_colors_inspection.json"),
        "semantic_overlay_dir": file_record(output_dir / "semantic_label_overlay_renders"),
        "semantic_overlay_contact_sheet": file_record(output_dir / "semantic_label_overlay_contact_sheet.png"),
    }


def validation_stage(output_dir: Path) -> dict[str, Any]:
    validation_path = output_dir / "task1_validation.json"
    validation = read_json(validation_path)
    return {
        "name": "Task 1 output validation",
        "validation": file_record(validation_path),
        "status": validation.get("status", "missing"),
        "label_count": validation.get("label_count"),
        "unlabeled_ratio": validation.get("unlabeled_ratio"),
        "semantic_ply": validation.get("semantic_ply", {}),
    }


def build_summary(output_dir: Path) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "stages": {
            "groundingdino_sam": grounded_sam_stage(output_dir),
            "flashsplat": flashsplat_stage(output_dir),
            "semantic_fusion_pruning": semantic_fusion_stage(output_dir),
            "exports": export_stage(output_dir),
            "validation": validation_stage(output_dir),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-name", default="pipeline_run_summary.json")
    args = parser.parse_args()

    summary = build_summary(args.output_dir)
    summary_path = args.output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
