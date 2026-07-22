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
    record = {"path": str(path), "exists": path.exists(), "bytes": 0}
    if path.exists() and path.is_file():
        record["bytes"] = path.stat().st_size
    elif path.exists() and path.is_dir():
        record["file_count"] = sum(1 for item in path.rglob("*") if item.is_file())
    return record


def first_path(output_dir: Path, *relative_paths: str) -> Path:
    candidates = [output_dir / relative for relative in relative_paths]
    return next((path for path in candidates if path.exists()), candidates[0])


def run_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "ground": first_path(output_dir, "stages/01_grounded_sam", "grounded_sam"),
        "proposals": first_path(output_dir, "stages/02_flashsplat", "flashsplat_proposals"),
        "fusion": first_path(output_dir, "stages/03_semantic_fusion", "."),
        "deliverables": first_path(output_dir, "deliverables", "."),
        "visualizations": first_path(output_dir, "visualizations", "."),
        "validation": first_path(output_dir, "validation", "."),
    }


def grounded_sam_stage(paths: dict[str, Path]) -> dict[str, Any]:
    root = paths["ground"]
    manifest_path = root / "grounded_sam_manifest.json"
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
    mask_stack_dir = root / "mask_stacks"
    if not mask_stack_dir.exists():
        mask_stack_dir = root / "grounded_sam_masks"
    overlay_dir = root / "overlays"
    if not overlay_dir.exists():
        overlay_dir = root / "grounded_sam_overlays"
    return {
        "name": "GroundingDINO + SAM masks",
        "manifest": file_record(manifest_path),
        "frame_count": len(frames),
        "selected_camera_indices": manifest.get("selected_camera_indices", []),
        "raw_detection_count": sum(int(frame.get("raw_detection_count", 0)) for frame in frames),
        "kept_mask_count": sum(int(frame.get("kept_mask_count", 0)) for frame in frames),
        "class_counts": dict(sorted(class_counts.items())),
        "top_phrases": dict(phrase_counts.most_common(20)),
        "rgb_render_dir": file_record(root / "rgb_renders"),
        "mask_stack_dir": file_record(mask_stack_dir),
        "binary_mask_dir": file_record(root / "binary_masks"),
        "overlay_dir": file_record(overlay_dir),
    }


def flashsplat_stage(paths: dict[str, Path]) -> dict[str, Any]:
    root = paths["proposals"]
    manifest_path = root / "proposal_manifest.json"
    manifest = read_json(manifest_path)
    class_evidence_path = root / "class_evidence" / "class_evidence_manifest.json"
    class_evidence = read_json(class_evidence_path)
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
        "proposal_support_dir": file_record(root / "proposal_supports"),
        "class_evidence_manifest": file_record(class_evidence_path),
        "class_evidence_class_count": len(class_evidence.get("classes", [])),
    }


def semantic_fusion_stage(paths: dict[str, Path]) -> dict[str, Any]:
    summary_path = paths["fusion"] / "semantic_group_summary.json"
    summary = read_json(summary_path)
    pruning = summary.get("pruning", {}) if not summary.get("missing") else {}
    return {
        "name": "3D semantic fusion and automatic pruning",
        "summary": file_record(summary_path),
        "labels_npy": file_record(paths["fusion"] / "gaussian_labels.npy"),
        "proposal_count": summary.get("proposal_count", 0),
        "group_count": summary.get("group_count", 0),
        "label_histogram": summary.get("label_histogram", {}),
        "parameters": summary.get("parameters", {}),
        "pruning": {
            "pruned_group_count": pruning.get("pruned_group_count", 0),
            "pruned_groups": pruning.get("pruned_groups", []),
        },
        "spatial_pruning": summary.get("spatial_pruning", {"enabled": False}),
        "instance_consolidation": summary.get("instance_consolidation", {"enabled": False}),
        "groups": summary.get("groups", []),
    }


def export_stage(paths: dict[str, Path], focus_name: str) -> dict[str, Any]:
    deliverables = paths["deliverables"]
    visualizations = paths["visualizations"]
    ply_dir = visualizations / "ply" if (visualizations / "ply").exists() else visualizations
    overlay_root = visualizations / "overlays"
    contact_root = visualizations / "contact_sheets"
    return {
        "name": "Semantic PLY and debug exports",
        "semantic_ply": file_record(deliverables / "semantic_point_cloud.ply"),
        "label_map": file_record(deliverables / "label_map.json"),
        "supersplat_debug_ply": file_record(ply_dir / "semantic_point_cloud_supersplat_debug.ply"),
        "focus_name": focus_name,
        "focus_debug_ply": file_record(ply_dir / f"{focus_name}_supersplat_debug.ply"),
        "semantic_overlay_dir": file_record(overlay_root / "semantic_labels"),
        "focus_overlay_dir": file_record(overlay_root / focus_name),
        "contact_sheet_dir": file_record(contact_root),
    }


def validation_stage(paths: dict[str, Path]) -> dict[str, Any]:
    validation_path = paths["validation"] / "task1_validation.json"
    validation = read_json(validation_path)
    return {
        "name": "Task 1 output validation",
        "validation": file_record(validation_path),
        "status": validation.get("status", "missing"),
        "label_count": validation.get("label_count"),
        "unlabeled_ratio": validation.get("unlabeled_ratio"),
        "visible_coverage": validation.get("visible_coverage"),
        "semantic_ply": validation.get("semantic_ply", {}),
    }


def build_summary(output_dir: Path, focus_name: str = "focus_classes") -> dict[str, Any]:
    paths = run_paths(output_dir)
    return {
        "output_dir": str(output_dir),
        "layout": {name: str(path) for name, path in paths.items()},
        "stages": {
            "groundingdino_sam": grounded_sam_stage(paths),
            "flashsplat": flashsplat_stage(paths),
            "semantic_fusion_pruning": semantic_fusion_stage(paths),
            "exports": export_stage(paths, focus_name),
            "validation": validation_stage(paths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-name", default="pipeline_run_summary.json")
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--focus-name", default="focus_classes")
    args = parser.parse_args()

    summary = build_summary(args.output_dir, args.focus_name)
    summary_path = args.summary_path or args.output_dir / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
