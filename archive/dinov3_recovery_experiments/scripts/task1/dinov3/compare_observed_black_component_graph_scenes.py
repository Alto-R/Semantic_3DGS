#!/usr/bin/env python3
"""Compare the common observed-black component graph across two scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.task1.dinov3.observed_black_component_graph_audit import (
    CONTRACT,
    POLICY,
    SOURCE,
)


SCENES = ("playroom", "drjohnson")


def validate_report(report: Mapping[str, Any], scene: str) -> None:
    expected = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": scene,
        "report_only": True,
        "policy": POLICY,
        "component_graph_is_class_agnostic": True,
        "component_graph_uses_mutual_knn": True,
        "component_labels_selected_after_grouping": True,
        "candidate_labels_limited_to_component_camera_evidence": True,
        "class_reliability_is_soft_weighting": True,
        "global_class_veto_used": False,
        "anchor_support_is_component_normalized": True,
        "broad_surface_population_normalized": True,
        "zero_camera_gaussians_forced_black": True,
        "single_camera_gaussians_forced_black": False,
        "spatial_evidence_can_choose_semantic_class": False,
        "dinov3_inference_rerun": False,
        "flashsplat_lifting_rerun": False,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "manual_class_selection_used": False,
        "scene_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"{scene} report violates {field}={value!r}")
    for field in (
        "current_black_gaussian_count",
        "current_black_zero_combined_camera_count",
        "current_black_single_combined_camera_count",
        "current_black_multicamera_count",
        "camera_observed_black_gaussian_count",
        "component_count",
        "eligible_report_only_gaussian_count",
    ):
        if int(report.get(field, -1)) < 0:
            raise ValueError(f"{scene} report has invalid {field}")
    if not isinstance(report.get("graph_calibration"), dict):
        raise ValueError(f"{scene} report lacks graph calibration")
    if not isinstance(report.get("component_calibration"), dict):
        raise ValueError(f"{scene} report lacks component calibration")


def scene_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    black = int(report["current_black_gaussian_count"])
    eligible = int(report["eligible_report_only_gaussian_count"])
    return {
        "current_black_gaussian_count": black,
        "zero_combined_camera_count": int(
            report["current_black_zero_combined_camera_count"]
        ),
        "single_combined_camera_count": int(
            report["current_black_single_combined_camera_count"]
        ),
        "multicamera_count": int(report["current_black_multicamera_count"]),
        "camera_observed_black_gaussian_count": int(
            report["camera_observed_black_gaussian_count"]
        ),
        "component_count": int(report["component_count"]),
        "eligible_report_only_gaussian_count": eligible,
        "eligible_ratio_of_current_black": eligible / black if black else 0.0,
        "eligible_by_class": report.get("eligible_by_class", []),
        "decision_counts": report.get("decision_counts", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playroom-report", required=True, type=Path)
    parser.add_argument("--drjohnson-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = {
        "playroom": json.loads(args.playroom_report.read_text(encoding="utf-8")),
        "drjohnson": json.loads(args.drjohnson_report.read_text(encoding="utf-8")),
    }
    for scene in SCENES:
        validate_report(reports[scene], scene)
    summaries = {scene: scene_summary(reports[scene]) for scene in SCENES}
    ready = all(
        summary["eligible_report_only_gaussian_count"] > 0
        for summary in summaries.values()
    )
    output = {
        "source": "dinov3_observed_black_component_graph_two_scene_comparison",
        "contract": "paired_common_component_graph_report_only_gate_v1",
        "policy": POLICY,
        "scenes": summaries,
        "both_scenes_present": True,
        "common_policy_verified": True,
        "ready_for_heldout_round_trip_validation": ready,
        "accepted_for_materialization": False,
        "materialization_blocker": (
            "component candidates must pass a new held-out overall, interior, "
            "boundary, and per-class validation in both scenes"
        ),
        "manual_scene_selection_used": False,
        "manual_class_selection_used": False,
        "semantic_ply_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
