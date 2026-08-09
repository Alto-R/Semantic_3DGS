#!/usr/bin/env python3
"""Compare the fixed observed-black audit across Playroom and DrJohnson."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.task1.dinov3.observed_black_calibrated_spatial_audit import (
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
        "zero_camera_gaussians_forced_black": True,
        "single_camera_gaussians_forced_black": True,
        "spatial_evidence_can_choose_semantic_class": False,
        "spatial_evidence_can_relax_stuff_semantic_thresholds": False,
        "semantic_candidate_must_precede_spatial_corroboration": True,
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
        "eligible_report_only_gaussian_count",
    ):
        if int(report.get(field, -1)) < 0:
            raise ValueError(f"{scene} report has invalid {field}")
    if not isinstance(report.get("decision_counts"), dict):
        raise ValueError(f"{scene} report lacks decision counts")
    if not isinstance(report.get("heldout_class_reliability"), list):
        raise ValueError(f"{scene} report lacks held-out class reliability")


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
        "eligible_report_only_gaussian_count": eligible,
        "eligible_ratio_of_current_black": eligible / black if black else 0.0,
        "eligible_by_class": report.get("eligible_by_class", []),
        "decision_counts": report["decision_counts"],
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
    rows = {scene: scene_summary(reports[scene]) for scene in SCENES}
    ready = all(row["eligible_report_only_gaussian_count"] > 0 for row in rows.values())
    output = {
        "source": "dinov3_observed_black_calibrated_spatial_two_scene_comparison",
        "contract": "paired_common_policy_report_only_observed_black_gate_v1",
        "policy": POLICY,
        "scenes": rows,
        "both_scenes_present": True,
        "common_policy_verified": True,
        "ready_for_heldout_round_trip_validation": ready,
        "accepted_for_materialization": False,
        "materialization_blocker": (
            "the report-only candidate must pass a new held-out overall, interior, "
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
