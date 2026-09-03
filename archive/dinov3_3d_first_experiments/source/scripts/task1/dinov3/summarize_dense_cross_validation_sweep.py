#!/usr/bin/env python3
"""Summarize globally defined dense cross-validation confidence profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CONTRACT = "report_only_dinov3_core_first_dense_cross_validation_v1"


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one profile report is required")
    scenes = {str(report.get("scene", "")) for report in reports}
    if len(scenes) != 1:
        raise ValueError("all profile reports must describe the same scene")
    profile_names = [str(report.get("profile_name", "")) for report in reports]
    if any(not name for name in profile_names) or len(set(profile_names)) != len(
        profile_names
    ):
        raise ValueError("profile names must be non-empty and unique")
    for report in reports:
        if report.get("contract") != EXPECTED_CONTRACT:
            raise ValueError("profile report has the wrong contract")
        if not bool(report.get("report_only")):
            raise ValueError("profile report is not report-only")
        for field in (
            "semantic_labels_written",
            "semantic_project_class_arrays_written",
            "label_map_written",
            "semantic_ply_written",
            "scene_specific_rules",
            "class_specific_thresholds",
            "manual_component_decisions",
        ):
            if report.get(field) is not False:
                raise ValueError(f"profile report violates {field}=false")

    profiles = []
    for report in reports:
        candidate_count = int(report["black_core_first_candidate_gaussian_count"])
        survivor_count = int(
            report["dense_survivor_before_spatial_gaussian_count"]
        )
        validated_count = int(report["validated_fill_gaussian_count"])
        profiles.append(
            {
                "profile_name": str(report["profile_name"]),
                "pixel_confidence_thresholds": report[
                    "pixel_confidence_thresholds"
                ],
                "black_core_first_candidate_gaussian_count": candidate_count,
                "dense_survivor_before_spatial_gaussian_count": survivor_count,
                "dense_survivor_ratio": (
                    survivor_count / float(candidate_count)
                    if candidate_count
                    else 0.0
                ),
                "validated_fill_gaussian_count": validated_count,
                "validated_fill_ratio": (
                    validated_count / float(candidate_count)
                    if candidate_count
                    else 0.0
                ),
                "preferred_unlabeled_recovery_ratio": float(
                    report["preferred_unlabeled_recovery_ratio"]
                ),
                "accepted_spatial_component_count": int(
                    report["accepted_spatial_component_count"]
                ),
                "candidate_gate_reason_counts": report[
                    "candidate_gate_reason_counts"
                ],
                "class_validated_fill_gaussian_counts": report[
                    "class_validated_fill_gaussian_counts"
                ],
            }
        )
    return {
        "source": "dinov3_core_first_dense_cross_validation_sweep",
        "contract": "report_only_global_confidence_profile_sweep_v1",
        "scene": next(iter(scenes)),
        "report_only": True,
        "automatic_profile_selection": False,
        "manual_component_decisions": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "profile_count": len(profiles),
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for path in args.report:
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    summary = summarize_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
