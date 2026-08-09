#!/usr/bin/env python3
"""Require paired Playroom and DrJohnson recovery reports before promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.task1.dinov3.recover_detected_abstentions import CONTRACT, SOURCE


def validate_report(report: dict[str, Any], scene: str) -> None:
    expected = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": scene,
        "report_only": True,
        "immutable_anchor_labels_changed": 0,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "accepted_gaussian_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"{scene} report violates {field}={value!r}")


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
    for scene, report in reports.items():
        validate_report(report, scene)
    rows = {}
    for scene, report in reports.items():
        recovery = report["recovery"]
        calibration = report["calibration"]
        rows[scene] = {
            "detected_abstention_count": report["detected_abstention_count"],
            "additional_strict_majority_count": recovery[
                "additional_strict_majority_count"
            ],
            "calibrated_weighted_majority_count": recovery[
                "calibrated_weighted_majority_count"
            ],
            "total_recovered_count": recovery["total_recovered_count"],
            "total_recovered_ratio_of_detected_abstentions": recovery[
                "total_recovered_ratio_of_detected_abstentions"
            ],
            "remaining_detected_abstention_count": recovery[
                "remaining_detected_abstention_count"
            ],
            "anchor_candidate_precision": calibration["anchor_candidate_precision"],
            "anchor_candidate_coverage": calibration["anchor_candidate_coverage"],
        }
    output = {
        "source": "dinov3_detected_abstention_two_scene_comparison",
        "contract": "paired_playroom_drjohnson_report_only_recovery_gate_v1",
        "scenes": rows,
        "both_scenes_present": True,
        "immutable_anchor_labels_changed": 0,
        "ready_for_round_trip_validation": all(
            row["total_recovered_count"] > 0 for row in rows.values()
        ),
        "accepted_for_materialization": False,
        "materialization_blocker": (
            "candidate must next pass held-out overall, interior, and boundary "
            "round-trip comparisons independently in both scenes"
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
