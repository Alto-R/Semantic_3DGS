#!/usr/bin/env python3
"""Apply the fixed two-scene held-out gate to recovery validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


SOURCE = "dinov3_detected_abstention_recovery_round_trip_validation"
CONTRACT = "report_only_leave_one_camera_out_recovery_comparison_v1"
SCENE_NAMES = ("playroom", "drjohnson")
METRICS = (
    "projected_ratio",
    "agreement_of_projected",
    "interior_agreement_of_projected",
    "boundary_agreement_of_projected",
)
TOLERANCE = 1e-6


def validate_report(report: Mapping[str, Any], scene: str) -> None:
    if report.get("source") != SOURCE:
        raise ValueError(f"{scene} report has the wrong source")
    if report.get("contract") != CONTRACT:
        raise ValueError(f"{scene} report has the wrong contract")
    if report.get("scene") != scene or report.get("report_only") is not True:
        raise ValueError(f"{scene} report has the wrong scene or mode")
    for field in (
        "immutable_anchor_labels_changed",
        "manual_camera_selection_used",
        "manual_gaussian_selection_used",
        "accepted_gaussian_labels_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if report.get(field) not in (0, False):
            raise ValueError(f"{scene} report violates {field}")
    for section in ("baseline_metrics", "candidate_metrics", "delta"):
        if not isinstance(report.get(section), dict):
            raise ValueError(f"{scene} report lacks {section}")


def scene_gate(report: Mapping[str, Any]) -> Dict[str, Any]:
    baseline = report["baseline_metrics"]
    candidate = report["candidate_metrics"]
    deltas = {
        metric: float(candidate[metric]) - float(baseline[metric])
        for metric in METRICS
    }
    non_regression = {
        metric: deltas[metric] >= -TOLERANCE for metric in METRICS
    }
    recovered = int(report.get("candidate_recovered_count", 0))
    return {
        "baseline_metrics": {metric: float(baseline[metric]) for metric in METRICS},
        "candidate_metrics": {metric: float(candidate[metric]) for metric in METRICS},
        "delta": deltas,
        "non_regression": non_regression,
        "candidate_recovered_count": recovered,
        "passes": recovered > 0 and all(non_regression.values()),
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
    for scene, report in reports.items():
        validate_report(report, scene)
    gates = {scene: scene_gate(report) for scene, report in reports.items()}
    accepted = all(gate["passes"] for gate in gates.values())
    output = {
        "source": "dinov3_abstention_round_trip_two_scene_gate",
        "contract": "paired_playroom_drjohnson_recovery_round_trip_gate_v1",
        "scenes": gates,
        "both_scenes_present": True,
        "accepted_for_materialization": accepted,
        "materialization_blocker": None if accepted else (
            "candidate regressed or recovered no labels in at least one scene"
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
