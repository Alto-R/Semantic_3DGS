#!/usr/bin/env python3
"""Apply the fixed two-scene held-out gate to component-graph validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.task1.dinov3.observed_black_component_graph_round_trip_validation import (
    CONTRACT,
    SOURCE,
)


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
        "manual_class_selection_used",
        "scene_specific_rules",
        "accepted_gaussian_labels_written",
        "gaussian_project_class_array_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if report.get(field) not in (0, False):
            raise ValueError(f"{scene} report violates {field}")
    for section in ("baseline_metrics", "candidate_metrics", "delta"):
        if not isinstance(report.get(section), dict):
            raise ValueError(f"{scene} report lacks {section}")
    if not isinstance(report.get("per_class_recovery"), list):
        raise ValueError(f"{scene} report lacks per-class recovery")


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
    per_class = report.get("per_class_recovery", [])
    per_class_reported = isinstance(per_class, list) and bool(per_class)
    return {
        "baseline_metrics": {metric: float(baseline[metric]) for metric in METRICS},
        "candidate_metrics": {metric: float(candidate[metric]) for metric in METRICS},
        "delta": deltas,
        "non_regression": non_regression,
        "candidate_recovered_count": recovered,
        "full_evidence_eligible_gaussian_count": int(
            report.get("full_evidence_eligible_gaussian_count", 0)
        ),
        "per_class_recovery_reported": per_class_reported,
        "passes": (
            recovered > 0
            and per_class_reported
            and all(non_regression.values())
        ),
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
        "source": "dinov3_observed_black_component_graph_two_scene_gate",
        "contract": "paired_playroom_drjohnson_component_graph_round_trip_gate_v1",
        "scenes": gates,
        "both_scenes_present": True,
        "accepted_for_materialization": accepted,
        "materialization_blocker": None if accepted else (
            "component candidates regressed, recovered no held-out labels, or "
            "reported no per-class recovery in at least one scene"
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
