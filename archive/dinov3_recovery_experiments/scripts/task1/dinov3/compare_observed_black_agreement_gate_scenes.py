#!/usr/bin/env python3
"""Apply the fixed two-scene DINOv2 agreement-gate acceptance decision.

The DINOv2 agreement gate admits camera-observed black fills only when every
camera's component vote required agreement between the DINOv3 and DINOv2
winners.  This comparator requires that both scenes ran with the gate, that
both scenes still pass the four held-out non-regression metrics, and that
directly measured fill precision is at least 0.50 in both scenes.  It is
report-only and never writes labels or PLY.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.task1.dinov3.compare_observed_black_component_graph_round_trip_scenes import (
    scene_gate,
    validate_report as validate_round_trip_report,
)
from scripts.task1.dinov3.compare_observed_black_fill_precision_scenes import (
    validate_report as validate_fill_report,
)


SCENE_NAMES = ("playroom", "drjohnson")
FILL_PRECISION_THRESHOLD = 0.50


def validate_agreement_gate_reports(
    round_trip: Mapping[str, Any],
    fill: Mapping[str, Any],
    scene: str,
) -> None:
    validate_round_trip_report(round_trip, scene)
    validate_fill_report(fill, scene)
    if round_trip.get("dinov2_agreement_gate_used") is not True:
        raise ValueError(f"{scene} round-trip report did not use the DINOv2 agreement gate")
    if fill.get("dinov2_agreement_gate_used") is not True:
        raise ValueError(f"{scene} fill report did not use the DINOv2 agreement gate")
    if not isinstance(fill.get("fill_precision_of_source"), (int, float)):
        raise ValueError(f"{scene} fill report lacks fill precision")


def scene_decision(
    round_trip: Mapping[str, Any],
    fill: Mapping[str, Any],
    scene: str,
) -> Dict[str, Any]:
    gate = scene_gate(round_trip)
    fill_precision = float(fill["fill_precision_of_source"])
    fill_passes = fill_precision >= FILL_PRECISION_THRESHOLD
    return {
        "dinov2_agreement_gate_used": True,
        "round_trip_gate": gate,
        "fill_precision_of_source": fill_precision,
        "fill_precision_threshold": FILL_PRECISION_THRESHOLD,
        "fill_precision_passes": fill_passes,
        "passes": gate["passes"] and fill_passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playroom-round-trip-report", required=True, type=Path)
    parser.add_argument("--drjohnson-round-trip-report", required=True, type=Path)
    parser.add_argument("--playroom-fill-report", required=True, type=Path)
    parser.add_argument("--drjohnson-fill-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    round_trip_reports = {
        "playroom": json.loads(
            args.playroom_round_trip_report.read_text(encoding="utf-8")
        ),
        "drjohnson": json.loads(
            args.drjohnson_round_trip_report.read_text(encoding="utf-8")
        ),
    }
    fill_reports = {
        "playroom": json.loads(
            args.playroom_fill_report.read_text(encoding="utf-8")
        ),
        "drjohnson": json.loads(
            args.drjohnson_fill_report.read_text(encoding="utf-8")
        ),
    }
    decisions = {}
    for scene in SCENE_NAMES:
        validate_agreement_gate_reports(
            round_trip_reports[scene], fill_reports[scene], scene
        )
        decisions[scene] = scene_decision(
            round_trip_reports[scene], fill_reports[scene], scene
        )
    accepted = all(decisions[scene]["passes"] for scene in SCENE_NAMES)
    blockers = []
    for scene in SCENE_NAMES:
        decision = decisions[scene]
        if not decision["round_trip_gate"]["passes"]:
            blockers.append(f"{scene} round-trip non-regression gate failed")
        if not decision["fill_precision_passes"]:
            blockers.append(
                f"{scene} fill precision "
                f"{decision['fill_precision_of_source']:.3f} below "
                f"{FILL_PRECISION_THRESHOLD:.2f}"
            )
    output = {
        "source": "dinov3_observed_black_agreement_gate_two_scene_gate",
        "contract": "paired_playroom_drjohnson_dinov2_agreement_gate_v1",
        "scenes": decisions,
        "both_scenes_present": True,
        "fill_precision_threshold": FILL_PRECISION_THRESHOLD,
        "class_agnostic": True,
        "accepted_for_materialization": accepted,
        "materialization_blocker": (
            None if accepted else "; ".join(blockers) or
            "one or both scenes failed the agreement gate"
        ),
        "manual_scene_selection_used": False,
        "manual_class_selection_used": False,
        "accepted_gaussian_labels_written": False,
        "semantic_ply_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
