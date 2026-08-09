#!/usr/bin/env python3
"""Summarize the raw-vs-weighted disagreement diagnostic across both scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.task1.dinov3.observed_black_winner_disagreement_audit import (
    CONTRACT,
    SOURCE,
)


SCENE_NAMES = ("playroom", "drjohnson")


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
    if not isinstance(report.get("disagreement_by_subtype"), dict):
        raise ValueError(f"{scene} report lacks disagreement decomposition")
    if not isinstance(report.get("fill_confusion"), list):
        raise ValueError(f"{scene} report lacks fill confusion")


def scene_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "disagreement_by_subtype": report["disagreement_by_subtype"],
        "fill_metrics": report["fill_metrics"],
        "candidate_recovered_count": int(report["candidate_recovered_count"]),
        "raw_winner_vs_heldout_confusion": report["raw_winner_vs_heldout_confusion"],
        "weighted_winner_vs_heldout_confusion": report[
            "weighted_winner_vs_heldout_confusion"
        ],
        "fill_confusion": report["fill_confusion"],
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
    summaries = {scene: scene_summary(report) for scene, report in reports.items()}
    output = {
        "source": "dinov3_observed_black_winner_disagreement_two_scene_summary",
        "contract": "paired_playroom_drjohnson_winner_disagreement_summary_v1",
        "scenes": summaries,
        "both_scenes_present": True,
        "measurement_only": True,
        "accepted_for_materialization": False,
        "materialization_blocker": (
            "winner disagreement is a diagnostic; acceptance still requires a "
            "reviewed rule change, validation, and materializer"
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
