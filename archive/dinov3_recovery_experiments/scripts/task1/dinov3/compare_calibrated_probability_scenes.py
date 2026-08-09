#!/usr/bin/env python3
"""Apply a fixed two-scene acceptance gate to calibrated DINOv3 fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.task1.dinov3.soft_probability_round_trip_audit import (
    CALIBRATED_CONTRACT,
    CALIBRATED_SOURCE,
)


SOURCE = "dinov3_calibrated_probability_two_scene_comparison"
CONTRACT = "locked_policy_playroom_drjohnson_acceptance_gate_v1"
METRICS = (
    "agreement_of_projected",
    "interior_agreement_of_projected",
    "boundary_agreement_of_projected",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_scene_reports(
    *,
    scene_role: str,
    hard: dict[str, Any],
    calibrated: dict[str, Any],
) -> dict[str, Any]:
    if calibrated.get("source") != CALIBRATED_SOURCE or calibrated.get("contract") != CALIBRATED_CONTRACT:
        raise ValueError(f"{scene_role} calibrated report has the wrong contract")
    if hard.get("report_only") is not True or calibrated.get("report_only") is not True:
        raise ValueError(f"{scene_role} inputs must both be report-only")
    if hard.get("camera_indices") != calibrated.get("camera_indices"):
        raise ValueError(f"{scene_role} hard and calibrated cameras differ")
    if int(hard.get("gaussian_count", -1)) != int(calibrated.get("gaussian_count", -2)):
        raise ValueError(f"{scene_role} hard and calibrated Gaussian counts differ")
    hard_metrics = hard.get("heldout_pixel_metrics", {})
    calibrated_metrics = calibrated.get("heldout_pixel_metrics", {})
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        hard_value = float(hard_metrics.get(metric, -1.0))
        calibrated_value = float(calibrated_metrics.get(metric, -1.0))
        if not 0.0 <= hard_value <= 1.0 or not 0.0 <= calibrated_value <= 1.0:
            raise ValueError(f"{scene_role} report has invalid {metric}")
        comparisons[metric] = {
            "hard": hard_value,
            "calibrated": calibrated_value,
            "delta": calibrated_value - hard_value,
            "non_regression": calibrated_value >= hard_value,
        }
    return {
        "scene_role": scene_role,
        "camera_count": int(calibrated["camera_count"]),
        "gaussian_count": int(calibrated["gaussian_count"]),
        "calibration_policy": calibrated["calibration_policy"],
        "metrics": comparisons,
        "all_metrics_non_regressing": all(
            item["non_regression"] for item in comparisons.values()
        ),
    }


def compare_scene_pair(
    playroom: dict[str, Any],
    drjohnson: dict[str, Any],
) -> dict[str, Any]:
    if playroom["calibration_policy"] != drjohnson["calibration_policy"]:
        raise ValueError("DrJohnson did not use the locked Playroom calibration policy")
    strict_overall_improvement = any(
        scene["metrics"]["agreement_of_projected"]["delta"] > 0.0
        for scene in (playroom, drjohnson)
    )
    accepted = (
        playroom["all_metrics_non_regressing"]
        and drjohnson["all_metrics_non_regressing"]
        and strict_overall_improvement
    )
    return {
        "accepted": accepted,
        "decision": "accept_calibrated_policy" if accepted else "retain_hard_vote_baseline",
        "locked_policy": playroom["calibration_policy"],
        "scenes": [playroom, drjohnson],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playroom-hard-report", required=True, type=Path)
    parser.add_argument("--playroom-calibrated-report", required=True, type=Path)
    parser.add_argument("--drjohnson-hard-report", required=True, type=Path)
    parser.add_argument("--drjohnson-calibrated-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    required = (
        args.playroom_hard_report,
        args.playroom_calibrated_report,
        args.drjohnson_hard_report,
        args.drjohnson_calibrated_report,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in required]
    playroom = compare_scene_reports(
        scene_role="playroom_development",
        hard=documents[0],
        calibrated=documents[1],
    )
    drjohnson = compare_scene_reports(
        scene_role="drjohnson_locked_validation",
        hard=documents[2],
        calibrated=documents[3],
    )
    pair = compare_scene_pair(playroom, drjohnson)
    output = {
        "source": SOURCE,
        "contract": CONTRACT,
        "report_only": True,
        "acceptance_rule": (
            "same_locked_policy_on_both_scenes_all_overall_interior_boundary_"
            "metrics_non_regressing_and_at_least_one_strict_overall_improvement"
        ),
        **pair,
        "input_sha256": {str(path): sha256_file(path) for path in required},
        "manual_scene_selection_used": False,
        "manual_candidate_selection_used": False,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "accepted_gaussian_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
