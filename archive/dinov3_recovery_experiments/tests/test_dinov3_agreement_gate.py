from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.task1.dinov3.compare_observed_black_agreement_gate_scenes import (
    FILL_PRECISION_THRESHOLD,
    main as gate_main,
    scene_decision,
    validate_agreement_gate_reports,
)
from scripts.task1.dinov3.observed_black_component_graph_round_trip_validation import (
    CONTRACT as ROUND_TRIP_CONTRACT,
    SOURCE as ROUND_TRIP_SOURCE,
)
from scripts.task1.dinov3.observed_black_fill_precision_audit import (
    CONTRACT as FILL_CONTRACT,
    SOURCE as FILL_SOURCE,
)


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_agreement_gate_pipeline_scene.sbatch"
)


def round_trip_report(
    scene: str,
    *,
    gate_used: bool = True,
    candidate_metrics_shift: float = 0.0,
) -> dict:
    metrics = {
        "projected_ratio": 1.0,
        "agreement_of_projected": 0.95,
        "interior_agreement_of_projected": 0.96,
        "boundary_agreement_of_projected": 0.90,
    }
    candidate = {key: value + candidate_metrics_shift for key, value in metrics.items()}
    return {
        "source": ROUND_TRIP_SOURCE,
        "contract": ROUND_TRIP_CONTRACT,
        "scene": scene,
        "report_only": True,
        "component_audit_report": "audit.json",
        "component_diagnostics": "diagnostics.npz",
        "dinov2_vote_manifest": "vote_manifest.json",
        "dinov2_agreement_gate_used": gate_used,
        "baseline_metrics": metrics,
        "candidate_metrics": candidate,
        "delta": {key: candidate[key] - value for key, value in metrics.items()},
        "per_class_recovery": [{"class_id": 15, "recovered": 10}],
        "candidate_recovered_count": 10,
        "full_evidence_eligible_gaussian_count": 20,
        "immutable_anchor_labels_changed": 0,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "manual_class_selection_used": False,
        "scene_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }


def fill_report(
    scene: str,
    *,
    gate_used: bool = True,
    precision: float = 0.60,
) -> dict:
    return {
        "source": FILL_SOURCE,
        "contract": FILL_CONTRACT,
        "scene": scene,
        "report_only": True,
        "component_audit_report": "audit.json",
        "component_diagnostics": "diagnostics.npz",
        "dinov2_vote_manifest": "vote_manifest.json",
        "dinov2_agreement_gate_used": gate_used,
        "fill_metrics": {
            "fill_pixels": 100,
            "fill_source_pixels": 60,
            "fill_agreed": int(60 * precision),
            "fill_wrong": 60 - int(60 * precision),
            "fill_unverifiable_pixels": 40,
        },
        "fill_precision_of_source": precision,
        "per_class_fill": [{"class_id": 15, "fill_precision": precision}],
        "candidate_recovered_count": 10,
        "immutable_anchor_labels_changed": 0,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "manual_class_selection_used": False,
        "scene_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }


class AgreementGateDecisionTest(unittest.TestCase):
    def test_accepts_when_both_scenes_pass_all_conditions(self) -> None:
        for scene in ("playroom", "drjohnson"):
            decision = scene_decision(
                round_trip_report(scene), fill_report(scene), scene
            )
            self.assertTrue(decision["round_trip_gate"]["passes"])
            self.assertTrue(decision["fill_precision_passes"])
            self.assertTrue(decision["passes"])

    def test_rejects_fill_precision_below_threshold(self) -> None:
        decision = scene_decision(
            round_trip_report("playroom"),
            fill_report("playroom", precision=FILL_PRECISION_THRESHOLD - 0.01),
            "playroom",
        )
        self.assertFalse(decision["fill_precision_passes"])
        self.assertFalse(decision["passes"])

    def test_requires_gate_used_in_both_reports(self) -> None:
        with self.assertRaisesRegex(ValueError, "round-trip report did not use"):
            validate_agreement_gate_reports(
                round_trip_report("playroom", gate_used=False),
                fill_report("playroom"),
                "playroom",
            )
        with self.assertRaisesRegex(ValueError, "fill report did not use"):
            validate_agreement_gate_reports(
                round_trip_report("playroom"),
                fill_report("playroom", gate_used=False),
                "playroom",
            )


class AgreementGateMainTest(unittest.TestCase):
    def test_main_writes_paired_gate_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {}
            for scene in ("playroom", "drjohnson"):
                (root / f"{scene}_round_trip.json").write_text(
                    json.dumps(round_trip_report(scene)), encoding="utf-8"
                )
                (root / f"{scene}_fill.json").write_text(
                    json.dumps(fill_report(scene)), encoding="utf-8"
                )
                outputs[scene] = root
            output = root / "paired.json"
            argv = [
                "gate",
                "--playroom-round-trip-report",
                str(root / "playroom_round_trip.json"),
                "--drjohnson-round-trip-report",
                str(root / "drjohnson_round_trip.json"),
                "--playroom-fill-report",
                str(root / "playroom_fill.json"),
                "--drjohnson-fill-report",
                str(root / "drjohnson_fill.json"),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                gate_main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["accepted_for_materialization"])
            self.assertEqual(result["fill_precision_threshold"], FILL_PRECISION_THRESHOLD)
            self.assertTrue(result["class_agnostic"])

    def test_main_rejects_below_threshold_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in ("playroom", "drjohnson"):
                precision = 0.60 if scene == "playroom" else 0.40
                (root / f"{scene}_round_trip.json").write_text(
                    json.dumps(round_trip_report(scene)), encoding="utf-8"
                )
                (root / f"{scene}_fill.json").write_text(
                    json.dumps(fill_report(scene, precision=precision)),
                    encoding="utf-8",
                )
            output = root / "paired.json"
            argv = [
                "gate",
                "--playroom-round-trip-report",
                str(root / "playroom_round_trip.json"),
                "--drjohnson-round-trip-report",
                str(root / "drjohnson_round_trip.json"),
                "--playroom-fill-report",
                str(root / "playroom_fill.json"),
                "--drjohnson-fill-report",
                str(root / "drjohnson_fill.json"),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                gate_main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(result["accepted_for_materialization"])
            self.assertIn("drjohnson fill precision", result["materialization_blocker"])


class PipelineSchedulerContractTest(unittest.TestCase):
    def test_pipeline_is_gated_report_only_and_reuses_caches(self) -> None:
        source = PIPELINE_SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "DINOV2_VOTE_MANIFEST",
            "export DINOV2_VOTE_MANIFEST",
            "export COMPONENT_AUDIT_OUTPUT_NAME ROUND_TRIP_OUTPUT_NAME",
            'OUTPUT_NAME="${output_name}"',
            "slurm_task1_dinov3_observed_black_component_graph_audit_scene.sbatch",
            "slurm_task1_dinov3_observed_black_component_graph_round_trip_validation_scene.sbatch",
            "slurm_task1_dinov3_observed_black_fill_precision_audit_scene.sbatch",
            "slurm_task1_dinov3_observed_black_winner_disagreement_audit_scene.sbatch",
            "compare_observed_black_agreement_gate_scenes",
            "class_agnostic=1",
            "report_only=1",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("v" + "5", source.lower())


if __name__ == "__main__":
    unittest.main()
