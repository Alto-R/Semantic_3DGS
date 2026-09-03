from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.materialize_abstention_recovery import (
    CONTRACT,
    SOURCE,
    build_label_map,
    validate_candidate_arrays,
    validate_gate,
    validate_recovery_report,
)
from scripts.task1.dinov2.dinov2_ontology import load_ontology


ROOT = Path(__file__).resolve().parents[1]


def validation_report() -> dict:
    baseline = {
        "projected_ratio": 0.90,
        "agreement_of_projected": 0.80,
        "interior_agreement_of_projected": 0.82,
        "boundary_agreement_of_projected": 0.50,
    }
    candidate = {
        "projected_ratio": 0.91,
        "agreement_of_projected": 0.81,
        "interior_agreement_of_projected": 0.83,
        "boundary_agreement_of_projected": 0.51,
    }
    return {
        "source": "dinov3_detected_abstention_recovery_round_trip_validation",
        "contract": "report_only_leave_one_camera_out_recovery_comparison_v1",
        "scene": "playroom",
        "report_only": True,
        "recovery_candidate_reproduced": True,
        "candidate_recovered_count": 12,
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "delta": {key: candidate[key] - baseline[key] for key in baseline},
        "coverage_non_regression": True,
        "overall_non_regression": True,
        "interior_non_regression": True,
        "boundary_non_regression": True,
        "accepted_gaussian_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }


def gate_report() -> dict:
    validation = validation_report()
    return {
        "source": "dinov3_abstention_round_trip_two_scene_gate",
        "contract": "paired_playroom_drjohnson_recovery_round_trip_gate_v1",
        "both_scenes_present": True,
        "accepted_for_materialization": True,
        "scenes": {
            "playroom": {
                "passes": True,
                "candidate_recovered_count": 12,
                "baseline_metrics": validation["baseline_metrics"],
                "candidate_metrics": validation["candidate_metrics"],
                "delta": validation["delta"],
            },
            "drjohnson": {"passes": True, "candidate_recovered_count": 8},
        },
    }


class GateValidationTest(unittest.TestCase):
    def test_accepts_exact_passed_scene_gate(self) -> None:
        validate_gate(gate_report(), "playroom", validation_report())

    def test_rejects_unaccepted_pair(self) -> None:
        gate = gate_report()
        gate["accepted_for_materialization"] = False
        with self.assertRaisesRegex(ValueError, "has not accepted"):
            validate_gate(gate, "playroom", validation_report())

    def test_rejects_candidate_count_mismatch(self) -> None:
        gate = gate_report()
        gate["scenes"]["playroom"]["candidate_recovered_count"] = 13
        with self.assertRaisesRegex(ValueError, "recovered counts differ"):
            validate_gate(gate, "playroom", validation_report())


class RecoveryContractTest(unittest.TestCase):
    def test_requires_report_only_recovery_contract(self) -> None:
        report = {
            "source": "dinov3_detected_abstention_recovery_audit",
            "contract": "immutable_hard_anchor_incremental_strict_then_calibrated_v1",
            "scene": "playroom",
            "gaussian_count": 3,
            "report_only": True,
            "accepted_gaussian_labels_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
        }
        validate_recovery_report(report, "playroom", 3)
        report["semantic_ply_written"] = True
        with self.assertRaisesRegex(ValueError, "semantic_ply_written"):
            validate_recovery_report(report, "playroom", 3)


class CandidateArrayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.status = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)
        self.hard = np.asarray([1, 0, 0, 0, 0], dtype=np.int32)
        self.labels = np.asarray([1, 0, 2, 0, 3], dtype=np.int32)
        self.codes = np.asarray([1, 0, 2, 0, 3], dtype=np.uint8)

    def test_accepts_locked_anchor_and_detected_recoveries(self) -> None:
        recovered = validate_candidate_arrays(
            self.labels,
            self.codes,
            self.status,
            self.hard,
            expected_recovered_count=2,
        )
        self.assertEqual(recovered, 2)

    def test_rejects_zero_camera_recovery(self) -> None:
        self.labels[1] = 2
        self.codes[1] = 2
        with self.assertRaisesRegex(ValueError, "zero-camera"):
            validate_candidate_arrays(
                self.labels,
                self.codes,
                self.status,
                self.hard,
                expected_recovered_count=3,
            )

    def test_rejects_locked_anchor_code_on_abstention(self) -> None:
        self.labels[2] = 1
        self.codes[2] = 1
        with self.assertRaisesRegex(ValueError, "outside accepted anchors"):
            validate_candidate_arrays(
                self.labels,
                self.codes,
                self.status,
                self.hard,
                expected_recovered_count=1,
            )


class LabelMapTest(unittest.TestCase):
    def test_label_map_uses_project_ontology_ids(self) -> None:
        ontology = load_ontology(ROOT / "configs" / "ade20k_to_project.json")
        result = build_label_map(
            "playroom",
            ontology,
            np.asarray([0, 1, 1, 2], dtype=np.uint16),
            validation_report=Path("validation.json"),
            ontology=Path("ontology.json"),
        )
        self.assertEqual(result["source"], SOURCE)
        self.assertEqual(result["contract"], CONTRACT)
        self.assertEqual(result["ontology"], "ontology.json")
        self.assertEqual([item["id"] for item in result["labels"]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
