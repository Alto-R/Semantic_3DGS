from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.recover_detected_abstentions import (
    collapse_camera_with_mass,
    unique_weighted_strict_majority,
    wilson_lower_bound,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    STATUS_UNOBSERVED,
)
from scripts.task1.dinov3.select_abstention_evidence_cameras import (
    TARGET_COVERAGE,
    greedy_abstention_camera_selection,
    required_additional_evidence,
)
from scripts.task1.dinov3.compare_detected_abstention_scenes import validate_report
from scripts.task1.dinov3.recover_detected_abstentions import CONTRACT, SOURCE


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_detected_abstention_recovery_scene.sbatch"
)


class EvidenceRequirementTest(unittest.TestCase):
    def test_uses_exact_two_camera_and_majority_deficits(self) -> None:
        required = required_additional_evidence(
            np.asarray([0, 1, 2, 4, 5], dtype=np.uint16),
            np.asarray([0, 1, 1, 2, 2], dtype=np.uint8),
            np.asarray(
                [
                    STATUS_UNOBSERVED,
                    STATUS_SINGLE_CAMERA,
                    STATUS_EXACT_TIE,
                    STATUS_NO_STRICT_MAJORITY,
                    STATUS_NO_STRICT_MAJORITY,
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(required, [0, 1, 1, 1, 2])

    def test_greedy_selection_excludes_baseline_and_reaches_fixed_target(self) -> None:
        rows = np.asarray(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        packed = np.packbits(rows, axis=1, bitorder="little")
        cameras = [
            {
                "id": index,
                "position": [float(index), 0.0, 0.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
            for index in range(4)
        ]
        result = greedy_abstention_camera_selection(
            packed,
            gaussian_count=4,
            camera_indices=np.arange(4, dtype=np.int32),
            baseline_camera_indices=np.asarray([0], dtype=np.int32),
            required=np.asarray([1, 1, 1, 0], dtype=np.uint16),
            cameras=cameras,
        )
        self.assertEqual(result["target_coverage"], TARGET_COVERAGE)
        self.assertNotIn(0, result["selected_camera_indices"].tolist())
        self.assertGreaterEqual(result["achieved_units"], result["target_units"])
        self.assertEqual(result["selected_camera_indices"].tolist(), [3, 1])


class CalibratedRecoveryCoreTest(unittest.TestCase):
    def test_collapse_returns_unique_winner_mass_and_abstains_on_tie(self) -> None:
        winners, mass, _ = collapse_camera_with_mass(
            np.asarray([0, 0, 1, 1], dtype=np.uint32),
            np.asarray([1, 2, 1, 2], dtype=np.uint16),
            np.asarray([0.8, 0.2, 0.5, 0.5], dtype=np.float32),
            gaussian_count=2,
            class_count=2,
        )
        np.testing.assert_array_equal(winners, [1, 0])
        np.testing.assert_allclose(mass, [0.8, 0.0])

    def test_weighted_fusion_requires_unique_strict_majority(self) -> None:
        scores = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.5, 0.4],
                [0.3, 0.5, 0.35],
                [0.0, 0.0, 0.25],
            ],
            dtype=np.float32,
        )
        result = unique_weighted_strict_majority(scores)
        np.testing.assert_array_equal(result["prediction"], [1, 0, 0])

    def test_wilson_weight_is_conservative(self) -> None:
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)
        self.assertLess(wilson_lower_bound(90, 100), 0.9)
        self.assertGreater(wilson_lower_bound(90, 100), 0.8)


class SchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCHEDULER.read_text(encoding="utf-8")

    def test_is_automatic_report_only_and_standalone(self) -> None:
        for expected in (
            "Manual CAMERA_INDICES and VIEW_COUNT are not accepted",
            "accepted_hard_labels_immutable=1",
            "manual_camera_selection_used=0",
            "manual_gaussian_selection_used=0",
            "v5_used=0",
            "dinov2_used=0",
            "semantic_ply_written=0",
            "Report-only recovery unexpectedly wrote a PLY",
        ):
            self.assertIn(expected, self.source)

    def test_orders_selection_inference_lift_and_recovery(self) -> None:
        selection = self.source.index("01_select_additional_evidence_cameras")
        inference = self.source.index("02_incremental_dinov3_views")
        lift = self.source.index("03_lift_additional_hard_evidence")
        recovery = self.source.index("04_report_detected_abstention_recovery")
        self.assertLess(selection, inference)
        self.assertLess(inference, lift)
        self.assertLess(lift, recovery)


class TwoSceneGateContractTest(unittest.TestCase):
    def test_requires_report_only_immutable_scene_contract(self) -> None:
        report = {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "immutable_anchor_labels_changed": 0,
            "manual_camera_selection_used": False,
            "manual_gaussian_selection_used": False,
            "accepted_gaussian_labels_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
        }
        validate_report(report, "playroom")
        report["semantic_ply_written"] = True
        with self.assertRaisesRegex(ValueError, "semantic_ply_written"):
            validate_report(report, "playroom")


if __name__ == "__main__":
    unittest.main()
