from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.abstention_round_trip_validation import (
    CONTRACT,
    SOURCE,
    candidate_without_camera,
    reproduce_recovery_candidate,
    update_metrics,
)
from scripts.task1.dinov3.compare_abstention_round_trip_scenes import (
    scene_gate,
    validate_report,
)
from scripts.task1.dinov3.recover_detected_abstentions import (
    accumulate_weighted_scores,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import consensus_statistics


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_abstention_round_trip_validation_scene.sbatch"
)


def evidence(camera_index: int, winners: list[int], mass: list[float]) -> dict:
    return {
        "camera_index": camera_index,
        "camera_id": camera_index,
        "file": f"camera_{camera_index}.png",
        "winners": np.asarray(winners, dtype=np.uint16),
        "mass": np.asarray(mass, dtype=np.float32),
    }


class CandidateExclusionTest(unittest.TestCase):
    def test_heldout_vote_is_removed_from_strict_and_weighted_evidence(self) -> None:
        baseline = [
            evidence(0, [1, 1], [1.0, 1.0]),
            evidence(1, [2, 1], [1.0, 1.0]),
        ]
        additional = [evidence(2, [1, 2], [1.0, 1.0])]
        baseline_counts = np.zeros((3, 2), dtype=np.uint8)
        combined_counts = np.zeros((3, 2), dtype=np.uint8)
        for item in baseline:
            columns = np.flatnonzero(item["winners"])
            baseline_counts[item["winners"][columns], columns] += 1
            combined_counts[item["winners"][columns], columns] += 1
        for item in additional:
            columns = np.flatnonzero(item["winners"])
            combined_counts[item["winners"][columns], columns] += 1
        reliabilities = {0: 0.1, 1: 0.1, 2: 0.9}
        target_indices = np.asarray([0], dtype=np.int64)
        weighted_scores, contributing = accumulate_weighted_scores(
            [baseline[1], *additional],
            {1: reliabilities[1], 2: reliabilities[2]},
            target_indices,
            class_count=2,
        )
        candidate, summary = candidate_without_camera(
            np.asarray([0, 0], dtype=np.uint16),
            np.asarray([0, 0], dtype=np.uint16),
            np.asarray([True, False]),
            weighted_scores,
            contributing,
            target_indices,
        )
        np.testing.assert_array_equal(candidate, [1, 0])
        self.assertEqual(summary["newly_resolved_count"], 1)
        self.assertEqual(summary["weighted_newly_resolved_count"], 1)

    def test_full_reproduction_matches_strict_then_weighted_recovery(self) -> None:
        baseline = [
            evidence(0, [1, 1], [1.0, 1.0]),
            evidence(1, [2, 1], [1.0, 1.0]),
        ]
        additional = [evidence(2, [1, 2], [1.0, 1.0])]
        baseline_counts = np.zeros((3, 2), dtype=np.uint8)
        combined_counts = np.zeros((3, 2), dtype=np.uint8)
        for item in baseline:
            columns = np.flatnonzero(item["winners"])
            baseline_counts[item["winners"][columns], columns] += 1
            combined_counts[item["winners"][columns], columns] += 1
        for item in additional:
            columns = np.flatnonzero(item["winners"])
            combined_counts[item["winners"][columns], columns] += 1
        baseline_statistics = consensus_statistics(baseline_counts)
        combined_statistics = consensus_statistics(combined_counts)
        labels, source_codes = reproduce_recovery_candidate(
            baseline_statistics,
            combined_statistics,
            baseline,
            additional,
            np.asarray([True, False]),
            np.asarray([0], dtype=np.int64),
            class_count=2,
        )
        np.testing.assert_array_equal(labels, [1, 1])
        np.testing.assert_array_equal(source_codes, [2, 1])


class PixelMetricTest(unittest.TestCase):
    def test_counts_overall_interior_and_boundary_from_shared_partition(self) -> None:
        values = {
            "pixels": 0,
            "projected": 0,
            "agreed": 0,
            "boundary_pixels": 0,
            "boundary_projected": 0,
            "boundary_agreed": 0,
            "interior_pixels": 0,
            "interior_projected": 0,
            "interior_agreed": 0,
        }
        update_metrics(
            values,
            np.asarray([[1, 2], [2, 2]], dtype=np.uint16),
            np.asarray([[1, 1], [1, 0]], dtype=bool),
            np.asarray([[1, 1], [2, 2]], dtype=np.uint16),
            np.asarray([[0, 1], [0, 1]], dtype=bool),
        )
        self.assertEqual(values["projected"], 3)
        self.assertEqual(values["agreed"], 2)
        self.assertEqual(values["boundary_projected"], 1)
        self.assertEqual(values["boundary_agreed"], 0)
        self.assertEqual(values["interior_agreed"], 2)


class TwoSceneGateTest(unittest.TestCase):
    @staticmethod
    def report() -> dict:
        return {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "baseline_metrics": {
                "projected_ratio": 0.90,
                "agreement_of_projected": 0.80,
                "interior_agreement_of_projected": 0.82,
                "boundary_agreement_of_projected": 0.50,
            },
            "candidate_metrics": {
                "projected_ratio": 0.91,
                "agreement_of_projected": 0.81,
                "interior_agreement_of_projected": 0.82,
                "boundary_agreement_of_projected": 0.50,
            },
            "delta": {},
            "candidate_recovered_count": 10,
            "immutable_anchor_labels_changed": 0,
            "manual_camera_selection_used": False,
            "manual_gaussian_selection_used": False,
            "accepted_gaussian_labels_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
        }

    def test_accepts_recovery_with_no_metric_regression(self) -> None:
        report = self.report()
        validate_report(report, "playroom")
        self.assertTrue(scene_gate(report)["passes"])

    def test_rejects_boundary_regression(self) -> None:
        report = self.report()
        report["candidate_metrics"]["boundary_agreement_of_projected"] = 0.49
        gate = scene_gate(report)
        self.assertFalse(gate["passes"])
        self.assertFalse(gate["non_regression"]["boundary_agreement_of_projected"])


class SchedulerContractTest(unittest.TestCase):
    def test_scheduler_is_automatic_report_only_and_uses_existing_caches(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "Manual camera selection is not accepted",
            "exclude_each_original_baseline_camera_from_baseline_and_recovery_evidence",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
            "SOURCE_CACHE_OUTPUT_NAME",
            "SOURCE_HARD_AUDIT_OUTPUT_NAME",
            "SOURCE_RECOVERY_OUTPUT_NAME",
            "--selection-report",
            "recompute_all_camera_reliabilities_after_excluding_the_heldout_camera",
            "candidate_heldout_overlays.png",
            '--output-dir "${AUDIT_DIR}"',
        ):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
