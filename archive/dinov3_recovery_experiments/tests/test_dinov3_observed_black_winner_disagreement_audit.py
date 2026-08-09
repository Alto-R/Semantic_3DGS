from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.compare_observed_black_winner_disagreement_scenes import (
    scene_summary,
    validate_report,
)
from scripts.task1.dinov3.observed_black_winner_disagreement_audit import (
    CONTRACT,
    RAW_TIED,
    SOURCE,
    WEIGHTED_NOT_ACCEPTED,
    WINNERS_DISAGREE,
    camera_component_votes,
    disagreement_subtype,
    update_fill_confusion,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_observed_black_winner_disagreement_audit_scene.sbatch"
)
MODULE = ROOT / "scripts" / "task1" / "dinov3" / (
    "observed_black_winner_disagreement_audit.py"
)
COMPARATOR = ROOT / "scripts" / "task1" / "dinov3" / (
    "compare_observed_black_winner_disagreement_scenes.py"
)


class CameraComponentVoteTest(unittest.TestCase):
    def test_mirrors_component_audit_mass_normalization(self) -> None:
        component_ids = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
        observed_mask = np.asarray([True, True, True, True, True])
        black_indices = np.arange(5, dtype=np.int64)
        winners = np.asarray([15, 15, 20, 21, 0], dtype=np.uint16)
        masses = np.asarray([1.0, 0.6, 0.5, 0.5, 0.0], dtype=np.float32)
        votes = camera_component_votes(
            component_ids,
            observed_mask,
            winners,
            masses,
            black_indices,
            class_count=25,
        )
        # Component 0: class 15 with 1.6/1.6 -> unique strict majority.
        self.assertEqual(int(votes["winner"][0]), 15)
        self.assertTrue(bool(votes["accepted"][0]))
        # Component 1: exact tie -> abstain.
        self.assertEqual(int(votes["winner"][1]), 0)
        self.assertFalse(bool(votes["accepted"][1]))
        self.assertTrue(bool(votes["tied"][1]))
        # Component 2: no evidence -> abstain.
        self.assertEqual(int(votes["winner"][2]), 0)


class DisagreementSubtypeTest(unittest.TestCase):
    def test_classifies_all_conflict_cases(self) -> None:
        self.assertEqual(
            disagreement_subtype(15, 15, True, True, False), WINNERS_DISAGREE
        )
        self.assertEqual(
            disagreement_subtype(15, 20, True, True, False), WINNERS_DISAGREE
        )
        self.assertEqual(
            disagreement_subtype(15, 15, True, False, False),
            WEIGHTED_NOT_ACCEPTED,
        )
        self.assertEqual(
            disagreement_subtype(15, 15, False, True, True), RAW_TIED
        )


class FillConfusionTest(unittest.TestCase):
    def test_counts_predicted_vs_source_pixels(self) -> None:
        matrix = np.zeros((26, 26), dtype=np.uint64)
        update_fill_confusion(
            matrix,
            np.asarray([[15, 20], [21, 0]], dtype=np.uint16),
            np.asarray([[1, 1], [1, 1]], dtype=bool),
            np.asarray([[15, 15], [20, 0]], dtype=np.uint16),
        )
        self.assertEqual(int(matrix[15, 15]), 1)
        self.assertEqual(int(matrix[20, 15]), 1)
        self.assertEqual(int(matrix[21, 20]), 1)


class TwoSceneSummaryTest(unittest.TestCase):
    @staticmethod
    def report() -> dict:
        return {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "disagreement_by_subtype": {
                RAW_TIED: {"component_count": 1},
                WINNERS_DISAGREE: {"component_count": 2},
                WEIGHTED_NOT_ACCEPTED: {"component_count": 0},
            },
            "fill_metrics": {
                "fill_pixels": 100,
                "fill_source_pixels": 80,
                "fill_agreed": 60,
                "fill_precision_of_source": 0.75,
            },
            "candidate_recovered_count": 10,
            "raw_winner_vs_heldout_confusion": [],
            "weighted_winner_vs_heldout_confusion": [],
            "fill_confusion": [{"pixels": 5}],
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

    def test_summary_reports_disagreement_decomposition(self) -> None:
        report = self.report()
        validate_report(report, "playroom")
        summary = scene_summary(report)
        self.assertEqual(
            summary["disagreement_by_subtype"][WINNERS_DISAGREE]["component_count"], 2
        )

    def test_rejects_missing_fill_confusion(self) -> None:
        report = self.report()
        report["fill_confusion"] = "missing"
        with self.assertRaisesRegex(ValueError, "fill confusion"):
            validate_report(report, "playroom")


class SchedulerContractTest(unittest.TestCase):
    def test_scheduler_is_automatic_report_only_and_reuses_caches(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "DINOV2_VOTE_MANIFEST",
            "--dinov2-vote-manifest",
            "Manual camera selection is not accepted",
            "measurement=raw_vs_weighted_winner_disagreement_and_fill_confusion",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
            "SOURCE_CACHE_OUTPUT_NAME",
            "SOURCE_HARD_AUDIT_OUTPUT_NAME",
            "SOURCE_RECOVERY_OUTPUT_NAME",
            "SOURCE_COMPONENT_AUDIT_OUTPUT_NAME",
            "--component-audit-report",
            "--component-diagnostics",
            '--output-dir "${AUDIT_DIR}"',
        ):
            self.assertIn(expected, source)
        disallowed = "v" + "5"
        for path in (MODULE, COMPARATOR, SCHEDULER):
            self.assertNotIn(disallowed, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
