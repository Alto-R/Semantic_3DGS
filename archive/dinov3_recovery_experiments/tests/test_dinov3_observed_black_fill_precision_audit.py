from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.compare_observed_black_fill_precision_scenes import (
    scene_summary,
    validate_report,
)
from scripts.task1.dinov3.observed_black_fill_precision_audit import (
    CONTRACT,
    SOURCE,
    fill_labels,
    update_fill_metrics,
    update_fill_per_class,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_observed_black_fill_precision_audit_scene.sbatch"
)
MODULE = ROOT / "scripts" / "task1" / "dinov3" / (
    "observed_black_fill_precision_audit.py"
)
COMPARATOR = ROOT / "scripts" / "task1" / "dinov3" / (
    "compare_observed_black_fill_precision_scenes.py"
)


class FillLabelTest(unittest.TestCase):
    def test_fill_labels_keep_only_newly_resolved_black_gaussians(self) -> None:
        baseline = np.asarray([0, 0, 5, 5, 0], dtype=np.uint16)
        candidate = np.asarray([0, 15, 5, 5, 20], dtype=np.uint16)
        fills = fill_labels(baseline, candidate)
        np.testing.assert_array_equal(fills, [0, 15, 0, 0, 20])

    def test_fill_labels_require_same_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "different shapes"):
            fill_labels(np.zeros((3,), dtype=np.uint16), np.zeros((4,), dtype=np.uint16))


class FillMetricTest(unittest.TestCase):
    def test_metrics_separate_source_unverifiable_and_wrong(self) -> None:
        values = {
            "fill_pixels": 0,
            "fill_source_pixels": 0,
            "fill_agreed": 0,
            "fill_wrong": 0,
            "fill_unverifiable_pixels": 0,
        }
        update_fill_metrics(
            values,
            np.asarray([[1, 2], [3, 0]], dtype=np.uint16),
            np.asarray([[1, 1], [1, 1]], dtype=bool),
            np.asarray([[1, 1], [2, 0]], dtype=np.uint16),
        )
        self.assertEqual(values["fill_pixels"], 4)
        self.assertEqual(values["fill_source_pixels"], 3)
        self.assertEqual(values["fill_agreed"], 1)
        self.assertEqual(values["fill_wrong"], 2)
        self.assertEqual(values["fill_unverifiable_pixels"], 1)

    def test_per_class_counts_agreed_and_wrong(self) -> None:
        values = {}
        update_fill_per_class(
            values,
            np.asarray([[1, 2], [3, 0]], dtype=np.uint16),
            np.asarray([[1, 1], [1, 1]], dtype=bool),
            np.asarray([[1, 1], [2, 0]], dtype=np.uint16),
        )
        self.assertEqual(values[1]["source_pixels"], 2)
        self.assertEqual(values[1]["agreed"], 1)
        self.assertEqual(values[1]["wrong"], 1)
        self.assertEqual(values[2]["source_pixels"], 1)
        self.assertEqual(values[2]["agreed"], 0)
        self.assertEqual(values[2]["wrong"], 1)


class TwoSceneSummaryTest(unittest.TestCase):
    @staticmethod
    def report() -> dict:
        return {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "fill_metrics": {
                "fill_pixels": 100,
                "fill_source_pixels": 80,
                "fill_agreed": 60,
                "fill_wrong": 20,
                "fill_unverifiable_pixels": 20,
            },
            "fill_precision_of_source": 0.75,
            "candidate_recovered_count": 10,
            "per_class_fill": [{"project_id": 15}],
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

    def test_summary_reports_fill_precision(self) -> None:
        report = self.report()
        validate_report(report, "playroom")
        summary = scene_summary(report)
        self.assertEqual(summary["fill_precision_of_source"], 0.75)
        self.assertEqual(summary["fill_metrics"]["fill_agreed"], 60)

    def test_rejects_missing_per_class_fill(self) -> None:
        report = self.report()
        report["per_class_fill"] = "missing"
        with self.assertRaisesRegex(ValueError, "per-class fill precision"):
            validate_report(report, "playroom")


class SchedulerContractTest(unittest.TestCase):
    def test_scheduler_is_automatic_report_only_and_reuses_caches(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "DINOV2_VOTE_MANIFEST",
            "--dinov2-vote-manifest",
            "RUNNER_UP_CAP",
            "--runner-up-cap",
            "CLASS_AWARE",
            "--class-aware",
            "Manual camera selection is not accepted",
            "measurement=fill_precision_of_black_spot_fills",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
            "SOURCE_CACHE_OUTPUT_NAME",
            "SOURCE_HARD_AUDIT_OUTPUT_NAME",
            "SOURCE_RECOVERY_OUTPUT_NAME",
            "SOURCE_COMPONENT_AUDIT_OUTPUT_NAME",
            "--component-audit-report",
            "--component-diagnostics",
            "fill_heldout_overlays.png",
            '--output-dir "${AUDIT_DIR}"',
        ):
            self.assertIn(expected, source)
        disallowed = "v" + "5"
        for path in (MODULE, COMPARATOR, SCHEDULER):
            self.assertNotIn(disallowed, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
