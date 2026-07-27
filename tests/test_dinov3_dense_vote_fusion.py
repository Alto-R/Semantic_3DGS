from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.fuse_dense_view_votes import (
    REASON_ACCEPTED,
    REASON_EXACT_TIE,
    REASON_INSUFFICIENT_VIEWS,
    REASON_UNOBSERVED,
    add_view_votes,
    decide_dense_labels,
)
from scripts.task1.dinov3.lift_dense_view_votes import dense_sparse_view_votes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_dense_vote_scene.sbatch"
)


class DensePixelLiftTest(unittest.TestCase):
    def test_every_visible_gaussian_receives_unit_class_mass(self) -> None:
        used = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        class_ids = np.asarray([0, 7, 11], dtype=np.uint16)
        indices, classes, weights = dense_sparse_view_votes(used, class_ids)

        totals = np.zeros((3,), dtype=np.float32)
        np.add.at(totals, indices, weights)
        np.testing.assert_allclose(totals, [1.0, 1.0, 1.0])

        by_pair = {
            (int(index), int(class_id)): float(weight)
            for index, class_id, weight in zip(indices, classes, weights)
        }
        self.assertAlmostEqual(by_pair[(0, 7)], 0.75)
        self.assertAlmostEqual(by_pair[(0, 11)], 0.25)
        self.assertAlmostEqual(by_pair[(1, 11)], 1.0)
        self.assertAlmostEqual(by_pair[(2, 7)], 1.0)


class DenseFusionTest(unittest.TestCase):
    def test_unique_soft_winner_and_structural_abstention_reasons(self) -> None:
        votes = np.asarray(
            [
                [1.4, 0.5, 0.0, 0.0],
                [0.6, 0.5, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        views = np.asarray([2, 2, 0, 1], dtype=np.uint16)
        winners, reasons, share, margin, _runner = decide_dense_labels(
            votes,
            views,
            min_views=2,
        )
        np.testing.assert_array_equal(winners, [1, 0, 0, 0])
        np.testing.assert_array_equal(
            reasons,
            [
                REASON_ACCEPTED,
                REASON_EXACT_TIE,
                REASON_UNOBSERVED,
                REASON_INSUFFICIENT_VIEWS,
            ],
        )
        self.assertAlmostEqual(float(share[0]), 0.7, places=6)
        self.assertAlmostEqual(float(margin[0]), 0.4, places=6)

    def test_default_one_view_policy_materializes_unique_observations(self) -> None:
        votes = np.asarray([[0.0], [1.0]], dtype=np.float32)
        winners, reasons, *_ = decide_dense_labels(
            votes,
            np.asarray([1], dtype=np.uint16),
            min_views=1,
        )
        np.testing.assert_array_equal(winners, [2])
        np.testing.assert_array_equal(reasons, [REASON_ACCEPTED])

    def test_each_camera_must_contribute_unit_mass(self) -> None:
        matrix = np.zeros((4, 3), dtype=np.float32)
        view_counts = np.zeros((3,), dtype=np.uint16)
        add_view_votes(
            matrix,
            view_counts,
            np.asarray([0, 0, 2], dtype=np.uint32),
            np.asarray([1, 2, 3], dtype=np.uint16),
            np.asarray([0.75, 0.25, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(view_counts, [1, 0, 1])
        self.assertAlmostEqual(float(matrix[:, 0].sum()), 1.0)
        self.assertAlmostEqual(float(matrix[:, 2].sum()), 1.0)


class DenseFusionSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_consumes_existing_dense_dinov3_cache(self) -> None:
        self.assertIn("SOURCE_VIEW_DIR", self.text)
        self.assertIn("dinov3_manifest.json", self.text)
        self.assertIn("lift_dense_view_votes", self.text)
        self.assertIn("fuse_dense_view_votes", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)

    def test_scheduler_has_complete_global_contract(self) -> None:
        self.assertIn("query_region_filtering_used=0", self.text)
        self.assertIn("confidence_threshold_used=0", self.text)
        self.assertIn("prior_semantic_labels_used=0", self.text)
        self.assertIn("dinov2_used=0", self.text)
        self.assertIn('MIN_VIEWS="${MIN_VIEWS:-1}"', self.text)
        self.assertNotIn("fuse_query_region_votes", self.text)
        self.assertNotIn("associate_3d_query_regions", self.text)

    def test_scheduler_writes_end_to_end_review_artifacts(self) -> None:
        self.assertIn("semantic_point_cloud_supersplat_debug.ply", self.text)
        self.assertIn("render_auto_label_overlays", self.text)
        self.assertIn("visible_overlay_coverage.json", self.text)
        self.assertIn("semantic_labels.png", self.text)


if __name__ == "__main__":
    unittest.main()
