from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.fuse_query_region_votes import (
    REASON_ACCEPTED,
    REASON_INSUFFICIENT_VIEWS,
    REASON_NO_OBJECT_MAJORITY,
    REASON_NO_STRICT_MAJORITY,
    REASON_SOFT_HARD_DISAGREEMENT,
    choose_view_owners,
    decide_multiview_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_region_vote_scene.sbatch"
)
FUSION = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "fuse_query_region_votes.py"
)


class PerViewOwnershipTest(unittest.TestCase):
    def test_each_camera_selects_strongest_region_once_per_gaussian(self) -> None:
        touched, owners = choose_view_owners(
            [
                (
                    np.asarray([0, 1, 2], dtype=np.uint32),
                    np.asarray([5.0, 2.0, 1.0], dtype=np.float32),
                ),
                (
                    np.asarray([1, 2, 3], dtype=np.uint32),
                    np.asarray([1.0, 4.0, 3.0], dtype=np.float32),
                ),
            ],
            vertex_count=5,
        )
        np.testing.assert_array_equal(touched, [0, 1, 2, 3])
        np.testing.assert_array_equal(owners, [0, 0, 1, 1, -1])


class StrictMajorityFusionTest(unittest.TestCase):
    def test_acceptance_and_abstention_reasons_are_global(self) -> None:
        # Four semantic classes plus the no-object channel at index 4.
        hard = np.zeros((5, 5), dtype=np.uint8)
        soft = np.zeros((5, 5), dtype=np.float32)
        views = np.asarray([3, 2, 3, 3, 1], dtype=np.uint8)

        hard[0, 1] = 2
        hard[0, 2] = 1
        soft[0] = [0.1, 1.8, 1.1, 0.0, 0.0]

        hard[1, 1] = 1
        hard[1, 2] = 1
        soft[1] = [0.0, 1.0, 1.0, 0.0, 0.0]

        hard[2, 1] = 2
        hard[2, 2] = 1
        soft[2] = [0.0, 1.1, 1.8, 0.1, 0.0]

        hard[3, 4] = 2
        hard[3, 1] = 1
        soft[3] = [0.0, 0.8, 0.0, 0.0, 2.2]

        hard[4, 1] = 1
        soft[4, 1] = 1.0

        winners, reasons, agreement, confidence = decide_multiview_labels(
            hard,
            soft,
            views,
            no_object_index=4,
            min_views=2,
        )
        np.testing.assert_array_equal(winners, [1, -1, -1, -1, -1])
        np.testing.assert_array_equal(
            reasons,
            [
                REASON_ACCEPTED,
                REASON_NO_STRICT_MAJORITY,
                REASON_SOFT_HARD_DISAGREEMENT,
                REASON_NO_OBJECT_MAJORITY,
                REASON_INSUFFICIENT_VIEWS,
            ],
        )
        self.assertAlmostEqual(float(agreement[0]), 2.0 / 3.0, places=3)
        self.assertGreater(float(confidence[0]), 0.5)


class DirectRegionVoteContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER.read_text(encoding="utf-8")
        cls.fusion = FUSION.read_text(encoding="utf-8")

    def test_scheduler_uses_cached_regions_without_association_or_prior_labels(self) -> None:
        self.assertIn("fuse_query_region_votes", self.scheduler)
        self.assertIn("region_association_used=0", self.scheduler)
        self.assertNotIn("associate_3d_query_regions", self.scheduler)
        for forbidden in (
            "BASE_LABELS",
            "BASE_LABEL_MAP",
            "DINOV2_LABELS",
            "DINOV2_OUTPUT",
        ):
            self.assertNotIn(forbidden, self.scheduler.upper())
        self.assertIn("prior_semantic_labels_used=0", self.scheduler)
        self.assertIn("dinov2_used=0", self.scheduler)

    def test_fusion_has_one_camera_vote_and_strict_majority_contract(self) -> None:
        self.assertIn("choose_view_owners", self.fusion)
        self.assertIn("hard_counts.astype(np.uint16) * 2 >", self.fusion)
        self.assertIn("soft_winners != hard_winners", self.fusion)
        self.assertIn('"region_association_used": False', self.fusion)
        self.assertIn('"scene_specific_rules": False', self.fusion)
        self.assertIn('"class_specific_thresholds": False', self.fusion)


if __name__ == "__main__":
    unittest.main()
