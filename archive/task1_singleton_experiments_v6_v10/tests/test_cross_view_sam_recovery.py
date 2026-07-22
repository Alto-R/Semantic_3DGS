from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from cluster_semantic_flashsplat_proposals import (  # noqa: E402
    SemanticGroup,
    filter_groups,
)
from recover_cross_view_sam_masks import (  # noqa: E402
    mask_containment,
    nested_competing_thing,
    project_points,
    projected_mask_coverage,
    projected_mask_membership,
    robust_projection_box,
)


def camera() -> dict[str, object]:
    return {
        "rotation": np.eye(3).tolist(),
        "position": [0.0, 0.0, 0.0],
        "fx": 100.0,
        "fy": 100.0,
        "width": 200,
        "height": 100,
    }


class ProjectionTest(unittest.TestCase):
    def test_project_points_matches_pinhole_camera(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 10.0],
                [1.0, 1.0, 10.0],
                [20.0, 0.0, 10.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        xy, depth, inside = project_points(points, camera(), 200, 100)

        np.testing.assert_allclose(xy[0], [100.0, 50.0])
        np.testing.assert_allclose(xy[1], [110.0, 60.0])
        self.assertEqual(depth.tolist(), [10.0, 10.0, 10.0, -1.0])
        self.assertEqual(inside.tolist(), [True, True, False, False])

    def test_projection_box_uses_robust_extent_and_padding(self) -> None:
        xy = np.asarray([[20.0, 30.0], [40.0, 50.0], [30.0, 40.0]])

        box = robust_projection_box(xy, 100, 80, padding_ratio=0.10, quantile=0.0)

        self.assertEqual(box, (18.0, 28.0, 42.0, 52.0))

    def test_projected_mask_coverage_samples_projected_points(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True
        xy = np.asarray([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]])

        self.assertAlmostEqual(projected_mask_coverage(mask, xy), 2.0 / 3.0)
        self.assertEqual(
            projected_mask_membership(mask, xy).tolist(),
            [True, True, False],
        )


class PartWholeGuardTest(unittest.TestCase):
    def test_nested_different_thing_is_reported_but_stuff_is_ignored(self) -> None:
        seed = np.zeros((12, 12), dtype=bool)
        seed[4:8, 4:8] = True
        wall = np.ones((12, 12), dtype=bool)
        door = np.zeros((12, 12), dtype=bool)
        door[2:10, 2:10] = True
        masks = np.stack([seed, wall, door])
        metadata = [
            {"class": "windowpane"},
            {"class": "wall"},
            {"class": "door"},
        ]

        result = nested_competing_thing(
            seed,
            "windowpane",
            masks,
            metadata,
            {"windowpane", "door"},
            minimum_outer_area_ratio=1.5,
        )

        self.assertEqual(result["class"], "door")
        self.assertEqual(result["mask_index"], 2)
        self.assertEqual(result["containment"], 1.0)
        self.assertEqual(mask_containment(seed, door), 1.0)


class RecoveryFusionGuardTest(unittest.TestCase):
    @staticmethod
    def group(verification_views: int) -> SemanticGroup:
        return SemanticGroup(
            group_id=0,
            class_name="door",
            indices=np.arange(20, dtype=np.uint32),
            proposal_ids=list(range(1, verification_views + 2)),
            source_frames={"seed", *{f"verify_{i}" for i in range(verification_views)}},
            independent_source_frames={"seed"},
            verification_source_frames={
                f"verify_{i}" for i in range(verification_views)
            },
            recovery_seed_keys={"seed:0"},
            scores=[0.5] * (verification_views + 1),
        )

    def test_singleton_recovery_requires_two_verification_views(self) -> None:
        rejected = filter_groups(
            [self.group(1)],
            min_group_gaussians=10,
            min_group_proposals=2,
            max_groups=10,
            priorities={},
            min_recovery_verification_views=2,
        )
        accepted = filter_groups(
            [self.group(2)],
            min_group_gaussians=10,
            min_group_proposals=2,
            max_groups=10,
            priorities={},
            min_recovery_verification_views=2,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)

    def test_two_independent_detections_remain_sufficient(self) -> None:
        group = self.group(1)
        group.independent_source_frames.add("independent_2")
        group.source_frames.add("independent_2")
        group.proposal_ids.append(99)

        accepted = filter_groups(
            [group],
            min_group_gaussians=10,
            min_group_proposals=2,
            max_groups=10,
            priorities={},
            min_recovery_verification_views=2,
        )

        self.assertEqual(len(accepted), 1)


if __name__ == "__main__":
    unittest.main()
