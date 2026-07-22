from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK1 = ROOT / "scripts" / "task1"
if str(TASK1) not in sys.path:
    sys.path.insert(0, str(TASK1))

from audit_independent_multiview_evidence import (  # noqa: E402
    IndependentDetection,
    angle_degrees,
    box_iou,
    contribution_support,
    diagnostic_lift_decision,
    identity_consistency_gate,
    match_independent_detection,
    radius_connected_components,
    retain_seed_touching_components,
    select_geometry_separated_views,
)
from recover_cross_view_sam_masks import ProjectionPrompt  # noqa: E402


def camera(position: tuple[float, float, float]) -> dict[str, object]:
    return {
        "position": list(position),
        "rotation": np.eye(3).tolist(),
        "width": 100,
        "height": 100,
        "fx": 50.0,
        "fy": 50.0,
    }


def prompt(camera_index: int, projected_fraction: float = 0.8) -> ProjectionPrompt:
    xy = np.asarray([[40.0, 40.0], [50.0, 50.0], [60.0, 60.0]])
    return ProjectionPrompt(
        seed_proposal_id=1,
        seed_key="source.png:0",
        class_name="door",
        target_frame_file=f"target_{camera_index}.png",
        target_camera_index=camera_index,
        bbox_xyxy=(35.0, 35.0, 65.0, 65.0),
        projected_xy=xy,
        projected_seed_positions=np.arange(xy.shape[0]),
        projected_gaussian_count=xy.shape[0],
        projected_fraction=projected_fraction,
        baseline_depth_ratio=0.2,
        seed_grounding_score=0.9,
        seed_sam_score=0.95,
        seed_frame_file="source.png",
    )


class GeometrySeparationTests(unittest.TestCase):
    def test_angle_degrees(self) -> None:
        self.assertAlmostEqual(
            angle_degrees(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])),
            90.0,
        )

    def test_near_duplicate_views_do_not_fill_two_slots(self) -> None:
        cameras = [
            camera((0.0, 0.0, -5.0)),
            camera((1.0, 0.0, -5.0)),
            camera((1.1, 0.0, -5.0)),
            camera((-1.0, 0.0, -5.0)),
        ]
        selected = select_geometry_separated_views(
            [prompt(1), prompt(2), prompt(3)],
            cameras[0],
            cameras,
            np.asarray([0.0, 0.0, 0.0]),
            min_source_angle_degrees=5.0,
            min_pairwise_angle_degrees=10.0,
            max_views=3,
        )
        selected_indices = {item.prompt.target_camera_index for item in selected}
        self.assertEqual(len(selected_indices), 2)
        self.assertIn(3, selected_indices)
        self.assertEqual(len(selected_indices & {1, 2}), 1)


class IndependentAssociationTests(unittest.TestCase):
    def test_detection_box_is_independent_and_projection_only_matches_it(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True
        detection = IndependentDetection(
            mask=mask,
            class_name="door",
            phrase="door",
            grounding_score=0.8,
            sam_score=0.95,
            bbox_xyxy=(30.0, 30.0, 70.0, 70.0),
        )
        matched, metrics = match_independent_detection(prompt(1), [detection])
        self.assertIs(matched, detection)
        self.assertEqual(metrics["projected_seed_coverage"], 1.0)
        self.assertGreater(metrics["projected_box_iou"], 0.0)

    def test_wrong_class_does_not_match_even_with_perfect_overlap(self) -> None:
        mask = np.ones((100, 100), dtype=bool)
        detection = IndependentDetection(
            mask=mask,
            class_name="wall",
            phrase="wall",
            grounding_score=0.99,
            sam_score=0.99,
            bbox_xyxy=(35.0, 35.0, 65.0, 65.0),
        )
        matched, metrics = match_independent_detection(prompt(1), [detection])
        self.assertIsNone(matched)
        self.assertEqual(metrics["association_score"], 0.0)

    def test_box_iou_is_symmetric(self) -> None:
        left = (0.0, 0.0, 10.0, 10.0)
        right = (5.0, 0.0, 15.0, 10.0)
        self.assertAlmostEqual(box_iou(left, right), box_iou(right, left))
        self.assertAlmostEqual(box_iou(left, right), 1.0 / 3.0)


class ContributionSupportTests(unittest.TestCase):
    def test_requires_visibility_and_positive_dominance(self) -> None:
        positive = np.asarray([0.06, 0.04, 0.08, 0.01], dtype=np.float32)
        negative = np.asarray([0.01, 0.00, 0.08, 0.00], dtype=np.float32)
        indices, metrics = contribution_support(
            positive,
            negative,
            min_total_contribution=0.05,
            min_positive_fraction=0.60,
        )
        np.testing.assert_array_equal(indices, np.asarray([0], dtype=np.uint32))
        self.assertEqual(metrics["visible_gaussian_count"], 2)
        self.assertEqual(metrics["positive_gaussian_count"], 1)


class IdentityConsistencyTests(unittest.TestCase):
    def test_accepts_balanced_two_view_seed_support(self) -> None:
        accepted, metrics = identity_consistency_gate([0.437, 0.399], 0.20, 0.50)
        self.assertTrue(accepted)
        self.assertAlmostEqual(metrics["second_seed_contribution_fraction"], 0.399)
        self.assertGreater(metrics["second_to_best_seed_contribution_ratio"], 0.90)

    def test_rejects_same_class_instance_confusion_pattern(self) -> None:
        accepted, metrics = identity_consistency_gate([0.739, 0.109], 0.20, 0.50)
        self.assertFalse(accepted)
        self.assertIn(
            "second_seed_contribution_fraction<0.2",
            metrics["reasons"],
        )
        self.assertIn(
            "second_to_best_seed_contribution_ratio<0.5",
            metrics["reasons"],
        )

    def test_near_threshold_failure_is_diagnostic_only(self) -> None:
        decision = diagnostic_lift_decision(
            grounding_score=0.80,
            sam_score=0.89819,
            association_overlap=0.985,
            min_grounding_score=0.30,
            min_sam_score=0.90,
            min_association_overlap=0.05,
            max_grounding_shortfall=0.05,
            max_sam_shortfall=0.025,
        )
        self.assertFalse(decision["quality_accepted"])
        self.assertTrue(decision["diagnostic_lift"])

    def test_low_quality_failure_is_not_lifted(self) -> None:
        decision = diagnostic_lift_decision(
            grounding_score=0.80,
            sam_score=0.87,
            association_overlap=0.985,
            min_grounding_score=0.30,
            min_sam_score=0.90,
            min_association_overlap=0.05,
            max_grounding_shortfall=0.05,
            max_sam_shortfall=0.025,
        )
        self.assertFalse(decision["quality_accepted"])
        self.assertFalse(decision["diagnostic_lift"])


class AdaptiveComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.xyz = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [11.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def test_radius_graph_separates_distant_surfaces(self) -> None:
        component_ids, component_sizes, metrics = radius_connected_components(
            self.xyz, radius=1.5
        )
        self.assertEqual(metrics["component_count"], 2)
        np.testing.assert_array_equal(np.sort(component_sizes), np.asarray([4, 4]))
        self.assertNotEqual(component_ids[0], component_ids[4])

    def test_only_components_touching_source_seed_are_retained(self) -> None:
        intersection = np.arange(self.xyz.shape[0], dtype=np.uint32)
        retained, removed, _component_ids, metrics = retain_seed_touching_components(
            intersection,
            np.asarray([0, 1], dtype=np.uint32),
            self.xyz,
            neighbor_k=1,
            radius_multiplier=1.5,
            spacing_max_samples=32,
        )
        np.testing.assert_array_equal(retained, np.asarray([0, 1, 2, 3]))
        np.testing.assert_array_equal(removed, np.asarray([4, 5, 6, 7]))
        self.assertEqual(metrics["seed_touching_component_count"], 1)

    def test_adaptive_radius_is_scale_free(self) -> None:
        intersection = np.arange(self.xyz.shape[0], dtype=np.uint32)
        _retained, _removed, _ids, base = retain_seed_touching_components(
            intersection,
            np.asarray([0], dtype=np.uint32),
            self.xyz,
            neighbor_k=1,
            radius_multiplier=1.5,
            spacing_max_samples=32,
        )
        _retained, _removed, _ids, scaled = retain_seed_touching_components(
            intersection,
            np.asarray([0], dtype=np.uint32),
            self.xyz * 10.0,
            neighbor_k=1,
            radius_multiplier=1.5,
            spacing_max_samples=32,
        )
        self.assertAlmostEqual(
            scaled["connectivity_radius"],
            10.0 * base["connectivity_radius"],
        )
        self.assertEqual(scaled["retained_gaussian_count"], 4)


if __name__ == "__main__":
    unittest.main()
