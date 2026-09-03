from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.audit_automatic_anchor_guard_fill import (
    AnchorGuardThresholds,
    PROFILES,
    class_consistent_anchor_mask,
    load_upstream_components,
)
from scripts.task1.dinov3.render_incremental_spatial_core_fill import (
    build_exact_residual_colors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts/slurm/slurm_task1_dinov3_automatic_anchor_guard_audit_scene.sbatch"
)


class AnchorGuardDecisionTest(unittest.TestCase):
    def test_global_profiles_only_vary_radius(self) -> None:
        self.assertEqual(
            [(item.name, item.radius_voxel_multiplier) for item in PROFILES],
            [("radius_1x", 1.0), ("radius_2x", 2.0), ("radius_4x", 4.0)],
        )

    def test_same_class_majority_and_distance_accept(self) -> None:
        distances = np.array([[0.10, 0.12, 0.14, 0.16, 0.30, 0.40]])
        neighbor_ids = np.array([[0, 1, 2, 3, 4, 5]])
        labels = np.array([15, 15, 15, 15, 1, 1], dtype=np.int32)
        accepted, metrics = class_consistent_anchor_mask(
            distances,
            neighbor_ids,
            labels,
            target_project_id=15,
            radius=0.5,
            min_same_class_neighbors=4,
            min_same_class_fraction=0.60,
            max_same_to_competing_distance_ratio=1.0,
        )
        np.testing.assert_array_equal(accepted, [True])
        np.testing.assert_array_equal(metrics["same_class_neighbor_count"], [4])

    def test_competing_class_closer_rejects_halo(self) -> None:
        distances = np.array([[0.05, 0.10, 0.12, 0.14, 0.16]])
        neighbor_ids = np.array([[4, 0, 1, 2, 3]])
        labels = np.array([23, 23, 23, 23, 1], dtype=np.int32)
        accepted, _metrics = class_consistent_anchor_mask(
            distances,
            neighbor_ids,
            labels,
            target_project_id=23,
            radius=0.5,
            min_same_class_neighbors=4,
            min_same_class_fraction=0.75,
            max_same_to_competing_distance_ratio=1.0,
        )
        np.testing.assert_array_equal(accepted, [False])

    def test_unanchored_candidate_abstains(self) -> None:
        distances = np.array([[0.10, 0.20, np.inf, np.inf]])
        neighbor_ids = np.array([[0, 1, 2, 2]])
        labels = np.array([6, 6], dtype=np.int32)
        accepted, metrics = class_consistent_anchor_mask(
            distances,
            neighbor_ids,
            labels,
            target_project_id=37,
            radius=0.5,
            min_same_class_neighbors=2,
            min_same_class_fraction=0.75,
            max_same_to_competing_distance_ratio=1.0,
        )
        np.testing.assert_array_equal(accepted, [False])
        np.testing.assert_array_equal(metrics["same_class_neighbor_count"], [0])

    def test_thresholds_reject_invalid_neighbor_requirement(self) -> None:
        with self.assertRaises(ValueError):
            AnchorGuardThresholds(
                neighbor_count=4,
                min_same_class_neighbors=5,
            ).validate()


class AnchorGuardSourceContractTest(unittest.TestCase):
    def test_loader_requires_disjoint_reported_source_supports(self) -> None:
        report = {
            "contract": "report_only_dinov3_incremental_spatial_core_fill_v1",
            "scene": "synthetic",
            "report_only": True,
            "preferred_labels_read_only": True,
            "preferred_labels_modified": False,
            "vertex_count": 5,
            "accepted_residual_component_count": 1,
            "incremental_fill_gaussian_count": 2,
            "components": [
                {
                    "component_id": 7,
                    "accepted": True,
                    "class": "door",
                    "project_id": 15,
                    "voxel_size": 0.1,
                    "source_proposal_ids": [2, 3],
                    "support_gaussian_count": 2,
                }
            ],
        }
        archive = {
            "component_000007_indices": np.array([1, 4], dtype=np.uint32),
            "component_000007_camera_counts": np.array([2, 3], dtype=np.uint16),
        }
        components = load_upstream_components(
            report,
            archive,
            vertex_count=5,
            scene="synthetic",
        )
        self.assertEqual(len(components), 1)
        np.testing.assert_array_equal(components[0].indices, [1, 4])

    def test_renderer_accepts_automatic_report_contract(self) -> None:
        report = {
            "contract": "report_only_dinov3_automatic_anchor_guard_fill_v1",
            "vertex_count": 4,
            "incremental_fill_gaussian_count": 2,
            "components": [
                {
                    "component_id": 1,
                    "accepted": True,
                    "class": "door",
                }
            ],
        }
        supports = {
            "component_000001_indices": np.array([1, 3], dtype=np.uint32)
        }
        _colors, mask, palette = build_exact_residual_colors(4, report, supports)
        np.testing.assert_array_equal(mask, [False, True, False, True])
        self.assertEqual(palette[0]["class"], "door")


class AnchorGuardSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_is_report_only_and_has_no_manual_decisions(self) -> None:
        self.assertIn("report_only=1", self.text)
        self.assertIn("manual_component_decisions=0", self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)
        self.assertNotIn("REVIEW_CONFIG", self.text)
        self.assertNotIn("gaussian_labels.npy\" \\", self.text)

    def test_scheduler_renders_all_global_profiles(self) -> None:
        self.assertIn(
            "PROFILES=(radius_1x radius_2x radius_4x)",
            self.text,
        )
        self.assertIn("render_incremental_spatial_core_fill", self.text)
        self.assertIn("automatic_fill_supports.npz", self.text)


if __name__ == "__main__":
    unittest.main()
