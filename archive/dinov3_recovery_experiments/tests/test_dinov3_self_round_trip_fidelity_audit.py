from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.self_round_trip_fidelity_audit import (
    CONTRACT,
    aggregate_hardening_metrics,
    metric_ratios,
    pixel_metric_counts,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = (
    ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "self_round_trip_fidelity_audit.py"
)
SCHEDULER = (
    ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_self_round_trip_fidelity_audit_scene.sbatch"
)


class SelfRoundTripFidelityAuditTest(unittest.TestCase):
    def test_contract_is_same_camera_report_only(self) -> None:
        self.assertEqual(
            CONTRACT,
            "report_only_same_camera_flashsplat_self_round_trip_fidelity_v1",
        )

    def test_pixel_metrics_separate_boundary_and_interior(self) -> None:
        source = np.asarray(
            [[1, 1, 2], [1, 1, 2]],
            dtype=np.uint16,
        )
        predicted = np.asarray(
            [[1, 1, 1], [1, 2, 2]],
            dtype=np.uint16,
        )
        valid = np.asarray(
            [[True, True, True], [True, False, True]],
            dtype=bool,
        )
        counts, boundary, agreement = pixel_metric_counts(
            source, predicted, valid
        )
        self.assertEqual(
            counts,
            {
                "pixels": 6,
                "projected": 5,
                "agreed": 4,
                "boundary_pixels": 4,
                "boundary_projected": 3,
                "boundary_agreed": 2,
                "interior_pixels": 2,
                "interior_projected": 2,
                "interior_agreed": 2,
            },
        )
        np.testing.assert_array_equal(
            boundary,
            [[False, True, True], [False, True, True]],
        )
        np.testing.assert_array_equal(
            agreement,
            [[True, True, False], [True, False, True]],
        )
        ratios = metric_ratios(counts)
        self.assertAlmostEqual(ratios["projected_ratio"], 5 / 6)
        self.assertAlmostEqual(ratios["agreement_of_projected"], 4 / 5)
        self.assertAlmostEqual(ratios["boundary_agreement_of_projected"], 2 / 3)
        self.assertEqual(ratios["interior_agreement_of_projected"], 1.0)

    def test_pixel_metrics_reject_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            pixel_metric_counts(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.uint8),
                np.zeros((2, 2), dtype=bool),
            )

    def test_ratios_are_safe_when_no_pixels_project(self) -> None:
        counts = {
            "pixels": 4,
            "projected": 0,
            "agreed": 0,
            "boundary_pixels": 0,
            "boundary_projected": 0,
            "boundary_agreed": 0,
            "interior_pixels": 4,
            "interior_projected": 0,
            "interior_agreed": 0,
        }
        ratios = metric_ratios(counts)
        self.assertEqual(ratios["projected_ratio"], 0.0)
        self.assertEqual(ratios["agreement_of_projected"], 0.0)
        self.assertEqual(ratios["boundary_agreement_of_projected"], 0.0)
        self.assertEqual(ratios["interior_agreement_of_projected"], 0.0)

    def test_hardening_metrics_weight_gaussians_not_cameras(self) -> None:
        summaries = [
            {
                "visible_gaussian_count": 4,
                "unique_camera_winner_count": 3,
                "camera_abstain_count": 1,
                "exact_tie_count": 1,
                "winning_mass": {"count": 3, "mean": 0.8},
            },
            {
                "visible_gaussian_count": 2,
                "unique_camera_winner_count": 2,
                "camera_abstain_count": 0,
                "exact_tie_count": 0,
                "winning_mass": {"count": 2, "mean": 0.5},
            },
        ]
        metrics = aggregate_hardening_metrics(summaries)
        self.assertEqual(metrics["visible_gaussian_count"], 6)
        self.assertEqual(metrics["unique_camera_winner_count"], 5)
        self.assertAlmostEqual(metrics["unique_winner_ratio_of_visible"], 5 / 6)
        self.assertEqual(metrics["winning_mass_count"], 5)
        self.assertAlmostEqual(metrics["winning_mass_mean"], 0.68)

    def test_module_performs_no_cross_camera_fusion(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("collapse_camera_distribution", source)
        self.assertIn("render_binary_project_ids", source)
        self.assertNotIn("consensus_statistics(", source)
        self.assertNotIn("leave_one_out_consensus(", source)
        self.assertIn('"cross_camera_fusion_used": False', source)
        self.assertIn('"flashsplat_lifting_rerun": False', source)
        self.assertIn('"semantic_ply_written": False', source)

    def test_scheduler_reuses_both_caches_and_is_report_only(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("SOURCE_CACHE_OUTPUT_NAME", source)
        self.assertIn("SOURCE_VOTE_OUTPUT_NAME", source)
        self.assertIn("self_round_trip_fidelity_audit", source)
        self.assertNotIn("dinov3_segment_views", source)
        self.assertNotIn("lift_dense_view_votes", source)
        self.assertIn("dinov3_inference_rerun=0", source)
        self.assertIn("flashsplat_lifting_rerun=0", source)
        self.assertIn("flashsplat_vote_cache_reused=1", source)
        self.assertIn("cross_camera_fusion_used=0", source)
        self.assertIn("Manual CAMERA_INDICES and VIEW_COUNT are not accepted", source)
        self.assertIn("gaussian_labels.npy", source)
        self.assertIn("unexpectedly wrote a PLY", source)


if __name__ == "__main__":
    unittest.main()
