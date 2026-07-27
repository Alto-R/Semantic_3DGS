from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.audit_core_first_dense_cross_validation import (
    decide_camera_winners,
    validate_vote_manifest,
)
from scripts.task1.dinov3.calibrate_dense_confidence import (
    empirical_cdf,
    histogram_indices,
    profile_minimum_bins,
    weakest_empirical_rank_bins,
)
from scripts.task1.dinov3.lift_calibrated_dense_view_votes import (
    CONTRACT as CALIBRATED_VOTE_CONTRACT,
    confidence_levels_from_bins,
    pair_index_map,
    profile_used_count,
)
from scripts.task1.dinov3.lift_confident_dense_view_votes import (
    SOURCE as VOTE_SOURCE,
    confident_sparse_view_votes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / (
        "slurm_task1_dinov3_core_first_calibrated_"
        "dense_cross_validation_audit_scene.sbatch"
    )
)


class JointEmpiricalCalibrationTest(unittest.TestCase):
    def test_metric_cdfs_produce_weakest_empirical_rank(self) -> None:
        values = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
        histogram = np.bincount(
            histogram_indices(values, 4).ravel(),
            minlength=4,
        )
        cdf = empirical_cdf(histogram)
        segment = {
            "confidence": values,
            "max_softmax_probability": values,
            "normalized_entropy_confidence": values,
        }
        score_bins = weakest_empirical_rank_bins(
            segment,
            {
                "relative_margin": cdf,
                "max_softmax_probability": cdf,
                "normalized_entropy_confidence": cdf,
            },
            4,
        )
        np.testing.assert_array_equal(score_bins, [[1, 2, 3]])

    def test_profiles_are_automatic_nested_joint_quantiles(self) -> None:
        histogram = np.ones((100,), dtype=np.uint64)
        profiles = profile_minimum_bins(histogram)
        minimum_bins = [
            int(profile["minimum_weakest_rank_bin"])
            for profile in profiles
        ]
        self.assertEqual(
            [profile["profile_name"] for profile in profiles],
            ["baseline", "permissive", "balanced", "strict"],
        )
        self.assertEqual(minimum_bins[0], 0)
        self.assertEqual(minimum_bins, sorted(minimum_bins))
        retained = [
            float(profile["actual_joint_retained_ratio"])
            for profile in profiles
        ]
        self.assertEqual(retained, sorted(retained, reverse=True))
        self.assertTrue(all(value > 0.0 for value in retained))


class OneRenderNestedProfileTest(unittest.TestCase):
    def test_pair_encoding_reconstructs_profiles_and_abstain_mass(self) -> None:
        project_ids = np.asarray([[1, 2], [1, 2]], dtype=np.uint16)
        levels = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
        indexed, row_projects, row_levels = pair_index_map(
            project_ids,
            levels,
            project_class_count=2,
        )
        self.assertEqual(indexed.shape, project_ids.shape)
        used = np.zeros((row_projects.size, 1), dtype=np.float32)
        for row in range(1, row_projects.size):
            project_id = int(row_projects[row])
            level = int(row_levels[row])
            used[row, 0] = {
                (1, 0): 1.0,
                (2, 1): 2.0,
                (1, 2): 3.0,
                (2, 3): 4.0,
            }[(project_id, level)]

        baseline, baseline_classes = profile_used_count(
            used,
            row_projects,
            row_levels,
            minimum_level=0,
        )
        strict, strict_classes = profile_used_count(
            used,
            row_projects,
            row_levels,
            minimum_level=3,
        )
        np.testing.assert_array_equal(baseline_classes, [0, 1, 2])
        np.testing.assert_allclose(baseline[:, 0], [0.0, 4.0, 6.0])
        np.testing.assert_array_equal(strict_classes, [0, 2])
        np.testing.assert_allclose(strict[:, 0], [6.0, 4.0])

        _indices, _classes, _weights, visible, accepted = (
            confident_sparse_view_votes(strict, strict_classes)
        )
        np.testing.assert_array_equal(visible, [0])
        np.testing.assert_allclose(accepted, [0.4])

    def test_pixel_profile_membership_is_nested(self) -> None:
        profiles = [
            {"profile_name": "baseline", "minimum_weakest_rank_bin": 0},
            {"profile_name": "permissive", "minimum_weakest_rank_bin": 4},
            {"profile_name": "balanced", "minimum_weakest_rank_bin": 7},
            {"profile_name": "strict", "minimum_weakest_rank_bin": 9},
        ]
        levels = confidence_levels_from_bins(
            np.asarray([0, 4, 7, 9], dtype=np.int32),
            profiles,
        )
        np.testing.assert_array_equal(levels, [0, 1, 2, 3])


class CalibratedCameraDecisionTest(unittest.TestCase):
    def test_semantic_winner_is_normalized_by_accepted_mass(self) -> None:
        reliable, matches = decide_camera_winners(
            1,
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([1, 2], dtype=np.uint16),
            np.asarray([0.20, 0.05], dtype=np.float32),
            np.asarray([0.25], dtype=np.float32),
            np.asarray([1], dtype=np.uint16),
            min_accepted_fraction=0.10,
            min_winner_share=0.50,
            min_winner_margin=0.10,
        )
        np.testing.assert_array_equal(reliable, [1])
        np.testing.assert_array_equal(matches, [True])

        insufficient_coverage, _matches = decide_camera_winners(
            1,
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([1, 2], dtype=np.uint16),
            np.asarray([0.03, 0.01], dtype=np.float32),
            np.asarray([0.04], dtype=np.float32),
            np.asarray([1], dtype=np.uint16),
            min_accepted_fraction=0.10,
            min_winner_share=0.50,
            min_winner_margin=0.10,
        )
        np.testing.assert_array_equal(insufficient_coverage, [0])

    def test_calibrated_manifest_accepts_unfiltered_baseline_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            source_ply = temporary / "source.ply"
            source_ply.write_bytes(b"placeholder")
            manifest = {
                "source": VOTE_SOURCE,
                "contract": CALIBRATED_VOTE_CONTRACT,
                "profile_name": "baseline",
                "gaussian_count": 3,
                "ply_path": str(source_ply),
                "query_region_filtering_used": False,
                "confidence_threshold_used": False,
                "inference_rerun": False,
                "flashsplat_rerun": True,
                "abstain_mass_preserved": True,
                "scene_specific_rules": False,
                "class_specific_thresholds": False,
                "manual_component_decisions": False,
                "v5_used": False,
                "dinov2_used": False,
                "joint_calibration_used": True,
                "single_flashsplat_render_shared_across_profiles": True,
                "automatic_min_camera_accepted_fraction": 0.5,
                "frames": [{"file": "camera.png", "camera_index": 0}],
            }
            validate_vote_manifest(
                manifest,
                vertex_count=3,
                source_ply=source_ply,
            )
            manifest["profile_name"] = "strict"
            with self.assertRaisesRegex(ValueError, "confidence_threshold_used"):
                validate_vote_manifest(
                    manifest,
                    vertex_count=3,
                    source_ply=source_ply,
                )


class CalibratedSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_joint_calibration_and_one_lift_are_automatic(self) -> None:
        self.assertIn("calibrate_dense_confidence", self.text)
        self.assertIn("CALIBRATION_SOURCE_A_DIR", self.text)
        self.assertIn("CALIBRATION_SOURCE_B_DIR", self.text)
        self.assertEqual(
            self.text.count(
                "python -m scripts.task1.dinov3."
                "lift_calibrated_dense_view_votes"
            ),
            1,
        )
        self.assertIn(
            "single_flashsplat_render_shared_across_profiles=1",
            self.text,
        )
        self.assertIn(
            "profile_target_retention=1.00,0.50,0.25,0.10",
            self.text,
        )
        self.assertNotIn("RELATIVE_MARGINS=", self.text)
        self.assertNotIn("ENTROPY_CONFIDENCES=", self.text)
        self.assertNotIn("MAX_PROBABILITIES=", self.text)

    def test_report_only_and_no_manual_decisions(self) -> None:
        self.assertIn("preferred_labels_read_only=1", self.text)
        self.assertIn("manual_component_decisions=0", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("semantic_project_class_arrays_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn("-name 'gaussian_labels.npy'", self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)


if __name__ == "__main__":
    unittest.main()
