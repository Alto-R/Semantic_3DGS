from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.lift_soft_probability_view_votes import (
    FIXED_SAMPLE_COUNT,
    deterministic_stratified_thresholds,
    normalize_soft_support,
    validate_one_to_one_ade20k_ontology,
    validate_probability_tensor,
    winner_agreement,
)
from scripts.task1.dinov3.soft_probability_round_trip_audit import (
    confusion_counts,
    soft_consensus,
    validate_camera_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_soft_probability_round_trip_scene.sbatch"
)


class SoftProbabilityRoundTripTest(unittest.TestCase):
    def test_thresholds_are_deterministic_stratified_and_camera_specific(self) -> None:
        first = deterministic_stratified_thresholds(3, 5, 8, 7)
        repeated = deterministic_stratified_thresholds(3, 5, 8, 7)
        another = deterministic_stratified_thresholds(3, 5, 8, 8)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, another))
        self.assertTrue(np.all((first >= 0.0) & (first < 1.0)))
        np.testing.assert_array_equal(np.sum(first < 0.25, axis=1), 2)

    def test_probability_validation_rejects_incomplete_mass(self) -> None:
        values = np.zeros((150, 2, 3), dtype=np.float16)
        values[0] = 1.0
        validate_probability_tensor(values, (150, 2, 3))
        values[0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_probability_tensor(values, (150, 2, 3))

    def test_real_ade20k_ontology_is_the_required_one_to_one_mapping(self) -> None:
        ontology = load_ontology(PROJECT_ROOT / "configs" / "ade20k_to_project.json")
        lookup = validate_one_to_one_ade20k_ontology(ontology)
        np.testing.assert_array_equal(
            lookup[:150], np.arange(1, 151, dtype=np.uint16)
        )
        np.testing.assert_array_equal(lookup[150:], np.zeros(106, dtype=np.uint16))

    def test_ontology_validation_rejects_nonzero_unused_lookup_entries(self) -> None:
        lookup = np.zeros(256, dtype=np.uint16)
        lookup[:150] = np.arange(1, 151, dtype=np.uint16)
        lookup[200] = 1
        ontology = SimpleNamespace(class_count=150, ade_to_project=lookup)
        with self.assertRaisesRegex(ValueError, "must map to zero"):
            validate_one_to_one_ade20k_ontology(ontology)

    def test_normalizes_soft_flashsplat_support(self) -> None:
        support = np.zeros((151, 4), dtype=np.float32)
        support[1, 0] = 1.0
        support[2, 0] = 3.0
        support[3, 2] = 2.0
        indices, probabilities, _total = normalize_soft_support(support)
        np.testing.assert_array_equal(indices, [0, 2])
        self.assertEqual(probabilities.shape, (2, 150))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(probabilities[0, :2], [0.25, 0.75])

    def test_half_full_convergence_reports_winner_agreement(self) -> None:
        first = np.zeros((151, 2), dtype=np.float32)
        second = np.zeros((151, 2), dtype=np.float32)
        first[1, :] = [3, 1]
        first[2, :] = [1, 3]
        second[1, :] = [6, 1]
        second[2, :] = [2, 7]
        result = winner_agreement(first, second)
        self.assertEqual(result["common_visible_gaussian_count"], 2)
        self.assertEqual(result["winner_agreement_ratio"], 1.0)

    def test_soft_consensus_argmaxes_only_after_equal_camera_fusion(self) -> None:
        evidence = np.asarray(
            [
                [1.4, 1.0, 0.0],
                [0.6, 1.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = soft_consensus(evidence, np.asarray([2, 2, 1], dtype=np.uint16))
        np.testing.assert_array_equal(result["labels"], [1, 0, 0])
        self.assertAlmostEqual(float(result["winning_probability"][0]), 0.7)
        self.assertTrue(result["exact_tie"][1])

    def test_camera_distribution_requires_sorted_normalized_rows(self) -> None:
        indices = np.asarray([1, 3], dtype=np.uint32)
        values = np.zeros((2, 3), dtype=np.float16)
        values[:, 0] = 1.0
        checked_indices, checked = validate_camera_distribution(
            indices, values, gaussian_count=4, class_count=3
        )
        np.testing.assert_array_equal(checked_indices, indices)
        np.testing.assert_allclose(checked.sum(axis=1), 1.0)
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            validate_camera_distribution(
                indices[::-1], values, gaussian_count=4, class_count=3
            )

    def test_confusion_counts_accumulate_without_signed_unsigned_promotion(self) -> None:
        source = np.asarray([[1, 2], [2, 0]], dtype=np.uint16)
        predicted = np.asarray([[1, 1], [2, 0]], dtype=np.uint16)
        valid = np.asarray([[True, True], [True, False]])
        counts = confusion_counts(
            source,
            predicted,
            valid,
            class_count=2,
        )
        self.assertEqual(counts.dtype, np.uint64)
        self.assertEqual(int(counts[1, 1]), 1)
        self.assertEqual(int(counts[2, 1]), 1)
        self.assertEqual(int(counts[2, 2]), 1)
        aggregate = np.zeros((3, 3), dtype=np.uint64)
        aggregate += counts
        np.testing.assert_array_equal(aggregate, counts)

    def test_scheduler_is_automatic_fixed_report_only_and_rental_compatible(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("Manual CAMERA_INDICES and VIEW_COUNT are not accepted", source)
        self.assertIn("TARGET_TWO_VIEW_PERCENT=99", source)
        self.assertIn("DINOV3_SAVE_PROBABILITIES=1", source)
        self.assertIn("SOURCE_CACHE_OUTPUT_NAME", source)
        self.assertIn("compare-hard-cache", source)
        self.assertIn("RESUME_FROM_SOFT_LIFT", source)
        self.assertIn("RESUME_FROM_SOFT_AUDIT", source)
        self.assertIn("03_soft_probability_round_trip_retry_v1", source)
        self.assertIn("08_retry_soft_leave_one_camera_out_round_trip", source)
        self.assertIn('if [[ "${RESUME_FROM_SOFT_AUDIT}" != "1" ]]', source)
        self.assertIn("selected_view_cache validate-resume", source)
        self.assertIn("Resume refuses partial or completed stages 7-8", source)
        self.assertNotIn("rm -rf", source)
        self.assertIn("DINOV3_CROP_SIZE=512", source)
        self.assertIn("DINOV3_STRIDE=384", source)
        self.assertIn("DINOV3_MAX_CUDA_MEMORY_GIB=40", source)
        self.assertIn("--sample-count 8", source)
        self.assertIn("semantic_3dgs_renderer", source)
        self.assertIn("equal_camera_full_probability_sum_then_argmax", source)
        self.assertIn("unexpectedly wrote a PLY", source)
        self.assertEqual(FIXED_SAMPLE_COUNT, 8)


if __name__ == "__main__":
    unittest.main()
