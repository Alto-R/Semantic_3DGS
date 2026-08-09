from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.calibrated_probability_fusion import (
    CONTRACT,
    SELECTION_RULE,
    SOURCE,
    calibrate_probability_distribution,
    evaluate_policy,
    fixed_candidate_policies,
    load_selected_policy,
    select_candidate_result,
)
from scripts.task1.dinov3.compare_calibrated_probability_scenes import (
    compare_scene_pair,
    compare_scene_reports,
)
from scripts.task1.dinov3.soft_probability_round_trip_audit import (
    CALIBRATED_CONTRACT,
    CALIBRATED_SOURCE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_calibrated_probability_scene.sbatch"
)
SOFT_SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_soft_probability_round_trip_scene.sbatch"
)


def probability_rows() -> np.ndarray:
    values = np.zeros((3, 150), dtype=np.float32)
    values[0, :4] = [0.6, 0.25, 0.1, 0.05]
    values[1, :4] = [0.4, 0.4, 0.1, 0.1]
    values[2, :] = 1.0 / 150.0
    return values


class CalibratedProbabilityFusionTest(unittest.TestCase):
    def test_fixed_candidate_family_has_required_anchors_and_no_duplicates(self) -> None:
        policies = fixed_candidate_policies(150)
        self.assertEqual(len(policies), 18)
        self.assertEqual(len({item["id"] for item in policies}), len(policies))
        self.assertEqual(policies[0]["id"], "full_t1_uniform")
        self.assertEqual(policies[1]["id"], "top1_t1_uniform")
        self.assertTrue(any(item["temperature"] < 1.0 for item in policies))
        self.assertTrue(any(item["confidence_weight"] == "margin" for item in policies))
        self.assertTrue(any(item["confidence_weight"] == "entropy" for item in policies))
        with self.assertRaisesRegex(ValueError, "ADE20K-150"):
            fixed_candidate_policies(3)

    def test_top_k_retains_boundary_ties_and_renormalizes(self) -> None:
        policy = next(
            item for item in fixed_candidate_policies(150)
            if item["id"] == "top1_t1_uniform"
        )
        result = calibrate_probability_distribution(probability_rows(), policy)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(int(np.count_nonzero(result[0])), 1)
        self.assertEqual(int(np.count_nonzero(result[1])), 2)
        np.testing.assert_allclose(result[1, :2], [0.5, 0.5])

    def test_temperature_sharpens_without_changing_the_winner(self) -> None:
        policies = {item["id"]: item for item in fixed_candidate_policies(150)}
        plain = calibrate_probability_distribution(
            probability_rows()[:1], policies["full_t1_uniform"]
        )
        sharp = calibrate_probability_distribution(
            probability_rows()[:1], policies["full_t05_uniform"]
        )
        self.assertEqual(int(np.argmax(plain[0])), int(np.argmax(sharp[0])))
        self.assertGreater(float(sharp[0, 0]), float(plain[0, 0]))

    def test_confidence_weights_are_automatic_and_penalize_uncertainty(self) -> None:
        policies = {item["id"]: item for item in fixed_candidate_policies(150)}
        margin = calibrate_probability_distribution(
            probability_rows(), policies["top3_t075_margin"]
        )
        entropy = calibrate_probability_distribution(
            probability_rows(), policies["top3_t075_entropy"]
        )
        self.assertAlmostEqual(float(margin[0].sum()), 0.35, places=6)
        self.assertAlmostEqual(float(margin[1].sum()), 0.0, places=6)
        self.assertGreater(float(entropy[0].sum()), float(entropy[2].sum()))
        self.assertAlmostEqual(float(entropy[2].sum()), 0.0, places=5)

    def test_candidate_selection_penalizes_abstention_and_breaks_ties_by_order(self) -> None:
        results = [
            {
                "fixed_candidate_order": 0,
                "accuracy_of_all_eligible": 0.7,
                "prediction_coverage_of_eligible": 1.0,
            },
            {
                "fixed_candidate_order": 1,
                "accuracy_of_all_eligible": 0.8,
                "prediction_coverage_of_eligible": 0.8,
            },
            {
                "fixed_candidate_order": 2,
                "accuracy_of_all_eligible": 0.8,
                "prediction_coverage_of_eligible": 0.9,
            },
            {
                "fixed_candidate_order": 3,
                "accuracy_of_all_eligible": 0.8,
                "prediction_coverage_of_eligible": 0.9,
            },
        ]
        self.assertEqual(select_candidate_result(results)["fixed_candidate_order"], 2)

    def test_policy_report_must_contain_complete_fixed_automatic_sweep(self) -> None:
        policies = fixed_candidate_policies(150)
        report = {
            "source": SOURCE,
            "contract": CONTRACT,
            "selection_rule": SELECTION_RULE,
            "candidate_policies": policies,
            "candidate_results": [{"policy": item} for item in policies],
            "selected_policy": policies[4],
            "manual_candidate_selection_used": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            selected, loaded = load_selected_policy(path, class_count=150)
            self.assertEqual(selected, policies[4])
            self.assertEqual(loaded["contract"], CONTRACT)
            report["manual_candidate_selection_used"] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "automatically"):
                load_selected_policy(path, class_count=150)

    def test_synthetic_leave_one_camera_out_policy_evaluation(self) -> None:
        policies = {item["id"]: item for item in fixed_candidate_policies(150)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for camera_index in range(3):
                indices = np.arange(4, dtype=np.uint32)
                values = np.zeros((4, 150), dtype=np.float16)
                values[np.arange(4), np.arange(4)] = 1.0
                index_path = root / f"{camera_index}_indices.npy"
                value_path = root / f"{camera_index}_probabilities.npy"
                np.save(index_path, indices, allow_pickle=False)
                np.save(value_path, values, allow_pickle=False)
                frames.append(
                    {
                        "file": f"{camera_index}.png",
                        "camera_index": camera_index,
                        "gaussian_indices_file": index_path.name,
                        "class_probabilities_file": value_path.name,
                    }
                )
            result = evaluate_policy(
                policy=policies["top1_t1_uniform"],
                frames=frames,
                vote_root=root,
                gaussian_count=4,
                class_count=150,
                base_camera_counts=np.full(4, 3, dtype=np.uint16),
                temporary_root=root,
                chunk_size=2,
            )
            self.assertEqual(result["eligible_gaussian_observation_count"], 12)
            self.assertEqual(result["predicted_gaussian_observation_count"], 12)
            self.assertEqual(result["correct_gaussian_observation_count"], 12)
            self.assertEqual(result["accuracy_of_all_eligible"], 1.0)

    def test_two_scene_gate_requires_same_policy_and_no_metric_regressions(self) -> None:
        policy = fixed_candidate_policies(150)[2]
        base = {
            "report_only": True,
            "camera_indices": [1, 2, 3],
            "camera_count": 3,
            "gaussian_count": 10,
            "heldout_pixel_metrics": {
                "agreement_of_projected": 0.7,
                "interior_agreement_of_projected": 0.72,
                "boundary_agreement_of_projected": 0.4,
            },
        }
        calibrated = {
            **base,
            "source": CALIBRATED_SOURCE,
            "contract": CALIBRATED_CONTRACT,
            "calibration_policy": policy,
            "heldout_pixel_metrics": {
                "agreement_of_projected": 0.71,
                "interior_agreement_of_projected": 0.73,
                "boundary_agreement_of_projected": 0.41,
            },
        }
        result = compare_scene_reports(
            scene_role="test", hard=base, calibrated=calibrated
        )
        self.assertTrue(result["all_metrics_non_regressing"])
        validation = {**result, "scene_role": "validation"}
        pair = compare_scene_pair(result, validation)
        self.assertTrue(pair["accepted"])
        self.assertEqual(pair["decision"], "accept_calibrated_policy")
        calibrated["heldout_pixel_metrics"]["boundary_agreement_of_projected"] = 0.39
        result = compare_scene_reports(
            scene_role="test", hard=base, calibrated=calibrated
        )
        self.assertFalse(result["all_metrics_non_regressing"])
        other_policy = dict(validation)
        other_policy["calibration_policy"] = fixed_candidate_policies(150)[3]
        with self.assertRaisesRegex(ValueError, "locked Playroom"):
            compare_scene_pair(validation, other_policy)

    def test_schedulers_are_fixed_report_only_and_support_locked_validation(self) -> None:
        calibrated = CALIBRATED_SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("POLICY_MODE must be select or validate", calibrated)
        self.assertIn("Manual calibration policies are not accepted", calibrated)
        self.assertIn("POLICY_SOURCE_OUTPUT_NAME", calibrated)
        self.assertIn("calibrated_probability_fusion", calibrated)
        self.assertIn("--calibration-policy-report", calibrated)
        self.assertIn("report_only=1", calibrated)
        self.assertIn("unexpectedly wrote a PLY", calibrated)
        self.assertNotIn("rm -rf", calibrated)
        soft = SOFT_SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("STOP_AFTER_SOFT_LIFT", soft)
        self.assertIn("automatic_dinov3_soft_probability_cache_only", soft)
        self.assertIn("test ! -e \"${PRIMARY_AUDIT_DIR}\"", soft)


if __name__ == "__main__":
    unittest.main()
