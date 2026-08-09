from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.round_trip_fidelity_audit import sha256_file
from scripts.task1.dinov3.compare_observed_black_component_graph_round_trip_scenes import (
    scene_gate,
    validate_report,
)
from scripts.task1.dinov3.observed_black_component_graph_round_trip_validation import (
    CONTRACT,
    DECISION_ELIGIBLE_COMPONENT,
    DECISION_ZERO_CAMERA,
    SOURCE,
    _validate_camera_counts,
    _validate_ontology_provenance,
    observed_component_candidate,
    per_gaussian_plurality_constraint,
    plurality_runner_up_candidate,
    update_per_class,
    _validate_reproduction,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_observed_black_component_graph_round_trip_validation_scene.sbatch"
)
MODULE = ROOT / "scripts" / "task1" / "dinov3" / (
    "observed_black_component_graph_round_trip_validation.py"
)
COMPARATOR = ROOT / "scripts" / "task1" / "dinov3" / (
    "compare_observed_black_component_graph_round_trip_scenes.py"
)


def vertex_arrays(count: int) -> np.ndarray:
    dtype = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    vertices = np.zeros((count,), dtype=dtype)
    for index in range(count):
        vertices["x"][index] = index * 0.02
    vertices["scale_0"] = -3.0
    vertices["scale_1"] = -3.0
    vertices["scale_2"] = -3.0
    vertices["rot_0"] = 1.0
    return vertices


def evidence(
    camera_index: int,
    indices: np.ndarray,
    *,
    gaussian_count: int,
    class_id: int = 15,
) -> dict:
    winners = np.zeros((gaussian_count,), dtype=np.uint16)
    mass = np.zeros((gaussian_count,), dtype=np.float32)
    selected = np.asarray(indices, dtype=np.int64)
    winners[selected] = np.uint16(class_id)
    mass[selected] = 1.0
    return {
        "camera_index": camera_index,
        "camera_id": camera_index,
        "file": f"camera_{camera_index}.png",
        "winners": winners,
        "mass": mass,
    }


def combined_total(evidence_items: list[dict], gaussian_count: int) -> np.ndarray:
    total = np.zeros((gaussian_count,), dtype=np.uint16)
    for item in evidence_items:
        total += (item["winners"] > 0).astype(np.uint16)
    return total


class HeldOutEvidenceTest(unittest.TestCase):
    def test_heldout_camera_removes_its_evidence_from_candidate(self) -> None:
        gaussian_count = 8
        vertices = vertex_arrays(gaussian_count)
        anchors = np.arange(4, dtype=np.int64)
        black = np.arange(4, 8, dtype=np.int64)
        baseline = [
            evidence(0, anchors, gaussian_count=gaussian_count),
            evidence(1, np.concatenate([anchors, [4, 5]]), gaussian_count=gaussian_count),
        ]
        additional = [
            evidence(2, np.concatenate([anchors, [5, 6, 7]]), gaussian_count=gaussian_count),
            evidence(3, np.concatenate([anchors, [6, 7]]), gaussian_count=gaussian_count),
        ]
        all_items = [*baseline, *additional]
        reliabilities = {item["camera_index"]: 0.95 for item in all_items}
        class_reliability = np.ones((16,), dtype=np.float32)
        anchor_points = np.column_stack(
            [vertices["x"][anchors], vertices["y"][anchors], vertices["z"][anchors]]
        )
        from scipy.spatial import cKDTree

        anchor_tree = cKDTree(anchor_points)
        anchor_labels = np.full((anchors.size,), 15, dtype=np.uint16)
        full = observed_component_candidate(
            vertices,
            black,
            combined_total(all_items, gaussian_count),
            all_items,
            reliabilities,
            class_reliability,
            anchor_tree,
            anchor_labels,
            edge_threshold=0.50,
            distance_scale=0.10,
            feature_scales={},
            component_score_threshold=0.62,
            class_count=15,
            workers=1,
        )
        # Gaussian 4 is observed only by baseline camera 1.
        self.assertEqual(int(full["combined_camera_count"][0]), 1)
        self.assertTrue(bool(full["observed_mask"][0]))

        remaining = [baseline[0], *additional]
        fold = observed_component_candidate(
            vertices,
            black,
            combined_total(remaining, gaussian_count),
            remaining,
            {item["camera_index"]: 0.95 for item in remaining},
            class_reliability,
            anchor_tree,
            anchor_labels,
            edge_threshold=0.50,
            distance_scale=0.10,
            feature_scales={},
            component_score_threshold=0.62,
            class_count=15,
            workers=1,
        )
        self.assertEqual(int(fold["combined_camera_count"][0]), 0)
        self.assertEqual(int(fold["decision_code"][0]), DECISION_ZERO_CAMERA)
        self.assertEqual(int(fold["candidate_project_id"][0]), 0)
        # Gaussian 5 loses camera 1 but remains observed by camera 2, and its
        # component still carries multicamera evidence from cameras 2 and 3.
        self.assertEqual(int(fold["combined_camera_count"][1]), 1)
        self.assertTrue(bool(fold["observed_mask"][1]))
        self.assertGreaterEqual(int(fold["combined_camera_count"][2]), 2)
        self.assertGreaterEqual(int(fold["combined_camera_count"][3]), 2)


class PluralityRunnerUpTest(unittest.TestCase):
    def test_accepts_plurality_winner_with_small_runner_up(self) -> None:
        # Classes 1..3 votes: 4, 1, 1 -> winner share 4/6, runner-up 1/6.
        raw = np.zeros((4, 1), dtype=np.uint16)
        raw[1, 0] = 4
        raw[2, 0] = 1
        raw[3, 0] = 1
        result = plurality_runner_up_candidate(
            raw, runner_up_cap=0.30, minimum_camera_count=2
        )
        self.assertTrue(bool(result["accepted"][0]))
        self.assertEqual(int(result["winner"][0]), 1)
        self.assertAlmostEqual(float(result["runner_share"][0]), 1.0 / 6.0)

    def test_rejects_winner_with_runner_up_above_cap(self) -> None:
        # Classes 1..3 votes: 4, 3, 1 -> runner-up share 3/8 = 0.375.
        raw = np.zeros((4, 1), dtype=np.uint16)
        raw[1, 0] = 4
        raw[2, 0] = 3
        raw[3, 0] = 1
        result = plurality_runner_up_candidate(
            raw, runner_up_cap=0.30, minimum_camera_count=2
        )
        self.assertFalse(bool(result["accepted"][0]))
        self.assertEqual(int(result["winner"][0]), 0)
        self.assertAlmostEqual(float(result["runner_share"][0]), 3.0 / 8.0)

    def test_rejects_tie_and_too_few_cameras(self) -> None:
        raw = np.zeros((3, 2), dtype=np.uint16)
        raw[1, 0] = 3
        raw[2, 0] = 3
        raw[1, 1] = 1
        result = plurality_runner_up_candidate(
            raw, runner_up_cap=0.50, minimum_camera_count=2
        )
        self.assertFalse(bool(result["accepted"][0]))
        self.assertFalse(bool(result["accepted"][1]))
        self.assertEqual(int(result["winner"][0]), 0)


class PerGaussianConstraintTest(unittest.TestCase):
    def test_plurality_winner_and_tie_abstention(self) -> None:
        cache = {
            "raw": np.asarray(
                [
                    [0, 0, 0, 0],
                    [3, 1, 2, 0],
                    [1, 2, 2, 0],
                    [0, 0, 1, 0],
                ],
                dtype=np.uint16,
            )
        }
        constraint = per_gaussian_plurality_constraint(cache)
        # Gaussian 0: class 1 unique winner; Gaussian 1: class 2 unique
        # winner; Gaussian 2: classes 2 and 3 tied -> abstain; Gaussian 3:
        # no votes -> abstain.
        np.testing.assert_array_equal(constraint, [1, 2, 0, 0])


class ReproductionContractTest(unittest.TestCase):
    def test_camera_counts_must_match_evidence(self) -> None:
        baseline = [{"camera_index": 0}, {"camera_index": 1}]
        additional = [{"camera_index": 2}]
        _validate_camera_counts(
            {"baseline_camera_count": 2, "additional_camera_count": 1},
            baseline,
            additional,
        )
        with self.assertRaisesRegex(ValueError, "baseline camera count"):
            _validate_camera_counts(
                {"baseline_camera_count": 3, "additional_camera_count": 1},
                baseline,
                additional,
            )
        with self.assertRaisesRegex(ValueError, "additional camera count"):
            _validate_camera_counts(
                {"baseline_camera_count": 2, "additional_camera_count": 2},
                baseline,
                additional,
            )

    def test_ontology_provenance_accepts_matching_hash_in_other_checkout(self) -> None:
        ontology = ROOT / "configs" / "ade20k_to_project.json"
        recorded = "/other/checkout/configs/ade20k_to_project.json"
        report = {
            "ontology": recorded,
            "input_sha256": {recorded: sha256_file(ontology)},
        }
        _validate_ontology_provenance(report, ontology)

    def test_ontology_provenance_rejects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "ade20k_to_project.json"
            changed.write_text('{"changed": true}', encoding="utf-8")
            recorded = "/other/checkout/configs/ade20k_to_project.json"
            report = {
                "ontology": recorded,
                "input_sha256": {
                    recorded: sha256_file(ROOT / "configs" / "ade20k_to_project.json")
                },
            }
            with self.assertRaisesRegex(ValueError, "different ontology"):
                _validate_ontology_provenance(report, changed)

    def test_reproduction_validation_accepts_matching_archive(self) -> None:
        black_count = 3
        values = {
            "component_id": np.asarray([-1, 0, 0], dtype=np.int32),
            "candidate_project_id": np.asarray([0, 15, 15], dtype=np.uint16),
            "decision_code": np.asarray(
                [DECISION_ZERO_CAMERA, DECISION_ELIGIBLE_COMPONENT, DECISION_ELIGIBLE_COMPONENT],
                dtype=np.uint8,
            ),
            "combined_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "component_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "observed_mask": np.asarray([False, True, True]),
            "component_score": np.asarray([0.0, 0.9, 0.9], dtype=np.float32),
            "component_size": np.asarray([0, 2, 2], dtype=np.uint32),
            "component_winner_share": np.asarray([0.0, 0.8, 0.8], dtype=np.float32),
            "component_winner_margin": np.asarray([0.0, 0.5, 0.5], dtype=np.float32),
            "component_normalized_entropy": np.asarray([0.0, 0.4, 0.4], dtype=np.float32),
            "component_internal_affinity": np.asarray([0.0, 0.7, 0.7], dtype=np.float32),
            "component_boundary_pressure": np.asarray([0.0, 0.2, 0.2], dtype=np.float32),
            "component_anchor_support": np.asarray([0.0, 0.1, 0.1], dtype=np.float32),
            "component_soft_class_reliability": np.asarray(
                [0.0, 0.9, 0.9], dtype=np.float32
            ),
            "eligible": np.asarray([False, True, True]),
        }
        diagnostics = {
            "component_id": np.asarray([-1, 0, 0], dtype=np.int32),
            "component_candidate_project_id": np.asarray([0, 15, 15], dtype=np.uint16),
            "decision_code": np.asarray(
                [DECISION_ZERO_CAMERA, DECISION_ELIGIBLE_COMPONENT, DECISION_ELIGIBLE_COMPONENT],
                dtype=np.uint8,
            ),
            "combined_semantic_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "component_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "observed_by_semantic_camera": np.asarray([False, True, True]),
            "component_score": np.asarray([0.0, 0.9, 0.9], dtype=np.float16),
            "component_size": np.asarray([0, 2, 2], dtype=np.uint32),
            "component_winner_share": np.asarray([0.0, 0.8, 0.8], dtype=np.float16),
            "component_winner_margin": np.asarray([0.0, 0.5, 0.5], dtype=np.float16),
            "component_normalized_entropy": np.asarray([0.0, 0.4, 0.4], dtype=np.float16),
            "component_internal_affinity": np.asarray([0.0, 0.7, 0.7], dtype=np.float16),
            "component_boundary_pressure": np.asarray([0.0, 0.2, 0.2], dtype=np.float16),
            "component_anchor_support": np.asarray([0.0, 0.1, 0.1], dtype=np.float16),
            "component_soft_class_reliability": np.asarray(
                [0.0, 0.9, 0.9], dtype=np.float16
            ),
        }
        report = {
            "eligible_report_only_gaussian_count": 2,
            "component_count": 1,
            "decision_counts": {
                "remain_black_zero_combined_semantic_cameras": 1,
                "remain_black_component_has_too_little_camera_evidence": 0,
                "remain_black_component_raw_and_weighted_winners_disagree": 0,
                "remain_black_component_boundary_is_ambiguous": 0,
                "remain_black_component_score_below_calibrated_threshold": 0,
                "eligible_report_only_component_graph_candidate": 2,
            },
        }
        _validate_reproduction(values, diagnostics, report)

    def test_reproduction_validation_rejects_changed_decision(self) -> None:
        values = {
            "component_id": np.asarray([-1, 0, 0], dtype=np.int32),
            "candidate_project_id": np.asarray([0, 15, 15], dtype=np.uint16),
            "decision_code": np.asarray([0, 4, 4], dtype=np.uint8),
            "combined_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "component_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "observed_mask": np.asarray([False, True, True]),
            "component_score": np.asarray([0.0, 0.9, 0.9], dtype=np.float32),
            "component_size": np.asarray([0, 2, 2], dtype=np.uint32),
            "component_winner_share": np.asarray([0.0, 0.8, 0.8], dtype=np.float32),
            "component_winner_margin": np.asarray([0.0, 0.5, 0.5], dtype=np.float32),
            "component_normalized_entropy": np.asarray([0.0, 0.4, 0.4], dtype=np.float32),
            "component_internal_affinity": np.asarray([0.0, 0.7, 0.7], dtype=np.float32),
            "component_boundary_pressure": np.asarray([0.0, 0.2, 0.2], dtype=np.float32),
            "component_anchor_support": np.asarray([0.0, 0.1, 0.1], dtype=np.float32),
            "component_soft_class_reliability": np.asarray(
                [0.0, 0.9, 0.9], dtype=np.float32
            ),
            "eligible": np.asarray([False, True, True]),
        }
        diagnostics = {
            "component_id": np.asarray([-1, 0, 0], dtype=np.int32),
            "component_candidate_project_id": np.asarray([0, 15, 15], dtype=np.uint16),
            "decision_code": np.asarray([0, 5, 5], dtype=np.uint8),
            "combined_semantic_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "component_camera_count": np.asarray([0, 2, 2], dtype=np.uint16),
            "observed_by_semantic_camera": np.asarray([False, True, True]),
            "component_score": np.asarray([0.0, 0.9, 0.9], dtype=np.float16),
            "component_size": np.asarray([0, 2, 2], dtype=np.uint32),
            "component_winner_share": np.asarray([0.0, 0.8, 0.8], dtype=np.float16),
            "component_winner_margin": np.asarray([0.0, 0.5, 0.5], dtype=np.float16),
            "component_normalized_entropy": np.asarray([0.0, 0.4, 0.4], dtype=np.float16),
            "component_internal_affinity": np.asarray([0.0, 0.7, 0.7], dtype=np.float16),
            "component_boundary_pressure": np.asarray([0.0, 0.2, 0.2], dtype=np.float16),
            "component_anchor_support": np.asarray([0.0, 0.1, 0.1], dtype=np.float16),
            "component_soft_class_reliability": np.asarray(
                [0.0, 0.9, 0.9], dtype=np.float16
            ),
        }
        report = {
            "eligible_report_only_gaussian_count": 2,
            "component_count": 1,
            "decision_counts": {
                "remain_black_zero_combined_semantic_cameras": 1,
                "remain_black_component_has_too_little_camera_evidence": 0,
                "remain_black_component_raw_and_weighted_winners_disagree": 0,
                "remain_black_component_boundary_is_ambiguous": 0,
                "remain_black_component_score_below_calibrated_threshold": 0,
                "eligible_report_only_component_graph_candidate": 2,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "decision_code"):
            _validate_reproduction(values, diagnostics, report)


class PixelMetricTest(unittest.TestCase):
    def test_per_class_recovery_counts_from_shared_partition(self) -> None:
        values = {}
        update_per_class(
            values,
            np.asarray([[1, 2], [2, 0]], dtype=np.uint16),
            np.asarray([[1, 1], [1, 0]], dtype=bool),
            np.asarray([[1, 1], [2, 2]], dtype=np.uint16),
            np.asarray([[0, 2], [2, 0]], dtype=np.uint16),
        )
        self.assertEqual(values[1]["source_pixels"], 2)
        self.assertEqual(values[1]["candidate_agreed"], 1)
        self.assertEqual(values[1]["baseline_agreed"], 0)
        self.assertEqual(values[1]["newly_recovered_pixels"], 1)
        self.assertEqual(values[2]["source_pixels"], 1)
        self.assertEqual(values[2]["candidate_agreed"], 1)
        self.assertEqual(values[2]["newly_recovered_pixels"], 0)


class TwoSceneGateTest(unittest.TestCase):
    @staticmethod
    def report() -> dict:
        return {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "baseline_metrics": {
                "projected_ratio": 0.90,
                "agreement_of_projected": 0.80,
                "interior_agreement_of_projected": 0.82,
                "boundary_agreement_of_projected": 0.50,
            },
            "candidate_metrics": {
                "projected_ratio": 0.91,
                "agreement_of_projected": 0.81,
                "interior_agreement_of_projected": 0.83,
                "boundary_agreement_of_projected": 0.51,
            },
            "delta": {},
            "per_class_recovery": [{"project_id": 15}],
            "candidate_recovered_count": 10,
            "full_evidence_eligible_gaussian_count": 12,
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

    def test_accepts_recovery_with_no_metric_regression(self) -> None:
        report = self.report()
        validate_report(report, "playroom")
        self.assertTrue(scene_gate(report)["passes"])

    def test_rejects_boundary_regression(self) -> None:
        report = self.report()
        report["candidate_metrics"]["boundary_agreement_of_projected"] = 0.49
        gate = scene_gate(report)
        self.assertFalse(gate["passes"])
        self.assertFalse(gate["non_regression"]["boundary_agreement_of_projected"])

    def test_rejects_missing_per_class_recovery(self) -> None:
        report = self.report()
        report["per_class_recovery"] = []
        self.assertFalse(scene_gate(report)["passes"])


class SchedulerContractTest(unittest.TestCase):
    def test_scheduler_is_automatic_report_only_and_uses_existing_caches(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "DINOV2_VOTE_MANIFEST",
            "--dinov2-vote-manifest",
            "Manual camera selection is not accepted",
            "exclude_each_original_baseline_camera_from_baseline_and_component_graph_evidence",
            "recompute_all_camera_reliabilities_and_rebuild_the_component_graph_after_excluding_the_heldout_camera",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
            "SOURCE_CACHE_OUTPUT_NAME",
            "SOURCE_HARD_AUDIT_OUTPUT_NAME",
            "SOURCE_RECOVERY_OUTPUT_NAME",
            "SOURCE_COMPONENT_AUDIT_OUTPUT_NAME",
            "--component-audit-report",
            "--component-diagnostics",
            "--hard-confusion",
            "candidate_heldout_overlays.png",
            '--output-dir "${AUDIT_DIR}"',
        ):
            self.assertIn(expected, source)
        disallowed = "v" + "5"
        for path in (MODULE, COMPARATOR, SCHEDULER):
            self.assertNotIn(disallowed, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
