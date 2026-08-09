from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.compare_observed_black_calibrated_spatial_scenes import (
    validate_report,
)
from scripts.task1.dinov3.observed_black_calibrated_spatial_audit import (
    CONTRACT,
    DECISION_ELIGIBLE_SPATIAL_CORROBORATION,
    DECISION_ELIGIBLE_STRONG_SEMANTIC,
    DECISION_SEMANTIC_TOO_WEAK,
    DECISION_SINGLE_CAMERA,
    DECISION_SPATIAL_CONFLICT,
    DECISION_ZERO_CAMERA,
    POLICY,
    SOURCE,
    build_calibration_tables,
    calibrated_lower_bounds,
    class_reliability_rows,
    component_support,
    decide_candidates,
    evidence_bin,
    main as audit_main,
    neighbor_metrics,
    score_features,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CONTRACT as HARD_AUDIT_CONTRACT,
    SOURCE as HARD_AUDIT_SOURCE,
    STATUS_ACCEPTED,
    STATUS_UNOBSERVED,
    VOTE_CONTRACT,
    VOTE_SOURCE,
)


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = load_ontology(ROOT / "configs" / "ade20k_to_project.json")
SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_observed_black_calibrated_spatial_audit_scene.sbatch"
)
MODULE = ROOT / "scripts" / "task1" / "dinov3" / (
    "observed_black_calibrated_spatial_audit.py"
)
COMPARATOR = ROOT / "scripts" / "task1" / "dinov3" / (
    "compare_observed_black_calibrated_spatial_scenes.py"
)


class SemanticFeatureTest(unittest.TestCase):
    def test_evidence_bins_keep_zero_and_one_outside_calibration(self) -> None:
        np.testing.assert_array_equal(
            evidence_bin(np.asarray([0, 1, 2, 3, 4, 5, 9, 10, 44])),
            [-1, -1, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_weighted_features_require_unique_strict_majority(self) -> None:
        result = score_features(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.7, 0.5, 0.4],
                    [0.3, 0.5, 0.35],
                    [0.0, 0.0, 0.25],
                ],
                dtype=np.float32,
            )
        )
        np.testing.assert_array_equal(result["prediction"], [1, 0, 0])
        self.assertGreater(float(result["winner_margin"][0]), 0.0)
        self.assertLess(float(result["normalized_entropy"][0]), 1.0)


class HeldoutClassReliabilityTest(unittest.TestCase):
    def test_catastrophic_source_confusion_marks_wall_and_door_targets(self) -> None:
        counts = np.zeros((ONTOLOGY.class_count + 1, ONTOLOGY.class_count + 1), dtype=np.uint64)
        window_id = 9
        wall_id = 1
        door_id = 15
        counts[window_id, window_id] = 1
        counts[window_id, wall_id] = 700
        counts[window_id, door_id] = 100
        counts[wall_id, wall_id] = 1000
        counts[door_id, door_id] = 1000
        rows = {row["project_id"]: row for row in class_reliability_rows(counts, ONTOLOGY)}
        self.assertEqual(rows[wall_id]["largest_catastrophic_incoming_source_project_id"], window_id)
        self.assertGreater(rows[wall_id]["largest_catastrophic_incoming_ratio"], 0.8)
        self.assertEqual(rows[door_id]["largest_catastrophic_incoming_source_project_id"], window_id)
        self.assertGreater(rows[door_id]["largest_catastrophic_incoming_ratio"], 0.1)
        self.assertLess(rows[window_id]["heldout_recall_lower_bound"], 0.01)


class CalibrationTest(unittest.TestCase):
    def test_candidate_requires_class_evidence_and_score_region_calibration(self) -> None:
        count = 160
        predictions = np.full((count,), 15, dtype=np.uint16)
        truth = predictions.copy()
        truth[-8:] = 1
        camera_count = np.full((count,), 3, dtype=np.uint16)
        share = np.full((count,), 0.75, dtype=np.float32)
        margin = np.full((count,), 0.25, dtype=np.float32)
        interior = np.ones((count,), dtype=bool)
        tables = build_calibration_tables(
            predictions,
            truth,
            camera_count,
            share,
            margin,
            interior,
            ONTOLOGY,
        )
        with patch.dict(
            POLICY,
            {
                "minimum_class_calibration_trials": 1,
                "minimum_class_evidence_calibration_trials": 1,
                "minimum_score_region_calibration_trials": 1,
            },
        ):
            lower, calibrated = calibrated_lower_bounds(
                np.asarray([15, 15], dtype=np.uint16),
                np.asarray([3, 10], dtype=np.uint16),
                np.asarray([0.75, 0.75], dtype=np.float32),
                np.asarray([0.25, 0.25], dtype=np.float32),
                np.asarray([True, True]),
                tables,
            )
        self.assertTrue(bool(calibrated[0]))
        self.assertGreater(float(lower[0]), 0.8)
        self.assertFalse(bool(calibrated[1]))


class SpatialEvidenceTest(unittest.TestCase):
    def test_neighbor_metrics_count_only_target_class_within_radius(self) -> None:
        distances = np.asarray([[0.1, 0.2, 0.3, 2.0]], dtype=np.float64)
        indices = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        metrics = neighbor_metrics(
            distances,
            indices,
            np.asarray([15, 15, 1, 15], dtype=np.uint16),
            np.asarray([15], dtype=np.uint16),
            maximum_distance=0.5,
        )
        self.assertEqual(int(metrics["valid_neighbor_count"][0]), 3)
        self.assertEqual(int(metrics["same_class_neighbor_count"][0]), 2)
        self.assertEqual(int(metrics["competing_class_neighbor_count"][0]), 1)

    def test_components_do_not_mix_semantic_classes(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float64,
        )
        project_ids = np.asarray([15, 15, 15, 1], dtype=np.uint16)
        sizes, anchored = component_support(
            points,
            project_ids,
            np.asarray([True, False, False, True]),
            voxel_size=0.25,
        )
        np.testing.assert_array_equal(sizes, [3, 3, 3, 1])
        np.testing.assert_array_equal(anchored, [1, 1, 1, 1])


class DecisionPolicyTest(unittest.TestCase):
    @staticmethod
    def reliable_class_rows() -> list[dict]:
        return [
            {
                "project_id": item.project_id,
                "heldout_recall_lower_bound": 0.99,
                "heldout_precision_lower_bound": 0.99,
                "largest_catastrophic_incoming_ratio": 0.0,
            }
            for item in ONTOLOGY.classes
        ]

    def test_object_spatial_support_corroborates_but_surface_support_does_not_relax(self) -> None:
        project_ids = np.asarray([0, 0, 15, 1, 1, 15], dtype=np.uint16)
        camera_count = np.asarray([0, 1, 3, 3, 5, 5], dtype=np.uint16)
        raw_winners = project_ids.copy()
        weighted_accepted = np.asarray([False, False, True, True, True, True])
        share = np.asarray([0.0, 1.0, 0.70, 0.70, 0.90, 0.90], dtype=np.float32)
        margin = np.asarray([0.0, 1.0, 0.20, 0.20, 0.60, 0.60], dtype=np.float32)
        entropy = np.asarray([0.0, 0.0, 0.50, 0.50, 0.10, 0.10], dtype=np.float32)
        calibrated = np.ones((6,), dtype=bool)
        lower = np.asarray([0.0, 0.0, 0.92, 0.99, 0.99, 0.99], dtype=np.float32)
        spatial = {
            "valid_neighbor_count": np.asarray([0, 0, 8, 8, 8, 8], dtype=np.uint8),
            "same_class_neighbor_count": np.asarray([0, 0, 6, 8, 8, 0], dtype=np.uint8),
            "same_class_fraction": np.asarray([0.0, 0.0, 0.75, 1.0, 1.0, 0.0], dtype=np.float32),
            "nearest_same_class_distance": np.asarray([np.inf, np.inf, 0.1, 0.1, 0.1, 1.0]),
            "nearest_competing_class_distance": np.asarray([np.inf, np.inf, 0.2, 0.2, 0.2, 0.1]),
        }
        decisions = decide_candidates(
            project_ids,
            camera_count,
            raw_winners,
            weighted_accepted,
            share,
            margin,
            entropy,
            calibrated,
            lower,
            self.reliable_class_rows(),
            ONTOLOGY,
            spatial,
            np.asarray([0, 0, 5, 5, 5, 5], dtype=np.uint32),
            np.asarray([0, 0, 4, 4, 4, 0], dtype=np.uint32),
        )
        self.assertEqual(int(decisions[0]), DECISION_ZERO_CAMERA)
        self.assertEqual(int(decisions[1]), DECISION_SINGLE_CAMERA)
        self.assertEqual(int(decisions[2]), DECISION_ELIGIBLE_SPATIAL_CORROBORATION)
        self.assertEqual(int(decisions[3]), DECISION_SEMANTIC_TOO_WEAK)
        self.assertEqual(int(decisions[4]), DECISION_ELIGIBLE_STRONG_SEMANTIC)
        self.assertEqual(int(decisions[5]), DECISION_SPATIAL_CONFLICT)


class ContractTest(unittest.TestCase):
    def test_two_scene_validator_requires_the_identical_report_only_policy(self) -> None:
        report = {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "policy": POLICY,
            "zero_camera_gaussians_forced_black": True,
            "single_camera_gaussians_forced_black": True,
            "spatial_evidence_can_choose_semantic_class": False,
            "spatial_evidence_can_relax_stuff_semantic_thresholds": False,
            "semantic_candidate_must_precede_spatial_corroboration": True,
            "dinov3_inference_rerun": False,
            "flashsplat_lifting_rerun": False,
            "manual_camera_selection_used": False,
            "manual_gaussian_selection_used": False,
            "manual_class_selection_used": False,
            "scene_specific_rules": False,
            "accepted_gaussian_labels_written": False,
            "gaussian_project_class_array_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
            "current_black_gaussian_count": 10,
            "current_black_zero_combined_camera_count": 3,
            "current_black_single_combined_camera_count": 2,
            "current_black_multicamera_count": 5,
            "eligible_report_only_gaussian_count": 1,
            "decision_counts": {},
            "heldout_class_reliability": [],
        }
        validate_report(report, "playroom")
        report["spatial_evidence_can_choose_semantic_class"] = True
        with self.assertRaisesRegex(ValueError, "spatial_evidence_can_choose_semantic_class"):
            validate_report(report, "playroom")

    def test_scheduler_is_cache_only_and_new_files_exclude_disallowed_method(self) -> None:
        scheduler = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "cache_only=1",
            "dinov3_inference_rerun=0",
            "flashsplat_lifting_rerun=0",
            "zero_camera_gaussians_forced_black=1",
            "single_camera_gaussians_forced_black=1",
            "semantic_class_selected_before_spatial_corroboration=1",
            "broad_surface_spatial_relaxation=0",
            "--hard-confusion",
            "--recovery-candidate-labels",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
        ):
            self.assertIn(expected, scheduler)
        disallowed = "v" + "5"
        for path in (SCHEDULER, MODULE, COMPARATOR):
            self.assertNotIn(disallowed, path.read_text(encoding="utf-8").lower())


class CacheOnlyIntegrationTest(unittest.TestCase):
    @staticmethod
    def write_ply(path: Path, points: np.ndarray) -> None:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        vertices = np.zeros(
            (points.shape[0],),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
        )
        for axis_index, axis in enumerate(("x", "y", "z")):
            vertices[axis] = points[:, axis_index]
        with path.open("wb") as stream:
            stream.write(header)
            vertices.tofile(stream)

    @staticmethod
    def write_vote_manifest(
        root: Path,
        camera_indices: list[int],
        supported_indices: np.ndarray,
        *,
        gaussian_count: int,
        ply_path: Path,
    ) -> Path:
        root.mkdir()
        frames = []
        for camera_index in camera_indices:
            vote_file = f"camera_{camera_index:04d}.npz"
            np.savez_compressed(
                root / vote_file,
                indices=supported_indices.astype(np.uint32),
                class_ids=np.full(supported_indices.shape, 15, dtype=np.uint16),
                weights=np.ones(supported_indices.shape, dtype=np.float32),
            )
            frames.append(
                {
                    "camera_index": camera_index,
                    "camera_id": camera_index,
                    "file": f"camera_{camera_index:04d}.png",
                    "vote_file": vote_file,
                }
            )
        path = root / "vote_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "source": VOTE_SOURCE,
                    "contract": VOTE_CONTRACT,
                    "query_region_filtering_used": False,
                    "confidence_threshold_used": False,
                    "one_normalized_vote_per_camera": True,
                    "gaussian_count": gaussian_count,
                    "ply_path": str(ply_path),
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_main_reuses_votes_and_writes_diagnostics_without_semantic_output(self) -> None:
        anchor_count = 160
        black_count = 10
        gaussian_count = anchor_count + black_count
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply_path = root / "point_cloud.ply"
            points = np.column_stack(
                [
                    np.arange(gaussian_count, dtype=np.float32) * np.float32(0.01),
                    np.zeros((gaussian_count,), dtype=np.float32),
                    np.zeros((gaussian_count,), dtype=np.float32),
                ]
            )
            self.write_ply(ply_path, points)
            anchor_indices = np.arange(anchor_count, dtype=np.uint32)
            all_indices = np.arange(gaussian_count, dtype=np.uint32)
            baseline_cameras = [0, 1, 2]
            additional_cameras = [3, 4, 5, 6, 7]
            baseline_manifest = self.write_vote_manifest(
                root / "baseline_votes",
                baseline_cameras,
                anchor_indices,
                gaussian_count=gaussian_count,
                ply_path=ply_path,
            )
            additional_manifest = self.write_vote_manifest(
                root / "additional_votes",
                additional_cameras,
                all_indices,
                gaussian_count=gaussian_count,
                ply_path=ply_path,
            )

            hard_report = root / "hard_report.json"
            hard_report.write_text(
                json.dumps(
                    {
                        "source": HARD_AUDIT_SOURCE,
                        "contract": HARD_AUDIT_CONTRACT,
                        "vote_manifest": str(baseline_manifest),
                        "camera_indices": baseline_cameras,
                        "gaussian_count": gaussian_count,
                    }
                ),
                encoding="utf-8",
            )
            hard_status = np.full((gaussian_count,), STATUS_UNOBSERVED, dtype=np.uint8)
            hard_status[:anchor_count] = STATUS_ACCEPTED
            hard_diagnostics = root / "hard_diagnostics.npz"
            np.savez_compressed(
                hard_diagnostics,
                semantic_camera_count=np.concatenate(
                    [
                        np.full((anchor_count,), 3, dtype=np.uint16),
                        np.zeros((black_count,), dtype=np.uint16),
                    ]
                ),
                winner_camera_count=np.concatenate(
                    [
                        np.full((anchor_count,), 3, dtype=np.uint8),
                        np.zeros((black_count,), dtype=np.uint8),
                    ]
                ),
                consensus_status=hard_status,
            )
            confusion = np.zeros(
                (ONTOLOGY.class_count + 1, ONTOLOGY.class_count + 1),
                dtype=np.uint64,
            )
            confusion[15, 15] = 10_000
            hard_confusion = root / "hard_confusion.npz"
            np.savez_compressed(hard_confusion, counts=confusion)

            recovery_report = root / "recovery_report.json"
            recovery_report.write_text(
                json.dumps(
                    {
                        "source": "dinov3_detected_abstention_recovery_audit",
                        "contract": "immutable_hard_anchor_incremental_strict_then_calibrated_v1",
                        "scene": "playroom",
                        "report_only": True,
                        "gaussian_count": gaussian_count,
                        "baseline_camera_indices": baseline_cameras,
                        "additional_camera_indices": additional_cameras,
                    }
                ),
                encoding="utf-8",
            )
            candidate_labels = root / "candidate.npy"
            current = np.zeros((gaussian_count,), dtype=np.uint16)
            current[:anchor_count] = 15
            np.save(candidate_labels, current)
            recovery_codes = root / "source_codes.npy"
            codes = np.zeros((gaussian_count,), dtype=np.uint8)
            codes[:anchor_count] = 1
            np.save(recovery_codes, codes)
            output_dir = root / "audit"

            argv = [
                "observed_black_calibrated_spatial_audit",
                "--scene",
                "playroom",
                "--baseline-vote-manifest",
                str(baseline_manifest),
                "--additional-vote-manifest",
                str(additional_manifest),
                "--hard-audit-report",
                str(hard_report),
                "--hard-diagnostics",
                str(hard_diagnostics),
                "--hard-confusion",
                str(hard_confusion),
                "--recovery-report",
                str(recovery_report),
                "--recovery-candidate-labels",
                str(candidate_labels),
                "--recovery-source-codes",
                str(recovery_codes),
                "--source-ply",
                str(ply_path),
                "--ontology",
                str(ROOT / "configs" / "ade20k_to_project.json"),
                "--output-dir",
                str(output_dir),
                "--query-workers",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                audit_main()

            report = json.loads(
                (output_dir / "observed_black_calibrated_spatial_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["current_black_gaussian_count"], black_count)
            self.assertGreater(report["eligible_report_only_gaussian_count"], 0)
            self.assertLessEqual(report["eligible_report_only_gaussian_count"], black_count)
            self.assertTrue((output_dir / "observed_black_diagnostics.npz").is_file())
            self.assertFalse((output_dir / "gaussian_labels.npy").exists())
            self.assertFalse((output_dir / "label_map.json").exists())
            self.assertFalse(any(output_dir.rglob("*.ply")))


if __name__ == "__main__":
    unittest.main()
