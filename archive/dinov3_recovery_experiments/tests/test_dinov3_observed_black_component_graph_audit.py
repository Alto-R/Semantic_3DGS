from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.compare_observed_black_component_graph_scenes import (
    validate_report,
)
from scripts.task1.dinov3.observed_black_component_graph_audit import (
    CONTRACT,
    DECISION_ELIGIBLE_COMPONENT,
    DECISION_ZERO_CAMERA,
    POLICY,
    SOURCE,
    aggregate_component_votes,
    build_components,
    calibrate_edge_threshold,
    component_anchor_support,
    main as audit_main,
    score_features,
    soft_class_reliability,
)
from scripts.task1.dinov3.recover_detected_abstentions import (
    CONTRACT as RECOVERY_CONTRACT,
    SOURCE as RECOVERY_SOURCE,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CONTRACT as HARD_AUDIT_CONTRACT,
    SOURCE as HARD_AUDIT_SOURCE,
    STATUS_ACCEPTED,
    STATUS_UNOBSERVED,
    VOTE_CONTRACT,
    VOTE_SOURCE,
)
from scripts.task1.dinov3.dinov2_second_source import SOURCE as DINOV2_SOURCE


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = load_ontology(ROOT / "configs" / "ade20k_to_project.json")
MODULE = ROOT / "scripts" / "task1" / "dinov3" / "observed_black_component_graph_audit.py"
COMPARATOR = ROOT / "scripts" / "task1" / "dinov3" / "compare_observed_black_component_graph_scenes.py"
SCHEDULER = ROOT / "scripts" / "slurm" / "slurm_task1_dinov3_observed_black_component_graph_audit_scene.sbatch"


class GraphGroupingTest(unittest.TestCase):
    @staticmethod
    def graph(probabilities: np.ndarray) -> np.ndarray:
        points = np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.00, 0.00],
                [0.04, 0.00, 0.00],
                [0.00, 0.08, 0.00],
                [0.02, 0.08, 0.00],
                [0.04, 0.08, 0.00],
            ],
            dtype=np.float64,
        )
        appearance = np.asarray(
            [[0.0, 0.0, 0.0]] * 3 + [[3.0, 3.0, 3.0]] * 3,
            dtype=np.float32,
        )
        result = build_components(
            {
                "points": points,
                "appearance": appearance,
                "log_scale": None,
                "normal": None,
            },
            probabilities,
            np.ones((4, points.shape[0]), dtype=bool),
            edge_threshold=0.70,
            distance_scale=0.10,
            feature_scales={"appearance": 1.0},
            neighbor_count=5,
            workers=1,
        )
        return result["component_ids"]

    def test_class_id_permutation_does_not_change_grouping(self) -> None:
        probabilities = np.zeros((3, 6), dtype=np.float32)
        probabilities[0, :3] = 1.0
        probabilities[1, 3:] = 1.0
        original = self.graph(probabilities)
        permuted = self.graph(probabilities[[2, 0, 1]])
        np.testing.assert_array_equal(original, permuted)
        self.assertEqual(len(np.unique(original)), 2)

    def test_large_surface_does_not_absorb_touching_object(self) -> None:
        wall = np.column_stack(
            [
                np.linspace(0.0, 0.30, 24),
                np.zeros((24,)),
                np.zeros((24,)),
            ]
        )
        chair = np.asarray(
            [[0.13, 0.025, 0.0], [0.15, 0.025, 0.0], [0.17, 0.025, 0.0]]
        )
        points = np.vstack([wall, chair])
        probabilities = np.zeros((3, points.shape[0]), dtype=np.float32)
        probabilities[0, : wall.shape[0]] = 1.0
        probabilities[1, wall.shape[0] :] = 1.0
        appearance = np.vstack(
            [
                np.zeros((wall.shape[0], 3), dtype=np.float32),
                np.full((chair.shape[0], 3), 4.0, dtype=np.float32),
            ]
        )
        result = build_components(
            {
                "points": points,
                "appearance": appearance,
                "log_scale": None,
                "normal": None,
            },
            probabilities,
            np.ones((3, points.shape[0]), dtype=bool),
            edge_threshold=0.72,
            distance_scale=0.05,
            feature_scales={"appearance": 1.0},
            neighbor_count=8,
            workers=1,
        )
        component_ids = result["component_ids"]
        wall_components = set(component_ids[: wall.shape[0]].tolist())
        chair_components = set(component_ids[wall.shape[0] :].tolist())
        self.assertTrue(wall_components.isdisjoint(chair_components))

    def test_class_constraint_splits_touching_different_classes(self) -> None:
        points = np.asarray(
            [[0.00, 0.00, 0.00], [0.02, 0.00, 0.00]],
            dtype=np.float64,
        )
        probabilities = np.zeros((3, 2), dtype=np.float32)
        probabilities[0, :] = 1.0
        appearance = np.zeros((2, 3), dtype=np.float32)
        common = dict(
            features={
                "points": points,
                "appearance": appearance,
                "log_scale": None,
                "normal": None,
            },
            semantic_probabilities=probabilities,
            visibility=np.ones((2, 2), dtype=bool),
            edge_threshold=0.70,
            distance_scale=0.10,
            feature_scales={"appearance": 1.0},
            neighbor_count=5,
            workers=1,
        )
        same = build_components(
            class_constraint=np.asarray([15, 15], dtype=np.int64), **common
        )["component_ids"]
        different = build_components(
            class_constraint=np.asarray([15, 20], dtype=np.int64), **common
        )["component_ids"]
        self.assertEqual(int(same[0]), int(same[1]))
        self.assertNotEqual(int(different[0]), int(different[1]))

    def test_single_view_members_gain_component_multiview_evidence(self) -> None:
        component_ids = np.zeros((3,), dtype=np.int32)
        cache = {
            "winners": np.asarray([[8, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=np.uint16),
            "masses": np.ones((3, 3), dtype=np.float32),
        }
        evidence = [{"camera_index": index} for index in range(3)]
        votes = aggregate_component_votes(
            component_ids,
            cache,
            evidence,
            {0: 1.0, 1: 1.0, 2: 1.0},
            class_count=ONTOLOGY.class_count,
        )
        features = score_features(votes["weighted"])
        self.assertEqual(int(votes["camera_count"][0]), 3)
        self.assertTrue(bool(features["accepted"][0]))
        self.assertEqual(int(features["winner"][0]), 8)

    def test_anchor_support_is_normalized_by_component_size(self) -> None:
        anchors = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0]]
        )
        labels = np.ones((4,), dtype=np.uint16)
        large = np.column_stack(
            [np.zeros((10,)), np.linspace(0.0, 1.0, 10), np.zeros((10,))]
        )
        small = np.asarray([[10.0, 0.2, 0.0], [10.0, 0.8, 0.0]])
        points = np.vstack([large, small])
        components = np.asarray([0] * 10 + [1] * 2, dtype=np.int32)
        support = component_anchor_support(
            points,
            components,
            __import__("scipy").spatial.cKDTree(anchors),
            labels,
            neighbor_count=2,
            maximum_distance=2.0,
            workers=1,
            class_count=ONTOLOGY.class_count,
        )
        self.assertAlmostEqual(float(support[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(support[1, 1]), 1.0, places=5)


class CalibrationTest(unittest.TestCase):
    def test_edge_calibration_uses_one_common_threshold(self) -> None:
        affinity = np.asarray([0.99] * 100 + [0.20] * 20, dtype=np.float32)
        same = np.asarray([True] * 100 + [False] * 20)
        threshold, report = calibrate_edge_threshold(affinity, same)
        self.assertGreater(threshold, 0.20)
        self.assertFalse(report["class_specific_thresholds_used"])

    def test_bad_class_metrics_are_a_soft_penalty_not_a_veto(self) -> None:
        rows = [
            {
                "project_id": 9,
                "heldout_recall_lower_bound": 0.0001,
                "heldout_precision_lower_bound": 0.0001,
            }
        ]
        reliability = soft_class_reliability(rows, ONTOLOGY.class_count)
        self.assertEqual(
            float(reliability[9]),
            float(POLICY["soft_class_reliability_floor"]),
        )
        self.assertGreater(float(reliability[9]), 0.0)


class ContractTest(unittest.TestCase):
    def test_comparator_requires_generalized_component_contract(self) -> None:
        report = {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": "playroom",
            "report_only": True,
            "policy": POLICY,
            "component_graph_is_class_agnostic": True,
            "component_graph_uses_mutual_knn": True,
            "component_labels_selected_after_grouping": True,
            "candidate_labels_limited_to_component_camera_evidence": True,
            "class_reliability_is_soft_weighting": True,
            "global_class_veto_used": False,
            "anchor_support_is_component_normalized": True,
            "broad_surface_population_normalized": True,
            "zero_camera_gaussians_forced_black": True,
            "single_camera_gaussians_forced_black": False,
            "spatial_evidence_can_choose_semantic_class": False,
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
            "current_black_zero_combined_camera_count": 2,
            "current_black_single_combined_camera_count": 2,
            "current_black_multicamera_count": 6,
            "camera_observed_black_gaussian_count": 8,
            "component_count": 3,
            "eligible_report_only_gaussian_count": 5,
            "graph_calibration": {},
            "component_calibration": {},
        }
        validate_report(report, "playroom")
        report["global_class_veto_used"] = True
        with self.assertRaisesRegex(ValueError, "global_class_veto_used"):
            validate_report(report, "playroom")

    def test_scheduler_is_cache_only_and_has_no_class_specific_rules(self) -> None:
        scheduler = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "DINOV2_VOTE_MANIFEST",
            "--dinov2-vote-manifest",
            "cache_only=1",
            "class_agnostic_mutual_knn_component_graph=1",
            "component_labels_selected_after_grouping=1",
            "single_camera_gaussians_forced_black=0",
            "global_class_veto_used=0",
            "broad_surface_population_normalized=1",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
        ):
            self.assertIn(expected, scheduler)
        disallowed = "v" + "5"
        for path in (MODULE, COMPARATOR, SCHEDULER):
            self.assertNotIn(disallowed, path.read_text(encoding="utf-8").lower())


class CacheOnlyIntegrationTest(unittest.TestCase):
    @staticmethod
    def write_ply(path: Path, points: np.ndarray) -> None:
        properties = (
            ("x", "float"), ("y", "float"), ("z", "float"),
            ("f_dc_0", "float"), ("f_dc_1", "float"), ("f_dc_2", "float"),
            ("scale_0", "float"), ("scale_1", "float"), ("scale_2", "float"),
            ("rot_0", "float"), ("rot_1", "float"), ("rot_2", "float"), ("rot_3", "float"),
        )
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {points.shape[0]}"]
        header.extend(f"property {kind} {name}" for name, kind in properties)
        header.extend(["end_header", ""])
        dtype = [(name, "<f4") for name, _ in properties]
        vertices = np.zeros((points.shape[0],), dtype=dtype)
        for axis_index, axis in enumerate(("x", "y", "z")):
            vertices[axis] = points[:, axis_index]
        for name in ("scale_0", "scale_1", "scale_2"):
            vertices[name] = -3.0
        vertices["rot_0"] = 1.0
        with path.open("wb") as stream:
            stream.write("\n".join(header).encode("ascii"))
            vertices.tofile(stream)

    @staticmethod
    def write_manifest(
        root: Path,
        supports: dict[int, np.ndarray],
        *,
        gaussian_count: int,
        ply_path: Path,
    ) -> Path:
        root.mkdir()
        frames = []
        for camera_index, indices in supports.items():
            vote_file = f"camera_{camera_index:04d}.npz"
            np.savez_compressed(
                root / vote_file,
                indices=indices.astype(np.uint32),
                class_ids=np.full(indices.shape, 15, dtype=np.uint16),
                weights=np.ones(indices.shape, dtype=np.float32),
            )
            frames.append(
                {
                    "camera_index": camera_index,
                    "camera_id": camera_index,
                    "file": f"camera_{camera_index:04d}.png",
                    "vote_file": vote_file,
                }
            )
        manifest = {
            "source": VOTE_SOURCE,
            "contract": VOTE_CONTRACT,
            "query_region_filtering_used": False,
            "confidence_threshold_used": False,
            "one_normalized_vote_per_camera": True,
            "dinov2_used": False,
            "gaussian_count": gaussian_count,
            "ply_path": str(ply_path),
            "frames": frames,
        }
        manifest["v" + "5_used"] = False
        path = root / "vote_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    @staticmethod
    def write_dinov2_manifest(
        root: Path,
        supports: dict[int, np.ndarray],
        *,
        gaussian_count: int,
    ) -> Path:
        root.mkdir()
        frames = []
        for camera_index, indices in supports.items():
            vote_file = f"camera_{camera_index:04d}.npz"
            np.savez_compressed(
                root / vote_file,
                indices=indices.astype(np.uint32),
                class_ids=np.full(indices.shape, 15, dtype=np.uint16),
                weights=np.ones(indices.shape, dtype=np.float32),
            )
            frames.append(
                {
                    "camera_index": camera_index,
                    "camera_id": camera_index,
                    "file": f"camera_{camera_index:04d}.png",
                    "vote_file": vote_file,
                }
            )
        manifest = {
            "source": DINOV2_SOURCE,
            "gaussian_count": gaussian_count,
            "camera_count": len(frames),
            "frames": frames,
        }
        path = root / "vote_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_main_groups_single_camera_gaussians_and_keeps_zero_camera_black(self) -> None:
        anchor_count = 160
        black_count = 6
        gaussian_count = anchor_count + black_count
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.column_stack(
                [
                    np.arange(gaussian_count, dtype=np.float32) * np.float32(0.01),
                    np.zeros((gaussian_count,), dtype=np.float32),
                    np.zeros((gaussian_count,), dtype=np.float32),
                ]
            )
            points[-1, 0] = 10.0
            ply_path = root / "point_cloud.ply"
            self.write_ply(ply_path, points)
            anchors = np.arange(anchor_count, dtype=np.uint32)
            baseline_manifest = self.write_manifest(
                root / "baseline_votes",
                {0: anchors, 1: anchors, 2: anchors},
                gaussian_count=gaussian_count,
                ply_path=ply_path,
            )
            additional_manifest = self.write_manifest(
                root / "additional_votes",
                {
                    3: np.concatenate([anchors, [160, 164]]).astype(np.uint32),
                    4: np.concatenate([anchors, [160, 161]]).astype(np.uint32),
                    5: np.concatenate([anchors, [161, 162]]).astype(np.uint32),
                    6: np.concatenate([anchors, [162, 163]]).astype(np.uint32),
                },
                gaussian_count=gaussian_count,
                ply_path=ply_path,
            )
            hard_report = root / "hard_report.json"
            hard_report.write_text(
                json.dumps(
                    {
                        "source": HARD_AUDIT_SOURCE,
                        "contract": HARD_AUDIT_CONTRACT,
                        "camera_indices": [0, 1, 2],
                        "gaussian_count": gaussian_count,
                    }
                ),
                encoding="utf-8",
            )
            status = np.full((gaussian_count,), STATUS_UNOBSERVED, dtype=np.uint8)
            status[:anchor_count] = STATUS_ACCEPTED
            np.savez_compressed(
                root / "hard_diagnostics.npz",
                semantic_camera_count=np.concatenate(
                    [np.full((anchor_count,), 3, dtype=np.uint16), np.zeros((black_count,), dtype=np.uint16)]
                ),
                winner_camera_count=np.concatenate(
                    [np.full((anchor_count,), 3, dtype=np.uint8), np.zeros((black_count,), dtype=np.uint8)]
                ),
                consensus_status=status,
            )
            confusion = np.zeros((ONTOLOGY.class_count + 1, ONTOLOGY.class_count + 1), dtype=np.uint64)
            confusion[15, 15] = 10_000
            np.savez_compressed(root / "hard_confusion.npz", counts=confusion)
            recovery_report = root / "recovery_report.json"
            recovery_report.write_text(
                json.dumps(
                    {
                        "source": RECOVERY_SOURCE,
                        "contract": RECOVERY_CONTRACT,
                        "scene": "playroom",
                        "gaussian_count": gaussian_count,
                        "baseline_camera_indices": [0, 1, 2],
                        "additional_camera_indices": [3, 4, 5, 6],
                    }
                ),
                encoding="utf-8",
            )
            candidate = np.zeros((gaussian_count,), dtype=np.uint16)
            candidate[:anchor_count] = 15
            np.save(root / "candidate.npy", candidate)
            source_codes = np.zeros((gaussian_count,), dtype=np.uint8)
            source_codes[:anchor_count] = 1
            np.save(root / "source_codes.npy", source_codes)
            output = root / "output"
            argv = [
                "audit",
                "--scene", "playroom",
                "--baseline-vote-manifest", str(baseline_manifest),
                "--additional-vote-manifest", str(additional_manifest),
                "--hard-audit-report", str(hard_report),
                "--hard-diagnostics", str(root / "hard_diagnostics.npz"),
                "--hard-confusion", str(root / "hard_confusion.npz"),
                "--recovery-report", str(recovery_report),
                "--recovery-candidate-labels", str(root / "candidate.npy"),
                "--recovery-source-codes", str(root / "source_codes.npy"),
                "--source-ply", str(ply_path),
                "--ontology", str(ROOT / "configs" / "ade20k_to_project.json"),
                "--output-dir", str(output),
                "--query-workers", "1",
            ]
            with patch.object(sys, "argv", argv):
                audit_main()
            report = json.loads(
                (output / "observed_black_component_graph_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["current_black_zero_combined_camera_count"], 1)
            self.assertEqual(report["current_black_single_combined_camera_count"], 2)
            self.assertIs(report["dinov2_agreement_gate_used"], False)
            with np.load(output / "observed_black_component_graph_diagnostics.npz") as diagnostics:
                decisions = diagnostics["decision_code"]
                camera_counts = diagnostics["combined_semantic_camera_count"]
                self.assertEqual(int(decisions[camera_counts == 0][0]), DECISION_ZERO_CAMERA)
                self.assertTrue(np.all(decisions[camera_counts == 1] == DECISION_ELIGIBLE_COMPONENT))
            self.assertFalse(any(output.rglob("*.ply")))

            # Re-run the same audit with an agreeing DINOv2 second source and
            # verify the report records the agreement gate.
            dinov2_manifest = self.write_dinov2_manifest(
                root / "dinov2_votes",
                {
                    0: anchors,
                    1: anchors,
                    2: anchors,
                    3: np.concatenate([anchors, [160, 164]]).astype(np.uint32),
                    4: np.concatenate([anchors, [160, 161]]).astype(np.uint32),
                    5: np.concatenate([anchors, [161, 162]]).astype(np.uint32),
                    6: np.concatenate([anchors, [162, 163]]).astype(np.uint32),
                },
                gaussian_count=gaussian_count,
            )
            output_gated = root / "output_gated"
            output_dir_index = argv.index("--output-dir")
            argv_gated = (
                argv[:output_dir_index] + argv[output_dir_index + 2 :]
            )
            argv_gated.extend(
                [
                    "--dinov2-vote-manifest",
                    str(dinov2_manifest),
                    "--output-dir",
                    str(output_gated),
                ]
            )
            with patch.object(sys, "argv", argv_gated):
                audit_main()
            report_gated = json.loads(
                (
                    output_gated
                    / "observed_black_component_graph_audit.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIs(report_gated["dinov2_agreement_gate_used"], True)
            self.assertEqual(
                report_gated["dinov2_vote_manifest"], str(dinov2_manifest)
            )
            self.assertEqual(
                report_gated["current_black_zero_combined_camera_count"], 1
            )


if __name__ == "__main__":
    unittest.main()
