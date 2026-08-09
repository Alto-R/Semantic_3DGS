from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.audit_core_first_dense_cross_validation import (
    CONTRACT,
    decide_camera_winners,
    main,
)
from scripts.task1.dinov3.lift_confident_dense_view_votes import (
    CONTRACT as VOTE_CONTRACT,
    SOURCE as VOTE_SOURCE,
    confidence_keep_mask,
    confident_sparse_view_votes,
)
from scripts.task1.dinov3.render_incremental_spatial_core_fill import (
    build_exact_residual_colors,
)
from scripts.task1.dinov3.summarize_dense_cross_validation_sweep import (
    summarize_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "ade20k_to_project.json"
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_core_first_dense_cross_validation_audit_scene.sbatch"
)


def vertex_array(count: int) -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
        ]
    )
    vertices = np.zeros((count,), dtype=dtype)
    vertices["x"] = np.arange(count, dtype=np.float32) * 0.01
    for name in ("scale_0", "scale_1", "scale_2"):
        vertices[name] = np.log(0.005)
    return vertices


class ConfidentDenseLiftTest(unittest.TestCase):
    def test_abstain_mass_is_not_renormalized_away(self) -> None:
        used = np.asarray(
            [
                [1.0, 3.0, 0.0],
                [3.0, 0.0, 2.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        class_ids = np.asarray([0, 7, 11], dtype=np.uint16)
        indices, classes, weights, visible, accepted = (
            confident_sparse_view_votes(used, class_ids)
        )
        np.testing.assert_array_equal(visible, [0, 1, 2])
        np.testing.assert_allclose(accepted, [0.75, 0.25, 1.0])
        totals = np.zeros((3,), dtype=np.float32)
        np.add.at(totals, indices, weights)
        np.testing.assert_allclose(totals, accepted)
        by_pair = {
            (int(index), int(class_id)): float(weight)
            for index, class_id, weight in zip(indices, classes, weights)
        }
        self.assertAlmostEqual(by_pair[(0, 7)], 0.75)
        self.assertAlmostEqual(by_pair[(1, 11)], 0.25)

    def test_confidence_profile_is_conjunctive(self) -> None:
        segment = {
            "class_id": np.zeros((1, 4), dtype=np.uint8),
            "confidence": np.asarray([[0.8, 0.4, 0.8, 0.8]], dtype=np.float16),
            "normalized_entropy_confidence": np.asarray(
                [[0.8, 0.8, 0.3, 0.8]],
                dtype=np.float16,
            ),
            "max_softmax_probability": np.asarray(
                [[0.8, 0.8, 0.8, 0.2]],
                dtype=np.float16,
            ),
        }
        keep = confidence_keep_mask(
            segment,
            min_relative_margin=0.5,
            min_entropy_confidence=0.5,
            min_max_probability=0.5,
        )
        np.testing.assert_array_equal(keep, [[True, False, False, False]])


class PerGaussianCameraDecisionTest(unittest.TestCase):
    def test_reliable_camera_winner_must_match_proposed_class(self) -> None:
        reliable, matches = decide_camera_winners(
            3,
            np.asarray([0, 0, 1, 1, 2], dtype=np.int32),
            np.asarray([1, 2, 1, 2, 1], dtype=np.uint16),
            np.asarray([0.7, 0.2, 0.3, 0.6, 0.4], dtype=np.float32),
            np.asarray([0.9, 0.9, 0.4], dtype=np.float32),
            np.asarray([1, 1, 1], dtype=np.uint16),
            min_accepted_fraction=0.5,
            min_winner_share=0.5,
            min_winner_margin=0.1,
        )
        np.testing.assert_array_equal(reliable, [1, 2, 0])
        np.testing.assert_array_equal(matches, [True, False, False])

    def test_exact_tie_abstains(self) -> None:
        reliable, matches = decide_camera_winners(
            1,
            np.asarray([0, 0], dtype=np.int32),
            np.asarray([1, 2], dtype=np.uint16),
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1], dtype=np.uint16),
            min_accepted_fraction=0.5,
            min_winner_share=0.5,
            min_winner_margin=0.0,
        )
        np.testing.assert_array_equal(reliable, [0])
        np.testing.assert_array_equal(matches, [False])


class DenseCrossValidationEndToEndTest(unittest.TestCase):
    def test_only_black_dense_validated_gaussians_reach_report_support(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        wall = next(
            item for item in ontology.classes if item.project_class == "wall"
        )
        other = next(
            item for item in ontology.classes if item.project_class == "building"
        )
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            source_ply = temporary / "source.ply"
            source_ply.write_bytes(b"placeholder")
            preferred = np.zeros((8,), dtype=np.int32)
            preferred[0] = wall.project_id
            preferred_path = temporary / "preferred.npy"
            np.save(preferred_path, preferred)

            core_report_path = temporary / "core_report.json"
            core_report_path.write_text(
                json.dumps(
                    {
                        "contract": (
                            "report_only_dinov3_core_first_semantic_identity_v1"
                        ),
                        "report_only": True,
                        "vertex_count": 8,
                        "components": [
                            {
                                "component_id": 1,
                                "accepted": True,
                                "class": wall.project_class,
                                "project_id": wall.project_id,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            core_support_path = temporary / "core_supports.npz"
            np.savez_compressed(
                core_support_path,
                component_000001_indices=np.arange(8, dtype=np.uint32),
            )

            evidence_dir = temporary / "evidence"
            vote_dir = evidence_dir / "view_votes"
            vote_dir.mkdir(parents=True)
            frames = []
            for camera_index in range(3):
                indices = np.repeat(
                    np.arange(8, dtype=np.uint32),
                    2,
                )
                classes = np.tile(
                    np.asarray(
                        [wall.project_id, other.project_id],
                        dtype=np.uint16,
                    ),
                    8,
                )
                weights = np.tile(
                    np.asarray([0.8, 0.1], dtype=np.float32),
                    8,
                )
                # The last Gaussian is a repeatable dense disagreement:
                # the core proposes wall, but every camera independently
                # identifies building.
                weights[-2:] = np.asarray([0.1, 0.8], dtype=np.float32)
                vote_file = vote_dir / f"camera_{camera_index:04d}.npz"
                np.savez_compressed(
                    vote_file,
                    indices=indices,
                    class_ids=classes,
                    weights=weights,
                    visible_indices=np.arange(8, dtype=np.uint32),
                    accepted_fractions=np.full(
                        (8,),
                        0.9,
                        dtype=np.float32,
                    ),
                )
                frames.append(
                    {
                        "file": f"camera_{camera_index:04d}.png",
                        "camera_index": camera_index,
                        "vote_file": vote_file.relative_to(
                            evidence_dir
                        ).as_posix(),
                    }
                )
            vote_manifest_path = evidence_dir / "vote_manifest.json"
            vote_manifest_path.write_text(
                json.dumps(
                    {
                        "source": VOTE_SOURCE,
                        "contract": VOTE_CONTRACT,
                        "profile_name": "balanced",
                        "thresholds": {
                            "min_relative_margin": 0.5,
                            "min_entropy_confidence": 0.4,
                            "min_max_probability": 0.3,
                        },
                        "gaussian_count": 8,
                        "camera_count": 3,
                        "ply_path": str(source_ply),
                        "query_region_filtering_used": False,
                        "confidence_threshold_used": True,
                        "inference_rerun": False,
                        "flashsplat_rerun": True,
                        "abstain_mass_preserved": True,
                        "scene_specific_rules": False,
                        "class_specific_thresholds": False,
                        "manual_component_decisions": False,
                        "v5_used": False,
                        "dinov2_used": False,
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )

            output_dir = temporary / "output"
            argv = [
                "audit_core_first_dense_cross_validation",
                "--core-first-report",
                str(core_report_path),
                "--core-first-supports",
                str(core_support_path),
                "--vote-manifest",
                str(vote_manifest_path),
                "--preferred-labels",
                str(preferred_path),
                "--source-ply",
                str(source_ply),
                "--ontology",
                str(ONTOLOGY_PATH),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
                "--min-spatial-component-gaussians",
                "2",
                "--min-spatial-component-cameras",
                "3",
            ]
            module = (
                "scripts.task1.dinov3."
                "audit_core_first_dense_cross_validation"
            )
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(
                    f"{module}.vertex_data_memmap",
                    return_value=(
                        SimpleNamespace(elements=[SimpleNamespace(count=8)]),
                        vertex_array(8),
                    ),
                ),
            ):
                main()

            report = json.loads(
                (
                    output_dir / "dense_cross_validation_audit.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(report["contract"], CONTRACT)
            self.assertEqual(
                report["black_core_first_candidate_gaussian_count"],
                7,
            )
            self.assertEqual(report["validated_fill_gaussian_count"], 6)
            self.assertEqual(
                report["candidate_gate_reason_counts"][
                    "global_winner_mismatch"
                ],
                1,
            )
            self.assertTrue(report["preferred_labels_read_only"])
            self.assertFalse(report["semantic_labels_written"])
            validated = np.load(output_dir / "dense_validated_fill_mask.npy")
            np.testing.assert_array_equal(
                validated,
                [False, True, True, True, True, True, True, False],
            )
            self.assertFalse((output_dir / "gaussian_labels.npy").exists())
            self.assertFalse((output_dir / "label_map.json").exists())
            self.assertEqual(list(output_dir.glob("*.ply")), [])

            with np.load(
                output_dir / "dense_validated_supports.npz",
                allow_pickle=False,
            ) as supports:
                colors, selected, _palette = build_exact_residual_colors(
                    8,
                    report,
                    supports,
                )
            np.testing.assert_array_equal(selected, validated)
            self.assertEqual(int(np.count_nonzero(colors.max(axis=1))), 6)


class SweepSummaryTest(unittest.TestCase):
    def test_summary_compares_profiles_without_selecting_one(self) -> None:
        base = {
            "contract": CONTRACT,
            "scene": "synthetic",
            "report_only": True,
            "semantic_labels_written": False,
            "semantic_project_class_arrays_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
            "scene_specific_rules": False,
            "class_specific_thresholds": False,
            "manual_component_decisions": False,
            "black_core_first_candidate_gaussian_count": 100,
            "dense_survivor_before_spatial_gaussian_count": 50,
            "validated_fill_gaussian_count": 40,
            "preferred_unlabeled_recovery_ratio": 0.1,
            "accepted_spatial_component_count": 2,
            "candidate_gate_reason_counts": {"global_winner_mismatch": 50},
            "class_validated_fill_gaussian_counts": {"wall": 40},
        }
        reports = [
            {
                **base,
                "profile_name": "permissive",
                "pixel_confidence_thresholds": {"min_relative_margin": 0.25},
            },
            {
                **base,
                "profile_name": "strict",
                "pixel_confidence_thresholds": {"min_relative_margin": 0.70},
                "validated_fill_gaussian_count": 20,
            },
        ]
        summary = summarize_reports(reports)
        self.assertFalse(summary["automatic_profile_selection"])
        self.assertEqual(summary["profile_count"], 2)


class DenseCrossValidationSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_reuses_inference_but_lifts_all_70_cached_views(self) -> None:
        self.assertIn("SEGMENTATION_MANIFEST", self.text)
        self.assertIn("lift_confident_dense_view_votes", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertIn("inference_rerun=0", self.text)
        self.assertIn("flashsplat_rerun=1", self.text)

    def test_three_global_profiles_preserve_explicit_abstain_mass(self) -> None:
        self.assertIn("PROFILES=(permissive balanced strict)", self.text)
        self.assertIn("RELATIVE_MARGINS=(0.25 0.50 0.70)", self.text)
        self.assertIn("ENTROPY_CONFIDENCES=(0.25 0.40 0.55)", self.text)
        self.assertIn("MAX_PROBABILITIES=(0.20 0.30 0.45)", self.text)
        self.assertIn("abstain_mass_preserved=1", self.text)
        self.assertIn("boundary_classes=wall,ceiling,floor", self.text)

    def test_is_report_only_automatic_and_immutable(self) -> None:
        self.assertIn("preferred_labels_read_only=1", self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)
        self.assertIn("manual_component_decisions=0", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("semantic_project_class_arrays_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn("-name 'gaussian_labels.npy'", self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)


if __name__ == "__main__":
    unittest.main()
