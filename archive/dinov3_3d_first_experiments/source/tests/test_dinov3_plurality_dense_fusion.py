from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.task1.dinov3.combine_plurality_component_seeds import (
    ALLOWED_COMPONENT_CONTRACTS,
    CAMERA_OWNED_COMPONENT_CONTRACT,
    CAMERA_OWNED_COMPONENT_SOURCE,
    CONTRACT as COMBINED_SEED_CONTRACT,
    SOURCE as COMBINED_SEED_SOURCE,
    combine_seed_labels,
)
from scripts.task1.dinov3.materialize_3d_component_labels import ComponentSupport
from scripts.task1.dinov3.materialize_camera_owned_component_labels import (
    CONTRACT as MATERIALIZED_CAMERA_OWNED_CONTRACT,
    SOURCE as MATERIALIZED_CAMERA_OWNED_SOURCE,
    aggregate_supporting_camera_counts,
    main as camera_owned_main,
    resolve_camera_support_ownership,
)
from scripts.task1.dinov3.materialize_plurality_component_labels import (
    unique_camera_plurality,
)
from scripts.task1.dinov3.propagate_dense_labels_from_region_seeds import (
    ALLOWED_SEED_CONTRACTS,
    main as propagate_main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_plurality_dense_scene.sbatch"
)


class CameraPluralityTest(unittest.TestCase):
    def test_each_camera_has_one_equal_vote(self) -> None:
        winner, votes, counts = unique_camera_plurality(
            ["windowpane", "door", "wall", "windowpane", "signboard"]
        )
        self.assertEqual(winner, "windowpane")
        self.assertEqual(votes, 2)
        self.assertEqual(
            counts,
            {"door": 1, "signboard": 1, "wall": 1, "windowpane": 2},
        )

    def test_tied_plurality_abstains(self) -> None:
        winner, votes, counts = unique_camera_plurality(
            ["cabinet", "wardrobe", "wall", "wardrobe", "cabinet"]
        )
        self.assertIsNone(winner)
        self.assertEqual(votes, 2)
        self.assertEqual(counts["cabinet"], 2)
        self.assertEqual(counts["wardrobe"], 2)

    def test_single_camera_is_not_multiview_identity(self) -> None:
        winner, votes, counts = unique_camera_plurality(["door"])
        self.assertIsNone(winner)
        self.assertEqual(votes, 0)
        self.assertEqual(counts, {"door": 1})


class SeedCombinationTest(unittest.TestCase):
    def test_components_add_and_override_region_identity_globally(self) -> None:
        combined, sources, additions, overrides = combine_seed_labels(
            np.asarray([0, 15, 1, 4, 0], dtype=np.int32),
            np.asarray([9, 9, 0, 4, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(combined, [9, 9, 1, 4, 0])
        np.testing.assert_array_equal(sources, [2, 2, 1, 2, 0])
        np.testing.assert_array_equal(additions, [True, False, False, False, False])
        np.testing.assert_array_equal(overrides, [False, True, False, False, False])

    def test_combined_seed_contract_is_accepted_by_dense_propagation(self) -> None:
        self.assertIn(
            (COMBINED_SEED_SOURCE, COMBINED_SEED_CONTRACT),
            ALLOWED_SEED_CONTRACTS,
        )

    def test_camera_owned_component_contract_is_accepted(self) -> None:
        self.assertEqual(
            CAMERA_OWNED_COMPONENT_SOURCE,
            MATERIALIZED_CAMERA_OWNED_SOURCE,
        )
        self.assertEqual(
            CAMERA_OWNED_COMPONENT_CONTRACT,
            MATERIALIZED_CAMERA_OWNED_CONTRACT,
        )
        self.assertIn(
            (
                CAMERA_OWNED_COMPONENT_SOURCE,
                CAMERA_OWNED_COMPONENT_CONTRACT,
            ),
            ALLOWED_COMPONENT_CONTRACTS,
        )

    def test_propagation_reports_region_and_component_seed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            source_ply = temp_dir / "point_cloud.ply"
            header = [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "end_header",
                "",
            ]
            log_scale = float(np.log(0.01))
            payload = b"".join(
                struct.pack("<6f", x, 0.0, 0.0, log_scale, log_scale, log_scale)
                for x in (0.0, 0.01, 0.02)
            )
            source_ply.write_bytes("\n".join(header).encode("ascii") + payload)
            seeds = temp_dir / "seeds.npy"
            dense = temp_dir / "dense.npy"
            np.save(seeds, np.asarray([4, 9, 0], dtype=np.int32))
            np.save(dense, np.zeros((3,), dtype=np.int32))
            base = {
                "scene": "synthetic",
                "source_ply": str(source_ply),
                "vertex_count": 3,
                "v5_used": False,
                "dinov2_used": False,
            }
            seed_summary = temp_dir / "combined_seed_summary.json"
            seed_summary.write_text(
                json.dumps(
                    {
                        **base,
                        "source": COMBINED_SEED_SOURCE,
                        "contract": COMBINED_SEED_CONTRACT,
                        "region_seed_gaussian_count": 1,
                        "component_seed_gaussian_count": 1,
                        "combined_seed_gaussian_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            dense_summary = temp_dir / "dense_vote_summary.json"
            dense_summary.write_text(
                json.dumps(
                    {
                        **base,
                        "source": "dinov3_dense_pixel_flashsplat_soft_class_mass",
                        "contract": "complete_dense_pixels_unique_soft_argmax_v1",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = temp_dir / "output"
            argv = [
                "propagate_dense_labels_from_region_seeds",
                "--seed-labels",
                str(seeds),
                "--seed-summary",
                str(seed_summary),
                "--dense-labels",
                str(dense),
                "--dense-summary",
                str(dense_summary),
                "--ontology",
                str(PROJECT_ROOT / "configs" / "ade20k_to_project.json"),
                "--source-ply",
                str(source_ply),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
                "--no-semantic-ply",
            ]
            with mock.patch.object(sys, "argv", argv):
                propagate_main()
            summary = json.loads(
                (output_dir / "propagation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["seed_gaussian_count"], 2)
            self.assertEqual(summary["region_seed_gaussian_count"], 1)
            self.assertEqual(summary["plurality_component_seed_gaussian_count"], 1)
            self.assertTrue((output_dir / "seed_mask.npy").is_file())


class CameraSupportOwnershipTest(unittest.TestCase):
    @staticmethod
    def component(
        component_id: int,
        project_id: int,
        indices: list[int],
        scores: list[float],
        *,
        source_view_count: int = 3,
    ) -> ComponentSupport:
        return ComponentSupport(
            component_id=component_id,
            project_id=project_id,
            indices=np.asarray(indices, dtype=np.uint32),
            scores=np.asarray(scores, dtype=np.float32),
            record={"source_view_count": source_view_count},
        )

    def test_unique_camera_count_wins_and_cross_class_tie_abstains(self) -> None:
        components = [
            self.component(1, 9, [0, 1, 2, 3], [1.0, 1.0, 1.0, 0.1]),
            self.component(2, 15, [1, 2, 3, 4], [9.0, 9.0, 9.0, 9.0]),
        ]
        counts = [
            np.asarray([1, 2, 2, 1], dtype=np.uint16),
            np.asarray([1, 2, 3, 1], dtype=np.uint16),
        ]
        labels, classes, ties, overlaps, winners = (
            resolve_camera_support_ownership(components, counts, 5)
        )
        np.testing.assert_array_equal(classes, [9, 9, 0, 15, 15])
        np.testing.assert_array_equal(ties, [False, False, True, False, False])
        np.testing.assert_array_equal(
            overlaps,
            [False, True, True, True, False],
        )
        np.testing.assert_array_equal(winners, [1, 2, 2, 3, 1])
        self.assertEqual(labels[2], 0)

    def test_cross_class_decisions_ignore_support_magnitude(self) -> None:
        components = [
            self.component(1, 9, [0, 1], [0.01, 0.01]),
            self.component(2, 15, [0, 1], [100.0, 100.0]),
        ]
        counts = [
            np.asarray([2, 2], dtype=np.uint16),
            np.asarray([1, 2], dtype=np.uint16),
        ]
        _, classes, ties, *_ = resolve_camera_support_ownership(
            components,
            counts,
            2,
        )
        np.testing.assert_array_equal(classes, [9, 0])
        np.testing.assert_array_equal(ties, [False, True])

    def test_project_class_result_is_component_order_invariant(self) -> None:
        components = [
            self.component(1, 9, [0, 1], [1.0, 1.0]),
            self.component(2, 15, [0, 1], [2.0, 2.0]),
        ]
        counts = [
            np.asarray([2, 1], dtype=np.uint16),
            np.asarray([1, 1], dtype=np.uint16),
        ]
        forward = resolve_camera_support_ownership(components, counts, 2)
        reverse = resolve_camera_support_ownership(
            list(reversed(components)),
            list(reversed(counts)),
            2,
        )
        np.testing.assert_array_equal(forward[1], reverse[1])
        np.testing.assert_array_equal(forward[2], reverse[2])

    def test_same_class_camera_tie_does_not_abstain(self) -> None:
        components = [
            self.component(1, 9, [0], [1.0]),
            self.component(2, 9, [0], [2.0]),
        ]
        counts = [
            np.asarray([1], dtype=np.uint16),
            np.asarray([1], dtype=np.uint16),
        ]
        labels, classes, ties, *_ = resolve_camera_support_ownership(
            components,
            counts,
            1,
        )
        np.testing.assert_array_equal(classes, [9])
        np.testing.assert_array_equal(ties, [False])
        np.testing.assert_array_equal(labels, [2])

    def test_camera_counts_are_one_presence_vote_per_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            np.savez_compressed(
                temp_dir / "a.npz",
                indices=np.asarray([0, 1, 3], dtype=np.uint32),
                counts=np.ones((3,), dtype=np.float32),
            )
            np.savez_compressed(
                temp_dir / "b.npz",
                indices=np.asarray([1, 2, 3], dtype=np.uint32),
                counts=np.ones((3,), dtype=np.float32),
            )
            result = aggregate_supporting_camera_counts(
                [10, 11],
                {
                    10: {"support_file": "a.npz"},
                    11: {"support_file": "b.npz"},
                },
                temp_dir,
                np.asarray([0, 1, 2, 3], dtype=np.uint32),
            )
            np.testing.assert_array_equal(result, [1, 2, 1, 2])

    def test_camera_owned_materializer_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            source_ply = temp_dir / "point_cloud.ply"
            source_ply.write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 5",
                        "property float x",
                        "property float y",
                        "property float z",
                        "end_header",
                        "0 0 0",
                        "1 0 0",
                        "2 0 0",
                        "3 0 0",
                        "4 0 0",
                        "",
                    ]
                ),
                encoding="ascii",
            )
            support_dir = temp_dir / "proposal_supports"
            support_dir.mkdir()
            supports = {
                0: [0, 1, 2],
                1: [0, 1],
                2: [1, 2, 3],
                3: [1, 3, 4],
            }
            proposals = []
            for proposal_id, indices in supports.items():
                filename = f"support_{proposal_id}.npz"
                np.savez_compressed(
                    support_dir / filename,
                    indices=np.asarray(indices, dtype=np.uint32),
                    counts=np.ones((len(indices),), dtype=np.float32),
                )
                proposals.append(
                    {
                        "proposal_id": proposal_id,
                        "frame_file": f"frame_{proposal_id}.png",
                        "support_file": filename,
                    }
                )
            manifest = temp_dir / "proposal_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "vertex_count": 5,
                        "ply_path": str(source_ply),
                        "proposals": proposals,
                    }
                ),
                encoding="utf-8",
            )
            report = temp_dir / "component_report.json"
            report.write_text(
                json.dumps(
                    {
                        "contract": "report_only_class_agnostic_3d_association_v1",
                        "v5_used": False,
                        "dinov2_used": False,
                        "semantic_identity_hardened_before_3d": False,
                        "proposal_count": 4,
                        "components": [
                            {
                                "component_id": 1,
                                "class": "windowpane",
                                "probability": 0.6,
                                "proposal_ids": [0, 1],
                                "source_frames": ["frame_0.png", "frame_1.png"],
                                "source_view_count": 2,
                                "support_gaussian_count": 3,
                                "semantic_stability": {
                                    "view_winners": ["windowpane", "windowpane"]
                                },
                            },
                            {
                                "component_id": 2,
                                "class": "door",
                                "probability": 0.7,
                                "proposal_ids": [2, 3],
                                "source_frames": ["frame_2.png", "frame_3.png"],
                                "source_view_count": 2,
                                "support_gaussian_count": 4,
                                "semantic_stability": {
                                    "view_winners": ["door", "door"]
                                },
                            },
                        ],
                        "outputs": {"report_only": True},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = temp_dir / "output"
            argv = [
                "materialize_camera_owned_component_labels",
                "--proposal-manifest",
                str(manifest),
                "--component-report",
                str(report),
                "--ontology",
                str(PROJECT_ROOT / "configs" / "ade20k_to_project.json"),
                "--source-ply",
                str(source_ply),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
                "--no-semantic-ply",
            ]
            with mock.patch.object(sys, "argv", argv):
                camera_owned_main()
            np.testing.assert_array_equal(
                np.load(output_dir / "gaussian_project_class_ids.npy"),
                [9, 0, 0, 15, 15],
            )
            summary = json.loads(
                (
                    output_dir / "camera_owned_component_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["cross_class_overlap_gaussian_count"], 2)
            self.assertEqual(
                summary["cross_class_camera_count_tie_gaussian_count"],
                2,
            )
            self.assertEqual(summary["assigned_gaussian_count"], 3)


class PluralityDenseSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_reuses_all_cached_dinov3_evidence(self) -> None:
        self.assertIn("materialize_plurality_component_labels", self.text)
        self.assertIn("combine_plurality_component_seeds", self.text)
        self.assertIn("propagate_dense_labels_from_region_seeds", self.text)
        self.assertIn("region_vote_summary.json", self.text)
        self.assertIn("dense_vote_summary.json", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("lift_dense_view_votes", self.text)

    def test_scheduler_uses_structural_plurality_without_probability_gates(self) -> None:
        self.assertIn('MIN_COMPONENT_VIEWS="${MIN_COMPONENT_VIEWS:-2}"', self.text)
        self.assertIn("component_acceptance=unique_plurality", self.text)
        self.assertIn("soft_probability_weighting_used=0", self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)
        self.assertIn("offline_threshold_sweep_used=0", self.text)
        self.assertNotIn("MIN_PROBABILITY", self.text)
        self.assertNotIn("MIN_MARGIN", self.text)

    def test_scheduler_supports_unique_camera_ownership_without_thresholds(self) -> None:
        self.assertIn(
            'COMPONENT_OWNERSHIP="${COMPONENT_OWNERSHIP:-cross_class_abstain}"',
            self.text,
        )
        self.assertIn("unique_camera_support", self.text)
        self.assertIn("materialize_camera_owned_component_labels", self.text)
        self.assertIn(
            "cross_class_geometric_strength_weighting_used=0",
            self.text,
        )
        self.assertNotIn("MIN_CAMERA_SUPPORT", self.text)

    def test_scheduler_writes_end_to_end_review_artifacts(self) -> None:
        self.assertIn("semantic_point_cloud_supersplat_debug.ply", self.text)
        self.assertIn("render_auto_label_overlays", self.text)
        self.assertIn("visible_overlay_coverage.json", self.text)
        self.assertIn("semantic_labels.png", self.text)


if __name__ == "__main__":
    unittest.main()
