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
from scripts.task1.dinov3.associate_3d_query_regions import QueryProposal
from scripts.task1.dinov3.audit_core_first_semantic_identity import (
    CoreFirstThresholds,
    fuse_core_identity,
    main,
    pre_identity_spatial_split,
)
from scripts.task1.dinov3.render_incremental_spatial_core_fill import (
    build_exact_residual_colors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "ade20k_to_project.json"
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_core_first_semantic_identity_audit_scene.sbatch"
)


def proposal(
    proposal_id: int,
    camera_index: int,
    indices: list[int] | np.ndarray,
    *,
    winner: int,
) -> QueryProposal:
    values = np.asarray(indices, dtype=np.uint32)
    probabilities = np.full((150,), 1.0e-6, dtype=np.float32)
    probabilities[winner] = 0.9
    probabilities[(winner + 1) % 150] = 0.1
    probabilities /= probabilities.sum()
    return QueryProposal(
        proposal_id=proposal_id,
        frame_file=f"camera_{camera_index:04d}.png",
        camera_index=camera_index,
        region_id=proposal_id,
        indices=values,
        counts=np.ones(values.shape, dtype=np.float32),
        class_probabilities=probabilities,
        no_object_probability=0.01,
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        quality=0.9,
        metadata={},
    )


def vertex_array() -> np.ndarray:
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
    vertices = np.zeros((8,), dtype=dtype)
    vertices["x"] = [0.00, 0.01, 0.02, 0.03, 1.00, 1.01, 1.02, 1.03]
    for name in ("scale_0", "scale_1", "scale_2"):
        vertices[name] = np.log(0.005)
    return vertices


def mixed_source_record(
    proposals: list[QueryProposal],
    *,
    class_name: str,
    project_id: int,
    ade_id: int,
    kind: str,
) -> dict[str, object]:
    return {
        "component_id": 1,
        "status": "abstained_unstable_multiview_identity",
        "accepted": False,
        "class": class_name,
        "project_id": project_id,
        "ade_id": ade_id,
        "kind": kind,
        "probability": 0.5,
        "proposal_ids": [item.proposal_id for item in proposals],
        "source_frames": [item.frame_file for item in proposals],
        "source_view_count": len(proposals),
        "support_gaussian_count": 8,
        "semantic_stability": {
            "view_count": len(proposals),
            "winner_view_count": 2,
            "view_winners": [class_name, class_name, "other", "other"],
            "leave_one_out_winners": [],
            "stable": False,
        },
    }


class CoreFirstOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = load_ontology(ONTOLOGY_PATH)

    def test_mixed_graph_is_spatially_split_before_identity(self) -> None:
        proposals = [
            proposal(1, 0, [0, 1, 2, 3], winner=0),
            proposal(2, 1, [0, 1, 2, 3], winner=0),
            proposal(3, 2, [4, 5, 6, 7], winner=1),
            proposal(4, 3, [4, 5, 6, 7], winner=1),
        ]
        splits, median, voxel_size, geometry, union_count = (
            pre_identity_spatial_split(
                proposals,
                vertex_array(),
                CoreFirstThresholds(
                    min_spatial_component_gaussians=2,
                ),
            )
        )
        self.assertEqual(union_count, 8)
        self.assertAlmostEqual(median, 0.005)
        self.assertAlmostEqual(voxel_size, 0.02)
        self.assertEqual(geometry["component_count"], 2)
        self.assertEqual([item[0].tolist() for item in splits], [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ])

        first_tier, first_winner, _, _, _ = fuse_core_identity(
            proposals[:2],
            self.ontology,
        )
        second_tier, second_winner, _, _, _ = fuse_core_identity(
            proposals[2:],
            self.ontology,
        )
        self.assertEqual(first_tier, "strict_unanimous")
        self.assertEqual(second_tier, "strict_unanimous")
        self.assertEqual((first_winner, second_winner), (0, 1))

    def test_one_core_outlier_is_removed_only_after_core_identity(self) -> None:
        proposals = [
            proposal(1, 0, [0, 1, 2, 3], winner=0),
            proposal(2, 1, [0, 1, 2, 3], winner=0),
            proposal(3, 2, [0, 1, 2, 3], winner=0),
            proposal(4, 3, [0, 1, 2, 3], winner=1),
        ]
        tier, winner, agreeing, dissenting, stability = fuse_core_identity(
            proposals,
            self.ontology,
        )
        self.assertEqual(tier, "one_view_outlier_leave_one_out_stable")
        self.assertEqual(winner, 0)
        self.assertEqual([item.proposal_id for item in agreeing], [1, 2, 3])
        self.assertEqual([item.proposal_id for item in dissenting], [4])
        self.assertEqual(stability["winner_view_count"], 3)


class CoreFirstEndToEndTest(unittest.TestCase):
    def test_mixed_source_graph_recovers_two_independent_semantic_cores(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        proposals = [
            proposal(1, 0, [0, 1, 2, 3], winner=0),
            proposal(2, 1, [0, 1, 2, 3], winner=0),
            proposal(3, 2, [4, 5, 6, 7], winner=1),
            proposal(4, 3, [4, 5, 6, 7], winner=1),
        ]
        first = ontology.classes[0]
        source_record = mixed_source_record(
            proposals,
            class_name=first.project_class,
            project_id=first.project_id,
            ade_id=first.ade_id,
            kind=first.kind,
        )
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            source_ply = temporary / "source.ply"
            source_ply.write_bytes(b"placeholder")
            proposal_manifest_path = temporary / "proposal_manifest.json"
            proposal_manifest_path.write_text("{}", encoding="utf-8")
            component_report_path = temporary / "component_report.json"
            component_report_path.write_text(
                json.dumps(
                    {
                        "contract": "report_only_class_agnostic_3d_association_v1",
                        "v5_used": False,
                        "dinov2_used": False,
                        "semantic_identity_hardened_before_3d": False,
                        "outputs": {"report_only": True},
                        "proposal_count": len(proposals),
                        "components": [source_record],
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "vertex_count": 8,
                "ply_path": str(source_ply),
                "proposals": [
                    {
                        "proposal_id": item.proposal_id,
                        "frame_file": item.frame_file,
                    }
                    for item in proposals
                ],
            }
            header = SimpleNamespace(elements=[SimpleNamespace(count=8)])
            output_dir = temporary / "output"
            argv = [
                "audit_core_first_semantic_identity",
                "--component-report",
                str(component_report_path),
                "--proposal-manifest",
                str(proposal_manifest_path),
                "--ontology",
                str(ONTOLOGY_PATH),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
                "--min-spatial-component-gaussians",
                "2",
            ]
            module = (
                "scripts.task1.dinov3.audit_core_first_semantic_identity"
            )
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(
                    f"{module}.load_query_proposals",
                    return_value=(proposals, manifest),
                ),
                mock.patch(
                    f"{module}.vertex_data_memmap",
                    return_value=(header, vertex_array()),
                ),
            ):
                main()

            report = json.loads(
                (
                    output_dir / "core_first_semantic_identity_audit.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(report["mixed_source_component_count"], 1)
            self.assertEqual(
                report["mixed_source_component_with_accepted_core_count"],
                1,
            )
            self.assertEqual(report["pre_identity_spatial_core_count"], 2)
            self.assertEqual(report["accepted_core_first_component_count"], 2)
            self.assertEqual(report["proposed_core_gaussian_count"], 8)
            self.assertEqual(
                {item["class"] for item in report["components"]},
                {
                    ontology.classes[0].project_class,
                    ontology.classes[1].project_class,
                },
            )
            self.assertFalse(report["semantic_labels_written"])
            self.assertTrue((output_dir / "core_first_mask.npy").is_file())
            self.assertTrue((output_dir / "core_first_supports.npz").is_file())
            self.assertFalse((output_dir / "gaussian_labels.npy").exists())
            self.assertFalse(
                (output_dir / "gaussian_project_class_ids.npy").exists()
            )
            self.assertFalse((output_dir / "label_map.json").exists())
            self.assertEqual(list(output_dir.glob("*.ply")), [])

    def test_exact_renderer_accepts_core_first_sparse_support_contract(self) -> None:
        report = {
            "contract": "report_only_dinov3_core_first_semantic_identity_v1",
            "vertex_count": 4,
            "proposed_core_gaussian_count": 3,
            "components": [
                {
                    "component_id": 1,
                    "accepted": True,
                    "class": "wall",
                }
            ],
        }
        supports = {
            "component_000001_indices": np.asarray(
                [0, 1, 3],
                dtype=np.uint32,
            )
        }
        colors, selected, palette = build_exact_residual_colors(
            4,
            report,
            supports,
        )
        np.testing.assert_array_equal(selected, [True, True, False, True])
        self.assertEqual(int(np.count_nonzero(colors.max(axis=1))), 3)
        self.assertEqual(palette[0]["class"], "wall")


class CoreFirstSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_reuses_cached_evidence_and_includes_mixed_graphs(self) -> None:
        self.assertIn("SOURCE_COMPONENT_REPORT", self.text)
        self.assertIn("SOURCE_PROPOSAL_MANIFEST", self.text)
        self.assertIn("source_mixed_components_included=1", self.text)
        self.assertIn(
            "all_graphs_to_multiview_spatial_cores_then_per_core_identity",
            self.text,
        )
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("run_flashsplat_mask_proposals", self.text)
        self.assertIn("inference_rerun=0", self.text)
        self.assertIn("flashsplat_rerun=0", self.text)

    def test_scheduler_has_one_global_report_only_policy(self) -> None:
        for value in (
            'MIN_GAUSSIAN_CAMERA_SUPPORT="${MIN_GAUSSIAN_CAMERA_SUPPORT:-2}"',
            'MIN_SPATIAL_COMPONENT_GAUSSIANS="${MIN_SPATIAL_COMPONENT_GAUSSIANS:-500}"',
            'MIN_SPATIAL_COMPONENT_CAMERAS="${MIN_SPATIAL_COMPONENT_CAMERAS:-2}"',
            'VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"',
            'MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"',
            'MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"',
        ):
            self.assertIn(value, self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)
        self.assertIn("manual_component_decisions=0", self.text)
        self.assertIn("report_only=1", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)

    def test_scheduler_renders_exact_support_and_forbids_materialization(self) -> None:
        self.assertIn("render_incremental_spatial_core_fill", self.text)
        self.assertIn("core_first_supports.npz", self.text)
        self.assertIn("exact_overlays.png", self.text)
        self.assertIn("exact_masks.png", self.text)
        self.assertIn("-name 'gaussian_labels.npy'", self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)


if __name__ == "__main__":
    unittest.main()
