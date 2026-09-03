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
from scripts.task1.dinov3.audit_multiview_spatial_core import (
    AuditThresholds,
    SpatialCoreSupport,
    aggregate_camera_presence,
    classify_component_identity,
    main,
    resolve_spatial_core_ownership,
    split_multiview_spatial_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "ade20k_to_project.json"
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_multiview_spatial_core_audit_scene.sbatch"
)
RENDERER = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "render_3d_component_overlays.py"
)


def proposal(
    proposal_id: int,
    camera_index: int,
    indices: list[int] | np.ndarray,
    *,
    winner: int = 0,
) -> QueryProposal:
    probabilities = np.full((150,), 1.0e-6, dtype=np.float32)
    probabilities[winner] = 0.9
    probabilities[(winner + 1) % 150] = 0.1
    probabilities /= probabilities.sum()
    return QueryProposal(
        proposal_id=proposal_id,
        frame_file=f"camera_{camera_index:04d}.png",
        camera_index=camera_index,
        region_id=proposal_id,
        indices=np.asarray(indices, dtype=np.uint32),
        counts=np.ones((len(indices),), dtype=np.float32),
        class_probabilities=probabilities,
        no_object_probability=0.01,
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        quality=0.9,
        metadata={},
    )


def source_record(proposals: list[QueryProposal], winner: int = 0) -> dict[str, object]:
    ontology = load_ontology(ONTOLOGY_PATH)
    item = ontology.classes[winner]
    return {
        "component_id": 7,
        "status": "abstained_unstable_multiview_identity",
        "accepted": False,
        "class": item.project_class,
        "project_id": item.project_id,
        "ade_id": item.ade_id,
        "kind": item.kind,
        "probability": 0.9,
        "proposal_ids": [value.proposal_id for value in proposals],
        "source_frames": [value.frame_file for value in proposals],
        "source_view_count": len(proposals),
        "support_gaussian_count": len(
            np.unique(np.concatenate([value.indices for value in proposals]))
        ),
    }


class MultiviewIdentityAndSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = load_ontology(ONTOLOGY_PATH)

    def test_each_camera_contributes_one_presence_vote(self) -> None:
        proposals = [
            proposal(1, 0, [0, 1, 2]),
            proposal(2, 1, [1, 2, 3]),
            proposal(3, 2, [2, 3, 4]),
        ]
        indices, counts = aggregate_camera_presence(proposals)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(counts, [1, 2, 3, 2, 1])

    def test_duplicate_camera_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple proposals from one camera"):
            aggregate_camera_presence(
                [proposal(1, 0, [0, 1]), proposal(2, 0, [1, 2])]
            )

    def test_exactly_one_outlier_is_excluded_before_geometry(self) -> None:
        proposals = [
            proposal(1, 0, [0, 1]),
            proposal(2, 1, [0, 1]),
            proposal(3, 2, [0, 1]),
            proposal(4, 3, [2, 3], winner=1),
        ]
        tier, agreeing, dissenting, stability = classify_component_identity(
            source_record(proposals), proposals, self.ontology
        )
        self.assertEqual(tier, "one_view_outlier_leave_one_out_stable")
        self.assertEqual([item.proposal_id for item in agreeing], [1, 2, 3])
        self.assertEqual([item.proposal_id for item in dissenting], [4])
        self.assertEqual(stability["winner_view_count"], 3)

    def test_two_outliers_remain_mixed(self) -> None:
        proposals = [
            proposal(1, 0, [0]),
            proposal(2, 1, [0]),
            proposal(3, 2, [0]),
            proposal(4, 3, [1], winner=1),
            proposal(5, 4, [1], winner=1),
        ]
        tier, agreeing, _dissenting, _stability = classify_component_identity(
            source_record(proposals), proposals, self.ontology
        )
        self.assertEqual(tier, "mixed_or_unstable")
        self.assertEqual(agreeing, [])


class SpatialCoreGeometryTest(unittest.TestCase):
    def test_adaptive_voxels_split_disconnected_support_and_apply_global_size(self) -> None:
        indices = np.arange(7, dtype=np.uint32)
        counts = np.full((7,), 2, dtype=np.uint16)
        points = np.asarray(
            [
                [0.00, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [1.00, 0.0, 0.0],
                [1.01, 0.0, 0.0],
                [1.02, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        log_scales = np.full((7, 3), np.log(0.005), dtype=np.float64)
        agreeing = [proposal(1, 0, indices), proposal(2, 1, indices)]
        records, median, voxel_size, geometry = split_multiview_spatial_support(
            7,
            indices,
            counts,
            agreeing,
            points,
            log_scales,
            AuditThresholds(min_spatial_component_gaussians=4),
        )
        self.assertAlmostEqual(median, 0.005)
        self.assertAlmostEqual(voxel_size, 0.02)
        self.assertEqual(geometry["component_count"], 2)
        self.assertEqual([item["support_gaussian_count"] for item in records], [4, 3])
        self.assertEqual([item["accepted"] for item in records], [True, False])
        self.assertEqual(records[0]["independent_camera_count"], 2)


class SpatialCoreOwnershipTest(unittest.TestCase):
    @staticmethod
    def core(
        component_id: int,
        project_id: int,
        indices: list[int],
        counts: list[int],
    ) -> SpatialCoreSupport:
        return SpatialCoreSupport(
            component_id=component_id,
            source_component_id=component_id,
            project_id=project_id,
            indices=np.asarray(indices, dtype=np.uint32),
            camera_counts=np.asarray(counts, dtype=np.uint16),
            record={},
        )

    def test_unique_camera_maximum_wins_and_cross_class_tie_abstains(self) -> None:
        cores = [
            self.core(1, 9, [0, 1, 2], [2, 2, 2]),
            self.core(2, 15, [1, 2, 3], [1, 2, 3]),
        ]
        proposed, owners, overlap, cross_overlap, ties, exclusive = (
            resolve_spatial_core_ownership(cores, 4)
        )
        np.testing.assert_array_equal(proposed, [True, True, False, True])
        np.testing.assert_array_equal(owners, [1, 2, 2, 1])
        np.testing.assert_array_equal(overlap, [False, True, True, False])
        np.testing.assert_array_equal(
            cross_overlap, [False, True, True, False]
        )
        np.testing.assert_array_equal(ties, [False, False, True, False])
        np.testing.assert_array_equal(exclusive[1][0], [0, 1])
        np.testing.assert_array_equal(exclusive[2][0], [3])

    def test_later_stronger_support_clears_an_earlier_tie(self) -> None:
        cores = [
            self.core(1, 9, [0], [2]),
            self.core(2, 15, [0], [2]),
            self.core(3, 15, [0], [3]),
        ]
        proposed, _owners, _overlap, _cross, ties, exclusive = (
            resolve_spatial_core_ownership(cores, 1)
        )
        np.testing.assert_array_equal(proposed, [True])
        np.testing.assert_array_equal(ties, [False])
        np.testing.assert_array_equal(exclusive[3][0], [0])


class MultiviewSpatialCoreEndToEndTest(unittest.TestCase):
    def test_main_writes_only_report_masks_and_sparse_supports(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        proposals = [
            proposal(1, 0, [0, 1, 2, 3]),
            proposal(2, 1, [0, 1, 2, 3]),
            proposal(3, 2, [0, 1, 2, 3]),
        ]
        record = source_record(proposals)
        record["status"] = "accepted_stable_multiview_identity"
        record["accepted"] = True
        manifest_proposals = [
            {
                "proposal_id": item.proposal_id,
                "frame_file": item.frame_file,
            }
            for item in proposals
        ]
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            source_ply = temp_dir / "source.ply"
            source_ply.write_bytes(b"placeholder")
            proposal_manifest_path = temp_dir / "proposal_manifest.json"
            proposal_manifest_path.write_text("{}", encoding="utf-8")
            component_report_path = temp_dir / "component_report.json"
            component_report_path.write_text(
                json.dumps(
                    {
                        "contract": "report_only_class_agnostic_3d_association_v1",
                        "v5_used": False,
                        "dinov2_used": False,
                        "semantic_identity_hardened_before_3d": False,
                        "outputs": {"report_only": True},
                        "proposal_count": 3,
                        "components": [record],
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "vertex_count": 4,
                "ply_path": str(source_ply),
                "proposals": manifest_proposals,
            }
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
            vertices = np.zeros((4,), dtype=dtype)
            vertices["x"] = [0.00, 0.01, 0.02, 0.03]
            for name in ("scale_0", "scale_1", "scale_2"):
                vertices[name] = np.log(0.005)
            header = SimpleNamespace(elements=[SimpleNamespace(count=4)])
            output_dir = temp_dir / "output"
            argv = [
                "audit_multiview_spatial_core",
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
            module = "scripts.task1.dinov3.audit_multiview_spatial_core"
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(
                    f"{module}.load_query_proposals",
                    return_value=(proposals, manifest),
                ),
                mock.patch(
                    f"{module}.vertex_data_memmap",
                    return_value=(header, vertices),
                ),
            ):
                main()

            report = json.loads(
                (output_dir / "multiview_spatial_core_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["proposed_core_gaussian_count"], 4)
            self.assertFalse(report["semantic_labels_written"])
            self.assertTrue((output_dir / "proposed_core_mask.npy").is_file())
            self.assertTrue((output_dir / "proposed_core_supports.npz").is_file())
            self.assertFalse((output_dir / "gaussian_labels.npy").exists())
            self.assertFalse((output_dir / "gaussian_project_class_ids.npy").exists())
            self.assertFalse((output_dir / "label_map.json").exists())
            self.assertEqual(list(output_dir.glob("*.ply")), [])
            self.assertEqual(ontology.class_count, 150)


class MultiviewSpatialCoreSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")
        cls.renderer = RENDERER.read_text(encoding="utf-8")

    def test_scheduler_reuses_only_cached_targeted_report(self) -> None:
        self.assertIn("SOURCE_COMPONENT_REPORT", self.text)
        self.assertIn("SOURCE_PROPOSAL_MANIFEST", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("run_flashsplat_mask_proposals", self.text)
        self.assertIn("inference_rerun=0", self.text)
        self.assertIn("flashsplat_rerun=0", self.text)

    def test_scheduler_exposes_global_spatial_core_defaults(self) -> None:
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

    def test_scheduler_is_fresh_report_only_and_forbids_semantic_outputs(self) -> None:
        self.assertIn("report_only=1", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("semantic_project_class_arrays_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn("refuses RESET_OUTPUT=1", self.text)
        self.assertIn('test ! -e "${AUDIT_DIR}/gaussian_labels.npy"', self.text)
        self.assertIn('test ! -e "${AUDIT_DIR}/label_map.json"', self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)

    def test_renderer_accepts_the_spatial_core_contract(self) -> None:
        self.assertIn(
            "report_only_dinov3_multiview_spatial_core_v1",
            self.renderer,
        )
        self.assertIn('"visualization_components"', self.renderer)


if __name__ == "__main__":
    unittest.main()
