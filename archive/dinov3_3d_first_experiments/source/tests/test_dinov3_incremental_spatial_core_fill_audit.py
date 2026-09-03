from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.task1.dinov3.associate_3d_query_regions import QueryProposal
from scripts.task1.dinov3.audit_incremental_spatial_core_fill import (
    IncrementalFillThresholds,
    load_resolved_core_supports,
    main,
    split_incremental_residual,
)
from scripts.task1.dinov3.render_incremental_spatial_core_fill import (
    build_exact_residual_colors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_incremental_spatial_core_fill_audit_scene.sbatch"
)
RENDERER = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "render_incremental_spatial_core_fill.py"
)
REPORT_CONTRACT = "report_only_dinov3_incremental_spatial_core_fill_v1"


def proposal(
    proposal_id: int,
    camera_index: int,
    indices: list[int] | np.ndarray,
) -> QueryProposal:
    return QueryProposal(
        proposal_id=proposal_id,
        frame_file=f"camera_{camera_index:04d}.png",
        camera_index=camera_index,
        region_id=proposal_id,
        indices=np.asarray(indices, dtype=np.uint32),
        counts=np.ones((len(indices),), dtype=np.float32),
        class_probabilities=np.asarray([1.0], dtype=np.float32),
        no_object_probability=0.0,
        query_embedding=np.asarray([1.0], dtype=np.float32),
        quality=1.0,
        metadata={},
    )


def spatial_report(vertex_count: int = 6) -> dict[str, object]:
    return {
        "contract": "report_only_dinov3_multiview_spatial_core_v1",
        "report_only": True,
        "vertex_count": vertex_count,
        "components": [
            {
                "component_id": 1,
                "source_component_id": 7,
                "accepted": True,
                "class": "door",
                "project_id": 9,
                "source_proposal_ids": [1, 2],
            }
        ],
    }


class ResolvedCoreLoadingTest(unittest.TestCase):
    def test_loads_exclusive_post_ownership_support(self) -> None:
        archive = {
            "component_000001_indices": np.asarray([0, 2, 4], dtype=np.uint32),
            "component_000001_camera_counts": np.asarray(
                [2, 3, 2], dtype=np.uint16
            ),
        }
        supports = load_resolved_core_supports(spatial_report(), archive, 6)
        self.assertEqual(len(supports), 1)
        np.testing.assert_array_equal(supports[0].indices, [0, 2, 4])
        np.testing.assert_array_equal(supports[0].camera_counts, [2, 3, 2])

    def test_rejects_nonexclusive_resolved_supports(self) -> None:
        report = spatial_report()
        report["components"].append(
            {
                "component_id": 2,
                "source_component_id": 8,
                "accepted": True,
                "class": "wall",
                "project_id": 15,
                "source_proposal_ids": [3, 4],
            }
        )
        archive = {
            "component_000001_indices": np.asarray([0, 1], dtype=np.uint32),
            "component_000001_camera_counts": np.asarray([2, 2], dtype=np.uint16),
            "component_000002_indices": np.asarray([1, 2], dtype=np.uint32),
            "component_000002_camera_counts": np.asarray([2, 2], dtype=np.uint16),
        }
        with self.assertRaisesRegex(ValueError, "not exclusive"):
            load_resolved_core_supports(report, archive, 6)


class IncrementalResidualGeometryTest(unittest.TestCase):
    def test_applies_only_global_size_and_independent_camera_gates(self) -> None:
        indices = np.arange(5, dtype=np.uint32)
        counts = np.full((5,), 2, dtype=np.uint16)
        points = np.column_stack(
            [np.arange(5, dtype=np.float64) * 0.01, np.zeros((5, 2))]
        )
        scales = np.full((5, 3), np.log(0.005), dtype=np.float64)
        records, median, voxel_size, geometry = split_incremental_residual(
            1,
            indices,
            counts,
            [proposal(1, 0, indices), proposal(2, 1, indices)],
            points,
            scales,
            IncrementalFillThresholds(min_spatial_component_gaussians=4),
        )
        self.assertAlmostEqual(median, 0.005)
        self.assertAlmostEqual(voxel_size, 0.02)
        self.assertEqual(geometry["component_count"], 1)
        self.assertTrue(records[0]["accepted"])
        self.assertEqual(records[0]["independent_camera_count"], 2)

    def test_rejects_residual_seen_by_only_one_independent_camera(self) -> None:
        indices = np.arange(4, dtype=np.uint32)
        counts = np.full((4,), 2, dtype=np.uint16)
        points = np.column_stack(
            [np.arange(4, dtype=np.float64) * 0.01, np.zeros((4, 2))]
        )
        scales = np.full((4, 3), np.log(0.005), dtype=np.float64)
        records, _median, _voxel_size, _geometry = split_incremental_residual(
            1,
            indices,
            counts,
            [proposal(1, 0, indices)],
            points,
            scales,
            IncrementalFillThresholds(min_spatial_component_gaussians=2),
        )
        self.assertFalse(records[0]["accepted"])
        self.assertEqual(
            records[0]["status"],
            "rejected_insufficient_independent_cameras",
        )


class ExactResidualColorTest(unittest.TestCase):
    def test_builds_colors_from_sparse_support_without_a_label_array(self) -> None:
        report = {
            "contract": REPORT_CONTRACT,
            "vertex_count": 5,
            "incremental_fill_gaussian_count": 3,
            "components": [
                {
                    "component_id": 1,
                    "accepted": True,
                    "class": "door",
                },
                {
                    "component_id": 2,
                    "accepted": True,
                    "class": "chair",
                },
            ],
        }
        supports = {
            "component_000001_indices": np.asarray([0, 2], dtype=np.uint32),
            "component_000002_indices": np.asarray([4], dtype=np.uint32),
        }
        colors, selected, palette = build_exact_residual_colors(
            5, report, supports
        )
        np.testing.assert_array_equal(selected, [True, False, True, False, True])
        self.assertTrue(np.all(colors[~selected] == 0.0))
        self.assertGreater(float(colors[selected].max()), 0.0)
        self.assertEqual([item["class"] for item in palette], ["chair", "door"])

    def test_rejects_overlapping_sparse_residuals(self) -> None:
        report = {
            "contract": REPORT_CONTRACT,
            "vertex_count": 3,
            "incremental_fill_gaussian_count": 3,
            "components": [
                {"component_id": 1, "accepted": True, "class": "door"},
                {"component_id": 2, "accepted": True, "class": "wall"},
            ],
        }
        supports = {
            "component_000001_indices": np.asarray([0, 1], dtype=np.uint32),
            "component_000002_indices": np.asarray([1, 2], dtype=np.uint32),
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_exact_residual_colors(3, report, supports)


class IncrementalFillEndToEndTest(unittest.TestCase):
    def test_main_preserves_preferred_labels_and_writes_report_only_supports(
        self,
    ) -> None:
        proposals = [
            proposal(1, 0, list(range(6))),
            proposal(2, 1, list(range(6))),
        ]
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            source_ply = temp_dir / "source.ply"
            source_ply.write_bytes(b"placeholder")
            preferred_labels_path = temp_dir / "preferred_labels.npy"
            preferred_labels = np.asarray([11, 0, 0, 0, 0, 0], dtype=np.int32)
            np.save(preferred_labels_path, preferred_labels)
            preferred_bytes = preferred_labels_path.read_bytes()

            report_path = temp_dir / "spatial_report.json"
            report_path.write_text(
                json.dumps(spatial_report()),
                encoding="utf-8",
            )
            support_path = temp_dir / "spatial_supports.npz"
            np.savez_compressed(
                support_path,
                component_000001_indices=np.arange(6, dtype=np.uint32),
                component_000001_camera_counts=np.full(
                    (6,), 2, dtype=np.uint16
                ),
            )
            proposal_manifest_path = temp_dir / "proposal_manifest.json"
            proposal_manifest_path.write_text("{}", encoding="utf-8")
            proposal_manifest = {
                "vertex_count": 6,
                "proposals": [
                    {
                        "proposal_id": item.proposal_id,
                        "frame_file": item.frame_file,
                    }
                    for item in proposals
                ],
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
            vertices = np.zeros((6,), dtype=dtype)
            vertices["x"] = np.arange(6, dtype=np.float32) * 0.01
            for name in ("scale_0", "scale_1", "scale_2"):
                vertices[name] = np.log(0.005)
            header = SimpleNamespace(elements=[SimpleNamespace(count=6)])
            output_dir = temp_dir / "output"
            argv = [
                "audit_incremental_spatial_core_fill",
                "--spatial-core-report",
                str(report_path),
                "--spatial-core-supports",
                str(support_path),
                "--proposal-manifest",
                str(proposal_manifest_path),
                "--preferred-labels",
                str(preferred_labels_path),
                "--source-ply",
                str(source_ply),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
                "--min-spatial-component-gaussians",
                "2",
            ]
            module = (
                "scripts.task1.dinov3.audit_incremental_spatial_core_fill"
            )
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(
                    f"{module}.load_query_proposals",
                    return_value=(proposals, proposal_manifest),
                ),
                mock.patch(
                    f"{module}.vertex_data_memmap",
                    return_value=(header, vertices),
                ),
            ):
                main()

            result = json.loads(
                (
                    output_dir / "incremental_spatial_core_fill_audit.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(result["incremental_fill_gaussian_count"], 5)
            self.assertEqual(
                result["resolved_core_preferred_overlap_gaussian_count"], 1
            )
            self.assertTrue(result["preferred_labels_read_only"])
            self.assertFalse(result["semantic_labels_written"])
            self.assertEqual(preferred_labels_path.read_bytes(), preferred_bytes)
            fill_mask = np.load(output_dir / "incremental_fill_mask.npy")
            np.testing.assert_array_equal(
                fill_mask, [False, True, True, True, True, True]
            )
            self.assertTrue(
                (output_dir / "incremental_fill_supports.npz").is_file()
            )
            self.assertFalse((output_dir / "gaussian_labels.npy").exists())
            self.assertFalse(
                (output_dir / "gaussian_project_class_ids.npy").exists()
            )
            self.assertFalse((output_dir / "label_map.json").exists())
            self.assertEqual(list(output_dir.glob("*.ply")), [])


class IncrementalFillSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")
        cls.renderer = RENDERER.read_text(encoding="utf-8")

    def test_scheduler_uses_cached_core_and_immutable_preferred_labels(self) -> None:
        self.assertIn("SPATIAL_CORE_REPORT", self.text)
        self.assertIn("SPATIAL_CORE_SUPPORTS", self.text)
        self.assertIn("PREFERRED_LABELS", self.text)
        self.assertIn("preferred_labels_read_only=1", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("run_flashsplat_mask_proposals", self.text)
        self.assertIn("inference_rerun=0", self.text)
        self.assertIn("flashsplat_rerun=0", self.text)

    def test_scheduler_exposes_only_global_residual_gates(self) -> None:
        for value in (
            'MIN_SPATIAL_COMPONENT_GAUSSIANS="${MIN_SPATIAL_COMPONENT_GAUSSIANS:-500}"',
            'MIN_SPATIAL_COMPONENT_CAMERAS="${MIN_SPATIAL_COMPONENT_CAMERAS:-2}"',
            'VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"',
            'MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"',
            'MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"',
        ):
            self.assertIn(value, self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)

    def test_scheduler_renders_exact_support_and_forbids_semantic_outputs(
        self,
    ) -> None:
        self.assertIn(
            "exact_projection_of_accepted_unlabeled_residual_3d_support",
            self.text,
        )
        self.assertIn("render_incremental_spatial_core_fill", self.text)
        self.assertIn("report_only=1", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("semantic_project_class_arrays_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn("refuses RESET_OUTPUT=1", self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)

    def test_renderer_reconstructs_colors_in_memory_from_sparse_supports(
        self,
    ) -> None:
        self.assertIn("build_exact_residual_colors", self.renderer)
        self.assertIn("support-npz", self.renderer)
        self.assertNotIn("labels-npy", self.renderer)
        self.assertIn(REPORT_CONTRACT, self.renderer)


if __name__ == "__main__":
    unittest.main()
