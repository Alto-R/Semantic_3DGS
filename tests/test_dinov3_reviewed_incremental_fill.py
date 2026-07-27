from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.task1.dinov3.materialize_reviewed_incremental_fill import (
    FillComponent,
    build_candidate_labels,
    load_review_config,
    main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"
DRJOHNSON_CONFIG = (
    PROJECT_ROOT / "configs" / "dinov3_incremental_fill_review.drjohnson.json"
)
PLAYROOM_CONFIG = (
    PROJECT_ROOT / "configs" / "dinov3_incremental_fill_review.playroom.json"
)
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_reviewed_incremental_fill_scene.sbatch"
)


def component(
    component_id: int,
    project_id: int,
    class_name: str,
    indices: list[int],
) -> FillComponent:
    array = np.asarray(indices, dtype=np.uint32)
    return FillComponent(
        component_id=component_id,
        project_id=project_id,
        class_name=class_name,
        indices=array,
        camera_counts=np.full(array.shape, 2, dtype=np.uint16),
        record={
            "component_id": component_id,
            "project_id": project_id,
            "class": class_name,
            "accepted": True,
            "support_gaussian_count": len(indices),
        },
    )


class ReviewedSelectionTest(unittest.TestCase):
    def test_candidate_fills_only_zero_labels_and_applies_exclusion(self) -> None:
        preferred = np.asarray([1, 0, 0, 0], dtype=np.int32)
        components = [
            component(1, 15, "door", [1, 2]),
            component(2, 20, "chair", [3]),
        ]
        config = {
            "scene": "synthetic",
            "source_contract": (
                "report_only_dinov3_incremental_spatial_core_fill_v1"
            ),
            "default_decision": "accept",
            "excluded_components": [
                {
                    "component_id": 2,
                    "class": "chair",
                    "reason": "Synthetic exact-QA rejection.",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir_value:
            path = Path(temp_dir_value) / "review.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            exclusions, _raw = load_review_config(path, scene="synthetic")
        final, fill, excluded, ids, included, rejected = build_candidate_labels(
            preferred,
            components,
            exclusions,
        )
        np.testing.assert_array_equal(final, [1, 15, 15, 0])
        np.testing.assert_array_equal(fill, [False, True, True, False])
        np.testing.assert_array_equal(excluded, [False, False, False, True])
        np.testing.assert_array_equal(ids, [0, 1, 1, 0])
        self.assertEqual([item.component_id for item in included], [1])
        self.assertEqual([item.component_id for item in rejected], [2])

    def test_candidate_rejects_any_overlap_with_preferred_nonzero_label(
        self,
    ) -> None:
        preferred = np.asarray([1, 15, 0], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "preferred nonzero"):
            build_candidate_labels(
                preferred,
                [component(1, 15, "door", [1, 2])],
                {},
            )


class ReviewedSceneConfigTest(unittest.TestCase):
    def test_drjohnson_excludes_all_observed_painting_halo_components(self) -> None:
        exclusions, raw = load_review_config(
            DRJOHNSON_CONFIG,
            scene="drjohnson",
        )
        self.assertEqual(
            sorted(exclusions),
            [1, 3, 13, 21, 24, 29, 30, 37, 40, 47, 49],
        )
        self.assertEqual(
            {item.class_name for item in exclusions.values()},
            {"painting"},
        )
        self.assertEqual(raw["default_decision"], "accept")

    def test_playroom_excludes_door_halo_and_misidentified_hanging_item(
        self,
    ) -> None:
        exclusions, _raw = load_review_config(
            PLAYROOM_CONFIG,
            scene="playroom",
        )
        self.assertEqual(
            {
                component_id: decision.class_name
                for component_id, decision in exclusions.items()
            },
            {54: "door", 105: "lamp"},
        )


class ReviewedMaterializationEndToEndTest(unittest.TestCase):
    def test_main_writes_fresh_candidate_and_preserves_preferred_nonzero_labels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            preferred_path = temp_dir / "preferred.npy"
            preferred = np.asarray([1, 0, 0, 0], dtype=np.int32)
            np.save(preferred_path, preferred)
            preferred_bytes = preferred_path.read_bytes()
            preferred_map_path = temp_dir / "preferred_label_map.json"
            preferred_map_path.write_text(
                json.dumps(
                    {
                        "scene": "synthetic",
                        "source": "synthetic_preferred_v2",
                        "labels": [
                            {
                                "id": 0,
                                "name": "unlabeled",
                                "class": "unlabeled",
                                "type": "unlabeled",
                            },
                            {
                                "id": 1,
                                "name": "wall",
                                "class": "wall",
                                "project_id": 1,
                                "ade_id": 0,
                                "type": "stuff",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit_path = temp_dir / "audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "contract": (
                            "report_only_dinov3_incremental_spatial_core_fill_v1"
                        ),
                        "scene": "synthetic",
                        "report_only": True,
                        "preferred_labels_read_only": True,
                        "preferred_labels_modified": False,
                        "vertex_count": 4,
                        "accepted_residual_component_count": 2,
                        "incremental_fill_gaussian_count": 3,
                        "components": [
                            {
                                "component_id": 1,
                                "project_id": 15,
                                "class": "door",
                                "accepted": True,
                                "support_gaussian_count": 2,
                            },
                            {
                                "component_id": 2,
                                "project_id": 20,
                                "class": "chair",
                                "accepted": True,
                                "support_gaussian_count": 1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            supports_path = temp_dir / "supports.npz"
            np.savez_compressed(
                supports_path,
                component_000001_indices=np.asarray([1, 2], dtype=np.uint32),
                component_000001_camera_counts=np.asarray(
                    [2, 3], dtype=np.uint16
                ),
                component_000002_indices=np.asarray([3], dtype=np.uint32),
                component_000002_camera_counts=np.asarray([2], dtype=np.uint16),
            )
            config_path = temp_dir / "review.json"
            config_path.write_text(
                json.dumps(
                    {
                        "scene": "synthetic",
                        "source_contract": (
                            "report_only_dinov3_incremental_spatial_core_fill_v1"
                        ),
                        "default_decision": "accept",
                        "excluded_components": [
                            {
                                "component_id": 2,
                                "class": "chair",
                                "reason": "Synthetic exact-QA rejection.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source_ply = temp_dir / "source.ply"
            source_ply.write_bytes(b"placeholder")
            output_dir = temp_dir / "output"
            argv = [
                "materialize_reviewed_incremental_fill",
                "--preferred-labels",
                str(preferred_path),
                "--preferred-label-map",
                str(preferred_map_path),
                "--incremental-fill-report",
                str(audit_path),
                "--incremental-fill-supports",
                str(supports_path),
                "--review-config",
                str(config_path),
                "--ontology",
                str(ONTOLOGY),
                "--source-ply",
                str(source_ply),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
            ]
            module = (
                "scripts.task1.dinov3.materialize_reviewed_incremental_fill"
            )

            def fake_write_ply(
                _source: Path,
                output: Path,
                _labels: np.ndarray,
            ) -> None:
                output.write_bytes(b"semantic")

            header = SimpleNamespace(
                elements=[SimpleNamespace(name="vertex", count=4)]
            )
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(f"{module}.read_ply_header", return_value=header),
                mock.patch(
                    f"{module}.write_ply_with_labels",
                    side_effect=fake_write_ply,
                ),
            ):
                main()

            final = np.load(output_dir / "gaussian_labels.npy")
            np.testing.assert_array_equal(final, [1, 15, 15, 0])
            self.assertEqual(preferred_path.read_bytes(), preferred_bytes)
            label_map = json.loads(
                (output_dir / "label_map.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {int(item["id"]) for item in label_map["labels"]},
                {0, 1, 15},
            )
            summary = json.loads(
                (
                    output_dir / "reviewed_incremental_fill_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(summary["candidate_only"])
            self.assertTrue(summary["preferred_nonzero_labels_preserved"])
            self.assertEqual(summary["reviewed_fill_gaussian_count"], 2)
            self.assertEqual(summary["reviewed_excluded_gaussian_count"], 1)
            self.assertTrue((output_dir / "semantic_point_cloud.ply").is_file())


class ReviewedIncrementalFillSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_uses_reviewed_cached_sources_only(self) -> None:
        self.assertIn("PREFERRED_LABELS", self.text)
        self.assertIn("INCREMENTAL_FILL_REPORT", self.text)
        self.assertIn("INCREMENTAL_FILL_SUPPORTS", self.text)
        self.assertIn("REVIEW_CONFIG", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("run_flashsplat_mask_proposals", self.text)
        self.assertIn("inference_rerun=0", self.text)
        self.assertIn("flashsplat_rerun=0", self.text)

    def test_scheduler_is_fresh_candidate_and_preserves_v2(self) -> None:
        self.assertIn("candidate_only=1", self.text)
        self.assertIn("preferred_labels_read_only=1", self.text)
        self.assertIn("fill_only_preferred_zero_gaussians", self.text)
        self.assertIn("refuses RESET_OUTPUT=1", self.text)

    def test_scheduler_writes_full_candidate_qa(self) -> None:
        self.assertIn("validate_task1_outputs", self.text)
        self.assertIn("export_supersplat_label_colors", self.text)
        self.assertIn("render_auto_label_overlays", self.text)
        self.assertIn("measure_overlay_coverage", self.text)
        self.assertIn("make_contact_sheet", self.text)
        self.assertIn("write_semantic_color_legend", self.text)
        self.assertIn("semantic_point_cloud.ply", self.text)
        self.assertIn("QA_CAMERA_INDICES", self.text)


if __name__ == "__main__":
    unittest.main()
