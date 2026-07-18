from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from build_ade_refinement_config import (  # noqa: E402
    augment_with_extensions,
    build_refinement_config,
)
from dinov2_ontology import load_ontology  # noqa: E402
from merge_ade_refinement import refine_ade_labels  # noqa: E402


class AdeRefinementConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.label_map = {
            "scene": "example",
            "labels": [
                {"id": 0, "class": "unlabeled", "gaussian_count": 1000},
                {
                    "id": 1,
                    "class": "windowpane",
                    "gaussian_count": 700,
                    "source_view_count": 8,
                },
                {
                    "id": 2,
                    "class": "bench",
                    "gaussian_count": 400,
                    "source_view_count": 20,
                },
                {
                    "id": 3,
                    "class": "bicycle",
                    "gaussian_count": 800,
                    "source_view_count": 1,
                },
            ],
        }

    def test_only_supported_present_ade_classes_are_selected(self) -> None:
        config = build_refinement_config(
            self.label_map,
            ROOT / "configs" / "ade20k_to_project.json",
            {"windowpane": ["window", "window pane"]},
            min_anchor_gaussians=500,
            min_source_views=2,
            max_prompt_words=20,
        )
        self.assertEqual(config["selected_refinement_classes"], ["windowpane"])
        self.assertFalse(config["missing_vocabulary_enabled"])
        self.assertEqual(config["classes"][0]["prompts"], ["window", "window pane", "windowpane"])
        rejected = {item["class"]: item["reasons"] for item in config["rejected_classes"]}
        self.assertIn("gaussian_count<500", rejected["bench"])
        self.assertIn("source_view_count<2", rejected["bicycle"])

    def test_prompt_budget_is_enforced_globally(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt vocabulary"):
            build_refinement_config(
                self.label_map,
                ROOT / "configs" / "ade20k_to_project.json",
                {"windowpane": ["large interior window"]},
                min_anchor_gaussians=500,
                min_source_views=2,
                max_prompt_words=1,
            )

    def test_missing_vocabulary_extension_is_added_to_unified_grounding_config(self) -> None:
        config = build_refinement_config(
            self.label_map,
            ROOT / "configs" / "ade20k_to_project.json",
            {"windowpane": ["window"]},
            min_anchor_gaussians=500,
            min_source_views=2,
            max_prompt_words=20,
        )
        unified = augment_with_extensions(
            config,
            {
                "classes": [
                    {
                        "class": "speaker",
                        "type": "thing",
                        "prompts": ["speaker", "floor speaker"],
                    }
                ]
            },
            {
                "scene": "example",
                "candidate_classes": ["speaker"],
                "default_enabled_classes": ["speaker"],
            },
            include_classes="",
            ontology_classes={
                item.project_class
                for item in load_ontology(ROOT / "configs" / "ade20k_to_project.json").classes
            },
            max_prompt_words=20,
        )
        self.assertTrue(unified["missing_vocabulary_enabled"])
        self.assertEqual(unified["selected_extension_classes"], ["speaker"])
        self.assertEqual(unified["classes"][-1]["role"], "extension")


class AdeRefinementMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_labels = np.asarray([1, 1, 2, 2, 3, 0, 1], dtype=np.int32)
        self.base_items = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            1: {"id": 1, "name": "bicycle_001", "class": "bicycle"},
            2: {"id": 2, "name": "bench_001", "class": "bench"},
            3: {"id": 3, "name": "wall", "class": "wall"},
        }
        self.grounding_labels = np.asarray([10, 10, 10, 10, 11, 11, 11], dtype=np.int32)
        self.grounding_items = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            10: {
                "id": 10,
                "name": "bench_01",
                "class": "bench",
                "proposal_count": 5,
                "source_view_count": 4,
            },
            11: {
                "id": 11,
                "name": "bicycle_01",
                "class": "bicycle",
                "proposal_count": 6,
                "source_view_count": 5,
            },
        }
        self.claim_count = np.asarray([1, 2, 1, 1, 1, 0, 1], dtype=np.uint16)
        self.robust = {
            "bench": np.asarray([0, 1, 2, 3], dtype=np.uint32),
            "bicycle": np.asarray([1, 4, 6], dtype=np.uint32),
        }

    def run_merge(self, min_anchor_gaussians: int = 1):
        return refine_ade_labels(
            self.base_labels,
            self.base_items,
            self.grounding_labels,
            self.grounding_items,
            ["bench", "bicycle"],
            self.claim_count,
            self.robust,
            min_anchor_gaussians=min_anchor_gaussians,
            min_anchor_coverage=0.30,
            min_group_proposals=2,
            min_group_source_views=2,
        )

    def test_unique_grounding_claims_refine_and_ambiguous_claims_do_not(self) -> None:
        merged, changes, appended, report = self.run_merge()
        self.assertEqual(merged.tolist(), [4, 1, 2, 2, 5, 0, 1])
        self.assertEqual(changes.tolist(), [4, 0, 0, 0, 5, 0, 0])
        self.assertEqual([item["class"] for item in appended], ["bench", "bicycle"])
        self.assertEqual(report["changed_gaussian_count"], 2)
        self.assertEqual(report["relabeled_count"], 2)
        self.assertEqual(report["newly_labeled_count"], 0)
        self.assertTrue(report["unchanged_outside_accepted_masks"])

    def test_global_anchor_threshold_rejects_underanchored_group(self) -> None:
        merged, changes, appended, report = self.run_merge(min_anchor_gaussians=2)
        self.assertEqual(merged.tolist(), [4, 1, 2, 2, 3, 0, 1])
        self.assertEqual(changes.tolist(), [4, 0, 0, 0, 0, 0, 0])
        self.assertEqual([item["class"] for item in appended], ["bench"])
        bicycle = next(item for item in report["groups"] if item["class"] == "bicycle")
        self.assertEqual(bicycle["status"], "rejected")
        self.assertIn("anchor_overlap<2", bicycle["reasons"])

    def test_missing_vocabulary_cannot_enter_refinement(self) -> None:
        with self.assertRaisesRegex(ValueError, "no DINO anchor class"):
            refine_ade_labels(
                self.base_labels,
                self.base_items,
                self.grounding_labels,
                self.grounding_items,
                ["train"],
                self.claim_count,
                {"train": np.zeros((0,), dtype=np.uint32)},
                min_anchor_gaussians=1,
                min_anchor_coverage=0.0,
                min_group_proposals=1,
                min_group_source_views=1,
            )

    def test_spatial_anchor_precision_rejects_disconnected_piggyback(self) -> None:
        base_labels = np.asarray([2, 2, 0, 2, *([1] * 10)], dtype=np.int32)
        grounding_labels = np.full(base_labels.shape, 10, dtype=np.int32)
        claim_count = np.ones(base_labels.shape, dtype=np.uint16)
        robust = {"bench": np.arange(base_labels.shape[0], dtype=np.uint32)}
        vertex = np.zeros(
            base_labels.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
            ],
        )
        vertex["x"][:3] = np.arange(3) * 0.05
        vertex["x"][3:] = 10.0 + np.arange(11) * 0.05
        for name in ("scale_0", "scale_1", "scale_2"):
            vertex[name] = -3.0

        merged, changes, appended, report = refine_ade_labels(
            base_labels,
            self.base_items,
            grounding_labels,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "bench_01",
                    "class": "bench",
                    "proposal_count": 5,
                    "source_view_count": 4,
                },
            },
            ["bench"],
            claim_count,
            robust,
            min_anchor_gaussians=1,
            min_anchor_coverage=0.30,
            min_group_proposals=2,
            min_group_source_views=2,
            min_anchor_precision=0.10,
            vertex_data=vertex,
        )
        self.assertEqual(merged[:3].tolist(), [2, 2, 4])
        self.assertEqual(merged[3:].tolist(), base_labels[3:].tolist())
        self.assertEqual(int(np.count_nonzero(changes)), 1)
        self.assertEqual([item["class"] for item in appended], ["bench"])
        group = report["groups"][0]
        self.assertEqual(group["spatial_anchor_guard"]["component_count"], 2)
        self.assertEqual(group["spatial_anchor_guard"]["kept_component_count"], 1)
        self.assertEqual(group["spatially_removed_gaussian_count"], 11)

    def test_grounding_stuff_does_not_overwrite_base_thing(self) -> None:
        merged, changes, appended, report = refine_ade_labels(
            np.asarray([1, 2, 0], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "cabinet_001", "class": "cabinet"},
                2: {"id": 2, "name": "ceiling", "class": "ceiling"},
            },
            np.asarray([10, 10, 10], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "ceiling",
                    "class": "ceiling",
                    "proposal_count": 5,
                    "source_view_count": 4,
                },
            },
            ["ceiling"],
            np.ones((3,), dtype=np.uint16),
            {"ceiling": np.arange(3, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=1.0,
            min_group_proposals=2,
            min_group_source_views=2,
            min_anchor_precision=0.10,
            class_kinds={"ceiling": "stuff", "cabinet": "thing"},
        )
        self.assertEqual(merged.tolist(), [1, 2, 3])
        self.assertEqual(changes.tolist(), [0, 0, 3])
        self.assertEqual([item["class"] for item in appended], ["ceiling"])
        self.assertEqual(report["groups"][0]["protected_base_thing_gaussian_count"], 1)

    @staticmethod
    def spatial_vertex(x_values: list[float]) -> np.ndarray:
        vertex = np.zeros(
            len(x_values),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
            ],
        )
        vertex["x"] = x_values
        for name in ("scale_0", "scale_1", "scale_2"):
            vertex[name] = -3.0
        return vertex

    def test_thing_anchor_precision_can_replace_low_instance_coverage(self) -> None:
        merged, _changes, _appended, report = refine_ade_labels(
            np.asarray([1, 1, 1, 1, 1, 2, 2], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "wardrobe_001", "class": "wardrobe"},
                2: {"id": 2, "name": "wall", "class": "wall"},
            },
            np.asarray([0, 0, 0, 10, 10, 10, 10], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "wardrobe_01",
                    "class": "wardrobe",
                    "proposal_count": 3,
                    "source_view_count": 3,
                },
            },
            ["wardrobe"],
            np.ones((7,), dtype=np.uint16),
            {"wardrobe": np.arange(3, 7, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=0.50,
            min_group_proposals=2,
            min_group_source_views=2,
            min_anchor_precision=0.10,
            class_kinds={"wardrobe": "thing", "wall": "stuff"},
        )
        self.assertEqual(merged.tolist(), [1, 1, 1, 1, 1, 3, 3])
        self.assertEqual(report["groups"][0]["acceptance_mode"], "thing_anchor_precision_fallback")

    def test_ambiguous_strong_thing_claims_follow_large_anchored_group(self) -> None:
        base = np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        merged, _changes, _appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "bench_001", "class": "bench"},
                2: {"id": 2, "name": "bicycle_001", "class": "bicycle"},
            },
            np.full(base.shape, 10, dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "bench_01",
                    "class": "bench",
                    "proposal_count": 5,
                    "source_view_count": 5,
                },
            },
            ["bench"],
            np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint16),
            {"bench": np.arange(10, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45]
            ),
            class_kinds={"bench": "thing", "bicycle": "thing"},
        )
        self.assertTrue(np.all(merged[6:] == 3))
        self.assertEqual(report["groups"][0]["ambiguous_strong_thing_added_count"], 4)

    def test_thing_anchor_envelope_trims_connected_stuff_leakage(self) -> None:
        base = np.asarray([1, 1, 1, 2, 2], dtype=np.int32)
        merged, _changes, _appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "fireplace_001", "class": "fireplace"},
                2: {"id": 2, "name": "wall", "class": "wall"},
            },
            np.full(base.shape, 10, dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "fireplace_01",
                    "class": "fireplace",
                    "proposal_count": 4,
                    "source_view_count": 4,
                },
            },
            ["fireplace"],
            np.ones(base.shape, dtype=np.uint16),
            {"fireplace": np.arange(5, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex([0.0, 0.1, 0.2, 0.3, 0.4]),
            class_kinds={"fireplace": "thing", "wall": "stuff"},
        )
        self.assertEqual(merged.tolist(), base.tolist())
        self.assertEqual(report["groups"][0]["anchor_envelope_removed_change_count"], 2)

    def test_repeated_thing_instance_can_use_anchored_geometry_prototype(self) -> None:
        base = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
        merged, _changes, _appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "door_001", "class": "door"},
                2: {"id": 2, "name": "wall", "class": "wall"},
            },
            np.asarray([10, 10, 10, 11, 11, 11], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "door_01",
                    "class": "door",
                    "proposal_count": 4,
                    "source_view_count": 4,
                },
                11: {
                    "id": 11,
                    "name": "door_02",
                    "class": "door",
                    "proposal_count": 4,
                    "source_view_count": 4,
                },
            },
            ["door"],
            np.ones(base.shape, dtype=np.uint16),
            {"door": np.arange(6, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex([0.0, 0.1, 0.2, 1.0, 1.1, 1.2]),
            class_kinds={"door": "thing", "wall": "stuff"},
        )
        self.assertTrue(np.all(merged[3:] == 3))
        self.assertEqual(report["groups"][1]["acceptance_mode"], "anchored_instance_prototype")

    def test_high_precision_fusion_halo_resolves_weak_thing_competition(self) -> None:
        base = np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        merged, _changes, _appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "truck_001", "class": "truck"},
                2: {"id": 2, "name": "car_001", "class": "car"},
            },
            np.full(base.shape, 10, dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "truck_01",
                    "class": "truck",
                    "proposal_count": 6,
                    "source_view_count": 6,
                },
            },
            ["truck"],
            np.ones(base.shape, dtype=np.uint16),
            {"truck": np.arange(6, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45]
            ),
            class_kinds={"truck": "thing", "car": "thing"},
        )
        self.assertTrue(np.all(merged[6:] == 3))
        self.assertEqual(report["groups"][0]["fusion_thing_halo_added_count"], 4)

    def test_strong_view_prototype_cannot_take_over_dominant_competing_thing(self) -> None:
        base = np.asarray([1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        merged, _changes, _appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "curtain_001", "class": "curtain"},
                2: {"id": 2, "name": "ottoman_001", "class": "ottoman"},
            },
            np.asarray([10, 10, 10, 11, 11, 11, 11], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "curtain_01",
                    "class": "curtain",
                    "proposal_count": 5,
                    "source_view_count": 5,
                },
                11: {
                    "id": 11,
                    "name": "curtain_02",
                    "class": "curtain",
                    "proposal_count": 5,
                    "source_view_count": 5,
                },
            },
            ["curtain"],
            np.ones(base.shape, dtype=np.uint16),
            {"curtain": np.arange(7, dtype=np.uint32)},
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex([0.0, 0.1, 0.2, 10.0, 20.0, 30.0, 40.0]),
            class_kinds={"curtain": "thing", "ottoman": "thing"},
        )
        self.assertEqual(merged.tolist(), base.tolist())
        self.assertEqual(report["groups"][1]["status"], "rejected")
        self.assertGreater(report["groups"][1]["dominant_competing_thing_fraction"], 0.50)

    def test_matching_prototype_can_resolve_multiclass_ambiguity(self) -> None:
        base = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
        merged, _changes, _appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "window_001", "class": "windowpane"},
                2: {"id": 2, "name": "picture_001", "class": "picture"},
            },
            np.asarray([10, 10, 10, 11, 11, 11], dtype=np.int32),
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "windowpane_01",
                    "class": "windowpane",
                    "proposal_count": 5,
                    "source_view_count": 5,
                },
                11: {
                    "id": 11,
                    "name": "windowpane_02",
                    "class": "windowpane",
                    "proposal_count": 6,
                    "source_view_count": 6,
                },
                12: {
                    "id": 12,
                    "name": "picture_01",
                    "class": "picture",
                    "proposal_count": 3,
                    "source_view_count": 3,
                },
            },
            ["windowpane", "picture"],
            np.asarray([1, 1, 1, 2, 2, 2], dtype=np.uint16),
            {
                "windowpane": np.arange(6, dtype=np.uint32),
                "picture": np.arange(3, 6, dtype=np.uint32),
            },
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex([0.0, 0.1, 0.2, 1.0, 1.1, 1.2]),
            class_kinds={"windowpane": "thing", "picture": "thing"},
        )
        self.assertTrue(np.all(merged[3:] == 3))
        self.assertTrue(
            report["groups"][1]["prototype_ambiguous_geometry_recovery"]
        )
        self.assertEqual(
            report["groups"][1]["prototype_recovered_ambiguous_count"], 3
        )
        self.assertEqual(
            report["groups"][1][
                "prototype_candidate_dominant_competing_thing_fraction"
            ],
            1.0,
        )

    def test_nested_smaller_thing_instance_is_taken_over_by_stronger_parent(self) -> None:
        base = np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        grounding = np.asarray([10, 10, 10, 10, 10, 10, 10, 10, 11, 11], dtype=np.int32)
        merged, _changes, appended, report = refine_ade_labels(
            base,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                1: {"id": 1, "name": "truck_001", "class": "truck"},
                2: {"id": 2, "name": "car_001", "class": "car"},
            },
            grounding,
            {
                0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
                10: {
                    "id": 10,
                    "name": "truck_01",
                    "class": "truck",
                    "proposal_count": 8,
                    "source_view_count": 8,
                },
                11: {
                    "id": 11,
                    "name": "car_01",
                    "class": "car",
                    "proposal_count": 2,
                    "source_view_count": 2,
                },
            },
            ["truck", "car"],
            np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 1, 1], dtype=np.uint16),
            {
                "truck": np.arange(8, dtype=np.uint32),
                "car": np.arange(6, 10, dtype=np.uint32),
            },
            min_anchor_gaussians=1,
            min_anchor_coverage=0.10,
            min_group_proposals=2,
            min_group_source_views=2,
            vertex_data=self.spatial_vertex(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45]
            ),
            class_kinds={"truck": "thing", "car": "thing"},
        )
        self.assertNotIn(2, merged[6:].tolist())
        self.assertEqual([item["class"] for item in appended], ["truck", "truck"])
        self.assertEqual(report["groups"][1]["status"], "nested_parent_takeover")
        self.assertEqual(report["groups"][1]["takeover_class"], "truck")


if __name__ == "__main__":
    unittest.main()
