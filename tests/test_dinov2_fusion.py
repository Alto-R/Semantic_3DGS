from __future__ import annotations

import unittest

import numpy as np


from scripts.task1.dinov2.dinov2_ontology import OntologyClass
from scripts.task1.dinov2.fuse_dinov2_multiview_votes import (
    CandidateGroup,
    adaptive_threshold,
    build_candidate_groups,
    finalize_groups,
)


class Dinov2FusionTest(unittest.TestCase):
    def test_final_ids_are_sorted_by_class_then_gaussian_index(self) -> None:
        class_one = OntologyClass(0, "wall", 1, "wall", "stuff")
        class_two = OntologyClass(1, "object", 2, "object", "thing")
        candidates = [
            CandidateGroup(1, class_two, np.asarray([4, 5]), 0.1, 2),
            CandidateGroup(2, class_one, np.asarray([2, 3]), None, 0),
            CandidateGroup(3, class_two, np.asarray([0, 1]), 0.1, 1),
        ]
        labels, kept, pruned = finalize_groups(
            candidates,
            total_views=2,
            minimum=0,
            minimum_thing=0,
            minimum_stuff=0,
            stuff_min_view_ratio=0.5,
            stuff_threshold_ratio=0.75,
            thing_floor_ratio=0.5,
            gaussian_count=6,
        )

        self.assertEqual([item.ontology_class.project_id for item in kept], [1, 2, 2])
        self.assertEqual(labels.tolist(), [2, 2, 1, 1, 3, 3])
        self.assertEqual(pruned, [])

    def test_adaptive_thing_threshold_respects_global_floor_ratio(self) -> None:
        item = OntologyClass(0, "object", 1, "object", "thing")
        candidate = CandidateGroup(1, item, np.arange(10), 0.1, 1)
        candidate.source_frames.update({f"view_{index}" for index in range(9)})

        threshold = adaptive_threshold(
            candidate,
            total_views=10,
            minimum=0,
            minimum_thing=5000,
            minimum_stuff=10000,
            stuff_min_view_ratio=0.5,
            stuff_threshold_ratio=0.75,
            thing_floor_ratio=0.5,
        )

        self.assertEqual(threshold, 2500)

    def test_thing_class_is_split_into_deterministic_spatial_components(self) -> None:
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ]
        vertices = np.zeros((6,), dtype=dtype)
        vertices["x"] = np.asarray([0.00, 0.01, 0.02, 1.00, 1.01, 1.02])
        for name in ("scale_0", "scale_1", "scale_2"):
            vertices[name] = np.log(0.01)
        ontology_class = OntologyClass(0, "object", 1, "object", "thing")

        candidates, temporary_labels, reports = build_candidate_groups(
            np.ones((6,), dtype=np.uint16),
            {1: ontology_class},
            vertices,
            voxel_multiplier=4.0,
            min_voxel_size=0.01,
            max_voxel_size=0.20,
            min_component_gaussians=1,
            min_component_ratio=0.0,
        )

        self.assertEqual([item.indices.tolist() for item in candidates], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(temporary_labels.tolist(), [1, 1, 1, 2, 2, 2])
        self.assertEqual(reports[0]["kept_component_count"], 2)


if __name__ == "__main__":
    unittest.main()
