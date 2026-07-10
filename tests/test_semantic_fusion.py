from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cluster_semantic_flashsplat_proposals import (  # noqa: E402
    SemanticGroup,
    consolidate_thing_instances,
)


def vertex_data(points: list[tuple[float, float, float]]) -> np.ndarray:
    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
    ]
    data = np.zeros((len(points),), dtype=dtype)
    xyz = np.asarray(points, dtype=np.float32)
    for axis, values in zip(("x", "y", "z"), xyz.T):
        data[axis] = values
    log_scale = np.float32(np.log(0.02))
    for axis in ("scale_0", "scale_1", "scale_2"):
        data[axis] = log_scale
    return data


def thing_group(group_id: int, indices: list[int]) -> SemanticGroup:
    return SemanticGroup(
        group_id=group_id,
        class_name="bicycle",
        indices=np.asarray(indices, dtype=np.uint32),
        proposal_ids=[group_id],
        source_frames={f"frame_{group_id}"},
        scores=[0.8],
        assigned_count=len(indices),
    )


class InstanceConsolidationTest(unittest.TestCase):
    def consolidate(
        self,
        labels: np.ndarray,
        groups: list[SemanticGroup],
        points: list[tuple[float, float, float]],
    ) -> tuple[np.ndarray, list[SemanticGroup], list[dict[str, object]]]:
        return consolidate_thing_instances(
            labels,
            groups,
            vertex_data(points),
            voxel_scale_multiplier=4.0,
            min_voxel_size=0.10,
            max_voxel_size=0.20,
            min_component_gaussians=1,
            min_component_ratio=0.0,
        )

    def test_adjacent_same_class_fragments_merge(self) -> None:
        labels = np.asarray([1, 1, 2, 2], dtype=np.int32)
        groups = [thing_group(1, [0, 1]), thing_group(2, [2, 3])]
        points = [(0.00, 0.0, 0.0), (0.02, 0.0, 0.0), (0.11, 0.0, 0.0), (0.13, 0.0, 0.0)]

        consolidated, output_groups, reports = self.consolidate(labels, groups, points)

        self.assertEqual(len(output_groups), 1)
        self.assertEqual(np.unique(consolidated).shape[0], 1)
        self.assertEqual(reports[0]["before_group_count"], 2)
        self.assertEqual(reports[0]["after_instance_count"], 1)
        self.assertEqual(reports[0]["instances"][0]["source_label_ids"], [1, 2])

    def test_disconnected_same_class_objects_keep_separate_ids(self) -> None:
        labels = np.asarray([1, 1, 2, 2], dtype=np.int32)
        groups = [thing_group(1, [0, 1]), thing_group(2, [2, 3])]
        points = [(0.00, 0.0, 0.0), (0.02, 0.0, 0.0), (1.00, 0.0, 0.0), (1.02, 0.0, 0.0)]

        consolidated, output_groups, reports = self.consolidate(labels, groups, points)

        self.assertEqual(len(output_groups), 2)
        self.assertEqual(np.unique(consolidated).shape[0], 2)
        self.assertEqual(reports[0]["after_instance_count"], 2)

    def test_disconnected_parts_of_one_accepted_instance_are_not_split(self) -> None:
        labels = np.asarray([1, 1, 1, 1], dtype=np.int32)
        groups = [thing_group(1, [0, 1, 2, 3])]
        points = [(0.00, 0.0, 0.0), (0.02, 0.0, 0.0), (1.00, 0.0, 0.0), (1.02, 0.0, 0.0)]

        consolidated, output_groups, reports = self.consolidate(labels, groups, points)

        self.assertEqual(len(output_groups), 1)
        self.assertEqual(np.unique(consolidated).shape[0], 1)
        self.assertEqual(reports[0]["before_gaussians"], 4)
        self.assertEqual(reports[0]["after_gaussians"], 4)
        self.assertEqual(reports[0]["removed_gaussians"], 0)


if __name__ == "__main__":
    unittest.main()
