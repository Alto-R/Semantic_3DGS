from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from apply_singleton_point_recovery import (  # noqa: E402
    PointPatch,
    apply_point_patches,
    confirmed_patch_indices,
    decision_with_anchor_target,
    evaluate_anchor_candidate_policy,
    evaluate_seed_policy,
    fresh_recovery_label_item,
    geometry_compatibility,
    intersect_confirmed_views,
    resolve_verification_view_manifest,
    robust_geometry_signature,
    update_label_map_counts,
)
from recover_cross_view_sam_masks import ProjectionPrompt  # noqa: E402


LABEL_ITEMS = {
    0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
    1: {"id": 1, "name": "door_01", "class": "door"},
    2: {"id": 2, "name": "wall_01", "class": "wall"},
    3: {"id": 3, "name": "cabinet_01", "class": "cabinet"},
    4: {"id": 4, "name": "floor_01", "class": "floor"},
    5: {"id": 5, "name": "door_02", "class": "door"},
}
KINDS = {"door": "thing", "wall": "stuff", "cabinet": "thing", "floor": "stuff"}


def policy(labels: np.ndarray):
    return evaluate_seed_policy(
        np.arange(labels.size, dtype=np.uint32),
        labels,
        LABEL_ITEMS,
        KINDS,
        "door",
        min_breadcrumb_fraction=0.10,
        max_breadcrumb_fraction=0.50,
        max_competing_thing_fraction=0.05,
        min_recoverable_background_fraction=0.50,
    )


class SeedPolicyTest(unittest.TestCase):
    def test_partial_same_class_breadcrumb_with_background_is_accepted(self) -> None:
        labels = np.asarray([1] * 20 + [2] * 70 + [0] * 10, dtype=np.int32)

        decision = policy(labels)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.target_label_id, 1)
        self.assertAlmostEqual(decision.metrics["breadcrumb_fraction"], 0.20)
        self.assertEqual(int(decision.patchable_local.sum()), 80)

    def test_missing_breadcrumb_is_rejected(self) -> None:
        labels = np.asarray([2] * 90 + [0] * 10, dtype=np.int32)

        decision = policy(labels)

        self.assertFalse(decision.accepted)
        self.assertIn("same_class_breadcrumb_missing", decision.reasons)

    def test_competing_thing_protects_candidate(self) -> None:
        labels = np.asarray([1] * 20 + [2] * 69 + [3] * 11, dtype=np.int32)

        decision = policy(labels)

        self.assertFalse(decision.accepted)
        self.assertIn("competing_thing_fraction>0.05", decision.reasons)

    def test_other_stuff_class_is_not_patchable(self) -> None:
        labels = np.asarray([1] * 20 + [2] * 50 + [4] * 30, dtype=np.int32)

        decision = policy(labels)

        self.assertTrue(decision.accepted)
        self.assertEqual(int(decision.patchable_local.sum()), 50)
        self.assertFalse(decision.patchable_local[70:].any())

    def test_dominant_same_class_instance_becomes_patch_target(self) -> None:
        labels = np.asarray([1] * 5 + [5] * 15 + [2] * 80, dtype=np.int32)

        decision = policy(labels)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.target_label_id, 5)


class PointIntersectionTest(unittest.TestCase):
    def test_alternate_mask_can_only_keep_original_patchable_seed_points(self) -> None:
        baseline = np.asarray([1, 2, 2, 4, 0, 3], dtype=np.int32)
        seed_indices = np.arange(baseline.size, dtype=np.uint32)
        decision = evaluate_seed_policy(
            seed_indices,
            baseline,
            LABEL_ITEMS,
            KINDS,
            "door",
            min_breadcrumb_fraction=0.10,
            max_breadcrumb_fraction=0.50,
            max_competing_thing_fraction=0.20,
            min_recoverable_background_fraction=0.40,
        )
        prompt = ProjectionPrompt(
            seed_proposal_id=7,
            seed_key="frame.png:0",
            class_name="door",
            target_frame_file="alternate.png",
            target_camera_index=2,
            bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
            projected_xy=np.zeros((5, 2), dtype=np.float64),
            projected_seed_positions=np.asarray([0, 1, 2, 3, 4]),
            projected_gaussian_count=5,
            projected_fraction=5.0 / 6.0,
            baseline_depth_ratio=0.1,
            seed_grounding_score=0.9,
            seed_sam_score=0.95,
            seed_frame_file="frame.png",
        )

        confirmed, detail = confirmed_patch_indices(
            seed_indices,
            decision,
            prompt,
            np.asarray([True, True, False, True, True]),
        )

        # Wall index 1 and unlabeled index 4 pass. Floor 3 is protected, wall
        # index 2 fails the alternate mask, and cabinet 5 was not projected.
        self.assertEqual(confirmed.tolist(), [1, 4])
        self.assertEqual(detail["confirmed_patch_gaussian_count"], 2)

    def test_patch_changes_nothing_outside_confirmed_indices(self) -> None:
        baseline = np.asarray([2, 2, 2, 4, 3, 0], dtype=np.int32)
        patch = PointPatch(
            seed_key="seed:0",
            target_label_id=1,
            indices=np.asarray([1, 5], dtype=np.uint32),
        )

        merged, changes, report, per_seed = apply_point_patches(baseline, [patch])

        self.assertEqual(merged.tolist(), [2, 1, 2, 4, 3, 1])
        self.assertEqual(changes.tolist(), [0, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(merged[[0, 2, 3, 4]], baseline[[0, 2, 3, 4]])
        self.assertTrue(report["unchanged_outside_patch"])
        self.assertEqual(per_seed["seed:0"], 2)

    def test_conflicting_target_claims_are_left_unchanged(self) -> None:
        baseline = np.asarray([2, 2, 2], dtype=np.int32)
        patches = [
            PointPatch("door:a", 1, np.asarray([0, 1], dtype=np.uint32)),
            PointPatch("door:b", 5, np.asarray([1, 2], dtype=np.uint32)),
        ]

        merged, _changes, report, per_seed = apply_point_patches(baseline, patches)

        self.assertEqual(merged.tolist(), [1, 2, 5])
        self.assertEqual(report["ambiguous_gaussian_count"], 1)
        self.assertEqual(per_seed, {"door:a": 1, "door:b": 1})

    def test_tier2_uses_intersection_not_union_of_confirmed_views(self) -> None:
        confirmed = intersect_confirmed_views(
            [
                np.asarray([1, 2, 3, 4], dtype=np.uint32),
                np.asarray([2, 3, 4, 5], dtype=np.uint32),
                np.asarray([3, 4, 6], dtype=np.uint32),
            ],
            required_views=2,
        )

        self.assertEqual(confirmed.tolist(), [2, 3, 4])
        self.assertNotIn(1, confirmed)
        self.assertNotIn(5, confirmed)

    def test_tier2_refuses_to_intersect_too_few_views(self) -> None:
        confirmed = intersect_confirmed_views(
            [np.asarray([1, 2], dtype=np.uint32)],
            required_views=2,
        )

        self.assertEqual(confirmed.size, 0)


class AnchorCandidatePolicyTest(unittest.TestCase):
    def test_zero_breadcrumb_clean_background_can_wait_for_anchor(self) -> None:
        labels = np.asarray([2] * 95 + [0] * 5, dtype=np.int32)
        decision = policy(labels)

        reasons = evaluate_anchor_candidate_policy(
            decision,
            tier1_min_breadcrumb_fraction=0.10,
            max_breadcrumb_fraction=0.50,
            max_competing_thing_fraction=0.01,
            min_recoverable_background_fraction=0.90,
        )

        self.assertEqual(reasons, ())
        anchored = decision_with_anchor_target(decision, target_label_id=5)
        self.assertTrue(anchored.accepted)
        self.assertEqual(anchored.target_label_id, 5)
        self.assertTrue(anchored.metrics["requires_fresh_instance_label"])
        np.testing.assert_array_equal(
            anchored.patchable_local, decision.patchable_local
        )

    def test_small_candidate_breadcrumb_keeps_its_own_instance_id(self) -> None:
        labels = np.asarray([5] * 5 + [2] * 95, dtype=np.int32)
        decision = policy(labels)

        anchored = decision_with_anchor_target(decision, target_label_id=1)

        self.assertEqual(anchored.target_label_id, 5)
        self.assertFalse(anchored.metrics["requires_fresh_instance_label"])
        self.assertEqual(anchored.metrics["anchor_target_label_id"], 1)

    def test_competing_thing_blocks_anchor_candidate(self) -> None:
        labels = np.asarray([2] * 89 + [3] * 11, dtype=np.int32)
        decision = policy(labels)

        reasons = evaluate_anchor_candidate_policy(
            decision,
            tier1_min_breadcrumb_fraction=0.10,
            max_breadcrumb_fraction=0.50,
            max_competing_thing_fraction=0.01,
            min_recoverable_background_fraction=0.80,
        )

        self.assertIn("competing_thing_fraction>0.01", reasons)


class RecoveryInstanceLabelTest(unittest.TestCase):
    def test_zero_breadcrumb_recovery_gets_fresh_same_class_instance(self) -> None:
        item = fresh_recovery_label_item(
            LABEL_ITEMS,
            anchor_label_id=1,
            new_label_id=6,
            seed_key="frame.png:7",
        )

        self.assertEqual(item["id"], 6)
        self.assertEqual(item["class"], "door")
        self.assertNotEqual(item["name"], LABEL_ITEMS[1]["name"])
        self.assertEqual(item["source_label_id"], 1)

        output_map = update_label_map_counts(
            {"labels": list(LABEL_ITEMS.values())},
            np.asarray([0, 1, 2, 6, 6], dtype=np.int32),
            "test",
            [item],
        )
        added = next(value for value in output_map["labels"] if value["id"] == 6)
        self.assertEqual(added["gaussian_count"], 2)


class GeometryCompatibilityTest(unittest.TestCase):
    @staticmethod
    def box_points(extents: tuple[float, float, float]) -> np.ndarray:
        x = np.linspace(-extents[0] / 2.0, extents[0] / 2.0, 9)
        y = np.linspace(-extents[1] / 2.0, extents[1] / 2.0, 7)
        z = np.linspace(-extents[2] / 2.0, extents[2] / 2.0, 3)
        return np.asarray(
            [(a, b, c) for a in x for b in y for c in z],
            dtype=np.float64,
        )

    def test_rotated_scaled_same_shape_is_compatible(self) -> None:
        anchor_points = self.box_points((2.0, 1.0, 0.1))
        candidate_points = self.box_points((3.0, 1.5, 0.15))
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        candidate_points = candidate_points @ rotation.T

        compatible, metrics = geometry_compatibility(
            robust_geometry_signature(candidate_points),
            robust_geometry_signature(anchor_points),
            max_size_ratio=3.0,
            max_shape_ratio=2.5,
        )

        self.assertTrue(compatible)
        self.assertLess(metrics["size_ratio"], 2.0)
        self.assertLess(metrics["shape_ratio"], 1.1)

    def test_incompatible_shape_is_rejected(self) -> None:
        anchor = robust_geometry_signature(self.box_points((2.0, 1.0, 0.1)))
        candidate = robust_geometry_signature(self.box_points((5.0, 0.2, 0.1)))

        compatible, metrics = geometry_compatibility(
            candidate,
            anchor,
            max_size_ratio=3.0,
            max_shape_ratio=2.5,
        )

        self.assertFalse(compatible)
        self.assertGreater(metrics["shape_ratio"], 2.5)


class VerificationViewManifestTest(unittest.TestCase):
    def test_explicit_full_view_manifest_overrides_targeted_grounding_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grounded_path = root / "grounded_sam_manifest.json"
            full_path = root / "view_manifest.json"
            full_manifest = {
                "camera_count": 3,
                "frames": [
                    {"file": f"frame_{index}.png", "camera_index": index}
                    for index in range(3)
                ],
            }
            full_path.write_text(json.dumps(full_manifest), encoding="utf-8")

            resolved_path, resolved = resolve_verification_view_manifest(
                grounded_path,
                {"frames": [{"file": "frame_0.png", "camera_index": 0}]},
                full_path,
            )

            self.assertEqual(resolved_path, full_path)
            self.assertEqual(len(resolved["frames"]), 3)

    def test_grounded_subset_is_compatibility_fallback(self) -> None:
        grounded_path = Path("cache/grounded_sam_manifest.json")
        grounded_manifest = {
            "frames": [{"file": "frame_0.png", "camera_index": 0}]
        }

        resolved_path, resolved = resolve_verification_view_manifest(
            grounded_path, grounded_manifest, None
        )

        self.assertEqual(resolved_path, grounded_path)
        self.assertIs(resolved, grounded_manifest)


if __name__ == "__main__":
    unittest.main()
