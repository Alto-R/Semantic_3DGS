from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.audit_partial_anchor_dense_completion import (
    AuditThresholds,
    audit_partial_anchor_completions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_partial_anchor_audit_scene.sbatch"
)


def thresholds(**overrides: float | int) -> AuditThresholds:
    base = AuditThresholds(
        min_anchor_gaussians=2,
        min_anchor_coverage=0.80,
        min_anchor_precision=0.10,
        max_anchor_precision=0.25,
        min_robust_fraction=0.90,
        min_source_views=5,
        min_candidate_supporting_views=5,
        max_competing_thing_fraction=0.05,
        min_spatial_keep_fraction=0.95,
        max_partial_competing_thing_risk=0.40,
        voxel_scale_multiplier=4.0,
        min_voxel_size=1.0,
        max_voxel_size=1.0,
    )
    return replace(base, **overrides)


def item(
    label_id: int,
    project_id: int,
    class_name: str,
    *,
    source_views: int = 5,
) -> dict[str, object]:
    return {
        "id": label_id,
        "component_id": 100 + label_id,
        "project_id": project_id,
        "class": class_name,
        "type": "thing",
        "source_view_count": source_views,
    }


def geometry(count: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = np.arange(count, dtype=np.float64) + 0.25
    log_scales = np.full((count, 3), np.log(0.25), dtype=np.float64)
    return points, log_scales


def run_audit(
    *,
    seeds: np.ndarray,
    instances: np.ndarray,
    component_projects: np.ndarray,
    dense: np.ndarray,
    component_items: dict[int, dict[str, object]],
    points: np.ndarray | None = None,
    views: np.ndarray | None = None,
    audit_thresholds: AuditThresholds | None = None,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    count = int(seeds.size)
    default_points, log_scales = geometry(count)
    return audit_partial_anchor_completions(
        combined_seed_labels=seeds,
        component_instance_labels=instances,
        component_project_labels=component_projects,
        dense_labels=dense,
        supporting_views=(
            np.full((count,), 5, dtype=np.uint16) if views is None else views
        ),
        winner_share=np.full((count,), 0.9, dtype=np.float32),
        points=default_points if points is None else points,
        log_scales=log_scales,
        component_items=component_items,
        thresholds=audit_thresholds or thresholds(),
    )


class PartialAnchorAuditTest(unittest.TestCase):
    def test_one_strong_anchor_connected_completion_is_proposed(self) -> None:
        count = 24
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:4] = 4
        records, fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=np.full((count,), 4, dtype=np.int32),
            component_items={1: item(1, 4, "windowpane")},
        )
        self.assertEqual(records[0]["status"], "accepted_proposed_completion")
        self.assertAlmostEqual(records[0]["anchor_coverage"], 1.0)
        self.assertAlmostEqual(records[0]["anchor_precision"], 4.0 / 24.0)
        np.testing.assert_array_equal(fills[0], np.arange(4, 24, dtype=np.uint32))

    def test_conflicting_seed_does_not_discard_safe_retained_subcomponent(self) -> None:
        count = 24
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        seeds[-1] = 9
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        instances[-1] = 2
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:4] = 4
        component_projects[-1] = 9
        records, fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=np.full((count,), 4, dtype=np.int32),
            component_items={
                1: item(1, 4, "windowpane"),
                2: item(2, 9, "cabinet"),
            },
        )
        target = records[0]
        self.assertEqual(target["conflicting_seed_gaussian_count"], 1)
        self.assertEqual(target["status"], "accepted_proposed_completion")
        self.assertNotIn(23, fills[0])

    def test_dominant_competing_instance_partial_overlap_is_rejected(self) -> None:
        count = 40
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        seeds[4:8] = 9
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        instances[4:8] = 2
        instances[24:] = 2
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:4] = 4
        component_projects[4:8] = 9
        component_projects[24:] = 9
        dense = np.zeros((count,), dtype=np.int32)
        dense[:24] = 4
        records, _fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=dense,
            component_items={
                1: item(1, 4, "windowpane"),
                2: item(2, 9, "cabinet"),
            },
            audit_thresholds=thresholds(
                max_competing_thing_fraction=1.0,
                max_partial_competing_thing_risk=0.10,
                min_spatial_keep_fraction=0.0,
            ),
        )
        target = records[0]
        self.assertGreater(
            target["dominant_competing_thing_partial_overlap_risk"], 0.10
        )
        self.assertIn("partial_competing_thing_risk>0.1", target["reasons"])

    def test_multiple_retained_anchor_components_are_rejected(self) -> None:
        count = 24
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[[0, 1, 22, 23]] = 4
        seeds[12] = 9
        instances = np.zeros((count,), dtype=np.int32)
        instances[[0, 1, 22, 23]] = 1
        instances[12] = 2
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[[0, 1, 22, 23]] = 4
        component_projects[12] = 9
        records, fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=np.full((count,), 4, dtype=np.int32),
            component_items={
                1: item(1, 4, "windowpane"),
                2: item(2, 9, "cabinet"),
            },
        )
        target = records[0]
        self.assertEqual(target["retained_anchor_component_count"], 2)
        self.assertIn("retained_anchor_components!=1", target["reasons"])
        self.assertEqual(fills, [])

    def test_low_anchor_coverage_is_rejected(self) -> None:
        count = 26
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        seeds[20:] = 4
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        instances[20:] = 1
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[instances == 1] = 4
        dense = np.zeros((count,), dtype=np.int32)
        dense[:20] = 4
        points, _log_scales = geometry(count)
        points[20:, 0] += 100.0
        records, _fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=dense,
            component_items={1: item(1, 4, "windowpane")},
            points=points,
        )
        self.assertAlmostEqual(records[0]["anchor_coverage"], 0.4)
        self.assertIn("anchor_coverage<0.8", records[0]["reasons"])

    def test_low_anchor_precision_is_rejected(self) -> None:
        count = 32
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:2] = 4
        instances = np.zeros((count,), dtype=np.int32)
        instances[:2] = 1
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:2] = 4
        records, _fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=np.full((count,), 4, dtype=np.int32),
            component_items={1: item(1, 4, "windowpane")},
        )
        self.assertLess(records[0]["maximum_initial_component_anchor_precision"], 0.10)
        self.assertIn("anchor_precision<0.1", records[0]["reasons"])

    def test_weak_candidate_view_support_is_rejected(self) -> None:
        count = 24
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:4] = 4
        records, _fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=np.full((count,), 4, dtype=np.int32),
            component_items={1: item(1, 4, "windowpane")},
            views=np.full((count,), 4, dtype=np.uint16),
        )
        self.assertIn("robust_fraction<0.9", records[0]["reasons"])

    def test_proposed_fill_never_contains_an_immutable_seed(self) -> None:
        count = 24
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:4] = 4
        _records, fills = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=np.full((count,), 4, dtype=np.int32),
            component_items={1: item(1, 4, "windowpane")},
        )
        self.assertTrue(np.all(seeds[fills[0]] == 0))

    def test_component_item_order_does_not_change_results(self) -> None:
        count = 48
        seeds = np.zeros((count,), dtype=np.int32)
        seeds[:4] = 4
        seeds[24:28] = 5
        instances = np.zeros((count,), dtype=np.int32)
        instances[:4] = 1
        instances[24:28] = 2
        component_projects = np.zeros((count,), dtype=np.int32)
        component_projects[:4] = 4
        component_projects[24:28] = 5
        dense = np.full((count,), 4, dtype=np.int32)
        dense[24:] = 5
        points, _log_scales = geometry(count)
        points[24:, 0] += 100.0
        forward = {
            1: item(1, 4, "windowpane"),
            2: item(2, 5, "door"),
        }
        reverse = {2: forward[2], 1: forward[1]}
        first = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=dense,
            component_items=forward,
            points=points,
        )
        second = run_audit(
            seeds=seeds,
            instances=instances,
            component_projects=component_projects,
            dense=dense,
            component_items=reverse,
            points=points,
        )
        self.assertEqual(first[0], second[0])
        for left, right in zip(first[1], second[1]):
            np.testing.assert_array_equal(left, right)


class PartialAnchorAuditSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_uses_cached_dinov3_sources_only(self) -> None:
        self.assertIn("camera_owned_component_summary.json", self.text)
        self.assertIn("combined_seed_summary.json", self.text)
        self.assertIn("dense_vote_summary.json", self.text)
        self.assertIn("supporting_views.npy", self.text)
        self.assertIn("winner_share.npy", self.text)
        self.assertNotIn("scripts.task1.dinov2", self.text)
        self.assertNotIn("_dinov2_", self.text)
        self.assertIn("dinov2_used=0", self.text)

    def test_scheduler_is_report_only_without_label_or_ply_outputs(self) -> None:
        self.assertIn("report_only=1", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn('test ! -e "${AUDIT_DIR}/gaussian_labels.npy"', self.text)
        self.assertIn('test ! -e "${AUDIT_DIR}/label_map.json"', self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)
        self.assertNotIn("export_supersplat_label_colors", self.text)
        self.assertNotIn("write_ply_with_labels", self.text)
        self.assertNotIn('mkdir -p "${AUDIT_DIR}"', self.text)

    def test_scheduler_exposes_the_global_v5_style_gates(self) -> None:
        for expected in (
            'MIN_ANCHOR_GAUSSIANS="${MIN_ANCHOR_GAUSSIANS:-500}"',
            'MIN_ANCHOR_COVERAGE="${MIN_ANCHOR_COVERAGE:-0.80}"',
            'MIN_ANCHOR_PRECISION="${MIN_ANCHOR_PRECISION:-0.10}"',
            'MAX_ANCHOR_PRECISION="${MAX_ANCHOR_PRECISION:-0.25}"',
            'MIN_ROBUST_FRACTION="${MIN_ROBUST_FRACTION:-0.90}"',
            'MIN_SOURCE_VIEWS="${MIN_SOURCE_VIEWS:-5}"',
            'MAX_COMPETING_THING_FRACTION="${MAX_COMPETING_THING_FRACTION:-0.05}"',
            'MIN_SPATIAL_KEEP_FRACTION="${MIN_SPATIAL_KEEP_FRACTION:-0.95}"',
            'MAX_PARTIAL_COMPETING_THING_RISK="${MAX_PARTIAL_COMPETING_THING_RISK:-0.40}"',
        ):
            self.assertIn(expected, self.text)
        self.assertIn("v5_used=0", self.text)
        self.assertIn("v5_method_adapted=1", self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)


if __name__ == "__main__":
    unittest.main()
