from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.associate_3d_query_regions import (
    QueryProposal,
    semantic_stability,
)
from scripts.task1.dinov3.audit_anchored_component_association import (
    AuditThresholds,
    build_anchor_component,
    classify_component_record,
    classify_numeric_consensus,
    evaluate_anchor_extensions,
    match_candidates_to_anchors,
    resolve_fill_overlaps,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_anchored_association_audit_scene.sbatch"
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
    indices: np.ndarray,
    *,
    winner: int = 0,
    feature: tuple[float, float] = (1.0, 0.0),
) -> QueryProposal:
    probabilities = np.asarray([0.9, 0.1], dtype=np.float32)
    if winner == 1:
        probabilities = probabilities[::-1].copy()
    return QueryProposal(
        proposal_id=proposal_id,
        frame_file=f"camera_{camera_index:04d}.png",
        camera_index=camera_index,
        region_id=proposal_id,
        indices=np.asarray(indices, dtype=np.uint32),
        counts=np.ones((len(indices),), dtype=np.float32),
        class_probabilities=probabilities,
        no_object_probability=0.01,
        query_embedding=np.asarray(feature, dtype=np.float32),
        quality=0.9,
        metadata={},
    )


def anchor(
    component_id: int,
    start: int,
    *,
    cameras: tuple[int, int] = (0, 1),
):
    proposals = [
        proposal(component_id * 10 + index, camera, np.arange(start, start + 100))
        for index, camera in enumerate(cameras)
    ]
    record = {
        "component_id": component_id,
        "accepted": True,
        "class": f"class_{component_id}",
        "project_id": component_id,
        "ade_id": 0,
        "kind": "thing",
    }
    return build_anchor_component(record, proposals)


def thresholds(**overrides: float | int) -> AuditThresholds:
    values: dict[str, float | int] = {
        "min_shared_gaussians": 10,
        "min_iou": 0.05,
        "min_containment": 0.25,
        "min_feature_similarity": 0.0,
        "min_candidate_anchor_containment": 0.50,
        "min_unique_anchor_score_ratio": 1.10,
        "min_extension_views": 2,
        "min_extension_gaussians": 20,
    }
    values.update(overrides)
    return AuditThresholds(**values)


class AnchoredAssociationAuditTest(unittest.TestCase):
    def test_unique_direct_candidate_attaches_to_one_anchor(self) -> None:
        anchors = [anchor(1, 0), anchor(2, 100)]
        direct = proposal(100, 10, np.arange(0, 100))
        attached, diagnostics = match_candidates_to_anchors(
            anchors, [direct], thresholds()
        )
        self.assertEqual([item[0].proposal_id for item in attached[1]], [100])
        self.assertEqual(attached[2], [])
        self.assertEqual(diagnostics["attached_proposal_count"], 1)

    def test_equal_bridge_between_anchors_is_rejected_as_ambiguous(self) -> None:
        anchors = [anchor(1, 0), anchor(2, 100)]
        bridge = proposal(100, 10, np.arange(50, 150))
        attached, diagnostics = match_candidates_to_anchors(
            anchors, [bridge], thresholds()
        )
        self.assertEqual(attached[1], [])
        self.assertEqual(attached[2], [])
        self.assertEqual(diagnostics["ambiguous_anchor_proposal_count"], 1)

    def test_candidate_cannot_chain_through_an_earlier_extension(self) -> None:
        immutable = anchor(1, 0)
        first = proposal(100, 10, np.arange(50, 150))
        second = proposal(101, 11, np.arange(100, 200))
        attached, _diagnostics = match_candidates_to_anchors(
            [immutable], [first, second], thresholds()
        )
        self.assertEqual([item[0].proposal_id for item in attached[1]], [100])
        self.assertNotIn(101, [item[0].proposal_id for item in attached[1]])

    def test_one_camera_contributes_at_most_one_region_to_an_anchor(self) -> None:
        immutable = anchor(1, 0)
        stronger = proposal(100, 10, np.arange(0, 100))
        weaker = proposal(101, 10, np.arange(0, 80))
        attached, diagnostics = match_candidates_to_anchors(
            [immutable], [weaker, stronger], thresholds()
        )
        self.assertEqual([item[0].proposal_id for item in attached[1]], [100])
        self.assertEqual(diagnostics["attached_proposal_count"], 1)

    def test_exactly_one_outlier_is_separate_from_strict_and_mixed(self) -> None:
        stability = semantic_stability(
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.1, 0.9],
                ],
                dtype=np.float32,
            )
        )
        self.assertEqual(
            classify_numeric_consensus(stability, 0),
            "one_view_outlier_leave_one_out_stable",
        )
        self.assertEqual(classify_numeric_consensus(stability, 1), "mixed_or_unstable")

    def test_serialized_one_outlier_component_is_reported_not_accepted(self) -> None:
        record = {
            "status": "abstained_unstable_multiview_identity",
            "class": "door",
            "semantic_stability": {
                "view_count": 27,
                "winner_view_count": 26,
                "leave_one_out_winners": ["door"] * 27,
            },
        }
        self.assertEqual(
            classify_component_record(record),
            "one_view_outlier_leave_one_out_stable",
        )
        self.assertFalse(record.get("accepted", False))

    def test_extension_requires_matching_support_from_multiple_new_cameras(self) -> None:
        immutable = anchor(1, 0)
        first = proposal(100, 10, np.arange(50, 150))
        second = proposal(101, 11, np.arange(40, 140))
        attached, _diagnostics = match_candidates_to_anchors(
            [immutable], [first, second], thresholds()
        )
        records, fills = evaluate_anchor_extensions(
            [immutable], attached, thresholds()
        )
        self.assertEqual(records[0]["status"], "accepted_report_only_extension_proposal")
        np.testing.assert_array_equal(fills[1], np.arange(100, 140, dtype=np.uint32))

    def test_single_view_outside_support_is_not_proposed(self) -> None:
        immutable = anchor(1, 0)
        candidate = proposal(100, 10, np.arange(50, 150))
        attached, _diagnostics = match_candidates_to_anchors(
            [immutable], [candidate], thresholds()
        )
        records, fills = evaluate_anchor_extensions(
            [immutable], attached, thresholds()
        )
        self.assertEqual(
            records[0]["status"], "rejected_insufficient_multiview_extension"
        )
        self.assertEqual(fills, {})

    def test_cross_anchor_fill_overlap_abstains(self) -> None:
        records = [
            {"component_id": 1},
            {"component_id": 2},
        ]
        proposed, owner_count, overlap, exclusive = resolve_fill_overlaps(
            records,
            {
                1: np.asarray([0, 1, 2], dtype=np.uint32),
                2: np.asarray([2, 3], dtype=np.uint32),
            },
            vertex_count=5,
        )
        np.testing.assert_array_equal(owner_count, [1, 1, 2, 1, 0])
        np.testing.assert_array_equal(overlap, [False, False, True, False, False])
        np.testing.assert_array_equal(proposed, [True, True, False, True, False])
        np.testing.assert_array_equal(exclusive[1], [0, 1])
        np.testing.assert_array_equal(exclusive[2], [3])


class AnchoredAssociationSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")
        cls.renderer = RENDERER.read_text(encoding="utf-8")

    def test_scheduler_reuses_cached_dinov3_reports(self) -> None:
        self.assertIn("ANCHOR_COMPONENT_REPORT", self.text)
        self.assertIn("ANCHOR_PROPOSAL_MANIFEST", self.text)
        self.assertIn("CANDIDATE_COMPONENT_REPORT", self.text)
        self.assertIn("CANDIDATE_PROPOSAL_MANIFEST", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("run_flashsplat_mask_proposals", self.text)
        self.assertIn("dinov2_used=0", self.text)

    def test_scheduler_is_fresh_report_only_and_writes_no_ply(self) -> None:
        self.assertIn("report_only=1", self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("label_map_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn("refuses RESET_OUTPUT=1", self.text)
        self.assertIn('test ! -e "${AUDIT_DIR}/gaussian_labels.npy"', self.text)
        self.assertIn('test ! -e "${AUDIT_DIR}/label_map.json"', self.text)
        self.assertIn("Report-only audit unexpectedly wrote a PLY", self.text)
        self.assertNotIn("write_ply_with_labels", self.text)
        self.assertNotIn("export_supersplat_label_colors", self.text)

    def test_scheduler_exposes_global_class_neutral_gates(self) -> None:
        for expected in (
            'MIN_SHARED_GAUSSIANS="${MIN_SHARED_GAUSSIANS:-250}"',
            'MIN_IOU="${MIN_IOU:-0.05}"',
            'MIN_CONTAINMENT="${MIN_CONTAINMENT:-0.25}"',
            'MIN_FEATURE_SIMILARITY="${MIN_FEATURE_SIMILARITY:-0.0}"',
            'MIN_CANDIDATE_ANCHOR_CONTAINMENT="${MIN_CANDIDATE_ANCHOR_CONTAINMENT:-0.50}"',
            'MIN_UNIQUE_ANCHOR_SCORE_RATIO="${MIN_UNIQUE_ANCHOR_SCORE_RATIO:-1.10}"',
            'MIN_EXTENSION_VIEWS="${MIN_EXTENSION_VIEWS:-2}"',
            'MIN_EXTENSION_GAUSSIANS="${MIN_EXTENSION_GAUSSIANS:-500}"',
        ):
            self.assertIn(expected, self.text)
        self.assertIn("scene_specific_rules=0", self.text)
        self.assertIn("class_specific_thresholds=0", self.text)

    def test_renderer_accepts_the_new_report_contract(self) -> None:
        self.assertIn(
            "report_only_dinov3_anchored_component_association_v1",
            self.renderer,
        )


if __name__ == "__main__":
    unittest.main()
