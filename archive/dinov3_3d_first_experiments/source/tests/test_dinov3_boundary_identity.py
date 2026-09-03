from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.boundary_identity_audit import (
    IdentityThresholds,
    score_region_identity,
)
from scripts.task1.dinov3.query_regions import (
    QueryRegionThresholds,
    class_agnostic_query_regions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_boundary_identity_audit_scene.sbatch"
)
ADAPTER = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "dinov3_segment_views.py"
)
PROPOSAL_LIFT = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "grounding"
    / "run_flashsplat_mask_proposals.py"
)
BOUNDARY_AUDIT = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "boundary_identity_audit.py"
)


class Dinov3QueryRegionTest(unittest.TestCase):
    def test_region_boundaries_do_not_depend_on_query_semantic_class(self) -> None:
        masks = np.zeros((2, 6, 8), dtype=np.float32)
        masks[0, 1:5, 1:4] = 0.95
        masks[1, 1:5, 5:7] = 0.90
        logits = np.asarray(
            [
                [5.0, 0.0, -5.0],
                [0.0, 5.0, -5.0],
            ],
            dtype=np.float32,
        )
        thresholds = QueryRegionThresholds(
            min_area=4,
            max_area_ratio=0.90,
        )
        region_id, confidence, regions = class_agnostic_query_regions(
            logits,
            masks,
            thresholds,
        )
        swapped_id, swapped_confidence, swapped_regions = (
            class_agnostic_query_regions(
                logits[:, [1, 0, 2]],
                masks,
                thresholds,
            )
        )
        np.testing.assert_array_equal(region_id, swapped_id)
        np.testing.assert_allclose(confidence, swapped_confidence)
        self.assertEqual(len(regions), 2)
        self.assertEqual(len(swapped_regions), 2)
        self.assertNotEqual(
            regions[0]["diagnostic_ade20k_class"],
            swapped_regions[0]["diagnostic_ade20k_class"],
        )

    def test_no_object_and_oversized_queries_are_filtered(self) -> None:
        masks = np.ones((2, 5, 5), dtype=np.float32) * 0.95
        logits = np.asarray(
            [
                [5.0, 0.0, -5.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )
        region_id, _confidence, regions = class_agnostic_query_regions(
            logits,
            masks,
            QueryRegionThresholds(
                min_area=2,
                max_area_ratio=0.50,
            ),
        )
        self.assertFalse(region_id.any())
        self.assertEqual(regions, [])


class Dinov2RegionIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.region = np.ones((10, 10), dtype=bool)
        self.lookup = np.asarray([15, 9, 1], dtype=np.int16)
        self.names = {1: "wall", 9: "windowpane", 15: "door"}
        self.thresholds = IdentityThresholds(
            min_dinov2_confidence=0.50,
            min_evidence_pixels=20,
            min_evidence_coverage=0.25,
            min_class_share=0.60,
            min_class_margin=0.20,
        )

    def test_clear_dinov2_identity_is_accepted(self) -> None:
        class_id = np.zeros((10, 10), dtype=np.uint8)
        class_id[:, :2] = 1
        confidence = np.full((10, 10), 0.90, dtype=np.float32)
        result = score_region_identity(
            self.region,
            class_id,
            confidence,
            self.lookup,
            self.names,
            {"door", "windowpane"},
            self.thresholds,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_class"], "door")
        self.assertFalse(result["dinov3_semantic_class_used"])

    def test_close_door_window_vote_abstains(self) -> None:
        class_id = np.zeros((10, 10), dtype=np.uint8)
        class_id[:, 5:] = 1
        confidence = np.full((10, 10), 0.90, dtype=np.float32)
        result = score_region_identity(
            self.region,
            class_id,
            confidence,
            self.lookup,
            self.names,
            {"door", "windowpane"},
            self.thresholds,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "abstained_low_identity_share")

    def test_stronger_non_target_class_abstains(self) -> None:
        class_id = np.full((10, 10), 2, dtype=np.uint8)
        confidence = np.full((10, 10), 0.95, dtype=np.float32)
        result = score_region_identity(
            self.region,
            class_id,
            confidence,
            self.lookup,
            self.names,
            {"door", "windowpane"},
            self.thresholds,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["status"],
            "abstained_stronger_non_target_identity",
        )


class BoundaryIdentityPipelineContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER.read_text(encoding="utf-8")
        cls.adapter = ADAPTER.read_text(encoding="utf-8")
        cls.proposal_lift = PROPOSAL_LIFT.read_text(encoding="utf-8")
        cls.boundary_audit = BOUNDARY_AUDIT.read_text(encoding="utf-8")

    def test_scheduler_is_report_only_and_identity_separated(self) -> None:
        self.assertIn('REPORT_ONLY="${REPORT_ONLY:-1}"', self.scheduler)
        self.assertIn('if [[ "${REPORT_ONLY}" != "1" ]]', self.scheduler)
        self.assertIn("--save-regions", self.scheduler)
        self.assertIn(
            "scripts.task1.dinov3.boundary_identity_audit",
            self.scheduler,
        )
        self.assertIn("--report-only", self.scheduler)
        self.assertIn("--no-semantic-ply", self.scheduler)
        self.assertNotIn("candidate_gaussian_labels.npy", self.scheduler)

    def test_scheduler_reuses_v5_and_existing_global_audits(self) -> None:
        self.assertIn("BASE_LABELS", self.scheduler)
        self.assertIn("audit_proposals_against_base", self.scheduler)
        self.assertIn("cluster_semantic_flashsplat_proposals", self.scheduler)
        self.assertIn("--spatial-prune-all-classes", self.scheduler)

    def test_scheduler_does_not_precreate_fail_closed_identity_output(self) -> None:
        mkdir_block = self.scheduler.split("mkdir -p \\\n", 1)[1].split(
            "\n\n",
            1,
        )[0]
        self.assertNotIn('"${IDENTITY_DIR}"', mkdir_block)

    def test_adapter_exports_diagnostic_only_query_regions(self) -> None:
        self.assertIn('"--save-regions"', self.adapter)
        self.assertIn("inference_query_regions", self.adapter)
        self.assertIn("class_agnostic_query_regions", self.adapter)
        self.assertIn('"semantic_class_used_for_identity": False', self.adapter)

    def test_identity_audit_survives_proposal_lifting(self) -> None:
        self.assertIn('"identity_audit"', self.proposal_lift)

    def test_identity_overlay_is_python39_compatible(self) -> None:
        self.assertNotIn("strict=True", self.boundary_audit)
        self.assertIn("mask and metadata counts differ", self.boundary_audit)


if __name__ == "__main__":
    unittest.main()
