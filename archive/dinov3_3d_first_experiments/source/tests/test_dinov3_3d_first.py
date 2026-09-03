from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.associate_3d_query_regions import (
    QueryProposal,
    associate_components,
    component_record,
    load_query_proposals,
    mutual_best_edges,
    semantic_stability,
)
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.prepare_3d_region_proposals import prepare_frame
from scripts.task1.dinov3.query_regions import compact_query_evidence
from scripts.task1.dinov3.materialize_3d_component_labels import (
    ComponentSupport,
    remap_nonempty_labels,
    resolve_component_assignments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_3d_first_scene.sbatch"
)
SEGMENTER = (
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
ASSOCIATION = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "associate_3d_query_regions.py"
)
MATERIALIZER = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "materialize_3d_component_labels.py"
)
MATERIALIZE_SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_3d_first_materialize_scene.sbatch"
)


def proposal(
    proposal_id: int,
    frame: str,
    indices: np.ndarray,
    probabilities: np.ndarray,
    embedding: np.ndarray,
) -> QueryProposal:
    normalized_embedding = embedding / np.linalg.norm(embedding)
    return QueryProposal(
        proposal_id=proposal_id,
        frame_file=frame,
        camera_index=proposal_id,
        region_id=proposal_id,
        indices=np.asarray(indices, dtype=np.uint32),
        counts=np.ones((len(indices),), dtype=np.float32),
        class_probabilities=np.asarray(probabilities, dtype=np.float32),
        no_object_probability=0.01,
        query_embedding=normalized_embedding.astype(np.float32),
        quality=0.9,
        metadata={},
    )


class QueryEvidenceExportTest(unittest.TestCase):
    def test_compact_evidence_keeps_full_probabilities_and_embeddings(self) -> None:
        logits = np.asarray(
            [
                [4.0, 1.0, 0.0, -2.0],
                [1.0, 4.0, 0.0, -1.0],
                [0.0, 1.0, 4.0, -3.0],
            ],
            dtype=np.float32,
        )
        regions = [
            {"query_index": 2},
            {"query_index": 0},
        ]
        embeddings = np.asarray(
            [
                [3.0, 0.0],
                [0.0, 2.0],
                [4.0, 3.0],
            ],
            dtype=np.float32,
        )
        evidence = compact_query_evidence(logits, regions, embeddings)
        np.testing.assert_array_equal(evidence["query_indices"], [2, 0])
        self.assertEqual(evidence["class_probabilities"].shape, (2, 3))
        np.testing.assert_allclose(
            evidence["class_probabilities"].sum(axis=1),
            1.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(evidence["query_embeddings"], axis=1),
            1.0,
            atol=1e-6,
        )
        self.assertGreater(
            evidence["class_probabilities"][0, 2],
            evidence["class_probabilities"][0, 0],
        )

    def test_prepare_frame_preserves_every_region_without_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            mask_dir = root / "masks"
            evidence_dir = root / "evidence"
            mask_dir.mkdir()
            evidence_dir.mkdir()
            region_id = np.zeros((4, 5), dtype=np.uint16)
            region_id[0:2, 0:2] = 1
            region_id[2:4, 3:5] = 2
            np.savez_compressed(
                source / "frame.npz",
                region_id=region_id,
                region_confidence=np.where(region_id > 0, 0.9, 0.0),
                query_indices=np.asarray([7, 11], dtype=np.int16),
                class_probabilities=np.asarray(
                    [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]],
                    dtype=np.float16,
                ),
                no_object_probabilities=np.asarray([0.01, 0.02], dtype=np.float16),
                query_embeddings=np.asarray(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=np.float16
                ),
            )
            frame = {
                "file": "frame.png",
                "camera_index": 3,
                "camera_id": 3,
                "region_file": "frame.npz",
                "regions": [
                    {
                        "region_id": 1,
                        "query_index": 7,
                        "area": 4,
                        "bbox_xyxy": [0, 0, 2, 2],
                        "mean_region_confidence": 0.9,
                    },
                    {
                        "region_id": 2,
                        "query_index": 11,
                        "area": 4,
                        "bbox_xyxy": [3, 2, 5, 4],
                        "mean_region_confidence": 0.8,
                    },
                ],
            }
            prepared = prepare_frame(frame, source, mask_dir, evidence_dir)
            self.assertEqual(prepared["region_count"], 2)
            self.assertTrue(
                all(
                    not item["semantic_identity_assigned"]
                    for item in prepared["masks"]
                )
            )
            with np.load(mask_dir / "frame.npz") as data:
                self.assertEqual(data["masks"].shape, (2, 4, 5))


class ClassAgnosticAssociationTest(unittest.TestCase):
    def test_association_uses_geometry_and_features_not_class_identity(self) -> None:
        first = proposal(
            1,
            "a.png",
            np.arange(0, 100),
            np.asarray([0.9, 0.1]),
            np.asarray([1.0, 0.0]),
        )
        second = proposal(
            2,
            "b.png",
            np.arange(5, 105),
            np.asarray([0.1, 0.9]),
            np.asarray([0.99, 0.01]),
        )
        third = proposal(
            3,
            "a.png",
            np.arange(200, 300),
            np.asarray([0.2, 0.8]),
            np.asarray([0.0, 1.0]),
        )
        fourth = proposal(
            4,
            "b.png",
            np.arange(205, 305),
            np.asarray([0.8, 0.2]),
            np.asarray([0.01, 0.99]),
        )
        proposals = [first, second, third, fourth]
        edges = mutual_best_edges(
            proposals,
            min_shared_gaussians=20,
            min_iou=0.05,
            min_containment=0.25,
            min_feature_similarity=0.0,
        )
        components = associate_components(proposals, edges)
        self.assertEqual(
            [[item.proposal_id for item in component] for component in components],
            [[1, 2], [3, 4]],
        )

        swapped = [
            proposal(
                item.proposal_id,
                item.frame_file,
                item.indices,
                item.class_probabilities[::-1],
                item.query_embedding,
            )
            for item in proposals
        ]
        swapped_edges = mutual_best_edges(
            swapped,
            min_shared_gaussians=20,
            min_iou=0.05,
            min_containment=0.25,
            min_feature_similarity=0.0,
        )
        self.assertEqual(
            [
                (item["left_proposal_id"], item["right_proposal_id"])
                for item in edges
            ],
            [
                (item["left_proposal_id"], item["right_proposal_id"])
                for item in swapped_edges
            ],
        )

    def test_same_view_regions_are_never_merged(self) -> None:
        proposals = [
            proposal(
                1,
                "same.png",
                np.arange(100),
                np.asarray([0.8, 0.2]),
                np.asarray([1.0, 0.0]),
            ),
            proposal(
                2,
                "same.png",
                np.arange(100),
                np.asarray([0.8, 0.2]),
                np.asarray([1.0, 0.0]),
            ),
        ]
        self.assertEqual(
            mutual_best_edges(
                proposals,
                min_shared_gaussians=1,
                min_iou=0.0,
                min_containment=0.0,
                min_feature_similarity=-1.0,
            ),
            [],
        )


class MultiviewSemanticFusionTest(unittest.TestCase):
    def test_stable_identity_survives_every_leave_one_out_view(self) -> None:
        result = semantic_stability(
            np.asarray(
                [
                    [0.80, 0.15, 0.05],
                    [0.65, 0.25, 0.10],
                    [0.70, 0.10, 0.20],
                ],
                dtype=np.float32,
            )
        )
        self.assertEqual(result["winner"], 0)
        self.assertTrue(result["stable"])

    def test_view_dependent_identity_remains_ambiguous(self) -> None:
        result = semantic_stability(
            np.asarray(
                [
                    [0.80, 0.10, 0.10],
                    [0.10, 0.80, 0.10],
                    [0.10, 0.10, 0.80],
                ],
                dtype=np.float32,
            )
        )
        self.assertFalse(result["stable"])

    def test_lifted_proposals_resolve_compact_evidence_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "regions"
            lift = root / "lift"
            (source / "query_evidence").mkdir(parents=True)
            (lift / "proposal_supports").mkdir(parents=True)
            (source / "query_region_manifest.json").write_text(
                '{"contract":"class_agnostic_query_masks_with_soft_evidence_v1"}',
                encoding="utf-8",
            )
            probabilities = np.full((2, 150), 0.001, dtype=np.float32)
            probabilities[:, 14] = 0.851
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            np.savez_compressed(
                source / "query_evidence" / "views.npz",
                query_indices=np.asarray([4, 7], dtype=np.int16),
                class_probabilities=probabilities.astype(np.float16),
                no_object_probabilities=np.asarray([0.01, 0.02], dtype=np.float16),
                query_embeddings=np.asarray(
                    [[1.0, 0.0], [0.99, 0.01]], dtype=np.float16
                ),
            )
            proposal_metadata = []
            for proposal_id, frame_file, row, start in (
                (1, "a.png", 0, 0),
                (2, "b.png", 1, 5),
            ):
                support_file = f"proposal_{proposal_id:06d}.npz"
                indices = np.arange(start, start + 100, dtype=np.uint32)
                np.savez_compressed(
                    lift / "proposal_supports" / support_file,
                    indices=indices,
                    counts=np.ones((100,), dtype=np.float32),
                )
                proposal_metadata.append(
                    {
                        "proposal_id": proposal_id,
                        "support_file": support_file,
                        "frame_file": frame_file,
                        "camera_index": proposal_id,
                        "region_id": 1,
                        "confidence": 0.9,
                        "semantic_identity_assigned": False,
                        "query_evidence_file": "query_evidence/views.npz",
                        "query_evidence_row": row,
                    }
                )
            proposal_manifest = lift / "proposal_manifest.json"
            proposal_manifest.write_text(
                json.dumps(
                    {
                        "source_manifest": str(
                            source / "query_region_manifest.json"
                        ),
                        "vertex_count": 200,
                        "proposals": proposal_metadata,
                    }
                ),
                encoding="utf-8",
            )
            loaded, _manifest = load_query_proposals(proposal_manifest)
            self.assertEqual(len(loaded), 2)
            record, indices, _scores = component_record(
                1,
                loaded,
                load_ontology(PROJECT_ROOT / "configs" / "ade20k_to_project.json"),
                min_source_views=2,
                min_component_gaussians=50,
            )
            self.assertTrue(record["accepted"])
            self.assertEqual(record["class"], "door")
            self.assertEqual(indices.shape[0], 105)


class ConflictAbstainingMaterializationTest(unittest.TestCase):
    @staticmethod
    def component(
        component_id: int,
        project_id: int,
        indices: list[int],
        scores: list[float],
    ) -> ComponentSupport:
        return ComponentSupport(
            component_id=component_id,
            project_id=project_id,
            indices=np.asarray(indices, dtype=np.uint32),
            scores=np.asarray(scores, dtype=np.float32),
            record={
                "class": f"class_{project_id}",
                "ade_id": project_id - 1,
                "kind": "thing",
                "probability": 0.9,
                "proposal_ids": [component_id],
                "source_frames": ["a.png", "b.png"],
                "source_view_count": 2,
                "support_gaussian_count": len(indices),
            },
        )

    def test_cross_class_overlap_abstains_and_same_class_uses_support(self) -> None:
        components = [
            self.component(10, 3, [0, 1, 2], [1.0, 2.0, 1.0]),
            self.component(20, 3, [2, 3], [2.0, 1.0]),
            self.component(30, 7, [1, 4], [4.0, 1.0]),
        ]
        labels, project_classes, conflicts = resolve_component_assignments(
            components,
            vertex_count=6,
        )
        np.testing.assert_array_equal(labels, [1, 0, 2, 2, 3, 0])
        np.testing.assert_array_equal(project_classes, [3, 0, 3, 3, 7, 0])
        np.testing.assert_array_equal(conflicts, [False, True, False, False, False, False])

        remapped, records = remap_nonempty_labels(labels, components)
        np.testing.assert_array_equal(remapped, labels)
        self.assertEqual([item["component_id"] for item in records], [10, 20, 30])
        self.assertEqual([item["assigned_gaussian_count"] for item in records], [1, 2, 1])

    def test_empty_component_labels_are_removed_and_ids_become_contiguous(self) -> None:
        components = [
            self.component(10, 3, [0], [1.0]),
            self.component(20, 3, [0], [2.0]),
            self.component(30, 7, [1], [1.0]),
        ]
        labels, _classes, _conflicts = resolve_component_assignments(components, 2)
        remapped, records = remap_nonempty_labels(labels, components)
        np.testing.assert_array_equal(remapped, [1, 2])
        self.assertEqual([item["component_id"] for item in records], [20, 30])


class IndependentPipelineContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER.read_text(encoding="utf-8")
        cls.segmenter = SEGMENTER.read_text(encoding="utf-8")
        cls.proposal_lift = PROPOSAL_LIFT.read_text(encoding="utf-8")
        cls.association = ASSOCIATION.read_text(encoding="utf-8")
        cls.materializer = MATERIALIZER.read_text(encoding="utf-8")
        cls.materialize_scheduler = MATERIALIZE_SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_has_no_prior_semantic_input_or_base_merge(self) -> None:
        for forbidden in (
            "BASE_LABELS",
            "BASE_LABEL_MAP",
            "BASE_OUTPUT_NAME",
            "audit_proposals_against_base",
            "candidate_gaussian_labels.npy",
        ):
            self.assertNotIn(forbidden, self.scheduler)
        self.assertIn("prior_semantic_labels_used=0", self.scheduler)

    def test_scheduler_lifts_before_semantic_fusion(self) -> None:
        lift = self.scheduler.index("04_lift_all_query_regions")
        fusion = self.scheduler.index("05_associate_and_fuse_3d_components")
        self.assertLess(lift, fusion)
        self.assertNotIn("boundary_identity_audit", self.scheduler)

    def test_pipeline_is_permanently_report_only(self) -> None:
        self.assertIn('if [[ "${REPORT_ONLY}" != "1" ]]', self.scheduler)
        self.assertIn("--report-only", self.scheduler)
        self.assertIn("never Gaussian labels", self.association)
        self.assertNotIn("np.save", self.association)

    def test_segmenter_exports_full_soft_query_evidence(self) -> None:
        for expected in (
            "class_probabilities=query_evidence",
            "no_object_probabilities=query_evidence",
            "query_embeddings=query_evidence",
            "register_forward_pre_hook",
            "semantic_identity_hardened_per_view",
        ):
            self.assertIn(expected, self.segmenter)

    def test_proposal_lift_preserves_query_evidence_references(self) -> None:
        for expected in (
            '"query_evidence_file"',
            '"query_evidence_row"',
            '"region_id"',
            '"semantic_identity_assigned"',
        ):
            self.assertIn(expected, self.proposal_lift)

    def test_materializer_is_separate_and_conflict_abstaining(self) -> None:
        self.assertIn("cross_class_conflict", self.materializer)
        self.assertIn("labels[cross_class_conflict] = 0", self.materializer)
        self.assertIn("project_classes[cross_class_conflict] = 0", self.materializer)
        for forbidden in (
            "BASE_LABELS",
            "BASE_LABEL_MAP",
            "DINOV2_LABELS",
            "DINOV2_OUTPUT",
        ):
            self.assertNotIn(forbidden, self.materialize_scheduler.upper())
        self.assertIn("dinov2_used=0", self.materialize_scheduler)
        self.assertIn("SOURCE_OUTPUT_NAME", self.materialize_scheduler)
        self.assertIn("materialize_3d_component_labels", self.materialize_scheduler)


if __name__ == "__main__":
    unittest.main()
