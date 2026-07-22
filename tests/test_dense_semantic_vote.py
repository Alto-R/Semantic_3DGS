from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dense_seg.ade20k_ontology import ADE20K_NUM_CLASSES, load_ontology
from scripts.task1.dense_seg.fuse_semantic_votes import (
    LabelBuilder,
    decide_vote_labels,
    merge_fill,
    prune_small_new_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "ade20k_to_project.dense_backends.json"
SCHEDULER_PATH = (
    PROJECT_ROOT / "scripts" / "slurm" / "slurm_task1_dense_semantic_scene.sbatch"
)


class OntologyConfigTest(unittest.TestCase):
    def test_ontology_covers_all_150_classes(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        self.assertEqual(ontology.ade_to_project.shape[0], ADE20K_NUM_CLASSES)
        self.assertEqual(ontology.class_names[0], "ignore")
        self.assertGreater(ontology.num_project_classes, 0)

    def test_every_project_class_has_one_type(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        for compact_id in range(1, len(ontology.class_names)):
            self.assertIn(ontology.class_types[compact_id], {"thing", "stuff"})

    def test_key_scene_classes_are_mapped(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        names = set(ontology.class_names)
        for expected in ("bicycle", "bench", "truck", "television", "sky", "wall", "floor"):
            self.assertIn(expected, names)

    def test_remap_rejects_out_of_range(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        with self.assertRaises(ValueError):
            ontology.remap(np.asarray([[0, 150]], dtype=np.int16))

    def test_remap_sends_wall_to_stuff_class(self) -> None:
        ontology = load_ontology(ONTOLOGY_PATH)
        compact = ontology.remap(np.asarray([[0]], dtype=np.int16))
        wall_id = int(compact[0, 0])
        self.assertEqual(ontology.class_names[wall_id], "wall")
        self.assertTrue(ontology.is_stuff(wall_id))


class DecideVoteLabelsTest(unittest.TestCase):
    def test_clear_winner_is_assigned(self) -> None:
        votes = np.asarray([[9.0, 1.0]], dtype=np.float32)
        support = np.asarray([[5, 1]], dtype=np.uint16)
        visible = np.asarray([6], dtype=np.uint16)
        result = decide_vote_labels(votes, support, visible, 2, 0.5, 0.0)
        self.assertEqual(result["class_choice"][0], 1)

    def test_no_votes_stays_unlabeled(self) -> None:
        votes = np.zeros((3, 2), dtype=np.float32)
        support = np.zeros((3, 2), dtype=np.uint16)
        visible = np.asarray([4, 0, 2], dtype=np.uint16)
        result = decide_vote_labels(votes, support, visible, 2, 0.5, 0.0)
        self.assertTrue((result["class_choice"] == 0).all())
        self.assertFalse(result["raw_voted"].any())

    def test_min_views_gate(self) -> None:
        votes = np.asarray([[10.0, 0.0]], dtype=np.float32)
        support = np.asarray([[1, 0]], dtype=np.uint16)
        visible = np.asarray([10], dtype=np.uint16)
        result = decide_vote_labels(votes, support, visible, 2, 0.5, 0.0)
        self.assertEqual(result["class_choice"][0], 0)

    def test_min_agreement_gate(self) -> None:
        votes = np.asarray([[5.0, 4.9]], dtype=np.float32)
        support = np.asarray([[6, 6]], dtype=np.uint16)
        visible = np.asarray([12], dtype=np.uint16)
        result = decide_vote_labels(votes, support, visible, 2, 0.6, 0.0)
        self.assertEqual(result["class_choice"][0], 0)

    def test_min_visible_ratio_gate(self) -> None:
        # 3 supporting views out of 60 visible views: systematic artifact case.
        votes = np.asarray([[10.0, 0.0]], dtype=np.float32)
        support = np.asarray([[3, 0]], dtype=np.uint16)
        visible = np.asarray([60], dtype=np.uint16)
        loose = decide_vote_labels(votes, support, visible, 2, 0.5, 0.0)
        strict = decide_vote_labels(votes, support, visible, 2, 0.5, 0.25)
        self.assertEqual(loose["class_choice"][0], 1)
        self.assertEqual(strict["class_choice"][0], 0)

    def test_zero_visible_views_does_not_divide_by_zero(self) -> None:
        votes = np.asarray([[1.0, 0.0]], dtype=np.float32)
        support = np.asarray([[2, 0]], dtype=np.uint16)
        visible = np.asarray([0], dtype=np.uint16)
        result = decide_vote_labels(votes, support, visible, 2, 0.5, 0.0)
        self.assertEqual(result["class_choice"][0], 1)


class MergeFillTest(unittest.TestCase):
    # compact ids: 0 ignore, 1 thing, 2 stuff
    CLASS_TYPES = ["ignore", "thing", "stuff"]

    def test_base_labels_are_never_overwritten(self) -> None:
        base = np.asarray([7, 0, 3, 0], dtype=np.int32)
        choice = np.asarray([2, 2, 1, 1], dtype=np.int16)
        filled = merge_fill(base, choice, self.CLASS_TYPES, stuff_only=False)
        self.assertEqual(filled[0], 0)  # base 7 kept, no fill vote applied
        self.assertEqual(filled[2], 0)  # base 3 kept
        self.assertEqual(filled[1], 2)
        self.assertEqual(filled[3], 1)

    def test_stuff_only_drops_thing_votes(self) -> None:
        base = np.asarray([0, 0], dtype=np.int32)
        choice = np.asarray([1, 2], dtype=np.int16)
        filled = merge_fill(base, choice, self.CLASS_TYPES, stuff_only=True)
        self.assertEqual(filled[0], 0)  # thing vote suppressed
        self.assertEqual(filled[1], 2)  # stuff vote kept

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            merge_fill(
                np.zeros(3, dtype=np.int32),
                np.zeros(4, dtype=np.int16),
                self.CLASS_TYPES,
                stuff_only=True,
            )


class LabelBuilderTest(unittest.TestCase):
    BASE = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled"},
        {"id": 1, "name": "bicycle_01", "class": "bicycle"},
        {"id": 5, "name": "sky", "class": "sky"},
    ]

    def test_new_ids_start_after_base_max(self) -> None:
        builder = LabelBuilder(self.BASE)
        new_id = builder.add("wall", "wall")
        self.assertEqual(new_id, 6)

    def test_existing_stuff_is_found(self) -> None:
        builder = LabelBuilder(self.BASE)
        self.assertEqual(builder.existing_stuff_id("sky"), 5)
        self.assertIsNone(builder.existing_stuff_id("wall"))

    def test_thing_style_entries_are_never_merge_targets(self) -> None:
        # bicycle_01 shares class "bicycle" but is an accepted thing instance;
        # a stuff vote for "bicycle" must open a new id, not extend it.
        builder = LabelBuilder(self.BASE)
        self.assertIsNone(builder.existing_stuff_id("bicycle"))

    def test_empty_builder_creates_unlabeled_entry(self) -> None:
        builder = LabelBuilder()
        label_map = builder.label_map("room")
        self.assertEqual(label_map["labels"][0]["id"], 0)
        self.assertEqual(builder.add("wall", "wall"), 1)

    def test_base_without_id_zero_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LabelBuilder([{"id": 3, "name": "sky", "class": "sky"}])


class PruneSmallNewLabelsTest(unittest.TestCase):
    def test_small_new_stuff_label_is_pruned(self) -> None:
        builder = LabelBuilder()
        label_id = builder.add("wall", "wall")
        labels = np.full(50, label_id, dtype=np.int32)
        records = [
            {"id": label_id, "name": "wall", "class": "wall", "type": "stuff", "gaussians": 50}
        ]
        pruned = prune_small_new_labels(labels, records, builder, 10, 100)
        self.assertEqual(len(pruned), 1)
        self.assertTrue((labels == 0).all())
        self.assertNotIn(label_id, [item["id"] for item in builder.entries])

    def test_merged_stuff_labels_are_never_pruned(self) -> None:
        builder = LabelBuilder(LabelBuilderTest.BASE)
        labels = np.full(3, 5, dtype=np.int32)  # extends the accepted sky group
        records = [
            {
                "id": 5,
                "name": "sky",
                "class": "sky",
                "type": "stuff",
                "gaussians": 3,
                "merged_into_existing": True,
            }
        ]
        pruned = prune_small_new_labels(labels, records, builder, 10, 100)
        self.assertEqual(pruned, [])
        self.assertTrue((labels == 5).all())


class DenseSemanticSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER_PATH.read_text(encoding="utf-8")

    def test_explicit_project_root_is_honored(self) -> None:
        self.assertIn(
            'PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"',
            self.scheduler,
        )

    def test_report_only_contract_writes_no_labels_or_ply(self) -> None:
        self.assertIn('echo "mode=report_only_dense_seg"', self.scheduler)
        self.assertIn('echo "semantic_labels_written=0"', self.scheduler)
        self.assertIn('echo "semantic_ply_written=0"', self.scheduler)

    def test_report_only_exits_before_vote_lifting(self) -> None:
        report_only = self.scheduler.index('if [ "${REPORT_ONLY}" = "1" ]; then')
        vote_lift = self.scheduler.index("run_stage 02_vote_lift")
        self.assertLess(report_only, vote_lift)

    def test_report_only_does_not_require_fill_baseline(self) -> None:
        self.assertIn(
            'if [ "${MODE}" = "fill" ] && [ "${REPORT_ONLY}" != "1" ]; then',
            self.scheduler,
        )


if __name__ == "__main__":
    unittest.main()
