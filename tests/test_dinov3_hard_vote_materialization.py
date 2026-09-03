from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.materialize_hard_vote_consensus import (
    add_camera_winners,
    labels_from_statistics,
    status_counts,
    validate_inputs,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_ACCEPTED,
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    STATUS_UNOBSERVED,
    consensus_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "scripts" / "task1" / "dinov3" / "materialize_hard_vote_consensus.py"


class HardVoteMaterializationTest(unittest.TestCase):
    def test_adds_at_most_one_identity_vote_per_camera_and_gaussian(self) -> None:
        counts = np.zeros((4, 5), dtype=np.uint8)
        added = add_camera_winners(
            counts,
            np.asarray([1, 0, 3, 1, 0], dtype=np.uint16),
        )
        self.assertEqual(added, 3)
        np.testing.assert_array_equal(counts[1], [1, 0, 0, 1, 0])
        np.testing.assert_array_equal(counts[3], [0, 0, 1, 0, 0])

    def test_materializes_only_audited_strict_majorities(self) -> None:
        # Columns: accepted 3/4, tie 2/2, plurality 2/1/1, single, unseen.
        counts = np.zeros((5, 5), dtype=np.uint8)
        counts[1:, 0] = [3, 1, 0, 0]
        counts[1:, 1] = [2, 2, 0, 0]
        counts[1:, 2] = [2, 1, 1, 0]
        counts[1:, 3] = [1, 0, 0, 0]
        statistics = consensus_statistics(counts, chunk_size=2)
        labels = labels_from_statistics(statistics)
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0])
        self.assertEqual(
            status_counts(statistics["status"]),
            {
                "accepted_strict_majority": 1,
                "unobserved": 1,
                "single_camera": 1,
                "exact_tie": 1,
                "no_strict_majority": 1,
            },
        )

    def test_status_codes_match_audit_contract(self) -> None:
        statistics = {
            "winner": np.asarray([4, 8, 9, 2, 7], dtype=np.uint16),
            "status": np.asarray(
                [
                    STATUS_ACCEPTED,
                    STATUS_EXACT_TIE,
                    STATUS_NO_STRICT_MAJORITY,
                    STATUS_SINGLE_CAMERA,
                    STATUS_UNOBSERVED,
                ],
                dtype=np.uint8,
            ),
        }
        np.testing.assert_array_equal(labels_from_statistics(statistics), [4, 0, 0, 0, 0])

    def test_materializer_reuses_exact_audit_helpers(self) -> None:
        source = MATERIALIZER.read_text(encoding="utf-8")
        self.assertIn("collapse_camera_distribution", source)
        self.assertIn("consensus_statistics", source)
        self.assertIn("audit_status_counts_reproduced_exactly", source)
        self.assertIn("label_zero_without_fill_or_propagation", source)

    def test_accepts_identical_ontology_from_isolated_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vote_manifest_path = root / "vote_manifest.json"
            recorded_ontology = root / "active" / "ontology.json"
            isolated_ontology = root / "isolated" / "ontology.json"
            source_ply = root / "point_cloud.ply"
            recorded_ontology.parent.mkdir()
            isolated_ontology.parent.mkdir()
            ontology_bytes = b'{"same": "ontology"}'
            recorded_ontology.write_bytes(ontology_bytes)
            isolated_ontology.write_bytes(ontology_bytes)
            source_ply.write_bytes(b"ply")
            vote_manifest = {
                "source": "dinov3_dense_pixel_flashsplat_votes",
                "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
                "ply_path": str(source_ply),
                "camera_count": 1,
                "frames": [{"camera_index": 0}],
                "query_region_filtering_used": False,
                "confidence_threshold_used": False,
                "one_normalized_vote_per_camera": True,
                "v5_used": False,
                "dinov2_used": False,
            }
            vote_manifest_path.write_text(json.dumps(vote_manifest), encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            audit = {
                "source": "dinov3_cached_2d_3d_round_trip_fidelity_audit",
                "contract": "report_only_leave_one_camera_out_round_trip_fidelity_v1",
                "report_only": True,
                "vote_manifest": str(vote_manifest_path),
                "ontology": str(recorded_ontology),
                "camera_count": 1,
                "camera_vote_policy": "one_unique_max_class_per_camera_and_gaussian_else_abstain",
                "camera_vote_scale": "one_equal_identity_vote_per_camera",
                "consensus_policy": "at_least_two_cameras_unique_strict_majority_else_abstain",
                "manual_camera_selection_used": False,
                "manual_gaussian_selection_used": False,
                "accepted_gaussian_labels_written": False,
                "gaussian_project_class_array_written": False,
                "label_map_written": False,
                "semantic_ply_written": False,
                "input_sha256": {
                    str(vote_manifest_path): digest(vote_manifest_path),
                    str(recorded_ontology): digest(recorded_ontology),
                },
            }
            validate_inputs(
                audit,
                vote_manifest,
                root / "audit.json",
                vote_manifest_path,
                isolated_ontology,
                source_ply,
            )

if __name__ == "__main__":
    unittest.main()
