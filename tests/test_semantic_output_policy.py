from __future__ import annotations

import unittest
from pathlib import Path


from scripts.task1.common.ply_utils import resolve_semantic_ply_output


class SemanticOutputPolicyTest(unittest.TestCase):
    def test_stage_output_is_disabled_by_default(self) -> None:
        self.assertIsNone(resolve_semantic_ply_output(Path("stage")))

    def test_final_deliverable_path_is_respected(self) -> None:
        path = Path("deliverables") / "semantic_point_cloud.ply"
        self.assertEqual(
            resolve_semantic_ply_output(Path("stage"), semantic_ply_path=path),
            path,
        )

    def test_legacy_relative_name_requires_explicit_opt_in(self) -> None:
        self.assertEqual(
            resolve_semantic_ply_output(
                Path("stage"), semantic_ply_name="semantic_point_cloud.ply"
            ),
            Path("stage") / "semantic_point_cloud.ply",
        )

    def test_conflicting_output_controls_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both enabled and disabled"):
            resolve_semantic_ply_output(
                Path("stage"),
                semantic_ply_path=Path("deliverables") / "semantic_point_cloud.ply",
                disabled=True,
            )


if __name__ == "__main__":
    unittest.main()
