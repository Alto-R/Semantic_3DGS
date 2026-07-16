from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from run_grounding_semantic_branch import run_logged  # noqa: E402


class GroundingBranchDriverTest(unittest.TestCase):
    def test_cuda_kernel_failure_text_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "cuda.log"
            command = [
                sys.executable,
                "-c",
                "print('no kernel image is available for execution on the device')",
            ]
            with self.assertRaisesRegex(RuntimeError, "CUDA extension execution failed"):
                run_logged(command, log_path)
            self.assertIn("no kernel image", log_path.read_text(encoding="utf-8"))

    def test_normal_zero_exit_output_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "ok.log"
            run_logged([sys.executable, "-c", "print('ok')"], log_path)
            self.assertEqual(log_path.read_text(encoding="utf-8").strip(), "ok")


if __name__ == "__main__":
    unittest.main()
