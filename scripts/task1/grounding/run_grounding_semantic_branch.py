#!/usr/bin/env python3
"""Run the shared GroundingDINO/SAM, FlashSplat, and semantic-fusion branch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FATAL_OUTPUT_PATTERNS = (
    "no kernel image is available for execution on the device",
    "invalid device function",
)


def run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"===== {log_path.stem} =====", flush=True)
    fatal_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture branch command output")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            if any(pattern in line.lower() for pattern in FATAL_OUTPUT_PATTERNS):
                fatal_lines.append(line.strip())
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if fatal_lines:
        raise RuntimeError(
            "CUDA extension execution failed even though the child process returned zero: "
            + " | ".join(fatal_lines[:3])
        )


def add_optional(command: list[str], flag: str, value: str | Path | None) -> None:
    if value is not None and str(value).strip():
        command.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--ground-output-dir", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--fusion-dir", required=True, type=Path)
    parser.add_argument("--label-map-path", required=True, type=Path)
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--no-semantic-ply", action="store_true")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--class-config", required=True, type=Path)
    parser.add_argument("--include-classes", default="")
    parser.add_argument("--source-view-manifest", type=Path)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--view-count", default=50, type=int)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--groundingdino-root", required=True, type=Path)
    parser.add_argument("--groundingdino-config", required=True, type=Path)
    parser.add_argument("--groundingdino-checkpoint", required=True, type=Path)
    parser.add_argument("--segment-anything-root", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--render-max-width", default=960, type=int)
    parser.add_argument("--max-detections-per-view", default=32, type=int)
    parser.add_argument("--mask-batch-size", default=4, type=int)
    parser.add_argument("--flashsplat-support-threshold", default=0.05, type=float)
    parser.add_argument("--box-threshold", default=0.30, type=float)
    parser.add_argument("--text-threshold", default=0.25, type=float)
    parser.add_argument("--min-assigned-gaussians", default=0, type=int)
    parser.add_argument("--min-assigned-thing-gaussians", default=5000, type=int)
    parser.add_argument("--min-assigned-stuff-gaussians", default=10000, type=int)
    parser.add_argument("--adaptive-stuff-min-view-ratio", default=0.50, type=float)
    parser.add_argument("--adaptive-stuff-threshold-ratio", default=0.75, type=float)
    parser.add_argument("--adaptive-thing-threshold-floor-ratio", default=0.50, type=float)
    parser.add_argument("--min-label-score", default=0.0, type=float)
    parser.add_argument("--skip-mask-generation", action="store_true")
    parser.add_argument("--skip-proposal-generation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.no_semantic_ply and args.semantic_ply_path is not None:
        raise ValueError("--no-semantic-ply and --semantic-ply-path are mutually exclusive")

    python = sys.executable
    if not args.skip_mask_generation:
        command = [
            python,
            "-m",
            "scripts.task1.grounding.generate_grounded_sam_masks",
            "--model-path",
            str(args.model_path),
            "--output-dir",
            str(args.ground_output_dir),
            "--flashsplat-root",
            str(args.flashsplat_root),
            "--groundingdino-root",
            str(args.groundingdino_root),
            "--groundingdino-config",
            str(args.groundingdino_config),
            "--groundingdino-checkpoint",
            str(args.groundingdino_checkpoint),
            "--segment-anything-root",
            str(args.segment_anything_root),
            "--sam-checkpoint",
            str(args.sam_checkpoint),
            "--class-config",
            str(args.class_config),
            "--iteration",
            "30000",
            "--max-width",
            str(args.render_max_width),
            "--camera-indices",
            args.camera_indices,
            "--count",
            str(args.view_count),
            "--max-detections-per-view",
            str(args.max_detections_per_view),
            "--box-threshold",
            str(args.box_threshold),
            "--text-threshold",
            str(args.text_threshold),
            "--min-mask-area",
            "100",
            "--max-mask-area-ratio",
            "0.80",
        ]
        add_optional(command, "--include-classes", args.include_classes)
        add_optional(command, "--source-view-manifest", args.source_view_manifest)
        run_logged(command, args.log_dir / "grounding_01_grounded_sam.log")

    if not args.skip_proposal_generation:
        command = [
            python,
            "-m",
            "scripts.task1.grounding.run_flashsplat_mask_proposals",
            "--model-path",
            str(args.model_path),
            "--sam-output-dir",
            str(args.ground_output_dir),
            "--output-dir",
            str(args.proposal_dir),
            "--manifest-name",
            "grounded_sam_manifest.json",
            "--mask-dir-name",
            "mask_stacks",
            "--flashsplat-root",
            str(args.flashsplat_root),
            "--iteration",
            "30000",
            "--max-width",
            str(args.render_max_width),
            "--mask-batch-size",
            str(args.mask_batch_size),
            "--max-masks-per-view",
            str(args.max_detections_per_view),
            "--support-threshold",
            str(args.flashsplat_support_threshold),
            "--write-class-evidence",
            "--class-evidence-threshold",
            str(args.flashsplat_support_threshold),
            "--min-support-gaussians",
            "250",
        ]
        run_logged(command, args.log_dir / "grounding_02_flashsplat.log")

    command = [
        python,
        "-m",
        "scripts.task1.grounding.cluster_semantic_flashsplat_proposals",
        "--model-path",
        str(args.model_path),
        "--proposal-dir",
        str(args.proposal_dir),
        "--output-dir",
        str(args.fusion_dir),
        "--labels-path",
        str(args.fusion_dir / "gaussian_labels.npy"),
        "--summary-path",
        str(args.fusion_dir / "semantic_group_summary.json"),
        "--label-map-path",
        str(args.label_map_path),
        "--class-config",
        str(args.class_config),
        "--iteration",
        "30000",
        "--scene",
        args.scene,
        "--min-proposal-gaussians",
        "500",
        "--min-group-gaussians",
        "1500",
        "--min-group-proposals",
        "2",
        "--merge-iou",
        "0.35",
        "--containment-threshold",
        "0.70",
        "--max-groups",
        "96",
        "--assignment-reliability-views",
        "2",
        "--assignment-min-quality",
        "0.08",
        "--class-evidence-dir",
        str(args.proposal_dir / "class_evidence"),
        "--class-evidence-min-positive-views",
        "2",
        "--class-evidence-min-ratio",
        "0.50",
        "--spatial-prune-thing-islands",
        "--spatial-voxel-scale-multiplier",
        "4",
        "--spatial-min-voxel-size",
        "0.01",
        "--spatial-max-voxel-size",
        "0.20",
        "--spatial-min-component-gaussians",
        "500",
        "--spatial-min-component-ratio",
        "0.01",
        "--consolidate-thing-instances",
        "--instance-voxel-scale-multiplier",
        "4",
        "--instance-min-voxel-size",
        "0.01",
        "--instance-max-voxel-size",
        "0.20",
        "--instance-min-component-gaussians",
        "500",
        "--instance-min-component-ratio",
        "0.01",
        "--min-assigned-gaussians",
        str(args.min_assigned_gaussians),
        "--min-assigned-thing-gaussians",
        str(args.min_assigned_thing_gaussians),
        "--min-assigned-stuff-gaussians",
        str(args.min_assigned_stuff_gaussians),
        "--adaptive-stuff-min-view-ratio",
        str(args.adaptive_stuff_min_view_ratio),
        "--adaptive-stuff-threshold-ratio",
        str(args.adaptive_stuff_threshold_ratio),
        "--adaptive-thing-threshold-floor-ratio",
        str(args.adaptive_thing_threshold_floor_ratio),
        "--min-label-score",
        str(args.min_label_score),
    ]
    if args.no_semantic_ply:
        command.append("--no-semantic-ply")
    else:
        add_optional(command, "--semantic-ply-path", args.semantic_ply_path)
    if args.overwrite:
        command.append("--overwrite")
    run_logged(command, args.log_dir / "grounding_03_semantic_fusion.log")


if __name__ == "__main__":
    main()
