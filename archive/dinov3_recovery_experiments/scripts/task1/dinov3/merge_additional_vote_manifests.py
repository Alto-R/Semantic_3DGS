#!/usr/bin/env python3
"""Merge existing and newly selected additional camera votes.

The black-evidence selector adds a second round of unused cameras.  This
module combines the existing additional vote manifest with the newly lifted
votes into one cache, and writes a merged automatic-selection report whose
camera lists satisfy the recovery-stage provenance contract.  It copies vote
files, verifies both input manifests, and never changes either source cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SOURCE = "dinov3_merged_additional_vote_cache"
CONTRACT = "complete_dense_argmax_pixels_normalized_per_camera_v1"
SELECTION_CONTRACT = "automatic_visibility_and_pose_diverse_abstention_evidence_v1"
BLACK_EVIDENCE_SELECTION_CONTRACT = "automatic_visibility_black_evidence_selection_v1"


def merge_manifests(
    existing_manifest: dict[str, Any],
    new_manifest: dict[str, Any],
) -> dict[str, Any]:
    for field in (
        "source",
        "contract",
        "gaussian_count",
        "ply_path",
        "render_max_width",
    ):
        if existing_manifest.get(field) != new_manifest.get(field):
            raise ValueError(f"vote manifests differ in {field}")
    if existing_manifest.get("contract") != CONTRACT:
        raise ValueError("existing vote manifest has the wrong contract")
    existing_frames = existing_manifest["frames"]
    new_frames = new_manifest["frames"]
    existing_indices = [int(frame["camera_index"]) for frame in existing_frames]
    new_indices = [int(frame["camera_index"]) for frame in new_frames]
    if len(set(existing_indices)) != len(existing_indices):
        raise ValueError("existing vote manifest repeats a camera")
    if len(set(new_indices)) != len(new_indices):
        raise ValueError("new vote manifest repeats a camera")
    if set(existing_indices) & set(new_indices):
        raise ValueError("new vote manifest repeats an existing additional camera")
    merged = dict(existing_manifest)
    merged["frames"] = [*existing_frames, *new_frames]
    merged["camera_count"] = len(merged["frames"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-vote-dir", required=True, type=Path)
    parser.add_argument("--new-vote-dir", required=True, type=Path)
    parser.add_argument("--existing-selection-report", required=True, type=Path)
    parser.add_argument("--new-selection-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    existing_manifest_path = args.existing_vote_dir / "vote_manifest.json"
    new_manifest_path = args.new_vote_dir / "vote_manifest.json"
    for path in (
        existing_manifest_path,
        new_manifest_path,
        args.existing_selection_report,
        args.new_selection_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
    new_manifest = json.loads(new_manifest_path.read_text(encoding="utf-8"))
    existing_selection = json.loads(
        args.existing_selection_report.read_text(encoding="utf-8")
    )
    new_selection = json.loads(args.new_selection_report.read_text(encoding="utf-8"))
    if existing_selection.get("contract") != SELECTION_CONTRACT:
        raise ValueError("existing selection report has the wrong contract")
    if new_selection.get("contract") not in (
        SELECTION_CONTRACT,
        BLACK_EVIDENCE_SELECTION_CONTRACT,
    ):
        raise ValueError("new selection report has the wrong contract")
    merged = merge_manifests(existing_manifest, new_manifest)

    output_manifest_dir = args.output_dir / "vote_manifest"
    output_manifest_dir.mkdir(parents=True)
    copied = []
    for frame in merged["frames"]:
        source = Path(str(frame["vote_file"]))
        if source.is_absolute():
            raise ValueError("vote manifests must use relative vote_file paths")
        existing_source = (existing_manifest_path.parent / source).resolve()
        new_source = (new_manifest_path.parent / source).resolve()
        if existing_source.is_file():
            selected_source = existing_source
        elif new_source.is_file():
            selected_source = new_source
        else:
            raise FileNotFoundError(source)
        destination = output_manifest_dir / source.name
        shutil.copy2(selected_source, destination)
        frame["vote_file"] = "vote_manifest/" + source.name
        copied.append(str(source.name))
    merged_manifest_path = args.output_dir / "vote_manifest.json"
    merged_manifest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    baseline_indices = [
        int(value) for value in existing_selection.get("baseline_camera_indices", [])
    ]
    additional_indices = [
        int(frame["camera_index"]) for frame in merged["frames"]
    ]
    merged_selection = {
        "source": "dinov3_merged_additional_camera_selection",
        "contract": SELECTION_CONTRACT,
        "merged_selection": True,
        "target_policy": "detected_abstentions_then_current_black_evidence",
        "existing_selection_report": str(args.existing_selection_report),
        "new_selection_report": str(args.new_selection_report),
        "baseline_camera_indices": baseline_indices,
        "additional_camera_indices": additional_indices,
        "existing_additional_camera_indices": [
            int(frame["camera_index"]) for frame in existing_manifest["frames"]
        ],
        "new_additional_camera_indices": [
            int(frame["camera_index"]) for frame in new_manifest["frames"]
        ],
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    merged_selection_path = args.output_dir / "additional_camera_selection_report.json"
    merged_selection_path.write_text(
        json.dumps(merged_selection, indent=2), encoding="utf-8"
    )
    (args.output_dir / "experiment_mode.txt").write_text(
        "mode=merged_additional_vote_cache\n"
        "merged_selection=1\n"
        "accepted_gaussian_labels_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": SOURCE,
                "camera_count": len(merged["frames"]),
                "existing_camera_count": len(existing_manifest["frames"]),
                "new_camera_count": len(new_manifest["frames"]),
                "copied_vote_files": len(copied),
                "vote_manifest": str(merged_manifest_path),
                "selection_report": str(merged_selection_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
