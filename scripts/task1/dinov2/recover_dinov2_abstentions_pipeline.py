#!/usr/bin/env python3
"""End-to-end DINOv2 abstention-recovery materialization for one scene.

This pipeline runs every stage in one invocation and writes a review
materialization:

1. load the DINOv2 per-view FlashSplat votes (weights are not normalized);
2. equal-camera strict-majority consensus -> hard labels + status;
3. leave-one-camera-out reliability calibration (95% Wilson lower bound);
4. calibrated weighted strict-majority recovery for detected abstentions
   (single camera, exact tie, no strict majority);
5. materialize labels, label map, semantic PLY, and a SuperSplat debug PLY;
6. write a summary report.

No inference or vote lifting is rerun.  The output is a review artifact for
visual comparison; it is not auto-accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.dinov2_second_source import (
    SOURCE as DINOV2_SOURCE,
    collapse_dinov2_camera,
)
from scripts.task1.dinov3.materialize_abstention_recovery import build_label_map
from scripts.task1.dinov3.recover_detected_abstentions import (
    unique_weighted_strict_majority,
    wilson_lower_bound,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_ACCEPTED,
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    STATUS_UNOBSERVED,
    consensus_statistics,
    leave_one_out_consensus,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


SOURCE = "dinov2_abstention_recovery_materialization"
CONTRACT = "review_dinov2_abstention_recovery_materialization_v1"
TARGET_STATUSES = (
    STATUS_SINGLE_CAMERA,
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
)
STATUS_NAMES = {
    STATUS_UNOBSERVED: "unobserved",
    STATUS_SINGLE_CAMERA: "single_camera",
    STATUS_EXACT_TIE: "exact_tie",
    STATUS_NO_STRICT_MAJORITY: "no_strict_majority",
    STATUS_ACCEPTED: "accepted_strict_majority",
}


def load_dinov2_votes(
    manifest_path: Path,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[list[dict], np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != DINOV2_SOURCE:
        raise ValueError("vote manifest is not the DINOv2 FlashSplat vote source")
    if int(manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifest Gaussian count differs")
    counts = np.zeros((class_count + 1, gaussian_count), dtype=np.uint16)
    evidence = []
    for frame in manifest["frames"]:
        vote_path = manifest_path.parent / str(frame["vote_file"])
        if not vote_path.is_file():
            raise FileNotFoundError(vote_path)
        with np.load(vote_path, allow_pickle=False) as data:
            indices = np.asarray(data["indices"], dtype=np.int64)
            winners, mass = collapse_dinov2_camera(
                data["indices"],
                data["class_ids"],
                data["weights"],
                gaussian_count=gaussian_count,
                class_count=class_count,
            )
            totals = np.bincount(
                indices, weights=np.asarray(data["weights"], dtype=np.float32),
                minlength=gaussian_count,
            )
            norm_mass = np.divide(
                mass,
                totals,
                out=np.zeros_like(mass),
                where=totals > 0.0,
            )
        supported = np.flatnonzero(winners)
        if supported.size:
            np.add.at(
                counts,
                (winners[supported].astype(np.int64), supported),
                1,
            )
        evidence.append(
            {
                "camera_index": int(frame["camera_index"]),
                "camera_id": int(frame["camera_id"]),
                "winners": winners,
                "norm_mass": norm_mass,
            }
        )
    return evidence, counts


def camera_reliabilities(
    statistics: dict,
    evidence: list[dict],
    *,
    z: float,
) -> dict[int, float]:
    reliabilities = {}
    for item in evidence:
        winners = item["winners"]
        loo_labels, _ = leave_one_out_consensus(statistics, winners)
        compared = (loo_labels > 0) & (winners > 0)
        trials = int(np.count_nonzero(compared))
        if trials == 0:
            reliabilities[item["camera_index"]] = 0.0
            continue
        agreed = int(np.count_nonzero(compared & (loo_labels == winners)))
        reliabilities[item["camera_index"]] = wilson_lower_bound(
            agreed, trials, z=z
        )
    return reliabilities


def recover_weighted(
    evidence: list[dict],
    reliabilities: dict[int, float],
    status: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.isin(status, np.asarray(TARGET_STATUSES, dtype=np.uint8))
    target_indices = np.flatnonzero(target)
    scores = np.zeros((class_count + 1, target_indices.size), dtype=np.float32)
    camera_count = np.zeros((target_indices.size,), dtype=np.uint16)
    columns = np.arange(target_indices.size, dtype=np.int64)
    for item in evidence:
        winner = item["winners"][target_indices]
        mass = item["norm_mass"][target_indices]
        supported = winner > 0
        if not np.any(supported):
            continue
        contribution = (
            np.float32(reliabilities[item["camera_index"]]) * mass[supported]
        )
        np.add.at(
            scores,
            (winner[supported].astype(np.int64), columns[supported]),
            contribution,
        )
        camera_count[supported] += np.uint16(1)
    resolved = unique_weighted_strict_majority(scores)
    accepted = resolved["accepted"] & (camera_count >= 2)
    recovered_labels = np.zeros((gaussian_count,), dtype=np.uint16)
    recovered_labels[target_indices[accepted]] = resolved["prediction"][accepted]
    source_codes = np.zeros((gaussian_count,), dtype=np.uint8)
    source_codes[status == STATUS_ACCEPTED] = 1
    source_codes[target_indices[accepted]] = 2
    return recovered_labels, source_codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--vote-manifest", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    parser.add_argument("--wilson-z", default=1.959963984540054, type=float)
    parser.add_argument("--no-supersplat", action="store_true")
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    header = read_vertex_count(args.source_ply)
    gaussian_count = header

    print("loading DINOv2 votes...", flush=True)
    evidence, counts = load_dinov2_votes(
        args.vote_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    print("computing consensus...", flush=True)
    statistics = consensus_statistics(counts, chunk_size=args.chunk_size)
    status = statistics["status"]
    hard_labels = np.where(
        status == STATUS_ACCEPTED, statistics["winner"], 0
    ).astype(np.uint16)

    print("calibrating camera reliability...", flush=True)
    reliabilities = camera_reliabilities(
        statistics, evidence, z=args.wilson_z
    )
    reliability_values = np.asarray(list(reliabilities.values()), dtype=np.float32)

    print("weighted recovery...", flush=True)
    recovered_labels, source_codes = recover_weighted(
        evidence,
        reliabilities,
        status,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    labels = np.where(recovered_labels > 0, recovered_labels, hard_labels).astype(
        np.uint16
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gaussian_labels.npy", labels)
    np.save(output_dir / "recovery_source_codes.npy", source_codes)
    np.save(output_dir / "original_consensus_status.npy", status)
    label_map = build_label_map(
        args.scene,
        ontology,
        labels,
        vote_manifest=args.vote_manifest,
        source_ply=args.source_ply,
        ontology_path=args.ontology,
    )
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    semantic_ply = output_dir / "semantic_point_cloud.ply"
    write_ply_with_labels(args.source_ply, semantic_ply, labels)

    if not args.no_supersplat:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.task1.qa.export_supersplat_label_colors",
                "--input-ply",
                str(semantic_ply),
                "--label-map",
                str(output_dir / "label_map.json"),
                "--output-ply",
                str(output_dir / "semantic_point_cloud_supersplat_debug.ply"),
                "--color-mode",
                "class",
                "--overwrite",
            ],
            check=True,
        )

    status_counts = {
        STATUS_NAMES[code]: int(np.count_nonzero(status == code))
        for code in sorted(STATUS_NAMES)
    }
    recovered_count = int(np.count_nonzero(source_codes == 2))
    per_class = {
        ontology.by_project_id[int(project_id)].project_class: int(
            np.count_nonzero((labels == project_id) & (source_codes == 2))
        )
        for project_id in np.unique(labels[source_codes == 2])
        if int(project_id) in ontology.by_project_id
    }
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": False,
        "gaussian_count": gaussian_count,
        "camera_count": len(evidence),
        "status_counts": status_counts,
        "recovered_gaussian_count": recovered_count,
        "unlabeled_gaussian_count": int(np.count_nonzero(labels == 0)),
        "accepted_gaussian_count": int(np.count_nonzero(labels > 0)),
        "recovered_per_class": per_class,
        "reliability": {
            "mean": float(reliability_values.mean()) if reliability_values.size else 0.0,
            "min": float(reliability_values.min()) if reliability_values.size else 0.0,
            "zero_cameras": int(np.count_nonzero(reliability_values == 0.0)),
        },
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_rules": False,
        "zero_camera_fill_used": False,
        "propagation_used": False,
        "semantic_labels_written": True,
        "semantic_ply_written": True,
        "review_only": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_mode.txt").write_text(
        "mode=review_dinov2_abstention_recovery_materialization\n"
        "accepted_gaussian_labels_written=1\n"
        "semantic_ply_written=1\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def read_vertex_count(path: Path) -> int:
    with open(path, "rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path} has no PLY header")
            if line.strip() == b"end_header":
                break
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                return int(text.split()[-1])
    raise ValueError(f"{path} has no vertex element")


if __name__ == "__main__":
    main()
