#!/usr/bin/env python3
"""Materialize an audited DINOv3 hard-camera consensus as semantic PLY data.

This command deliberately reuses the hard-camera collapse and strict-majority
consensus functions from the report-only round-trip audit.  It does not fill,
propagate, or manually override abstentions: every Gaussian rejected by the
audited policy remains label 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CONTRACT as AUDIT_CONTRACT,
    SOURCE as AUDIT_SOURCE,
    STATUS_ACCEPTED,
    STATUS_NAMES,
    VOTE_CONTRACT,
    VOTE_SOURCE,
    collapse_camera_distribution,
    consensus_statistics,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


SOURCE = "dinov3_audited_hard_vote_consensus_materialization"
CONTRACT = "exact_report_audited_strict_majority_materialization_v1"
CAMERA_VOTE_POLICY = "one_unique_max_class_per_camera_and_gaussian_else_abstain"
CAMERA_VOTE_SCALE = "one_equal_identity_vote_per_camera"
CONSENSUS_POLICY = "at_least_two_cameras_unique_strict_majority_else_abstain"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_sha256(audit: dict[str, Any], path: Path) -> str:
    hashes = audit.get("input_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("audit report is missing input_sha256")
    resolved = path.resolve()
    matches = [
        str(value)
        for raw_path, value in hashes.items()
        if Path(str(raw_path)).resolve() == resolved
    ]
    if len(matches) != 1:
        raise ValueError(f"audit report does not uniquely record the hash of {path}")
    return matches[0]


def status_counts(status: np.ndarray) -> dict[str, int]:
    values = np.asarray(status)
    return {
        STATUS_NAMES[code]: int(np.count_nonzero(values == code))
        for code in sorted(STATUS_NAMES)
    }


def add_camera_winners(camera_counts: np.ndarray, winners: np.ndarray) -> int:
    """Add one equal identity vote for every nonzero camera winner."""

    counts = np.asarray(camera_counts)
    identities = np.asarray(winners)
    if counts.ndim != 2 or identities.ndim != 1 or counts.shape[1] != identities.size:
        raise ValueError("camera count matrix and winner array are incompatible")
    supported = np.flatnonzero(identities).astype(np.int64)
    if supported.size:
        classes = identities[supported].astype(np.int64, copy=False)
        if np.any(classes <= 0) or int(classes.max()) >= counts.shape[0]:
            raise ValueError("camera winner references an invalid project class")
        counts[classes, supported] += np.uint8(1)
    return int(supported.size)


def labels_from_statistics(statistics: dict[str, np.ndarray]) -> np.ndarray:
    """Return strict-majority winners and keep every other status unlabeled."""

    winner = np.asarray(statistics["winner"], dtype=np.uint16)
    status = np.asarray(statistics["status"], dtype=np.uint8)
    if winner.shape != status.shape or winner.ndim != 1:
        raise ValueError("consensus winner and status arrays must be aligned")
    return np.where(status == STATUS_ACCEPTED, winner, 0).astype(np.int32)


def validate_inputs(
    audit: dict[str, Any],
    vote_manifest: dict[str, Any],
    audit_report_path: Path,
    vote_manifest_path: Path,
    ontology_path: Path,
    source_ply: Path,
) -> None:
    expected_audit = {
        "source": AUDIT_SOURCE,
        "contract": AUDIT_CONTRACT,
        "report_only": True,
        "camera_vote_policy": CAMERA_VOTE_POLICY,
        "camera_vote_scale": CAMERA_VOTE_SCALE,
        "consensus_policy": CONSENSUS_POLICY,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    for field, expected in expected_audit.items():
        if audit.get(field) != expected:
            raise ValueError(f"audit report violates {field}={expected!r}")

    if Path(str(audit.get("vote_manifest", ""))).resolve() != vote_manifest_path.resolve():
        raise ValueError("audit report belongs to a different vote manifest")
    if recorded_sha256(audit, vote_manifest_path) != sha256_file(vote_manifest_path):
        raise ValueError("vote manifest has changed since the audit")
    audit_ontology_path = Path(str(audit.get("ontology", "")))
    if not str(audit.get("ontology", "")):
        raise ValueError("audit report does not record its ontology path")
    if recorded_sha256(audit, audit_ontology_path) != sha256_file(ontology_path):
        raise ValueError("ontology has changed since the audit")

    if vote_manifest.get("source") != VOTE_SOURCE:
        raise ValueError("vote manifest has the wrong source")
    if vote_manifest.get("contract") != VOTE_CONTRACT:
        raise ValueError("vote manifest has the wrong contract")
    for field, expected in (
        ("query_region_filtering_used", False),
        ("confidence_threshold_used", False),
        ("one_normalized_vote_per_camera", True),
        ("v5_used", False),
        ("dinov2_used", False),
    ):
        if vote_manifest.get(field) is not expected:
            raise ValueError(f"vote manifest violates {field}={expected}")
    if Path(str(vote_manifest.get("ply_path", ""))).resolve() != source_ply.resolve():
        raise ValueError("vote manifest belongs to a different source PLY")
    frames = vote_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("vote manifest has no camera frames")
    if int(vote_manifest.get("camera_count", -1)) != len(frames):
        raise ValueError("vote manifest camera count differs from its frames")
    if int(audit.get("camera_count", -1)) != len(frames):
        raise ValueError("audit camera count differs from the vote manifest")
    if audit_report_path.resolve() == vote_manifest_path.resolve():
        raise ValueError("audit report and vote manifest cannot be the same file")


def build_label_map(
    scene: str,
    ontology: Any,
    labels: np.ndarray,
    audit_report: Path,
    vote_manifest: Path,
    ontology_path: Path,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    for raw_id in np.unique(labels[labels > 0]):
        item = ontology.by_project_id[int(raw_id)]
        items.append(
            {
                "id": int(raw_id),
                "name": item.project_class,
                "class": item.project_class,
                "project_id": item.project_id,
                "ade_id": item.ade_id,
                "type": item.kind,
            }
        )
    return {
        "scene": scene,
        "source": SOURCE,
        "contract": CONTRACT,
        "audit_report": str(audit_report),
        "vote_manifest": str(vote_manifest),
        "ontology": str(ontology_path),
        "labels": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--vote-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.source_ply, args.audit_report, args.vote_manifest, args.ontology):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    vote_manifest = json.loads(args.vote_manifest.read_text(encoding="utf-8"))
    validate_inputs(
        audit,
        vote_manifest,
        args.audit_report,
        args.vote_manifest,
        args.ontology,
        args.source_ply,
    )
    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    gaussian_count = int(vote_manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(audit.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("audit and vote manifest Gaussian counts differ")
    header = read_ply_header(args.source_ply)
    vertex = header.element("vertex")
    if vertex is None or vertex.count != gaussian_count:
        raise ValueError("source PLY vertex count differs from audited Gaussian count")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    statistics: dict[str, np.ndarray]
    camera_summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hard_vote_counts_", dir=args.output_dir) as temporary:
        camera_counts = np.memmap(
            Path(temporary) / "camera_counts.uint8",
            mode="w+",
            dtype=np.uint8,
            shape=(class_count + 1, gaussian_count),
        )
        camera_counts[:] = 0
        for frame in vote_manifest["frames"]:
            vote_path = args.vote_manifest.parent / str(frame["vote_file"])
            if not vote_path.is_file():
                raise FileNotFoundError(vote_path)
            with np.load(vote_path, allow_pickle=False) as data:
                winners, collapse_summary = collapse_camera_distribution(
                    data["indices"],
                    data["class_ids"],
                    data["weights"],
                    gaussian_count=gaussian_count,
                    class_count=class_count,
                )
            winner_count = add_camera_winners(camera_counts, winners)
            camera_summaries.append(
                {
                    "camera_index": int(frame["camera_index"]),
                    "camera_id": int(frame["camera_id"]),
                    "file": str(frame["file"]),
                    "unique_hard_winner_count": winner_count,
                    **collapse_summary,
                }
            )
            print(f"collapsed camera {frame['camera_index']}: {winner_count} unique winners")
        camera_counts.flush()
        statistics = consensus_statistics(camera_counts, chunk_size=args.chunk_size)
        del camera_counts

    labels = labels_from_statistics(statistics)
    computed_status_counts = status_counts(statistics["status"])
    expected_status_counts = audit.get("gaussian_agreement", {}).get("status_counts")
    if computed_status_counts != expected_status_counts:
        raise RuntimeError(
            "materialized consensus status counts differ from the audited hard-vote result"
        )

    np.save(args.output_dir / "gaussian_labels.npy", labels)
    np.save(args.output_dir / "gaussian_project_class_ids.npy", labels)
    np.save(args.output_dir / "semantic_camera_count.npy", statistics["total"])
    np.save(args.output_dir / "winner_camera_count.npy", statistics["maximum"])
    np.save(args.output_dir / "winner_share.npy", statistics["winner_share"].astype(np.float32))
    np.save(args.output_dir / "abstain_reason_codes.npy", statistics["status"])

    label_map = build_label_map(
        args.scene,
        ontology,
        labels,
        args.audit_report,
        args.vote_manifest,
        args.ontology,
    )
    label_map_path = args.output_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    partial_ply = semantic_ply.with_suffix(".ply.partial")
    write_ply_with_labels(args.source_ply, partial_ply, labels)
    partial_ply.replace(semantic_ply)

    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(*np.unique(labels[labels > 0], return_counts=True))
    }
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "audit_report": str(args.audit_report),
        "audit_report_sha256": sha256_file(args.audit_report),
        "vote_manifest": str(args.vote_manifest),
        "vote_manifest_sha256": sha256_file(args.vote_manifest),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "camera_count": len(vote_manifest["frames"]),
        "gaussian_count": gaussian_count,
        "camera_vote_policy": CAMERA_VOTE_POLICY,
        "camera_vote_scale": CAMERA_VOTE_SCALE,
        "consensus_policy": CONSENSUS_POLICY,
        "unaccepted_policy": "label_zero_without_fill_or_propagation",
        "status_counts": computed_status_counts,
        "accepted_gaussian_count": int(np.count_nonzero(labels)),
        "accepted_gaussian_ratio": float(np.count_nonzero(labels) / labels.size),
        "unlabeled_gaussian_count": int(np.count_nonzero(labels == 0)),
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "per_camera": camera_summaries,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "audit_status_counts_reproduced_exactly": True,
        "semantic_labels_written": True,
        "label_map_written": True,
        "semantic_ply_written": True,
    }
    (args.output_dir / "hard_vote_materialization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_camera"}, indent=2))


if __name__ == "__main__":
    main()
