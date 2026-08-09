#!/usr/bin/env python3
"""Leakage-controlled held-out validation of abstention recovery candidates.

The recovery audit creates one diagnostic candidate from all baseline and
additional cameras.  This validator recomputes the candidate for every
original baseline camera after removing that camera's vote, then compares the
candidate and the immutable hard-vote baseline against the cached DINOv3 map.
It writes reports and contact sheets only; no labels or PLY are accepted.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
)
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.recover_detected_abstentions import (
    RECOVERY_ADDITIONAL_STRICT_MAJORITY,
    RECOVERY_CALIBRATED_WEIGHTED_MAJORITY,
    RECOVERY_LOCKED_ANCHOR,
    TARGET_STATUSES,
    add_hard_camera_counts,
    accumulate_weighted_scores,
    camera_reliability_rows,
    load_camera_evidence,
    unique_weighted_strict_majority,
    validate_vote_manifest,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CACHE_CONTRACT,
    CONTRACT as AUDIT_CONTRACT,
    SOURCE as AUDIT_SOURCE,
    STATUS_ACCEPTED,
    STATUS_NAMES,
    boundary_mask,
    consensus_statistics,
    leave_one_out_consensus,
    quantile_summary,
    render_binary_project_ids,
    save_visuals,
    sha256_file,
    validate_provenance,
)
from scripts.task1.dinov3.select_abstention_evidence_cameras import (
    CONTRACT as SELECTION_CONTRACT,
    SOURCE as SELECTION_SOURCE,
)


SOURCE = "dinov3_detected_abstention_recovery_round_trip_validation"
CONTRACT = "report_only_leave_one_camera_out_recovery_comparison_v1"
METRIC_KEYS = (
    "projected_ratio",
    "agreement_of_projected",
    "interior_agreement_of_projected",
    "boundary_agreement_of_projected",
)


def _metrics() -> Dict[str, int]:
    return {
        "pixels": 0,
        "projected": 0,
        "agreed": 0,
        "boundary_pixels": 0,
        "boundary_projected": 0,
        "boundary_agreed": 0,
        "interior_pixels": 0,
        "interior_projected": 0,
        "interior_agreed": 0,
    }


def _metric_values(values: Mapping[str, int]) -> Dict[str, float]:
    pixels = int(values["pixels"])
    projected = int(values["projected"])
    boundary_pixels = int(values["boundary_pixels"])
    boundary_projected = int(values["boundary_projected"])
    interior_pixels = int(values["interior_pixels"])
    interior_projected = int(values["interior_projected"])
    return {
        **{key: int(value) for key, value in values.items()},
        "projected_ratio": projected / pixels if pixels else 0.0,
        "agreement_of_projected": values["agreed"] / projected if projected else 0.0,
        "boundary_projected_ratio": (
            boundary_projected / boundary_pixels if boundary_pixels else 0.0
        ),
        "boundary_agreement_of_projected": (
            values["boundary_agreed"] / boundary_projected
            if boundary_projected
            else 0.0
        ),
        "interior_projected_ratio": (
            interior_projected / interior_pixels if interior_pixels else 0.0
        ),
        "interior_agreement_of_projected": (
            values["interior_agreed"] / interior_projected
            if interior_projected
            else 0.0
        ),
    }


def update_metrics(
    values: Dict[str, int],
    predicted: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
) -> None:
    interior = ~boundary
    agreement = valid & (predicted == source)
    values["pixels"] += int(source.size)
    values["projected"] += int(np.count_nonzero(valid))
    values["agreed"] += int(np.count_nonzero(agreement))
    values["boundary_pixels"] += int(np.count_nonzero(boundary))
    values["boundary_projected"] += int(np.count_nonzero(valid & boundary))
    values["boundary_agreed"] += int(np.count_nonzero(agreement & boundary))
    values["interior_pixels"] += int(np.count_nonzero(interior))
    values["interior_projected"] += int(np.count_nonzero(valid & interior))
    values["interior_agreed"] += int(np.count_nonzero(agreement & interior))


def candidate_without_camera(
    baseline_labels: np.ndarray,
    combined_labels: np.ndarray,
    target_mask: np.ndarray,
    weighted_scores: np.ndarray,
    weighted_contributing: np.ndarray,
    target_indices: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build one fold candidate from evidence that already excludes its camera."""

    candidate = np.asarray(baseline_labels, dtype=np.uint16).copy()
    strict_target = target_mask & (combined_labels > 0)
    candidate[strict_target] = combined_labels[strict_target]

    remaining = target_indices[~strict_target[target_indices]]
    weighted = unique_weighted_strict_majority(weighted_scores)
    target_positions = np.flatnonzero(~strict_target[target_indices])
    if remaining.size != target_positions.size:
        raise RuntimeError("recovery target indexing became inconsistent")
    weighted_accept = weighted["accepted"][target_positions] & (
        weighted_contributing[target_positions] >= 2
    )
    if np.any(weighted_accept):
        candidate[remaining[weighted_accept]] = weighted["prediction"][
            target_positions[weighted_accept]
        ]

    changed = (candidate > 0) & (baseline_labels == 0)
    return candidate, {
        "newly_resolved_count": int(np.count_nonzero(changed)),
        "strict_newly_resolved_count": int(
            np.count_nonzero(strict_target & (baseline_labels == 0))
        ),
        "weighted_newly_resolved_count": int(
            np.count_nonzero(
                (candidate > 0)
                & (baseline_labels == 0)
                & ~strict_target
            )
        ),
    }


def reproduce_recovery_candidate(
    baseline_statistics: Dict[str, np.ndarray],
    combined_statistics: Dict[str, np.ndarray],
    baseline_evidence: List[Dict[str, Any]],
    additional_evidence: List[Dict[str, Any]],
    target_mask: np.ndarray,
    target_indices: np.ndarray,
    *,
    class_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the saved full-evidence recovery output exactly."""

    status = baseline_statistics["status"]
    locked_labels = np.where(
        status == STATUS_ACCEPTED,
        baseline_statistics["winner"],
        0,
    ).astype(np.uint16)
    labels = locked_labels.copy()
    source_codes = np.zeros(status.shape, dtype=np.uint8)
    source_codes[status == STATUS_ACCEPTED] = RECOVERY_LOCKED_ANCHOR
    strict_fill = target_mask & (combined_statistics["status"] == STATUS_ACCEPTED)
    labels[strict_fill] = combined_statistics["winner"][strict_fill]
    source_codes[strict_fill] = RECOVERY_ADDITIONAL_STRICT_MAJORITY

    reliability_rows = camera_reliability_rows(
        baseline_evidence,
        additional_evidence,
        baseline_statistics,
        locked_labels,
    )
    reliabilities = {
        int(row["camera_index"]): float(row["reliability_weight"])
        for row in reliability_rows
    }
    remaining = target_indices[~strict_fill[target_indices]]
    scores, contributing = accumulate_weighted_scores(
        [*baseline_evidence, *additional_evidence],
        reliabilities,
        remaining,
        class_count=class_count,
    )
    weighted = unique_weighted_strict_majority(scores)
    accepted = weighted["accepted"] & (contributing >= 2)
    selected = remaining[accepted]
    labels[selected] = weighted["prediction"][accepted]
    source_codes[selected] = RECOVERY_CALIBRATED_WEIGHTED_MAJORITY
    return labels, source_codes


def _validate_report_contract(
    report: Mapping[str, Any],
    *,
    scene: str,
    gaussian_count: int,
) -> None:
    if report.get("source") != "dinov3_detected_abstention_recovery_audit":
        raise ValueError("recovery report has the wrong source")
    if report.get("contract") != "immutable_hard_anchor_incremental_strict_then_calibrated_v1":
        raise ValueError("recovery report has the wrong contract")
    if report.get("scene") != scene or not report.get("report_only"):
        raise ValueError("recovery report has the wrong scene or mode")
    if int(report.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("recovery report has a different Gaussian count")
    for field in (
        "immutable_anchor_labels_changed",
        "manual_camera_selection_used",
        "manual_gaussian_selection_used",
        "accepted_gaussian_labels_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if report.get(field) not in (0, False):
            raise ValueError("recovery report violates " + field)


def _validate_baseline_reproduction(
    diagnostics: Mapping[str, np.ndarray],
    statistics: Dict[str, np.ndarray],
    audit: Mapping[str, Any],
) -> np.ndarray:
    for key, diagnostic_key in (
        ("total", "semantic_camera_count"),
        ("maximum", "winner_camera_count"),
        ("status", "consensus_status"),
    ):
        if not np.array_equal(statistics[key], diagnostics[diagnostic_key]):
            raise RuntimeError("baseline hard consensus does not reproduce " + key)
    status = statistics["status"]
    expected_counts = audit.get("gaussian_agreement", {}).get("status_counts")
    actual_counts = {
        STATUS_NAMES[code]: int(np.count_nonzero(status == code))
        for code in sorted(STATUS_NAMES)
    }
    if actual_counts != expected_counts:
        raise RuntimeError("hard diagnostics do not reproduce audit status counts")
    return status


def _validate_baseline_metrics(
    actual: Mapping[str, float], expected: Mapping[str, Any]
) -> None:
    for key in (
        "projected_ratio",
        "agreement_of_projected",
        "boundary_agreement_of_projected",
        "interior_agreement_of_projected",
    ):
        if abs(float(actual[key]) - float(expected[key])) > 2e-5:
            raise RuntimeError("baseline round-trip metric does not reproduce " + key)


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-dir", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--additional-vote-manifest", required=True, type=Path)
    parser.add_argument("--hard-audit-report", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path)
    parser.add_argument("--candidate-labels", required=True, type=Path)
    parser.add_argument("--recovery-source-codes", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required = (
        args.source_view_dir / "view_manifest.json",
        args.source_view_dir / "dinov3_manifest.json",
        args.selected_cache_report,
        args.baseline_vote_manifest,
        args.additional_vote_manifest,
        args.hard_audit_report,
        args.hard_diagnostics,
        args.selection_report,
        args.recovery_report,
        args.candidate_labels,
        args.recovery_source_codes,
        args.ontology,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    cache_report = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    view_manifest = json.loads(
        (args.source_view_dir / "view_manifest.json").read_text(encoding="utf-8")
    )
    dino_manifest = json.loads(
        (args.source_view_dir / "dinov3_manifest.json").read_text(encoding="utf-8")
    )
    baseline_manifest = json.loads(
        args.baseline_vote_manifest.read_text(encoding="utf-8")
    )
    additional_manifest = json.loads(
        args.additional_vote_manifest.read_text(encoding="utf-8")
    )
    hard_audit = json.loads(args.hard_audit_report.read_text(encoding="utf-8"))
    selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    recovery_report = json.loads(args.recovery_report.read_text(encoding="utf-8"))
    if cache_report.get("contract") != CACHE_CONTRACT:
        raise ValueError("source cache has the wrong contract")
    baseline_indices = validate_provenance(
        cache_report, view_manifest, dino_manifest, baseline_manifest
    )
    validate_vote_manifest(baseline_manifest, name="baseline")
    validate_vote_manifest(additional_manifest, name="additional")
    gaussian_count = int(baseline_manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(additional_manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifests have different Gaussian counts")
    additional_indices = [
        int(frame["camera_index"]) for frame in additional_manifest["frames"]
    ]
    if set(baseline_indices) & set(additional_indices):
        raise ValueError("additional vote manifest repeats a baseline camera")
    if Path(str(baseline_manifest.get("ply_path", ""))).resolve() != Path(
        str(additional_manifest.get("ply_path", ""))
    ).resolve():
        raise ValueError("vote manifests belong to different Gaussian PLYs")
    if hard_audit.get("source") != AUDIT_SOURCE or hard_audit.get("contract") != AUDIT_CONTRACT:
        raise ValueError("hard audit has the wrong contract")
    if Path(str(hard_audit.get("vote_manifest", ""))).resolve() != args.baseline_vote_manifest.resolve():
        raise ValueError("hard audit belongs to a different baseline vote manifest")
    if [int(value) for value in hard_audit.get("camera_indices", [])] != baseline_indices:
        raise ValueError("hard audit camera list differs from the selected cache")
    _validate_report_contract(recovery_report, scene=args.scene, gaussian_count=gaussian_count)
    if selection_report.get("source") != SELECTION_SOURCE or selection_report.get(
        "contract"
    ) != SELECTION_CONTRACT:
        raise ValueError("additional camera selection report has the wrong contract")
    if [int(value) for value in selection_report.get("baseline_camera_indices", [])] != baseline_indices:
        raise ValueError("selection report has different baseline cameras")
    if [int(value) for value in selection_report.get("additional_camera_indices", [])] != additional_indices:
        raise ValueError("selection report differs from the additional vote manifest")
    if [int(value) for value in recovery_report.get("baseline_camera_indices", [])] != baseline_indices:
        raise ValueError("recovery report has different baseline cameras")
    if [int(value) for value in recovery_report.get("additional_camera_indices", [])] != additional_indices:
        raise ValueError("recovery report differs from the additional vote manifest")
    if Path(str(recovery_report.get("additional_camera_selection_report", ""))).resolve() != args.selection_report.resolve():
        raise ValueError("recovery report belongs to a different camera selection")

    candidate_labels = np.load(args.candidate_labels, allow_pickle=False)
    source_codes = np.load(args.recovery_source_codes, allow_pickle=False)
    if candidate_labels.shape != (gaussian_count,) or source_codes.shape != (gaussian_count,):
        raise ValueError("recovery arrays have the wrong shape")
    if candidate_labels.dtype.kind not in "ui" or source_codes.dtype.kind not in "ui":
        raise ValueError("recovery arrays must be integer arrays")
    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    if np.any(candidate_labels > class_count):
        raise ValueError("recovery candidate contains a class outside the ontology")
    baseline_evidence = load_camera_evidence(
        args.baseline_vote_manifest,
        baseline_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    additional_evidence = load_camera_evidence(
        args.additional_vote_manifest,
        additional_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    # The scheduler creates the output root for its log, but the report stage
    # itself is intentionally created by this report-only module.  Ensure the
    # temporary evidence workspace has a valid parent on a fresh run.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix="recovery_round_trip_", dir=args.output_dir.parent
    )
    counts = np.memmap(
        Path(temporary.name) / "baseline_counts.uint8",
        mode="w+",
        dtype=np.uint8,
        shape=(class_count + 1, gaussian_count),
    )
    counts[:] = 0
    for item in baseline_evidence:
        add_hard_camera_counts(counts, item["winners"])
    baseline_statistics = consensus_statistics(counts, chunk_size=args.chunk_size)
    with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
        status = _validate_baseline_reproduction(diagnostics, baseline_statistics, hard_audit)
    locked = baseline_statistics["winner"].copy()
    for item in additional_evidence:
        add_hard_camera_counts(counts, item["winners"])
    combined_statistics = consensus_statistics(counts, chunk_size=args.chunk_size)
    counts.flush()

    target_mask = np.isin(status, np.asarray(TARGET_STATUSES, dtype=np.uint8))
    target_indices = np.flatnonzero(target_mask).astype(np.int64)
    locked_labels = np.where(status == STATUS_ACCEPTED, locked, 0).astype(np.uint16)
    if np.any(candidate_labels[status == STATUS_ACCEPTED] != locked_labels[status == STATUS_ACCEPTED]):
        raise ValueError("recovery candidate changed an immutable anchor")
    if np.any(source_codes[status == STATUS_ACCEPTED] != RECOVERY_LOCKED_ANCHOR):
        raise ValueError("recovery source codes do not lock accepted anchors")
    reproduced_labels, reproduced_codes = reproduce_recovery_candidate(
        baseline_statistics,
        combined_statistics,
        baseline_evidence,
        additional_evidence,
        target_mask,
        target_indices,
        class_count=class_count,
    )
    if not np.array_equal(candidate_labels, reproduced_labels):
        raise ValueError("saved recovery candidate does not reproduce from its vote caches")
    if not np.array_equal(source_codes, reproduced_codes):
        raise ValueError("saved recovery source codes do not reproduce from its vote caches")
    recovered_count = int(
        np.count_nonzero(
            reproduced_codes >= RECOVERY_ADDITIONAL_STRICT_MAJORITY
        )
    )
    if recovered_count != int(recovery_report["recovery"]["total_recovered_count"]):
        raise ValueError("recovery report count does not reproduce from its arrays")

    # Keep the baseline count matrix available for leakage-controlled folds.
    # The combined statistics above were computed after temporarily adding the
    # additional-camera evidence; restore the baseline-only matrix here.
    for item in additional_evidence:
        winners = item["winners"]
        supported = np.flatnonzero(winners).astype(np.int64)
        counts[winners[supported], supported] -= np.uint8(1)

    modules = load_flashsplat(args.flashsplat_root)
    cameras = load_cameras(args.model_path)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    if Path(str(baseline_manifest.get("ply_path", ""))).resolve() != ply_path.resolve():
        raise ValueError("vote manifest belongs to a different model PLY")
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    if int(gaussians.get_xyz.shape[0]) != gaussian_count:
        raise ValueError("model Gaussian count differs from vote manifest")
    pipeline = default_pipeline()
    background = background_tensor(False)
    rgb_dir = args.source_view_dir / "rgb_renders"
    lookup = ontology.ade_to_project
    baseline_values = _metrics()
    candidate_values = _metrics()
    per_camera: List[Dict[str, Any]] = []
    group_counts = {
        STATUS_NAMES[code]: {"baseline": 0, "candidate": 0}
        for code in TARGET_STATUSES
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_overlay_dir = args.output_dir / "candidate_heldout_overlays"
    candidate_disagreement_dir = args.output_dir / "candidate_heldout_disagreement"
    candidate_overlay_dir.mkdir()
    candidate_disagreement_dir.mkdir()

    for frame in baseline_manifest["frames"]:
        camera_index = int(frame["camera_index"])
        evidence = next(item for item in baseline_evidence if item["camera_index"] == camera_index)
        baseline_labels, remaining_views = leave_one_out_consensus(
            baseline_statistics, evidence["winners"]
        )
        supported = np.flatnonzero(evidence["winners"]).astype(np.int64)
        counts[evidence["winners"][supported], supported] -= np.uint8(1)
        try:
            fold_statistics = consensus_statistics(counts, chunk_size=args.chunk_size)
        finally:
            counts[evidence["winners"][supported], supported] += np.uint8(1)
        fold_labels = np.where(
            fold_statistics["status"] == STATUS_ACCEPTED,
            fold_statistics["winner"],
            0,
        ).astype(np.uint16)
        if not np.array_equal(fold_labels, baseline_labels):
            raise RuntimeError("fold baseline consensus does not reproduce leave-one-out labels")
        remaining_baseline = [
            item for item in baseline_evidence if item["camera_index"] != camera_index
        ]
        fold_locked = np.where(
            (status == STATUS_ACCEPTED) & (fold_labels > 0),
            fold_labels,
            0,
        ).astype(np.uint16)
        fold_rows = camera_reliability_rows(
            remaining_baseline,
            additional_evidence,
            fold_statistics,
            fold_locked,
        )
        fold_reliabilities = {
            int(row["camera_index"]): float(row["reliability_weight"])
            for row in fold_rows
        }
        fold_scores, fold_contributing = accumulate_weighted_scores(
            [*remaining_baseline, *additional_evidence],
            fold_reliabilities,
            target_indices,
            class_count=class_count,
        )
        combined_labels, _ = leave_one_out_consensus(
            combined_statistics, evidence["winners"]
        )
        candidate, resolution = candidate_without_camera(
            baseline_labels,
            combined_labels,
            target_mask,
            fold_scores,
            fold_contributing,
            target_indices,
        )
        camera = make_camera(cameras[camera_index], modules, int(cache_report["render"]["max_width"]))
        baseline_predicted, baseline_valid, _ = render_binary_project_ids(
            baseline_labels, camera, gaussians, modules, pipeline, background,
            class_count=class_count,
        )
        candidate_predicted, candidate_valid, candidate_margin = render_binary_project_ids(
            candidate, camera, gaussians, modules, pipeline, background,
            class_count=class_count,
        )
        dino_frame = next(
            item for item in dino_manifest["frames"]
            if int(item["camera_index"]) == camera_index
        )
        with np.load(args.source_view_dir / str(dino_frame["segment_file"]), allow_pickle=False) as segment:
            source_project = lookup[np.asarray(segment["class_id"], dtype=np.uint8)]
        if source_project.shape != baseline_predicted.shape:
            raise ValueError("held-out DINO map and 3D projection shapes differ")
        boundary = boundary_mask(source_project)
        update_metrics(baseline_values, baseline_predicted, baseline_valid, source_project, boundary)
        update_metrics(candidate_values, candidate_predicted, candidate_valid, source_project, boundary)
        base_rgb = np.asarray(Image.open(rgb_dir / str(frame["file"])).convert("RGB"), dtype=np.uint8)
        save_visuals(
            base_rgb,
            candidate_predicted,
            candidate_valid,
            source_project,
            ontology,
            candidate_overlay_dir / str(frame["file"]),
            candidate_disagreement_dir / str(frame["file"]),
        )
        for code in TARGET_STATUSES:
            name = STATUS_NAMES[code]
            eligible = status == code
            group_counts[name]["baseline"] += int(np.count_nonzero(eligible & (baseline_labels > 0)))
            group_counts[name]["candidate"] += int(np.count_nonzero(eligible & (candidate > 0)))
        per_camera.append(
            {
                "camera_index": camera_index,
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                "baseline_projected_ratio": int(np.count_nonzero(baseline_valid)) / int(source_project.size),
                "candidate_projected_ratio": int(np.count_nonzero(candidate_valid)) / int(source_project.size),
                "baseline_agreement_of_projected": int(np.count_nonzero(baseline_valid & (baseline_predicted == source_project))) / int(np.count_nonzero(baseline_valid)) if np.any(baseline_valid) else 0.0,
                "candidate_agreement_of_projected": int(np.count_nonzero(candidate_valid & (candidate_predicted == source_project))) / int(np.count_nonzero(candidate_valid)) if np.any(candidate_valid) else 0.0,
                "candidate_binary_margin": quantile_summary(candidate_margin[candidate_valid]),
                **resolution,
                "remaining_camera_evidence": quantile_summary(remaining_views[baseline_labels > 0]),
            }
        )
        print(
            "held out camera %s: baseline=%.4f candidate=%.4f"
            % (
                camera_index,
                per_camera[-1]["baseline_agreement_of_projected"],
                per_camera[-1]["candidate_agreement_of_projected"],
            )
        )

    baseline_metrics = _metric_values(baseline_values)
    candidate_metrics = _metric_values(candidate_values)
    _validate_baseline_metrics(
        baseline_metrics,
        hard_audit["heldout_pixel_metrics"],
    )
    delta = {
        key: float(candidate_metrics[key]) - float(baseline_metrics[key])
        for key in METRIC_KEYS
    }
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": True,
        "gaussian_count": gaussian_count,
        "camera_count": len(baseline_indices),
        "camera_indices": baseline_indices,
        "heldout_policy": "exclude_each_original_baseline_camera_from_baseline_and_recovery_evidence",
        "heldout_calibration_policy": "recompute_all_camera_reliabilities_after_excluding_the_heldout_camera",
        "baseline_audit_report": str(args.hard_audit_report),
        "recovery_report": str(args.recovery_report),
        "recovery_candidate_reproduced": True,
        "recovery_candidate_sha256": sha256_file(args.candidate_labels),
        "recovery_source_codes_sha256": sha256_file(args.recovery_source_codes),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "group_label_counts": group_counts,
        "per_camera": per_camera,
        "candidate_recovered_count": recovered_count,
        "candidate_recovered_ratio": float(
            recovery_report["recovery"]["total_recovered_ratio_of_detected_abstentions"]
        ),
        "coverage_non_regression": candidate_metrics["projected_ratio"]
        >= baseline_metrics["projected_ratio"] - 1e-6,
        "overall_non_regression": candidate_metrics["agreement_of_projected"]
        >= baseline_metrics["agreement_of_projected"] - 1e-6,
        "interior_non_regression": candidate_metrics["interior_agreement_of_projected"]
        >= baseline_metrics["interior_agreement_of_projected"] - 1e-6,
        "boundary_non_regression": candidate_metrics["boundary_agreement_of_projected"]
        >= baseline_metrics["boundary_agreement_of_projected"] - 1e-6,
        "immutable_anchor_labels_changed": 0,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "accepted_gaussian_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    (args.output_dir / "recovery_round_trip_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "experiment_mode.txt").write_text(
        "mode=report_only_leave_one_out_recovery_comparison\n"
        "accepted_gaussian_labels_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "scene", "camera_count", "baseline_metrics", "candidate_metrics", "delta",
        "coverage_non_regression", "overall_non_regression", "interior_non_regression",
        "boundary_non_regression",
    )}, indent=2))
    del counts
    temporary.cleanup()


if __name__ == "__main__":
    main()
