#!/usr/bin/env python3
"""Select one global DINOv3 probability-fusion policy automatically."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.lift_soft_probability_view_votes import (
    CONTRACT as SOFT_VOTE_CONTRACT,
    SOURCE as SOFT_VOTE_SOURCE,
)


SOURCE = "dinov3_calibrated_probability_policy_sweep"
CONTRACT = "automatic_global_calibrated_probability_policy_sweep_v1"
SELECTION_RULE = (
    "maximize_correct_over_all_eligible_leave_one_camera_out_gaussian_"
    "observations_then_prediction_coverage_then_fixed_candidate_order"
)
CONFIDENCE_MODES = ("uniform", "margin", "entropy")

# This is a fixed, scene-neutral candidate family. The full-soft and top-1
# policies anchor the two already measured extremes. The remaining policies
# suppress weak probability tails without introducing per-class thresholds.
FIXED_POLICY_SPECS: tuple[tuple[str, int, float, str], ...] = (
    ("full_t1_uniform", 150, 1.0, "uniform"),
    ("top1_t1_uniform", 1, 1.0, "uniform"),
    ("full_t075_uniform", 150, 0.75, "uniform"),
    ("full_t05_uniform", 150, 0.5, "uniform"),
    ("top2_t1_uniform", 2, 1.0, "uniform"),
    ("top2_t075_uniform", 2, 0.75, "uniform"),
    ("top2_t05_uniform", 2, 0.5, "uniform"),
    ("top3_t1_uniform", 3, 1.0, "uniform"),
    ("top3_t075_uniform", 3, 0.75, "uniform"),
    ("top3_t05_uniform", 3, 0.5, "uniform"),
    ("top5_t075_uniform", 5, 0.75, "uniform"),
    ("top5_t05_uniform", 5, 0.5, "uniform"),
    ("top3_t075_margin", 3, 0.75, "margin"),
    ("top3_t075_entropy", 3, 0.75, "entropy"),
    ("top5_t075_margin", 5, 0.75, "margin"),
    ("top5_t075_entropy", 5, 0.75, "entropy"),
    ("full_t075_margin", 150, 0.75, "margin"),
    ("full_t075_entropy", 150, 0.75, "entropy"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_candidate_policies(class_count: int) -> list[dict[str, Any]]:
    if class_count != 150:
        raise ValueError("the calibrated fusion contract requires ADE20K-150")
    return [
        {
            "id": policy_id,
            "top_k": top_k,
            "temperature": temperature,
            "confidence_weight": confidence,
            "top_k_tie_policy": "retain_all_classes_tied_at_kth_probability",
        }
        for policy_id, top_k, temperature, confidence in FIXED_POLICY_SPECS
    ]


def validate_policy(policy: dict[str, Any], *, class_count: int) -> dict[str, Any]:
    candidates = fixed_candidate_policies(class_count)
    by_id = {candidate["id"]: candidate for candidate in candidates}
    policy_id = str(policy.get("id", ""))
    if policy_id not in by_id or policy != by_id[policy_id]:
        raise ValueError("calibration policy is not one of the fixed candidates")
    return dict(by_id[policy_id])


def load_selected_policy(path: Path, *, class_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("source") != SOURCE or report.get("contract") != CONTRACT:
        raise ValueError("input is not a calibrated probability policy report")
    if report.get("selection_rule") != SELECTION_RULE:
        raise ValueError("calibrated probability selection rule differs")
    expected = fixed_candidate_policies(class_count)
    if report.get("candidate_policies") != expected:
        raise ValueError("calibrated probability candidate family differs")
    selected = validate_policy(report.get("selected_policy", {}), class_count=class_count)
    result_ids = [str(item.get("policy", {}).get("id", "")) for item in report.get("candidate_results", [])]
    if result_ids != [item["id"] for item in expected]:
        raise ValueError("calibrated probability candidate results are incomplete or reordered")
    if report.get("manual_candidate_selection_used") is not False:
        raise ValueError("calibrated probability policy was not selected automatically")
    return selected, report


def _row_top_two(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have at least two classes")
    top = np.partition(values, -2, axis=1)[:, -2:]
    return top[:, 1], top[:, 0]


def calibrate_probability_distribution(
    probabilities: np.ndarray,
    policy: dict[str, Any],
) -> np.ndarray:
    """Apply one fixed global calibration policy to camera/Gaussian rows."""

    raw = np.asarray(probabilities, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 150:
        raise ValueError("calibration requires rows of 150 ADE20K probabilities")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("calibration input is non-finite or negative")
    sums = raw.sum(axis=1, dtype=np.float32)
    if raw.shape[0] and not np.allclose(sums, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError("calibration input rows must sum to one")
    checked = validate_policy(policy, class_count=raw.shape[1])
    normalized = np.divide(
        raw,
        sums[:, None],
        out=np.zeros_like(raw),
        where=sums[:, None] > 0.0,
    )
    top_one, top_two = _row_top_two(normalized)
    confidence_mode = checked["confidence_weight"]
    if confidence_mode == "uniform":
        weights = np.ones(normalized.shape[0], dtype=np.float32)
    elif confidence_mode == "margin":
        weights = np.maximum(top_one - top_two, 0.0).astype(np.float32, copy=False)
    elif confidence_mode == "entropy":
        logarithms = np.zeros_like(normalized)
        np.log(normalized, out=logarithms, where=normalized > 0.0)
        entropy = -np.sum(normalized * logarithms, axis=1, dtype=np.float32)
        weights = np.clip(
            1.0 - entropy / np.float32(np.log(normalized.shape[1])),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
    else:  # pragma: no cover - validate_policy prevents this branch.
        raise ValueError(f"unknown confidence mode: {confidence_mode}")

    transformed = normalized.copy()
    temperature = float(checked["temperature"])
    if temperature != 1.0:
        np.power(transformed, np.float32(1.0 / temperature), out=transformed)
    top_k = int(checked["top_k"])
    if top_k < transformed.shape[1]:
        kth = np.partition(transformed, -top_k, axis=1)[:, -top_k]
        transformed[transformed < kth[:, None]] = 0.0
    transformed_sums = transformed.sum(axis=1, dtype=np.float32)
    transformed = np.divide(
        transformed,
        transformed_sums[:, None],
        out=np.zeros_like(transformed),
        where=transformed_sums[:, None] > 0.0,
    )
    transformed *= weights[:, None]
    return transformed


def select_candidate_result(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_results:
        raise ValueError("candidate results are empty")
    for index, result in enumerate(candidate_results):
        if int(result.get("fixed_candidate_order", -1)) != index:
            raise ValueError("candidate results are not in fixed order")
        for key in ("accuracy_of_all_eligible", "prediction_coverage_of_eligible"):
            value = float(result.get(key, -1.0))
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"candidate result has invalid {key}")
    return max(
        candidate_results,
        key=lambda item: (
            float(item["accuracy_of_all_eligible"]),
            float(item["prediction_coverage_of_eligible"]),
            -int(item["fixed_candidate_order"]),
        ),
    )


def _validate_distribution(
    indices: np.ndarray,
    probabilities: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices)
    values = np.asarray(probabilities)
    if idx.ndim != 1 or values.shape != (idx.size, class_count):
        raise ValueError("soft camera evidence has an invalid shape")
    if idx.size and (np.any(idx < 0) or int(idx.max()) >= gaussian_count):
        raise ValueError("soft camera evidence references an invalid Gaussian")
    if idx.size and np.any(idx[1:] <= idx[:-1]):
        raise ValueError("soft camera Gaussian indices must be unique and sorted")
    values32 = values.astype(np.float32)
    if not np.isfinite(values32).all() or np.any(values32 < 0.0):
        raise ValueError("soft camera probabilities are non-finite or negative")
    sums = values32.sum(axis=1, dtype=np.float32)
    if idx.size and not np.allclose(sums, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError("soft camera probabilities do not sum to one")
    values32 = np.divide(
        values32,
        sums[:, None],
        out=np.zeros_like(values32),
        where=sums[:, None] > 0.0,
    )
    return idx.astype(np.int64, copy=False), values32


def _load_frame_distribution(
    vote_root: Path,
    frame: dict[str, Any],
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.load(vote_root / str(frame["gaussian_indices_file"]), mmap_mode="r")
    probabilities = np.load(
        vote_root / str(frame["class_probabilities_file"]), mmap_mode="r"
    )
    return _validate_distribution(
        indices,
        probabilities,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )


def _local_unique_winner(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum, second = _row_top_two(values)
    return np.argmax(values, axis=1).astype(np.int64), maximum > second


def evaluate_policy(
    *,
    policy: dict[str, Any],
    frames: Iterable[dict[str, Any]],
    vote_root: Path,
    gaussian_count: int,
    class_count: int,
    base_camera_counts: np.ndarray,
    temporary_root: Path,
    chunk_size: int,
) -> dict[str, Any]:
    checked = validate_policy(policy, class_count=class_count)
    frame_list = list(frames)
    evidence_path = temporary_root / f"{checked['id']}.evidence.float32"
    evidence = np.memmap(
        evidence_path,
        mode="w+",
        dtype=np.float32,
        shape=(class_count, gaussian_count),
    )
    evidence[:] = 0.0
    support_counts = np.zeros(gaussian_count, dtype=np.uint16)
    for frame in frame_list:
        idx, values = _load_frame_distribution(
            vote_root,
            frame,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        contribution = calibrate_probability_distribution(values, checked)
        support = contribution.sum(axis=1, dtype=np.float32) > 0.0
        evidence[:, idx] += contribution.T
        support_counts[idx[support]] += np.uint16(1)
    evidence.flush()

    aggregate = {"eligible": 0, "predicted": 0, "correct": 0}
    class_eligible = np.zeros(class_count, dtype=np.uint64)
    class_predicted = np.zeros(class_count, dtype=np.uint64)
    class_correct = np.zeros(class_count, dtype=np.uint64)
    per_camera: list[dict[str, Any]] = []
    for frame in frame_list:
        idx, values = _load_frame_distribution(
            vote_root,
            frame,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        contribution = calibrate_probability_distribution(values, checked)
        contribution_support = contribution.sum(axis=1, dtype=np.float32) > 0.0
        target, target_unique = _local_unique_winner(values)
        raw_remaining = base_camera_counts[idx].astype(np.int32) - 1
        eligible = target_unique & (raw_remaining >= 2)
        predicted = np.zeros(idx.size, dtype=bool)
        correct = np.zeros(idx.size, dtype=bool)
        candidate_remaining = support_counts[idx].astype(np.int32)
        candidate_remaining -= contribution_support.astype(np.int32)
        for start in range(0, idx.size, chunk_size):
            end = min(start + chunk_size, idx.size)
            local = np.asarray(evidence[:, idx[start:end]].T, dtype=np.float32).copy()
            local -= contribution[start:end]
            np.maximum(local, 0.0, out=local)
            total = local.sum(axis=1, dtype=np.float32)
            maximum, second = _row_top_two(local)
            local_predicted = (
                eligible[start:end]
                & (candidate_remaining[start:end] >= 2)
                & (total > 0.0)
                & (maximum > second)
            )
            winners = np.argmax(local, axis=1)
            predicted[start:end] = local_predicted
            correct[start:end] = local_predicted & (winners == target[start:end])
        eligible_count = int(np.count_nonzero(eligible))
        predicted_count = int(np.count_nonzero(predicted))
        correct_count = int(np.count_nonzero(correct))
        aggregate["eligible"] += eligible_count
        aggregate["predicted"] += predicted_count
        aggregate["correct"] += correct_count
        np.add.at(class_eligible, target[eligible], np.uint64(1))
        np.add.at(class_predicted, target[predicted], np.uint64(1))
        np.add.at(class_correct, target[correct], np.uint64(1))
        per_camera.append(
            {
                "file": str(frame["file"]),
                "camera_index": int(frame["camera_index"]),
                "eligible_gaussian_observation_count": eligible_count,
                "predicted_gaussian_observation_count": predicted_count,
                "correct_gaussian_observation_count": correct_count,
                "prediction_coverage_of_eligible": (
                    predicted_count / eligible_count if eligible_count else 0.0
                ),
                "accuracy_of_all_eligible": (
                    correct_count / eligible_count if eligible_count else 0.0
                ),
                "agreement_of_predicted": (
                    correct_count / predicted_count if predicted_count else 0.0
                ),
            }
        )
    per_class = [
        {
            "ade20k_class_index": class_index,
            "eligible_gaussian_observation_count": int(class_eligible[class_index]),
            "predicted_gaussian_observation_count": int(class_predicted[class_index]),
            "correct_gaussian_observation_count": int(class_correct[class_index]),
            "accuracy_of_all_eligible": (
                int(class_correct[class_index]) / int(class_eligible[class_index])
                if class_eligible[class_index]
                else 0.0
            ),
        }
        for class_index in range(class_count)
        if class_eligible[class_index]
    ]
    del evidence
    gc.collect()
    evidence_path.unlink()
    eligible_total = aggregate["eligible"]
    predicted_total = aggregate["predicted"]
    return {
        "policy": checked,
        "eligible_gaussian_observation_count": eligible_total,
        "predicted_gaussian_observation_count": predicted_total,
        "correct_gaussian_observation_count": aggregate["correct"],
        "prediction_coverage_of_eligible": (
            predicted_total / eligible_total if eligible_total else 0.0
        ),
        "accuracy_of_all_eligible": (
            aggregate["correct"] / eligible_total if eligible_total else 0.0
        ),
        "agreement_of_predicted": (
            aggregate["correct"] / predicted_total if predicted_total else 0.0
        ),
        "per_camera": per_camera,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-vote-manifest", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=20_000, type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    required = (args.soft_vote_manifest, args.selected_cache_report, args.ontology)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256_file(path) for path in required}
    votes = json.loads(args.soft_vote_manifest.read_text(encoding="utf-8"))
    cache = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    if votes.get("source") != SOFT_VOTE_SOURCE or votes.get("contract") != SOFT_VOTE_CONTRACT:
        raise ValueError("input is not the reviewed soft probability vote cache")
    if votes.get("accepted_gaussian_labels_written") is not False:
        raise ValueError("soft vote cache is not report-only")
    expected = [int(value) for value in cache.get("camera_indices", [])]
    frames = list(votes.get("frames", []))
    if [int(frame["camera_index"]) for frame in frames] != expected:
        raise ValueError("soft vote cameras differ from the automatic selected prefix")
    gaussian_count = int(votes.get("gaussian_count", -1))
    if gaussian_count < 1 or len(frames) < 3:
        raise ValueError("soft vote cache has insufficient Gaussians or cameras")
    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    candidates = fixed_candidate_policies(class_count)
    vote_root = args.soft_vote_manifest.parent
    base_camera_counts = np.zeros(gaussian_count, dtype=np.uint16)
    for frame in frames:
        idx, _values = _load_frame_distribution(
            vote_root,
            frame,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        base_camera_counts[idx] += np.uint16(1)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="calibrated_sweep_", dir=args.output_dir) as temp:
        temporary_root = Path(temp)
        for order, policy in enumerate(candidates):
            print(f"evaluating calibrated fusion candidate {order + 1}/{len(candidates)}: {policy['id']}")
            result = evaluate_policy(
                policy=policy,
                frames=frames,
                vote_root=vote_root,
                gaussian_count=gaussian_count,
                class_count=class_count,
                base_camera_counts=base_camera_counts,
                temporary_root=temporary_root,
                chunk_size=args.chunk_size,
            )
            result["fixed_candidate_order"] = order
            candidate_results.append(result)
            print(
                json.dumps(
                    {
                        "policy": policy["id"],
                        "accuracy_of_all_eligible": result["accuracy_of_all_eligible"],
                        "prediction_coverage_of_eligible": result["prediction_coverage_of_eligible"],
                    },
                    sort_keys=True,
                )
            )
    selected_result = select_candidate_result(candidate_results)
    hashes_after = {str(path): sha256_file(path) for path in required}
    if hashes_before != hashes_after:
        raise RuntimeError("calibration provenance inputs changed during execution")
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "report_only": True,
        "selection_rule": SELECTION_RULE,
        "selection_metric": "correct_predictions_divided_by_all_policy_independent_eligible_observations",
        "development_scope": "one_scene_global_policy_selection",
        "required_external_validation": "apply_selected_policy_unchanged_to_another_scene",
        "soft_vote_manifest": str(args.soft_vote_manifest),
        "selected_cache_report": str(args.selected_cache_report),
        "ontology": str(args.ontology),
        "camera_count": len(frames),
        "gaussian_count": gaussian_count,
        "camera_indices": expected,
        "candidate_policies": candidates,
        "candidate_results": candidate_results,
        "selected_policy": selected_result["policy"],
        "selected_policy_metrics": {
            key: selected_result[key]
            for key in (
                "eligible_gaussian_observation_count",
                "predicted_gaussian_observation_count",
                "correct_gaussian_observation_count",
                "prediction_coverage_of_eligible",
                "accuracy_of_all_eligible",
                "agreement_of_predicted",
            )
        },
        "input_sha256": hashes_before,
        "manual_candidate_selection_used": False,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "calibrated_probability_policy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_policy": report["selected_policy"],
                "selected_policy_metrics": report["selected_policy_metrics"],
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
