"""Pure helpers for FlashSplat-backed DINOv2 multi-view voting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def sparse_view_votes(
    used_count: np.ndarray,
    project_class_ids: np.ndarray,
    mean_confidences: np.ndarray,
    view_quality: float,
    support_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-class FlashSplat mass into normalized sparse votes.

    Row zero is the abstain class. ``project_class_ids`` maps each local row to
    its global project id and must therefore also start with zero.
    """

    used = np.asarray(used_count, dtype=np.float32)
    class_ids = np.asarray(project_class_ids, dtype=np.uint16)
    confidences = np.asarray(mean_confidences, dtype=np.float32)
    if used.ndim != 2:
        raise ValueError("used_count must have shape (classes, gaussians)")
    if class_ids.shape != (used.shape[0],):
        raise ValueError("project_class_ids must match the used_count class axis")
    if confidences.shape != (used.shape[0],):
        raise ValueError("mean_confidences must match the used_count class axis")
    if class_ids[0] != 0:
        raise ValueError("local class row zero must be the abstain class")
    if not 0.0 <= view_quality <= 1.0:
        raise ValueError("view_quality must be between zero and one")
    if support_threshold < 0.0:
        raise ValueError("support_threshold must be non-negative")

    visibility = used.sum(axis=0, dtype=np.float32)
    all_indices: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for local_id, project_id in enumerate(class_ids):
        supported = (used[local_id] > support_threshold) & (visibility > 0.0)
        indices = np.flatnonzero(supported).astype(np.uint32)
        if indices.shape[0] == 0:
            continue
        fractions = used[local_id, indices] / visibility[indices]
        weights = fractions * confidences[local_id] * np.float32(view_quality)
        positive = weights > 0.0
        indices = indices[positive]
        weights = weights[positive]
        if indices.shape[0] == 0:
            continue
        all_indices.append(indices)
        all_classes.append(np.full(indices.shape, project_id, dtype=np.uint16))
        all_weights.append(weights.astype(np.float32))

    if not all_indices:
        return (
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.uint16),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.concatenate(all_indices),
        np.concatenate(all_classes),
        np.concatenate(all_weights),
    )


def accumulate_vote_arrays(
    vote_matrix: np.ndarray,
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
) -> None:
    indices = np.asarray(indices)
    class_ids = np.asarray(class_ids)
    weights = np.asarray(weights, dtype=np.float32)
    if not (indices.shape == class_ids.shape == weights.shape):
        raise ValueError("indices, class_ids, and weights must have matching shapes")
    if indices.ndim != 1:
        raise ValueError("vote arrays must be one-dimensional")
    if indices.shape[0] == 0:
        return
    if int(indices.max()) >= vote_matrix.shape[1]:
        raise IndexError("vote references a Gaussian outside the vote matrix")
    if int(class_ids.max()) >= vote_matrix.shape[0]:
        raise IndexError("vote references a class outside the vote matrix")

    for class_id in np.unique(class_ids):
        selected = class_ids == class_id
        class_indices = indices[selected].astype(np.int64, copy=False)
        class_weights = weights[selected]
        np.add.at(vote_matrix[int(class_id)], class_indices, class_weights)


def winner_metrics(votes: np.ndarray, tie_epsilon: float = 0.0) -> tuple[np.ndarray, ...]:
    values = np.asarray(votes, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("votes must have shape (at least two classes, gaussians)")
    winners = np.argmax(values, axis=0).astype(np.uint16)
    columns = np.arange(values.shape[1], dtype=np.int64)
    winner_scores = values[winners, columns]
    second_scores = np.partition(values, -2, axis=0)[-2]
    totals = values.sum(axis=0, dtype=np.float32)
    agreements = np.divide(
        winner_scores,
        totals,
        out=np.zeros_like(winner_scores, dtype=np.float32),
        where=totals > 0.0,
    )
    unique = (winner_scores - second_scores) > np.float32(tie_epsilon)
    raw = np.where(unique & (winners != 0), winners, 0).astype(np.uint16)
    return raw, winner_scores.astype(np.float32), second_scores.astype(np.float32), agreements


def threshold_winners(
    raw_winners: np.ndarray,
    agreements: np.ndarray,
    supporting_views: np.ndarray,
    min_views: int,
    min_agreement: float,
) -> np.ndarray:
    if min_views < 0:
        raise ValueError("min_views must be non-negative")
    if not 0.0 <= min_agreement <= 1.0:
        raise ValueError("min_agreement must be between zero and one")
    raw = np.asarray(raw_winners, dtype=np.uint16)
    agreement = np.asarray(agreements, dtype=np.float32)
    support = np.asarray(supporting_views)
    if not (raw.shape == agreement.shape == support.shape):
        raise ValueError("winner threshold arrays must have matching shapes")
    keep = (raw != 0) & (support >= min_views) & (agreement >= min_agreement)
    return np.where(keep, raw, 0).astype(np.uint16)


def supporting_view_counts(
    vote_files: Iterable[Path],
    raw_winners: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_winners, dtype=np.uint16)
    counts = np.zeros(raw.shape, dtype=np.uint16)
    for path in vote_files:
        with np.load(path) as data:
            indices = data["indices"].astype(np.int64, copy=False)
            class_ids = data["class_ids"].astype(np.uint16, copy=False)
        selected = (raw[indices] != 0) & (raw[indices] == class_ids)
        if selected.any():
            # A view contributes at most one support count to each Gaussian,
            # even if a malformed sparse file repeats a matching record.
            matched = np.unique(indices[selected])
            counts[matched] += np.uint16(1)
    return counts
