"""S4: per-concept equal-camera consensus into a sparse membership matrix.

Each global instance is voted on independently: a camera that observes a
Gaussian supports the instance when the Gaussian's within-concept winner in
that view maps to it. Statuses mirror the audited hard-vote policy, applied
per (Gaussian, instance) pair instead of per Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATUS_ACCEPTED = 1
STATUS_SINGLE_CAMERA = 2
STATUS_WEAK_MAJORITY = 3

STATUS_NAMES = {
    STATUS_ACCEPTED: "accepted",
    STATUS_SINGLE_CAMERA: "single_camera",
    STATUS_WEAK_MAJORITY: "weak_majority",
}

_ID_SPACE = 65536  # uint16 instance id space, id 0 reserved


@dataclass(frozen=True)
class MembershipCSR:
    """Sparse (gaussians x instances) membership with vote statistics."""

    indptr: np.ndarray
    instance_ids: np.ndarray
    support_counts: np.ndarray
    observe_counts: np.ndarray
    scores: np.ndarray
    status: np.ndarray


def view_concept_winners(
    indices: np.ndarray,
    mask_ids: np.ndarray,
    weights: np.ndarray,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-Gaussian unique dominant mask within one concept pass.

    A Gaussian yields a winner only when its highest mask membership reaches
    ``min_weight`` and is strictly unique; exact ties abstain, mirroring the
    audited one-unique-max-per-camera policy.
    """

    gaussians = np.asarray(indices, dtype=np.int64)
    masks = np.asarray(mask_ids, dtype=np.uint16)
    mass = np.asarray(weights, dtype=np.float32)
    if not (gaussians.shape == masks.shape == mass.shape):
        raise ValueError("indices, mask_ids, and weights must align")
    if gaussians.size == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint16)

    order = np.lexsort((masks.astype(np.int64), -mass.astype(np.float64), gaussians))
    sorted_g = gaussians[order]
    sorted_m = masks[order]
    sorted_w = mass[order]

    first = np.ones(sorted_g.size, dtype=bool)
    first[1:] = sorted_g[1:] != sorted_g[:-1]
    first_pos = np.flatnonzero(first)
    winner_g = sorted_g[first_pos]
    winner_m = sorted_m[first_pos]
    winner_w = sorted_w[first_pos]

    next_pos = first_pos + 1
    tie = np.zeros(first_pos.size, dtype=bool)
    has_next = next_pos < sorted_g.size
    candidates = np.flatnonzero(has_next)
    same_gaussian = sorted_g[next_pos[candidates]] == winner_g[candidates]
    runners_up = candidates[same_gaussian]
    tie[runners_up] = sorted_w[next_pos[runners_up]] == winner_w[runners_up]

    keep = (winner_w >= np.float32(min_weight)) & ~tie
    return winner_g[keep].astype(np.uint32), winner_m[keep].astype(np.uint16)


def accumulate_membership(
    events: list[tuple[np.ndarray, np.ndarray]],
    observe_counts: np.ndarray,
    gaussian_count: int,
) -> MembershipCSR:
    """Combine per-view (gaussian, global instance) winner events.

    ``observe_counts`` holds, per Gaussian, the number of cameras with any
    rendered visibility of it. A vote on an unobserved Gaussian is a driver
    bug and raises.
    """

    observed = np.asarray(observe_counts, dtype=np.uint16)
    if observed.shape != (gaussian_count,):
        raise ValueError("observe_counts must have one entry per Gaussian")

    keys_parts: list[np.ndarray] = []
    for gaussians, instances in events:
        g = np.asarray(gaussians, dtype=np.int64)
        i = np.asarray(instances, dtype=np.int64)
        if g.shape != i.shape:
            raise ValueError("event gaussians and instances must align")
        if g.size and (g.min() < 0 or g.max() >= gaussian_count):
            raise ValueError("event references a Gaussian out of range")
        if i.size and (i.min() < 1 or i.max() >= _ID_SPACE):
            raise ValueError("instance ids must be 1-based uint16 values")
        keys_parts.append(g * _ID_SPACE + i)

    if keys_parts:
        keys = np.concatenate(keys_parts)
    else:
        keys = np.zeros(0, dtype=np.int64)
    unique_keys, support = np.unique(keys, return_counts=True)
    entry_gaussians = unique_keys // _ID_SPACE
    entry_instances = (unique_keys % _ID_SPACE).astype(np.uint16)
    support = support.astype(np.uint16)

    entry_observed = observed[entry_gaussians]
    if np.any(entry_observed == 0):
        raise RuntimeError("a camera voted on a Gaussian it never observed")
    scores = (support / entry_observed).astype(np.float32)

    status = np.full(support.shape, STATUS_WEAK_MAJORITY, dtype=np.uint8)
    status[support.astype(np.int64) * 2 > entry_observed] = STATUS_ACCEPTED
    status[support == 1] = STATUS_SINGLE_CAMERA

    counts_per_gaussian = np.bincount(entry_gaussians, minlength=gaussian_count)
    indptr = np.zeros(gaussian_count + 1, dtype=np.int64)
    np.cumsum(counts_per_gaussian, out=indptr[1:])

    return MembershipCSR(
        indptr=indptr,
        instance_ids=entry_instances,
        support_counts=support,
        observe_counts=entry_observed.astype(np.uint16),
        scores=scores,
        status=status,
    )


def save_membership(path: Path, membership: MembershipCSR) -> None:
    np.savez_compressed(
        Path(path),
        indptr=membership.indptr,
        instance_ids=membership.instance_ids,
        support_counts=membership.support_counts,
        observe_counts=membership.observe_counts,
        scores=membership.scores,
        status=membership.status,
    )


def load_membership(path: Path) -> MembershipCSR:
    with np.load(Path(path), allow_pickle=False) as data:
        return MembershipCSR(
            indptr=data["indptr"],
            instance_ids=data["instance_ids"],
            support_counts=data["support_counts"],
            observe_counts=data["observe_counts"],
            scores=data["scores"],
            status=data["status"],
        )
