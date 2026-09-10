"""Pure core of the S2 per-concept FlashSplat mask lift.

One FlashSplat pass covers one concept in one view: instances of a concept
are disjoint in image space, so they share an index map, while different
concepts may overlap and therefore get separate passes. Membership weights
are per-mask fractions of a Gaussian's rendered mass; there is deliberately
no sum-to-one constraint across concepts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def concept_index_map(mask_stack: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Collapse one concept's masks into a FlashSplat index map.

    Row k becomes index k+1; 0 is background. Overlapping pixels (which SAM3
    should not produce within a concept, but may) go to the higher-scoring
    mask.
    """

    stack = np.asarray(mask_stack)
    score_values = np.asarray(scores, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError("mask_stack must have shape masks x height x width")
    if score_values.shape != (stack.shape[0],):
        raise ValueError("scores must align with the mask axis")

    index_map = np.zeros(stack.shape[1:], dtype=np.float32)
    best_score = np.full(stack.shape[1:], -np.inf, dtype=np.float32)
    for row in range(stack.shape[0]):
        covered = stack[row] > 0
        wins = covered & (score_values[row] > best_score)
        index_map[wins] = np.float32(row + 1)
        best_score[wins] = score_values[row]
    return index_map


def concat_or_empty(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
    """Join sparse vote columns, yielding a typed empty array when there are none."""

    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts)


def mask_membership_votes(
    used_count: np.ndarray,
    visibility: np.ndarray,
    view_mask_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one concept pass's FlashSplat support into per-mask memberships.

    ``used_count`` holds one background row plus one row per mask in this
    pass. ``view_mask_indices`` maps mask rows back to the S1 per-view mask
    indices. Weights are ``used / visibility`` and are independent per mask.
    """

    used = np.asarray(used_count, dtype=np.float32)
    seen = np.asarray(visibility, dtype=np.float32)
    mask_indices = np.asarray(view_mask_indices, dtype=np.uint16)
    if used.ndim != 2:
        raise ValueError("used_count must have shape rows x gaussians")
    if used.shape[0] != mask_indices.shape[0] + 1:
        raise ValueError("used_count needs one background row plus mask rows")
    if seen.shape != (used.shape[1],):
        raise ValueError("visibility must align with the gaussian axis")
    if not np.isfinite(used).all() or np.any(used < 0.0):
        raise ValueError("FlashSplat support must be finite and non-negative")

    all_indices: list[np.ndarray] = []
    all_mask_ids: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for row, mask_index in enumerate(mask_indices, start=1):
        supported = (used[row] > 0.0) & (seen > 0.0)
        indices = np.flatnonzero(supported).astype(np.uint32)
        if indices.size == 0:
            continue
        weights = (used[row, indices] / seen[indices]).astype(np.float32)
        all_indices.append(indices)
        all_mask_ids.append(np.full(indices.shape, mask_index, dtype=np.uint16))
        all_weights.append(weights)

    return (
        concat_or_empty(all_indices, np.uint32),
        concat_or_empty(all_mask_ids, np.uint16),
        concat_or_empty(all_weights, np.float32),
    )


def observed_gaussians(visibility: np.ndarray) -> np.ndarray:
    """Indices of Gaussians with any rendered mass in this view."""

    seen = np.asarray(visibility, dtype=np.float32)
    if seen.ndim != 1:
        raise ValueError("visibility must be one-dimensional")
    return np.flatnonzero(seen > 0.0).astype(np.uint32)
