"""SAM3 backend interface and the deterministic mock used by tests.

The real transformers-backed model lives behind the same ``segment`` call in
the S1 driver and is only importable on the cluster. Nothing in this module
imports torch or transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class InstanceMask:
    """One detected instance for one prompt: a boolean mask and its score."""

    mask: np.ndarray
    score: float


class Sam3Backend(Protocol):
    def segment(self, image: np.ndarray, phrase: str) -> list[InstanceMask]:
        """Return every instance of the prompted concept in the image."""
        ...


class MockSam3Backend:
    """Deterministic rectangle-mask backend for unit tests.

    ``boxes`` maps a phrase to ``(y0, y1, x0, x1, score)`` rectangles in
    half-open pixel coordinates. Rectangles are clipped to the image;
    rectangles that clip to nothing are dropped.
    """

    def __init__(self, boxes: dict[str, list[tuple[int, int, int, int, float]]]):
        self._boxes = boxes

    def segment(self, image: np.ndarray, phrase: str) -> list[InstanceMask]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape height x width x 3")
        height, width = image.shape[:2]
        masks: list[InstanceMask] = []
        for y0, y1, x0, x1, score in self._boxes.get(phrase, []):
            mask = np.zeros((height, width), dtype=np.bool_)
            mask[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)] = True
            if not mask.any():
                continue
            masks.append(InstanceMask(mask=mask, score=float(score)))
        return masks
