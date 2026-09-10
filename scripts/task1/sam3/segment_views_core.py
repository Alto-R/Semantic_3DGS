"""Backend-agnostic core of the S1 SAM3 view segmentation stage.

The driver loops over rendered camera views; this module owns the per-view
prompt loop, the on-disk mask stack format, and the masks manifest contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.sam3.sam3_backend import Sam3Backend
from scripts.task1.sam3.vocabulary import Vocabulary


MASKS_SOURCE = "sam3_promptable_concept_segmentation"
MASKS_CONTRACT = "per_view_binary_instance_masks_v1"


@dataclass(frozen=True)
class ViewMask:
    """One instance mask detected in one view."""

    phrase: str
    concept: str
    score: float
    mask: np.ndarray


def segment_view(
    image: np.ndarray,
    vocabulary: Vocabulary,
    backend: Sam3Backend,
    min_score: float,
) -> list[ViewMask]:
    """Prompt every vocabulary phrase and keep scored, non-empty masks."""

    concept_of = vocabulary.phrase_to_concept()
    view_masks: list[ViewMask] = []
    for phrase in vocabulary.prompt_phrases():
        for instance in backend.segment(image, phrase):
            if instance.score < min_score:
                continue
            mask = np.asarray(instance.mask)
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"backend mask shape {mask.shape} does not match the "
                    f"image shape {image.shape[:2]}"
                )
            if not mask.any():
                continue
            view_masks.append(
                ViewMask(
                    phrase=phrase,
                    concept=concept_of[phrase],
                    score=float(instance.score),
                    mask=mask.astype(np.bool_, copy=False),
                )
            )
    return view_masks


def write_view_masks(path: Path, view_masks: list[ViewMask]) -> dict[str, Any]:
    """Save the view's mask stack and return its manifest fragment."""

    path = Path(path)
    if view_masks:
        stack = np.stack([mask.mask for mask in view_masks]).astype(np.uint8)
    else:
        stack = np.zeros((0, 0, 0), dtype=np.uint8)
    np.savez_compressed(path, mask_stack=stack)
    return {
        "mask_file": path.name,
        "mask_count": len(view_masks),
        "masks": [
            {
                "mask_index": index,
                "phrase": mask.phrase,
                "concept": mask.concept,
                "score": mask.score,
                "pixel_count": int(mask.mask.sum()),
            }
            for index, mask in enumerate(view_masks)
        ],
    }


def validate_masks_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("source") != MASKS_SOURCE:
        raise ValueError(f"unsupported masks source: {manifest.get('source')!r}")
    if manifest.get("contract") != MASKS_CONTRACT:
        raise ValueError(
            f"unsupported masks contract: {manifest.get('contract')!r}"
        )
    if not isinstance(manifest.get("frames"), list):
        raise ValueError("masks manifest must list its frames")


def build_masks_manifest(
    scene: str,
    vocabulary: Vocabulary,
    model_id: str,
    model_revision: str,
    min_score: float,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": MASKS_SOURCE,
        "contract": MASKS_CONTRACT,
        "scene": scene,
        "model_id": model_id,
        "model_revision": model_revision,
        "min_score": float(min_score),
        "vocabulary_sha256": vocabulary.sha256(),
        "prompt_phrases": list(vocabulary.prompt_phrases()),
        "frame_count": len(frames),
        "frames": frames,
    }
