"""Class-agnostic regions from raw DINOv3 Mask2Former queries.

The query's ADE20K prediction is retained for diagnostics only.  Region
boundaries and confidence depend on no semantic class, which lets downstream
code assign identity from an independent source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QueryRegionThresholds:
    min_objectness: float = 0.30
    mask_threshold: float = 0.50
    min_pixel_score: float = 0.25
    min_area: int = 100
    max_area_ratio: float = 0.80
    max_queries: int = 64

    def validate(self) -> None:
        for name in ("min_objectness", "mask_threshold", "min_pixel_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_area < 1:
            raise ValueError("min_area must be positive")
        if not 0.0 < self.max_area_ratio <= 1.0:
            raise ValueError("max_area_ratio must be greater than zero and at most one")
        if self.max_queries < 1 or self.max_queries > np.iinfo(np.uint16).max:
            raise ValueError("max_queries must fit in a positive uint16 region id")


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values.astype(np.float32, copy=False) - np.max(
        values,
        axis=axis,
        keepdims=True,
    )
    exponent = np.exp(shifted)
    return exponent / np.maximum(
        exponent.sum(axis=axis, keepdims=True),
        np.finfo(np.float32).tiny,
    )


def compact_query_evidence(
    class_logits: np.ndarray,
    regions: list[dict[str, Any]],
    query_embeddings: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Select full semantic evidence for the compact query-region partition.

    Mask2Former includes a final no-object logit.  Semantic probabilities are
    stored conditionally on the query being an object, while the no-object
    probability is retained separately.  Optional decoder-query embeddings
    are L2-normalized so downstream association can compare views without
    depending on their raw scale.
    """

    logits = np.asarray(class_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("class_logits must have shape Qx(C+1)")
    if not np.isfinite(logits).all():
        raise ValueError("class_logits contain non-finite values")

    query_indices = np.asarray(
        [int(region["query_index"]) for region in regions],
        dtype=np.int16,
    )
    if query_indices.size and (
        int(query_indices.min()) < 0 or int(query_indices.max()) >= logits.shape[0]
    ):
        raise ValueError("region query index is outside class_logits")

    probabilities = _softmax(logits, axis=1)
    selected = probabilities[query_indices] if query_indices.size else probabilities[:0]
    semantic = selected[:, :-1]
    semantic_sum = semantic.sum(axis=1, keepdims=True)
    semantic = np.divide(
        semantic,
        np.maximum(semantic_sum, np.finfo(np.float32).tiny),
    ).astype(np.float32, copy=False)
    no_object = selected[:, -1].astype(np.float32, copy=False)

    result = {
        "query_indices": query_indices,
        "class_probabilities": semantic,
        "no_object_probabilities": no_object,
    }
    if query_embeddings is not None:
        embeddings = np.asarray(query_embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != logits.shape[0]:
            raise ValueError("query_embeddings must have shape QxD")
        if not np.isfinite(embeddings).all():
            raise ValueError("query_embeddings contain non-finite values")
        selected_embeddings = (
            embeddings[query_indices]
            if query_indices.size
            else embeddings[:0]
        )
        norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
        result["query_embeddings"] = np.divide(
            selected_embeddings,
            np.maximum(norms, np.finfo(np.float32).tiny),
        ).astype(np.float32, copy=False)
    return result


def class_agnostic_query_regions(
    class_logits: np.ndarray,
    mask_probabilities: np.ndarray,
    thresholds: QueryRegionThresholds,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return a disjoint region partition without using query class identity."""

    thresholds.validate()
    class_logits = np.asarray(class_logits, dtype=np.float32)
    mask_probabilities = np.asarray(mask_probabilities, dtype=np.float32)
    if class_logits.ndim != 2 or class_logits.shape[1] < 2:
        raise ValueError("class_logits must have shape Qx(C+1)")
    if mask_probabilities.ndim != 3:
        raise ValueError("mask_probabilities must have shape QxHxW")
    if class_logits.shape[0] != mask_probabilities.shape[0]:
        raise ValueError("query counts differ between class logits and masks")
    if not np.isfinite(class_logits).all() or not np.isfinite(mask_probabilities).all():
        raise ValueError("query tensors contain non-finite values")
    if np.any(mask_probabilities < 0.0) or np.any(mask_probabilities > 1.0):
        raise ValueError("mask probabilities must be between zero and one")

    height, width = mask_probabilities.shape[1:]
    empty_id = np.zeros((height, width), dtype=np.uint16)
    empty_confidence = np.zeros((height, width), dtype=np.float32)
    class_probabilities = _softmax(class_logits, axis=1)
    objectness = 1.0 - class_probabilities[:, -1]
    diagnostic_class = np.argmax(class_probabilities[:, :-1], axis=1)
    diagnostic_confidence = np.max(class_probabilities[:, :-1], axis=1)

    candidate_indices = np.flatnonzero(objectness >= thresholds.min_objectness)
    if candidate_indices.size == 0:
        return empty_id, empty_confidence, []

    candidate_masks = mask_probabilities[candidate_indices]
    candidate_areas = np.sum(
        candidate_masks >= thresholds.mask_threshold,
        axis=(1, 2),
    )
    maximum_area = max(
        int(round(height * width * thresholds.max_area_ratio)),
        1,
    )
    keep = (candidate_areas >= thresholds.min_area) & (
        candidate_areas <= maximum_area
    )
    candidate_indices = candidate_indices[keep]
    candidate_masks = candidate_masks[keep]
    if candidate_indices.size == 0:
        return empty_id, empty_confidence, []

    order = np.argsort(objectness[candidate_indices])[::-1][
        : thresholds.max_queries
    ]
    candidate_indices = candidate_indices[order]
    candidate_masks = candidate_masks[order]
    candidate_objectness = objectness[candidate_indices]

    pixel_scores = (
        candidate_masks
        * candidate_objectness[:, np.newaxis, np.newaxis]
    )
    winner_position = np.argmax(pixel_scores, axis=0)
    winner_score = np.max(pixel_scores, axis=0)
    winner_mask_probability = np.take_along_axis(
        candidate_masks,
        winner_position[np.newaxis, ...],
        axis=0,
    )[0]
    assigned = (
        (winner_score >= thresholds.min_pixel_score)
        & (winner_mask_probability >= thresholds.mask_threshold)
    )

    region_id = np.zeros((height, width), dtype=np.uint16)
    region_confidence = np.zeros((height, width), dtype=np.float32)
    regions: list[dict[str, Any]] = []
    compact_id = 1
    for candidate_position, query_index in enumerate(candidate_indices):
        pixels = assigned & (winner_position == candidate_position)
        area = int(pixels.sum())
        if area < thresholds.min_area:
            continue
        rows, columns = np.nonzero(pixels)
        region_id[pixels] = compact_id
        region_confidence[pixels] = winner_score[pixels]
        regions.append(
            {
                "region_id": compact_id,
                "query_index": int(query_index),
                "area": area,
                "bbox_xyxy": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                ],
                "mean_region_confidence": float(winner_score[pixels].mean()),
                "objectness": float(objectness[query_index]),
                "diagnostic_ade20k_class": int(
                    diagnostic_class[query_index]
                ),
                "diagnostic_class_confidence": float(
                    diagnostic_confidence[query_index]
                ),
            }
        )
        compact_id += 1
    return region_id, region_confidence, regions


def describe_query_regions(
    thresholds: QueryRegionThresholds,
) -> dict[str, Any]:
    thresholds.validate()
    return {
        "available": True,
        "source": "whole_image_dinov3_mask2former_queries",
        "identity_source": None,
        "semantic_class_used_for_identity": False,
        "semantic_class_is_diagnostic_only": True,
        "thresholds": asdict(thresholds),
    }
