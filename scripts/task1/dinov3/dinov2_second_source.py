#!/usr/bin/env python3
"""DINOv2 second-source evidence for the component-graph agreement gate.

The cluster maintains complete per-camera DINOv2 FlashSplat vote caches
(separate-abstain allviews).  This module loads those votes as a second
evidence stream with the same per-camera winner/mass shape used by the
component-graph pipeline, validates that every DINOv3 camera has a DINOv2
counterpart, and provides an agreement-gated component vote aggregation:
a camera contributes to a component only when its DINOv3 and DINOv2 component
winners agree.  DINOv2 never changes reliability, anchors, or consensus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


SOURCE = "dinov2_flashsplat_per_view_votes"


def collapse_dinov2_camera(
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one unique per-Gaussian winner and its mass for one DINOv2 camera.

    DINOv2 vote weights are not normalized to sum to one per Gaussian, so the
    DINOv3 collapse contract does not apply.  The winner is the class with the
    largest weight for that Gaussian; exact ties abstain, and class id zero is
    treated as unlabeled.  Mass is the winning weight, used only for the
    agreement comparison (the DINOv3 mass is what counts in the final vote).
    """

    idx = np.asarray(indices, dtype=np.int64)
    classes = np.asarray(class_ids, dtype=np.uint16)
    values = np.asarray(weights, dtype=np.float32)
    if not (idx.shape == classes.shape == values.shape) or idx.ndim != 1:
        raise ValueError("DINOv2 vote arrays must be aligned one-dimensional arrays")
    if idx.size and (np.any(idx < 0) or int(idx.max()) >= gaussian_count):
        raise ValueError("DINOv2 vote references a Gaussian outside the model")
    if idx.size and (int(classes.max()) > class_count):
        raise ValueError("DINOv2 vote references an invalid project class")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("DINOv2 vote weights must be finite and non-negative")

    positive = classes > 0
    idx = idx[positive]
    classes = classes[positive]
    values = values[positive]
    winner = np.zeros((gaussian_count,), dtype=np.uint16)
    mass = np.zeros((gaussian_count,), dtype=np.float32)
    tied = np.zeros((gaussian_count,), dtype=bool)
    if idx.size:
        keys = idx.astype(np.int64) * np.int64(class_count + 1) + classes.astype(np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        aggregated = np.zeros((unique_keys.size,), dtype=np.float32)
        np.add.at(aggregated, inverse, values)
        gaussian_ids = unique_keys // np.int64(class_count + 1)
        class_ids = (unique_keys % np.int64(class_count + 1)).astype(np.uint16)
        maximum = np.zeros((gaussian_count,), dtype=np.float32)
        np.maximum.at(maximum, gaussian_ids, aggregated)
        is_maximum = aggregated == maximum[gaussian_ids]
        counts = np.zeros((gaussian_count,), dtype=np.uint8)
        np.add.at(counts, gaussian_ids[is_maximum], np.uint8(1))
        tied = counts > 1
        winner[gaussian_ids[is_maximum]] = class_ids[is_maximum]
        mass[gaussian_ids[is_maximum]] = aggregated[is_maximum]
        winner[tied] = 0
        mass[tied] = 0.0
    return winner, mass


def load_dinov2_evidence(
    manifest_path: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    *,
    gaussian_count: int,
    class_count: int,
) -> List[Dict[str, Any]]:
    """Load every DINOv2 camera vote as an evidence row."""

    if manifest is None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != SOURCE:
        raise ValueError("DINOv2 vote manifest has the wrong source")
    if int(manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("DINOv2 vote manifest has a different Gaussian count")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("DINOv2 vote manifest has no frames")
    evidence: List[Dict[str, Any]] = []
    for frame in frames:
        camera_index = int(frame["camera_index"])
        camera_id = int(frame["camera_id"])
        vote_path = manifest_path.parent / str(frame["vote_file"])
        if not vote_path.is_file():
            raise FileNotFoundError(vote_path)
        with np.load(vote_path, allow_pickle=False) as data:
            winners, mass = collapse_dinov2_camera(
                data["indices"],
                data["class_ids"],
                data["weights"],
                gaussian_count=gaussian_count,
                class_count=class_count,
            )
        evidence.append(
            {
                "camera_index": camera_index,
                "camera_id": camera_id,
                "file": str(frame.get("file", "")),
                "winners": winners,
                "mass": mass,
                "source": SOURCE,
            }
        )
    return evidence


def align_dinov2_evidence(
    evidence: Sequence[Mapping[str, Any]],
    dinov2_evidence: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return DINOv2 evidence ordered to match the DINOv3 evidence by camera_id."""

    by_camera_id = {
        int(item["camera_id"]): item for item in dinov2_evidence
    }
    if len(by_camera_id) != len(dinov2_evidence):
        raise ValueError("DINOv2 evidence repeats a camera id")
    aligned: List[Dict[str, Any]] = []
    for item in evidence:
        counterpart = by_camera_id.get(int(item["camera_id"]))
        if counterpart is None:
            raise ValueError(
                f"DINOv3 camera {item['camera_id']} has no DINOv2 counterpart"
            )
        if int(counterpart["camera_index"]) != int(item["camera_index"]):
            raise ValueError("DINOv3 and DINOv2 camera indices disagree")
        aligned.append(dict(counterpart))
    return aligned


def exclude_dinov2_camera(
    dinov2_evidence: Sequence[Mapping[str, Any]],
    camera_id: int,
) -> List[Dict[str, Any]]:
    """Return DINOv2 evidence without the held-out camera's row."""

    return [
        item
        for item in dinov2_evidence
        if int(item["camera_id"]) != int(camera_id)
    ]


def _component_totals(
    components: np.ndarray,
    winners: np.ndarray,
    masses: np.ndarray,
    supported: np.ndarray,
    *,
    component_count: int,
    class_count: int,
) -> Dict[str, np.ndarray]:
    selected = np.flatnonzero(supported)
    if selected.size == 0:
        return {
            "winner": np.zeros((component_count,), dtype=np.uint16),
            "maximum": np.zeros((component_count,), dtype=np.float32),
            "total": np.zeros((component_count,), dtype=np.float32),
            "unique": np.zeros((component_count,), dtype=bool),
        }
    keys = (
        components[selected] * np.int64(class_count + 1)
        + winners[selected].astype(np.int64)
    )
    totals = np.bincount(
        keys,
        weights=masses[selected],
        minlength=component_count * (class_count + 1),
    ).reshape(component_count, class_count + 1)
    semantic = totals[:, 1:]
    maximum = semantic.max(axis=1)
    winner = semantic.argmax(axis=1).astype(np.uint16) + np.uint16(1)
    tied = (semantic == maximum[:, None]).sum(axis=1) > 1
    total = semantic.sum(axis=1)
    unique = (maximum > 0.0) & ~tied & (maximum * 2.0 > total)
    return {
        "winner": np.where(unique, winner, 0).astype(np.uint16),
        "maximum": maximum,
        "total": total,
        "unique": unique,
    }


def aggregate_component_votes_agreement_gated(
    component_ids: np.ndarray,
    cache: Mapping[str, np.ndarray],
    dinov2_cache: Mapping[str, np.ndarray],
    evidence: Sequence[Mapping[str, Any]],
    dinov2_evidence: Sequence[Mapping[str, Any]],
    reliabilities: Mapping[int, float],
    *,
    class_count: int,
    excluded_camera_by_component: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Aggregate component votes only where DINOv3 and DINOv2 winners agree.

    Mirrors ``aggregate_component_votes`` except that a camera's vote for a
    component is cast only when the DINOv3 component winner and the DINOv2
    component winner are the same nonzero class.  The contribution always uses
    the DINOv3 winner, DINOv3 normalized mass, and the DINOv3 reliability.
    """

    components = np.asarray(component_ids, dtype=np.int64)
    component_count = int(components.max() + 1) if components.size else 0
    raw = np.zeros((class_count + 1, component_count), dtype=np.uint16)
    weighted = np.zeros((class_count + 1, component_count), dtype=np.float32)
    camera_count = np.zeros((component_count,), dtype=np.uint16)
    winners = np.asarray(cache["winners"], dtype=np.uint16)
    masses = np.asarray(cache["masses"], dtype=np.float32)
    d2_winners = np.asarray(dinov2_cache["winners"], dtype=np.uint16)
    d2_masses = np.asarray(dinov2_cache["masses"], dtype=np.float32)
    d2_ordinal_by_camera = {
        int(item["camera_id"]): ordinal
        for ordinal, item in enumerate(dinov2_evidence)
    }
    if len(d2_ordinal_by_camera) != len(dinov2_evidence):
        raise ValueError("DINOv2 evidence repeats a camera id")

    for ordinal, item in enumerate(evidence):
        local_winner = winners[ordinal]
        local_mass = masses[ordinal]
        supported = local_winner > 0
        if excluded_camera_by_component is not None:
            supported &= excluded_camera_by_component[components] != ordinal
        if not np.any(supported):
            continue
        d3 = _component_totals(
            components,
            local_winner,
            local_mass,
            supported,
            component_count=component_count,
            class_count=class_count,
        )
        d2_ordinal = d2_ordinal_by_camera.get(int(item["camera_id"]))
        if d2_ordinal is None:
            continue
        d2_supported = d2_winners[d2_ordinal] > 0
        if excluded_camera_by_component is not None:
            d2_supported &= excluded_camera_by_component[components] != ordinal
        d2 = _component_totals(
            components,
            d2_winners[d2_ordinal],
            d2_masses[d2_ordinal],
            d2_supported,
            component_count=component_count,
            class_count=class_count,
        )
        agreed = (
            d3["unique"]
            & d2["unique"]
            & (d3["winner"] == d2["winner"])
            & (d3["winner"] > 0)
        )
        selected = np.flatnonzero(agreed)
        if selected.size == 0:
            continue
        raw[d3["winner"][selected], selected] += np.uint16(1)
        normalized_mass = np.divide(
            d3["maximum"][selected],
            d3["total"][selected],
            out=np.zeros_like(d3["maximum"][selected]),
            where=d3["total"][selected] > 0.0,
        )
        weighted[d3["winner"][selected], selected] += (
            np.float32(reliabilities[int(item["camera_index"])]) * normalized_mass
        )
        camera_count[selected] += np.uint16(1)
    return {"raw": raw, "weighted": weighted, "camera_count": camera_count}
