#!/usr/bin/env python3
"""Compose closed-set DINOv3 maps with OOV masks and lift them once.

The dense map is the only semantic image sent to FlashSplat for a camera:
the ADE20K project class is used everywhere, except where a validated OOV
mask wins the pixel.  Per-Gaussian votes are then reduced across cameras.
The accepted base labels remain the fallback; only an OOV class with a
multiview majority can replace a base label.

This keeps the OOV correction and its competitors in one multiclass render.
It does not delete Gaussians, choose cameras manually, or apply a scene-
specific spatial rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
)
from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import (
    load_ontology,
    normalize_class_name,
)
from scripts.task1.dinov3.lift_dense_view_votes import (
    dense_sparse_view_votes,
    flashsplat_class_rows,
    local_index_map,
    validate_dinov3_manifest,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"

SOURCE = "dinov3_dinotxt_sam_multiclass_oov_fusion"
CONTRACT = "single_composed_multiclass_map_oov_majority_fallback_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_label_items(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    label_map = load_json(path)
    raw_items = label_map.get("labels")
    if not isinstance(raw_items, list):
        raise ValueError(f"{path} must contain a labels list")
    items: dict[int, dict[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError(f"Malformed label-map item in {path}: {raw!r}")
        if not {"id", "name", "class"}.issubset(raw):
            raise ValueError(f"Malformed label-map item in {path}: {raw!r}")
        label_id = int(raw["id"])
        if label_id in items:
            raise ValueError(f"Duplicate label id {label_id} in {path}")
        item = dict(raw)
        item["id"] = label_id
        item["class"] = normalize_class_name(item["class"])
        items[label_id] = item
    if 0 not in items:
        raise ValueError(f"{path} is missing label id 0")
    return label_map, items


def validate_labels(labels: np.ndarray, items: dict[int, dict[str, Any]], path: Path) -> np.ndarray:
    result = np.asarray(labels)
    if result.ndim != 1 or result.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{path} must contain a one-dimensional integer array")
    result = result.astype(np.int32, copy=False)
    if np.any(result < 0):
        raise ValueError(f"{path} contains negative labels")
    missing = sorted(set(int(value) for value in np.unique(result)) - set(items))
    if missing:
        raise ValueError(f"{path} contains ids absent from its label map: {missing}")
    return result


def _class_name_from_record(record: dict[str, Any]) -> str:
    return normalize_class_name(record.get("class_name", record.get("class", "")))


def discover_oov_classes(manifest: dict[str, Any]) -> list[str]:
    """Return OOV classes in deterministic first-seen order."""

    names: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        name = normalize_class_name(value)
        if name and name not in {"unknown", "object_candidate"} and name not in seen:
            seen.add(name)
            names.append(name)

    add(manifest.get("target_class", ""))
    raw_classes = manifest.get("classes", [])
    if isinstance(raw_classes, list):
        for item in raw_classes:
            add(item.get("class", item.get("name", "")) if isinstance(item, dict) else item)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("OOV manifest frames must be a list")
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("OOV manifest frame must be an object")
        for key in ("masks", "selected"):
            records = frame.get(key, [])
            if not isinstance(records, list):
                raise ValueError(f"OOV manifest frame {key} must be a list")
            for record in records:
                if isinstance(record, dict):
                    add(record.get("class_name", record.get("class", "")))
    return names


def parse_name_id_map(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError("--oov-label-ids entries must use class=id")
        name_raw, id_raw = raw.split("=", 1)
        name = normalize_class_name(name_raw)
        label_id = int(id_raw)
        if not name or label_id <= 0:
            raise ValueError("OOV class names and ids must be positive")
        if name in result:
            raise ValueError(f"Duplicate OOV class in --oov-label-ids: {name}")
        result[name] = label_id
    return result


def resolve_oov_label_ids(
    base_items: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    requested_classes: str,
    explicit_ids: str,
) -> dict[str, int]:
    discovered = discover_oov_classes(manifest)
    requested = [
        normalize_class_name(value)
        for value in requested_classes.split(",")
        if value.strip()
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("--oov-classes contains duplicates")
    classes = requested or discovered
    if not classes:
        raise ValueError("No OOV classes were found")
    unknown = [name for name in classes if name not in discovered]
    if unknown:
        raise ValueError(f"Requested OOV classes are absent from the manifest: {unknown}")
    explicit = parse_name_id_map(explicit_ids)
    unknown_explicit = sorted(set(explicit) - set(classes))
    if unknown_explicit:
        raise ValueError(f"Explicit OOV ids name unselected classes: {unknown_explicit}")

    existing_by_class: dict[str, list[int]] = {}
    for label_id, item in base_items.items():
        existing_by_class.setdefault(normalize_class_name(item.get("class", "")), []).append(label_id)
    used = set(base_items)
    next_id = max(used, default=0) + 1
    result: dict[str, int] = {}
    for name in classes:
        existing = existing_by_class.get(name, [])
        if len(existing) > 1:
            raise ValueError(f"Base label map has multiple ids for OOV class {name}")
        if existing and name not in explicit:
            result[name] = existing[0]
            continue
        label_id = explicit.get(name)
        if label_id is None:
            while next_id in used:
                next_id += 1
            label_id = next_id
            next_id += 1
        if label_id in used and label_id not in existing:
            raise ValueError(f"OOV id {label_id} collides with an existing base id")
        if label_id in result.values() and result.get(name) != label_id:
            raise ValueError(f"OOV id {label_id} is assigned more than once")
        result[name] = label_id
        used.add(label_id)
    return result


def load_mask_stack(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    with np.load(mask_path, allow_pickle=False) as data:
        if "masks" not in data:
            raise ValueError(f"{mask_path} is missing a masks array")
        masks = np.asarray(data["masks"]).astype(bool, copy=False)
    if masks.ndim != 3:
        raise ValueError(f"{mask_path} masks must have shape (K,H,W)")
    if masks.shape[1:] == shape:
        return masks
    resized: list[np.ndarray] = []
    for mask in masks:
        image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        resized.append(np.asarray(image, dtype=np.uint8) > 0)
    return np.stack(resized, axis=0) if resized else np.zeros((0, *shape), dtype=bool)


def resolve_mask_path(mask_dir: Path, frame: dict[str, Any]) -> Path:
    raw = str(frame.get("mask_file", "")).strip()
    if not raw:
        stem = Path(str(frame.get("file", ""))).stem
        raw = f"{stem}.npz"
    candidates = [
        mask_dir / raw,
        mask_dir / "mask_stacks" / raw,
        mask_dir / "grounded_sam_masks" / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask stack for {frame.get('file')}: {candidates}")


def compose_oov_pixel_ids(
    masks: np.ndarray,
    frame: dict[str, Any],
    class_ids: dict[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve overlapping OOV masks by per-mask confidence."""

    if masks.ndim != 3:
        raise ValueError("masks must have shape (K,H,W)")
    metadata = frame.get("masks", frame.get("selected", []))
    if not isinstance(metadata, list):
        raise ValueError("mask metadata must be a list")
    output = np.zeros(masks.shape[1:], dtype=np.uint16)
    score_map = np.full(masks.shape[1:], -np.inf, dtype=np.float32)
    scores_by_class = {name: 0.0 for name in class_ids}
    for index, mask in enumerate(masks):
        record = metadata[index] if index < len(metadata) and isinstance(metadata[index], dict) else {}
        name = _class_name_from_record(record)
        if not name:
            target = normalize_class_name(frame.get("target_class", ""))
            name = target if target in class_ids else ""
        if name not in class_ids:
            continue
        raw_score = record.get(
            "confidence",
            record.get("dinotxt_target_probability", record.get("score", 1.0)),
        )
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 1.0
        if not math.isfinite(score):
            score = 1.0
        scores_by_class[name] = max(scores_by_class[name], score)
        eligible = mask & (score > score_map)
        output[eligible] = np.uint16(class_ids[name])
        score_map[eligible] = np.float32(score)
    return output, {
        name: int(np.count_nonzero(output == label_id))
        for name, label_id in class_ids.items()
    }


def compose_project_class_map(
    raw_class: np.ndarray,
    ontology_lookup: np.ndarray,
    oov_pixel_ids: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_class)
    oov = np.asarray(oov_pixel_ids)
    if raw.ndim != 2 or oov.shape != raw.shape:
        raise ValueError("raw class and OOV pixel maps must have the same HxW shape")
    if raw.dtype.kind not in {"u", "i"}:
        raise ValueError("raw class map must be integer-valued")
    if np.any(raw < 0) or int(raw.max(initial=0)) >= len(ontology_lookup):
        raise ValueError("raw class map contains an out-of-range ADE20K id")
    output = ontology_lookup[raw.astype(np.uint8, copy=False)].astype(np.uint16, copy=True)
    oov_mask = oov > 0
    output[oov_mask] = oov[oov_mask].astype(np.uint16, copy=False)
    return output


def camera_winners(
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
    gaussian_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse sparse fractional support into one deterministic winner."""

    indices = np.asarray(indices, dtype=np.uint32)
    class_ids = np.asarray(class_ids, dtype=np.uint16)
    weights = np.asarray(weights, dtype=np.float32)
    if not (indices.ndim == class_ids.ndim == weights.ndim == 1):
        raise ValueError("vote arrays must be one-dimensional")
    if not (indices.shape == class_ids.shape == weights.shape):
        raise ValueError("vote arrays must have the same shape")
    if np.any(indices >= gaussian_count) or np.any(class_ids == 0):
        raise ValueError("vote arrays contain an invalid index or class")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("vote weights must be finite and positive")
    if indices.size == 0:
        empty = np.zeros((0,), dtype=np.uint32)
        return empty, empty.astype(np.uint16), empty.astype(np.float32)

    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    sorted_classes = class_ids[order]
    sorted_weights = weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_indices)) + 1]
    lengths = np.diff(np.r_[starts, sorted_indices.size])
    group_ids = np.repeat(np.arange(starts.size, dtype=np.int64), lengths)
    maximum = np.maximum.reduceat(sorted_weights, starts)
    is_maximum = sorted_weights == maximum[group_ids]
    winners = np.full(starts.size, np.iinfo(np.uint16).max, dtype=np.uint16)
    np.minimum.at(winners, group_ids[is_maximum], sorted_classes[is_maximum])
    if np.any(winners == np.iinfo(np.uint16).max):
        raise RuntimeError("failed to resolve a sparse vote winner")
    return sorted_indices[starts], winners, maximum


def choose_oov_replacements(
    base_labels: np.ndarray,
    visible_views: np.ndarray,
    oov_winner_views: np.ndarray,
    oov_positive_views: np.ndarray,
    oov_mass: np.ndarray,
    incumbent_winner_views: np.ndarray,
    oov_ids: np.ndarray,
    *,
    min_visible_views: int = 3,
    min_oov_winner_views: int = 2,
    min_oov_winner_share: float = 0.50,
    min_oov_mass_share: float = 0.35,
    min_oov_positive_views: int = 2,
    winner_margin: float = 0.05,
    already_target_is_accepted: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Choose one OOV winner per Gaussian, retaining the base otherwise."""

    base = np.asarray(base_labels, dtype=np.int32)
    visible = np.asarray(visible_views, dtype=np.float32)
    winners = np.asarray(oov_winner_views, dtype=np.float32)
    positive = np.asarray(oov_positive_views, dtype=np.float32)
    mass = np.asarray(oov_mass, dtype=np.float32)
    incumbent = np.asarray(incumbent_winner_views, dtype=np.float32)
    ids = np.asarray(oov_ids, dtype=np.uint16)
    if winners.ndim != 2 or positive.shape != winners.shape or mass.shape != winners.shape:
        raise ValueError("OOV evidence must have shape classes x gaussians")
    if winners.shape[0] != ids.size or winners.shape[1] != base.size:
        raise ValueError("OOV evidence and labels are incompatible")
    if visible.shape != base.shape or incumbent.shape != base.shape:
        raise ValueError("per-Gaussian evidence arrays are incompatible")
    if ids.size == 0:
        return base.copy(), {"candidate_id": np.zeros(base.shape, dtype=np.uint16), "fill": np.zeros(base.shape, dtype=bool)}
    if min_visible_views < 1 or min_oov_winner_views < 1 or min_oov_positive_views < 1:
        raise ValueError("view thresholds must be positive")
    if not 0.0 <= min_oov_winner_share <= 1.0 or not 0.0 <= min_oov_mass_share <= 1.0:
        raise ValueError("OOV shares must be between zero and one")
    if winner_margin < 0.0:
        raise ValueError("winner margin must be non-negative")

    best_winners = winners.max(axis=0)
    best_mass = np.full(base.shape, -np.inf, dtype=np.float32)
    for class_index in range(ids.size):
        tied = winners[class_index] == best_winners
        best_mass[tied] = np.maximum(best_mass[tied], mass[class_index, tied])
    candidate_id = np.zeros(base.shape, dtype=np.uint16)
    candidate_index = np.full(base.shape, -1, dtype=np.int32)
    # Iterating in ascending ID order makes exact ties deterministic.
    for class_index, label_id in enumerate(ids):
        tied = (winners[class_index] == best_winners) & (mass[class_index] == best_mass)
        unset = tied & (candidate_index < 0)
        candidate_index[unset] = class_index
        candidate_id[unset] = label_id

    safe_visible = np.maximum(visible, 1.0)
    winner_share = best_winners / safe_visible
    mass_share = np.zeros(base.shape, dtype=np.float32)
    positive_count = np.zeros(base.shape, dtype=np.float32)
    for class_index in range(ids.size):
        selected = candidate_index == class_index
        mass_share[selected] = mass[class_index, selected] / safe_visible[selected]
        positive_count[selected] = positive[class_index, selected]
    incumbent_share = incumbent / safe_visible
    eligible = (
        (visible >= float(min_visible_views))
        & (best_winners >= float(min_oov_winner_views))
        & (winner_share >= float(min_oov_winner_share))
        & (winner_share >= incumbent_share + float(winner_margin))
        & (mass_share >= float(min_oov_mass_share))
        & (positive_count >= float(min_oov_positive_views))
        & (candidate_id > 0)
    )
    if already_target_is_accepted:
        eligible &= ~np.isin(base, ids)
    output = base.copy()
    output[eligible] = candidate_id[eligible].astype(np.int32)
    return output, {
        "candidate_id": candidate_id,
        "candidate_index": candidate_index,
        "fill": eligible,
        "winner_share": winner_share,
        "mass_share": mass_share,
        "positive_views": positive_count,
        "incumbent_share": incumbent_share,
        "best_winner_views": best_winners,
    }


def _frame_lookup(manifest: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_camera: dict[int, dict[str, Any]] = {}
    by_file: dict[str, dict[str, Any]] = {}
    for frame in manifest.get("frames", []):
        if not isinstance(frame, dict):
            raise ValueError("manifest frame must be an object")
        if "camera_index" in frame:
            camera = int(frame["camera_index"])
            if camera in by_camera:
                raise ValueError(f"OOV manifest repeats camera index {camera}")
            by_camera[camera] = frame
        file_name = str(frame.get("file", ""))
        if file_name:
            by_file[file_name] = frame
    return by_camera, by_file


def _histogram(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dense-input-dir", required=True, type=Path)
    parser.add_argument("--segmentation-manifest", required=True, type=Path)
    parser.add_argument("--oov-manifest", required=True, type=Path)
    parser.add_argument("--oov-mask-dir", required=True, type=Path)
    parser.add_argument("--base-labels", required=True, type=Path)
    parser.add_argument("--base-label-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--oov-classes", default="")
    parser.add_argument("--oov-label-ids", default="")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--min-visible-views", default=3, type=int)
    parser.add_argument("--min-oov-winner-views", default=2, type=int)
    parser.add_argument("--min-oov-winner-share", default=0.50, type=float)
    parser.add_argument("--min-oov-mass-share", default=0.35, type=float)
    parser.add_argument("--min-oov-positive-views", default=2, type=int)
    parser.add_argument("--oov-positive-mass-threshold", default=0.50, type=float)
    parser.add_argument("--winner-margin", default=0.05, type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-semantic-ply", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        raise ValueError("--overwrite is intentionally unsupported for immutable output roots")
    for path in (
        args.model_path,
        args.dense_input_dir,
        args.segmentation_manifest,
        args.oov_manifest,
        args.oov_mask_dir,
        args.base_labels,
        args.base_label_map,
        args.ontology,
        args.flashsplat_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 <= args.oov_positive_mass_threshold <= 1.0:
        raise ValueError("--oov-positive-mass-threshold must be between zero and one")

    segmentation_manifest = load_json(args.segmentation_manifest)
    validate_dinov3_manifest(segmentation_manifest)
    oov_manifest = load_json(args.oov_manifest)
    dense_frames = segmentation_manifest.get("frames", [])
    if not isinstance(dense_frames, list) or not dense_frames:
        raise ValueError("DINOv3 segmentation manifest has no frames")
    base_map, base_items = load_label_items(args.base_label_map)
    base_labels = validate_labels(np.load(args.base_labels, allow_pickle=False), base_items, args.base_labels)
    ontology = load_ontology(args.ontology)
    oov_label_ids = resolve_oov_label_ids(
        base_items,
        oov_manifest,
        args.oov_classes,
        args.oov_label_ids,
    )
    oov_ids = np.asarray([oov_label_ids[name] for name in oov_label_ids], dtype=np.uint16)
    if np.any(oov_ids == 0):
        raise ValueError("OOV ids must be positive")

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    if base_labels.size != gaussian_count:
        raise ValueError("base label count differs from the Gaussian PLY")
    header = read_ply_header(ply_path)
    vertex = header.element("vertex")
    if vertex is None or vertex.count != gaussian_count:
        raise ValueError("Gaussian PLY vertex count is inconsistent")

    oov_by_camera, oov_by_file = _frame_lookup(oov_manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    vote_dir = output_dir / "view_votes"
    map_dir = output_dir / "composed_maps"
    vote_dir.mkdir()
    map_dir.mkdir()
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)
    vote_frames: list[dict[str, Any]] = []
    oov_pixel_totals = {name: 0 for name in oov_label_ids}

    with torch.no_grad():
        for frame in dense_frames:
            filename = str(frame["file"])
            camera_index = int(frame["camera_index"])
            if camera_index < 0 or camera_index >= len(cameras):
                raise ValueError(f"DINOv3 frame references invalid camera {camera_index}")
            camera = make_camera(cameras[camera_index], modules, args.max_width)
            segment_path = args.dense_input_dir / str(frame["segment_file"])
            with np.load(segment_path, allow_pickle=False) as segment:
                raw_class = np.asarray(segment["class_id"], dtype=np.uint8)
            expected_shape = (int(camera.image_height), int(camera.image_width))
            if raw_class.shape != expected_shape:
                raise ValueError(
                    f"{segment_path} has shape {raw_class.shape}; expected {expected_shape}"
                )

            oov_frame = oov_by_camera.get(camera_index) or oov_by_file.get(filename)
            if oov_frame is None:
                oov_pixel_ids = np.zeros(expected_shape, dtype=np.uint16)
                pixel_counts = {name: 0 for name in oov_label_ids}
            else:
                mask_path = resolve_mask_path(args.oov_mask_dir, oov_frame)
                masks = load_mask_stack(mask_path, expected_shape)
                oov_pixel_ids, pixel_counts = compose_oov_pixel_ids(
                    masks,
                    oov_frame,
                    oov_label_ids,
                )
            for name, count in pixel_counts.items():
                oov_pixel_totals[name] += count
            composed = compose_project_class_map(raw_class, ontology.ade_to_project, oov_pixel_ids)
            map_path = map_dir / f"{Path(filename).stem}.npz"
            np.savez_compressed(
                map_path,
                project_class_id=composed,
                oov_class_id=oov_pixel_ids,
            )
            indexed, class_ids = local_index_map(composed)
            gt_mask = torch.from_numpy(indexed).to(device="cuda", dtype=torch.float32)
            render_pkg = render_flashsplat(
                camera,
                gaussians,
                modules,
                pipeline,
                background,
                gt_mask=gt_mask,
                obj_num=int(class_ids.shape[0]),
            )
            used_count = flashsplat_class_rows(
                render_pkg["used_count"].detach().float().cpu().numpy(),
                int(class_ids.shape[0]),
                gaussian_count,
            )
            indices, vote_classes, weights = dense_sparse_view_votes(used_count, class_ids)
            vote_path = vote_dir / f"{Path(filename).stem}.npz"
            np.savez_compressed(vote_path, indices=indices, class_ids=vote_classes, weights=weights)
            vote_frames.append(
                {
                    "file": filename,
                    "camera_index": camera_index,
                    "camera_id": int(frame.get("camera_id", camera_index)),
                    "vote_file": vote_path.relative_to(output_dir).as_posix(),
                    "composed_map": map_path.relative_to(output_dir).as_posix(),
                    "oov_mask_file": str(oov_frame.get("mask_file", "")) if oov_frame else "",
                    "oov_pixel_counts": pixel_counts,
                    "present_project_class_ids": [int(value) for value in class_ids[1:]],
                    "visible_gaussian_count": int(np.unique(indices).size),
                    "sparse_vote_count": int(weights.size),
                }
            )
            print(
                f"lifted {filename}: visible_gaussians={np.unique(indices).size} "
                f"class_votes={weights.size} oov_pixels={sum(pixel_counts.values())}"
            )
            del render_pkg, used_count, gt_mask
            torch.cuda.empty_cache()

    visible_views = np.zeros((gaussian_count,), dtype=np.uint16)
    incumbent_winner_views = np.zeros((gaussian_count,), dtype=np.uint16)
    oov_winner_views = np.zeros((oov_ids.size, gaussian_count), dtype=np.uint16)
    oov_positive_views = np.zeros((oov_ids.size, gaussian_count), dtype=np.uint16)
    oov_mass = np.zeros((oov_ids.size, gaussian_count), dtype=np.float32)
    for frame in vote_frames:
        vote_path = output_dir / str(frame["vote_file"])
        with np.load(vote_path, allow_pickle=False) as data:
            indices = data["indices"]
            classes = data["class_ids"]
            weights = data["weights"]
        unique, winners, _ = camera_winners(indices, classes, weights, gaussian_count)
        visible_views[unique] += np.uint16(1)
        for class_index, label_id in enumerate(oov_ids):
            class_mask = classes == label_id
            if np.any(class_mask):
                class_indices = indices[class_mask].astype(np.int64, copy=False)
                class_weights = weights[class_mask]
                np.add.at(oov_mass[class_index], class_indices, class_weights)
                per_view_mass = np.zeros((gaussian_count,), dtype=np.float32)
                np.add.at(per_view_mass, class_indices, class_weights)
                oov_positive_views[class_index, per_view_mass >= args.oov_positive_mass_threshold] += np.uint16(1)
            winner_mask = winners == label_id
            if np.any(winner_mask):
                oov_winner_views[class_index, unique[winner_mask].astype(np.int64, copy=False)] += np.uint16(1)
        base_at_winner = base_labels[unique.astype(np.int64, copy=False)]
        incumbent_mask = (base_at_winner > 0) & (winners == base_at_winner)
        if np.any(incumbent_mask):
            incumbent_winner_views[unique[incumbent_mask].astype(np.int64, copy=False)] += np.uint16(1)

    labels, evidence = choose_oov_replacements(
        base_labels,
        visible_views,
        oov_winner_views,
        oov_positive_views,
        oov_mass,
        incumbent_winner_views,
        oov_ids,
        min_visible_views=args.min_visible_views,
        min_oov_winner_views=args.min_oov_winner_views,
        min_oov_winner_share=args.min_oov_winner_share,
        min_oov_mass_share=args.min_oov_mass_share,
        min_oov_positive_views=args.min_oov_positive_views,
        winner_margin=args.winner_margin,
    )
    np.save(output_dir / "gaussian_labels.npy", labels)
    np.save(output_dir / "gaussian_project_class_ids.npy", labels)
    np.save(output_dir / "visible_view_count.npy", visible_views)
    np.save(output_dir / "incumbent_winner_view_count.npy", incumbent_winner_views)
    np.save(output_dir / "oov_winner_view_count.npy", oov_winner_views)
    np.save(output_dir / "oov_positive_view_count.npy", oov_positive_views)
    np.save(output_dir / "oov_mass.npy", oov_mass)

    output_map = dict(base_map)
    output_map.update({"source": SOURCE, "contract": CONTRACT})
    output_map["labels"] = [
        dict(item)
        for label_id, item in sorted(base_items.items())
        if label_id == 0 or np.any(labels == label_id)
    ]
    for name, label_id in oov_label_ids.items():
        output_map["labels"].append(
            {
                "id": label_id,
                "name": name,
                "class": name,
                "type": "thing",
                "source_pipeline": SOURCE,
                "gaussian_count": int(np.count_nonzero(labels == label_id)),
            }
        )
    output_map["oov_label_ids"] = oov_label_ids
    (output_dir / "label_map.json").write_text(json.dumps(output_map, indent=2), encoding="utf-8")

    semantic_ply = None
    if not args.no_semantic_ply:
        semantic_ply = output_dir / "semantic_point_cloud.ply"
        partial = output_dir / "semantic_point_cloud.ply.partial"
        write_ply_with_labels(ply_path, partial, labels)
        partial.replace(semantic_ply)

    changed = labels != base_labels
    transitions: list[dict[str, Any]] = []
    for label_id, count in zip(*np.unique(base_labels[changed], return_counts=True)):
        item = base_items[int(label_id)]
        transitions.append(
            {
                "base_label_id": int(label_id),
                "base_class": normalize_class_name(item.get("class", "")),
                "base_name": str(item.get("name", item.get("class", label_id))),
                "gaussian_count": int(count),
            }
        )
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "status": "ok",
        "model_path": str(args.model_path),
        "source_ply": str(ply_path),
        "segmentation_manifest": str(args.segmentation_manifest),
        "oov_manifest": str(args.oov_manifest),
        "oov_mask_dir": str(args.oov_mask_dir),
        "base_labels": str(args.base_labels),
        "base_label_map": str(args.base_label_map),
        "gaussian_count": gaussian_count,
        "camera_count": len(vote_frames),
        "oov_label_ids": oov_label_ids,
        "oov_pixel_totals": oov_pixel_totals,
        "selection_thresholds": {
            "min_visible_views": args.min_visible_views,
            "min_oov_winner_views": args.min_oov_winner_views,
            "min_oov_winner_share": args.min_oov_winner_share,
            "min_oov_mass_share": args.min_oov_mass_share,
            "min_oov_positive_views": args.min_oov_positive_views,
            "oov_positive_mass_threshold": args.oov_positive_mass_threshold,
            "winner_margin": args.winner_margin,
        },
        "changed_gaussian_count": int(np.count_nonzero(changed)),
        "newly_labeled_count": int(np.count_nonzero(changed & (base_labels == 0))),
        "relabeled_count": int(np.count_nonzero(changed & (base_labels > 0))),
        "output_label_histogram": _histogram(labels),
        "base_label_histogram": _histogram(base_labels),
        "base_transitions": transitions,
        "candidate_counts": {
            name: int(np.count_nonzero(labels == label_id))
            for name, label_id in oov_label_ids.items()
        },
        "evidence_summary": {
            "visible_at_least_minimum": int(np.count_nonzero(visible_views >= args.min_visible_views)),
            "accepted_oov": int(np.count_nonzero(evidence["fill"])),
            "candidate_winner_share_median": float(np.median(evidence["winner_share"][visible_views > 0]))
            if np.any(visible_views > 0)
            else 0.0,
        },
        "semantic_ply": str(semantic_ply) if semantic_ply else None,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "frames": vote_frames,
    }
    (output_dir / "multiclass_fusion_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    vote_manifest = {
        "source": SOURCE,
        "contract": CONTRACT,
        "ply_path": str(ply_path),
        "segmentation_manifest": str(args.segmentation_manifest),
        "oov_manifest": str(args.oov_manifest),
        "base_labels": str(args.base_labels),
        "gaussian_count": gaussian_count,
        "camera_count": len(vote_frames),
        "oov_label_ids": oov_label_ids,
        "one_composed_multiclass_map_per_camera": True,
        "base_fallback_used": True,
        "frames": vote_frames,
    }
    (output_dir / "vote_manifest.json").write_text(json.dumps(vote_manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
