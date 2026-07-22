#!/usr/bin/env python3
"""Fuse accumulated multi-view semantic votes into final per-Gaussian labels.

Two fusion modes share the same vote-thresholding core:

- ``full``: every Gaussian is labeled from the votes alone. This is the pure
  dense-semantic route (ablation / replacement front-end).
- ``fill``: an accepted GroundingDINO baseline is loaded and kept atomic; only
  Gaussians the baseline left unlabeled (label 0) receive vote labels. By
  default only stuff classes are filled, so accepted thing instances can never
  be contradicted. This is the primary production mode.

Vote decision per Gaussian g (column c wins):

    assign iff  sum_c votes[g] > 0
            and views_supporting[g, c] >= --min-views
            and votes[g, c] / sum votes[g] >= --min-agreement
            and views_supporting[g, c] / visible_views[g] >= --min-visible-ratio

The decision functions are pure numpy so tests can exercise them without GPU
or torch. Thing classes are split into instances by voxel connected
components, reusing voxel_components from the accepted fusion stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import voxel_components
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


# ---------------------------------------------------------------------------
# Pure vote logic (numpy only, unit-testable)
# ---------------------------------------------------------------------------


def decide_vote_labels(
    votes: np.ndarray,
    views_supporting: np.ndarray,
    visible_views: np.ndarray,
    min_views: int,
    min_agreement: float,
    min_visible_ratio: float,
) -> dict[str, np.ndarray]:
    """Threshold accumulated votes into compact class choices (0 = unlabeled)."""
    if votes.ndim != 2:
        raise ValueError("votes must have shape (N, C)")
    if views_supporting.shape != votes.shape:
        raise ValueError("views_supporting must match votes shape")
    if visible_views.shape != (votes.shape[0],):
        raise ValueError("visible_views must have shape (N,)")

    total = votes.sum(axis=1)
    winner_col = votes.argmax(axis=1)
    rows = np.arange(votes.shape[0])
    winner_mass = votes[rows, winner_col]
    agreement = np.divide(
        winner_mass,
        total,
        out=np.zeros_like(winner_mass),
        where=total > 0,
    )
    support = views_supporting[rows, winner_col].astype(np.float32)
    visible = np.maximum(visible_views.astype(np.float32), 1.0)
    visible_ratio = support / visible

    assign = (
        (total > 0)
        & (support >= float(min_views))
        & (agreement >= float(min_agreement))
        & (visible_ratio >= float(min_visible_ratio))
    )
    class_choice = np.where(assign, winner_col + 1, 0).astype(np.int16)
    return {
        "class_choice": class_choice,
        "agreement": agreement.astype(np.float32),
        "support_views": support.astype(np.float32),
        "raw_voted": (total > 0),
    }


def merge_fill(
    base_labels: np.ndarray,
    class_choice: np.ndarray,
    class_types: list[str],
    stuff_only: bool,
) -> np.ndarray:
    """Return the vote class choice restricted to base-unlabeled Gaussians."""
    if base_labels.shape != class_choice.shape:
        raise ValueError("base labels and vote choice must have the same shape")
    fill_choice = np.where(base_labels == 0, class_choice, 0).astype(np.int16)
    if stuff_only:
        keep = np.zeros(len(class_types), dtype=bool)
        for compact_id, kind in enumerate(class_types):
            keep[compact_id] = kind == "stuff"
        fill_choice = np.where(keep[fill_choice], fill_choice, 0).astype(np.int16)
    return fill_choice


# ---------------------------------------------------------------------------
# Instancing and label assembly
# ---------------------------------------------------------------------------


def class_voxel_size(
    vertex_data: np.memmap,
    indices: np.ndarray,
    multiplier: float,
    min_voxel: float,
    max_voxel: float,
) -> float:
    log_scales = np.column_stack(
        [vertex_data[axis][indices].astype(np.float64) for axis in ("scale_0", "scale_1", "scale_2")]
    )
    gaussian_scales = np.exp(np.clip(log_scales.max(axis=1), -20.0, 5.0))
    finite = gaussian_scales[np.isfinite(gaussian_scales)]
    median_scale = float(np.median(finite)) if finite.shape[0] else min_voxel
    voxel_size = max(min_voxel, median_scale * multiplier)
    if max_voxel > 0:
        voxel_size = min(voxel_size, max_voxel)
    return voxel_size


def split_thing_instances(
    indices: np.ndarray,
    vertex_data: np.memmap,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
    min_instance_gaussians: int,
) -> tuple[list[np.ndarray], int]:
    """Split one thing class into connected-component instances."""
    points = np.column_stack(
        [vertex_data[axis][indices].astype(np.float64) for axis in ("x", "y", "z")]
    )
    voxel_size = class_voxel_size(
        vertex_data, indices, voxel_scale_multiplier, min_voxel_size, max_voxel_size
    )
    point_components, component_sizes, _stats = voxel_components(points, voxel_size)
    instances: list[np.ndarray] = []
    dropped = 0
    order = np.argsort(component_sizes)[::-1]
    for component in order:
        member = indices[point_components == component]
        if member.shape[0] >= min_instance_gaussians:
            instances.append(member)
        else:
            dropped += int(member.shape[0])
    return instances, dropped


class LabelBuilder:
    """Assign stable final label ids and accumulate label-map entries."""

    def __init__(self, base_entries: list[dict[str, Any]] | None = None) -> None:
        self.entries: list[dict[str, Any]] = [dict(item) for item in (base_entries or [])]
        ids = [int(item["id"]) for item in self.entries]
        if self.entries and 0 not in ids:
            raise ValueError("Base label map must include id 0")
        self.next_id = max(ids) + 1 if ids else 1
        if not self.entries:
            self.entries.append({"id": 0, "name": "unlabeled", "class": "unlabeled"})

    def existing_stuff_id(self, class_name: str) -> int | None:
        # Merge only into entries that follow the stuff naming convention
        # (name == class, e.g. "sky"). Base label maps carry no thing/stuff
        # type, and thing instances are named "<class>_NN"; matching by class
        # alone could silently extend an accepted thing instance.
        for item in self.entries:
            if (
                int(item["id"]) != 0
                and str(item["class"]) == class_name
                and str(item["name"]) == class_name
            ):
                return int(item["id"])
        return None

    def add(self, name: str, class_name: str) -> int:
        label_id = self.next_id
        self.next_id += 1
        self.entries.append({"id": label_id, "name": name, "class": class_name})
        return label_id

    def remove(self, label_id: int) -> None:
        self.entries = [item for item in self.entries if int(item["id"]) != label_id]

    def label_map(self, scene: str) -> dict[str, Any]:
        return {
            "scene": scene,
            "labels": sorted(self.entries, key=lambda item: int(item["id"])),
        }


def assemble_labels(
    class_choice: np.ndarray,
    class_names: list[str],
    class_types: list[str],
    vertex_data: np.memmap,
    builder: LabelBuilder,
    base_labels: np.ndarray | None,
    extend_existing_stuff: bool,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
    min_instance_gaussians: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Turn compact vote choices into final label ids on top of base labels."""
    labels = (
        base_labels.astype(np.int32).copy()
        if base_labels is not None
        else np.zeros(class_choice.shape, dtype=np.int32)
    )
    records: list[dict[str, Any]] = []
    # Continue instance numbering after any base label-map entries of the same
    # class, so fill --fill-things can never emit a duplicate instance name.
    thing_ordinals: dict[str, int] = {}
    for item in builder.entries:
        if int(item["id"]) == 0:
            continue
        class_name = str(item["class"])
        thing_ordinals[class_name] = thing_ordinals.get(class_name, 0) + 1

    for compact_id in np.unique(class_choice):
        compact_id = int(compact_id)
        if compact_id == 0:
            continue
        class_name = class_names[compact_id]
        kind = class_types[compact_id]
        indices = np.flatnonzero(class_choice == compact_id)

        if kind == "stuff":
            label_id = None
            merged = False
            if extend_existing_stuff:
                label_id = builder.existing_stuff_id(class_name)
                merged = label_id is not None
            if label_id is None:
                label_id = builder.add(class_name, class_name)
            labels[indices] = label_id
            records.append(
                {
                    "id": label_id,
                    "name": class_name,
                    "class": class_name,
                    "type": "stuff",
                    "gaussians": int(indices.shape[0]),
                    "merged_into_existing": merged,
                }
            )
            continue

        instances, dropped = split_thing_instances(
            indices,
            vertex_data,
            voxel_scale_multiplier,
            min_voxel_size,
            max_voxel_size,
            min_instance_gaussians,
        )
        for member in instances:
            ordinal = thing_ordinals.get(class_name, 0) + 1
            thing_ordinals[class_name] = ordinal
            name = f"{class_name}_{ordinal:02d}"
            label_id = builder.add(name, class_name)
            labels[member] = label_id
            records.append(
                {
                    "id": label_id,
                    "name": name,
                    "class": class_name,
                    "type": "thing",
                    "gaussians": int(member.shape[0]),
                    "dropped_fragment_gaussians": 0,
                }
            )
        if dropped:
            records.append(
                {
                    "id": 0,
                    "name": f"{class_name}_dropped_fragments",
                    "class": class_name,
                    "type": "thing",
                    "gaussians": dropped,
                    "dropped_fragment_gaussians": dropped,
                }
            )
    return labels, records


def prune_small_new_labels(
    labels: np.ndarray,
    records: list[dict[str, Any]],
    builder: LabelBuilder,
    min_thing_gaussians: int,
    min_stuff_gaussians: int,
) -> list[dict[str, Any]]:
    """Zero out newly created labels that end below the size thresholds."""
    pruned: list[dict[str, Any]] = []
    for record in records:
        label_id = int(record["id"])
        if label_id == 0 or record.get("merged_into_existing"):
            continue
        threshold = (
            min_stuff_gaussians if record["type"] == "stuff" else min_thing_gaussians
        )
        count = int((labels == label_id).sum())
        if count < threshold:
            labels[labels == label_id] = 0
            builder.remove(label_id)
            pruned.append({**record, "gaussians": count, "prune_threshold": threshold})
    return pruned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def point_cloud_path(model_path: Path, iteration: int) -> Path:
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    return ply_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--votes", required=True, type=Path, help="votes.npz from lift stage")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", default="fill", choices=["full", "fill"])
    parser.add_argument("--base-labels", type=Path, help="Accepted gaussian_labels.npy (fill)")
    parser.add_argument("--base-label-map", type=Path, help="Accepted label_map.json (fill)")
    parser.add_argument(
        "--fill-things",
        action="store_true",
        help="In fill mode also fill thing classes (default: stuff only)",
    )
    parser.add_argument("--min-views", default=2, type=int)
    parser.add_argument("--min-agreement", default=0.5, type=float)
    parser.add_argument("--min-visible-ratio", default=0.0, type=float)
    parser.add_argument("--min-thing-gaussians", default=5000, type=int)
    parser.add_argument("--min-stuff-gaussians", default=10000, type=int)
    parser.add_argument("--instance-voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--instance-min-voxel-size", default=0.01, type=float)
    parser.add_argument("--instance-max-voxel-size", default=0.20, type=float)
    parser.add_argument("--instance-min-component-gaussians", default=500, type=int)
    parser.add_argument("--no-extend-existing-stuff", action="store_true")
    parser.add_argument("--labels-path", required=True, type=Path)
    parser.add_argument("--label-map-path", required=True, type=Path)
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.labels_path, args.label_map_path, args.summary_path):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    with np.load(args.votes, allow_pickle=False) as data:
        votes = data["votes"]
        views_supporting = data["views_supporting"]
        visible_views = data["visible_views"]
        class_names = [str(name) for name in data["class_names"]]
        class_types = [str(kind) for kind in data["class_types"]]

    decision = decide_vote_labels(
        votes,
        views_supporting,
        visible_views,
        args.min_views,
        args.min_agreement,
        args.min_visible_ratio,
    )
    class_choice = decision["class_choice"]
    total_gaussians = int(class_choice.shape[0])
    raw_unlabeled_ratio = float(1.0 - decision["raw_voted"].mean())
    thresholded_unlabeled_ratio = float((class_choice == 0).mean())

    base_labels = None
    base_entries: list[dict[str, Any]] | None = None
    if args.mode == "fill":
        if args.base_labels is None or args.base_label_map is None:
            raise ValueError("fill mode requires --base-labels and --base-label-map")
        base_labels = np.load(args.base_labels)
        if base_labels.shape != class_choice.shape:
            raise ValueError(
                f"Base labels have {base_labels.shape[0]} Gaussians, "
                f"votes have {class_choice.shape[0]}"
            )
        base_map = json.loads(args.base_label_map.read_text(encoding="utf-8"))
        base_entries = base_map["labels"]
        class_choice = merge_fill(
            base_labels,
            class_choice,
            class_types,
            stuff_only=not args.fill_things,
        )

    ply_path = point_cloud_path(args.model_path, args.iteration)
    _header, vertex_data = vertex_data_memmap(ply_path)

    builder = LabelBuilder(base_entries)
    labels, records = assemble_labels(
        class_choice,
        class_names,
        class_types,
        vertex_data,
        builder,
        base_labels,
        extend_existing_stuff=not args.no_extend_existing_stuff,
        voxel_scale_multiplier=args.instance_voxel_scale_multiplier,
        min_voxel_size=args.instance_min_voxel_size,
        max_voxel_size=args.instance_max_voxel_size,
        min_instance_gaussians=args.instance_min_component_gaussians,
    )
    pruned = prune_small_new_labels(
        labels,
        records,
        builder,
        args.min_thing_gaussians,
        args.min_stuff_gaussians,
    )

    final_unlabeled_ratio = float((labels == 0).mean())
    label_map = builder.label_map(args.scene)

    args.labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.labels_path, labels.astype(np.int32))
    args.label_map_path.parent.mkdir(parents=True, exist_ok=True)
    args.label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    if args.semantic_ply_path is not None:
        if args.semantic_ply_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{args.semantic_ply_path} exists; pass --overwrite")
            args.semantic_ply_path.unlink()
        write_ply_with_labels(ply_path, args.semantic_ply_path, labels.astype(np.int32))

    summary = {
        "scene": args.scene,
        "mode": args.mode,
        "votes_path": str(args.votes),
        "ply_path": str(ply_path),
        "total_gaussians": total_gaussians,
        "thresholds": {
            "min_views": args.min_views,
            "min_agreement": args.min_agreement,
            "min_visible_ratio": args.min_visible_ratio,
            "min_thing_gaussians": args.min_thing_gaussians,
            "min_stuff_gaussians": args.min_stuff_gaussians,
        },
        "fill": {
            "enabled": args.mode == "fill",
            "stuff_only": args.mode == "fill" and not args.fill_things,
            "base_labels": str(args.base_labels) if args.base_labels else None,
            "base_unlabeled_ratio": (
                float((base_labels == 0).mean()) if base_labels is not None else None
            ),
        },
        "raw_unlabeled_ratio": raw_unlabeled_ratio,
        "thresholded_unlabeled_ratio": thresholded_unlabeled_ratio,
        "final_unlabeled_ratio": final_unlabeled_ratio,
        "labels": records,
        "pruned_labels": pruned,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"raw unlabeled ratio        {raw_unlabeled_ratio:.6f}")
    print(f"thresholded unlabeled ratio {thresholded_unlabeled_ratio:.6f}")
    print(f"final unlabeled ratio      {final_unlabeled_ratio:.6f}")
    print(f"wrote {args.labels_path}")
    print(f"wrote {args.label_map_path}")
    if args.semantic_ply_path is not None:
        print(f"wrote {args.semantic_ply_path}")
    print(f"wrote {args.summary_path}")


if __name__ == "__main__":
    main()
