#!/usr/bin/env python3
"""Apply a reviewed label CSV to automatic 3DGS Gaussian labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from add_labels_from_npy import write_ply_with_labels


@dataclass
class ReviewRow:
    source_id: int
    source_name: str
    final_id: int
    final_name: str
    final_class: str
    review_status: str = ""
    notes: str = ""


@dataclass
class FinalLabel:
    label_id: int
    name: str
    semantic_class: str
    source_ids: list[int] = field(default_factory=list)
    gaussian_count: int = 0


def parse_int(value: str, column: str, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer in column {column!r} at CSV line {line_number}: {value!r}") from exc


def read_review_csv(path: Path) -> dict[int, ReviewRow]:
    rows: dict[int, ReviewRow] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_id", "final_id", "final_name", "final_class"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            source_id = parse_int(str(row["source_id"]).strip(), "source_id", line_number)
            final_id = parse_int(str(row["final_id"]).strip(), "final_id", line_number)
            if source_id < 0:
                raise ValueError(f"source_id must be non-negative at CSV line {line_number}")
            if final_id < 0:
                raise ValueError(f"final_id must be non-negative at CSV line {line_number}")
            if source_id in rows:
                raise ValueError(f"Duplicate source_id {source_id} in {path}")
            rows[source_id] = ReviewRow(
                source_id=source_id,
                source_name=str(row.get("source_name", "")).strip(),
                final_id=final_id,
                final_name=str(row["final_name"]).strip(),
                final_class=str(row["final_class"]).strip(),
                review_status=str(row.get("review_status", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
    if not rows:
        raise ValueError(f"{path} has no review rows")
    return rows


def validate_review_rows(
    rows: dict[int, ReviewRow],
    used_source_ids: set[int],
    allow_object_candidate: bool,
    default_missing_to_zero: bool,
) -> dict[int, ReviewRow]:
    if 0 not in rows:
        rows[0] = ReviewRow(
            source_id=0,
            source_name="unlabeled",
            final_id=0,
            final_name="unlabeled",
            final_class="unlabeled",
            review_status="locked",
        )

    if rows[0].final_id != 0:
        raise ValueError("source label 0 must map to final label 0")

    missing = sorted(used_source_ids - set(rows))
    if missing and not default_missing_to_zero:
        raise ValueError(
            "Review CSV does not map every source label in the auto labels: "
            + ", ".join(str(label) for label in missing)
        )
    for source_id in missing:
        rows[source_id] = ReviewRow(
            source_id=source_id,
            source_name=f"source_{source_id}",
            final_id=0,
            final_name="unlabeled",
            final_class="unlabeled",
            notes="default_missing_to_zero",
        )

    for row in rows.values():
        if row.final_id == 0:
            row.final_name = row.final_name or "unlabeled"
            row.final_class = row.final_class or "unlabeled"
            continue
        if not row.final_name:
            raise ValueError(f"source_id {row.source_id} maps to final_id {row.final_id} without final_name")
        if not row.final_class:
            raise ValueError(f"source_id {row.source_id} maps to final_id {row.final_id} without final_class")
        if not allow_object_candidate and row.final_class == "object_candidate":
            raise ValueError(
                f"source_id {row.source_id} still has final_class=object_candidate; "
                "rename it or pass --allow-object-candidate for a non-final smoke run"
            )
    return rows


def remap_labels(labels: np.ndarray, rows: dict[int, ReviewRow]) -> np.ndarray:
    final_labels = np.zeros(labels.shape, dtype=np.int32)
    for source_id, row in rows.items():
        final_labels[labels == source_id] = row.final_id
    return final_labels


def build_final_labels(final_labels: np.ndarray, rows: dict[int, ReviewRow]) -> list[FinalLabel]:
    used_final_ids = {int(label) for label in np.unique(final_labels)}
    by_final_id: dict[int, list[ReviewRow]] = defaultdict(list)
    for row in rows.values():
        if row.final_id in used_final_ids:
            by_final_id[row.final_id].append(row)

    histogram = {int(label): int(count) for label, count in zip(*np.unique(final_labels, return_counts=True))}
    final_items: list[FinalLabel] = []
    for final_id in sorted(used_final_ids):
        candidate_rows = by_final_id.get(final_id, [])
        names = {row.final_name for row in candidate_rows if row.final_name}
        classes = {row.final_class for row in candidate_rows if row.final_class}
        if len(names) > 1:
            raise ValueError(f"final_id {final_id} has inconsistent final_name values: {sorted(names)}")
        if len(classes) > 1:
            raise ValueError(f"final_id {final_id} has inconsistent final_class values: {sorted(classes)}")
        name = next(iter(names), "unlabeled" if final_id == 0 else f"label_{final_id}")
        semantic_class = next(iter(classes), "unlabeled" if final_id == 0 else "unknown")
        final_items.append(
            FinalLabel(
                label_id=final_id,
                name=name,
                semantic_class=semantic_class,
                source_ids=sorted(row.source_id for row in candidate_rows),
                gaussian_count=histogram.get(final_id, 0),
            )
        )
    return final_items


def write_outputs(
    output_dir: Path,
    labels: np.ndarray,
    final_items: list[FinalLabel],
    scene: str,
    review_csv: Path,
    auto_labels: Path,
    input_ply: Path | None,
    semantic_ply_name: str,
    overwrite: bool,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "gaussian_labels.npy"
    label_map_path = output_dir / "label_map.json"
    summary_path = output_dir / "label_review_summary.json"
    semantic_ply_path = output_dir / semantic_ply_name if input_ply is not None else None

    output_paths = [labels_path, label_map_path, summary_path]
    if semantic_ply_path is not None:
        output_paths.append(semantic_ply_path)
    for path in output_paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    np.save(labels_path, labels)
    label_map = {
        "scene": scene,
        "labels": [
            {"id": item.label_id, "name": item.name, "class": item.semantic_class}
            for item in final_items
        ],
    }
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    if input_ply is not None and semantic_ply_path is not None:
        write_ply_with_labels(input_ply, semantic_ply_path, labels, batch_size=batch_size)

    summary = {
        "scene": scene,
        "auto_labels": str(auto_labels),
        "review_csv": str(review_csv),
        "labels_npy": str(labels_path),
        "label_map": str(label_map_path),
        "semantic_ply": str(semantic_ply_path) if semantic_ply_path else None,
        "final_label_count": len(final_items),
        "final_label_histogram": {str(item.label_id): item.gaussian_count for item in final_items},
        "final_labels": [
            {
                "id": item.label_id,
                "name": item.name,
                "class": item.semantic_class,
                "source_ids": item.source_ids,
                "gaussian_count": item.gaussian_count,
            }
            for item in final_items
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-labels", required=True, type=Path)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--input-ply", type=Path)
    parser.add_argument("--semantic-ply-name", default="semantic_point_cloud.ply")
    parser.add_argument("--batch-size", default=100_000, type=int)
    parser.add_argument("--allow-object-candidate", action="store_true")
    parser.add_argument("--default-missing-to-zero", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_labels = np.load(args.auto_labels)
    if source_labels.ndim != 1:
        raise ValueError(f"{args.auto_labels} must contain a one-dimensional label array")
    used_source_ids = {int(label) for label in np.unique(source_labels)}

    rows = read_review_csv(args.review_csv)
    rows = validate_review_rows(
        rows,
        used_source_ids=used_source_ids,
        allow_object_candidate=args.allow_object_candidate,
        default_missing_to_zero=args.default_missing_to_zero,
    )
    final_labels = remap_labels(source_labels, rows)
    final_items = build_final_labels(final_labels, rows)
    summary = write_outputs(
        output_dir=args.output_dir,
        labels=final_labels,
        final_items=final_items,
        scene=args.scene,
        review_csv=args.review_csv,
        auto_labels=args.auto_labels,
        input_ply=args.input_ply,
        semantic_ply_name=args.semantic_ply_name,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    print(f"wrote {summary['labels_npy']}")
    print(f"wrote {summary['label_map']}")
    if summary["semantic_ply"]:
        print(f"wrote {summary['semantic_ply']}")
    print(json.dumps(summary["final_label_histogram"], sort_keys=True))


if __name__ == "__main__":
    main()
