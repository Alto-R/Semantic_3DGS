#!/usr/bin/env python3
"""Prepare every DINOv3 query region for class-agnostic 3D lifting.

This stage deliberately assigns no semantic identity.  It converts the
disjoint per-view region map into the mask-stack contract consumed by the
existing FlashSplat proposal lifter and carries compact references to the full
DINOv3 query evidence exported alongside each map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_EVIDENCE_KEYS = (
    "query_indices",
    "class_probabilities",
    "no_object_probabilities",
    "query_embeddings",
)


def load_region_evidence(
    path: Path,
    expected_regions: int,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_EVIDENCE_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"{path} lacks full DINOv3 query evidence: {', '.join(missing)}"
            )
        arrays = {
            "region_id": np.asarray(data["region_id"], dtype=np.uint16),
            "region_confidence": np.asarray(
                data["region_confidence"], dtype=np.float32
            ),
            **{
                key: np.asarray(data[key])
                for key in REQUIRED_EVIDENCE_KEYS
            },
        }

    if arrays["region_id"].ndim != 2:
        raise ValueError(f"{path} region_id must be a 2D map")
    if arrays["region_confidence"].shape != arrays["region_id"].shape:
        raise ValueError(f"{path} region confidence shape differs from region_id")
    if arrays["query_indices"].shape != (expected_regions,):
        raise ValueError(f"{path} query_indices count differs from manifest")
    if arrays["class_probabilities"].ndim != 2 or arrays[
        "class_probabilities"
    ].shape[0] != expected_regions:
        raise ValueError(f"{path} class probability rows differ from manifest")
    if arrays["no_object_probabilities"].shape != (expected_regions,):
        raise ValueError(f"{path} no-object rows differ from manifest")
    if arrays["query_embeddings"].ndim != 2 or arrays[
        "query_embeddings"
    ].shape[0] != expected_regions:
        raise ValueError(f"{path} query embedding rows differ from manifest")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{path} contains non-finite query evidence")
    if expected_regions:
        expected_ids = np.arange(1, expected_regions + 1, dtype=np.uint16)
        actual_ids = np.unique(arrays["region_id"])
        actual_ids = actual_ids[actual_ids > 0]
        if not np.array_equal(actual_ids, expected_ids):
            raise ValueError(f"{path} compact region ids differ from manifest")
        probability_sums = arrays["class_probabilities"].astype(
            np.float32
        ).sum(axis=1)
        if not np.allclose(probability_sums, 1.0, atol=2e-3):
            raise ValueError(f"{path} conditional class probabilities do not sum to one")
    return arrays


def prepare_frame(
    frame: dict[str, Any],
    manifest_dir: Path,
    mask_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    region_file = manifest_dir / str(frame["region_file"])
    regions = list(frame.get("regions", []))
    arrays = load_region_evidence(region_file, len(regions))
    masks = np.stack(
        [arrays["region_id"] == region_id for region_id in range(1, len(regions) + 1)],
        axis=0,
    ) if regions else np.zeros((0, *arrays["region_id"].shape), dtype=bool)

    stem = Path(str(frame["file"])).stem
    mask_file = f"{stem}.npz"
    evidence_file = f"{stem}.npz"
    np.savez_compressed(mask_dir / mask_file, masks=masks)
    np.savez_compressed(
        evidence_dir / evidence_file,
        query_indices=arrays["query_indices"].astype(np.int16, copy=False),
        class_probabilities=arrays["class_probabilities"].astype(
            np.float16, copy=False
        ),
        no_object_probabilities=arrays["no_object_probabilities"].astype(
            np.float16, copy=False
        ),
        query_embeddings=arrays["query_embeddings"].astype(
            np.float16, copy=False
        ),
    )

    metadata: list[dict[str, Any]] = []
    for row, region in enumerate(regions):
        region_id = row + 1
        area = int(masks[row].sum())
        if int(region["region_id"]) != region_id:
            raise ValueError(
                f"{region_file} region metadata is not compact and ordered"
            )
        if area != int(region["area"]):
            raise ValueError(f"{region_file} region area differs from manifest")
        metadata.append(
            {
                "source": "dinov3_mask2former_query_region",
                "class": "unassigned_region",
                "class_name": "unassigned_region",
                "phrase": "unassigned_region",
                "area": area,
                "bbox_xyxy": list(region["bbox_xyxy"]),
                "confidence": float(region["mean_region_confidence"]),
                "predicted_iou": float(region["mean_region_confidence"]),
                "stability_score": 1.0,
                "region_id": region_id,
                "query_index": int(arrays["query_indices"][row]),
                "query_evidence_file": (
                    Path("query_evidence") / evidence_file
                ).as_posix(),
                "query_evidence_row": row,
                "dinov3_query_diagnostic": region,
                "semantic_identity_assigned": False,
            }
        )

    return {
        **{
            key: frame[key]
            for key in (
                "file",
                "camera_index",
                "camera_id",
                "image_name",
                "render_width",
                "render_height",
                "view_quality",
                "camera_source",
            )
            if key in frame
        },
        "mask_file": mask_file,
        "masks": metadata,
        "region_count": len(regions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.dinov3_manifest.is_file():
        raise FileNotFoundError(args.dinov3_manifest)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} exists; pass --overwrite only for a reviewed rerun"
        )

    manifest = json.loads(args.dinov3_manifest.read_text(encoding="utf-8"))
    storage = manifest.get("query_evidence_storage", {})
    if not bool(storage.get("available", False)):
        raise ValueError("DINOv3 manifest does not contain full query evidence")

    mask_dir = args.output_dir / "mask_stacks"
    evidence_dir = args.output_dir / "query_evidence"
    mask_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.dinov3_manifest.parent
    frames = [
        prepare_frame(frame, manifest_dir, mask_dir, evidence_dir)
        for frame in manifest.get("frames", [])
    ]
    output = {
        "source": "dinov3_mask2former_query_regions",
        "contract": "class_agnostic_query_masks_with_soft_evidence_v1",
        "dinov3_manifest": str(args.dinov3_manifest),
        "semantic_identity_assigned": False,
        "v5_used": False,
        "dinov2_used": False,
        "frame_count": len(frames),
        "region_count": sum(int(frame["region_count"]) for frame in frames),
        "frames": frames,
    }
    output_path = args.output_dir / "query_region_manifest.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
