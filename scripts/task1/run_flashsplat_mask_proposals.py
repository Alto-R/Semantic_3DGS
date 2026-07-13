#!/usr/bin/env python3
"""Lift 2D mask proposals to sparse Gaussian support sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"


def load_mask_stack(mask_path: Path, height: int, width: int) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    with np.load(mask_path) as data:
        masks = data["masks"].astype(bool)
    if masks.ndim != 3:
        raise ValueError(f"{mask_path} masks must have shape (K,H,W)")
    if masks.shape[1:] != (height, width):
        resized = []
        for mask in masks:
            image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            image = image.resize((width, height), Image.Resampling.NEAREST)
            resized.append(np.asarray(image, dtype=np.uint8) > 0)
        masks = np.stack(resized, axis=0) if resized else np.zeros((0, height, width), dtype=bool)
    return masks


def resolve_mask_manifest(input_dir: Path, manifest_name: str, mask_dir_name: str) -> tuple[Path, Path]:
    if manifest_name:
        manifest_path = input_dir / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
    else:
        candidates = [
            "grounded_sam_manifest.json",
            "sam_auto_manifest.json",
        ]
        manifest_path = next((input_dir / name for name in candidates if (input_dir / name).exists()), None)
        if manifest_path is None:
            raise FileNotFoundError(
                f"Could not find any mask manifest in {input_dir}: {', '.join(candidates)}"
            )

    if mask_dir_name:
        mask_dir = input_dir / mask_dir_name
    elif manifest_path.name == "grounded_sam_manifest.json":
        mask_dir = input_dir / "mask_stacks"
        if not mask_dir.exists():
            mask_dir = input_dir / "grounded_sam_masks"
    else:
        mask_dir = input_dir / "sam_auto_masks"
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)
    return manifest_path, mask_dir


def build_index_mask(masks: np.ndarray, start: int, end: int) -> torch.Tensor:
    height, width = masks.shape[1:]
    indexed = np.zeros((height, width), dtype=np.float32)
    for local_id, mask_index in enumerate(range(start, end), start=1):
        indexed[masks[mask_index].astype(bool)] = float(local_id)
    return torch.from_numpy(indexed).to(device="cuda", dtype=torch.float32)


def normalized_class_name(mask_meta: Dict[str, Any]) -> str:
    value = mask_meta.get("class_name", mask_meta.get("class", "unknown"))
    return str(value or "unknown").strip().lower().replace(" ", "_")


def masks_by_class(masks: np.ndarray, frame: Dict[str, Any]) -> Dict[str, np.ndarray]:
    combined: Dict[str, np.ndarray] = {}
    metadata = frame.get("masks", [])
    for mask_index, mask in enumerate(masks):
        mask_meta = metadata[mask_index] if mask_index < len(metadata) else {}
        class_name = normalized_class_name(mask_meta)
        if class_name in {"", "unknown", "object_candidate"}:
            continue
        if class_name in combined:
            combined[class_name] |= mask
        else:
            combined[class_name] = mask.copy()
    return combined


def increment_view_counts(counts: np.ndarray, indices: np.ndarray) -> None:
    if indices.shape[0] > 0:
        counts[indices] += np.uint16(1)


def update_class_evidence(
    class_masks: Dict[str, np.ndarray],
    camera: Any,
    gaussians: Any,
    modules: Dict[str, Any],
    pipeline: Any,
    background: torch.Tensor,
    vertex_count: int,
    threshold: float,
    evidence: Dict[str, Dict[str, np.ndarray]],
) -> None:
    for class_name, class_mask in class_masks.items():
        binary_mask = torch.from_numpy(class_mask.astype(np.float32)).to(device="cuda")
        render_pkg = render_flashsplat(
            camera,
            gaussians,
            modules,
            pipeline,
            background,
            gt_mask=binary_mask,
            obj_num=2,
        )
        used_count = render_pkg["used_count"].detach().cpu().numpy()
        class_record = evidence.setdefault(
            class_name,
            {
                "positive_views": np.zeros((vertex_count,), dtype=np.uint16),
                "negative_views": np.zeros((vertex_count,), dtype=np.uint16),
            },
        )
        positive = np.flatnonzero(used_count[1] > threshold)
        negative = np.flatnonzero(used_count[0] > threshold)
        increment_view_counts(class_record["positive_views"], positive)
        increment_view_counts(class_record["negative_views"], negative)
        del used_count
        del render_pkg
        del binary_mask
        torch.cuda.empty_cache()


def write_class_evidence(
    output_dir: Path,
    evidence: Dict[str, Dict[str, np.ndarray]],
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[Dict[str, Any]] = []
    for class_name, arrays in sorted(evidence.items()):
        observed = (arrays["positive_views"] > 0) | (arrays["negative_views"] > 0)
        indices = np.flatnonzero(observed).astype(np.uint32)
        filename = f"{class_name}.npz"
        np.savez_compressed(
            output_dir / filename,
            indices=indices,
            positive_views=arrays["positive_views"][indices],
            negative_views=arrays["negative_views"][indices],
        )
        records.append(
            {
                "class": class_name,
                "file": filename,
                "gaussian_count": int(indices.shape[0]),
                "positive_gaussian_count": int((arrays["positive_views"] > 0).sum()),
                "negative_gaussian_count": int((arrays["negative_views"] > 0).sum()),
                "max_positive_views": int(arrays["positive_views"].max()),
                "max_negative_views": int(arrays["negative_views"].max()),
            }
        )
    manifest = {
        "source": "flashsplat_class_positive_negative_visibility",
        "threshold": threshold,
        "classes": records,
    }
    (output_dir / "class_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def save_support(
    output_path: Path,
    support_indices: np.ndarray,
    support_counts: np.ndarray,
    proposal_id: int,
    frame_file: str,
    mask_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        indices=support_indices.astype(np.uint32),
        counts=support_counts.astype(np.float32),
        proposal_id=np.asarray([proposal_id], dtype=np.int32),
        frame_file=np.asarray([frame_file]),
        mask_index=np.asarray([mask_index], dtype=np.int32),
    )


def proposal_metadata(
    proposal_id: int,
    support_file: str,
    frame: Dict[str, Any],
    camera_index: int,
    mask_index: int,
    mask_meta: Dict[str, Any],
    gaussian_count: int,
    fallback_area: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "proposal_id": proposal_id,
        "support_file": support_file,
        "frame_file": frame["file"],
        "camera_index": camera_index,
        "camera_id": int(frame["camera_id"]),
        "image_name": frame.get("image_name", ""),
        "mask_index": int(mask_index),
        "mask_area": int(mask_meta.get("area", fallback_area)),
        "gaussian_count": int(gaussian_count),
        "predicted_iou": float(mask_meta.get("predicted_iou", mask_meta.get("sam_score", 0.0))),
        "stability_score": float(mask_meta.get("stability_score", 1.0)),
    }
    for key in [
        "source",
        "class",
        "class_name",
        "phrase",
        "confidence",
        "grounding_score",
        "sam_score",
        "bbox",
        "bbox_xyxy",
    ]:
        if key in mask_meta:
            record[key] = mask_meta[key]
    if "class_name" not in record and "class" in record:
        record["class_name"] = record["class"]
    if "class" not in record and "class_name" in record:
        record["class"] = record["class_name"]
    if "confidence" not in record:
        record["confidence"] = float(record.get("grounding_score", record["predicted_iou"]))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sam-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-name", default="")
    parser.add_argument("--mask-dir-name", default="")
    parser.add_argument(
        "--flashsplat-root",
        default=DEFAULT_FLASHSPLAT_ROOT,
        type=Path,
    )
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--mask-batch-size", default=4, type=int)
    parser.add_argument("--max-masks-per-view", default=32, type=int)
    parser.add_argument("--support-threshold", default=0.0, type=float)
    parser.add_argument("--write-class-evidence", action="store_true")
    parser.add_argument("--class-evidence-threshold", default=0.05, type=float)
    parser.add_argument("--min-support-gaussians", default=100, type=int)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    if args.mask_batch_size <= 0:
        raise ValueError("--mask-batch-size must be positive")

    manifest_path, mask_dir = resolve_mask_manifest(
        args.sam_output_dir,
        args.manifest_name,
        args.mask_dir_name,
    )
    sam_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    support_dir = args.output_dir / "proposal_supports"
    support_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    proposal_id = 1
    proposals: List[Dict[str, Any]] = []
    class_evidence: Dict[str, Dict[str, np.ndarray]] = {}
    output_manifest: Dict[str, Any] = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "sam_output_dir": str(args.sam_output_dir),
        "source_manifest": str(manifest_path),
        "mask_dir": str(mask_dir),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "mask_batch_size": args.mask_batch_size,
        "support_threshold": args.support_threshold,
        "min_support_gaussians": args.min_support_gaussians,
        "write_class_evidence": args.write_class_evidence,
        "class_evidence_threshold": args.class_evidence_threshold,
        "proposals": proposals,
    }

    with torch.no_grad():
        for frame in sam_manifest["frames"]:
            camera_index = int(frame["camera_index"])
            camera_json = cameras[camera_index]
            camera = make_camera(camera_json, modules, args.max_width)
            mask_path = mask_dir / frame["mask_file"]
            masks = load_mask_stack(mask_path, int(camera.image_height), int(camera.image_width))
            if args.max_masks_per_view > 0:
                masks = masks[: args.max_masks_per_view]
            if masks.shape[0] == 0:
                print(f"skipped {frame['file']}: no masks")
                continue

            if args.write_class_evidence:
                frame_class_masks = masks_by_class(masks, frame)
                update_class_evidence(
                    frame_class_masks,
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    int(gaussians.get_xyz.shape[0]),
                    args.class_evidence_threshold,
                    class_evidence,
                )
                print(
                    f"class evidence: frame={frame['file']} "
                    f"classes={','.join(sorted(frame_class_masks))}"
                )

            for start in range(0, masks.shape[0], args.mask_batch_size):
                end = min(start + args.mask_batch_size, masks.shape[0])
                gt_mask = build_index_mask(masks, start, end)
                render_pkg = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    gt_mask=gt_mask,
                    obj_num=(end - start) + 1,
                )
                used_count = render_pkg["used_count"].detach().cpu()

                for local_id, mask_index in enumerate(range(start, end), start=1):
                    counts = used_count[local_id].numpy()
                    support = np.flatnonzero(counts > args.support_threshold)
                    if support.shape[0] < args.min_support_gaussians:
                        continue

                    support_file = f"proposal_{proposal_id:06d}.npz"
                    save_support(
                        support_dir / support_file,
                        support,
                        counts[support],
                        proposal_id,
                        frame["file"],
                        mask_index,
                    )
                    mask_meta = {}
                    if mask_index < len(frame.get("masks", [])):
                        mask_meta = frame["masks"][mask_index]
                    proposals.append(
                        proposal_metadata(
                            proposal_id,
                            support_file,
                            frame,
                            camera_index,
                            mask_index,
                            mask_meta,
                            int(support.shape[0]),
                            int(masks[mask_index].sum()),
                        )
                    )
                    print(
                        f"proposal {proposal_id:06d}: frame={frame['file']} "
                        f"mask={mask_index} gaussians={support.shape[0]}"
                    )
                    proposal_id += 1

                del used_count
                del gt_mask
                torch.cuda.empty_cache()

    (args.output_dir / "proposal_manifest.json").write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    if args.write_class_evidence:
        write_class_evidence(
            args.output_dir / "class_evidence",
            class_evidence,
            args.class_evidence_threshold,
        )
    print(f"wrote {args.output_dir / 'proposal_manifest.json'}")


if __name__ == "__main__":
    main()
