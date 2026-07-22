#!/usr/bin/env python3
"""Render 3DGS views and run dense ADE20K semantic segmentation per view.

This is the dense-semantic replacement for the GroundingDINO+SAM stage. Each
selected camera is rendered with the FlashSplat rasterizer, segmented by a
pluggable backend (see dense_seg_backends), remapped from raw ADE20K classes
into the compact project ontology, and written as one npz per view:

    seg/<view>.npz
      project_class : (H, W) uint8   compact project ids, 0 = ignore
      confidence    : (H, W) float16 backend confidence in [0, 1]
      ade20k_class  : (H, W) int16   raw ADE20K ids (only with --save-raw)

Confidence gating happens later in lift_semantic_votes.py, so a lift rerun
with a different threshold never requires re-running the model.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ade20k_ontology import ProjectOntology, load_ontology
from dense_seg_backends import BACKEND_NAMES, build_backend
from flashsplat_cameras import (
    background_tensor,
    camera_filename,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
    selected_camera_items,
    tensor_to_rgb_array,
)
from semantic_palette import CLASS_COLORS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"


def class_color(class_name: str) -> tuple[int, int, int]:
    rgb = CLASS_COLORS.get(class_name)
    if rgb is None:
        # Deterministic fallback hue for classes outside the shared palette.
        # Built-in hash() is salted per process, so use a stable digest.
        digest = hashlib.md5(class_name.encode("utf-8")).digest()
        hue = (int.from_bytes(digest[:4], "big") % 360) / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return tuple(int(round(value * 255.0)) for value in rgb)


def overlay_image(
    rgb: np.ndarray,
    project_class: np.ndarray,
    ontology: ProjectOntology,
    alpha: float = 0.55,
) -> Image.Image:
    color_map = np.zeros((len(ontology.class_names), 3), dtype=np.uint8)
    for compact_id, name in enumerate(ontology.class_names):
        if compact_id == 0:
            continue
        color_map[compact_id] = class_color(name)
    colored = color_map[project_class]
    labeled = project_class > 0
    blended = rgb.astype(np.float32)
    blended[labeled] = (
        (1.0 - alpha) * blended[labeled] + alpha * colored[labeled].astype(np.float32)
    )
    return Image.fromarray(blended.clip(0.0, 255.0).astype(np.uint8))


def frame_histogram(project_class: np.ndarray, ontology: ProjectOntology) -> dict[str, int]:
    values, counts = np.unique(project_class, return_counts=True)
    return {
        ontology.class_names[int(value)]: int(count)
        for value, count in zip(values, counts)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument(
        "--count",
        default=0,
        type=int,
        help="Number of evenly spaced cameras; <=0 uses every cameras.json camera",
    )
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="Also store raw ADE20K ids")
    parser.add_argument("--overlay-alpha", default=0.55, type=float)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--backend", default="mask2former", choices=BACKEND_NAMES)
    parser.add_argument(
        "--mask2former-model",
        default="facebook/mask2former-swin-large-ade-semantic",
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--dinov3-repo", default="")
    parser.add_argument("--dinov3-backbone-weights", default="")
    parser.add_argument("--dinov3-segmentor-weights", default="")
    parser.add_argument("--dinov3-hub-entry", default="dinov3_vit7b16_ms")
    parser.add_argument("--dinov3-crop-size", default=896, type=int)
    parser.add_argument("--dinov3-stride", default=448, type=int)
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)
    backend = build_backend(args.backend, args)

    cameras = load_cameras(args.model_path)
    view_count = args.count if args.count > 0 else len(cameras)
    camera_items = selected_camera_items(cameras, args.camera_indices, view_count)

    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(args.white_background)

    seg_dir = args.output_dir / "seg"
    overlay_dir = args.output_dir / "overlays"
    rgb_dir = args.output_dir / "rgb_renders"
    for directory in (seg_dir, overlay_dir, rgb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    started = time.time()
    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(camera_items):
            camera = make_camera(camera_json, modules, args.max_width)
            render_pkg = render_flashsplat(camera, gaussians, modules, pipeline, background)
            rgb = tensor_to_rgb_array(render_pkg["render"])
            del render_pkg

            filename = camera_filename(output_index, camera_json)
            Image.fromarray(rgb).save(rgb_dir / filename)

            ade_class, confidence = backend.segment(rgb)
            project_class = ontology.remap(ade_class).astype(np.uint8)

            seg_file = f"{Path(filename).stem}.npz"
            arrays: dict[str, np.ndarray] = {
                "project_class": project_class,
                "confidence": confidence.astype(np.float16),
            }
            if args.save_raw:
                arrays["ade20k_class"] = ade_class.astype(np.int16)
            np.savez_compressed(seg_dir / seg_file, **arrays)

            overlay_image(rgb, project_class, ontology, args.overlay_alpha).save(
                overlay_dir / filename
            )

            histogram = frame_histogram(project_class, ontology)
            frames.append(
                {
                    "file": filename,
                    "seg_file": seg_file,
                    "camera_index": camera_index,
                    "camera_id": int(camera_json["id"]),
                    "image_name": camera_json.get("img_name", ""),
                    "width": int(project_class.shape[1]),
                    "height": int(project_class.shape[0]),
                    "mean_confidence": float(np.asarray(confidence, dtype=np.float32).mean()),
                    "labeled_pixel_ratio": float((project_class > 0).mean()),
                    "class_histogram": histogram,
                }
            )
            print(
                f"view {output_index:04d} cam={camera_index} "
                f"classes={len(histogram) - int('ignore' in histogram)} "
                f"labeled={frames[-1]['labeled_pixel_ratio']:.3f} "
                f"conf={frames[-1]['mean_confidence']:.3f}"
            )
            torch.cuda.empty_cache()

    manifest = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "camera_indices": args.camera_indices,
        "view_count": len(frames),
        "backend": backend.describe(),
        "ontology": ontology.describe(),
        "save_raw": bool(args.save_raw),
        "elapsed_seconds": time.time() - started,
        "frames": frames,
    }
    manifest_path = args.output_dir / "dense_seg_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
