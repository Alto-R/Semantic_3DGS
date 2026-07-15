#!/usr/bin/env python3
"""Run the official DINOv2 ViT-L/14 ADE20K linear head on rendered views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dinov2_ontology import load_ontology


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def strip_state_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("DINOv2 checkpoint must contain a state dictionary")
    for key in ("model", "teacher", "state_dict"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            raw = nested
            break
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def build_segmenter(
    dinov2_root: Path,
    backbone_checkpoint: Path,
    head_config: Path,
    head_checkpoint: Path,
    device: str,
) -> Any:
    sys.path.insert(0, str(dinov2_root.resolve()))
    import torch
    import torch.nn.functional as functional
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmseg.apis import init_segmentor

    # Import registers DINOv2's segmentation heads with mmseg's registries.
    import dinov2.eval.segmentation.models  # noqa: F401

    class CenterPadding(torch.nn.Module):
        def __init__(self, multiple: int) -> None:
            super().__init__()
            self.multiple = multiple

        def forward(self, value: Any) -> Any:
            shape = value.shape[-2:]
            new_shape = tuple(
                ((dimension + self.multiple - 1) // self.multiple) * self.multiple
                for dimension in shape
            )
            pads: list[int] = []
            for dimension, target in zip(reversed(shape), reversed(new_shape)):
                total = target - dimension
                pads.extend([total // 2, total - total // 2])
            return functional.pad(value, pads)

    cfg = Config.fromfile(str(head_config))
    backbone = torch.hub.load(
        str(dinov2_root.resolve()),
        "dinov2_vitl14",
        source="local",
        pretrained=False,
    )
    raw_state = torch.load(str(backbone_checkpoint), map_location="cpu")
    backbone.load_state_dict(strip_state_dict(raw_state), strict=True)
    backbone.eval().to(device)

    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone, "patch_size"):
        padding = CenterPadding(int(backbone.patch_size))

        def center_pad(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            return (padding(inputs[0]), *inputs[1:])

        model.backbone.register_forward_pre_hook(center_pad)
    model.init_weights()
    load_checkpoint(model, str(head_checkpoint), map_location="cpu")
    model.eval().to(device)
    model.cfg = cfg
    return model


def inference_probabilities(model: Any, image_path: Path) -> np.ndarray:
    """Return CxHxW probabilities using mmseg 0.27's configured test pipeline."""

    import torch
    from mmcv.parallel import collate, scatter
    from mmseg.datasets.pipelines import Compose
    from mmseg.datasets.pipelines import LoadImageFromFile

    class LoadImage:
        def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
            if isinstance(results["img"], np.ndarray):
                results = results.copy()
                results["filename"] = None
                results["ori_filename"] = None
                results["img"] = results["img"]
                results["img_shape"] = results["img"].shape
                results["ori_shape"] = results["img"].shape
                return results
            loader = LoadImageFromFile()
            return loader({"img_info": {"filename": str(results["img"])}, "img_prefix": None})

    pipeline = Compose([LoadImage()] + model.cfg.data.test.pipeline[1:])
    data = pipeline({"img": str(image_path)})
    data = collate([data], samples_per_gpu=1)
    device = next(model.parameters()).device
    if device.type == "cuda":
        data = scatter(data, [device])[0]
    else:
        data["img_metas"] = [item.data[0] for item in data["img_metas"]]

    images = data["img"] if isinstance(data["img"], list) else [data["img"]]
    metas = data["img_metas"]
    if metas and isinstance(metas[0], dict):
        metas = [metas]
    probabilities = []
    with torch.no_grad():
        for tensor, image_metas in zip(images, metas):
            probabilities.append(model.inference(tensor, image_metas, rescale=True)[0])
    if not probabilities:
        raise RuntimeError(f"No segmentation output for {image_path}")
    stacked = torch.stack(probabilities, dim=0).mean(dim=0)
    return stacked.detach().float().cpu().numpy()


def class_color(class_id: int) -> np.ndarray:
    if class_id <= 0:
        return np.zeros((3,), dtype=np.uint8)
    return np.asarray(
        [
            (37 * class_id + 53) % 256,
            (97 * class_id + 101) % 256,
            (17 * class_id + 199) % 256,
        ],
        dtype=np.uint8,
    )


def save_overlay(
    rgb_path: Path,
    project_ids: np.ndarray,
    output_path: Path,
    alpha: float,
) -> None:
    base = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    colors = np.zeros_like(base)
    for class_id in np.unique(project_ids):
        if class_id > 0:
            colors[project_ids == class_id] = class_color(int(class_id))
    mask = project_ids > 0
    overlay = base.copy()
    overlay[mask] = (
        (1.0 - alpha) * base[mask].astype(np.float32)
        + alpha * colors[mask].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--dinov2-root", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--head-config", required=True, type=Path)
    parser.add_argument("--head-checkpoint", required=True, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--min-pixel-confidence", default=0.5, type=float)
    parser.add_argument("--overlay-alpha", default=0.55, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.min_pixel_confidence <= 1.0:
        raise ValueError("min-pixel-confidence must be between zero and one")

    view_manifest_path = args.input_dir / "view_manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    ontology = load_ontology(args.ontology)
    lookup = ontology.ade_to_project
    segment_dir = args.input_dir / "dinov2_segments"
    overlay_dir = args.input_dir / "dinov2_overlays"
    output_manifest_path = args.input_dir / "dinov2_manifest.json"
    if output_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_manifest_path} exists; pass --overwrite to replace it"
        )
    segment_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    model = build_segmenter(
        args.dinov2_root,
        args.backbone_checkpoint,
        args.head_config,
        args.head_checkpoint,
        args.device,
    )
    frames: list[dict[str, Any]] = []
    for frame in view_manifest["frames"]:
        filename = str(frame["file"])
        rgb_path = args.input_dir / "rgb_renders" / filename
        stem = Path(filename).stem
        segment_path = segment_dir / f"{stem}.npz"
        overlay_path = overlay_dir / filename
        if (segment_path.exists() or overlay_path.exists()) and not args.overwrite:
            raise FileExistsError(f"Outputs for {filename} exist; pass --overwrite")

        probabilities = inference_probabilities(model, rgb_path)
        if probabilities.ndim != 3 or probabilities.shape[0] != ontology.class_count:
            raise ValueError(
                f"Expected {ontology.class_count} ADE20K probability channels for "
                f"{filename}; got {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite DINOv2 probabilities for {filename}")
        raw_class = np.argmax(probabilities, axis=0).astype(np.uint8)
        confidence = np.max(probabilities, axis=0).astype(np.float16)
        confidence_for_threshold = confidence.astype(np.float32)
        project_ids = lookup[raw_class]
        project_ids[confidence_for_threshold < args.min_pixel_confidence] = 0
        np.savez_compressed(
            segment_path,
            class_id=raw_class,
            confidence=confidence,
        )
        save_overlay(rgb_path, project_ids, overlay_path, args.overlay_alpha)
        frames.append(
            {
                **frame,
                "segment_file": segment_path.relative_to(args.input_dir).as_posix(),
                "overlay_file": overlay_path.relative_to(args.input_dir).as_posix(),
                "mean_confidence": float(confidence_for_threshold.mean()),
                "min_confidence": float(confidence_for_threshold.min()),
                "max_confidence": float(confidence_for_threshold.max()),
                "abstain_pixel_ratio": float(np.mean(project_ids == 0)),
            }
        )
        print(f"segmented {filename}")

    output_manifest = {
        "source": "dinov2_vitl14_ade20k_linear",
        "view_manifest": str(view_manifest_path),
        "model": {
            "backbone": "dinov2_vitl14",
            "head": "ade20k_linear",
            "backbone_checkpoint": str(args.backbone_checkpoint),
            "backbone_sha256": sha256_file(args.backbone_checkpoint),
            "head_config": str(args.head_config),
            "head_config_sha256": sha256_file(args.head_config),
            "head_checkpoint": str(args.head_checkpoint),
            "head_sha256": sha256_file(args.head_checkpoint),
        },
        "ontology": str(args.ontology),
        "ontology_sha256": sha256_file(args.ontology),
        "min_pixel_confidence": args.min_pixel_confidence,
        "confidence_storage": "float16_max_softmax_probability",
        "raw_class_storage": "uint8_ade20k_zero_based",
        "camera_count": len(frames),
        "frames": frames,
    }
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(f"wrote {output_manifest_path}")


if __name__ == "__main__":
    main()
