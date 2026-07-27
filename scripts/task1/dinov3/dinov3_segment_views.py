#!/usr/bin/env python3
"""Run the official DINOv3 ViT-7B ADE20K Mask2Former head on rendered views."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.task1.common.semantic_palette import rgb8_for_class
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.query_regions import (
    QueryRegionThresholds,
    class_agnostic_query_regions,
    compact_query_evidence,
    describe_query_regions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"
DEFAULT_HUB_ENTRY = "dinov3_vit7b16_ms"
PINNED_DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
EXPECTED_BACKBONE_SHA256 = (
    "a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7"
)
EXPECTED_SEGMENTOR_SHA256 = (
    "bf307cb1c2fd95046feb1bf9a8a13dae60a746bddd8f5297134da95525dbcb42"
)
ADE20K_CLASS_COUNT = 150
CONFIDENCE_METRIC = "relative_top1_top2_margin"
CONFIDENCE_FORMULA = (
    "(top1_probability - top2_probability) / max(top1_probability, epsilon)"
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_expected_sha256(path: Path, expected: str) -> str:
    normalized = expected.strip().lower()
    if normalized and not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"Expected SHA256 for {path} must contain 64 hexadecimal characters")
    actual = sha256_file(path)
    if normalized and actual != normalized:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {normalized}, got {actual}"
        )
    return actual


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_provenance(
    repository: Path,
    expected_commit: str,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if not (repository / ".git").exists():
        raise ValueError(f"DINOv3 repository is not a Git checkout: {repository}")
    commit = git_output(repository, "rev-parse", "HEAD").lower()
    normalized_expected = expected_commit.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized_expected):
        raise ValueError("expected-repo-commit must be a full 40-character Git commit")
    if commit != normalized_expected:
        raise ValueError(
            f"DINOv3 repository commit mismatch: expected {normalized_expected}, got {commit}"
        )
    tracked_changes = git_output(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes and not allow_dirty:
        raise ValueError(
            "DINOv3 repository has tracked local modifications; "
            "pass --allow-dirty-repo only for an explicitly reviewed checkout"
        )
    try:
        remote = git_output(repository, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        remote = ""
    return {
        "path": str(repository),
        "commit": commit,
        "expected_commit": normalized_expected,
        "tracked_worktree_clean": not bool(tracked_changes),
        "origin": remote,
    }


def numeric_version(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise ValueError(f"Cannot parse version: {value}")
    return tuple(int(part or 0) for part in match.groups())


def mask2former_confidence_maps(
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return scale-independent confidence and calibration diagnostics."""

    if probabilities.ndim != 3 or probabilities.shape[0] < 2:
        raise ValueError(
            "probabilities must have shape CxHxW with at least two classes"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if probabilities.min() < 0.0:
        raise ValueError("probabilities must be non-negative")

    class_count = probabilities.shape[0]
    top_two = np.partition(
        probabilities,
        class_count - 2,
        axis=0,
    )[-2:]
    top1 = np.maximum(top_two[0], top_two[1]).astype(np.float32, copy=False)
    top2 = np.minimum(top_two[0], top_two[1]).astype(np.float32, copy=False)
    absolute_margin = np.maximum(top1 - top2, 0.0)
    relative_margin = np.divide(
        absolute_margin,
        np.maximum(top1, np.finfo(np.float32).eps),
    )
    relative_margin = np.clip(relative_margin, 0.0, 1.0)

    safe_probabilities = np.maximum(
        probabilities.astype(np.float32, copy=False),
        np.finfo(np.float32).tiny,
    )
    entropy = -np.sum(
        safe_probabilities * np.log(safe_probabilities),
        axis=0,
        dtype=np.float32,
    )
    normalized_entropy_confidence = np.clip(
        1.0 - entropy / np.log(float(class_count)),
        0.0,
        1.0,
    )
    return {
        "confidence": relative_margin.astype(np.float32, copy=False),
        "max_softmax_probability": top1,
        "top1_top2_margin": absolute_margin.astype(np.float32, copy=False),
        "normalized_entropy_confidence": normalized_entropy_confidence.astype(
            np.float32,
            copy=False,
        ),
    }


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    quantiles = np.quantile(
        values,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"),
                quantiles,
                strict=True,
            )
        },
    }


def validate_runtime(torch: Any, device: str, precision: str) -> None:
    if numeric_version(str(torch.__version__)) < (2, 7, 1):
        raise RuntimeError(
            f"DINOv3 evaluation requires PyTorch >= 2.7.1; found {torch.__version__}"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if precision == "bfloat16" and device.startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 inference was requested but the selected CUDA device "
                "does not report native bfloat16 support"
            )


def build_segmenter(
    dinov3_root: Path,
    backbone_checkpoint: Path,
    segmentor_checkpoint: Path,
    hub_entry: str,
    device: str,
    precision: str,
) -> tuple[Any, Any]:
    sys.path.insert(0, str(dinov3_root.resolve()))
    import torch

    validate_runtime(torch, device, precision)
    autocast_dtype = (
        torch.bfloat16 if precision == "bfloat16" else torch.float32
    )
    model = torch.hub.load(
        str(dinov3_root.resolve()),
        hub_entry,
        source="local",
        pretrained=True,
        weights=str(segmentor_checkpoint.resolve()),
        backbone_weights=str(backbone_checkpoint.resolve()),
        autocast_dtype=autocast_dtype,
        check_hash=True,
    )
    model.eval().to(device)
    return model, torch


def normalized_image_tensor(
    torch: Any,
    rgb_path: Path,
    device: str,
) -> tuple[Any, int, int]:
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8, copy=True)
    height, width = rgb.shape[:2]
    image = (
        torch.from_numpy(rgb)
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        / 255.0
    )
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device,
        dtype=torch.float32,
    ).view(3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device,
        dtype=torch.float32,
    ).view(3, 1, 1)
    return ((image - mean) / std).unsqueeze(0), height, width


def inference_probabilities(
    model: Any,
    torch: Any,
    rgb_path: Path,
    device: str,
    precision: str,
    crop_size: int,
    stride: int,
) -> np.ndarray:
    """Return CxHxW ADE20K probabilities using official sliding inference."""

    from dinov3.eval.segmentation.inference import make_inference

    image, height, width = normalized_image_tensor(
        torch,
        rgb_path,
        device,
    )
    amp_enabled = precision == "bfloat16" and device.startswith("cuda")
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            probabilities = make_inference(
                image,
                model,
                inference_mode="slide",
                decoder_head_type="m2f",
                rescale_to=(height, width),
                n_output_channels=ADE20K_CLASS_COUNT,
                crop_size=(crop_size, crop_size),
                stride=(stride, stride),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
    if probabilities.ndim != 4 or tuple(probabilities.shape[:2]) != (
        1,
        ADE20K_CLASS_COUNT,
    ):
        raise ValueError(
            f"Expected DINOv3 probabilities shaped (1, 150, H, W); "
            f"got {tuple(probabilities.shape)}"
        )
    return probabilities[0].detach().float().cpu().numpy()


def inference_query_regions(
    model: Any,
    torch: Any,
    rgb_path: Path,
    device: str,
    precision: str,
    thresholds: QueryRegionThresholds,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    """Export boundaries plus full query evidence in a whole-image pass."""

    image, height, width = normalized_image_tensor(
        torch,
        rgb_path,
        device,
    )
    amp_enabled = precision == "bfloat16" and device.startswith("cuda")
    try:
        class_embed = model.segmentation_model[1].predictor.class_embed
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError(
            "Pinned DINOv3 Mask2Former model lacks the expected query class head"
        ) from error

    captured: dict[str, Any] = {}

    def capture_query_embedding(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1:
            raise ValueError("Mask2Former class head received unexpected inputs")
        captured["query_embeddings"] = inputs[0].detach()

    hook = class_embed.register_forward_pre_hook(capture_query_embedding)
    try:
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                prediction = model.predict(
                    torch.nn.functional.interpolate(
                        image,
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    rescale_to=(height, width),
                )
    finally:
        hook.remove()
    if not isinstance(prediction, dict):
        raise ValueError("DINOv3 Mask2Former predict() did not return a dictionary")
    if "pred_logits" not in prediction or "pred_masks" not in prediction:
        raise ValueError("DINOv3 Mask2Former prediction lacks raw query outputs")
    class_logits = prediction["pred_logits"]
    mask_logits = prediction["pred_masks"]
    if class_logits.ndim != 3 or mask_logits.ndim != 4:
        raise ValueError(
            "Expected batched DINOv3 query logits and masks; "
            f"got {tuple(class_logits.shape)} and {tuple(mask_logits.shape)}"
        )
    query_embeddings = captured.get("query_embeddings")
    if query_embeddings is None:
        raise ValueError("DINOv3 Mask2Former query embeddings were not captured")
    if query_embeddings.ndim != 3:
        raise ValueError(
            "Expected batched DINOv3 query embeddings; "
            f"got {tuple(query_embeddings.shape)}"
        )

    logits_numpy = class_logits[0].detach().float().cpu().numpy()
    region_id, region_confidence, regions = class_agnostic_query_regions(
        logits_numpy,
        mask_logits[0].detach().float().sigmoid().cpu().numpy(),
        thresholds,
    )
    evidence = compact_query_evidence(
        logits_numpy,
        regions,
        query_embeddings[0].float().cpu().numpy(),
    )
    return region_id, region_confidence, regions, evidence


def save_overlay(
    rgb_path: Path,
    project_ids: np.ndarray,
    output_path: Path,
    alpha: float,
    project_class_names: dict[int, str],
) -> None:
    base = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    colors = np.zeros_like(base)
    for class_id in np.unique(project_ids):
        if class_id > 0:
            class_name = project_class_names.get(
                int(class_id),
                f"project_class_{int(class_id)}",
            )
            colors[project_ids == class_id] = np.asarray(
                rgb8_for_class(class_name),
                dtype=np.uint8,
            )
    mask = project_ids > 0
    overlay = base.copy()
    overlay[mask] = (
        (1.0 - alpha) * base[mask].astype(np.float32)
        + alpha * colors[mask].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def save_region_overlay(
    rgb_path: Path,
    region_id: np.ndarray,
    output_path: Path,
    alpha: float,
) -> None:
    base = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    colors = np.zeros_like(base)
    for compact_id in np.unique(region_id):
        if compact_id == 0:
            continue
        compact_id = int(compact_id)
        color = np.asarray(
            [
                48 + (compact_id * 67) % 176,
                48 + (compact_id * 101) % 176,
                48 + (compact_id * 149) % 176,
            ],
            dtype=np.uint8,
        )
        colors[region_id == compact_id] = color
    selected = region_id > 0
    overlay = base.copy()
    overlay[selected] = (
        (1.0 - alpha) * base[selected].astype(np.float32)
        + alpha * colors[selected].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--dinov3-root", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--segmentor-checkpoint", required=True, type=Path)
    parser.add_argument("--hub-entry", default=DEFAULT_HUB_ENTRY)
    parser.add_argument(
        "--expected-repo-commit",
        default=PINNED_DINOV3_COMMIT,
    )
    parser.add_argument(
        "--expected-backbone-sha256",
        default=EXPECTED_BACKBONE_SHA256,
    )
    parser.add_argument(
        "--expected-segmentor-sha256",
        default=EXPECTED_SEGMENTOR_SHA256,
    )
    parser.add_argument("--allow-dirty-repo", action="store_true")
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--min-pixel-confidence", default=0.0, type=float)
    parser.add_argument("--overlay-alpha", default=0.55, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--crop-size", default=896, type=int)
    parser.add_argument("--stride", default=596, type=int)
    parser.add_argument("--save-regions", action="store_true")
    parser.add_argument("--region-min-objectness", default=0.30, type=float)
    parser.add_argument("--region-mask-threshold", default=0.50, type=float)
    parser.add_argument("--region-min-pixel-score", default=0.25, type=float)
    parser.add_argument("--region-min-area", default=100, type=int)
    parser.add_argument("--region-max-area-ratio", default=0.80, type=float)
    parser.add_argument("--region-max-queries", default=64, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.min_pixel_confidence <= 1.0:
        raise ValueError("min-pixel-confidence must be between zero and one")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be between zero and one")
    if args.crop_size <= 0 or args.stride <= 0:
        raise ValueError("crop-size and stride must be positive")
    if args.stride > args.crop_size:
        raise ValueError("stride must not exceed crop-size")
    region_thresholds = QueryRegionThresholds(
        min_objectness=args.region_min_objectness,
        mask_threshold=args.region_mask_threshold,
        min_pixel_score=args.region_min_pixel_score,
        min_area=args.region_min_area,
        max_area_ratio=args.region_max_area_ratio,
        max_queries=args.region_max_queries,
    )
    region_thresholds.validate()
    for path in (
        args.input_dir / "view_manifest.json",
        args.backbone_checkpoint,
        args.segmentor_checkpoint,
        args.ontology,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    repo_provenance = repository_provenance(
        args.dinov3_root,
        args.expected_repo_commit,
        allow_dirty=args.allow_dirty_repo,
    )
    backbone_sha256 = verify_expected_sha256(
        args.backbone_checkpoint,
        args.expected_backbone_sha256,
    )
    segmentor_sha256 = verify_expected_sha256(
        args.segmentor_checkpoint,
        args.expected_segmentor_sha256,
    )

    view_manifest_path = args.input_dir / "view_manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    ontology = load_ontology(args.ontology)
    if ontology.class_count != ADE20K_CLASS_COUNT:
        raise ValueError(
            f"Expected {ADE20K_CLASS_COUNT} ADE20K ontology classes; "
            f"got {ontology.class_count}"
        )
    lookup = ontology.ade_to_project
    project_class_names = {
        item.project_id: item.project_class
        for item in ontology.classes
    }
    segment_dir = args.input_dir / "dinov3_segments"
    overlay_dir = args.input_dir / "dinov3_overlays"
    region_dir = args.input_dir / "dinov3_regions"
    region_overlay_dir = args.input_dir / "dinov3_region_overlays"
    output_manifest_path = args.input_dir / "dinov3_manifest.json"
    if output_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_manifest_path} exists; pass --overwrite to replace it"
        )
    segment_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_regions:
        region_dir.mkdir(parents=True, exist_ok=True)
        region_overlay_dir.mkdir(parents=True, exist_ok=True)

    model, torch = build_segmenter(
        args.dinov3_root,
        args.backbone_checkpoint,
        args.segmentor_checkpoint,
        args.hub_entry,
        args.device,
        args.precision,
    )
    frames: list[dict[str, Any]] = []
    for frame in view_manifest["frames"]:
        filename = str(frame["file"])
        rgb_path = args.input_dir / "rgb_renders" / filename
        stem = Path(filename).stem
        segment_path = segment_dir / f"{stem}.npz"
        overlay_path = overlay_dir / filename
        region_path = region_dir / f"{stem}.npz"
        region_overlay_path = region_overlay_dir / filename
        expected_outputs = [segment_path, overlay_path]
        if args.save_regions:
            expected_outputs.extend([region_path, region_overlay_path])
        if any(path.exists() for path in expected_outputs) and not args.overwrite:
            raise FileExistsError(f"Outputs for {filename} exist; pass --overwrite")

        probabilities = inference_probabilities(
            model,
            torch,
            rgb_path,
            args.device,
            args.precision,
            args.crop_size,
            args.stride,
        )
        expected_shape = (
            ADE20K_CLASS_COUNT,
            int(frame["render_height"]),
            int(frame["render_width"]),
        )
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"Expected probabilities shaped {expected_shape} for {filename}; "
                f"got {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite DINOv3 probabilities for {filename}")
        raw_class = np.argmax(probabilities, axis=0).astype(np.uint8)
        confidence_maps = mask2former_confidence_maps(probabilities)
        confidence_for_threshold = confidence_maps["confidence"]
        project_ids = lookup[raw_class]
        project_ids[confidence_for_threshold < args.min_pixel_confidence] = 0
        np.savez_compressed(
            segment_path,
            class_id=raw_class,
            confidence=confidence_maps["confidence"].astype(np.float16),
            max_softmax_probability=confidence_maps[
                "max_softmax_probability"
            ].astype(np.float16),
            top1_top2_margin=confidence_maps["top1_top2_margin"].astype(
                np.float16
            ),
            normalized_entropy_confidence=confidence_maps[
                "normalized_entropy_confidence"
            ].astype(np.float16),
        )
        save_overlay(
            rgb_path,
            project_ids,
            overlay_path,
            args.overlay_alpha,
            project_class_names,
        )
        frame_record = {
            **frame,
            "segment_file": segment_path.relative_to(args.input_dir).as_posix(),
            "overlay_file": overlay_path.relative_to(args.input_dir).as_posix(),
            "mean_confidence": float(confidence_for_threshold.mean()),
            "min_confidence": float(confidence_for_threshold.min()),
            "max_confidence": float(confidence_for_threshold.max()),
            "confidence_distribution": distribution_summary(
                confidence_for_threshold
            ),
            "max_softmax_probability_distribution": distribution_summary(
                confidence_maps["max_softmax_probability"]
            ),
            "top1_top2_margin_distribution": distribution_summary(
                confidence_maps["top1_top2_margin"]
            ),
            "normalized_entropy_confidence_distribution": (
                distribution_summary(
                    confidence_maps["normalized_entropy_confidence"]
                )
            ),
            "abstain_pixel_ratio": float(np.mean(project_ids == 0)),
        }
        if args.save_regions:
            (
                region_id,
                region_confidence,
                regions,
                query_evidence,
            ) = inference_query_regions(
                model,
                torch,
                rgb_path,
                args.device,
                args.precision,
                region_thresholds,
            )
            np.savez_compressed(
                region_path,
                region_id=region_id,
                region_confidence=region_confidence.astype(np.float16),
                query_indices=query_evidence["query_indices"],
                class_probabilities=query_evidence[
                    "class_probabilities"
                ].astype(np.float16),
                no_object_probabilities=query_evidence[
                    "no_object_probabilities"
                ].astype(np.float16),
                query_embeddings=query_evidence["query_embeddings"].astype(
                    np.float16
                ),
            )
            save_region_overlay(
                rgb_path,
                region_id,
                region_overlay_path,
                args.overlay_alpha,
            )
            frame_record.update(
                {
                    "region_file": region_path.relative_to(
                        args.input_dir
                    ).as_posix(),
                    "region_overlay_file": region_overlay_path.relative_to(
                        args.input_dir
                    ).as_posix(),
                    "region_count": len(regions),
                    "regions": regions,
                }
            )
        frames.append(frame_record)
        print(f"segmented {filename}")

    output_manifest = {
        "source": "dinov3_vit7b16_ade20k_mask2former",
        "contract": "raw_ade20k_class_and_relative_margin_confidence_v2",
        "view_manifest": str(view_manifest_path),
        "model": {
            "backbone": "dinov3_vit7b16",
            "head": "ade20k_mask2former",
            "hub_entry": args.hub_entry,
            "repository": repo_provenance,
            "backbone_checkpoint": str(args.backbone_checkpoint),
            "backbone_sha256": backbone_sha256,
            "backbone_size_bytes": args.backbone_checkpoint.stat().st_size,
            "segmentor_checkpoint": str(args.segmentor_checkpoint),
            "segmentor_sha256": segmentor_sha256,
            "segmentor_size_bytes": args.segmentor_checkpoint.stat().st_size,
            "precision": args.precision,
            "crop_size": args.crop_size,
            "stride": args.stride,
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name(args.device)
                if args.device.startswith("cuda")
                else args.device
            ),
            "native_bfloat16": (
                bool(torch.cuda.is_bf16_supported())
                if args.device.startswith("cuda")
                else False
            ),
        },
        "ontology": str(args.ontology),
        "ontology_sha256": sha256_file(args.ontology),
        "min_pixel_confidence": args.min_pixel_confidence,
        "confidence_metric": CONFIDENCE_METRIC,
        "confidence_formula": CONFIDENCE_FORMULA,
        "confidence_storage": "float16_relative_top1_top2_margin",
        "diagnostic_storages": {
            "max_softmax_probability": "float16",
            "top1_top2_margin": "float16",
            "normalized_entropy_confidence": "float16_one_minus_normalized_entropy",
        },
        "raw_class_storage": "uint8_ade20k_zero_based",
        "class_agnostic_query_regions": (
            describe_query_regions(region_thresholds)
            if args.save_regions
            else {
                "available": False,
                "semantic_class_used_for_identity": False,
            }
        ),
        "query_evidence_storage": (
            {
                "available": True,
                "class_probabilities": (
                    "float16_Rx150_conditional_on_object"
                ),
                "no_object_probabilities": "float16_R",
                "query_embeddings": "float16_Rx2048_l2_normalized",
                "query_indices": "int16_R",
                "semantic_identity_hardened_per_view": False,
            }
            if args.save_regions
            else {"available": False}
        ),
        "camera_count": len(frames),
        "frames": frames,
    }
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_manifest_path}")


if __name__ == "__main__":
    main()
