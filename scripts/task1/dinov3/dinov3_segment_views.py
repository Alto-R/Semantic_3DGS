#!/usr/bin/env python3
"""Run the official DINOv3 ViT-7B ADE20K Mask2Former head on rendered views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

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
GIB_BYTES = 1024**3
DINO_ADAPTER_SPATIAL_ALIGNMENT = 32


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


def resolve_cuda_device_index(torch: Any, device: str) -> int:
    parsed_device = torch.device(device)
    if parsed_device.type != "cuda":
        raise ValueError("Expected a CUDA device")
    if parsed_device.index is None:
        return int(torch.cuda.current_device())
    return int(parsed_device.index)


def configure_cuda_memory_limit(
    torch: Any,
    device: str,
    max_cuda_memory_gib: float | None,
) -> dict[str, Any]:
    if max_cuda_memory_gib is None:
        return {
            "enabled": False,
            "allocator": "pytorch_cuda_caching_allocator",
        }
    if not device.startswith("cuda"):
        raise ValueError("max-cuda-memory-gib requires a CUDA device")
    if not math.isfinite(max_cuda_memory_gib) or max_cuda_memory_gib <= 0.0:
        raise ValueError("max-cuda-memory-gib must be a finite positive number")

    device_index = resolve_cuda_device_index(torch, device)
    total_memory = int(
        torch.cuda.get_device_properties(device_index).total_memory
    )
    requested_bytes = int(max_cuda_memory_gib * GIB_BYTES)
    if requested_bytes >= total_memory:
        raise ValueError(
            "max-cuda-memory-gib must be smaller than total visible GPU memory: "
            f"requested {requested_bytes} bytes, total {total_memory} bytes"
        )
    fraction = requested_bytes / total_memory
    torch.cuda.set_per_process_memory_fraction(
        fraction,
        device=device_index,
    )
    free_memory, reported_total = torch.cuda.mem_get_info(device_index)
    record = {
        "enabled": True,
        "allocator": "pytorch_cuda_caching_allocator",
        "logical_device_index": device_index,
        "requested_gib": float(max_cuda_memory_gib),
        "requested_bytes": requested_bytes,
        "total_device_memory_bytes": total_memory,
        "allocator_fraction": fraction,
        "free_memory_bytes_before_model": int(free_memory),
        "reported_total_memory_bytes": int(reported_total),
        "hard_hardware_partition": False,
    }
    print(
        "configured PyTorch CUDA allocator limit: "
        f"{max_cuda_memory_gib:.3f} GiB "
        f"({fraction:.6f} of {total_memory / GIB_BYTES:.3f} GiB)",
        flush=True,
    )
    return record


def cuda_memory_usage(torch: Any, device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {"available": False}
    device_index = resolve_cuda_device_index(torch, device)
    torch.cuda.synchronize(device_index)
    record = {
        "available": True,
        "logical_device_index": device_index,
        "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device_index)
        ),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device_index)
        ),
    }
    record.update(
        {
            f"{name.removesuffix('_bytes')}_gib": value / GIB_BYTES
            for name, value in tuple(record.items())
            if name.endswith("_bytes")
        }
    )
    print(
        "observed DINOv3 CUDA memory: "
        f"peak allocated {record['peak_allocated_gib']:.3f} GiB, "
        f"peak reserved {record['peak_reserved_gib']:.3f} GiB",
        flush=True,
    )
    return record


@contextmanager
def checkpoint_state_dict_loader(
    torch: Any,
    mode: str,
    checkpoint_paths: tuple[Path, ...],
    *,
    integrity_preverified: bool,
):
    record: dict[str, Any] = {
        "mode": mode,
        "mmap": mode == "local_mmap",
        "weights_only": mode == "local_mmap",
        "integrity_preverified": integrity_preverified,
        "intercepted_checkpoint_paths": [],
        "external_repository_modified": False,
    }
    if mode == "standard":
        yield record
        return
    if mode != "local_mmap":
        raise ValueError(f"Unsupported checkpoint load mode: {mode}")
    if not integrity_preverified:
        raise ValueError(
            "local_mmap requires preverified checkpoint integrity"
        )

    allowed_paths = {
        checkpoint_path.resolve()
        for checkpoint_path in checkpoint_paths
    }
    original_loader = torch.hub.load_state_dict_from_url

    def mmap_local_loader(url: str, *args: Any, **kwargs: Any) -> Any:
        parsed = urlparse(str(url))
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            return original_loader(url, *args, **kwargs)
        checkpoint_path = Path(url2pathname(parsed.path)).resolve()
        if checkpoint_path not in allowed_paths:
            return original_loader(url, *args, **kwargs)
        if args:
            raise ValueError(
                "local_mmap does not accept positional Hub loader arguments"
            )
        supported = {
            "map_location",
            "progress",
            "check_hash",
            "file_name",
            "weights_only",
            "model_dir",
        }
        unsupported = sorted(set(kwargs) - supported)
        if unsupported:
            raise ValueError(
                "Unsupported local mmap Hub loader arguments: "
                f"{unsupported}"
            )
        weights_only = kwargs.get("weights_only", False)
        if weights_only is not True:
            raise ValueError("local_mmap requires weights_only=True")
        intercepted = record["intercepted_checkpoint_paths"]
        assert isinstance(intercepted, list)
        intercepted.append(str(checkpoint_path))
        print(
            f"memory-mapping verified local checkpoint: {checkpoint_path}",
            flush=True,
        )
        return torch.load(
            checkpoint_path,
            map_location=kwargs.get("map_location"),
            weights_only=True,
            mmap=True,
        )

    torch.hub.load_state_dict_from_url = mmap_local_loader
    try:
        yield record
    finally:
        torch.hub.load_state_dict_from_url = original_loader


def build_segmenter(
    dinov3_root: Path,
    backbone_checkpoint: Path,
    segmentor_checkpoint: Path,
    hub_entry: str,
    device: str,
    precision: str,
    max_cuda_memory_gib: float | None = None,
    checkpoint_load_mode: str = "standard",
    checkpoints_preverified: bool = False,
    return_memory_limit: bool = False,
) -> (
    tuple[Any, Any]
    | tuple[Any, Any, dict[str, Any], dict[str, Any]]
):
    sys.path.insert(0, str(dinov3_root.resolve()))
    import torch

    validate_runtime(torch, device, precision)
    cuda_memory_limit = configure_cuda_memory_limit(
        torch,
        device,
        max_cuda_memory_gib,
    )
    autocast_dtype = (
        torch.bfloat16 if precision == "bfloat16" else torch.float32
    )
    with checkpoint_state_dict_loader(
        torch,
        checkpoint_load_mode,
        (backbone_checkpoint, segmentor_checkpoint),
        integrity_preverified=checkpoints_preverified,
    ) as checkpoint_loading:
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
    if checkpoint_load_mode == "local_mmap":
        intercepted = {
            Path(path).resolve()
            for path in checkpoint_loading[
                "intercepted_checkpoint_paths"
            ]
        }
        expected = {
            backbone_checkpoint.resolve(),
            segmentor_checkpoint.resolve(),
        }
        if intercepted != expected:
            raise RuntimeError(
                "local_mmap did not intercept exactly the verified "
                f"checkpoints: expected {sorted(map(str, expected))}, "
                f"got {sorted(map(str, intercepted))}"
            )
    model.eval().to(device)
    if return_memory_limit:
        return model, torch, cuda_memory_limit, checkpoint_loading
    return model, torch


def short_side_resize_dimensions(
    height: int,
    width: int,
    short_side: int,
) -> tuple[int, int]:
    """Match DINOv3's evaluation resize while preserving aspect ratio."""

    if height <= 0 or width <= 0 or short_side <= 0:
        raise ValueError("Image dimensions and short side must be positive")
    if height > width:
        resized_width = short_side
        resized_height = int(short_side * height / width + 0.5)
    else:
        resized_height = short_side
        resized_width = int(short_side * width / height + 0.5)
    return resized_height, resized_width


def validate_sliding_inference_geometry(crop_size: int, stride: int) -> None:
    """Reject crop geometry that is unsafe for the DINOv3 adapter."""

    if crop_size <= 0 or stride <= 0:
        raise ValueError("crop-size and stride must be positive")
    if stride > crop_size:
        raise ValueError("stride must not exceed crop-size")
    if crop_size % DINO_ADAPTER_SPATIAL_ALIGNMENT != 0:
        raise ValueError(
            "crop-size must be divisible by "
            f"{DINO_ADAPTER_SPATIAL_ALIGNMENT} for the DINOv3 spatial adapter"
        )


def normalized_image_tensor(
    torch: Any,
    rgb_path: Path,
    device: str,
    *,
    short_side: int | None = None,
) -> tuple[Any, int, int, int, int]:
    with Image.open(rgb_path) as source:
        rgb_image = source.convert("RGB")
    width, height = rgb_image.size
    inference_height, inference_width = height, width
    if short_side is not None:
        inference_height, inference_width = short_side_resize_dimensions(
            height,
            width,
            short_side,
        )
        if (inference_width, inference_height) != rgb_image.size:
            rgb_image = rgb_image.resize(
                (inference_width, inference_height),
                resample=Image.Resampling.BILINEAR,
            )
    rgb = np.array(rgb_image, dtype=np.uint8, copy=True)
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
    return (
        ((image - mean) / std).unsqueeze(0),
        height,
        width,
        inference_height,
        inference_width,
    )


def inference_probabilities(
    model: Any,
    torch: Any,
    rgb_path: Path,
    device: str,
    precision: str,
    crop_size: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return CxHxW ADE20K probabilities using official sliding inference."""

    from dinov3.eval.segmentation.inference import make_inference

    image, height, width, inference_height, inference_width = normalized_image_tensor(
        torch,
        rgb_path,
        device,
        short_side=crop_size,
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
    preprocessing = {
        "mode": "resize_short_side_to_crop_size_before_sliding",
        "interpolation": "pillow_bilinear",
        "original_height": height,
        "original_width": width,
        "inference_height": inference_height,
        "inference_width": inference_width,
        "crop_size": crop_size,
        "adapter_spatial_alignment": DINO_ADAPTER_SPATIAL_ALIGNMENT,
    }
    return probabilities[0].detach().float().cpu().numpy(), preprocessing


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

    image, height, width, _, _ = normalized_image_tensor(
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
    parser.add_argument(
        "--checkpoint-load-mode",
        choices=("standard", "local_mmap"),
        default="standard",
        help=(
            "Use local_mmap only for already verified local checkpoints "
            "to avoid eager state-dict storage allocation."
        ),
    )
    parser.add_argument(
        "--max-cuda-memory-gib",
        default=None,
        type=float,
        help=(
            "Optional PyTorch CUDA caching-allocator ceiling in GiB. "
            "This is not a hard hardware partition."
        ),
    )
    parser.add_argument("--save-regions", action="store_true")
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help=(
            "Persist the complete 150-class softmax tensor as float16 NPY. "
            "This is opt-in because the cache is substantially larger than "
            "the normal hard-label diagnostic cache."
        ),
    )
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
    validate_sliding_inference_geometry(args.crop_size, args.stride)
    if args.max_cuda_memory_gib is not None and (
        not math.isfinite(args.max_cuda_memory_gib)
        or args.max_cuda_memory_gib <= 0.0
    ):
        raise ValueError("max-cuda-memory-gib must be a finite positive number")
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
    probability_dir = args.input_dir / "dinov3_probabilities"
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
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)

    (
        model,
        torch,
        cuda_memory_limit,
        checkpoint_loading,
    ) = build_segmenter(
        args.dinov3_root,
        args.backbone_checkpoint,
        args.segmentor_checkpoint,
        args.hub_entry,
        args.device,
        args.precision,
        args.max_cuda_memory_gib,
        args.checkpoint_load_mode,
        checkpoints_preverified=True,
        return_memory_limit=True,
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(
            resolve_cuda_device_index(torch, args.device)
        )
    frames: list[dict[str, Any]] = []
    for frame in view_manifest["frames"]:
        filename = str(frame["file"])
        rgb_path = args.input_dir / "rgb_renders" / filename
        stem = Path(filename).stem
        segment_path = segment_dir / f"{stem}.npz"
        overlay_path = overlay_dir / filename
        region_path = region_dir / f"{stem}.npz"
        probability_path = probability_dir / f"{stem}.npy"
        region_overlay_path = region_overlay_dir / filename
        expected_outputs = [segment_path, overlay_path]
        if args.save_regions:
            expected_outputs.extend([region_path, region_overlay_path])
        if args.save_probabilities:
            expected_outputs.append(probability_path)
        if any(path.exists() for path in expected_outputs) and not args.overwrite:
            raise FileExistsError(f"Outputs for {filename} exist; pass --overwrite")

        probabilities, preprocessing = inference_probabilities(
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
        if args.save_probabilities:
            probability_sums = probabilities.sum(axis=0, dtype=np.float32)
            if not np.allclose(probability_sums, 1.0, rtol=2e-4, atol=2e-4):
                raise ValueError(
                    f"DINOv3 probabilities do not sum to one for {filename}"
                )
            stored_probabilities = probabilities.astype(np.float16)
            np.save(probability_path, stored_probabilities, allow_pickle=False)
            stored_argmax = np.argmax(stored_probabilities, axis=0).astype(np.uint8)
            quantized_argmax_disagreement = int(
                np.count_nonzero(stored_argmax != raw_class)
            )
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
            "dinov3_preprocessing": preprocessing,
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
        if args.save_probabilities:
            frame_record.update(
                {
                    "probability_file": probability_path.relative_to(
                        args.input_dir
                    ).as_posix(),
                    "probability_shape": [int(value) for value in probabilities.shape],
                    "probability_storage": "float16_npy_ade20k_class_first",
                    "probability_size_bytes": probability_path.stat().st_size,
                    "float16_argmax_disagreement_count": (
                        quantized_argmax_disagreement
                    ),
                    "float16_argmax_disagreement_ratio": (
                        quantized_argmax_disagreement / raw_class.size
                    ),
                }
            )
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

    memory_usage = cuda_memory_usage(torch, args.device)
    output_manifest = {
        "source": "dinov3_vit7b16_ade20k_mask2former",
        "contract": (
            "raw_ade20k_class_probabilities_and_relative_margin_confidence_v3"
            if args.save_probabilities
            else "raw_ade20k_class_and_relative_margin_confidence_v2"
        ),
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
            "sliding_preprocessing": {
                "mode": "resize_short_side_to_crop_size_before_sliding",
                "interpolation": "pillow_bilinear",
                "adapter_spatial_alignment": DINO_ADAPTER_SPATIAL_ALIGNMENT,
            },
            "checkpoint_loading": checkpoint_loading,
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name(
                    resolve_cuda_device_index(torch, args.device)
                )
                if args.device.startswith("cuda")
                else args.device
            ),
            "native_bfloat16": (
                bool(torch.cuda.is_bf16_supported())
                if args.device.startswith("cuda")
                else False
            ),
            "cuda_memory_limit": cuda_memory_limit,
            "cuda_memory_usage": memory_usage,
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
        "dense_probability_storage": (
            {
                "available": True,
                "dtype": "float16",
                "layout": "ade20k_class_height_width",
                "class_count": ADE20K_CLASS_COUNT,
                "normalization": "float32_softmax_before_float16_storage",
            }
            if args.save_probabilities
            else {"available": False}
        ),
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
