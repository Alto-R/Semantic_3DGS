#!/usr/bin/env python3
"""Four-view 2D pilot: classify automatic SAM masks with DINOv3 dino.txt.

This stage is deliberately isolated from FlashSplat and semantic fusion.  It
answers one question before any 2D evidence is lifted into 3D: does a
competitive open-vocabulary DINOv3 score select the intended object masks and
reject coherent lookalikes?
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.task1.common.flashsplat_cameras import evenly_spaced_indices
from scripts.task1.dinov3.dinov3_segment_views import (
    PINNED_DINOV3_COMMIT,
    checkpoint_state_dict_loader,
    repository_provenance,
    sha256_file,
)


SOURCE = "dinov3_dinotxt_sam_mask_classification_pilot"
CONTRACT = "review_only_dinotxt_sam_mask_pilot_v1"
DEFAULT_HUB_ENTRY = "dinov3_vitl16_dinotxt_tet1280d20h24l"
EXPECTED_VITL_SHA256_PREFIX = "8aa4cbdd"
EXPECTED_DINOTXT_SHA256_PREFIX = "a442d8f5"
EXPECTED_BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# A compact photographic ensemble.  The official notebook uses a larger CLIP
# ensemble; these prompts preserve its essential singular/plural and crop
# variation while keeping a small candidate vocabulary inexpensive.
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a cropped photo of the {}.",
    "a bright photo of a {}.",
    "a dark photo of the {}.",
    "a low resolution photo of a {}.",
)


@dataclass(frozen=True)
class TextClass:
    name: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class TextFeatureEnsemble:
    """Per-prompt text features and the class-level reduction to apply.

    Keeping prompt features separate is important for open-vocabulary labels:
    averaging aliases before comparing classes can dilute the one phrase that
    actually matches the object.  The reduction is applied to the image-side
    logits, where each alias can compete on equal footing with every other
    class.
    """

    features: Any
    prompt_counts: tuple[int, ...]
    aggregation: str
    top_k: int


@dataclass(frozen=True)
class PilotConfig:
    target_class: str
    classes: tuple[TextClass, ...]
    prompt_aggregation: str
    prompt_top_k: int
    min_target_probability: float
    min_competitor_margin: float
    min_target_win_fraction: float
    max_selected_masks: int
    selection_containment_threshold: float
    selection_nms_iou: float


def normalize_class_name(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def load_config(path: Path) -> PilotConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("classes"), list):
        raise ValueError("config must be an object containing a classes list")
    classes: list[TextClass] = []
    for record in payload["classes"]:
        if not isinstance(record, dict):
            raise ValueError("each class entry must be an object")
        name = normalize_class_name(str(record.get("class", "")))
        prompts = record.get("prompts", [])
        if not name or not isinstance(prompts, list):
            raise ValueError("each class requires a name and prompt list")
        cleaned = tuple(dict.fromkeys(str(item).strip().lower() for item in prompts if str(item).strip()))
        if not cleaned:
            raise ValueError(f"class {name} has no prompts")
        classes.append(TextClass(name=name, prompts=cleaned))
    names = [item.name for item in classes]
    if len(names) != len(set(names)):
        raise ValueError("class names must be unique")
    target = normalize_class_name(str(payload.get("target_class", "")))
    if target not in names:
        raise ValueError("target_class must name one configured class")

    selection = payload.get("selection", {})
    config = PilotConfig(
        target_class=target,
        classes=tuple(classes),
        prompt_aggregation=str(
            selection.get("prompt_aggregation", "mean")
        ).strip().lower(),
        prompt_top_k=int(selection.get("prompt_top_k", 2)),
        min_target_probability=float(selection.get("min_target_probability", 0.35)),
        min_competitor_margin=float(selection.get("min_competitor_margin", 0.05)),
        min_target_win_fraction=float(selection.get("min_target_win_fraction", 0.50)),
        max_selected_masks=int(selection.get("max_selected_masks", 8)),
        selection_containment_threshold=float(selection.get("containment_threshold", 0.90)),
        selection_nms_iou=float(selection.get("nms_iou", 0.70)),
    )
    if config.prompt_aggregation not in {"mean", "max", "topk_mean"}:
        raise ValueError(
            "prompt_aggregation must be one of mean, max, or topk_mean"
        )
    if config.prompt_top_k < 1:
        raise ValueError("prompt_top_k must be positive")
    for name in (
        "min_target_probability",
        "min_target_win_fraction",
        "selection_containment_threshold",
        "selection_nms_iou",
    ):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if not -1.0 <= config.min_competitor_margin <= 1.0:
        raise ValueError("min_competitor_margin must be between -1 and one")
    if config.max_selected_masks < 1:
        raise ValueError("max_selected_masks must be positive")
    return config


def select_frames(manifest: dict[str, Any], camera_indices: list[int]) -> list[dict[str, Any]]:
    if len(camera_indices) != len(set(camera_indices)):
        raise ValueError("camera indices must be unique")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("view manifest is missing frames")
    by_index = {int(frame["camera_index"]): frame for frame in frames}
    missing = [index for index in camera_indices if index not in by_index]
    if missing:
        raise ValueError(f"view manifest is missing camera indices: {missing}")
    return [by_index[index] for index in camera_indices]


def select_evenly_spaced_frames(
    manifest: dict[str, Any],
    view_count: int,
) -> list[dict[str, Any]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("view manifest is missing frames")
    positions = evenly_spaced_indices(len(frames), view_count)
    return [frames[position] for position in positions]


def flashsplat_mask_metadata(
    selected: list[dict[str, Any]],
    target_class: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in selected:
        records.append(
            {
                "source": SOURCE,
                "class": target_class,
                "class_name": target_class,
                "phrase": target_class.replace("_", " "),
                "confidence": float(record["target_probability"]),
                "predicted_iou": float(record["sam_predicted_iou"]),
                "stability_score": float(record["sam_stability_score"]),
                "area": int(record["area"]),
                "bbox": [int(value) for value in record["bbox_xywh"]],
                "dinotxt_target_probability": float(record["target_probability"]),
                "dinotxt_target_margin": float(record["target_margin"]),
                "dinotxt_target_win_fraction": float(record["target_win_fraction"]),
                "dinotxt_best_competitor": str(record["best_competitor"]),
            }
        )
    return records


def verify_checkpoint_prefix(path: Path, expected_prefix: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_prefix and not digest.startswith(expected_prefix.lower()):
        raise ValueError(
            f"checkpoint SHA256 mismatch for {path}: expected prefix "
            f"{expected_prefix}, got {digest}"
        )
    return digest


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(left, right).sum())
    return intersection / float(max(union, 1))


def mask_containment(
    inner: np.ndarray,
    outer: np.ndarray,
    *,
    inner_area: int | None = None,
) -> float:
    inner_area = int(inner.sum()) if inner_area is None else int(inner_area)
    if inner_area == 0:
        return 0.0
    intersection = int(np.logical_and(inner, outer).sum())
    return intersection / float(inner_area)


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x_min = int(xs.min())
    y_min = int(ys.min())
    return [x_min, y_min, int(xs.max()) - x_min, int(ys.max()) - y_min]


def bbox_intersection_area(left: list[int], right: list[int]) -> int:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    # SAM stores XYWH boxes from inclusive mask extrema.  Add one to the far
    # edge so this remains a conservative upper bound on pixel intersection.
    width = max(0, min(left_x + left_width + 1, right_x + right_width + 1) - max(left_x, right_x))
    height = max(0, min(left_y + left_height + 1, right_y + right_height + 1) - max(left_y, right_y))
    return width * height


def scale_cosine_logits(cosine: Any, logit_scale: float) -> Any:
    """Apply the learned dino.txt temperature before class softmax."""

    value = float(logit_scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dino.txt logit scale must be finite and positive")
    return cosine * value


def score_sam_masks(
    masks: list[dict[str, Any]],
    probabilities: np.ndarray,
    class_names: list[str],
    target_class: str,
) -> list[dict[str, Any]]:
    if probabilities.ndim != 3 or probabilities.shape[0] != len(class_names):
        raise ValueError("probabilities must have shape CxHxW matching classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    target_id = class_names.index(target_class)
    pixel_winners = np.argmax(probabilities, axis=0)
    records: list[dict[str, Any]] = []
    for mask_id, mask_record in enumerate(masks):
        segmentation = np.asarray(mask_record["segmentation"], dtype=bool)
        if segmentation.shape != probabilities.shape[1:]:
            raise ValueError("SAM mask and dino.txt score map shapes differ")
        area = int(segmentation.sum())
        if area == 0:
            continue
        means = probabilities[:, segmentation].mean(axis=1)
        competitor_ids = [index for index in range(len(class_names)) if index != target_id]
        competitor_id = max(competitor_ids, key=lambda index: float(means[index]))
        target_probability = float(means[target_id])
        competitor_probability = float(means[competitor_id])
        records.append(
            {
                "mask_id": mask_id,
                "area": area,
                "bbox_xywh": [int(value) for value in mask_record["bbox"]],
                "sam_predicted_iou": float(mask_record.get("predicted_iou", 0.0)),
                "sam_stability_score": float(mask_record.get("stability_score", 0.0)),
                "target_class": target_class,
                "target_probability": target_probability,
                "best_competitor": class_names[competitor_id],
                "best_competitor_probability": competitor_probability,
                "target_margin": target_probability - competitor_probability,
                "target_win_fraction": float((pixel_winners[segmentation] == target_id).mean()),
                "class_probabilities": {
                    name: float(means[index]) for index, name in enumerate(class_names)
                },
            }
        )
    records.sort(
        key=lambda item: (
            item["target_margin"],
            item["target_probability"],
            item["target_win_fraction"],
            item["sam_predicted_iou"],
        ),
        reverse=True,
    )
    return records


def select_target_masks(
    masks: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    config: PilotConfig,
) -> list[dict[str, Any]]:
    eligible: list[tuple[dict[str, Any], np.ndarray, int, list[int]]] = []
    for record in scores:
        if record["target_probability"] < config.min_target_probability:
            continue
        if record["target_margin"] < config.min_competitor_margin:
            continue
        if record["target_win_fraction"] < config.min_target_win_fraction:
            continue
        candidate = np.asarray(masks[int(record["mask_id"])]["segmentation"], dtype=bool)
        area = int(candidate.sum())
        bbox = mask_bbox_xywh(candidate)
        eligible.append((record, candidate, area, bbox))

    maximal: list[tuple[dict[str, Any], np.ndarray]] = []
    for record, candidate, candidate_area, candidate_bbox in eligible:
        contained = any(
            other_area > candidate_area
            and bbox_intersection_area(candidate_bbox, other_bbox)
            >= config.selection_containment_threshold * candidate_area
            and mask_containment(candidate, other, inner_area=candidate_area)
            >= config.selection_containment_threshold
            for other_record, other, other_area, other_bbox in eligible
            if other_record is not record
        )
        if not contained:
            maximal.append((record, candidate))

    selected: list[dict[str, Any]] = []
    selected_masks: list[np.ndarray] = []
    for record, candidate in maximal:
        if any(mask_iou(candidate, prior) >= config.selection_nms_iou for prior in selected_masks):
            continue
        selected.append(record)
        selected_masks.append(candidate)
        if len(selected) >= config.max_selected_masks:
            break
    return selected


def short_side_resize(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if min(width, height) == size:
        return image
    scale = size / float(min(width, height))
    dimensions = (int(round(width * scale)), int(round(height * scale)))
    return image.resize(dimensions, Image.Resampling.BICUBIC)


def prepare_image(torch: Any, rgb: np.ndarray, resize: int, device: str) -> Any:
    image = short_side_resize(Image.fromarray(rgb, mode="RGB"), resize)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (tensor - mean) / std


def prompt_texts(text_class: TextClass) -> list[str]:
    """Expand one class's aliases into the shared photographic template set."""

    return [
        template.format(prompt)
        for prompt in text_class.prompts
        for template in PROMPT_TEMPLATES
    ]


def encode_text_classes(
    model: Any,
    tokenizer: Any,
    config: PilotConfig,
    torch: Any,
    device: str,
) -> TextFeatureEnsemble:
    import torch.nn.functional as functional

    prompt_features = []
    prompt_counts: list[int] = []
    with torch.inference_mode():
        for text_class in config.classes:
            prompts = prompt_texts(text_class)
            tokens = tokenizer.tokenize(prompts).to(device, non_blocking=True)
            features = model.encode_text(tokens)
            features = features[:, features.shape[1] // 2 :]
            features = functional.normalize(features, p=2, dim=-1)
            prompt_features.append(features)
            prompt_counts.append(int(features.shape[0]))
    return TextFeatureEnsemble(
        features=torch.cat(prompt_features, dim=0),
        prompt_counts=tuple(prompt_counts),
        aggregation=config.prompt_aggregation,
        top_k=config.prompt_top_k,
    )


def aggregate_prompt_logits(
    prompt_logits: Any,
    prompt_counts: tuple[int, ...],
    aggregation: str,
    top_k: int,
) -> Any:
    """Reduce per-alias logits into one logit map per configured class."""

    import torch

    if prompt_logits.ndim != 3:
        raise ValueError("prompt_logits must have shape PxHxW")
    if not prompt_counts or sum(prompt_counts) != int(prompt_logits.shape[0]):
        raise ValueError("prompt_counts do not match prompt_logits")
    if aggregation not in {"mean", "max", "topk_mean"}:
        raise ValueError(f"unsupported prompt aggregation: {aggregation}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    class_logits = []
    offset = 0
    for count in prompt_counts:
        if count < 1:
            raise ValueError("each class must have at least one prompt")
        values = prompt_logits[offset : offset + count]
        offset += count
        if aggregation == "mean":
            reduced = values.mean(dim=0)
        elif aggregation == "max":
            reduced = values.max(dim=0).values
        else:
            selected_count = min(int(top_k), count)
            reduced = values.topk(selected_count, dim=0).values.mean(dim=0)
        class_logits.append(reduced)
    return torch.stack(class_logits, dim=0)


def predict_dinotxt_probabilities(
    model: Any,
    torch: Any,
    image: Any,
    text_features: TextFeatureEnsemble | Any,
    original_size: tuple[int, int],
    side: int,
    stride: int,
    device: str,
    logit_scale: float,
) -> np.ndarray:
    import torch.nn.functional as functional

    _, height, width = image.shape
    if isinstance(text_features, TextFeatureEnsemble):
        prompt_features = text_features.features
        class_count = len(text_features.prompt_counts)
    else:
        # Preserve compatibility with callers that provide one feature per
        # class from the original mean-ensemble implementation.
        prompt_features = text_features
        class_count = int(text_features.shape[0])
    totals = torch.zeros((class_count, height, width), device=device, dtype=torch.float32)
    counts = torch.zeros((height, width), device=device, dtype=torch.float32)
    row_count = max(height - side + stride - 1, 0) // stride + 1
    column_count = max(width - side + stride - 1, 0) // stride + 1
    patch_size = int(model.visual_model.backbone.patch_size)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.startswith("cuda"),
    ):
        for row in range(row_count):
            for column in range(column_count):
                y1 = row * stride
                x1 = column * stride
                y2 = min(y1 + side, height)
                x2 = min(x1 + side, width)
                y1 = max(y2 - side, 0)
                x1 = max(x2 - side, 0)
                crop = image[:, y1:y2, x1:x2].unsqueeze(0)
                crop_height = int(math.ceil(crop.shape[-2] / patch_size) * patch_size)
                crop_width = int(math.ceil(crop.shape[-1] / patch_size) * patch_size)
                if tuple(crop.shape[-2:]) != (crop_height, crop_width):
                    crop = functional.interpolate(
                        crop,
                        size=(crop_height, crop_width),
                        mode="bicubic",
                        align_corners=False,
                    )
                _, patch_tokens, _ = model.visual_model.get_class_and_patch_tokens(crop)
                features = patch_tokens.reshape(
                    1,
                    crop_height // patch_size,
                    crop_width // patch_size,
                    -1,
                )[0]
                features = functional.normalize(features, p=2, dim=-1)
                prompt_logits = torch.einsum("pd,hwd->phw", prompt_features, features)
                if isinstance(text_features, TextFeatureEnsemble):
                    cosine = aggregate_prompt_logits(
                        prompt_logits,
                        text_features.prompt_counts,
                        text_features.aggregation,
                        text_features.top_k,
                    )
                else:
                    cosine = prompt_logits
                scores = functional.interpolate(
                    scale_cosine_logits(cosine, logit_scale).unsqueeze(0),
                    size=(y2 - y1, x2 - x1),
                    mode="bilinear",
                    align_corners=False,
                )[0].softmax(dim=0)
                totals[:, y1:y2, x1:x2] += scores.float()
                counts[y1:y2, x1:x2] += 1.0
    totals /= counts.clamp_min(1.0)
    original_width, original_height = original_size
    totals = functional.interpolate(
        totals.unsqueeze(0),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    return totals.cpu().numpy().astype(np.float32, copy=False)


def load_dinotxt(
    dinov3_root: Path,
    backbone_checkpoint: Path,
    dinotxt_checkpoint: Path,
    bpe_path: Path,
    hub_entry: str,
    device: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    sys.path.insert(0, str(dinov3_root.resolve()))
    import torch

    provenance = repository_provenance(dinov3_root, PINNED_DINOV3_COMMIT)
    backbone_sha = verify_checkpoint_prefix(backbone_checkpoint, EXPECTED_VITL_SHA256_PREFIX)
    dinotxt_sha = verify_checkpoint_prefix(dinotxt_checkpoint, EXPECTED_DINOTXT_SHA256_PREFIX)
    bpe_sha = verify_checkpoint_prefix(bpe_path, EXPECTED_BPE_SHA256)
    with checkpoint_state_dict_loader(
        torch,
        "local_mmap",
        (backbone_checkpoint, dinotxt_checkpoint),
        integrity_preverified=True,
    ) as loading:
        model, tokenizer = torch.hub.load(
            str(dinov3_root.resolve()),
            hub_entry,
            source="local",
            pretrained=True,
            weights=str(dinotxt_checkpoint.resolve()),
            backbone_weights=str(backbone_checkpoint.resolve()),
            bpe_path_or_url=str(bpe_path.resolve()),
            check_hash=True,
        )
    intercepted = {Path(value).resolve() for value in loading["intercepted_checkpoint_paths"]}
    expected = {backbone_checkpoint.resolve(), dinotxt_checkpoint.resolve()}
    if intercepted != expected:
        raise RuntimeError("local mmap did not intercept both verified dino.txt checkpoints")
    model.eval().to(device)
    logit_scale = float(model.logit_scale.detach().float().exp().cpu().item())
    scale_cosine_logits(1.0, logit_scale)
    return model, tokenizer, torch, {
        "repository": provenance,
        "hub_entry": hub_entry,
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_sha256": backbone_sha,
        "dinotxt_checkpoint": str(dinotxt_checkpoint),
        "dinotxt_sha256": dinotxt_sha,
        "bpe_path": str(bpe_path),
        "bpe_sha256": bpe_sha,
        "logit_scale": logit_scale,
    }


def load_sam_generator(
    segment_anything_root: Path,
    checkpoint: Path,
    device: str,
    points_per_side: int,
    predicted_iou_threshold: float,
    stability_threshold: float,
) -> Any:
    sys.path.insert(0, str(segment_anything_root.resolve()))
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=predicted_iou_threshold,
        stability_score_thresh=stability_threshold,
        crop_n_layers=0,
        min_mask_region_area=0,
    )


def heatmap_image(probability: np.ndarray) -> Image.Image:
    values = np.clip(probability, 0.0, 1.0)
    red = np.clip(2.0 * values, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - values), 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * values - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def overlay_selected(
    rgb: np.ndarray,
    masks: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> Image.Image:
    overlay = rgb.astype(np.float32, copy=True)
    color = np.asarray([136, 60, 222], dtype=np.float32)
    union = np.zeros(rgb.shape[:2], dtype=bool)
    for record in selected:
        union |= np.asarray(masks[int(record["mask_id"])]["segmentation"], dtype=bool)
    overlay[union] = 0.45 * overlay[union] + 0.55 * color
    image = Image.fromarray(overlay.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for record in selected:
        x, y, width, height = record["bbox_xywh"]
        draw.rectangle((x, y, x + width, y + height), outline=(136, 60, 222), width=2)
        label = (
            f"mask {record['mask_id']} p={record['target_probability']:.2f} "
            f"margin={record['target_margin']:.2f}"
        )
        draw.text((x, max(0, y - 12)), label, fill=(136, 60, 222), font=font)
    return image


def overlay_ranked(
    rgb: np.ndarray,
    masks: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    limit: int = 5,
) -> Image.Image:
    image = rgb.astype(np.float32, copy=True)
    colors = (
        np.asarray([255, 80, 80], dtype=np.float32),
        np.asarray([80, 180, 255], dtype=np.float32),
        np.asarray([255, 190, 60], dtype=np.float32),
        np.asarray([100, 220, 130], dtype=np.float32),
        np.asarray([200, 100, 240], dtype=np.float32),
    )
    ranked = scores[:limit]
    for rank, record in reversed(list(enumerate(ranked, start=1))):
        mask = np.asarray(masks[int(record["mask_id"])]["segmentation"], dtype=bool)
        color = colors[(rank - 1) % len(colors)]
        image[mask] = 0.72 * image[mask] + 0.28 * color
    rendered = Image.fromarray(image.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    for rank, record in enumerate(ranked, start=1):
        x, y, width, height = record["bbox_xywh"]
        color = tuple(int(value) for value in colors[(rank - 1) % len(colors)])
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
        label = (
            f"#{rank} p={record['target_probability']:.2f} "
            f"m={record['target_margin']:.2f} vs {record['best_competitor']}"
        )
        draw.text((x, max(0, y - 12)), label, fill=color, font=font)
    return rendered


def save_review_panel(
    rgb: np.ndarray,
    target_probability: np.ndarray,
    selected_overlay: Image.Image,
    ranked_overlay: Image.Image,
    path: Path,
) -> None:
    original = Image.fromarray(rgb, mode="RGB")
    heatmap = heatmap_image(target_probability)
    width, height = original.size
    panel = Image.new("RGB", (width * 2, height * 2), "white")
    panel.paste(original, (0, 0))
    panel.paste(heatmap, (width, 0))
    panel.paste(selected_overlay, (0, height))
    panel.paste(ranked_overlay, (width, height))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 8), "RGB", fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((width + 8, 8), "dino.txt target probability", fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((8, height + 8), "automatically selected SAM masks", fill="white", font=font, stroke_width=2, stroke_fill="black")
    draw.text((width + 8, height + 8), "top five SAM masks by target margin", fill="white", font=font, stroke_width=2, stroke_fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-dir", required=True, type=Path)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument(
        "--view-count",
        default=0,
        type=int,
        help="Select this many evenly spaced manifest frames when camera indices are omitted",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dinov3-root", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--dinotxt-checkpoint", required=True, type=Path)
    parser.add_argument("--bpe-path", required=True, type=Path)
    parser.add_argument("--dinotxt-hub-entry", default=DEFAULT_HUB_ENTRY)
    parser.add_argument("--segment-anything-root", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--side", default=384, type=int)
    parser.add_argument("--stride", default=192, type=int)
    parser.add_argument("--sam-points-per-side", default=32, type=int)
    parser.add_argument("--sam-predicted-iou-threshold", default=0.88, type=float)
    parser.add_argument("--sam-stability-threshold", default=0.92, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resize < 1 or args.side < 1 or args.stride < 1 or args.stride > args.side:
        raise ValueError("resize/side/stride geometry is invalid")
    if args.sam_points_per_side < 1:
        raise ValueError("sam-points-per-side must be positive")
    manifest_path = args.view_dir / "view_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    config = load_config(args.config)
    camera_indices = [int(value.strip()) for value in args.camera_indices.split(",") if value.strip()]
    if camera_indices and args.view_count:
        raise ValueError("--camera-indices and --view-count are mutually exclusive")
    if not camera_indices and args.view_count < 1:
        raise ValueError("provide --camera-indices or a positive --view-count")
    view_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = (
        select_frames(view_manifest, camera_indices)
        if camera_indices
        else select_evenly_spaced_frames(view_manifest, args.view_count)
    )
    camera_indices = [int(frame["camera_index"]) for frame in frames]
    input_paths = [args.view_dir / "rgb_renders" / str(frame["file"]) for frame in frames]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.dinov3_root,
        args.backbone_checkpoint,
        args.dinotxt_checkpoint,
        args.bpe_path,
        args.segment_anything_root,
        args.sam_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, torch, model_record = load_dinotxt(
        args.dinov3_root,
        args.backbone_checkpoint,
        args.dinotxt_checkpoint,
        args.bpe_path,
        args.dinotxt_hub_entry,
        args.device,
    )
    text_features = encode_text_classes(model, tokenizer, config, torch, args.device)
    probability_paths: list[Path] = []
    for frame, path in zip(frames, input_paths, strict=True):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        tensor = prepare_image(torch, rgb, args.resize, args.device)
        probabilities = predict_dinotxt_probabilities(
            model,
            torch,
            tensor,
            text_features,
            (rgb.shape[1], rgb.shape[0]),
            args.side,
            args.stride,
            args.device,
            float(model_record["logit_scale"]),
        )
        probability_path = args.output_dir / "probabilities" / f"{Path(str(frame['file'])).stem}.npy"
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(probability_path, probabilities.astype(np.float16))
        probability_paths.append(probability_path)
    del text_features, model, tokenizer
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    sam_generator = load_sam_generator(
        args.segment_anything_root,
        args.sam_checkpoint,
        args.device,
        args.sam_points_per_side,
        args.sam_predicted_iou_threshold,
        args.sam_stability_threshold,
    )
    class_names = [item.name for item in config.classes]
    target_id = class_names.index(config.target_class)
    frame_records: list[dict[str, Any]] = []
    mask_manifest_frames: list[dict[str, Any]] = []
    for frame, path, probability_path in zip(frames, input_paths, probability_paths, strict=True):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        masks = sam_generator.generate(rgb)
        probabilities = np.load(probability_path).astype(np.float32)
        scores = score_sam_masks(masks, probabilities, class_names, config.target_class)
        selected = select_target_masks(masks, scores, config)
        stem = Path(str(frame["file"])).stem
        selected_stack = [
            np.asarray(masks[int(record["mask_id"])]["segmentation"], dtype=bool)
            for record in selected
        ]
        stack = (
            np.stack(selected_stack, axis=0)
            if selected_stack
            else np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=bool)
        )
        mask_stack_path = args.output_dir / "mask_stacks" / f"{stem}.npz"
        mask_stack_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_stack_path, masks=stack)
        for selected_index, record in enumerate(selected):
            mask = np.asarray(masks[int(record["mask_id"])]["segmentation"], dtype=np.uint8) * 255
            mask_dir = args.output_dir / "selected_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}__{selected_index:02d}.png")
        overlay = overlay_selected(rgb, masks, selected)
        overlay_dir = args.output_dir / "selected_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_dir / f"{stem}.png")
        ranked = overlay_ranked(rgb, masks, scores)
        ranked_dir = args.output_dir / "ranked_overlays"
        ranked_dir.mkdir(parents=True, exist_ok=True)
        ranked.save(ranked_dir / f"{stem}.png")
        panel_path = args.output_dir / "review_panels" / f"{stem}.png"
        save_review_panel(rgb, probabilities[target_id], overlay, ranked, panel_path)
        frame_records.append(
            {
                "camera_index": int(frame["camera_index"]),
                "file": str(frame["file"]),
                "sam_mask_count": len(masks),
                "selected_mask_count": len(selected),
                "selected": selected,
                "top_target_candidates": scores[:20],
                "probabilities": str(probability_path.relative_to(args.output_dir)),
                "selected_overlay": str((overlay_dir / f"{stem}.png").relative_to(args.output_dir)),
                "ranked_overlay": str((ranked_dir / f"{stem}.png").relative_to(args.output_dir)),
                "review_panel": str(panel_path.relative_to(args.output_dir)),
            }
        )
        mask_manifest_frames.append(
            {
                "camera_index": int(frame["camera_index"]),
                "camera_id": int(frame.get("camera_id", frame["camera_index"])),
                "image_name": str(frame.get("image_name", frame["file"])),
                "file": str(frame["file"]),
                "mask_file": mask_stack_path.name,
                "masks": flashsplat_mask_metadata(selected, config.target_class),
            }
        )

    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "review_only": True,
        "flashsplat_run": False,
        "semantic_labels_written": False,
        "semantic_ply_written": False,
        "config": {
            "target_class": config.target_class,
            "classes": [asdict(item) for item in config.classes],
            "selection": {
                "prompt_aggregation": config.prompt_aggregation,
                "prompt_top_k": config.prompt_top_k,
                "min_target_probability": config.min_target_probability,
                "min_competitor_margin": config.min_competitor_margin,
                "min_target_win_fraction": config.min_target_win_fraction,
                "max_selected_masks": config.max_selected_masks,
                "containment_threshold": config.selection_containment_threshold,
                "nms_iou": config.selection_nms_iou,
            },
        },
        "model": model_record,
        "sam": {
            "checkpoint": str(args.sam_checkpoint),
            "points_per_side": args.sam_points_per_side,
            "predicted_iou_threshold": args.sam_predicted_iou_threshold,
            "stability_threshold": args.sam_stability_threshold,
        },
        "inference": {
            "resize": args.resize,
            "side": args.side,
            "stride": args.stride,
            "camera_indices": camera_indices,
        },
        "frames": frame_records,
    }
    (args.output_dir / "dinotxt_sam_pilot_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    mask_manifest = {
        "source": "dinov3_dinotxt_sam_selected_masks",
        "contract": "flashsplat_mask_stack_manifest_v1",
        "review_only_2d_source": True,
        "target_class": config.target_class,
        "view_manifest": str(manifest_path),
        "camera_count": len(mask_manifest_frames),
        "frames_with_selected_masks": sum(
            bool(frame["masks"]) for frame in mask_manifest_frames
        ),
        "frames": mask_manifest_frames,
    }
    (args.output_dir / "grounded_sam_manifest.json").write_text(
        json.dumps(mask_manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "frames": len(frame_records), "selected_masks": sum(item["selected_mask_count"] for item in frame_records)}))


if __name__ == "__main__":
    main()
