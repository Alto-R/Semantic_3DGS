#!/usr/bin/env python3
"""Generate semantic GroundingDINO + SAM mask proposals from 3DGS renders."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.task1.common.flashsplat_cameras import (
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
from scripts.task1.common.semantic_palette import rgb8_for_class


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_FLASHSPLAT_ROOT = WORKSPACE_ROOT / "external" / "FlashSplat"
DEFAULT_DEVA_ROOT = WORKSPACE_ROOT / "gaussian-grouping" / "Tracking-Anything-with-DEVA"
DEFAULT_GROUNDED_SAM_ROOT = DEFAULT_DEVA_ROOT / "Grounded-Segment-Anything"
DEFAULT_GROUNDINGDINO_CONFIG = DEFAULT_DEVA_ROOT / "saves" / "GroundingDINO_SwinT_OGC.py"
DEFAULT_GROUNDINGDINO_CHECKPOINT = DEFAULT_DEVA_ROOT / "saves" / "groundingdino_swint_ogc.pth"
DEFAULT_SEGMENT_ANYTHING_ROOT = DEFAULT_GROUNDED_SAM_ROOT / "segment_anything"
DEFAULT_SAM_CHECKPOINT = WORKSPACE_ROOT / "InvRGBL_modif" / "pretrained" / "sam_vit_h_4b8939.pth"


DEFAULT_CLASSES = [
    {"class": "bicycle", "prompts": ["bicycle", "bike"], "type": "thing"},
    {"class": "car", "prompts": ["car", "vehicle"], "type": "thing"},
    {"class": "bench", "prompts": ["bench"], "type": "thing"},
    {"class": "tree", "prompts": ["tree"], "type": "thing"},
    {"class": "building", "prompts": ["building", "house"], "type": "stuff"},
    {"class": "ground", "prompts": ["ground", "terrain"], "type": "stuff"},
    {"class": "road", "prompts": ["road", "street"], "type": "stuff"},
    {"class": "sidewalk", "prompts": ["sidewalk", "pavement"], "type": "stuff"},
    {"class": "sky", "prompts": ["sky"], "type": "stuff"},
    {"class": "pole", "prompts": ["pole", "streetlight", "lamp post"], "type": "thing"},
    {"class": "sign", "prompts": ["sign", "traffic sign"], "type": "thing"},
    {"class": "fence", "prompts": ["fence"], "type": "thing"},
    {"class": "person", "prompts": ["person", "pedestrian"], "type": "thing"},
    {"class": "vegetation", "prompts": ["vegetation", "bush", "grass", "plant"], "type": "stuff"},
]


@dataclass(frozen=True)
class ClassSpec:
    name: str
    prompts: tuple[str, ...]
    kind: str = "thing"


@dataclass
class Detection:
    mask: np.ndarray
    class_name: str
    phrase: str
    grounding_score: float
    sam_score: float
    bbox_xyxy: list[float]
    area: int


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def split_prompt_terms(text_prompt: str) -> list[str]:
    return [term.strip().lower() for term in re.split(r"[.,;|]+", text_prompt) if term.strip()]


def parse_class_specs(path: Path | None, text_prompt: str | None) -> list[ClassSpec]:
    raw_specs: Any
    if path is not None:
        raw_specs = json.loads(path.read_text(encoding="utf-8"))
    elif text_prompt:
        raw_specs = [{"class": term, "prompts": [term], "type": "thing"} for term in split_prompt_terms(text_prompt)]
    else:
        raw_specs = DEFAULT_CLASSES

    if isinstance(raw_specs, dict):
        raw_specs = raw_specs.get("classes", raw_specs)
    if not isinstance(raw_specs, list):
        raise ValueError("Class config must be a JSON list or an object with a classes list")

    specs: list[ClassSpec] = []
    for item in raw_specs:
        if isinstance(item, str):
            specs.append(ClassSpec(name=normalize_text(item).replace(" ", "_"), prompts=(item,), kind="thing"))
            continue
        class_name = str(item.get("class", item.get("name", ""))).strip().lower().replace(" ", "_")
        if not class_name:
            raise ValueError(f"Class config item has no class/name: {item}")
        prompts = item.get("prompts", item.get("aliases", [class_name.replace("_", " ")]))
        if isinstance(prompts, str):
            prompts = [prompts]
        prompt_tuple = tuple(str(prompt).strip().lower() for prompt in prompts if str(prompt).strip())
        if not prompt_tuple:
            prompt_tuple = (class_name.replace("_", " "),)
        specs.append(ClassSpec(name=class_name, prompts=prompt_tuple, kind=str(item.get("type", "thing"))))
    return specs


def select_class_specs(specs: list[ClassSpec], include_classes: str) -> list[ClassSpec]:
    requested = [
        normalize_text(value).replace(" ", "_")
        for value in include_classes.split(",")
        if value.strip()
    ]
    if not requested:
        return specs
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate include classes: {requested}")
    by_name = {spec.name: spec for spec in specs}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"Included classes are absent from the class config: {unknown}")
    return [by_name[name] for name in requested]


def text_prompt_from_specs(specs: list[ClassSpec], text_prompt: str | None) -> str:
    if text_prompt:
        prompt = text_prompt.strip()
    else:
        terms: list[str] = []
        seen: set[str] = set()
        for spec in specs:
            for prompt in spec.prompts:
                normalized = prompt.strip().lower()
                if normalized and normalized not in seen:
                    terms.append(normalized)
                    seen.add(normalized)
        prompt = " . ".join(terms)
    return prompt if prompt.endswith(".") else prompt + " ."


def class_from_phrase(phrase: str, specs: list[ClassSpec]) -> str:
    normalized_phrase = normalize_text(phrase)
    exact_matches: list[tuple[str, frozenset[str]]] = []
    for spec in specs:
        for prompt in spec.prompts:
            normalized_prompt = normalize_text(prompt)
            if normalized_prompt and re.search(rf"\b{re.escape(normalized_prompt)}\b", normalized_phrase):
                exact_matches.append((spec.name, frozenset(normalized_prompt.split())))

    if exact_matches:
        # GroundingDINO sometimes emits a span containing several configured
        # prompts. Ignore a short match only when it is wholly contained in a
        # longer matched prompt ("floor" inside "floor speaker"). If unrelated
        # classes remain ("television stand table desk"), reject the ambiguous
        # phrase instead of assigning it to whichever class appears first.
        maximal_matches = [
            (class_name, prompt_tokens)
            for class_name, prompt_tokens in exact_matches
            if not any(prompt_tokens < other_tokens for _, other_tokens in exact_matches)
        ]
        matched_classes = {class_name for class_name, _ in maximal_matches}
        if len(matched_classes) == 1:
            return next(iter(matched_classes))
        return "unknown"

    # GroundingDINO can return only one token from a configured multiword
    # prompt (for example, "acoustic" for "acoustic guitar"). Resolve a
    # unique partial prompt back to the configured class instead of inventing
    # an out-of-vocabulary class from the first token.
    phrase_tokens = frozenset(normalized_phrase.split())
    partial_scores: dict[str, float] = {}
    if phrase_tokens:
        for spec in specs:
            for prompt in spec.prompts:
                prompt_tokens = frozenset(normalize_text(prompt).split())
                if phrase_tokens < prompt_tokens:
                    coverage = len(phrase_tokens) / float(len(prompt_tokens))
                    partial_scores[spec.name] = max(partial_scores.get(spec.name, 0.0), coverage)
    if partial_scores:
        best_score = max(partial_scores.values())
        best_names = [name for name, score in partial_scores.items() if score == best_score]
        if len(best_names) == 1:
            return best_names[0]
    return "unknown"


def load_grounding_imports(root: Path) -> dict[str, Any]:
    candidates = [root]
    if (root / "GroundingDINO").exists():
        candidates.append(root / "GroundingDINO")
    for candidate in candidates:
        if candidate and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        import GroundingDINO.groundingdino.datasets.transforms as transforms
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    except ModuleNotFoundError:
        import groundingdino.datasets.transforms as transforms
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

    return {
        "transforms": transforms,
        "build_model": build_model,
        "SLConfig": SLConfig,
        "clean_state_dict": clean_state_dict,
        "get_phrases_from_posmap": get_phrases_from_posmap,
    }


def load_grounding_model(root: Path, config: Path, checkpoint: Path, device: str) -> tuple[Any, Any, Any]:
    imports = load_grounding_imports(root)
    args = imports["SLConfig"].fromfile(str(config))
    args.device = device
    model = imports["build_model"](args)
    payload = torch.load(str(checkpoint), map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    load_result = model.load_state_dict(imports["clean_state_dict"](state_dict), strict=False)
    print(f"loaded GroundingDINO: {load_result}")
    model.eval()
    model.to(device)
    return model, imports["transforms"], imports["get_phrases_from_posmap"]


def load_sam_predictor(segment_anything_root: Path, checkpoint: Path, arch: str, device: str) -> Any:
    if segment_anything_root and str(segment_anything_root) not in sys.path:
        sys.path.insert(0, str(segment_anything_root))
    from segment_anything import SamPredictor, build_sam

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        sam_model_registry = {}

    if checkpoint and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if arch in sam_model_registry:
        sam = sam_model_registry[arch](checkpoint=str(checkpoint))
    else:
        sam = build_sam(checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def grounding_transform(transforms_module: Any) -> Any:
    return transforms_module.Compose(
        [
            transforms_module.RandomResize([800], max_size=1333),
            transforms_module.ToTensor(),
            transforms_module.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def run_grounding_dino(
    model: Any,
    transforms_module: Any,
    phrase_from_posmap: Any,
    rgb: np.ndarray,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[torch.Tensor, list[str], list[float]]:
    image_pil = Image.fromarray(rgb, mode="RGB")
    image_tensor, _ = grounding_transform(transforms_module)(image_pil, None)
    image_tensor = image_tensor.to(device)
    caption = text_prompt.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    keep = logits.max(dim=1)[0] > box_threshold
    logits = logits[keep]
    boxes = boxes[keep]

    tokenized = model.tokenizer(caption)
    phrases: list[str] = []
    scores: list[float] = []
    for logit in logits:
        phrase = phrase_from_posmap(logit > text_threshold, tokenized, model.tokenizer)
        phrases.append(str(phrase))
        scores.append(float(logit.max().item()))
    return boxes, phrases, scores


def cxcywh_to_xyxy_pixels(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, width - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, height - 1)
    return boxes_xyxy


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(left, right).sum())
    return intersection / float(max(union, 1))


def filter_detections(
    detections: list[Detection],
    min_area: int,
    max_area_ratio: float,
    image_area: int,
    mask_nms_iou: float,
    max_detections: int,
) -> list[Detection]:
    kept: list[Detection] = []
    detections.sort(key=lambda detection: (detection.grounding_score, detection.sam_score, detection.area), reverse=True)
    for detection in detections:
        if detection.area < min_area:
            continue
        if max_area_ratio > 0 and detection.area > image_area * max_area_ratio:
            continue
        duplicate = False
        for existing in kept:
            if existing.class_name == detection.class_name and mask_iou(existing.mask, detection.mask) >= mask_nms_iou:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(detection)
        if max_detections > 0 and len(kept) >= max_detections:
            break
    return kept


def run_sam_for_boxes(
    predictor: Any,
    rgb: np.ndarray,
    boxes_xyxy: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, list[float]]:
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=bool), []
    predictor.set_image(rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, rgb.shape[:2]).to(device)
    with torch.no_grad():
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    masks_np = masks[:, 0].detach().cpu().numpy().astype(bool)
    scores_np = scores[:, 0].detach().cpu().numpy().astype(float).tolist()
    return masks_np, scores_np


def record_for_detection(detection: Detection, index: int) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return {
        "mask_index": index,
        "source": "groundingdino_sam",
        "class": detection.class_name,
        "phrase": detection.phrase,
        "confidence": float(detection.grounding_score),
        "grounding_score": float(detection.grounding_score),
        "sam_score": float(detection.sam_score),
        "area": int(detection.area),
        "bbox": [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))],
        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
    }


def save_overlay(rgb: np.ndarray, detections: list[Detection], output_path: Path) -> None:
    overlay = rgb.copy()
    for detection in detections:
        color = np.asarray(rgb8_for_class(detection.class_name), dtype=np.uint8)
        selected = detection.mask.astype(bool)
        overlay[selected] = (0.55 * overlay[selected] + 0.45 * color).astype(np.uint8)

    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = tuple(rgb8_for_class(detection.class_name))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = f"{index}:{detection.class_name} {detection.grounding_score:.2f}"
        text_xy = (max(0, int(x1)), max(0, int(y1) - 12))
        draw.text(text_xy, label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "unknown"


def save_binary_masks(
    detections: list[Detection],
    stem: str,
    output_dir: Path,
) -> list[str]:
    filenames: list[str] = []
    for index, detection in enumerate(detections):
        filename = f"{stem}__mask_{index:03d}__{safe_filename_part(detection.class_name)}.png"
        mask_image = Image.fromarray(detection.mask.astype(np.uint8) * 255, mode="L")
        mask_image.save(output_dir / filename)
        filenames.append(filename)
    return filenames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", default=DEFAULT_FLASHSPLAT_ROOT, type=Path)
    parser.add_argument(
        "--groundingdino-root",
        default=DEFAULT_GROUNDED_SAM_ROOT,
        type=Path,
    )
    parser.add_argument(
        "--groundingdino-config",
        default=DEFAULT_GROUNDINGDINO_CONFIG,
        type=Path,
    )
    parser.add_argument(
        "--groundingdino-checkpoint",
        default=DEFAULT_GROUNDINGDINO_CHECKPOINT,
        type=Path,
    )
    parser.add_argument(
        "--segment-anything-root",
        default=DEFAULT_SEGMENT_ANYTHING_ROOT,
        type=Path,
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=DEFAULT_SAM_CHECKPOINT,
        type=Path,
    )
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--class-config", type=Path)
    parser.add_argument("--include-classes", default="")
    parser.add_argument("--text-prompt", default="")
    parser.add_argument("--source-view-manifest", type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--count", default=20, type=int)
    parser.add_argument("--max-width", default=320, type=int)
    parser.add_argument("--max-detections-per-view", default=32, type=int)
    parser.add_argument("--box-threshold", default=0.30, type=float)
    parser.add_argument("--text-threshold", default=0.25, type=float)
    parser.add_argument("--min-mask-area", default=100, type=int)
    parser.add_argument("--max-mask-area-ratio", default=0.80, type=float)
    parser.add_argument("--mask-nms-iou", default=0.80, type=float)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default GroundingDINO/SAM path")

    specs = select_class_specs(
        parse_class_specs(args.class_config, args.text_prompt or None),
        args.include_classes,
    )
    if not specs:
        raise ValueError("No GroundingDINO classes were selected")
    text_prompt = text_prompt_from_specs(specs, args.text_prompt or None)

    source_view_manifest: dict[str, Any] | None = None
    source_frames_by_camera: dict[int, dict[str, Any]] = {}
    if args.source_view_manifest is not None:
        source_view_manifest = json.loads(args.source_view_manifest.read_text(encoding="utf-8"))
        source_frames_by_camera = {
            int(frame["camera_index"]): frame
            for frame in source_view_manifest.get("frames", [])
        }
        rgb_dir = args.source_view_manifest.parent / "rgb_renders"
        if not rgb_dir.is_dir():
            raise FileNotFoundError(rgb_dir)
    else:
        rgb_dir = args.output_dir / "rgb_renders"
    mask_dir = args.output_dir / "mask_stacks"
    binary_mask_dir = args.output_dir / "binary_masks"
    overlay_dir = args.output_dir / "overlays"
    for directory in (mask_dir, binary_mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if source_view_manifest is None:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras(args.model_path)
    selected_items = selected_camera_items(cameras, args.camera_indices, args.count)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    missing_source_views = [
        camera_index
        for camera_index, _camera in selected_items
        if source_view_manifest is not None and camera_index not in source_frames_by_camera
    ]
    if missing_source_views:
        raise ValueError(
            "Selected cameras are absent from the source view manifest: "
            f"{missing_source_views}"
        )
    modules = None
    gaussians = None
    pipeline = None
    background = None
    if source_view_manifest is None:
        modules = load_flashsplat(args.flashsplat_root)
        gaussians = load_gaussians(modules, ply_path, args.sh_degree)
        pipeline = default_pipeline()
        background = background_tensor(args.white_background)

    grounding_model, transforms_module, phrase_from_posmap = load_grounding_model(
        args.groundingdino_root,
        args.groundingdino_config,
        args.groundingdino_checkpoint,
        device,
    )
    sam_predictor = load_sam_predictor(args.segment_anything_root, args.sam_checkpoint, args.sam_arch, device)

    manifest: dict[str, Any] = {
        "source": "groundingdino_sam",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "requested_view_count": args.count,
        "requested_camera_indices": args.camera_indices,
        "selected_camera_indices": [camera_index for camera_index, _ in selected_items],
        "source_view_manifest": (
            str(args.source_view_manifest) if args.source_view_manifest is not None else None
        ),
        "class_config": str(args.class_config) if args.class_config is not None else None,
        "included_classes": args.include_classes,
        "reused_rgb_renders": source_view_manifest is not None,
        "output_directories": {
            "rgb_renders": str(rgb_dir),
            "mask_stacks": str(mask_dir),
            "binary_masks": str(binary_mask_dir),
            "overlays": str(overlay_dir),
        },
        "text_prompt": text_prompt,
        "classes": [
            {"class": spec.name, "prompts": list(spec.prompts), "type": spec.kind}
            for spec in specs
        ],
        "groundingdino": {
            "root": str(args.groundingdino_root),
            "config": str(args.groundingdino_config),
            "checkpoint": str(args.groundingdino_checkpoint),
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        },
        "sam": {
            "root": str(args.segment_anything_root),
            "checkpoint": str(args.sam_checkpoint),
            "arch": args.sam_arch,
        },
        "frames": [],
    }

    with torch.no_grad():
        for output_index, (camera_index, camera_json) in enumerate(selected_items):
            if source_view_manifest is not None:
                source_frame = source_frames_by_camera[camera_index]
                filename = str(source_frame["file"])
                rgb_path = rgb_dir / filename
                if not rgb_path.exists():
                    raise FileNotFoundError(rgb_path)
                rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
                render_height, render_width = rgb.shape[:2]
            else:
                if modules is None or gaussians is None or pipeline is None or background is None:
                    raise RuntimeError("FlashSplat render state was not initialized")
                camera = make_camera(camera_json, modules, args.max_width)
                render_pkg = render_flashsplat(camera, gaussians, modules, pipeline, background)
                rgb = tensor_to_rgb_array(render_pkg["render"])
                filename = camera_filename(output_index, camera_json)
                render_width = int(camera.image_width)
                render_height = int(camera.image_height)
            stem = Path(filename).stem
            if source_view_manifest is None:
                Image.fromarray(rgb, mode="RGB").save(rgb_dir / filename)

            boxes, phrases, scores = run_grounding_dino(
                grounding_model,
                transforms_module,
                phrase_from_posmap,
                rgb,
                text_prompt,
                args.box_threshold,
                args.text_threshold,
                device,
            )
            boxes_xyxy = cxcywh_to_xyxy_pixels(boxes, render_width, render_height)
            masks, sam_scores = run_sam_for_boxes(sam_predictor, rgb, boxes_xyxy, device)

            detections: list[Detection] = []
            for index in range(masks.shape[0]):
                phrase = phrases[index] if index < len(phrases) else ""
                class_name = class_from_phrase(phrase, specs)
                area = int(masks[index].sum())
                detections.append(
                    Detection(
                        mask=masks[index],
                        class_name=class_name,
                        phrase=phrase,
                        grounding_score=float(scores[index]) if index < len(scores) else 0.0,
                        sam_score=float(sam_scores[index]) if index < len(sam_scores) else 0.0,
                        bbox_xyxy=[float(value) for value in boxes_xyxy[index].tolist()],
                        area=area,
                    )
                )
            detections = filter_detections(
                detections,
                min_area=args.min_mask_area,
                max_area_ratio=args.max_mask_area_ratio,
                image_area=int(render_width * render_height),
                mask_nms_iou=args.mask_nms_iou,
                max_detections=args.max_detections_per_view,
            )

            mask_stack = np.zeros((0, render_height, render_width), dtype=np.uint8)
            if detections:
                mask_stack = np.stack([detection.mask.astype(np.uint8) for detection in detections], axis=0)
            np.savez_compressed(mask_dir / f"{stem}.npz", masks=mask_stack)
            binary_mask_files = save_binary_masks(detections, stem, binary_mask_dir)
            save_overlay(rgb, detections, overlay_dir / filename)

            frame_record = {
                "file": filename,
                "mask_file": f"{stem}.npz",
                "binary_mask_files": binary_mask_files,
                "camera_index": camera_index,
                "camera_id": int(camera_json["id"]),
                "image_name": camera_json.get("img_name", ""),
                "render_width": render_width,
                "render_height": render_height,
                "raw_detection_count": int(boxes.shape[0]),
                "kept_mask_count": int(mask_stack.shape[0]),
                "masks": [
                    {
                        **record_for_detection(detection, index),
                        "binary_mask_file": binary_mask_files[index],
                    }
                    for index, detection in enumerate(detections)
                ],
            }
            manifest["frames"].append(frame_record)
            print(
                f"wrote {filename}: raw_detections={boxes.shape[0]} "
                f"kept_masks={mask_stack.shape[0]}"
            )

    (args.output_dir / "grounded_sam_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
