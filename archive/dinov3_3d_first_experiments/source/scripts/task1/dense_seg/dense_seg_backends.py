"""Pluggable dense ADE20K semantic-segmentation backends.

Every backend exposes the same contract:

    backend = build_backend(name, args)
    class_id, confidence = backend.segment(rgb)   # rgb: (H, W, 3) uint8

where ``class_id`` is an (H, W) int16 array of raw ADE20K class indices
(0..149) and ``confidence`` is an (H, W) float32 array in [0, 1].

Backends:

- ``mask2former``: HuggingFace ``facebook/mask2former-swin-large-ade-semantic``
  (or any compatible Mask2Former semantic checkpoint). The default v1 backend.
- ``dinov3``: official facebookresearch/dinov3 ViT-7B/16 + ADE20K M2F segmentor
  loaded through ``torch.hub`` from a local repo clone. Comparison backend;
  needs a 40GB-class GPU.

Model loading is deliberately deferred to first use so that importing this
module never requires GPU libraries beyond torch.

The Mask2Former backend also exposes ``segment_with_regions``. It returns the
same semantic arrays plus a compact class-agnostic region partition derived
from the model's mask queries. Downstream hybrid refinement deliberately
ignores each query's predicted ADE20K class and uses only its boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch


ADE20K_NUM_CLASSES = 150


class DenseSegBackend(Protocol):
    name: str

    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (class_id int16 HxW, confidence float32 HxW) for one image."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return manifest metadata for reproducibility."""
        ...


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB array, got dtype {rgb.dtype}")
    return rgb


@dataclass
class Mask2FormerBackend:
    """HuggingFace Mask2Former semantic segmentation on ADE20K."""

    model_id: str = "facebook/mask2former-swin-large-ade-semantic"
    device: str = "cuda"
    fp16: bool = True
    region_min_objectness: float = 0.30
    region_mask_threshold: float = 0.50
    region_min_pixel_score: float = 0.25
    region_min_area: int = 100
    region_max_area_ratio: float = 0.80
    region_max_queries: int = 64
    name: str = "mask2former"

    def __post_init__(self) -> None:
        if not 0.0 <= self.region_min_objectness <= 1.0:
            raise ValueError("region_min_objectness must be between 0 and 1")
        if not 0.0 < self.region_mask_threshold < 1.0:
            raise ValueError("region_mask_threshold must be between 0 and 1")
        if not 0.0 <= self.region_min_pixel_score <= 1.0:
            raise ValueError("region_min_pixel_score must be between 0 and 1")
        if self.region_min_area < 1:
            raise ValueError("region_min_area must be positive")
        if not 0.0 < self.region_max_area_ratio <= 1.0:
            raise ValueError("region_max_area_ratio must be greater than 0 and at most 1")
        if self.region_max_queries < 1 or self.region_max_queries > np.iinfo(np.uint16).max:
            raise ValueError("region_max_queries must fit in a positive uint16 id range")
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
        model = model.to(self.device).eval()
        # Keep FP32 master weights. Transformers 4.30 Mask2Former explicitly
        # promotes some decoder features with x.float(); converting the whole
        # model to half then pairs FP32 activations with FP16 convolution
        # biases. CUDA autocast in segment() provides mixed-precision kernels
        # without creating that invalid dtype combination.
        self._model = model

    def _class_agnostic_regions(
        self,
        class_probabilities: torch.Tensor,
        mask_probabilities: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Build a compact hard region partition from Mask2Former queries.

        Query semantic classes are retained only as diagnostic metadata. They
        never participate in region matching or downstream identity decisions.
        """

        full_class_probabilities = class_probabilities[0]
        mask_probabilities = mask_probabilities[0]
        objectness = 1.0 - full_class_probabilities[:, -1]
        raw_class_confidence, raw_class_id = full_class_probabilities[:, :-1].max(dim=-1)

        candidate_indices = torch.nonzero(
            objectness >= self.region_min_objectness,
            as_tuple=False,
        ).flatten()
        if candidate_indices.numel() == 0:
            return (
                np.zeros((height, width), dtype=np.uint16),
                np.zeros((height, width), dtype=np.float32),
                [],
            )

        candidate_masks = torch.nn.functional.interpolate(
            mask_probabilities[candidate_indices, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        candidate_areas = (candidate_masks >= self.region_mask_threshold).sum(dim=(1, 2))
        max_area = max(int(round(height * width * self.region_max_area_ratio)), 1)
        keep = (candidate_areas >= self.region_min_area) & (candidate_areas <= max_area)
        candidate_indices = candidate_indices[keep]
        candidate_masks = candidate_masks[keep]
        if candidate_indices.numel() == 0:
            return (
                np.zeros((height, width), dtype=np.uint16),
                np.zeros((height, width), dtype=np.float32),
                [],
            )

        ordering = torch.argsort(objectness[candidate_indices], descending=True)
        ordering = ordering[: self.region_max_queries]
        candidate_indices = candidate_indices[ordering]
        candidate_masks = candidate_masks[ordering]
        candidate_objectness = objectness[candidate_indices]

        pixel_scores = candidate_masks * candidate_objectness[:, None, None]
        winner_score, winner_index = pixel_scores.max(dim=0)
        winner_mask_probability = torch.gather(
            candidate_masks,
            0,
            winner_index[None],
        )[0]
        assigned = (
            (winner_score >= self.region_min_pixel_score)
            & (winner_mask_probability >= self.region_mask_threshold)
        )

        raw_region_id = winner_index.to(torch.int32) + 1
        raw_region_id = torch.where(assigned, raw_region_id, torch.zeros_like(raw_region_id))
        raw_region_id_np = raw_region_id.cpu().numpy()
        winner_score_np = torch.where(
            assigned,
            winner_score,
            torch.zeros_like(winner_score),
        ).to(torch.float32).cpu().numpy()

        region_id = np.zeros((height, width), dtype=np.uint16)
        region_confidence = np.zeros((height, width), dtype=np.float32)
        regions: list[dict[str, Any]] = []
        next_region_id = 1
        for candidate_position, query_index in enumerate(candidate_indices.tolist(), start=1):
            pixels = raw_region_id_np == candidate_position
            area = int(pixels.sum())
            if area < self.region_min_area:
                continue
            rows, columns = np.nonzero(pixels)
            compact_id = next_region_id
            next_region_id += 1
            region_id[pixels] = compact_id
            region_confidence[pixels] = winner_score_np[pixels]
            regions.append(
                {
                    "region_id": compact_id,
                    "query_index": int(query_index),
                    "area": area,
                    "bbox_xyxy": [
                        int(columns.min()),
                        int(rows.min()),
                        int(columns.max()) + 1,
                        int(rows.max()) + 1,
                    ],
                    "mean_region_confidence": float(winner_score_np[pixels].mean()),
                    "objectness": float(objectness[query_index].item()),
                    "diagnostic_ade20k_class": int(raw_class_id[query_index].item()),
                    "diagnostic_class_confidence": float(
                        raw_class_confidence[query_index].item()
                    ),
                }
            )
        return region_id, region_confidence, regions

    @torch.no_grad()
    def _segment(
        self,
        rgb: np.ndarray,
        *,
        include_regions: bool,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        list[dict[str, Any]],
    ]:
        rgb = _validate_rgb(rgb)
        self._ensure_loaded()
        height, width = rgb.shape[:2]

        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        amp_enabled = self.fp16 and self.device.startswith("cuda")
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = self._model(pixel_values=pixel_values)

        # Reproduce post_process_semantic_segmentation in float32, keeping the
        # per-pixel class-score stack so we can also emit a confidence map.
        class_queries = outputs.class_queries_logits.float()  # (1, Q, 151)
        mask_queries = outputs.masks_queries_logits.float()  # (1, Q, h, w)
        full_class_probs = class_queries.softmax(dim=-1)
        class_probs = full_class_probs[..., :-1]  # drop no-object
        mask_probs = mask_queries.sigmoid()
        segmentation = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        segmentation = torch.nn.functional.interpolate(
            segmentation,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]  # (150, H, W)

        scores, class_id = segmentation.max(dim=0)
        total = segmentation.sum(dim=0).clamp_min(1e-6)
        confidence = (scores / total).clamp(0.0, 1.0)
        region_id: np.ndarray | None = None
        region_confidence: np.ndarray | None = None
        regions: list[dict[str, Any]] = []
        if include_regions:
            region_id, region_confidence, regions = self._class_agnostic_regions(
                full_class_probs,
                mask_probs,
                height,
                width,
            )
        return (
            class_id.to(torch.int16).cpu().numpy(),
            confidence.to(torch.float32).cpu().numpy(),
            region_id,
            region_confidence,
            regions,
        )

    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        class_id, confidence, _region_id, _region_confidence, _regions = self._segment(
            rgb,
            include_regions=False,
        )
        return class_id, confidence

    def segment_with_regions(
        self,
        rgb: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        class_id, confidence, region_id, region_confidence, regions = self._segment(
            rgb,
            include_regions=True,
        )
        if region_id is None or region_confidence is None:
            raise AssertionError("Mask2Former region export was requested but not produced")
        return class_id, confidence, region_id, region_confidence, regions

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "fp16": self.fp16,
            "precision": (
                "amp_fp16" if self.fp16 and self.device.startswith("cuda") else "fp32"
            ),
            "num_classes": ADE20K_NUM_CLASSES,
            "class_agnostic_regions": {
                "available": True,
                "semantic_class_used_for_matching": False,
                "min_objectness": self.region_min_objectness,
                "mask_threshold": self.region_mask_threshold,
                "min_pixel_score": self.region_min_pixel_score,
                "min_area": self.region_min_area,
                "max_area_ratio": self.region_max_area_ratio,
                "max_queries": self.region_max_queries,
            },
        }


@dataclass
class DINOv3Backend:
    """Official DINOv3 ViT-7B/16 + ADE20K Mask2Former segmentor via torch.hub.

    Requires a local clone of https://github.com/facebookresearch/dinov3 plus
    the two downloaded checkpoints (backbone + segmentor head). The hub
    entrypoint is ``dinov3_vit7b16_ms`` (see dinov3/hub/segmentors.py):

        segmentor = torch.hub.load(repo_dir, "dinov3_vit7b16_ms",
                                   source="local",
                                   weights=<segmentor_ckpt>,
                                   backbone_weights=<backbone_ckpt>)
    """

    repo_dir: Path = Path(".")
    backbone_weights: Path = Path("dinov3_vit7b16_pretrain.pth")
    segmentor_weights: Path = Path("dinov3_vit7b16_ade20k_m2f_head.pth")
    hub_entry: str = "dinov3_vit7b16_ms"
    device: str = "cuda"
    # Slide-inference tiling; DINOv3 uses patch 16, crops must be multiples.
    crop_size: int = 896
    stride: int = 448
    name: str = "dinov3"

    def __post_init__(self) -> None:
        self._segmentor = None
        self._make_inference = None

    def _ensure_loaded(self) -> None:
        if self._segmentor is not None:
            return
        repo_dir = Path(self.repo_dir).resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo clone not found: {repo_dir}")
        for checkpoint in (self.backbone_weights, self.segmentor_weights):
            if not Path(checkpoint).exists():
                raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")

        segmentor = torch.hub.load(
            str(repo_dir),
            self.hub_entry,
            source="local",
            weights=str(self.segmentor_weights),
            backbone_weights=str(self.backbone_weights),
        )
        self._segmentor = segmentor.to(self.device).eval()

        # Prefer the repo's own slide-inference helper when available; fall
        # back to whole-image forward if the import path changes upstream.
        try:
            import sys

            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            from dinov3.eval.segmentation.inference import make_inference

            self._make_inference = make_inference
        except ImportError:
            self._make_inference = None

    @torch.no_grad()
    def _forward_probs(self, image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Return per-pixel class probabilities (150, H, W)."""
        from functools import partial

        if self._make_inference is not None:
            probs = self._make_inference(
                image,
                self._segmentor,
                inference_mode="slide",
                decoder_head_type="m2f",
                rescale_to=(height, width),
                n_output_channels=ADE20K_NUM_CLASSES,
                crop_size=(self.crop_size, self.crop_size),
                stride=(self.stride, self.stride),
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )
            return probs[0]

        logits = self._segmentor(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        logits = torch.nn.functional.interpolate(
            logits.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return logits.softmax(dim=1)[0]

    @torch.no_grad()
    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = _validate_rgb(rgb)
        self._ensure_loaded()
        height, width = rgb.shape[:2]

        # ImageNet normalization, matching the dinov3 segmentation notebook.
        image = torch.from_numpy(rgb).to(self.device).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        image = ((image - mean) / std).unsqueeze(0)

        probs = self._forward_probs(image, height, width)  # (150, H, W)
        confidence, class_id = probs.max(dim=0)
        return (
            class_id.to(torch.int16).cpu().numpy(),
            confidence.to(torch.float32).clamp(0.0, 1.0).cpu().numpy(),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "repo_dir": str(self.repo_dir),
            "backbone_weights": str(self.backbone_weights),
            "segmentor_weights": str(self.segmentor_weights),
            "hub_entry": self.hub_entry,
            "crop_size": self.crop_size,
            "stride": self.stride,
            "num_classes": ADE20K_NUM_CLASSES,
        }


BACKEND_NAMES = ("mask2former", "dinov3")


def build_backend(name: str, args: Any) -> DenseSegBackend:
    """Build a backend from parsed argparse args (see segment_views_semantic)."""
    if name == "mask2former":
        return Mask2FormerBackend(
            model_id=args.mask2former_model,
            device=args.device,
            fp16=not args.no_fp16,
            region_min_objectness=args.region_min_objectness,
            region_mask_threshold=args.region_mask_threshold,
            region_min_pixel_score=args.region_min_pixel_score,
            region_min_area=args.region_min_area,
            region_max_area_ratio=args.region_max_area_ratio,
            region_max_queries=args.region_max_queries,
        )
    if name == "dinov3":
        # Path("") silently becomes "." and passes exists() checks, so reject
        # unset paths up front with an actionable message.
        missing = [
            flag
            for flag, value in (
                ("--dinov3-repo", args.dinov3_repo),
                ("--dinov3-backbone-weights", args.dinov3_backbone_weights),
                ("--dinov3-segmentor-weights", args.dinov3_segmentor_weights),
            )
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(f"dinov3 backend requires: {', '.join(missing)}")
        return DINOv3Backend(
            repo_dir=Path(args.dinov3_repo),
            backbone_weights=Path(args.dinov3_backbone_weights),
            segmentor_weights=Path(args.dinov3_segmentor_weights),
            hub_entry=args.dinov3_hub_entry,
            device=args.device,
            crop_size=args.dinov3_crop_size,
            stride=args.dinov3_stride,
        )
    raise ValueError(f"Unknown backend {name!r}; expected one of {BACKEND_NAMES}")
