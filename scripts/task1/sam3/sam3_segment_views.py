#!/usr/bin/env python3
"""S1 driver: run SAM3 concept segmentation over rendered camera views.

The mock backend keeps the driver testable without a GPU or checkpoint. The
transformers backend is cluster-only and imported lazily; its exact pinned
revision is recorded in docs/EXTERNAL_REPOS.md at cluster setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.sam3.sam3_backend import (
    InstanceMask,
    MockSam3Backend,
    Sam3Backend,
)
from scripts.task1.sam3.segment_views_core import (
    build_masks_manifest,
    segment_view,
    write_view_masks,
)
from scripts.task1.sam3.vocabulary import load_vocabulary


class Sam3TransformersBackend:
    """Hugging Face transformers SAM3 backend (cluster only)."""

    def __init__(self, model_id: str, revision: str, device: str = "cuda"):
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:  # pragma: no cover - cluster environment
            raise RuntimeError(
                "the transformers SAM3 backend requires the cluster "
                "environment with torch and transformers installed"
            ) from exc
        self._torch = torch
        self._device = device
        self._processor = Sam3Processor.from_pretrained(model_id, revision=revision)
        self._model = (
            Sam3Model.from_pretrained(
                model_id, revision=revision, torch_dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )

    def segment(  # pragma: no cover - exercised only on the cluster
        self, image: np.ndarray, phrase: str
    ) -> list[InstanceMask]:
        torch = self._torch
        inputs = self._processor(images=image, text=phrase, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            target_sizes=[image.shape[:2]],
        )[0]
        masks: list[InstanceMask] = []
        for mask, score in zip(results["masks"], results["scores"]):
            masks.append(
                InstanceMask(
                    mask=mask.cpu().numpy().astype(bool),
                    score=float(score),
                )
            )
        return masks


def _load_mock_backend(boxes_path: Path) -> MockSam3Backend:
    payload = json.loads(Path(boxes_path).read_text(encoding="utf-8"))
    boxes: dict[str, list[tuple[int, int, int, int, float]]] = {}
    for phrase, rectangles in payload.items():
        boxes[str(phrase)] = [
            (int(y0), int(y1), int(x0), int(x1), float(score))
            for y0, y1, x0, x1, score in rectangles
        ]
    return MockSam3Backend(boxes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("transformers", "mock"), required=True)
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mock-boxes", type=Path)
    parser.add_argument("--min-score", default=0.4, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    vocabulary = load_vocabulary(args.vocabulary)
    manifest_path = args.output_dir / "sam3_masks_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(manifest_path)
    masks_dir = args.output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(Path(args.rgb_dir).glob("*.png"))
    if not images:
        raise FileNotFoundError(f"no PNG views found in {args.rgb_dir}")

    backend: Sam3Backend
    if args.backend == "mock":
        if args.mock_boxes is None:
            raise ValueError("--mock-boxes is required with the mock backend")
        backend = _load_mock_backend(args.mock_boxes)
        model_id, model_revision = "mock", "mock"
    else:
        backend = Sam3TransformersBackend(
            args.model_id, args.model_revision, args.device
        )
        model_id, model_revision = args.model_id, args.model_revision

    from PIL import Image

    frames: list[dict[str, Any]] = []
    for image_path in images:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        view_masks = segment_view(image, vocabulary, backend, args.min_score)
        fragment = write_view_masks(masks_dir / f"{image_path.stem}.npz", view_masks)
        frames.append({"file": image_path.name, **fragment})
        print(f"segmented {image_path.name}: {fragment['mask_count']} masks")

    manifest = build_masks_manifest(
        scene=vocabulary.scene,
        vocabulary=vocabulary,
        model_id=model_id,
        model_revision=model_revision,
        min_score=args.min_score,
        frames=frames,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
