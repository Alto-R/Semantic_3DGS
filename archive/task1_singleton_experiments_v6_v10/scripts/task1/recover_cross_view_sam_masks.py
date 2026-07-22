#!/usr/bin/env python3
"""Recover strong singleton Grounded-SAM proposals through cross-view SAM prompts.

The initial GroundingDINO pass remains the semantic source. A strong one-view
mask is lifted to 3D, projected into nearby selected views, and used only as a
box prompt for SAM. Semantic fusion later decides whether at least two of those
verification masks agree with the original proposal in 3D.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cluster_semantic_flashsplat_proposals import (
    SemanticProposal,
    cluster_class_proposals,
    load_proposals,
    load_stuff_classes,
)
from ply_utils import vertex_data_memmap


CROSS_VIEW_SOURCE = "cross_view_sam_verification"


@dataclass(frozen=True)
class ProjectionPrompt:
    seed_proposal_id: int
    seed_key: str
    class_name: str
    target_frame_file: str
    target_camera_index: int
    bbox_xyxy: tuple[float, float, float, float]
    projected_xy: np.ndarray
    projected_seed_positions: np.ndarray
    projected_gaussian_count: int
    projected_fraction: float
    baseline_depth_ratio: float
    seed_grounding_score: float
    seed_sam_score: float
    seed_frame_file: str


def normalize_class_name(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_") or "unknown"


def load_mask_stack(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        masks = data["masks"].astype(bool)
    if masks.ndim != 3:
        raise ValueError(f"{path} masks must have shape (K,H,W)")
    return masks


def load_cameras(model_path: Path) -> list[dict[str, Any]]:
    path = model_path / "cameras.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def point_cloud_path(model_path: Path, iteration: int) -> Path:
    path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def thing_classes_from_config(path: Path) -> set[str]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("classes", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Class config must contain a classes list")
    return {
        normalize_class_name(
            item.get("class", item.get("name", "")) if isinstance(item, dict) else item
        )
        for item in items
        if not isinstance(item, dict)
        or str(item.get("type", "thing")).lower() == "thing"
    }


def frame_mask_path(
    manifest_path: Path,
    ground_output_dir: Path,
    frame: dict[str, Any],
) -> Path:
    if str(frame.get("mask_path", "")).strip():
        path = Path(str(frame["mask_path"]))
        return path if path.is_absolute() else manifest_path.parent / path
    return ground_output_dir / "mask_stacks" / str(frame["mask_file"])


def rgb_directory(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    configured = str(manifest.get("output_directories", {}).get("rgb_renders", "")).strip()
    if configured:
        path = Path(configured)
        if path.is_dir():
            return path
    fallback = manifest_path.parent / "rgb_renders"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(fallback)


def camera_to_world_matrix(camera: dict[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(camera["rotation"], dtype=np.float64)
    matrix[:3, 3] = np.asarray(camera["position"], dtype=np.float64)
    return matrix


def project_points(
    points: np.ndarray,
    camera: dict[str, Any],
    width: int,
    height: int,
    min_depth: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points using the same c2w interpretation as FlashSplat."""
    w2c = np.linalg.inv(camera_to_world_matrix(camera))
    camera_points = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera_points[:, 2]
    valid = depth > min_depth
    xy = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        scale_x = width / float(max(int(camera["width"]), 1))
        scale_y = height / float(max(int(camera["height"]), 1))
        focal_x = float(camera["fx"]) * scale_x
        focal_y = float(camera["fy"]) * scale_y
        xy[valid, 0] = focal_x * camera_points[valid, 0] / depth[valid] + width * 0.5
        xy[valid, 1] = focal_y * camera_points[valid, 1] / depth[valid] + height * 0.5
    inside = (
        valid
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < height)
    )
    return xy, depth, inside


def robust_projection_box(
    xy: np.ndarray,
    width: int,
    height: int,
    padding_ratio: float,
    quantile: float = 0.02,
) -> tuple[float, float, float, float] | None:
    if xy.shape[0] == 0:
        return None
    low = np.quantile(xy, quantile, axis=0)
    high = np.quantile(xy, 1.0 - quantile, axis=0)
    extent = np.maximum(high - low, 1.0)
    low -= extent * padding_ratio
    high += extent * padding_ratio
    x1 = float(np.clip(low[0], 0.0, max(width - 1, 0)))
    y1 = float(np.clip(low[1], 0.0, max(height - 1, 0)))
    x2 = float(np.clip(high[0], 0.0, max(width - 1, 0)))
    y2 = float(np.clip(high[1], 0.0, max(height - 1, 0)))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return x1, y1, x2, y2


def projected_mask_coverage(mask: np.ndarray, xy: np.ndarray) -> float:
    membership = projected_mask_membership(mask, xy)
    return float(membership.mean()) if membership.size else 0.0


def projected_mask_membership(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Return one mask-membership value for every projected seed point."""
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    height, width = mask.shape
    pixels = np.rint(xy).astype(np.int64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return mask[pixels[:, 1], pixels[:, 0]].astype(bool, copy=False)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(left, right).sum())
    return intersection / float(max(union, 1))


def mask_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    inner_area = int(inner.sum())
    if inner_area == 0:
        return 0.0
    return int(np.logical_and(inner, outer).sum()) / float(inner_area)


def nested_competing_thing(
    seed_mask: np.ndarray,
    seed_class: str,
    masks: np.ndarray,
    mask_metadata: list[dict[str, Any]],
    thing_classes: set[str],
    minimum_outer_area_ratio: float,
) -> dict[str, Any]:
    seed_area = int(seed_mask.sum())
    best = {"containment": 0.0, "class": "", "mask_index": -1, "area_ratio": 0.0}
    for index, outer in enumerate(masks):
        if index >= len(mask_metadata):
            continue
        outer_class = normalize_class_name(
            mask_metadata[index].get("class_name", mask_metadata[index].get("class", ""))
        )
        if outer_class == seed_class or outer_class not in thing_classes:
            continue
        outer_area = int(outer.sum())
        area_ratio = outer_area / float(max(seed_area, 1))
        if area_ratio < minimum_outer_area_ratio:
            continue
        containment = mask_containment(seed_mask, outer)
        if containment > float(best["containment"]):
            best = {
                "containment": containment,
                "class": outer_class,
                "mask_index": index,
                "area_ratio": area_ratio,
            }
    return best


def proposal_sam_score(proposal: SemanticProposal) -> float:
    return float(
        proposal.metadata.get(
            "sam_score",
            proposal.metadata.get("predicted_iou", 0.0),
        )
    )


def proposal_grounding_score(proposal: SemanticProposal) -> float:
    return float(
        proposal.metadata.get(
            "grounding_score",
            proposal.metadata.get("confidence", 0.0),
        )
    )


def singleton_proposals(
    proposals: list[SemanticProposal],
    stuff_classes: set[str],
    merge_iou: float,
    containment_threshold: float,
) -> list[SemanticProposal]:
    groups = cluster_class_proposals(
        proposals,
        stuff_classes,
        merge_iou,
        containment_threshold,
    )
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    singletons = [
        proposals_by_id[group.proposal_ids[0]]
        for group in groups
        if not group.is_stuff and group.proposal_count == 1
    ]
    return sorted(
        singletons,
        key=lambda proposal: (
            proposal_sam_score(proposal),
            proposal.gaussian_count,
            proposal_grounding_score(proposal),
        ),
        reverse=True,
    )


def candidate_view_prompts(
    proposal: SemanticProposal,
    points: np.ndarray,
    source_camera: dict[str, Any],
    frame_records: list[dict[str, Any]],
    cameras: list[dict[str, Any]],
    rgb_dir: Path,
    min_projected_gaussians: int,
    min_projected_fraction: float,
    max_baseline_depth_ratio: float,
    box_padding_ratio: float,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    max_target_views: int,
) -> list[ProjectionPrompt]:
    source_xy, source_depth, source_inside = project_points(
        points,
        source_camera,
        int(source_camera["width"]),
        int(source_camera["height"]),
    )
    del source_xy
    visible_source_depth = source_depth[source_inside]
    median_depth = float(np.median(visible_source_depth)) if visible_source_depth.size else 0.0
    source_position = np.asarray(source_camera["position"], dtype=np.float64)
    source_frame = str(proposal.metadata.get("frame_file", ""))
    seed_key = f"{source_frame}:{int(proposal.metadata.get('mask_index', -1))}"
    prompts: list[tuple[tuple[float, float, float], ProjectionPrompt]] = []
    for frame in frame_records:
        target_frame = str(frame["file"])
        if target_frame == source_frame:
            continue
        target_camera_index = int(frame["camera_index"])
        target_camera = cameras[target_camera_index]
        rgb_path = rgb_dir / target_frame
        if not rgb_path.exists():
            continue
        with Image.open(rgb_path) as image:
            width, height = image.size
        xy, _depth, inside = project_points(points, target_camera, width, height)
        projected_seed_positions = np.flatnonzero(inside)
        projected_xy = xy[inside]
        projected_count = int(projected_xy.shape[0])
        projected_fraction = projected_count / float(max(points.shape[0], 1))
        if projected_count < min_projected_gaussians:
            continue
        if projected_fraction < min_projected_fraction:
            continue
        box = robust_projection_box(
            projected_xy,
            width,
            height,
            box_padding_ratio,
        )
        if box is None:
            continue
        box_area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / float(width * height)
        if box_area_ratio < min_box_area_ratio or box_area_ratio > max_box_area_ratio:
            continue
        target_position = np.asarray(target_camera["position"], dtype=np.float64)
        baseline = float(np.linalg.norm(target_position - source_position))
        baseline_depth_ratio = baseline / float(max(median_depth, 1.0e-6))
        if median_depth <= 0.0 or baseline_depth_ratio > max_baseline_depth_ratio:
            continue
        prompt = ProjectionPrompt(
            seed_proposal_id=proposal.proposal_id,
            seed_key=seed_key,
            class_name=proposal.class_name,
            target_frame_file=target_frame,
            target_camera_index=target_camera_index,
            bbox_xyxy=box,
            projected_xy=projected_xy,
            projected_seed_positions=projected_seed_positions,
            projected_gaussian_count=projected_count,
            projected_fraction=projected_fraction,
            baseline_depth_ratio=baseline_depth_ratio,
            seed_grounding_score=proposal_grounding_score(proposal),
            seed_sam_score=proposal_sam_score(proposal),
            seed_frame_file=source_frame,
        )
        rank = (baseline_depth_ratio, -projected_fraction, -box_area_ratio)
        prompts.append((rank, prompt))
    prompts.sort(key=lambda item: item[0])
    return [prompt for _rank, prompt in prompts[:max_target_views]]


def relative_path(path: Path, parent: Path) -> str:
    try:
        return str(path.relative_to(parent))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--ground-output-dir", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--class-config", required=True, type=Path)
    parser.add_argument("--segment-anything-root", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--input-manifest-name", default="grounded_sam_manifest.json")
    parser.add_argument(
        "--output-manifest-name",
        default="grounded_sam_cross_view_manifest.json",
    )
    parser.add_argument("--min-proposal-gaussians", default=500, type=int)
    parser.add_argument("--min-seed-gaussians", default=1500, type=int)
    parser.add_argument("--min-seed-sam-score", default=0.90, type=float)
    parser.add_argument("--min-seed-mask-area-ratio", default=0.02, type=float)
    parser.add_argument("--min-outer-area-ratio", default=1.50, type=float)
    parser.add_argument("--max-nested-thing-containment", default=0.80, type=float)
    parser.add_argument("--merge-iou", default=0.35, type=float)
    parser.add_argument("--containment-threshold", default=0.70, type=float)
    parser.add_argument("--min-projected-gaussians", default=250, type=int)
    parser.add_argument("--min-projected-fraction", default=0.05, type=float)
    parser.add_argument("--max-baseline-depth-ratio", default=0.75, type=float)
    parser.add_argument("--box-padding-ratio", default=0.05, type=float)
    parser.add_argument("--min-box-area-ratio", default=0.005, type=float)
    parser.add_argument("--max-box-area-ratio", default=0.80, type=float)
    parser.add_argument("--max-target-views", default=4, type=int)
    parser.add_argument("--min-verification-views", default=2, type=int)
    parser.add_argument("--min-verification-sam-score", default=0.90, type=float)
    parser.add_argument("--min-projection-coverage", default=0.60, type=float)
    parser.add_argument("--min-verification-mask-area", default=100, type=int)
    parser.add_argument("--max-verification-mask-area-ratio", default=0.80, type=float)
    parser.add_argument("--duplicate-mask-iou", default=0.85, type=float)
    parser.add_argument("--max-seeds", default=24, type=int)
    parser.add_argument("--max-masks-per-view", default=32, type=int)
    args = parser.parse_args()

    if args.min_verification_views < 2:
        raise ValueError("--min-verification-views must be at least 2")
    if args.max_target_views < args.min_verification_views:
        raise ValueError("--max-target-views must be at least --min-verification-views")

    # Import CUDA-dependent helpers only for actual recovery execution so the
    # geometry and policy helpers remain unit-testable without Torch installed.
    import torch

    from generate_grounded_sam_masks import load_sam_predictor, run_sam_for_boxes

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-view SAM recovery")

    input_manifest_path = args.ground_output_dir / args.input_manifest_name
    proposal_manifest_path = args.proposal_dir / "proposal_manifest.json"
    support_dir = args.proposal_dir / "proposal_supports"
    if not input_manifest_path.exists():
        raise FileNotFoundError(input_manifest_path)
    if not proposal_manifest_path.exists():
        raise FileNotFoundError(proposal_manifest_path)

    source_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    output_manifest = copy.deepcopy(source_manifest)
    frames = output_manifest.get("frames", [])
    frames_by_file = {str(frame["file"]): frame for frame in frames}
    cameras = load_cameras(args.model_path)
    rgb_dir = rgb_directory(input_manifest_path, source_manifest)
    stuff_classes = load_stuff_classes(args.class_config, "")
    thing_classes = thing_classes_from_config(args.class_config)
    proposals = load_proposals(
        proposal_manifest_path,
        support_dir,
        args.min_proposal_gaussians,
        0,
        True,
    )
    candidates = singleton_proposals(
        proposals,
        stuff_classes,
        args.merge_iou,
        args.containment_threshold,
    )

    ply_path = point_cloud_path(args.model_path, args.iteration)
    _header, vertex_data = vertex_data_memmap(ply_path)
    xyz = np.stack(
        (
            np.asarray(vertex_data["x"]),
            np.asarray(vertex_data["y"]),
            np.asarray(vertex_data["z"]),
        ),
        axis=1,
    )

    attempt_reports: list[dict[str, Any]] = []
    prompts: list[ProjectionPrompt] = []
    selected_seed_ids: set[int] = set()
    for proposal in candidates:
        report: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "class": proposal.class_name,
            "frame_file": str(proposal.metadata.get("frame_file", "")),
            "mask_index": int(proposal.metadata.get("mask_index", -1)),
            "gaussian_count": proposal.gaussian_count,
            "sam_score": proposal_sam_score(proposal),
            "grounding_score": proposal_grounding_score(proposal),
            "status": "rejected",
            "reasons": [],
        }
        source_frame = frames_by_file.get(report["frame_file"])
        if source_frame is None:
            report["reasons"].append("source_frame_missing")
            attempt_reports.append(report)
            continue
        mask_index = int(report["mask_index"])
        source_masks = load_mask_stack(
            frame_mask_path(input_manifest_path, args.ground_output_dir, source_frame)
        )
        if mask_index < 0 or mask_index >= source_masks.shape[0]:
            report["reasons"].append("source_mask_missing")
            attempt_reports.append(report)
            continue
        seed_mask = source_masks[mask_index]
        image_area = int(seed_mask.shape[0] * seed_mask.shape[1])
        mask_area_ratio = int(seed_mask.sum()) / float(max(image_area, 1))
        report["mask_area_ratio"] = mask_area_ratio
        nested = nested_competing_thing(
            seed_mask,
            proposal.class_name,
            source_masks,
            list(source_frame.get("masks", [])),
            thing_classes,
            args.min_outer_area_ratio,
        )
        report["nested_competing_thing"] = nested
        if proposal.gaussian_count < args.min_seed_gaussians:
            report["reasons"].append(f"gaussian_count<{args.min_seed_gaussians}")
        if proposal_sam_score(proposal) < args.min_seed_sam_score:
            report["reasons"].append(f"sam_score<{args.min_seed_sam_score}")
        if mask_area_ratio < args.min_seed_mask_area_ratio:
            report["reasons"].append(
                f"mask_area_ratio<{args.min_seed_mask_area_ratio}"
            )
        if float(nested["containment"]) > args.max_nested_thing_containment:
            report["reasons"].append(
                "nested_competing_thing_containment>"
                f"{args.max_nested_thing_containment}"
            )
        if report["reasons"]:
            attempt_reports.append(report)
            continue
        if len(selected_seed_ids) >= args.max_seeds:
            report["reasons"].append(f"max_seeds>{args.max_seeds}")
            attempt_reports.append(report)
            continue

        support_points = xyz[proposal.indices]
        source_camera = cameras[int(source_frame["camera_index"])]
        seed_prompts = candidate_view_prompts(
            proposal,
            support_points,
            source_camera,
            frames,
            cameras,
            rgb_dir,
            args.min_projected_gaussians,
            args.min_projected_fraction,
            args.max_baseline_depth_ratio,
            args.box_padding_ratio,
            args.min_box_area_ratio,
            args.max_box_area_ratio,
            args.max_target_views,
        )
        report["candidate_view_count"] = len(seed_prompts)
        report["candidate_views"] = [
            {
                "frame_file": prompt.target_frame_file,
                "camera_index": prompt.target_camera_index,
                "projected_gaussian_count": prompt.projected_gaussian_count,
                "projected_fraction": prompt.projected_fraction,
                "baseline_depth_ratio": prompt.baseline_depth_ratio,
                "bbox_xyxy": list(prompt.bbox_xyxy),
            }
            for prompt in seed_prompts
        ]
        if len(seed_prompts) < args.min_verification_views:
            report["reasons"].append(
                f"candidate_view_count<{args.min_verification_views}"
            )
            attempt_reports.append(report)
            continue
        report["status"] = "prompted"
        selected_seed_ids.add(proposal.proposal_id)
        prompts.extend(seed_prompts)
        attempt_reports.append(report)

    sam_predictor = load_sam_predictor(
        args.segment_anything_root,
        args.sam_checkpoint,
        args.sam_arch,
        args.device,
    )
    prompts_by_frame: dict[str, list[ProjectionPrompt]] = defaultdict(list)
    for prompt in prompts:
        prompts_by_frame[prompt.target_frame_file].append(prompt)

    accepted_by_seed: dict[str, list[tuple[ProjectionPrompt, np.ndarray, float, float]]] = (
        defaultdict(list)
    )
    for frame_file, frame_prompts in prompts_by_frame.items():
        rgb = np.asarray(Image.open(rgb_dir / frame_file).convert("RGB"), dtype=np.uint8)
        boxes = torch.tensor(
            [prompt.bbox_xyxy for prompt in frame_prompts],
            dtype=torch.float32,
            device=args.device,
        )
        masks, sam_scores = run_sam_for_boxes(
            sam_predictor,
            rgb,
            boxes,
            args.device,
        )
        target_frame = frames_by_file[frame_file]
        existing_masks = load_mask_stack(
            frame_mask_path(input_manifest_path, args.ground_output_dir, target_frame)
        )
        existing_metadata = list(target_frame.get("masks", []))
        for index, prompt in enumerate(frame_prompts):
            mask = masks[index]
            sam_score = float(sam_scores[index]) if index < len(sam_scores) else 0.0
            area = int(mask.sum())
            area_ratio = area / float(max(mask.shape[0] * mask.shape[1], 1))
            coverage = projected_mask_coverage(mask, prompt.projected_xy)
            duplicate = any(
                normalize_class_name(
                    existing_metadata[existing_index].get(
                        "class_name",
                        existing_metadata[existing_index].get("class", ""),
                    )
                )
                == prompt.class_name
                and mask_iou(mask, existing_mask) >= args.duplicate_mask_iou
                for existing_index, existing_mask in enumerate(existing_masks)
                if existing_index < len(existing_metadata)
            )
            if sam_score < args.min_verification_sam_score:
                continue
            if area < args.min_verification_mask_area:
                continue
            if area_ratio > args.max_verification_mask_area_ratio:
                continue
            if coverage < args.min_projection_coverage:
                continue
            if duplicate:
                continue
            accepted_by_seed[prompt.seed_key].append(
                (prompt, mask, sam_score, coverage)
            )

    committed_by_frame: dict[str, list[tuple[ProjectionPrompt, np.ndarray, float, float]]] = (
        defaultdict(list)
    )
    recovered_seed_keys: set[str] = set()
    for seed_key, accepted in accepted_by_seed.items():
        distinct_frames = {item[0].target_frame_file for item in accepted}
        if len(distinct_frames) < args.min_verification_views:
            continue
        recovered_seed_keys.add(seed_key)
        for item in accepted:
            committed_by_frame[item[0].target_frame_file].append(item)

    recovery_mask_dir = args.ground_output_dir / "cross_view_mask_stacks"
    recovery_mask_dir.mkdir(parents=True, exist_ok=True)
    added_mask_count = 0
    for frame_file, additions in committed_by_frame.items():
        frame = frames_by_file[frame_file]
        original_masks = load_mask_stack(
            frame_mask_path(input_manifest_path, args.ground_output_dir, frame)
        )
        if args.max_masks_per_view > 0:
            capacity = max(0, args.max_masks_per_view - original_masks.shape[0])
            additions = additions[:capacity]
        if not additions:
            continue
        new_masks = [item[1] for item in additions]
        combined = np.concatenate(
            [original_masks, np.stack(new_masks, axis=0)],
            axis=0,
        )
        output_mask_path = recovery_mask_dir / str(frame["mask_file"])
        np.savez_compressed(output_mask_path, masks=combined.astype(np.uint8))
        frame["mask_path"] = relative_path(output_mask_path, input_manifest_path.parent)
        for prompt, mask, sam_score, coverage in additions:
            x1, y1, x2, y2 = prompt.bbox_xyxy
            mask_index = len(frame.get("masks", []))
            frame.setdefault("masks", []).append(
                {
                    "mask_index": mask_index,
                    "source": CROSS_VIEW_SOURCE,
                    "class": prompt.class_name,
                    "class_name": prompt.class_name,
                    "phrase": prompt.class_name.replace("_", " "),
                    "confidence": prompt.seed_grounding_score,
                    "grounding_score": prompt.seed_grounding_score,
                    "sam_score": sam_score,
                    "area": int(mask.sum()),
                    "bbox": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2 - x1)),
                        int(round(y2 - y1)),
                    ],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "cross_view_seed_proposal_id": prompt.seed_proposal_id,
                    "cross_view_seed_key": prompt.seed_key,
                    "cross_view_seed_frame": prompt.seed_frame_file,
                    "projected_gaussian_count": prompt.projected_gaussian_count,
                    "projected_fraction": prompt.projected_fraction,
                    "projection_coverage": coverage,
                    "baseline_depth_ratio": prompt.baseline_depth_ratio,
                }
            )
            added_mask_count += 1
        frame["kept_mask_count"] = len(frame.get("masks", []))

    for report in attempt_reports:
        seed_key = f"{report['frame_file']}:{report['mask_index']}"
        if seed_key in recovered_seed_keys:
            report["status"] = "recovered"
            report["verification_view_count"] = len(
                {
                    item[0].target_frame_file
                    for item in accepted_by_seed.get(seed_key, [])
                }
            )
        elif report["status"] == "prompted":
            report["status"] = "rejected"
            report["reasons"].append(
                f"verified_view_count<{args.min_verification_views}"
            )

    output_manifest["source"] = "groundingdino_sam_with_cross_view_verification"
    output_manifest["source_manifest"] = str(input_manifest_path)
    output_manifest["cross_view_recovery"] = {
        "parameters": {
            "min_seed_gaussians": args.min_seed_gaussians,
            "min_seed_sam_score": args.min_seed_sam_score,
            "min_seed_mask_area_ratio": args.min_seed_mask_area_ratio,
            "min_outer_area_ratio": args.min_outer_area_ratio,
            "max_nested_thing_containment": args.max_nested_thing_containment,
            "min_projected_gaussians": args.min_projected_gaussians,
            "min_projected_fraction": args.min_projected_fraction,
            "max_baseline_depth_ratio": args.max_baseline_depth_ratio,
            "min_verification_views": args.min_verification_views,
            "min_verification_sam_score": args.min_verification_sam_score,
            "min_projection_coverage": args.min_projection_coverage,
            "max_seeds": args.max_seeds,
        },
        "singleton_candidate_count": len(candidates),
        "prompted_seed_count": len(selected_seed_ids),
        "recovered_seed_count": len(recovered_seed_keys),
        "added_mask_count": added_mask_count,
        "seeds": attempt_reports,
    }
    output_manifest_path = args.ground_output_dir / args.output_manifest_name
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"wrote {output_manifest_path}: singleton_candidates={len(candidates)} "
        f"recovered_seeds={len(recovered_seed_keys)} added_masks={added_mask_count}"
    )


if __name__ == "__main__":
    main()
