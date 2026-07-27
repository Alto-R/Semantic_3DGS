#!/usr/bin/env python3
"""Cross-validate core-first proposals with dense per-Gaussian DINOv3 votes.

The upstream core-first audit assigns one semantic identity to a complete 3D
core.  This downstream, permanently report-only gate checks that identity at
each currently-unlabeled Gaussian using independent dense camera evidence.
Only Gaussians whose global dense winner matches the proposed class, whose
per-camera winners form a strict majority, and whose confidence/margins pass
global class-neutral gates can reach the final spatial re-split.

Preferred-v2 labels are read-only and every nonzero preferred label is
excluded before semantic validation.  No semantic label array, label map,
project-class array, or PLY is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.lift_confident_dense_view_votes import (
    CONTRACT as VOTE_CONTRACT,
    SOURCE as VOTE_SOURCE,
)
from scripts.task1.dinov3.propagate_dense_labels_from_region_seeds import (
    adaptive_voxel_size,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)


SOURCE = "dinov3_core_first_dense_cross_validation_audit"
CONTRACT = "report_only_dinov3_core_first_dense_cross_validation_v1"
CORE_CONTRACT = "report_only_dinov3_core_first_semantic_identity_v1"

REASON_ACCEPTED_BEFORE_SPATIAL = 0
REASON_UNOBSERVED = 1
REASON_INSUFFICIENT_RELIABLE_CAMERAS = 2
REASON_GLOBAL_WINNER_MISMATCH = 3
REASON_NO_STRICT_CAMERA_MAJORITY = 4
REASON_LOW_GLOBAL_WINNER_SHARE = 5
REASON_LOW_GLOBAL_WINNER_MARGIN = 6
REASON_BOUNDARY_COMPETITOR = 7
REASON_REJECTED_SPATIAL_GATE = 8

REASON_NAMES = {
    REASON_ACCEPTED_BEFORE_SPATIAL: "accepted_before_spatial",
    REASON_UNOBSERVED: "unobserved",
    REASON_INSUFFICIENT_RELIABLE_CAMERAS: "insufficient_reliable_cameras",
    REASON_GLOBAL_WINNER_MISMATCH: "global_winner_mismatch",
    REASON_NO_STRICT_CAMERA_MAJORITY: "no_strict_camera_majority",
    REASON_LOW_GLOBAL_WINNER_SHARE: "low_global_winner_share",
    REASON_LOW_GLOBAL_WINNER_MARGIN: "low_global_winner_margin",
    REASON_BOUNDARY_COMPETITOR: "boundary_competitor",
    REASON_REJECTED_SPATIAL_GATE: "rejected_spatial_gate",
}


@dataclass(frozen=True)
class DenseCrossValidationThresholds:
    min_reliable_cameras: int = 3
    min_camera_accepted_fraction: float = 0.50
    min_camera_winner_share: float = 0.50
    min_camera_winner_margin: float = 0.10
    min_global_winner_share: float = 0.55
    min_global_winner_margin: float = 0.10
    min_boundary_margin: float = 0.10
    min_spatial_component_gaussians: int = 500
    min_spatial_component_cameras: int = 3
    voxel_scale_multiplier: float = 4.0
    min_voxel_size: float = 0.01
    max_voxel_size: float = 0.20

    def validate(self) -> None:
        if self.min_reliable_cameras < 2:
            raise ValueError("min_reliable_cameras must be at least two")
        for name in (
            "min_camera_accepted_fraction",
            "min_camera_winner_share",
            "min_camera_winner_margin",
            "min_global_winner_share",
            "min_global_winner_margin",
            "min_boundary_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_spatial_component_gaussians < 1:
            raise ValueError("min_spatial_component_gaussians must be positive")
        if self.min_spatial_component_cameras < 2:
            raise ValueError(
                "min_spatial_component_cameras must be at least two"
            )
        if self.voxel_scale_multiplier <= 0.0:
            raise ValueError("voxel_scale_multiplier must be positive")
        if self.min_voxel_size <= 0.0:
            raise ValueError("min_voxel_size must be positive")
        if self.max_voxel_size < 0.0:
            raise ValueError("max_voxel_size must be non-negative")
        if 0.0 < self.max_voxel_size < self.min_voxel_size:
            raise ValueError(
                "max_voxel_size must be zero or at least min_voxel_size"
            )


@dataclass(frozen=True)
class CoreCandidate:
    component_id: int
    project_id: int
    class_name: str
    indices: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0}
    quantiles = np.quantile(
        finite,
        [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
    )
    names = (
        "min",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    )
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        **{name: float(value) for name, value in zip(names, quantiles)},
    }


def load_core_candidates(
    report: dict[str, Any],
    supports: Mapping[str, np.ndarray],
    vertex_count: int,
) -> list[CoreCandidate]:
    if report.get("contract") != CORE_CONTRACT:
        raise ValueError("core-first report has the wrong contract")
    if not bool(report.get("report_only")):
        raise ValueError("core-first source must be report-only")
    if int(report.get("vertex_count", -1)) != vertex_count:
        raise ValueError("core-first vertex count differs from preferred labels")

    candidates: list[CoreCandidate] = []
    seen_ids: set[int] = set()
    seen_indices = np.zeros((vertex_count,), dtype=bool)
    for raw in report.get("components", []):
        if not bool(raw.get("accepted")):
            continue
        component_id = int(raw["component_id"])
        if component_id < 1 or component_id in seen_ids:
            raise ValueError("accepted core component IDs are invalid")
        seen_ids.add(component_id)
        key = f"component_{component_id:06d}_indices"
        if key not in supports:
            raise ValueError(f"missing support for core {component_id}")
        indices = np.asarray(supports[key], dtype=np.uint32)
        if indices.ndim != 1:
            raise ValueError("core support indices must be one-dimensional")
        if indices.size and (
            int(indices[-1]) >= vertex_count
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("core support indices are invalid")
        if indices.size and np.any(seen_indices[indices]):
            raise ValueError("post-ownership core supports overlap")
        seen_indices[indices] = True
        candidates.append(
            CoreCandidate(
                component_id=component_id,
                project_id=int(raw["project_id"]),
                class_name=str(raw["class"]),
                indices=indices,
            )
        )
    return candidates


def validate_vote_manifest(
    manifest: dict[str, Any],
    *,
    vertex_count: int,
    source_ply: Path,
) -> None:
    if manifest.get("source") != VOTE_SOURCE:
        raise ValueError("vote manifest is not confident dense DINOv3 evidence")
    if manifest.get("contract") != VOTE_CONTRACT:
        raise ValueError("unsupported confident dense vote contract")
    if int(manifest.get("gaussian_count", -1)) != vertex_count:
        raise ValueError("vote manifest Gaussian count differs from labels")
    if Path(str(manifest.get("ply_path", ""))).resolve() != source_ply.resolve():
        raise ValueError("vote manifest source PLY differs from requested PLY")
    for field, expected in (
        ("query_region_filtering_used", False),
        ("confidence_threshold_used", True),
        ("inference_rerun", False),
        ("flashsplat_rerun", True),
        ("abstain_mass_preserved", True),
        ("scene_specific_rules", False),
        ("class_specific_thresholds", False),
        ("manual_component_decisions", False),
        ("v5_used", False),
        ("dinov2_used", False),
    ):
        if manifest.get(field) is not expected:
            raise ValueError(f"vote manifest violates {field}={expected}")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("vote manifest has no frames")
    camera_indices = [int(frame["camera_index"]) for frame in frames]
    if len(set(camera_indices)) != len(camera_indices):
        raise ValueError("vote manifest contains duplicate cameras")
    if len(frames) > 128:
        raise ValueError("camera bitset contract supports at most 128 cameras")


def decide_camera_winners(
    candidate_count: int,
    local_indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
    accepted_fractions: np.ndarray,
    proposed_ids: np.ndarray,
    *,
    min_accepted_fraction: float,
    min_winner_share: float,
    min_winner_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reliable winner IDs and whether they match the core proposal."""

    local = np.asarray(local_indices, dtype=np.int64)
    classes = np.asarray(class_ids, dtype=np.uint16)
    values = np.asarray(weights, dtype=np.float32)
    accepted = np.asarray(accepted_fractions, dtype=np.float32)
    proposed = np.asarray(proposed_ids, dtype=np.uint16)
    if not (local.shape == classes.shape == values.shape) or local.ndim != 1:
        raise ValueError("camera sparse vote arrays must align")
    if accepted.shape != (candidate_count,) or proposed.shape != (candidate_count,):
        raise ValueError("camera decision arrays differ from candidate count")
    if local.size and (int(local.min()) < 0 or int(local.max()) >= candidate_count):
        raise ValueError("camera vote references an invalid candidate")

    winner = np.zeros((candidate_count,), dtype=np.uint16)
    best = np.zeros((candidate_count,), dtype=np.float32)
    runner = np.zeros((candidate_count,), dtype=np.float32)
    tied = np.zeros((candidate_count,), dtype=bool)
    for project_id in np.unique(classes):
        selected = classes == project_id
        positions = local[selected]
        scores = values[selected]
        if np.unique(positions).size != positions.size:
            raise ValueError("camera has duplicate class mass for a Gaussian")
        current = best[positions]
        greater = scores > current
        equal_positive = (scores == current) & (scores > 0.0)
        if np.any(greater):
            changed = positions[greater]
            runner[changed] = current[greater]
            best[changed] = scores[greater]
            winner[changed] = np.uint16(project_id)
            tied[changed] = False
        remaining = ~greater
        if np.any(remaining):
            unchanged = positions[remaining]
            runner[unchanged] = np.maximum(
                runner[unchanged],
                scores[remaining],
            )
        if np.any(equal_positive):
            tied[positions[equal_positive]] = True

    reliable = (
        (accepted >= min_accepted_fraction)
        & (best >= min_winner_share)
        & ((best - runner) >= min_winner_margin)
        & (winner > 0)
        & ~tied
    )
    reliable_winner = np.where(reliable, winner, 0).astype(np.uint16)
    return reliable_winner, reliable & (winner == proposed)


def _boundary_project_ids(ontology: Ontology) -> tuple[int, ...]:
    names = {"wall", "ceiling", "floor"}
    return tuple(
        sorted(
            item.project_id
            for item in ontology.classes
            if item.project_class in names
        )
    )


def _camera_count_from_bits(bits: np.ndarray) -> int:
    words = np.bitwise_or.reduce(np.asarray(bits, dtype=np.uint64), axis=0)
    return sum(int(value).bit_count() for value in words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-first-report", required=True, type=Path)
    parser.add_argument("--core-first-supports", required=True, type=Path)
    parser.add_argument("--vote-manifest", required=True, type=Path)
    parser.add_argument("--preferred-labels", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-reliable-cameras", default=3, type=int)
    parser.add_argument(
        "--min-camera-accepted-fraction",
        default=0.50,
        type=float,
    )
    parser.add_argument("--min-camera-winner-share", default=0.50, type=float)
    parser.add_argument("--min-camera-winner-margin", default=0.10, type=float)
    parser.add_argument("--min-global-winner-share", default=0.55, type=float)
    parser.add_argument("--min-global-winner-margin", default=0.10, type=float)
    parser.add_argument("--min-boundary-margin", default=0.10, type=float)
    parser.add_argument("--min-spatial-component-gaussians", default=500, type=int)
    parser.add_argument("--min-spatial-component-cameras", default=3, type=int)
    parser.add_argument("--voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--min-voxel-size", default=0.01, type=float)
    parser.add_argument("--max-voxel-size", default=0.20, type=float)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (
        args.core_first_report,
        args.core_first_supports,
        args.vote_manifest,
        args.preferred_labels,
        args.source_ply,
        args.ontology,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    thresholds = DenseCrossValidationThresholds(
        min_reliable_cameras=args.min_reliable_cameras,
        min_camera_accepted_fraction=args.min_camera_accepted_fraction,
        min_camera_winner_share=args.min_camera_winner_share,
        min_camera_winner_margin=args.min_camera_winner_margin,
        min_global_winner_share=args.min_global_winner_share,
        min_global_winner_margin=args.min_global_winner_margin,
        min_boundary_margin=args.min_boundary_margin,
        min_spatial_component_gaussians=args.min_spatial_component_gaussians,
        min_spatial_component_cameras=args.min_spatial_component_cameras,
        voxel_scale_multiplier=args.voxel_scale_multiplier,
        min_voxel_size=args.min_voxel_size,
        max_voxel_size=args.max_voxel_size,
    )
    thresholds.validate()

    preferred_hash = file_sha256(args.preferred_labels)
    preferred = np.load(args.preferred_labels, mmap_mode="r")
    if preferred.ndim != 1 or not np.issubdtype(preferred.dtype, np.integer):
        raise ValueError("preferred labels must be a one-dimensional integer array")
    if preferred.size < 1 or np.any(preferred < 0):
        raise ValueError("preferred labels must be non-negative and non-empty")
    vertex_count = int(preferred.size)
    ontology = load_ontology(args.ontology)

    core_report = json.loads(args.core_first_report.read_text(encoding="utf-8"))
    with np.load(args.core_first_supports, allow_pickle=False) as archive:
        cores = load_core_candidates(core_report, archive, vertex_count)
    vote_manifest = json.loads(args.vote_manifest.read_text(encoding="utf-8"))
    validate_vote_manifest(
        vote_manifest,
        vertex_count=vertex_count,
        source_ply=args.source_ply,
    )

    header, vertices = vertex_data_memmap(args.source_ply)
    if not header.elements or int(header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from preferred labels")
    required_fields = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    if not required_fields.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY lacks coordinates or Gaussian scales")

    preferred_labeled = np.asarray(preferred != 0)
    all_core_mask = np.zeros((vertex_count,), dtype=bool)
    excluded_preferred_mask = np.zeros((vertex_count,), dtype=bool)
    candidate_indices_parts: list[np.ndarray] = []
    candidate_project_parts: list[np.ndarray] = []
    candidate_parent_parts: list[np.ndarray] = []
    for core in cores:
        all_core_mask[core.indices] = True
        is_labeled = preferred_labeled[core.indices]
        excluded_preferred_mask[core.indices[is_labeled]] = True
        black = core.indices[~is_labeled]
        candidate_indices_parts.append(black)
        candidate_project_parts.append(
            np.full(black.shape, core.project_id, dtype=np.uint16)
        )
        candidate_parent_parts.append(
            np.full(black.shape, core.component_id, dtype=np.int32)
        )
    candidate_indices = (
        np.concatenate(candidate_indices_parts)
        if candidate_indices_parts
        else np.zeros((0,), dtype=np.uint32)
    )
    proposed_ids = (
        np.concatenate(candidate_project_parts)
        if candidate_project_parts
        else np.zeros((0,), dtype=np.uint16)
    )
    parent_ids = (
        np.concatenate(candidate_parent_parts)
        if candidate_parent_parts
        else np.zeros((0,), dtype=np.int32)
    )
    if np.unique(candidate_indices).size != candidate_indices.size:
        raise ValueError("black core-first candidate ownership is not exclusive")
    candidate_count = int(candidate_indices.size)
    if candidate_count == 0:
        raise ValueError("core-first audit has no black Gaussian candidates")
    global_to_local = np.full((vertex_count,), -1, dtype=np.int32)
    global_to_local[candidate_indices] = np.arange(
        candidate_count,
        dtype=np.int32,
    )

    visible_camera_count = np.zeros((candidate_count,), dtype=np.uint16)
    reliable_winner_count = np.zeros((candidate_count,), dtype=np.uint16)
    proposed_winner_count = np.zeros((candidate_count,), dtype=np.uint16)
    proposed_camera_bits = np.zeros((candidate_count, 2), dtype=np.uint64)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    class_count = ontology.class_count
    with tempfile.TemporaryDirectory(
        prefix="dense_cross_validation_",
        dir=args.output_dir,
    ) as cache:
        vote_matrix = np.memmap(
            Path(cache) / "pooled_votes.float32",
            mode="w+",
            dtype=np.float32,
            shape=(class_count + 1, candidate_count),
        )
        vote_matrix[:] = 0.0

        for camera_ordinal, frame in enumerate(vote_manifest["frames"]):
            vote_path = args.vote_manifest.parent / str(frame["vote_file"])
            with np.load(vote_path, allow_pickle=False) as data:
                global_vote_indices = np.asarray(data["indices"], dtype=np.uint32)
                local_vote_indices = global_to_local[global_vote_indices]
                selected = local_vote_indices >= 0
                local_vote_indices = local_vote_indices[selected]
                vote_classes = np.asarray(data["class_ids"], dtype=np.uint16)[
                    selected
                ]
                weights = np.asarray(data["weights"], dtype=np.float32)[selected]
                if local_vote_indices.size:
                    np.add.at(
                        vote_matrix,
                        (
                            vote_classes.astype(np.int64, copy=False),
                            local_vote_indices.astype(np.int64, copy=False),
                        ),
                        weights,
                    )

                visible_global = np.asarray(
                    data["visible_indices"],
                    dtype=np.uint32,
                )
                visible_local = global_to_local[visible_global]
                visible_selected = visible_local >= 0
                visible_local = visible_local[visible_selected]
                accepted = np.zeros((candidate_count,), dtype=np.float32)
                accepted[visible_local] = np.asarray(
                    data["accepted_fractions"],
                    dtype=np.float32,
                )[visible_selected]
            visible_camera_count[visible_local] += np.uint16(1)
            reliable_winner, matches_proposal = decide_camera_winners(
                candidate_count,
                local_vote_indices,
                vote_classes,
                weights,
                accepted,
                proposed_ids,
                min_accepted_fraction=thresholds.min_camera_accepted_fraction,
                min_winner_share=thresholds.min_camera_winner_share,
                min_winner_margin=thresholds.min_camera_winner_margin,
            )
            reliable = reliable_winner > 0
            reliable_winner_count[reliable] += np.uint16(1)
            proposed_winner_count[matches_proposal] += np.uint16(1)
            word = camera_ordinal // 64
            bit = np.uint64(1) << np.uint64(camera_ordinal % 64)
            proposed_camera_bits[matches_proposal, word] |= bit
            print(
                f"accumulated {frame['file']}: "
                f"candidate_visible={visible_local.size} "
                f"reliable={int(np.count_nonzero(reliable))} "
                f"proposal_matches={int(np.count_nonzero(matches_proposal))}"
            )

        vote_matrix.flush()
        global_winner = np.zeros((candidate_count,), dtype=np.uint16)
        global_share = np.zeros((candidate_count,), dtype=np.float32)
        global_margin = np.zeros((candidate_count,), dtype=np.float32)
        boundary_margin = np.zeros((candidate_count,), dtype=np.float32)
        boundary_ids = _boundary_project_ids(ontology)
        for start in range(0, candidate_count, args.chunk_size):
            end = min(start + args.chunk_size, candidate_count)
            votes = np.asarray(vote_matrix[1:, start:end])
            winners = np.argmax(votes, axis=0)
            columns = np.arange(end - start, dtype=np.int64)
            winner_scores = votes[winners, columns]
            runner_scores = np.partition(votes, -2, axis=0)[-2]
            denominators = votes.sum(axis=0, dtype=np.float32)
            global_winner[start:end] = (
                winners.astype(np.uint16) + np.uint16(1)
            )
            global_winner[start:end][winner_scores <= 0.0] = 0
            global_share[start:end] = np.divide(
                winner_scores,
                denominators,
                out=np.zeros_like(winner_scores),
                where=denominators > 0.0,
            )
            global_margin[start:end] = np.divide(
                winner_scores - runner_scores,
                denominators,
                out=np.zeros_like(winner_scores),
                where=denominators > 0.0,
            )
            proposed_chunk = proposed_ids[start:end]
            proposed_scores = votes[
                proposed_chunk.astype(np.int64) - 1,
                columns,
            ]
            boundary_scores = np.zeros((end - start,), dtype=np.float32)
            for project_id in boundary_ids:
                scores = votes[project_id - 1]
                scores = np.where(proposed_chunk == project_id, 0.0, scores)
                boundary_scores = np.maximum(boundary_scores, scores)
            boundary_margin[start:end] = np.divide(
                proposed_scores - boundary_scores,
                denominators,
                out=np.zeros_like(proposed_scores),
                where=denominators > 0.0,
            )
        vote_matrix.flush()
        mmap_handle = vote_matrix._mmap
        del votes
        del vote_matrix
        mmap_handle.close()

    reasons = np.full(
        (candidate_count,),
        REASON_ACCEPTED_BEFORE_SPATIAL,
        dtype=np.uint8,
    )
    observed = visible_camera_count > 0
    reasons[~observed] = REASON_UNOBSERVED
    enough_reliable = reliable_winner_count >= thresholds.min_reliable_cameras
    reasons[observed & ~enough_reliable] = REASON_INSUFFICIENT_RELIABLE_CAMERAS
    eligible = observed & enough_reliable
    winner_matches = global_winner == proposed_ids
    reasons[eligible & ~winner_matches] = REASON_GLOBAL_WINNER_MISMATCH
    eligible &= winner_matches
    strict_majority = (
        proposed_winner_count.astype(np.uint32) * 2
        > reliable_winner_count.astype(np.uint32)
    )
    reasons[eligible & ~strict_majority] = REASON_NO_STRICT_CAMERA_MAJORITY
    eligible &= strict_majority
    share_pass = global_share >= thresholds.min_global_winner_share
    reasons[eligible & ~share_pass] = REASON_LOW_GLOBAL_WINNER_SHARE
    eligible &= share_pass
    margin_pass = global_margin >= thresholds.min_global_winner_margin
    reasons[eligible & ~margin_pass] = REASON_LOW_GLOBAL_WINNER_MARGIN
    eligible &= margin_pass
    boundary_pass = boundary_margin >= thresholds.min_boundary_margin
    reasons[eligible & ~boundary_pass] = REASON_BOUNDARY_COMPETITOR
    dense_survivor = eligible & boundary_pass

    dense_survivor_mask = np.zeros((vertex_count,), dtype=bool)
    dense_survivor_mask[candidate_indices[dense_survivor]] = True
    validated_mask = np.zeros((vertex_count,), dtype=bool)
    components: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    accepted_supports: dict[str, np.ndarray] = {}
    class_counts: dict[str, int] = {}
    next_component_id = 1
    core_by_id = {core.component_id: core for core in cores}

    for parent_component_id in sorted(core_by_id):
        core = core_by_id[parent_component_id]
        parent_local = np.flatnonzero(parent_ids == parent_component_id)
        survivor_local = parent_local[dense_survivor[parent_local]]
        survivor_global = candidate_indices[survivor_local]
        child_ids: list[int] = []
        accepted_count = 0
        accepted_gaussians = 0
        if survivor_global.size:
            points = np.column_stack(
                [
                    vertices[axis][survivor_global].astype(np.float64)
                    for axis in ("x", "y", "z")
                ]
            )
            log_scales = np.column_stack(
                [
                    vertices[axis][survivor_global].astype(np.float64)
                    for axis in ("scale_0", "scale_1", "scale_2")
                ]
            )
            median_scale, voxel_size = adaptive_voxel_size(
                log_scales,
                voxel_scale_multiplier=thresholds.voxel_scale_multiplier,
                min_voxel_size=thresholds.min_voxel_size,
                max_voxel_size=thresholds.max_voxel_size,
            )
            point_components, component_sizes, geometry = voxel_components(
                points,
                voxel_size,
            )
        else:
            median_scale = 0.0
            voxel_size = 0.0
            point_components = np.zeros((0,), dtype=np.int32)
            component_sizes = np.zeros((0,), dtype=np.int64)
            geometry = {
                "voxel_count": 0,
                "component_count": 0,
                "largest_component_gaussians": 0,
            }

        for local_component_id, size_value in enumerate(component_sizes):
            selected = point_components == local_component_id
            local_candidate_positions = survivor_local[selected]
            indices = survivor_global[selected]
            size = int(size_value)
            independent_cameras = _camera_count_from_bits(
                proposed_camera_bits[local_candidate_positions]
            )
            large_enough = size >= thresholds.min_spatial_component_gaussians
            enough_cameras = (
                independent_cameras
                >= thresholds.min_spatial_component_cameras
            )
            accepted = large_enough and enough_cameras
            if not large_enough:
                status = "rejected_insufficient_spatial_gaussians"
            elif not enough_cameras:
                status = "rejected_insufficient_spatial_cameras"
            else:
                status = "accepted_report_only_dense_validated_fill"
            record = {
                "component_id": next_component_id,
                "parent_core_first_component_id": parent_component_id,
                "local_spatial_component_id": local_component_id,
                "class": core.class_name,
                "project_id": core.project_id,
                "status": status,
                "accepted": bool(accepted),
                "support_gaussian_count": size,
                "independent_camera_count": independent_cameras,
                "minimum_reliable_camera_count": int(
                    reliable_winner_count[local_candidate_positions].min()
                ),
                "maximum_reliable_camera_count": int(
                    reliable_winner_count[local_candidate_positions].max()
                ),
                "minimum_proposed_winner_camera_count": int(
                    proposed_winner_count[local_candidate_positions].min()
                ),
                "maximum_proposed_winner_camera_count": int(
                    proposed_winner_count[local_candidate_positions].max()
                ),
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
            }
            components.append(record)
            child_ids.append(next_component_id)
            if accepted:
                if np.any(validated_mask[indices]):
                    raise ValueError("validated spatial supports overlap")
                validated_mask[indices] = True
                accepted_count += 1
                accepted_gaussians += size
                prefix = f"component_{next_component_id:06d}"
                accepted_supports[f"{prefix}_indices"] = indices
                accepted_supports[f"{prefix}_camera_counts"] = (
                    proposed_winner_count[local_candidate_positions]
                )
                class_counts[core.class_name] = (
                    class_counts.get(core.class_name, 0) + size
                )
            else:
                reasons[local_candidate_positions] = (
                    REASON_REJECTED_SPATIAL_GATE
                )
            next_component_id += 1

        parent_records.append(
            {
                "parent_core_first_component_id": parent_component_id,
                "class": core.class_name,
                "project_id": core.project_id,
                "core_first_gaussian_count": int(core.indices.size),
                "excluded_preferred_label_gaussian_count": int(
                    np.count_nonzero(preferred_labeled[core.indices])
                ),
                "black_candidate_gaussian_count": int(parent_local.size),
                "dense_survivor_before_spatial_count": int(survivor_local.size),
                "spatial_component_count": int(component_sizes.size),
                "accepted_spatial_component_count": accepted_count,
                "validated_fill_gaussian_count": accepted_gaussians,
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                "spatial_geometry": geometry,
                "component_ids": child_ids,
            }
        )

    if np.any(validated_mask & preferred_labeled):
        raise AssertionError("dense-validated fill overlaps preferred labels")
    if file_sha256(args.preferred_labels) != preferred_hash:
        raise RuntimeError("preferred label source changed during the audit")

    candidate_mask = np.zeros((vertex_count,), dtype=bool)
    candidate_mask[candidate_indices] = True
    np.save(args.output_dir / "black_core_first_candidate_mask.npy", candidate_mask)
    np.save(args.output_dir / "excluded_preferred_label_mask.npy", excluded_preferred_mask)
    np.save(args.output_dir / "dense_survivor_before_spatial_mask.npy", dense_survivor_mask)
    np.save(args.output_dir / "dense_validated_fill_mask.npy", validated_mask)
    np.save(args.output_dir / "candidate_gate_reason_codes.npy", reasons)
    np.savez_compressed(
        args.output_dir / "dense_validated_supports.npz",
        **accepted_supports,
    )

    reason_counts = {
        REASON_NAMES[int(code)]: int(count)
        for code, count in zip(*np.unique(reasons, return_counts=True))
    }
    preferred_count = int(np.count_nonzero(preferred_labeled))
    preferred_black_count = vertex_count - preferred_count
    validated_count = int(np.count_nonzero(validated_mask))
    profile_name = str(vote_manifest["profile_name"])
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "profile_name": profile_name,
        "core_first_report": str(args.core_first_report),
        "core_first_supports": str(args.core_first_supports),
        "vote_manifest": str(args.vote_manifest),
        "pixel_confidence_thresholds": vote_manifest["thresholds"],
        "preferred_labels": str(args.preferred_labels),
        "preferred_labels_sha256": preferred_hash,
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "report_only": True,
        "preferred_labels_read_only": True,
        "preferred_labels_modified": False,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "inference_rerun": False,
        "flashsplat_rerun": True,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "manual_component_decisions": False,
        "offline_component_selection_used": False,
        "candidate_policy": (
            "core_first_support_intersect_preferred_v2_black_gaussians"
        ),
        "camera_vote_policy": (
            "one_confidence_filtered_dense_winner_per_independent_camera"
        ),
        "global_vote_policy": (
            "proposed_class_must_be_unique_pooled_winner_with_strict_camera_"
            "majority_share_margin_and_boundary_margin"
        ),
        "boundary_classes_compared": ["ceiling", "floor", "wall"],
        "geometry_policy": (
            "semantic_filter_first_then_adaptive_26_neighbor_voxel_components"
        ),
        "visualization_scope": (
            "exact_projection_of_dense_validated_black_fill_support"
        ),
        "parameters": asdict(thresholds),
        "vertex_count": vertex_count,
        "camera_count": int(vote_manifest["camera_count"]),
        "preferred_labeled_gaussian_count": preferred_count,
        "preferred_unlabeled_gaussian_count": preferred_black_count,
        "core_first_gaussian_count": int(np.count_nonzero(all_core_mask)),
        "core_first_preferred_overlap_gaussian_count": int(
            np.count_nonzero(excluded_preferred_mask)
        ),
        "black_core_first_candidate_gaussian_count": candidate_count,
        "dense_survivor_before_spatial_gaussian_count": int(
            np.count_nonzero(dense_survivor)
        ),
        "spatial_component_count": len(components),
        "accepted_spatial_component_count": sum(
            int(record["accepted"]) for record in components
        ),
        "validated_fill_gaussian_count": validated_count,
        "preferred_unlabeled_recovery_ratio": (
            validated_count / float(preferred_black_count)
            if preferred_black_count
            else 0.0
        ),
        "combined_assigned_gaussian_count_if_materialized": (
            preferred_count + validated_count
        ),
        "combined_assigned_ratio_if_materialized": (
            preferred_count + validated_count
        )
        / float(vertex_count),
        "candidate_gate_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_distributions": {
            "visible_camera_count": quantile_summary(visible_camera_count),
            "reliable_winner_camera_count": quantile_summary(
                reliable_winner_count
            ),
            "proposed_winner_camera_count": quantile_summary(
                proposed_winner_count
            ),
            "global_winner_share": quantile_summary(global_share),
            "global_winner_margin": quantile_summary(global_margin),
            "proposed_to_boundary_margin": quantile_summary(boundary_margin),
        },
        "class_validated_fill_gaussian_counts": dict(sorted(class_counts.items())),
        "parent_core_first_components": parent_records,
        "components": components,
        "outputs": {
            "report_only": True,
            "semantic_labels_written": False,
            "semantic_project_class_arrays_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
            "boolean_masks_written": True,
            "diagnostic_reason_codes_written": True,
            "compressed_sparse_supports_written": True,
        },
    }
    (args.output_dir / "dense_cross_validation_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "scene": args.scene,
        "profile_name": profile_name,
        "black_core_first_candidate_gaussian_count": candidate_count,
        "dense_survivor_before_spatial_gaussian_count": int(
            np.count_nonzero(dense_survivor)
        ),
        "validated_fill_gaussian_count": validated_count,
    }, indent=2))


if __name__ == "__main__":
    main()
