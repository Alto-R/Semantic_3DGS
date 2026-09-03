#!/usr/bin/env python3
"""Fuse every lifted DINOv3 query region directly at each Gaussian.

Each camera contributes at most one normalized 151-way distribution per
Gaussian: 150 ADE20K classes plus Mask2Former's no-object probability.  The
winning proposal within a camera is selected only by its FlashSplat support.
No cross-view region association or prior semantic labels are used.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


REASON_ACCEPTED = 0
REASON_INSUFFICIENT_VIEWS = 1
REASON_NO_STRICT_MAJORITY = 2
REASON_NO_OBJECT_MAJORITY = 3
REASON_SOFT_HARD_DISAGREEMENT = 4

REASON_NAMES = {
    REASON_ACCEPTED: "accepted",
    REASON_INSUFFICIENT_VIEWS: "insufficient_views",
    REASON_NO_STRICT_MAJORITY: "no_strict_majority",
    REASON_NO_OBJECT_MAJORITY: "no_object_majority",
    REASON_SOFT_HARD_DISAGREEMENT: "soft_hard_disagreement",
}


@dataclass(frozen=True)
class ViewProposal:
    proposal_id: int
    frame_file: str
    indices: np.ndarray
    counts: np.ndarray
    distribution: np.ndarray


def choose_view_owners(
    supports: list[tuple[np.ndarray, np.ndarray]],
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return touched indices and one winning proposal index per vertex."""
    best_counts = np.zeros((vertex_count,), dtype=np.float32)
    owners = np.full((vertex_count,), -1, dtype=np.int32)
    touched: list[np.ndarray] = []
    for proposal_index, (indices, counts) in enumerate(supports):
        proposal_indices = np.asarray(indices, dtype=np.int64)
        proposal_counts = np.asarray(counts, dtype=np.float32)
        if proposal_indices.ndim != 1 or proposal_counts.shape != proposal_indices.shape:
            raise ValueError("proposal support vectors must have matching shapes")
        if proposal_indices.size and (
            proposal_indices[0] < 0 or proposal_indices[-1] >= vertex_count
        ):
            raise ValueError("proposal support index is outside the source PLY")
        if not np.isfinite(proposal_counts).all() or np.any(proposal_counts <= 0.0):
            raise ValueError("proposal support counts must be finite positive values")
        selected = proposal_counts > best_counts[proposal_indices]
        selected_indices = proposal_indices[selected]
        best_counts[selected_indices] = proposal_counts[selected]
        owners[selected_indices] = proposal_index
        touched.append(proposal_indices)
    if not touched:
        return np.empty((0,), dtype=np.int64), owners
    return np.unique(np.concatenate(touched)), owners


def decide_multiview_labels(
    hard_votes: np.ndarray,
    soft_votes: np.ndarray,
    supporting_views: np.ndarray,
    *,
    no_object_index: int,
    min_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply strict camera majority and soft-winner agreement."""
    hard = np.asarray(hard_votes)
    soft = np.asarray(soft_votes, dtype=np.float32)
    views = np.asarray(supporting_views)
    if hard.ndim != 2 or soft.shape != hard.shape:
        raise ValueError("hard and soft votes must have matching NxC shapes")
    if views.shape != (hard.shape[0],):
        raise ValueError("supporting view counts must have shape N")
    if not 0 <= no_object_index < hard.shape[1]:
        raise ValueError("no-object index is outside the vote matrix")
    if min_views < 2:
        raise ValueError("min_views must preserve independent multiview evidence")

    hard_winners = np.argmax(hard, axis=1).astype(np.int16)
    hard_counts = hard[np.arange(hard.shape[0]), hard_winners].astype(np.uint16)
    soft_winners = np.argmax(soft, axis=1).astype(np.int16)
    soft_totals = soft.sum(axis=1)
    soft_confidence = np.divide(
        soft[np.arange(soft.shape[0]), soft_winners],
        np.maximum(soft_totals, np.finfo(np.float32).tiny),
    ).astype(np.float16)
    hard_agreement = np.divide(
        hard_counts,
        np.maximum(views, 1),
    ).astype(np.float16)

    reasons = np.full(
        (hard.shape[0],),
        REASON_INSUFFICIENT_VIEWS,
        dtype=np.uint8,
    )
    enough_views = views >= min_views
    strict_majority = hard_counts.astype(np.uint16) * 2 > views.astype(np.uint16)
    reasons[enough_views & ~strict_majority] = REASON_NO_STRICT_MAJORITY
    majority_no_object = enough_views & strict_majority & (
        hard_winners == no_object_index
    )
    reasons[majority_no_object] = REASON_NO_OBJECT_MAJORITY
    semantic_majority = enough_views & strict_majority & ~majority_no_object
    disagreement = semantic_majority & (soft_winners != hard_winners)
    reasons[disagreement] = REASON_SOFT_HARD_DISAGREEMENT
    accepted = semantic_majority & ~disagreement
    reasons[accepted] = REASON_ACCEPTED

    winners = np.full((hard.shape[0],), -1, dtype=np.int16)
    winners[accepted] = hard_winners[accepted]
    return winners, reasons, hard_agreement, soft_confidence


def load_view_proposals(
    proposal_manifest_path: Path,
    ontology: Ontology,
) -> tuple[dict[str, list[ViewProposal]], dict[str, Any]]:
    manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = Path(str(manifest["source_manifest"]))
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("contract") != "class_agnostic_query_masks_with_soft_evidence_v1":
        raise ValueError("proposal source is not the DINOv3 query-region contract")
    if source_manifest.get("v5_used") is not False:
        raise ValueError("query regions are not independent of v5")
    if source_manifest.get("dinov2_used") is not False:
        raise ValueError("query regions are not independent of DINOv2")
    if source_manifest.get("semantic_identity_assigned") is not False:
        raise ValueError("query regions were semantically hardened before voting")

    support_dir = proposal_manifest_path.parent / "proposal_supports"
    evidence_cache: dict[Path, dict[str, np.ndarray]] = {}
    by_frame: dict[str, list[ViewProposal]] = {}
    seen_ids: set[int] = set()
    for metadata in manifest.get("proposals", []):
        proposal_id = int(metadata["proposal_id"])
        if proposal_id in seen_ids:
            raise ValueError("proposal IDs are not unique")
        seen_ids.add(proposal_id)
        if bool(metadata.get("semantic_identity_assigned", True)):
            raise ValueError("proposal was semantically hardened before voting")
        evidence_path = source_manifest_path.parent / str(
            metadata["query_evidence_file"]
        )
        if evidence_path not in evidence_cache:
            with np.load(evidence_path, allow_pickle=False) as evidence:
                evidence_cache[evidence_path] = {
                    key: np.asarray(evidence[key])
                    for key in (
                        "class_probabilities",
                        "no_object_probabilities",
                    )
                }
        row = int(metadata["query_evidence_row"])
        evidence = evidence_cache[evidence_path]
        conditional = np.asarray(
            evidence["class_probabilities"][row],
            dtype=np.float32,
        )
        if conditional.shape != (ontology.class_count,):
            raise ValueError("query class probabilities do not match the ontology")
        if not np.isfinite(conditional).all() or np.any(conditional < 0.0):
            raise ValueError("query class probabilities are invalid")
        conditional /= max(float(conditional.sum()), np.finfo(np.float32).tiny)
        no_object = float(evidence["no_object_probabilities"][row])
        if not np.isfinite(no_object) or not 0.0 <= no_object <= 1.0:
            raise ValueError("query no-object probability is invalid")
        distribution = np.empty((ontology.class_count + 1,), dtype=np.float32)
        distribution[:-1] = conditional * np.float32(1.0 - no_object)
        distribution[-1] = np.float32(no_object)
        distribution /= max(float(distribution.sum()), np.finfo(np.float32).tiny)

        support_path = support_dir / str(metadata["support_file"])
        with np.load(support_path, allow_pickle=False) as support:
            indices = np.asarray(support["indices"], dtype=np.uint32)
            counts = np.asarray(support["counts"], dtype=np.float32)
        if indices.ndim != 1 or counts.shape != indices.shape:
            raise ValueError(f"invalid support vectors in {support_path}")
        if indices.size and np.any(indices[1:] <= indices[:-1]):
            raise ValueError(f"support indices are not strictly increasing: {support_path}")
        frame_file = str(metadata["frame_file"])
        by_frame.setdefault(frame_file, []).append(
            ViewProposal(
                proposal_id=proposal_id,
                frame_file=frame_file,
                indices=indices,
                counts=counts,
                distribution=distribution,
            )
        )
    for proposals in by_frame.values():
        proposals.sort(key=lambda item: item.proposal_id)
    return dict(sorted(by_frame.items())), manifest


def add_soft_votes(
    matrix: np.memmap,
    indices: np.ndarray,
    distribution: np.ndarray,
    batch_size: int,
) -> None:
    for start in range(0, indices.shape[0], batch_size):
        batch = indices[start : start + batch_size]
        matrix[batch, :] += distribution[None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--min-views", default=2, type=int)
    parser.add_argument("--batch-size", default=100_000, type=int)
    parser.add_argument("--no-semantic-ply", action="store_true")
    args = parser.parse_args()

    for path in (args.proposal_manifest, args.ontology, args.source_ply):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.min_views < 2:
        raise ValueError("min_views must be at least two")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    ontology = load_ontology(args.ontology)
    by_frame, proposal_manifest = load_view_proposals(
        args.proposal_manifest,
        ontology,
    )
    vertex_count = int(proposal_manifest["vertex_count"])
    manifest_ply = Path(str(proposal_manifest["ply_path"]))
    if manifest_ply.resolve() != args.source_ply.resolve():
        raise ValueError("source PLY differs from the lifted proposal provenance")
    ply_header = read_ply_header(args.source_ply)
    if not ply_header.elements or ply_header.elements[0].name != "vertex":
        raise ValueError("source PLY does not begin with a vertex element")
    if int(ply_header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from proposal manifest")

    args.output_dir.mkdir(parents=True)
    class_count_with_abstain = ontology.class_count + 1
    supporting_views = np.zeros((vertex_count,), dtype=np.uint8)
    winners = np.full((vertex_count,), -1, dtype=np.int16)
    reasons = np.empty((vertex_count,), dtype=np.uint8)
    hard_agreement = np.zeros((vertex_count,), dtype=np.float16)
    soft_confidence = np.zeros((vertex_count,), dtype=np.float16)

    with tempfile.TemporaryDirectory(prefix="vote_cache_", dir=args.output_dir) as cache:
        cache_path = Path(cache)
        soft_votes = np.memmap(
            cache_path / "soft_votes.float32",
            mode="w+",
            dtype=np.float32,
            shape=(vertex_count, class_count_with_abstain),
        )
        hard_votes = np.memmap(
            cache_path / "hard_votes.uint8",
            mode="w+",
            dtype=np.uint8,
            shape=(vertex_count, class_count_with_abstain),
        )
        soft_votes[:] = np.float32(0.0)
        hard_votes[:] = np.uint8(0)

        for frame_file, proposals in by_frame.items():
            touched, owners = choose_view_owners(
                [(item.indices, item.counts) for item in proposals],
                vertex_count,
            )
            assigned_count = 0
            for proposal_index, proposal in enumerate(proposals):
                owned = proposal.indices[owners[proposal.indices] == proposal_index]
                if owned.size == 0:
                    continue
                add_soft_votes(
                    soft_votes,
                    owned.astype(np.int64, copy=False),
                    proposal.distribution,
                    args.batch_size,
                )
                hard_winner = int(np.argmax(proposal.distribution))
                hard_votes[owned, hard_winner] += np.uint8(1)
                supporting_views[owned] += np.uint8(1)
                assigned_count += int(owned.shape[0])
            if assigned_count != int(touched.shape[0]):
                raise RuntimeError("a camera did not contribute exactly once per touched Gaussian")
            print(
                f"voted {frame_file}: proposals={len(proposals)} "
                f"gaussians={assigned_count}"
            )

        soft_votes.flush()
        hard_votes.flush()
        for start in range(0, vertex_count, args.batch_size):
            end = min(start + args.batch_size, vertex_count)
            hard_batch = np.asarray(hard_votes[start:end, :])
            soft_batch = np.asarray(soft_votes[start:end, :])
            view_batch = supporting_views[start:end]
            if not np.array_equal(
                hard_batch.sum(axis=1, dtype=np.uint16),
                view_batch.astype(np.uint16),
            ):
                raise RuntimeError("hard vote totals differ from supporting view counts")
            (
                winners[start:end],
                reasons[start:end],
                hard_agreement[start:end],
                soft_confidence[start:end],
            ) = decide_multiview_labels(
                hard_batch,
                soft_batch,
                view_batch,
                no_object_index=ontology.class_count,
                min_views=args.min_views,
            )
        del hard_batch
        del soft_batch
        del soft_votes
        del hard_votes

    project_lookup = np.asarray(
        [item.project_id for item in ontology.classes],
        dtype=np.int32,
    )
    project_classes = np.zeros((vertex_count,), dtype=np.int32)
    accepted = winners >= 0
    project_classes[accepted] = project_lookup[winners[accepted]]
    labels = project_classes.copy()

    outputs = {
        "gaussian_labels.npy": labels,
        "gaussian_project_class_ids.npy": project_classes,
        "winner_ade20k_ids.npy": winners,
        "supporting_views.npy": supporting_views,
        "hard_winner_agreement.npy": hard_agreement,
        "soft_winner_confidence.npy": soft_confidence,
        "abstain_reason_codes.npy": reasons,
    }
    for filename, array in outputs.items():
        np.save(args.output_dir / filename, array)

    present_project_ids = np.unique(project_classes[project_classes > 0])
    labels_json: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    for project_id in present_project_ids:
        item = ontology.by_project_id[int(project_id)]
        labels_json.append(
            {
                "id": int(project_id),
                "name": item.project_class,
                "class": item.project_class,
                "project_id": item.project_id,
                "ade_id": item.ade_id,
                "type": item.kind,
            }
        )
    scene = args.scene or args.source_ply.parents[2].name
    label_map = {
        "scene": scene,
        "source": "dinov3_query_regions_direct_multiview_majority",
        "ontology": str(args.ontology),
        "proposal_manifest": str(args.proposal_manifest),
        "labels": labels_json,
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2),
        encoding="utf-8",
    )

    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    if not args.no_semantic_ply:
        partial = semantic_ply.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial, labels)
        partial.replace(semantic_ply)

    reason_counts = {
        REASON_NAMES[int(code)]: int(count)
        for code, count in zip(*np.unique(reasons, return_counts=True))
    }
    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(
            *np.unique(project_classes[project_classes > 0], return_counts=True)
        )
    }
    summary = {
        "source": "dinov3_query_regions_direct_multiview_majority",
        "contract": "per_gaussian_one_vote_per_camera_strict_majority_v1",
        "scene": scene,
        "proposal_manifest": str(args.proposal_manifest),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "v5_used": False,
        "dinov2_used": False,
        "region_association_used": False,
        "camera_vote_policy": "strongest_flashsplat_region_once_per_camera",
        "semantic_vote_policy": "strict_hard_majority_and_soft_winner_agreement",
        "no_object_policy": "explicit_151st_vote_class",
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "min_views": args.min_views,
        "camera_count_with_lifted_regions": len(by_frame),
        "proposal_count": sum(len(items) for items in by_frame.values()),
        "vertex_count": vertex_count,
        "gaussians_with_any_view": int(np.count_nonzero(supporting_views)),
        "assigned_gaussian_count": int(np.count_nonzero(accepted)),
        "assigned_ratio": float(np.mean(accepted)),
        "unlabeled_gaussian_count": int(np.count_nonzero(~accepted)),
        "abstain_reason_counts": reason_counts,
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "semantic_labels_written": True,
        "label_map_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
    }
    (args.output_dir / "region_vote_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
