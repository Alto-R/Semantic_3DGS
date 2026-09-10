# SAM3 visibility and semantic consensus refinement

The old_street refinement separates semantic class agreement from global
instance identity. A window can retain its semantic label when different
views assigned it different candidate instance IDs. Its instance identity
remains unresolved until the independent instance consensus succeeds.

The original strict route remains the default. For the refined route:

```bash
export VISIBILITY_WEIGHTING=soft
export VISIBILITY_ABSOLUTE_SCALE=0.01
export VISIBILITY_RELATIVE_SCALE=0.05
export CONSENSUS_THRESHOLD=0.4
export PREVENT_DISJOINT_INSTANCE_MERGES=1
export POOL_CONCEPT_VOTES=1
export SEMANTIC_THRESHOLD=0.4
bash scripts/slurm/slurm_task1_sam3_instance_scene.sbatch
```

The mass cache requires one additional FlashSplat visibility pass per camera,
using the exact model, camera index and resolution of the original votes.
The observed Gaussian sets must match exactly. It does not rerun SAM3.

Per-Gaussian camera reliability is
`min(1, mass / max(0.01, 0.05 * maximum_mass_across_cameras))`.
Both supporting votes and observing views receive this same weight. Weak
views retain a small contribution; they are not discarded. Acceptance still
requires at least two distinct supporting cameras and a weighted support
ratio strictly above 0.4. The two-camera requirement counts distinct raw
supporting views; it does not guarantee two full-weight observations.
The unchanged within-camera dominant-mask membership cutoff is 0.5, and
the unchanged SAM3 detection score cutoff is 0.4.

`--visibility-weighting hard` is available for ablation. It discards views
below either visibility scale, from both the numerator and denominator.
This variant was too restrictive on old_street: even removing its ratio
threshold capped two-camera instance coverage near 67.3%.

Association optionally creates cannot-link constraints for `gnn_node`
concepts: two detections from the same camera and concept with 2D mask IoU
at most 0.1 cannot enter the same connected component, including through
transitive unions. Substantially overlapping synonym detections remain
eligible. Context and OOV concepts are not constrained by this option.
Candidate edge ranking remains deterministic intersection-count order.
This constraint reduces a known source of over-merging; it is not a complete
instance-matching or tracking solution.

The semantic stage pools same-concept support across candidate instance IDs.
S4 permits only one winner per Gaussian, camera and concept, so this pooling
does not count synonym prompts as separate camera votes. It validates that
pooled counts and reliability weights do not exceed their observation totals.
No new global instance IDs are assigned by the semantic stage.

Outputs under `stages/04_semantic_consensus`:

- `gaussian_labels.npy`: semantic concept ID per Gaussian, score-first.
- `gaussian_labels_objects_first.npy`: same accepted evidence and coverage,
  with accepted GNN object concepts prioritized over context for display.
- `label_map.json`: IDs, concepts, roles and colors.
- `concept_membership.npz`: pooled multi-label evidence, support counts,
  observation counts, reliability weights, scores and acceptance status.
- `instance_unresolved.npy`: semantic label accepted but no instance accepted.
- `semantic_summary.json`: configuration, source hashes, coverage, class
  distributions and threshold ablations.

`stages/05_scene_graph/gaussian_instances.npy` continues to represent instance
IDs. Semantic coverage must not be reported as instance coverage.

The full 129-view old_street refinement retained 3,490,864 / 4,327,550
Gaussians with semantic labels (80.6661%), including 278,770 whose instance
identity remains unresolved. Instance coverage is 74.2243%. The final graph
contains 1,395 nodes and 1,390 part_of edges. These are coverage and structure
counts, not ground-truth semantic or instance accuracy.

The cluster run reuses S1/S2 from `old_street_sam3_instance_pilot_v1` and is
stored separately as `old_street_sam3_instance_v3`. `old_street_sam3_instance_v2`
retains the visibility cache and hard/soft visibility ablations. Original
outputs and the production checkout remain unchanged. The full SAM3 suite
passes 92 tests, including visibility weighting, transitive cannot-link
constraints, class pooling, and display priority regressions.
