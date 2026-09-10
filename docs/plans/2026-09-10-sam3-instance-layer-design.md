# SAM3 instance layer design

Date: 2026-09-10
Status: validated design, pre-implementation
Branch: `feature/sam3-instance-layer`

## Goal

Produce object-level data for the downstream dynamic scene-graph GNN: every
car, tree, person, or storefront in a scene becomes an individual node with a
3D position, so subject-to-object distances can be computed per timestep as a
subject navigates the reconstructed scene.

The maintained DINOv3 route produces one semantic class per Gaussian
(`gaussian_labels.npy`). Class-level labels cannot distinguish two cars, so
they cannot serve as GNN nodes. This design adds an instance-level route.

## Decisions

1. **SAM3 replaces the recognition models.** The new route uses SAM3
   promptable concept segmentation as its only recognition model. DINOv3,
   dino.txt, and automatic SAM are not called anywhere in the new route. SAM3
   is open-vocabulary (any short noun phrase), instance-native (one mask and
   identity per object), and subsumes the dino.txt+SAM out-of-vocabulary
   mechanism.
2. **The lifting and voting skeleton is reused.** Rendered reconstruction
   cameras, FlashSplat support accumulation, equal-camera voting, reliability
   statistics, and abstention all carry over with one structural change
   described next.
3. **Multi-label membership replaces the single-label partition.** The dense
   route enforces that per-view class mass sums to one at every Gaussian
   because its upstream is a mutually exclusive argmax map. SAM3 concepts
   legitimately overlap (`building` ⊃ `storefront` ⊃ `window`), so each
   concept votes independently. A Gaussian may end with several concurrent
   memberships, each with its own multiview score. No stage forces one label
   per Gaussian.
4. **Hierarchy is derived from 3D evidence, not asserted.** Part-of relations
   between instances come from asymmetric containment of their 3D support
   sets after voting. Configured expectations are used only as QA checks.
5. **Existing DINOv3 outputs remain read-only reference products.** The new
   route neither depends on them nor modifies them. Where the vocabularies
   overlap, agreement with the DINOv3 base is reported as a free QA signal.

## Pipeline

```text
S1 sam3_segment_views        rendered RGBs + vocabulary -> per-view instance masks
S2 lift_mask_view_votes      FlashSplat, one pass per concept per view ->
                             per-view per-mask Gaussian support
S3 associate_global_instances  support-overlap matching -> global instance ids
S4 materialize_instance_membership  per-concept equal-camera consensus ->
                             sparse membership matrix + reliability arrays
S5 derive_instance_hierarchy  overlap classification -> merge / part_of / noise,
                             scene graph with nodes and part_of edges
S6 QA renders                instance overlays and contact sheets
```

### S1: SAM3 segmentation

- Input: the existing per-camera RGB renders (no re-rendering) and a per-scene
  vocabulary config.
- SAM3 runs once per (view, concept phrase); each detected instance yields a
  binary mask with a confidence score. Same-concept instances in one view are
  disjoint by construction; cross-concept masks may overlap freely.
- The model call sits behind a small backend interface so unit tests run with
  a mock backend and no checkpoint. The real backend (Hugging Face
  `facebook/sam3`, pinned revision) is exercised only on the cluster.
- Output: per-view compressed mask stacks plus a manifest recording source,
  contract, vocabulary hash, model revision, and per-mask concept and score.

### S2: FlashSplat lift, one pass per concept

FlashSplat renders an index map and accumulates per-Gaussian alpha mass per
index, which assumes the indices partition the pixels. Overlapping concepts
therefore cannot share one pass, but instances of a single concept can:

- For each view and each concept present in it, build one index map (0 =
  not-this-concept, k = instance k) and run one FlashSplat pass.
- Per-Gaussian membership for a mask is `used_count[mask_row] / visibility`,
  the fraction of the Gaussian's rendered mass falling inside the mask.
  Within a concept pass this reuses the dense route's normalization; across
  concepts nothing is normalized, preserving multi-label structure.
- Output: per-view NPZ per concept pass with `indices`, `mask_ids`,
  `weights`, plus a vote manifest. Cost is roughly `|vocabulary|` fast
  rasterization passes per view.

### S3: cross-view instance association

The selected pilot cameras are not in capture order (for example
`cam0192 -> cam3921 -> cam1811`), so SAM3 video tracking cannot supply
cross-view identities. Association instead uses 3D support overlap, keeping
the route order-free:

- Each per-view mask has a sparse lifted support vector from S2.
- For each concept independently, greedily merge masks from different views
  whose support vectors exceed a weighted-Jaccard threshold (union-find).
  Merging across different concepts is forbidden at this stage.
- Output: a global instance registry mapping every (view, mask) to a global
  instance id, with per-instance supporting-camera counts.
- Fallback if pilot association quality is poor: run SAM3 tracking over the
  full source video and subsample, at higher cost. Not part of the pilot.

### S4: per-concept consensus

For each global instance, an equal-camera vote generalizing the audited
strict-majority policy:

- A camera that observes Gaussian g (positive visibility in that concept
  pass) votes for instance i if g's within-concept winner is i's per-view
  mask; otherwise it votes against.
- Membership score = supporting cameras / observing cameras, with the same
  acceptance statuses (single camera, tie, weak majority) recorded per
  (Gaussian, instance) pair, mirroring the dense route's abstention codes.
- Output: sparse CSR membership matrix over (gaussians x instances) with
  scores, per-instance reliability summaries, and a vote manifest. Instance
  ids are uint16 (the uint8 class assumption does not carry over).

### S5: overlap classification and hierarchy

Support-set overlap between accepted instances has three distinct causes,
classified from post-vote 3D statistics:

| Pattern | 3D criterion | Action |
| --- | --- | --- |
| Duplicate (synonym prompts) | high mutual IoU, synonymous concepts | merge nodes |
| True hierarchy (window in building) | containment(A in B) high, containment(B in A) low | emit `part_of` edge |
| Boundary noise (adjacent cars) | thin low-ratio overlap | keep scores, no edge |

- Output: `hierarchy.json` (part_of edges with containment statistics) and
  `scene_graph.json` (nodes with concept, membership-weighted 3D centroid,
  axis-aligned bounding box, supporting-camera count; edges: part_of). The
  scene graph is the GNN-facing deliverable.
- A derived flat labeling (one chosen hierarchy cut, one instance per
  Gaussian by top score) feeds the existing PLY/SuperSplat tooling as a
  visualization view only; it is never the storage format.

## Vocabulary config

Per-scene JSON (`configs/task1_sam3_vocabulary.<scene>.json`):

- `phrases`: list of `{phrase, role, synonyms}` where role is one of
  `gnn_node`, `context_probe` (stuff such as road, wall, sky, building),
  `oov_probe` (concepts absent from ADE20K, for example manhole cover, air
  conditioning unit, electric scooter).
- `expected_part_of`: optional phrase pairs used only in QA sanity checks.
- Roles gate acceptance criteria, not model behavior; every phrase is
  prompted identically.

## Pilot

Scene: `old_street` (129 selected views, DINOv3 v1 outputs exist for
comparison).

Acceptance criteria:

1. Per-view instance mask quality (contact sheet review).
2. Cross-view association: supporting-camera distribution per instance,
   merge-conflict rate.
3. 3D coherence: spatial compactness of instance support; agreement with the
   DINOv3 base where vocabularies overlap.
4. Object sanity: each car and tree in the scene receives its own id.
5. OOV probes: concepts outside ADE20K are found when present.
6. Context probes: SAM3 stuff quality versus the DINOv3 base, deciding
   whether the DINOv3 route is retired entirely or kept for stuff context.
7. Hierarchy sanity: expected part_of pairs (window under building) appear;
   no cycles.

## Compute strategy

- Local machine (RTX 4060, 8 GB): code and unit tests only. Tests exercise
  pure functions and the mocked SAM3 backend; no checkpoint download, no
  CUDA. This mirrors the repository's existing test style.
- Cluster (Slurm): real SAM3 inference and FlashSplat passes. SAM3 is about
  0.85B parameters, an order of magnitude lighter than the DINOv3 ViT-7B
  already in production. New sbatch entry point follows the existing
  scheduler conventions; SAM3 setup is documented in EXTERNAL_REPOS.md with a
  pinned model revision.

## Risks

- SAM3 accepts short noun phrases, not long referring expressions; the
  vocabulary must stay phrase-shaped.
- Prompt granularity defines node granularity; the vocabulary is curated per
  scene against the GNN's needs.
- Association thresholds may need tuning; the pilot reports the sensitivity
  of instance counts to the Jaccard threshold before any threshold is frozen.
- Exact SAM3 API details are verified at cluster integration time; the
  backend interface isolates any API drift from the rest of the route.
