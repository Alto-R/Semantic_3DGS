# Semantic Annotation Pipeline

This document describes the maintained Task 1 pipeline from a pretrained 3DGS
scene to a reviewed semantic output. It distinguishes automatic ADE20K
refinement from custom classes that are absent from the DINOv2 ontology.

## 1. Inputs and invariants

Each scene requires:

- a pretrained Gaussian point cloud;
- the scene's real `cameras.json` camera definitions;
- the original RGB images used by those cameras;
- the repository's Python environment and external model repositories.

The workspace is expected to contain:

```text
<workspace>/
  projects/<repository>/
  data/3dgs_models/graphdeco/<scene>/
  external/
  outputs/eyenavgs_task1/
```

The schedulers derive these roots from the repository location. A job may
override a resolved path through an environment variable, but tracked files do
not embed user, host, or cluster-specific paths.

The pipeline follows four invariants:

1. A published base is immutable.
2. Every refinement writes a new versioned output.
3. Weak or conflicting evidence may abstain instead of forcing a label.
4. Custom classes require visual review before they can modify a base.

## 2. Build the DINOv2 semantic base

Scheduler: `scripts/slurm/slurm_task1_dinov2_scene.sbatch`

The base stage is deliberately prompt-free:

1. Render the exact cameras from `cameras.json`.
2. Run the official DINOv2 ViT-L/14 ADE20K linear head for every selected view.
3. Preserve per-pixel class confidence and abstain below the confidence gate.
4. Lift every view's semantic evidence onto the Gaussian point cloud with
   FlashSplat.
5. Fuse the disk-backed per-view votes exactly across views.
6. Apply minimum-view, agreement, evidence, and adaptive class-support gates.
7. Publish labels, maps, diagnostics, overlays, and validation reports.

Production settings are:

```text
VIEW_COUNT=0
FUSION_MODE=separate_abstain
MIN_PIXEL_CONFIDENCE=0.50
MIN_VIEWS=2
MIN_AGREEMENT=0.50
MIN_SEMANTIC_EVIDENCE=0.50
MIN_ASSIGNED_THING_GAUSSIANS=5000
MIN_ASSIGNED_STUFF_GAUSSIANS=10000
ENABLE_GROUNDINGDINO=0
```

`VIEW_COUNT=0` means all real cameras. Adaptive class thresholds remain enabled.
The separate-abstain fusion keeps visible-but-semantically-weak evidence distinct
from genuinely unseen Gaussians.

Example:

```bash
sbatch --chdir="$(pwd -P)" \
  --export=ALL,SCENE=<scene>,OUTPUT_NAME=<scene>_dinov2_separate_abstain_allviews_v1,VIEW_COUNT=0,FUSION_MODE=separate_abstain,MIN_SEMANTIC_EVIDENCE=0.50,ENABLE_GROUNDINGDINO=0,RESET_OUTPUT=0,COLOR_MODE=class \
  scripts/slurm/slurm_task1_dinov2_scene.sbatch
```

The base scheduler organizes intermediate evidence into:

```text
stages/
  01_real_camera_views/
  02_flashsplat_votes/
  03_exact_fusion/
  04_...08_/                 optional GroundingDINO stages when enabled
deliverables/
visualizations/
validation/
logs/
```

The production base keeps `ENABLE_GROUNDINGDINO=0`; the optional later stages
exist for experiments and review workflows, not for defining the base ontology.

## 3. Inspect and accept the base

Before refinement, inspect:

- original RGB views beside class overlays;
- the class contact sheet and color legend;
- fused class counts and abstained/unseen counts;
- `gaussian_labels.npy` and `label_map.json` consistency;
- validation reports and PLY vertex counts when a PLY is present.

Do not infer correctness from aggregate coverage alone. Coverage is an
image-space visibility diagnostic and can be high even when labels are wrong.

## 4. Run adaptive ADE20K refinement v5

Scheduler: `scripts/slurm/slurm_task1_ade_refinement_scene.sbatch`

Use v5 only to refine classes already represented by the DINOv2 ADE20K base. It
does not introduce missing vocabulary. The scheduler:

1. Reads the immutable DINOv2 base.
2. Builds a same-ADE-class GroundingDINO vocabulary from classes supported by
   that base.
3. Samples 50 seed views and up to 20 evidence-targeted views.
4. Produces GroundingDINO boxes and SAM masks.
5. Lifts masks into FlashSplat 3D proposal groups.
6. Merges only proposals that pass the global v5 gates.

The v5 merge uses class-neutral rules:

- a same-class anchor in the base;
- multiview support and agreement;
- adaptive 3D connected components;
- a robust geometry envelope around supported base evidence;
- competing-thing protection;
- partial-anchor completion only when overlap risk is at most `0.40` and all
  other evidence, geometry, and conflict gates pass.

Example:

```bash
sbatch --chdir="$(pwd -P)" \
  --export=ALL,SCENE=<scene>,SOURCE_NAME=<immutable-base-name>,OUTPUT_NAME=<versioned-v5-name> \
  scripts/slurm/slurm_task1_ade_refinement_scene.sbatch
```

To change only merge policy while reusing cached detections and masks, use:

```bash
sbatch --chdir="$(pwd -P)" \
  --export=ALL,SCENE=<scene>,SOURCE_NAME=<immutable-base-name>,CACHE_NAME=<cached-refinement-name>,OUTPUT_NAME=<new-versioned-name> \
  scripts/slurm/slurm_task1_ade_refinement_replay_scene.sbatch
```

The v5 result is a conservative research refinement, not a guarantee that every
physical instance is recovered. Known unrecoverable omissions are recorded in
`docs/PROJECT_STATUS.md`.

## 5. Generate evidence for missing custom classes

Scheduler: `scripts/slurm/slurm_task1_semantic_scene.sbatch`

Use this path only when a desired object identity is absent from the ADE20K
ontology. Each class is declarative: its prompts, type, and default review state
live in `configs/task1_semantic_classes.<scene>[.<version>].json`.

The standalone source job:

1. selects 50 evenly spaced real views by default;
2. runs the configured GroundingDINO prompt group;
3. converts boxes to SAM masks;
4. lifts and fuses masks into 3D Gaussian groups;
5. publishes source overlays, counts, and review artifacts.

Example using a scene's canonical config:

```bash
sbatch --chdir="$(pwd -P)" \
  --export=ALL,SCENE=<scene>,OUTPUT_NAME=<scene>_semantic_extensions_<version>,SEMANTIC_VIEW_COUNT=50 \
  scripts/slurm/slurm_task1_semantic_scene.sbatch
```

For a non-default versioned config, derive its absolute runtime path from the
checkout rather than hardcoding it:

```bash
PROJECT_ROOT="$(pwd -P)"
CLASS_CONFIG="${PROJECT_ROOT}/configs/task1_semantic_classes.<scene>.v2.json"
sbatch --chdir="${PROJECT_ROOT}" \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",SCENE=<scene>,OUTPUT_NAME=<source-name>,SEMANTIC_VIEW_COUNT=50,CLASS_CONFIG="${CLASS_CONFIG}" \
  scripts/slurm/slurm_task1_semantic_scene.sbatch
```

A source output is evidence for review. Even if it contains provisional labels
or a PLY, it is not an accepted hybrid annotation.

## 6. Review custom-class evidence

Review each proposed class independently against the original RGB views and the
DINOv2 overlay. Reject a class when any of the following occurs:

- a mask absorbs unrelated background, furniture, or another object;
- multiple physical identities collapse into one ambiguous proposal;
- no coherent multiview 3D group is produced;
- the proposed class already exists in ADE20K;
- replacing base labels would remove more reliable semantics.

Prompt variants can improve source evidence, but should not encode a particular
scene location or one known instance. Keep configurations semantic and reusable.

## 7. Merge only reviewed extensions

Scheduler: `scripts/slurm/slurm_task1_reviewed_extensions_scene.sbatch`

The reviewed merge requires an explicit class allowlist and writes a new output.
It preserves both the DINOv2 base and the source evidence. Hybrid policy lives in
`configs/task1_hybrid_extensions.<scene>[.<version>].json`.

The merge preflight checks:

- class and prompt uniqueness;
- source labels and label-map consistency;
- absence of ontology collisions;
- permitted base-label transitions;
- selected group support and replacement risk;
- output label/map consistency.

`source_class_unions` may intentionally combine reviewed source identities into
one target identity. For example, cached train evidence can union former
`railroad_track` and `railway_platform` source groups into one `railroad_track`
class; the platform is not retained as a separate semantic class.

Set the source label and label-map paths from a workspace root derived at runtime,
then submit the merge with the reviewed class allowlist. For a no-PLY merge
preflight, invoke
`python -m scripts.task1.merge.merge_semantic_extensions` with
`--no-semantic-ply`; the reviewed-extension scheduler intentionally publishes a
PLY. A final PLY is adopted only after the overlays and base-label transitions
are explicitly accepted.

## 8. Validate the final output

A final candidate must satisfy all of the following:

- one label per Gaussian;
- every assigned label exists in `label_map.json`;
- label-map identifiers are unique;
- a published PLY has the same vertex count as the source point cloud;
- the PLY `label` property is integer-valued;
- class and instance overlays align with original RGB views;
- no accepted extension shows broad unrelated-background leakage;
- provenance identifies the immutable base and every merged source.

Useful maintained modules include:

```text
scripts/task1/qa/validate_task1_outputs.py
scripts/task1/qa/summarize_task1_semantic_run.py
scripts/task1/qa/make_contact_sheet.py
scripts/task1/qa/measure_overlay_coverage.py
scripts/task1/qa/inspect_ply.py
```

## 9. Safe operational sequence

For each scene:

1. Resolve the job with `CONFIG_ONLY=1` when that scheduler supports it.
2. Submit the DINOv2 base.
3. Inspect and accept the base.
4. Run v5 only if same-ADE-class completion is useful.
5. Generate separate source evidence for genuinely missing classes.
6. Review each source class visually and quantitatively.
7. Run a no-PLY reviewed-merge preflight.
8. Inspect base-label transitions.
9. Publish a new semantic PLY only after explicit acceptance.

The user owns all Slurm submission and monitoring. Repository maintenance must
not submit or monitor cluster jobs implicitly.
