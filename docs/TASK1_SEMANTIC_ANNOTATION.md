# Task 1: Semantic Annotation of EyeNavGS 3DGS Scenes

Task 1 assigns one object/semantic label to every Gaussian in each EyeNavGS scene.
The deliverable for each scene is:

```text
semantic_point_cloud.ply
label_map.json
overlay_renders/
```

## Output Contract

The semantic PLY must preserve the original 3DGS Gaussian properties and add one
new scalar integer property:

```text
property int label
```

Do not modify the original model PLY in place. Write a separate semantic output
file.

The label map must use stable integer IDs:

```json
{
  "scene": "bicycle",
  "labels": [
    {"id": 0, "name": "unlabeled", "class": "unlabeled"},
    {"id": 1, "name": "ground", "class": "ground"}
  ]
}
```

## Pilot Scene

Default pilot scene: `bicycle`.

Fallback rule: use the first scene that has:

- a complete 3DGS `point_cloud.ply`
- usable camera/render metadata
- enough visual structure to validate labels from overlays

Current data/model status:

- EyeNavGS Rutgers and NTHU trace repos are cloned under the cluster `external/`
  directory.
- `/lab/haoq_lab/cse12312032/data/EyeNavGS/Rutgers` and `NTHU` are symlinks to
  those cloned trace repos.
- Trace inventory can be checked with:

  ```bash
  python scripts/inventory_eyenavgs_traces.py
  ```

  Current inventory notes:

  - Rutgers: 12 scenes, 264 CSVs, 2,481,596 rows.
  - NTHU: 12 expected scenes plus one anomalous `trian` folder, 289 CSVs,
    1,384,969 rows.
  - `trian/user21_trian.csv` has no matching `scene_setting.csv` entry and
    should not be merged into `train` without manual confirmation.

- GraphDeco official pretrained models are downloaded with
  `scripts/download_graphdeco_models.sh`.
- After extraction, discover available PLY files with:

  ```bash
  python scripts/find_point_clouds.py /lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco
  ```

  Build the initial scene/model manifest with:

  ```bash
  python scripts/build_task1_manifest.py
  ```

  Current GraphDeco model manifest:

  - 26 `point_cloud.ply` files extracted under
    `/lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco`.
  - 8/12 EyeNavGS scenes match official GraphDeco pretrained models:
    `truck`, `treehill`, `train`, `stump`, `room`, `playroom`, `drjohnson`,
    `bicycle`.
  - The four missing scenes are `nyc`, `london`, `berlin`, and `alameda`;
    these are ZipNeRF-derived scenes and require a separate 3DGS model source.
  - Manifest JSON was saved to:
    `/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/manifests/task1_scene_manifest_graphdeco.json`.
  - Pilot scene candidate: `bicycle`, iteration 30000, 6,131,954 Gaussians.

## Pipeline

1. Locate the scene model:

   ```text
   <scene>/point_cloud/iteration_30000/point_cloud.ply
   ```

2. Inspect the PLY:

   ```bash
   python scripts/inspect_ply.py /path/to/point_cloud.ply --json
   ```

3. Render representative RGB views from the original 3DGS model.

4. Generate semantic 2D masks automatically:

   - GroundingDINO proposes class-aware boxes from the configured vocabulary.
   - SAM converts those boxes into binary masks.
   - The mask metadata carries `class`, phrase, and confidence forward.

5. Lift semantic masks/groups to Gaussian labels:

   - Run FlashSplat to convert each 2D mask into a sparse Gaussian support.
   - Fuse same-class 3D supports into object/stuff groups.
   - Convert groups into the project label schema.

6. Prune and validate labels:

   - Drop tiny/low-confidence labels automatically after final assignment.
   - Compact label IDs so there are no gaps.
   - Create semantic overlay renders and contact sheets for QA.
   - Keep `0` as `unlabeled`.

7. Refine important objects with SAGA only if the automatic baseline fails:

   High-priority object types:

   - people
   - animals
   - vehicles
   - doors
   - signs

   Medium-priority object types:

   - buildings
   - trees
   - furniture
   - screens

   Low-priority object types:

   - ground
   - sky
   - walls
   - road

8. Export final scene artifacts:

   ```text
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/<scene>/semantic_point_cloud.ply
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/<scene>/label_map.json
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/<scene>/overlay_renders/
   ```

## Validation Criteria

For a scene to count as Task 1 complete:

- every Gaussian has a `label` value
- `label_map.json` covers every nonzero label ID
- overlay renders show plausible object boundaries from several viewpoints
- obvious high-priority objects are not merged into background labels
- the scene has a short notes file describing known weak labels

## Baseline PLY Round Trip

Before running semantic tools, verify that the repo can safely add a label column:

```bash
python scripts/add_dummy_labels.py /path/to/point_cloud.ply /tmp/semantic_test.ply --label 0
python scripts/inspect_ply.py /tmp/semantic_test.ply --json
```

The output PLY should have the same vertex count and all original properties,
plus `label`.

## Bicycle Pilot Bootstrap

Run the first full bootstrap on the cluster from the project checkout:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
bash scripts/build_graphdeco_rasterizer.sh
sbatch scripts/slurm_task1_bicycle_pilot.sbatch
```

Default resources target the L40 node:

```text
partition=a100
qos=a100
nodelist=l40gpu002
```

If that node is busy, submit the same job on the RTX8000 node:

```bash
sbatch --partition=titan --qos=titan --nodelist=rtx8000 scripts/slurm_task1_bicycle_pilot.sbatch
```

Expected first-stage outputs:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/point_cloud_inspection.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud.ply
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud_inspection.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/label_map.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/rgb_renders/
```

The rendered PNGs are RGB sanity frames from `cameras.json`. They are generated
without original ground-truth images, because the pretrained GraphDeco model
folder already contains camera intrinsics/extrinsics but its saved `cfg_args`
points to the original author's local image path. The renderer prefers the
official GraphDeco rasterizer submodule, which must be built once with
`scripts/build_graphdeco_rasterizer.sh`; this avoids using the
Gaussian-Grouping-modified rasterizer for plain RGB sanity renders.

The default pilot render uses a conservative camera set selected by
`scripts/check_graphdeco_projection.py`:

```text
41,43,148,70,115,68,64,87,69,39,85,120,42,153,123,88,37,40,7,111
```

These views avoid the worst near-plane/radius cases in the bicycle model. To
try a higher-resolution run after the 320px sanity pass:

```bash
sbatch --export=ALL,RENDER_MAX_WIDTH=640 scripts/slurm_task1_bicycle_pilot.sbatch
```

To return to evenly spaced camera sampling, submit with an empty camera list:

```bash
sbatch --export=ALL,RENDER_CAMERA_INDICES=,RENDER_COUNT=20 scripts/slurm_task1_bicycle_pilot.sbatch
```

## Automatic GroundingDINO/SAM Semantic Pipeline

The active Task 1 route is semantic 2D proposal generation, not manual prompt
points and not manual object naming:

```text
render selected 3DGS views
-> run GroundingDINO with a scene vocabulary
-> run SAM on GroundingDINO boxes to get semantic masks
-> lift each semantic mask to sparse Gaussian supports with FlashSplat
-> discard negligible FlashSplat support and fuse same-class evidence across views
-> accumulate positive and negative per-class visibility across the rendered views
-> assign Gaussian ownership by confidence-weighted multi-view agreement
-> prune tiny/low-confidence final labels automatically
-> remove disconnected 3D islands from thing labels using adaptive Gaussian-scale voxels
-> merge accepted same-class thing IDs when connected in 3D without splitting them
-> export label_map.json, semantic_point_cloud.ply, and debug artifacts
```

This path automatically carries class names from GroundingDINO into the final
3D Gaussian labels. The class vocabulary is tracked in:

```text
configs/task1_semantic_classes.example.json
```

Run the bicycle semantic quality pass. The tracked defaults are 50 evenly
spaced views, 960-pixel render width, and up to 32 detections per view:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
sbatch scripts/slurm_task1_bicycle_grounded_sam.sbatch
```

### Automatic Targeted Camera Expansion

The scheduled workflow can automatically expand the accepted 50-view bicycle
baseline. It does not require a person to choose camera IDs:

1. Sample the model geometry and measure projection characteristics for every
   unused camera.
2. Anchor safety thresholds to the 50 cameras that already rendered
   successfully, rejecting candidate views with substantially worse near-plane,
   frustum, or projected-scale behavior.
3. Choose a pose-diverse pool of safe candidates.
4. Render the accepted semantic labels and matching RGB images from those
   candidates.
5. Measure candidate image-space overlay coverage.
6. Select the lowest-coverage candidates with an 80/20 coverage-need versus
   pose-novelty score, then append them to the original 50 camera indices.
7. Run GroundingDINO, SAM, FlashSplat lifting, signed fusion, export, and
   validation on the combined camera set.

The first comparison is configured for 100 screened candidates and 20 targeted
additions, producing a 70-view semantic run:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
sbatch --export=ALL,OUTPUT_NAME=bicycle_semantic_targeted_v1,AUTO_TARGET_VIEW_COUNT=20,TARGET_CANDIDATE_COUNT=100,TARGET_SELECTION_SOURCE=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle scripts/slurm_task1_bicycle_grounded_sam.sbatch
```

Do not set `REUSE_SOURCE_OUT` for this experiment: the added views require new
masks, proposals, and class evidence. Selection records are written under:

```text
validation/camera_selection/candidate_screen.json
validation/camera_selection/candidate_visible_coverage.json
validation/camera_selection/targeted_camera_selection.json
validation/camera_selection/final_camera_indices.txt
```

Targeted-camera job `92426` completed in 9m05s and passed structural validation.
The selector found 144 safe unused cameras, rendered a 100-camera candidate
pool, and added 20 views automatically. The run produced 599 kept masks and 583
FlashSplat proposals across 70 cameras.

Compared on the same original 50 cameras, pooled visible coverage changed only
from `0.9244532306561101` to `0.9242405890153753`, while the worst original view
improved from `0.7032191672903455` to `0.7281843725791416`. The 20 deliberately
difficult additions measured `0.8244914556006355` pooled coverage, with minimum
`0.7465320176993474`. Focused and full contact sheets show no obvious return of
the bicycle-colored road streaks or bench-colored vegetation leakage.

Per-Gaussian comparison found 253,800 gains from baseline label 0 and 139,975
losses to label 0, a net increase of 113,825 labeled Gaussians. Of the 37,579
class changes, the largest was 10,229 baseline fence Gaussians reassigned to
bench; the accepted overlays indicate this corrects the earlier `bench fence`
ambiguity. Job `92426` is therefore accepted as the current bicycle baseline.

### Scene-Configurable Workflow And Train Pilot

`scripts/slurm_task1_semantic_scene.sbatch` generalizes the accepted pipeline
with configurable `SCENE`, `MODEL_DIR`, `OUTPUT_NAME`, `CLASS_CONFIG`,
`FOCUS_CLASSES`, and `FOCUS_NAME`. The existing bicycle script exports bicycle
defaults and delegates to this generic scheduler, preserving its old command.

When `CLASS_CONFIG` is not provided, the generic script first looks for:

```text
configs/task1_semantic_classes.<scene>.json
```

It falls back to `configs/task1_semantic_classes.example.json`. Focus artifacts
are optional and use scene-neutral filenames.

The `train` pilot uses an official matched model with 1,026,508 Gaussians and
301 cameras. The tracked vocabulary is:

```text
configs/task1_semantic_classes.train.json
```

The initial 50-view job `92467` completed structurally, but pruned its strongly
supported sky group because 8,451 assigned Gaussians fell below the default
10,000-Gaussian stuff threshold. Reuse job `92469` lowered that threshold to
8,000 and established the accepted baseline:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/train
  -> ../train_semantic_baseline_v2
```

Job `92469` retains two train instances of 428,735 and 18,476 Gaussians and adds
an 8,462-Gaussian sky label without admitting any previously pruned thing
group. The final unlabeled ratio is `0.4414675774567758`. Across the 50 source
views, the overlay-difference proxy is `0.9931697851099566` pooled with a
minimum of `0.9473586105137582`. Visual QA shows coherent train labeling and
corrected sky ownership. These coverage numbers measure visible semantic tint;
they are not semantic ground truth or gaze-hit accuracy.

Automatic targeted expansion must use this accepted train result as its
scene-local `TARGET_SELECTION_SOURCE`; bicycle evidence must not be reused.
Job `92470` performed that expansion and added 20 low-coverage/pose-diverse
cameras. It remains diagnostic because its default stuff threshold pruned sky
at 8,941 Gaussians and a 6,831-Gaussian shipping-container false positive was
retained as `building`.

Assigned-group pruning adapts automatically from cross-view support. A stuff
group observed in at least half of the proposal manifest's source views uses
75% of the normal stuff cutoff; other stuff and all thing groups retain their
normal thresholds. This rule is shared by every scene. `building` is part of
the canonical stuff ontology, consistent with panoptic labeling rather than
object-instance labeling. For job `92470`, the rule retains sky (59/70 views,
8,941 Gaussians, automatic threshold 7,500) and prunes the shipping-container
false building (14/70 views, 6,831 Gaussians, threshold 10,000) without repeating
camera selection, GroundingDINO/SAM, or FlashSplat.

Corrected reuse job `92482` confirmed the automatic fusion outcome, with 8,943
sky Gaussians retained and the 9,505-Gaussian merged building candidate pruned.
Its first 50 overlays are visually clean. The run also exposed a generic reuse
validation issue: an empty camera list previously fell back to 50 evenly spaced
cameras instead of the reused manifest's 70 cameras. Reuse mode now inherits
the exact camera-index list and view count from `grounded_sam_manifest.json`
unless the caller explicitly supplies a camera list.

Final job `92483` inherited the complete manifest and validated all 70 semantic
and 70 train-versus-track overlays. It is accepted at:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/train
  -> ../train_semantic_targeted_v3
```

The accepted result has an unlabeled ratio of `0.42898447941954665`. Its pooled
70-view overlay-difference proxy is `0.9902516319130663`, with minimum
`0.8961593702418886`. The original 50 views measure `0.9955632426577927`; the
20 targeted additions measure `0.9769726050512503`. Visual QA passed with no
building/container false label and no obvious train-colored background leakage.

### Room baseline

`room` is the smallest remaining matched model, with 1,593,376 Gaussians and
311 cameras. Its scene vocabulary covers salient furniture and indoor surfaces:

```text
configs/task1_semantic_classes.room.json
```

The initial run uses 50 evenly spaced cameras. Automatic targeted expansion is
enabled only after that baseline is visually accepted.

Job `92490` completed structurally, with 14 nonzero labels and an unlabeled
ratio of `0.7256222009117748`. Its pooled 50-view overlay-difference proxy is
`0.6700577548576253`, with minimum `0.07734897022925243`. Visual QA shows clean
sofa/table focus and plausible furniture/surface labels, but the initial
vocabulary omitted the prominent piano, television, speakers, media console,
and curtains. It is therefore diagnostic, not accepted.

The corrected room vocabulary includes those missing classes and uses
piano-versus-television focused artifacts. GroundingDINO can emit only part of
a multiword prompt; unique partial phrases now resolve back to the configured
class, while unmatched phrases are rejected rather than becoming arbitrary new
classes.

Corrected job `92492` reran detection and FlashSplat and is accepted at:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/room
  -> ../room_semantic_baseline_v2
```

It validates all 1,593,376 Gaussians with 18 nonzero labels. The unlabeled ratio
is `0.6849419095053522`. Its pooled 50-view overlay-difference proxy is
`0.8114288519054195`, with minimum `0.5680983763195268`, compared with
`0.6700577548576253` pooled and `0.07734897022925243` minimum for diagnostic
job `92490`. Visual QA shows consistent piano/television localization and
plausible room-wide labels; minor television spill remains on adjacent geometry
in the hardest close views. These measurements are visible-overlay proxies, not
semantic ground truth or gaze-hit accuracy.

The next room checkpoint is a fresh 20-view automatic targeted expansion from
the accepted baseline; it must not reuse the old masks or proposals.

Targeted job `92498` completed structurally with 70 views, 21 nonzero labels,
and an unlabeled ratio of `0.6529877442612415`. Its pooled overlay-difference
proxy is `0.8067012463456203`, with minimum `0.6063147492490233`. The separated
palette confirms that curtain and chair are distinct, but it also reveals table
geometry colored as television. The offending TV groups are dominated by
multi-class detector phrases such as `television stand table desk`; the run is
diagnostic and is not the accepted room result.

Phrase resolution now removes shorter matches contained within a specific
compound prompt, then rejects any phrase that still spans unrelated configured
classes. Fusion also consolidates connected same-class fragments before the
global thing-size cutoff. These are scene-neutral precision and ordering fixes,
not per-class thresholds. A fresh detection run is required.

Corrected job `92513` is accepted at:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/room
  -> ../room_semantic_targeted_v2
```

It validates all 1,593,376 Gaussians across 70 views with 18 nonzero labels and
an unlabeled ratio of `0.6658127146386038`. The pooled overlay-difference proxy
is `0.8138248674513162`. Original views measure `0.8422301555618091` pooled
with minimum `0.6223584924182434`; added views measure `0.742811647175084`
pooled with minimum `0.3502791212099397`. Visual QA confirms a clean television
label, a separate table label, and distinct curtain/chair semantics.

### Truck baseline

`truck` is the fourth-scene pilot, with 2,541,226 Gaussians and 251 cameras.
Its vocabulary is:

```text
configs/task1_semantic_classes.truck.json
```

The initial run uses 50 evenly spaced views and truck-versus-wheel focus.

For a short smoke run with three selected cameras:

```bash
SEMANTIC_CAMERA_INDICES='70,115,68' SEMANTIC_VIEW_COUNT=3 MAX_DETECTIONS_PER_VIEW=12 MASK_BATCH_SIZE=4 \
  sbatch --export=ALL,SEMANTIC_CAMERA_INDICES,SEMANTIC_VIEW_COUNT,MAX_DETECTIONS_PER_VIEW,MASK_BATCH_SIZE \
  scripts/slurm_task1_bicycle_grounded_sam.sbatch
```

Active semantic scripts:

- `scripts/generate_grounded_sam_masks.py`
  - renders views from `cameras.json`
  - runs GroundingDINO with the configured class vocabulary
  - runs SAM on each detected box
  - writes compressed per-view mask stacks for FlashSplat
  - writes every SAM detection as an 8-bit black/white PNG (`0` background,
    `255` mask) under `binary_masks/`
  - writes semantic mask overlays with class names and detection scores

- `scripts/run_flashsplat_mask_proposals.py`
  - loads either SAM-auto or GroundingDINO/SAM masks
  - batches mask IDs through FlashSplat
  - writes sparse Gaussian support files per semantic mask proposal
  - rejects negligible raster contributions with a default support threshold of
    `0.05`
  - records positive views inside each class mask and negative views where the
    same Gaussian contributes outside that mask
  - preserves `class`, `phrase`, and confidence metadata

- `scripts/cluster_semantic_flashsplat_proposals.py`
  - merges lifted proposals only when they have the same class
  - merges stuff classes such as `ground`, `road`, `sidewalk`, and `sky`
  - prunes tiny final labels after 3D assignment
  - creates instance labels such as `bicycle_01`, `tree_02`, `bench_01`
  - resolves ambiguous ownership using confidence-weighted multi-view support
  - requires thing-label Gaussians to have at least two positive views and a
    positive visibility ratio of at least `0.50`
  - penalizes one-view groups and uses class priority only as an exact tie-break
  - the bicycle quality pass requires at least two proposals per group and a
    minimum ownership quality of `0.08` to suppress one-view background leakage
  - removes small disconnected 3D components from thing labels while leaving
    ground, road, sky, vegetation, and other stuff classes unchanged
  - consolidates accepted same-class IDs when a sufficiently large connected
    3D component contains both IDs
  - treats every accepted input instance as atomic, so consolidation cannot
    split an existing label or reduce semantic coverage
  - records source-label contributions, component bounds, and before/after
    instance counts in `semantic_group_summary.json`
  - writes final D1-style `semantic_point_cloud.ply` and `label_map.json`

- `scripts/export_debug_label_colors.py`
  - appends `red`, `green`, and `blue` properties to a separate debug PLY
  - useful for generic PLY viewers that read vertex RGB columns
  - keeps the final deliverable PLY contract unchanged

- `scripts/export_supersplat_label_colors.py`
  - writes a separate SuperSplat-compatible debug PLY
  - bakes label colors into `f_dc_0`, `f_dc_1`, and `f_dc_2`
  - clears `f_rest_*` so SuperSplat shows semantic label colors instead of the
    original 3DGS appearance
  - can focus selected classes while dimming all other labels

- `scripts/semantic_palette.py`
  - provides one deterministic label-aware palette shared by overlays and debug
    PLYs
  - preserves configured class colors when they remain perceptually separated
  - replaces collisions through farthest-point selection in CIELAB space with
    a target Delta E of 30, including separate hues for same-class instances
  - uses red for bicycle and blue for bench in every validation artifact

- `scripts/summarize_task1_semantic_run.py`
  - writes one `pipeline_run_summary.json` across mask generation, FlashSplat
    lifting, 3D fusion/pruning, exports, and validation

Expected semantic outputs:

```text
bicycle_semantic/
  stages/
    01_grounded_sam/
      rgb_renders/
      mask_stacks/
      binary_masks/
      overlays/
      grounded_sam_manifest.json
    02_flashsplat/
      proposal_supports/
      proposal_manifest.json
    03_semantic_fusion/
      gaussian_labels.npy
      semantic_group_summary.json
  deliverables/
    semantic_point_cloud.ply
    label_map.json
  visualizations/
    ply/
      semantic_point_cloud_rgb_debug.ply
      semantic_point_cloud_supersplat_debug.ply
      bicycle_vs_bench_supersplat_debug.ply
    overlays/
      semantic_labels/
      bicycle_vs_bench/
    contact_sheets/
  validation/
    semantic_point_cloud_inspection.json
    task1_validation.json
    pipeline_run_summary.json
  logs/
```

Each executable stage also has its own text log under `logs/`. Machine-readable
stage records are:

```text
stages/01_grounded_sam/grounded_sam_manifest.json
stages/02_flashsplat/proposal_manifest.json
stages/03_semantic_fusion/semantic_group_summary.json
visualizations/ply/*.json
validation/task1_validation.json
validation/pipeline_run_summary.json
validation/visible_overlay_coverage.json
```

`RESET_OUTPUT=1` is the batch-script default. It recreates the scene output
directory before every run so stale masks and proposals cannot leak into a new
result. Set `RESET_OUTPUT=0` only while debugging a failed stage.

To test fusion or instance-consolidation changes without rerunning
GroundingDINO, SAM, and FlashSplat, write to a separate output and reuse the
accepted source stages:

```bash
sbatch --export=ALL,OUTPUT_NAME=bicycle_semantic_identity_test,REUSE_SOURCE_OUT=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_semantic \
  scripts/slurm_task1_bicycle_grounded_sam.sbatch
```

This mode symlinks the source mask/proposal stages into the test output, reruns
fusion and all validation visualizations, and never modifies the source result.

### Bicycle signed-evidence baseline

Slurm job `92344` completed from commit `385f8ba` on the RTX 8000:

- 50 evenly spaced views at 960-pixel width
- 411 GroundingDINO + SAM masks and 399 FlashSplat proposals
- signed evidence for 14 semantic classes
- 6,131,954 Gaussians and 25 nonzero output labels
- structural validation status `ok`
- unlabeled ratio `0.6025267`

Compared with spatial-only job `92331`, the signed negative evidence removes
the visible bicycle-colored road streaks and bench-colored vegetation patches.
The foreground silhouettes remain stable across the 50-view contact sheet.
The remaining known issues are conservative unlabeled coverage and fragmented
same-class instance IDs; the identity-test run above validates the automatic
connected-component consolidation before it becomes the new baseline.

Identity-test job `92353` demonstrated why accepted labels must remain atomic:
it merged `bicycle_01/02` into one 123,177-Gaussian bicycle and merged four bench
fragments into one 120,971-Gaussian bench, with clean 50-view overlays. However,
it also split 13 tree labels into 26 connected components. Subsequent minimum-
size pruning increased the unlabeled count by 95,610, almost entirely from tree
labels. Job `92353` is therefore diagnostic only. The corrected merge-only rule
uses connected components to union source IDs but never subdivides an accepted
source label.

Corrected identity-test job `92369` passed all acceptance checks and remains the
preserved 50-view comparison baseline:

- one `bicycle_01` with 123,197 Gaussians
- one `bench_01` with 120,977 Gaussians
- all 932,516 tree-labeled Gaussians retained and consolidated from 13 to 8 IDs
- 3,694,666 unlabeled Gaussians, exactly matching job `92344`
- clean bicycle/bench overlays across the same 50 views
- structural validation status `ok`

Automatic targeted-camera job `92426` is the current accepted bicycle output:

- 70 views: the original 50 plus 20 automatically selected additions
- one `bicycle_01` with 119,570 Gaussians
- one `bench_01` with 139,727 Gaussians
- eight tree IDs totaling 1,096,092 Gaussians
- 3,580,841 unlabeled Gaussians (`0.5839641001873138`)
- structural validation status `ok`
- clean focused and full semantic contact sheets

The accepted output is exposed without copying large artifacts, while the job
`92369` directory remains unchanged:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle
-> ../bicycle_semantic_targeted_v1
```

Raw Gaussian coverage is not the same as rendered surface coverage. The
pipeline therefore writes `validation/visible_overlay_coverage.json`, comparing
the original RGB renders with their semantic overlays. This is an image-space
proxy, not semantic ground truth: it measures where a visible overlay changed a
rendered pixel and may undercount labels whose palette color resembles the
original RGB value.

For comparison bicycle job `92369`, the report measured 28,162,913 changed pixels
out of 30,464,400 across 50 views: a pooled visible-coverage proxy of
`0.9244532306561101`. Per-frame coverage ranged from `0.7032191672903455` to
`0.9996783130473602`, with median `0.9522204606031959` and p10/p90 values of
`0.7895937553340949`/`0.9990764630191306`. This is substantially higher than
the raw nonzero-label Gaussian ratio of `0.3974733013326583` (the complement of
the `0.6025266986673417` unlabeled ratio). The measurement does not establish
semantic correctness, representative gaze coverage, or equivalence between a
changed pixel and a labeled Gaussian. Do not add label propagation solely to
reduce the raw label-0 count; first inspect the low-coverage views and measure
downstream gaze-hit behavior.

For accepted job `92426`, the full 70-view report measures
`0.895740836611164` pooled visible coverage. This lower aggregate is expected
because the selector deliberately adds difficult views. On the common original
50 cameras, coverage is `0.9242405890153753`; the 20 additions measure
`0.8244914556006355`. Compare like-for-like camera subsets rather than treating
the 50-view and 70-view pooled values as directly interchangeable.

Example final label map:

```json
{
  "scene": "bicycle",
  "labels": [
    {"id": 0, "name": "unlabeled", "class": "unlabeled"},
    {"id": 1, "name": "bicycle_01", "class": "bicycle"},
    {"id": 2, "name": "tree_01", "class": "tree"},
    {"id": 3, "name": "ground", "class": "ground"}
  ]
}
```

The class vocabulary is still an input prompt list, but the per-object naming is
automatic. For scenes with missing categories, update the class config and
rerun; do not manually edit the final labels for the primary D1 pipeline.

## Fallback SAM/FlashSplat Proposal Pilot

The older baseline is automatic class-agnostic mask proposal generation:

```text
render selected 3DGS views
-> run SAM automatic mask generation per view
-> lift each mask proposal to sparse Gaussian supports with FlashSplat
-> cluster overlapping 3D supports into object groups
-> export label_map_auto.json and semantic_point_cloud_auto.ply
```

Run the bicycle class-agnostic proposal pilot:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
sbatch scripts/slurm_task1_bicycle_auto_proposals.sbatch
```

For short test runs with a comma-separated camera list, set the environment
variables before `sbatch`; do not put the comma-valued list directly inside
`--export`, because Slurm splits `--export` on commas:

```bash
AUTO_CAMERA_INDICES='70,115,68' AUTO_VIEW_COUNT=3 MAX_MASKS_PER_VIEW=8 MASK_BATCH_SIZE=4 \
  sbatch --export=ALL,AUTO_CAMERA_INDICES,AUTO_VIEW_COUNT,MAX_MASKS_PER_VIEW,MASK_BATCH_SIZE \
  scripts/slurm_task1_bicycle_auto_proposals.sbatch
```

Active scripts:

- `scripts/generate_sam_auto_masks.py`
  - renders views from `cameras.json`
  - runs `SamAutomaticMaskGenerator`
  - filters tiny and near-full-image masks
  - writes per-view proposal stacks and mask overlays

- `scripts/run_flashsplat_mask_proposals.py`
  - loads per-view SAM masks
  - batches mask IDs through FlashSplat
  - writes sparse Gaussian support files per mask proposal

- `scripts/cluster_flashsplat_proposals.py`
  - greedily merges proposal supports with high 3D overlap
  - exports automatic object candidate labels
  - writes `semantic_point_cloud_auto.ply`

Expected outputs:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/sam_auto/
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/flashsplat_proposals/
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/semantic_point_cloud_auto.ply
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/label_map_auto.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/auto_group_summary.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/sam_auto_contact_sheet.png
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/label_overlay_renders/
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/label_overlay_contact_sheet.png
```

The fallback output labels are named `object_group_###` with class
`object_candidate`. This is useful for finding missed objects or diagnosing
GroundingDINO failures, but it is no longer the preferred automatic D1 path
because it has no semantic class names.

## Optional Review and Finalize Fallback Groups

The review tools remain available only for fallback or QA. Create a review CSV
from class-agnostic automatic group outputs:

```bash
python scripts/create_label_review_template.py \
  --auto-label-map /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/label_map_auto.json \
  --auto-summary /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/auto_group_summary.json \
  --output /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/label_review.csv
```

Review `label_overlay_contact_sheet.png` and the overlay render folder, then
edit these CSV columns:

- `final_id`: final object/semantic id. Use the same id on multiple source
  rows to merge groups. Use `0` to drop a source group back to unlabeled.
- `final_name`: human-readable final label such as `bicycle_01`, `tree_03`,
  `ground`, or `sky`.
- `final_class`: semantic class such as `bicycle`, `tree`, `ground`, or `sky`.
- `review_status` and `notes`: optional bookkeeping for uncertain groups.

Apply the reviewed CSV to generate the D1-style outputs:

```bash
python scripts/apply_label_review.py \
  --auto-labels /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/gaussian_labels_auto.npy \
  --review-csv /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_auto/label_review.csv \
  --input-ply /lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco/bicycle/point_cloud/iteration_30000/point_cloud.ply \
  --output-dir /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle \
  --scene bicycle \
  --overwrite
```

Expected final outputs:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud.ply
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/label_map.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/gaussian_labels.npy
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/label_review_summary.json
```

Validate the final scene package before counting it as complete:

```bash
python scripts/validate_task1_outputs.py \
  --labels-npy /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/gaussian_labels.npy \
  --label-map /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/label_map.json \
  --semantic-ply /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud.ply
```

The validator intentionally fails if any nonzero final class is still
`object_candidate`. Pass `--allow-object-candidate` only for debugging an
unreviewed automatic output.

The old prompt-based bicycle pilot is archived under:

```text
archive/task1_prompt_pilot/
```

Current automatic pilot result:

- Slurm job `91872` completed on `rtx8000`.
- Test scope: 3 views, 8 SAM masks per view.
- SAM generated 24 lifted mask proposals.
- Greedy 3D clustering produced 6 automatic object/stuff candidate groups.
- Label histogram:

  ```json
  {"0": 5021685, "1": 339902, "2": 253174, "3": 45990, "4": 47164, "5": 61232, "6": 362807}
  ```

- The label overlay contact sheet confirms the automatic pipeline works end to
  end, but this is still a baseline: groups are broad candidates and need
  semantic naming plus refinement before the scene can count as fully validated.
