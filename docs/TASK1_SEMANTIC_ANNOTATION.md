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
-> assign Gaussian ownership by confidence-weighted multi-view agreement
-> prune tiny/low-confidence final labels automatically
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
  - preserves `class`, `phrase`, and confidence metadata

- `scripts/cluster_semantic_flashsplat_proposals.py`
  - merges lifted proposals only when they have the same class
  - merges stuff classes such as `ground`, `road`, `sidewalk`, and `sky`
  - prunes tiny final labels after 3D assignment
  - creates instance labels such as `bicycle_01`, `tree_02`, `bench_01`
  - resolves ambiguous ownership using confidence-weighted multi-view support
  - penalizes one-view groups and uses class priority only as an exact tie-break
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
  - provides one class-aware palette shared by overlays and debug PLYs
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
```

`RESET_OUTPUT=1` is the batch-script default. It recreates the scene output
directory before every run so stale masks and proposals cannot leak into a new
result. Set `RESET_OUTPUT=0` only while debugging a failed stage.

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
