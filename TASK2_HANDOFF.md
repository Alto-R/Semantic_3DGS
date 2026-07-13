# Task 2 Handoff: Gaze-Target Auto-Annotation

Last prepared: 2026-07-13 (Asia/Shanghai)

This is the starting contract for a separate Codex task implementing Task 2 in
the same PKU 3DGS VR project. Read this file completely before acting. Verify
Git, cluster, data, and accepted-input state instead of assuming an interrupted
operation completed.

## 1. Objective And Dependency Boundary

Task 2 converts every valid EyeNavGS per-eye trace row into a gaze ray, aligns
that ray with the corresponding labeled 3DGS scene, finds its first geometry
hit, and appends a semantic gaze-target annotation.

Required D2 columns, in addition to every original input column:

```text
gaze_target_id
gaze_target_name
hit_point_xyz
hit_distance
hit_confidence
no_hit
```

Task 2 depends on accepted Task 1 semantic scenes, but it does not need to wait
for all Task 1 scenes. Begin with the accepted Rutgers `bicycle` scene. Expand
to accepted `train` and `room` only after the coordinate-alignment gate passes.

Task 3 is not part of this branch. In particular, do not apply semantic
snapping to the raw Task 2 baseline. Task 3 must compare unsnapped and snapped
results explicitly.

## 2. Repository And Branch Ownership

Canonical Windows repository:

```text
C:\Users\dhana\PROJECTS\PKU 3DGS VR
```

GitHub source of truth:

```text
https://github.com/DhanaKresnawijaya237/pku-3dgs-vr
```

Baseline immediately before this handoff was added:

```text
178b757 Document superseded Task 1 runs
```

Start from the commit containing this handoff or a later `main`. Create and use
a separate branch/worktree:

```text
codex/task2-gaze-targets
```

Do not implement Task 2 directly on the Task 1 thread's active checkout. Do not
force-push, reset, or rewrite Task 1 history.

Task 2 owns new files with Task 2-specific names, for example:

```text
docs/TASK2_GAZE_ANNOTATION.md
configs/task2_gaze_raycast.example.json
scripts/annotate_gaze_targets.py
scripts/validate_task2_outputs.py
scripts/render_task2_alignment.py
scripts/slurm_task2_gaze_annotation.sbatch
tests/test_task2_trace_loading.py
tests/test_task2_gaze_geometry.py
tests/test_task2_output_contract.py
```

Avoid modifying these Task 1-owned files unless a shared bug truly requires it
and the change is coordinated first:

```text
CURRENT_TASK_HANDOFF.md
docs/TASK1_SEMANTIC_ANNOTATION.md
configs/task1_semantic_classes.*.json
scripts/slurm_task1_*.sbatch
scripts/cluster_semantic_flashsplat_proposals.py
scripts/generate_grounded_sam_masks.py
```

The Task 1 thread currently owns the pending truck baseline and all future
accepted-scene pointer changes. Task 2 must consume accepted inputs read-only.

## 3. Non-Negotiable Operating Rules

- Use `login:false` for local PowerShell commands.
- Never submit `sbatch` as Codex. When a scheduled run is necessary, give Dhana
  the exact command and stop at that checkpoint.
- Read-only cluster inspection and job monitoring are allowed.
- Keep passwords, tokens, and private-key passphrases out of commands, files,
  logs, and Git.
- Do not commit datasets, PLYs, generated tables, renders, caches, or external
  repositories.
- Do not modify or delete anything below the Task 1 accepted paths.
- Preserve original trace rows, original column names, original order, and
  source row count. Add columns; do not silently filter or deduplicate.
- Separate measured facts, implementation choices, visual interpretation, and
  unresolved uncertainty in reports.
- Do not introduce per-class or hand-tuned per-scene hit thresholds. Spatial
  tolerance should be derived automatically from Gaussian covariance/scale or
  another documented scene-scale-aware quantity.

Cluster login and checkout:

```bash
ssh -p 10022 cse12312032@172.18.34.25
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
```

When the cluster cannot pull GitHub non-interactively, use the existing safe
bundle workflow: create a local Git bundle, copy it to `/tmp`, verify the
cluster checkout is clean, fetch it, and fast-forward with `git merge
--ff-only`. Do not recreate the deprecated cluster bare remote.

## 4. Accepted Task 1 Inputs

Task 2 may use only accepted scene pointers, never diagnostic or superseded
runs:

| Scene | Accepted pointer | Resolved Task 1 run | Gaussians |
| --- | --- | --- | ---: |
| bicycle | `/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle` | `bicycle_semantic_targeted_v1`, job `92426` | 6,131,954 |
| train | `/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/train` | `train_semantic_targeted_v3`, job `92483` | 1,026,508 |
| room | `/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/room` | `room_semantic_targeted_v2`, job `92513` | 1,593,376 |

For each scene, the required files are:

```text
<accepted-scene>/deliverables/semantic_point_cloud.ply
<accepted-scene>/deliverables/label_map.json
<accepted-scene>/validation/task1_validation.json
<accepted-scene>/validation/pipeline_run_summary.json
```

The deliverable PLY is binary little-endian. It preserves the GraphDeco
properties and adds a scalar integer `label` property. `label_map.json` maps
each integer ID to `name` and `class`. Label `0` is `unlabeled`; it is not a
background surface and must not be confused with a ray miss.

At the start of every annotation run, record in a Task 2 run manifest:

- the accepted symlink path and its resolved target;
- PLY and label-map sizes and SHA-256 hashes;
- Task 1 validation status and Gaussian count;
- Task 2 Git commit, configuration, dependency versions, site, scene, source
  file, and intersection method version.

This prevents an accepted pointer update from silently changing an existing
Task 2 result. Never overwrite an old Task 2 result when its source fingerprint
changes.

## 5. Verified Trace Inventory And Schemas

Cluster trace roots:

```text
/lab/haoq_lab/cse12312032/data/EyeNavGS/Rutgers
/lab/haoq_lab/cse12312032/data/EyeNavGS/NTHU
```

Trace layouts:

```text
Rutgers/dataset/<scene>/user*_<scene>.csv
NTHU/<scene>/*.csv
```

Measured inventory on 2026-07-13:

| Site | Scenes | CSVs | Rows |
| --- | ---: | ---: | ---: |
| Rutgers | 12 | 264 | 2,481,596 |
| NTHU | 13 directory names | 289 | 1,384,969 |

The extra NTHU directory name is `trian`: one CSV with 4,308 rows and no scene
setting. Quarantine it as a data-quality issue. Do not silently rename it to
`train` without explicit provenance and validation.

Accepted-scene trace volume available now:

| Site | bicycle | train | room | Total rows |
| --- | ---: | ---: | ---: | ---: |
| Rutgers | 157,286 | 254,008 | 256,888 | 668,182 |
| NTHU | 95,627 | 125,751 | 120,385 | 341,763 |
| Combined | 252,913 | 379,759 | 377,273 | 1,009,945 |

Both sites have one row per rendered eye view, normally left then right:

```text
ViewIndex: 0 = left, 1 = right
FOV1..FOV4
PositionX/Y/Z
QuaternionX/Y/Z/W
GazePosX/Y/Z
GazeQX/Y/Z/W
```

Do not trust alternation without validating it per file. Join and preserve rows
using the source filename plus zero-based source row index, not timestamp alone.

Known schema differences:

- Rutgers uses `Timestamp`.
- NTHU uses `timestep`.
- The ordering of gaze-position and gaze-quaternion columns differs, so access
  fields by name rather than column position.
- Rutgers `scene_setting.csv` contains `Scene_Name`, `Quaternion`, `Scale`, and
  `Initial_Position`.
- NTHU `scene_setting.csv` contains `Scene_Name`, `Quaternion`, and `Scale`, but
  no `Initial_Position`. Do not assume a zero translation without evidence.

The existing lightweight inventory command is safe on a login node:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
python scripts/inventory_eyenavgs_traces.py
```

## 6. Verified Gaze And Coordinate Evidence

Pinned EyeNavGS software:

```text
/lab/haoq_lab/cse12312032/external/EyeNavGS_Software
commit 2121d03548e126d04e824f09764df1d4fce9fcea
```

Relevant upstream references:

```text
utils/AddEyeGazeTracking/add_gaze.py
utils/WorldCoordConvert/coord_convert.py
utils/JsonCsvTraceConvert/csv_to_json.py
src/core/openxr/OpenXRRdrMode.cpp
src/core/openxr/OpenXRHMD.cpp
```

Verified facts from those files:

1. `GazeQX/Y/Z/W` is stored as an OpenXR gaze-pose orientation quaternion.
2. The official screen-overlay utility uses local forward vector `[0, 0, -1]`.
   Therefore the world gaze direction candidate is
   `Rotation.from_quat(GazeQ).apply([0, 0, -1])`.
3. The recorder stores a per-eye gaze-pose position in `GazePosX/Y/Z`; it is
   close to, but not identical to, the per-eye view `PositionX/Y/Z`.
4. Head/view and gaze positions are scaled around the initial adjustment during
   recording. Viewer movement is also added to both positions.
5. The supplied `WorldCoordConvert/coord_convert.py` is not a D2-ready
   converter: it keeps only `ViewIndex == 0`, subsamples again with `iloc[::2]`,
   smooths/drops rows, converts only head position, and does not transform gaze
   position or gaze orientation.
6. `JsonCsvTraceConvert/csv_to_json.py` flips Y and Z for a different viewer
   convention. That conversion is not, by itself, proof of the transform from
   recorded OpenXR coordinates to the semantic GraphDeco PLY.

The task brief says to use `Position` as ray origin and suggests
`GazePos - Position` as a possible direction. Actual recorder evidence makes
that second expression unsafe: `GazePos` is a gaze-pose origin, not a distant
fixation point. Treat both ray-origin candidates (`Position` and `GazePos`) as
an alignment experiment, and derive direction from `GazeQ` unless visual and
numerical validation disproves it.

## 7. Mandatory Coordinate-Alignment Gate

Do not mass-annotate any scene until this gate passes. A plausible hit rate is
not sufficient evidence; a wrong transform can still intersect geometry.

Start with one Rutgers bicycle trace because Rutgers has a complete scene
setting and bicycle has the strongest accepted Task 1 history.

Required gate procedure:

1. Parse a small, deterministic sample containing both eyes, multiple
   timestamps, and at least three separated user poses.
2. Normalize nonzero quaternions and record their pre-normalization norms.
3. Reproduce the official screen-space gaze projection using
   `R_head.inv() * R_gaze` and local forward `-Z`.
4. Derive a full position-and-direction transform into GraphDeco model space.
   Apply translation/scale to origins and rotation only to directions; normalize
   directions afterward.
5. Render the corresponding model view and overlay the projected gaze ray or
   hit point. Compare against EyeNavGS replay/overlay behavior.
6. Check numerically that transformed eye origins are plausible relative to
   scene bounds, direction norms are one, positive ray parameters point forward,
   and left/right rows from the same moment are mutually consistent.
7. Save the sample indices, transform formula, images, numerical report, and
   pass/fail decision under a versioned Task 2 validation directory.

Test transform variants explicitly; do not collapse the approximate “180°
rotation” note into an unverified hard-coded flip. The Rutgers and NTHU scene
settings differ, so each site requires its own validated transform. NTHU must
remain blocked until the missing-translation meaning is resolved from upstream
evidence or replay validation.

## 8. Raw D2 Semantic Contract

The baseline must distinguish three states:

1. Invalid gaze data: `ray_valid = false`; `no_hit` should be null/not
   applicable, with a machine-readable `no_hit_reason` such as `invalid_gaze`.
2. Valid ray with no geometry intersection: `ray_valid = true`, `no_hit = true`.
3. Valid ray with a geometry intersection: `ray_valid = true`, `no_hit = false`.

If the first geometry hit has label `0`, emit target ID `0` / name `unlabeled`
and `hit_is_labeled = false`. Do not see through an unlabeled foreground
surface to select a labeled object behind it. If “first labeled hit” is tested,
store it as a separate ablation rather than overwriting the physical first-hit
baseline.

Recommended additional provenance/diagnostic columns:

```text
source_site
source_scene
source_file
source_row_index
ray_valid
no_hit_reason
hit_is_labeled
gaze_origin_model_x/y/z
gaze_direction_model_x/y/z
intersection_method
method_version
```

Use a nullable boolean for `no_hit` in the canonical Parquet output. If CSV is
also emitted, leave `no_hit` empty for invalid rays and document the encoding.
Represent `hit_point_xyz` as a three-value list/struct in Parquet; if CSV is
emitted, use a documented JSON array or add lossless component columns.

`hit_confidence` is algorithmic confidence, not semantic accuracy. Its exact
formula must be documented, bounded, tested, and included in the run manifest.
Do not fill it with a constant or describe local density as ground truth.

Hit-rate denominators must include only `ray_valid == true`. Report separately:

- geometry-hit rate;
- labeled-hit rate among valid rays;
- label-0 first-hit rate;
- invalid-gaze rate;
- miss reasons.

## 9. Intersection Design Requirements

There is no literal hard surface in a 3DGS PLY. A single Gaussian center is not
equivalent to visible surface geometry. Implement the intersection method behind
a replaceable interface and version it.

The first candidate should use the stored Gaussian center, anisotropic scale,
rotation, and opacity rather than a fixed-radius center test. Verify GraphDeco's
parameter conventions in the pinned renderer before decoding them: stored
scales are normally log-scale, opacity is normally a logit, and quaternion
component order must not be guessed.

A reasonable staged design is:

1. Build a spatial acceleration structure over Gaussian bounds/centers.
2. Retrieve candidates intersecting or near the ray.
3. Evaluate positive forward distance and closest approach in each Gaussian's
   local ellipsoidal coordinates.
4. Apply a documented opacity/visibility rule.
5. Select the nearest supported surface region and aggregate local label
   evidence rather than trusting a single noisy Gaussian.
6. Emit diagnostics needed to compare an ellipsoid-aware method with the
   simpler KD-tree/ray-march baseline suggested by the task brief.

Any tolerance must scale automatically with Gaussian covariance or another
measured scene-scale statistic. Do not solve difficult scenes by introducing a
list of per-scene or per-class constants.

Before scaling up, benchmark at least two defensible intersection variants on
the same alignment sample and inspect disagreements. Lock a method version only
after the alignment renders and synthetic tests pass.

## 10. Implementation And Validation Milestones

### Milestone A: trace and source manifest

- Add schema-aware loaders for Rutgers and NTHU.
- Preserve every row and original field.
- Validate required columns, finite values, view indices, timestamps, gaze
  default patterns, and quaternion norms.
- Fingerprint accepted Task 1 inputs.
- Unit-test both site schemas and malformed rows.

### Milestone B: coordinate proof on Rutgers bicycle

- Implement quaternion-to-ray helpers and transform variants.
- Reproduce official 2D gaze projection.
- Generate a small alignment report and rendered overlays.
- Stop and report if alignment is ambiguous; do not hide it with a wider hit
  tolerance.

### Milestone C: intersection pilot

- Add synthetic ellipsoid/ray tests with known hits, misses, tangencies,
  behind-origin candidates, overlapping depths, label `0`, and deterministic
  tie handling.
- Run a small Rutgers bicycle trace subset.
- Inspect first-hit depth, hit labels, invalid rows, and misses.
- Confirm chunked and unchunked results are identical.

### Milestone D: one-user end-to-end D2 proof

- Annotate one complete Rutgers bicycle user CSV.
- Validate row-count and source-field preservation exactly.
- Produce canonical output, run manifest, summary metrics, and a compact visual
  sample.
- Obtain user review before a full-scene job.

### Milestone E: accepted-scene expansion

- Scale to all Rutgers bicycle users.
- Then validate/annotate Rutgers train and room.
- Resolve and independently validate NTHU alignment before processing NTHU.
- Add new Task 1 scenes only after their accepted pointer exists and is
  fingerprinted.

Minimum automated tests before committing a pilot:

```text
identity gaze quaternion produces world direction [0, 0, -1]
quaternion normalization and invalid-zero handling
position/direction transform round trip
Rutgers Timestamp and NTHU timestep schema handling
source rows and original fields preserved exactly
label-map IDs match PLY labels
label 0 hit differs from geometry miss
invalid ray differs from valid miss
nearest positive hit wins; behind-origin candidates lose
determinism across chunks and repeated runs
```

Run the existing suite as well so Task 2 cannot regress Task 1:

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/<new_task2_scripts>.py
bash -n scripts/<new_task2_slurm_script>.sbatch
```

## 11. Output Isolation And Storage

Task 2 output root:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task2
```

Use immutable method-versioned directories, for example:

```text
outputs/eyenavgs_task2/gaze_raycast_v1/
  manifests/
  logs/
  Rutgers/bicycle/<source_stem>.parquet
  validation/Rutgers/bicycle/<source_stem>_summary.json
  validation/alignment/<alignment_version>/
```

Never write inside:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1
/lab/haoq_lab/cse12312032/data/EyeNavGS
```

As measured on 2026-07-13, `/lab` had 361 GB available and Task 1 outputs used
7.5 GB. This is shared, changing state; recheck before every full run. Keep only
compact alignment samples in validation directories and avoid redundant renders.

Loading the bicycle semantic PLY requires reading about 1.5 GB and 6.1 million
Gaussians. Small parsing/unit tests may run on the login node, but full spatial
index construction, mass annotation, or rendering should use a scheduled job
with measured memory requirements. Codex must provide the exact `sbatch`
command to Dhana and stop; Codex never submits it.

## 12. Merge And Coordination Protocol

1. Keep Task 2 commits on `codex/task2-gaze-targets` until its tests pass.
2. Before requesting integration, fetch `main`, resolve conflicts on the Task 2
   branch, and rerun both Task 1 and Task 2 tests.
3. Do not change accepted Task 1 symlinks from the Task 2 thread.
4. If Task 1 accepts a newer scene run, keep existing Task 2 outputs pinned to
   their old fingerprints and create a new versioned Task 2 run.
5. Do not overwrite README status from stale branch context. Add a concise Task
   2 status only after an end-to-end pilot is measured.
6. Commit code, tests, configs, and documentation together. Push the Task 2
   branch; merge to `main` only after review.

## 13. First Actions For The New Thread

The new thread should do these in order:

1. Read this entire file, `TASK_BRIEF_EyeNavGS_Semantic_Annotation.md`,
   `docs/EXTERNAL_REPOS.md`, and the top-level README.
2. Verify local `main`, `origin/main`, and cluster HEAD; do not assume sync.
3. Create/switch to `codex/task2-gaze-targets` in its own worktree.
4. Inspect the accepted bicycle symlink, deliverables, validation JSON, and
   hashes without modifying them.
5. Inspect one Rutgers bicycle CSV and its Rutgers scene-setting row.
6. Implement Milestone A with tests.
7. Implement only the small alignment proof from Milestone B.
8. If a GPU render or full scheduled run becomes necessary, provide Dhana the
   exact `sbatch` command and stop.

## 14. Ready-To-Paste New Thread Prompt

```text
You are working on Task 2 in the canonical PKU 3DGS VR project. Read
TASK2_HANDOFF.md completely before doing anything, then follow its branch,
input-immutability, output-isolation, alignment-gate, testing, and scheduler
rules. Work in a separate codex/task2-gaze-targets branch/worktree starting from
the handoff commit or later. Do not modify Task 1 accepted outputs or accepted
symlinks. Verify local, origin, and cluster state instead of assuming they are
synchronized. Use login:false for local PowerShell. Never submit sbatch; give
me the exact command and stop at scheduler checkpoints. Begin with Milestone A
and the small Rutgers bicycle coordinate-alignment proof, not full-dataset
annotation.
```
