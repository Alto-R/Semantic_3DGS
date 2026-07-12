# Current Task Handoff

Last updated: 2026-07-11 (Asia/Shanghai)

This file is the authoritative compact handoff for continuing the current work
in a fresh Codex task launched directly in the canonical Git repository. Treat
the current workspace/repository root as authoritative, read this file before
running commands, and inspect Git and cluster state rather than assuming every
interrupted operation finished.

## 1. User And Operating Rules

- User name for Git and documentation: `Dhana Kresnawijaya`.
- Keep communication direct and evidence-based. Separate measured facts from
  visual interpretation.
- The semantic pipeline must be automatic. Human inspection is validation, not
  manual naming, mask editing, label correction, or pruning.
- Never put passwords or private-key passphrases in commands, scripts, files,
  logs, or this repository.
- Critical scheduler rule: Codex must **never submit `sbatch` itself**. When a
  scheduled GPU run is needed, provide Dhana the exact command and stop at that
  checkpoint. Read-only monitoring with `squeue`, `sacct`, and log inspection is
  allowed.
- Dhana has authorized moving/copying repository files between the old
  Documents workspace and the canonical `PROJECTS` folder without asking again.
- Avoid long unexplained gaps. Use `login: false` for PowerShell shell calls to
  skip the broken PowerShell profile. Minimize tool round trips and redundant
  status checks. Do not hide a real timeout behind an "orchestration" explanation.
- This handoff exists because the previous Codex task became very long. The
  large conversation context appeared to add substantial model-side latency
  between otherwise fast commands.

## 2. Canonical Locations

### Windows

Canonical Git repository:

```text
C:\Users\dhana\PROJECTS\PKU 3DGS VR
```

The fresh Codex task is expected to start with this directory as its workspace
and current working directory. Use repository-relative paths and edit files
directly in this checkout.

Historical staging workspace used by the previous Codex task:

```text
C:\Users\dhana\Documents\PKU 3DGS VR
```

The Documents directory is not the canonical Git checkout. Do not use it in the
fresh task when the canonical repository is writable. It is listed only to
explain old temporary files and paths in the prior run history. All new edits,
tests, Git commands, and commits should operate directly in the current
canonical repository.

PowerShell currently tries to load a profile that is blocked by execution
policy. For local `shell_command` calls, set:

```text
login: false
```

This removes the recurring `profile.ps1 cannot be loaded` error and reduces
shell startup noise.

### Cluster

Login:

```bash
ssh -p 10022 cse12312032@172.18.34.25
```

SSH alias when the local SSH config is available:

```text
haoqi
```

Cluster account root:

```text
/lab/haoq_lab/cse12312032
```

Cluster project checkout:

```text
/lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
```

Main cluster layout:

```text
/lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
/lab/haoq_lab/cse12312032/data/EyeNavGS
/lab/haoq_lab/cse12312032/data/3dgs_models
/lab/haoq_lab/cse12312032/external
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1
```

Do not recreate the deprecated cluster bare Git remote under a `git/` folder.
GitHub is the source of truth.

## 3. Git And SSH State

GitHub repository:

```text
https://github.com/DhanaKresnawijaya237/pku-3dgs-vr
```

GitHub account:

```text
DhanaKresnawijaya237
```

The local repository can push to GitHub. The cluster also has a GitHub SSH key,
but it is passphrase-protected and an `ssh-agent` started in Dhana's interactive
terminal is not automatically inherited by a new Codex SSH session.

When cluster `git pull` cannot run non-interactively, the working fallback is a
temporary Git bundle, not a cluster bare remote:

1. Create a bundle from the canonical Windows repository into the writable
   Documents `tmp/` directory.
2. Copy it with `scp -P 10022` to `/tmp/` on the cluster.
3. Verify the cluster checkout is clean.
4. `git fetch /tmp/<bundle> main` in the cluster checkout.
5. `git merge --ff-only FETCH_HEAD`.

Functional code commit before this handoff file:

```text
7b866fc Measure projected semantic coverage
```

Its predecessor is:

```text
95ef1d6 Preserve accepted instances during consolidation
```

The cluster checkout was verified at `95ef1d6`, then fast-forwarded through
`42edce6` before the CPU-only coverage measurement. The coverage-recording
documentation commit was also synchronized after it was pushed.

## 4. Research Goal And Task 1 Contract

The broader project uses static 3D Gaussian Splatting scenes in UE5/PICO VR to
study gaze and predict gaze targets. Task 1 creates semantic labels on the 3DGS
models.

Task 1 requirement:

- Assign each Gaussian an integer `label`.
- Prefer persistent object instances such as `bicycle_01`, `tree_03`, and
  `building_02` where feasible.
- At minimum provide a consistent semantic class.
- Preserve every original Gaussian property.
- Add exactly the required integer `label` property to the deliverable PLY.
- Write a separate `label_map.json` mapping IDs to names/classes.
- Aim for all 12 EyeNavGS scenes; at least 4 fully completed and validated scenes
  is the hard minimum.

The accepted deliverable is separate from debug PLY files. A normal 3DGS viewer
does not automatically visualize the integer `label` property.

## 5. Available Scene Models

EyeNavGS has 12 target scene names. Eight currently match downloaded official
GraphDeco models:

```text
bicycle
drjohnson
playroom
room
stump
train
treehill
truck
```

Four target scenes still need another 3DGS model source:

```text
nyc
london
berlin
alameda
```

Other GraphDeco models exist on the cluster but are not part of the current
8/12 EyeNavGS match. See `docs/TASK1_SEMANTIC_ANNOTATION.md` and the generated
scene manifest for details.

Pilot model:

```text
/lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco/bicycle
```

Pilot PLY has:

```text
6,131,954 Gaussians
```

## 6. Active Automatic Pipeline

Current pipeline:

```text
50 evenly spaced 3DGS camera views at width 960
-> GroundingDINO open-vocabulary detections
-> SAM masks from GroundingDINO boxes
-> black/white 8-bit mask PNG export
-> FlashSplat mask-to-Gaussian lifting
-> positive and negative class visibility evidence
-> confidence-weighted multi-view 3D fusion
-> automatic tiny-label pruning
-> disconnected 3D island pruning for thing labels
-> merge-only same-class instance consolidation
-> semantic PLY and label_map.json
-> semantic overlays, focused overlays, contact sheets, debug PLYs
-> structural and visual validation
```

Important behavior:

- GroundingDINO supplies semantic class names automatically.
- SAM supplies masks; it does not name objects.
- FlashSplat lifts masks into Gaussian support.
- Signed class evidence is essential. Positive-only evidence caused background
  leakage because more views could add false support without penalizing views
  where the same Gaussian was visible outside a class mask.
- Thing labels require at least 2 positive views and a positive ratio of at
  least `0.50` in the bicycle configuration.
- Instance consolidation is merge-only. It may merge accepted same-class IDs
  that share a sufficiently large connected 3D component, but it must never
  split an already accepted instance or drop its Gaussians.
- Object IDs are stored on the static 3D Gaussians. Therefore the same accepted
  object keeps the same ID when viewed again from another camera.

Primary environment and dependencies:

```text
Conda env: gaussian_grouping_true
FlashSplat: /lab/haoq_lab/cse12312032/external/FlashSplat
Grounded-SAM root:
/lab/haoq_lab/cse12312032/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything
```

Working scheduled configuration uses the RTX 8000 explicitly:

```text
account=gpulab02
partition=titan
qos=titan
nodelist=rtx8000
gres=gpu:1
cpus-per-task=4
mem=96G
```

Do not silently change this known-working GPU configuration. Dhana previously
preferred waiting for the RTX 8000 rather than changing to an unverified setup.

## 7. Output Layout And Visualization

Current clean scene layout:

```text
<scene_output>/
  stages/01_grounded_sam/
  stages/02_flashsplat/
  stages/03_semantic_fusion/
  deliverables/
    semantic_point_cloud.ply
    label_map.json
  visualizations/
    ply/
    overlays/semantic_labels/
    overlays/bicycle_vs_bench/
    contact_sheets/
  validation/
  logs/
```

Debug visualization facts:

- `semantic_point_cloud.ply` preserves original 3DGS appearance and adds
  integer `label`; SuperSplat will normally still show original colors.
- `semantic_point_cloud_supersplat_debug.ply` bakes semantic colors into SH DC
  coefficients and clears higher-order SH so SuperSplat shows labels.
- `bicycle_vs_bench_supersplat_debug.ply` dims all non-focus classes.
- Shared palette uses bicycle red and bench blue.
- The focused contact sheet is the main leakage check.
- Contact-sheet inspection is not ground truth. It is visual QA only.

## 8. Bicycle Run History

### Job 92261

- First useful 50-view automatic baseline.
- 411 GroundingDINO + SAM masks and 399 FlashSplat proposals.
- Semantic classes plausible, but foreground object instances fragmented.

### Job 92331

- Added spatial 3D island pruning.
- Removed many tiny disconnected false labels.
- Residual red road streaks and blue vegetation patches remained because larger
  false components survived.
- Diagnosis: missing camera angles were not the main problem; positive-only
  evidence could accumulate false labels.

### Job 92344

- Commit: `385f8ba`.
- Added signed positive/negative multi-view class evidence for 14 classes.
- Runtime: 5m56s on RTX 8000.
- 50 views, 411 masks, 399 proposals.
- Structural validation passed.
- Removed the obvious red road and blue vegetation leakage while preserving the
  bicycle and bench across the 50-view sheet.
- High-confidence source output:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_semantic
```

- It still had 2 bicycle IDs and 4 bench IDs.
- Unlabeled count: 3,694,666 (`0.6025266986673417`).

### Job 92353 - Rejected Diagnostic

- First connected-component identity consolidation.
- Correctly merged bicycle `2 -> 1` and bench `4 -> 1`.
- Incorrectly split 13 accepted tree groups into 26 components.
- Subsequent size pruning discarded 95,584 tree Gaussians.
- Retained as a diagnostic only, not accepted.

### Job 92369 - Preserved 50-View Bicycle Baseline

- Commit: `95ef1d6`.
- Runtime: 2m06s, reusing job 92344 masks/proposals.
- Structural validation: `ok`.
- Exact accepted metrics:

```text
total Gaussians:       6,131,954
unlabeled:             3,694,666
unlabeled ratio:       0.6025266986673417
nonzero labels:        16
bicycle_01:              123,197
bench_01:                120,977
tree total:              932,516
tree IDs:                8
```

- Tree coverage exactly matches job 92344; consolidation removed no accepted
  Gaussian coverage.
- Focused and full semantic contact sheets passed visual inspection across all
  50 views.
- No obvious return of the red road streaks or blue vegetation patches.
- Accepted output target:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_semantic_identity_test_v2
```

- Its original output remains preserved:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle_semantic_identity_test_v2
```

The accepted source job 92344 output was verified unchanged after reuse runs.

### Job 92426 - Current Accepted Targeted-Camera Baseline

- Code commits: `c16ff73`, `ef2eb52`.
- Runtime: 9m05s on the configured RTX 8000.
- Automatic selection: 144 safe unused cameras, 100 rendered candidates, 20
  additions, 70 final cameras.
- 669 raw detections, 599 kept GroundingDINO + SAM masks, 583 FlashSplat
  proposals; 566 proposals survived fusion filtering.
- Structural validation: `ok`.
- Exact accepted metrics:

```text
total Gaussians:       6,131,954
unlabeled:             3,580,841
unlabeled ratio:       0.5839641001873138
nonzero labels:        15
bicycle_01:              119,570
bench_01:                139,727
tree total:            1,096,092
tree IDs:                      8
```

- Same original 50 cameras: pooled visible coverage `0.9242405890153753`,
  minimum `0.7281843725791416`, median `0.9428833326768293`.
- Added 20 cameras: pooled visible coverage `0.8244914556006355`, minimum
  `0.7465320176993474`, median `0.8209549506965508`.
- Per-Gaussian comparison versus job 92369: 253,800 gained labels from label 0,
  139,975 lost to label 0, and 37,579 changed semantic class.
- The largest class change was 10,229 fence-to-bench Gaussians. This matches the
  earlier `bench fence` detector ambiguity and the clean focused overlays.
- Focused and full 70-view contact sheets show no obvious red road streaks,
  blue vegetation leakage, or implausible tree expansion.
- Current non-destructive accepted pointer:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle
-> ../bicycle_semantic_targeted_v1
```

## 9. Measured Visible Coverage

The accepted bicycle result has clean rendered overlays but 58.40% of raw
Gaussians are label 0. The nonzero-label Gaussian ratio is
`0.4160358998126862`. Raw Gaussian count overstates the apparent practical gap
because many Gaussians can be hidden, redundant, low-opacity, or not dominant
in the rendered views.

Commit `7b866fc` adds:

```text
scripts/measure_overlay_coverage.py
tests/test_overlay_coverage.py
```

The generated report is:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle/validation/visible_overlay_coverage.json
```

Top-level measured metrics for the current 70-view accepted output:

```text
frame count:                    70
evaluated pixels:              42,650,160
overlay-changed pixels:        38,203,490
overlay-changed ratio:         0.895740836611164
frame ratio min:               0.7281843725791416
frame ratio p10:               0.7737812003518861
frame ratio median:            0.8987268746471291
frame ratio p90:               0.9985289386956578
frame ratio max:               0.9995568598101391
difference threshold:          2
excluded border pixels:        1
```

This is measured image-space evidence that visible coverage is much higher than
the raw nonzero-label Gaussian ratio. It does not establish semantic correctness
or gaze-target hit rate, and palette similarity or opacity blending can
undercount changed pixels. Any render/compositing difference unrelated to the
semantic tint could instead overcount them. The current evidence does not
justify a new propagation stage solely to reduce the raw label-0 count. Inspect
the lowest-coverage views and downstream gaze hits before changing the policy.

## 10. Completed Coverage Checkpoint

- Local checkout was clean at `42edce6`; its recent history included
  `42edce6`, `13a61eb`, and `7b866fc`.
- Cluster checkout was initially `95ef1d6`; the interrupted synchronization had
  not completed.
- No report existed before the retry.
- The cluster checkout was clean and fast-forwarded to `42edce6` before the
  CPU-only measurement.
- The report above was created successfully without `sbatch`.

## 11. Next Engineering Milestones

1. Run the prepared reuse correction for train targeted job `92470`; Codex must
   not submit it.
2. Compare the corrected 70-view result against accepted job `92469` and
   diagnostic job `92470` before changing the accepted pointer.
3. Define a downstream gaze-hit acceptance criterion for accepted scenes before
   introducing any automatic propagation/refinement stage.
4. Prepare vocabularies and runs for at least two more matched scenes to reach
   the four-scene minimum.
5. Locate or train models for `nyc`, `london`, `berlin`, and `alameda` before
   claiming all 12 scenes.

Do not call Task 1 complete merely because bicycle passes. The hard minimum is
four validated scenes, and the visible-coverage proxy is not a substitute for
semantic or gaze-hit validation.

### Completed automatic targeted-view checkpoint

Job `92426` was submitted by Dhana with:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
sbatch --export=ALL,OUTPUT_NAME=bicycle_semantic_targeted_v1,AUTO_TARGET_VIEW_COUNT=20,TARGET_CANDIDATE_COUNT=100,TARGET_SELECTION_SOURCE=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle scripts/slurm_task1_bicycle_grounded_sam.sbatch
```

The comparison passed and the accepted pointer now targets
`bicycle_semantic_targeted_v1`. No repeat submission is currently required.

### Accepted train baseline and targeted scheduler checkpoint

Jobs `92467` and `92469` completed successfully. Job `92467` produced a valid
initial result, but its 10,000-Gaussian stuff cutoff pruned a sky group supported
by 43 of 50 source views. Reuse job `92469` lowered that cutoff to 8,000 and is
accepted as `train_semantic_baseline_v2`.

Measured accepted metrics:

```text
Gaussians:                     1,026,508
final labels including 0:            10
unlabeled ratio:               0.4414675774567758
visible frame count:                         50
evaluated pixels:              25,482,800
overlay-changed pixels:        25,308,747
overlay-changed ratio:         0.9931697851099566
frame ratio min:               0.9473586105137582
frame ratio p10:               0.9799849310122907
frame ratio median:            0.998994419765489
frame ratio p90:               0.9999923477796789
frame ratio max:               1.0
```

The two train instances are unchanged from job `92467` at 428,735 and 18,476
Gaussians. The 8,462-Gaussian sky label replaces the pruned group, while no
rejected thing group crossed its unchanged 5,000-Gaussian cutoff. Full and
train-versus-track contact sheets show coherent train coverage and corrected
sky ownership. The overlay-difference ratio is an image-space visibility proxy,
not semantic ground truth or gaze-hit accuracy.

The accepted cluster pointer is:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/train
  -> ../train_semantic_baseline_v2
```

Targeted expansion job `92470` completed successfully with 70 views, but it is
diagnostic rather than accepted. The selected 20 cameras are useful: their
pooled overlay-difference proxy is `0.9681440030137975` with a minimum of
`0.9091445994945611`. However, the run used the default 10,000-Gaussian stuff
cutoff, pruned sky at 8,941 Gaussians, and returned the earlier background
leakage. It also promoted a 6,831-Gaussian `building` group that visual QA shows
is the long shipping container.

The fusion code now reads optional per-class `min_assigned_gaussians` overrides
from the scene config. Train sets both `sky` and `building` to 8,000: sky can be
retained below the generic stuff cutoff, while the false building remains below
its stricter thing threshold. The next user-controlled scheduler checkpoint
reuses all masks and proposals from job `92470`:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
SCENE=train \
OUTPUT_NAME=train_semantic_targeted_v2 \
REUSE_SOURCE_OUT=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/train_semantic_targeted_v1 \
FOCUS_CLASSES='train,railroad_track' \
FOCUS_NAME=train_vs_track \
sbatch --export=ALL,SCENE,OUTPUT_NAME,REUSE_SOURCE_OUT,FOCUS_CLASSES,FOCUS_NAME scripts/slurm_task1_semantic_scene.sbatch
```

Do not set `AUTO_TARGET_VIEW_COUNT` for this correction; camera selection and
detection are already complete.

## 12. Important Files

```text
README.md
docs/WORKFLOW.md
docs/TASK1_SEMANTIC_ANNOTATION.md
docs/EXTERNAL_REPOS.md
configs/task1_semantic_classes.example.json
configs/task1_semantic_classes.train.json
scripts/generate_grounded_sam_masks.py
scripts/run_flashsplat_mask_proposals.py
scripts/cluster_semantic_flashsplat_proposals.py
scripts/render_auto_label_overlays.py
scripts/measure_overlay_coverage.py
scripts/select_targeted_cameras.py
scripts/validate_task1_outputs.py
scripts/summarize_task1_semantic_run.py
scripts/slurm_task1_bicycle_grounded_sam.sbatch
scripts/slurm_task1_semantic_scene.sbatch
tests/test_semantic_fusion.py
tests/test_overlay_coverage.py
tests/test_scene_configuration.py
```

External repository commit pins are recorded in `docs/EXTERNAL_REPOS.md`.

## 13. Known Failure Modes And Lessons

- An SSH key existing locally does not mean the public key is installed on the
  cluster. In this setup key login already works; do not repeat setup unless it
  actually fails.
- A passphrase prompt is for decrypting a private key, not the cluster account
  password.
- `ssh-agent` state is session-specific.
- Avoid PowerShell variable interpolation inside remote SSH commands. Earlier
  `$ROOT` variables expanded locally and broke remote Git commands. Prefer
  literal cluster paths or carefully quoted remote commands.
- PowerShell quoting also broke remote `stat -c` format strings containing `%`.
  Use simpler commands such as `ls -l --full-time` when exact formatting is not
  necessary.
- Do not infer pipeline quality from Slurm exit code alone. Always inspect
  validation JSON, label map, class counts, and rendered contact sheets.
- More camera views do not automatically fix false labels. With positive-only
  fusion, more views can add contamination. Signed negative evidence fixed the
  main leakage problem.
- Spatial island pruning alone cannot remove large coherent false components.
- Connected-component instance logic must not split accepted source labels.
- The final semantic PLY and the SuperSplat debug-color PLY serve different
  purposes; do not confuse visualization colors with stored semantic labels.
- Avoid giant multi-command orchestration with a single short timeout. The last
  batch timed out at 34 seconds. Group only fast dependent operations and give
  real image-processing commands an appropriate timeout.

## 14. Fresh Task Prompt

Start the new Codex task with:

```text
You are already running in the canonical PKU 3DGS VR Git repository. Read
CURRENT_TASK_HANDOFF.md completely before acting and use repository-relative
paths. Verify local and cluster state first. Never submit sbatch yourself; give
me the exact command and stop at scheduler checkpoints. Use login:false for
local PowerShell and minimize tool round trips. Begin with the interrupted
visible coverage check in Section 10.
```
