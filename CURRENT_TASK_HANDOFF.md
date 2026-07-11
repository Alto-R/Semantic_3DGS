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

The local/GitHub push of `7b866fc` succeeded. A batched cluster synchronization
was started, but the batch later timed out during the coverage measurement and
did not return intermediate outputs. Therefore the fresh task must verify the
cluster HEAD instead of assuming it reached `7b866fc`.

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

### Job 92369 - Accepted Bicycle Identity Baseline

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

- Non-destructive accepted pointer, created and verified:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle
-> ../bicycle_semantic_identity_test_v2
```

The accepted source job 92344 output was verified unchanged after reuse runs.

## 9. Current Coverage Question

The accepted bicycle result has clean rendered overlays but 60.25% of raw
Gaussians are label 0. Raw Gaussian count may overstate practical gaze-target
coverage because many Gaussians can be hidden, redundant, low-opacity, or not
dominant in the rendered views.

Commit `7b866fc` adds:

```text
scripts/measure_overlay_coverage.py
tests/test_overlay_coverage.py
```

The tool compares each original RGB render with its semantic overlay and
reports the fraction of pixels changed by a visible semantic overlay across all
50 cameras. This is explicitly an image-space proxy, not semantic ground truth,
and may undercount palette colors close to the original RGB.

The previous Codex task attempted to sync `7b866fc` and run this command in one
large batched tool operation. The operation timed out after about 34 seconds
during the coverage step and returned no intermediate outputs. It may have
partially completed. Do not guess.

## 10. First Actions In The Fresh Task

Use `login: false` for local PowerShell commands.

### Step 1: Verify local and cluster Git state

```powershell
git status --short
git log -3 --oneline
```

```bash
ssh -p 10022 cse12312032@172.18.34.25 \
  'git -C /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr rev-parse --short HEAD'
```

Expected local/GitHub functional commit is `7b866fc`. Verify cluster HEAD.

### Step 2: Check whether the interrupted coverage report exists

```bash
ssh -p 10022 cse12312032@172.18.34.25 \
  'test -f /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle/validation/visible_overlay_coverage.json && cat /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle/validation/visible_overlay_coverage.json'
```

If it exists and is valid, summarize its top-level metrics. Do not print all 50
frame records unless needed.

### Step 3: If missing, ensure cluster code is synced, then run CPU coverage

This command is CPU-only and does not use `sbatch`. Allow a timeout of at least
120 seconds because the earlier 30-second timeout was too short for reading 100
full-size PNGs through the Conda environment.

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
conda run -n gaussian_grouping_true \
  python scripts/measure_overlay_coverage.py \
  --rgb-dir /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle/stages/01_grounded_sam/rgb_renders \
  --overlay-dir /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle/visualizations/overlays/semantic_labels \
  --output /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/accepted/bicycle/validation/visible_overlay_coverage.json \
  --difference-threshold 2 \
  --exclude-border 1
```

After measuring coverage, decide based on evidence whether the pipeline needs a
new label-propagation stage. Do not infer this from raw unlabeled Gaussian count
alone.

## 11. Next Engineering Milestones

1. Finish and record image-space visible coverage for accepted bicycle.
2. Decide whether conservative label 0 coverage is acceptable for gaze-target
   experiments or whether automatic propagation/refinement is required.
3. Generalize the bicycle-specific scheduled script to configurable `SCENE`,
   `MODEL_DIR`, output name, class config, and optional focus classes.
4. Prepare scene-specific automatic vocabularies for the next matched scenes.
5. Run one next-scene pilot through the user-controlled `sbatch` checkpoint.
6. Reach at least four fully labeled and visually validated scenes.
7. Locate or train models for `nyc`, `london`, `berlin`, and `alameda` before
   claiming all 12 scenes.

Do not call Task 1 complete merely because bicycle passes. The hard minimum is
four validated scenes, and raw/visible unlabeled policy still needs an explicit
decision.

## 12. Important Files

```text
README.md
docs/WORKFLOW.md
docs/TASK1_SEMANTIC_ANNOTATION.md
docs/EXTERNAL_REPOS.md
configs/task1_semantic_classes.example.json
scripts/generate_grounded_sam_masks.py
scripts/run_flashsplat_mask_proposals.py
scripts/cluster_semantic_flashsplat_proposals.py
scripts/render_auto_label_overlays.py
scripts/measure_overlay_coverage.py
scripts/validate_task1_outputs.py
scripts/summarize_task1_semantic_run.py
scripts/slurm_task1_bicycle_grounded_sam.sbatch
tests/test_semantic_fusion.py
tests/test_overlay_coverage.py
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
