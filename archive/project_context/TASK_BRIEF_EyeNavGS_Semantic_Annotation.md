---
title: "Intern Task Brief — EyeNavGS Semantic Enhancement & Gaze-Target Auto-Annotation"
date: 2026-07-07
tags:
  - 实习生任务
  - EyeNavGS
  - 语义分割
  - 注视目标标注
  - TrackA
---

# Intern Task Brief
## Semantic Enhancement & Gaze-Target Auto-Annotation on the EyeNavGS Dataset

**Owner:** (supervisor)  **Assignee:** (intern)  **Start:** 2026-07-20  **Target completion:** 2026-08-09 (3 weeks)
**Priority:** 🔴 Critical — this is the "lifeline" data path for our CVPR 2027 submission.

---

## 1. One-line summary

Take the **public EyeNavGS dataset** (46 users navigating 12 photorealistic 3DGS scenes in VR, with per-frame head pose and gaze direction) and produce two things it currently lacks: **(1) object-level semantic labels for every 3D Gaussian scene**, and **(2) a per-frame "which object is the user looking at" label** obtained by intersecting each gaze ray with the semantic scene.

When you finish, every frame of EyeNavGS will carry a **gaze-target label** — turning a raw navigation dataset into a supervised gaze-prediction benchmark, **without recruiting a single new participant.**

---

## 2. Background & motivation (read this first)

**EyeNavGS** (Ding et al., ACM Multimedia 2025, arXiv:2506.02380) is the closest prior work to our project. It released:
- 12 real-world 3DGS scenes (8 from the original 3DGS paper + 4 from the ZipNeRF dataset), each gravity-aligned and metric-scaled.
- 46 participants × 6-DoF free exploration on Meta Quest Pro, with built-in eye tracking.
- A per-frame CSV log (head position, head orientation quaternion, FOV angles, **gaze position + gaze direction quaternion**, timestamp).
- A "record-n-replay" SIBR viewer fork + coordinate-alignment / gaze-visualization utilities.

**Quick primer — what a 3DGS scene is:** 3D Gaussian Splatting (3DGS) represents a scene as millions of small 3D "Gaussians" (blobs), each with a position, shape, color (RGB), and opacity. It renders photorealistically and fast, but it is **purely geometric + appearance** — there is no notion of objects. A Gaussian "knows" it is red and semi-transparent, but not that it belongs to a car.

**What EyeNavGS does NOT have** (this is exactly our opportunity):
- ❌ **No scene semantics.** The Gaussians carry only RGB + opacity, no "this is a car / tree / door."
- ❌ **No gaze-target labels.** The data records a gaze *direction vector*, but never says *which object* the user is looking at. You cannot train "gaze target prediction" without this ground truth.

**Your job closes exactly these two gaps.** The result (early August) is the training data our whole method paper stands on. If everything else in the project slips, this deliverable alone can still become a paper (a re-annotated gaze-prediction benchmark), which is why it is the highest-priority, lowest-risk task on the team right now.

---

## 3. Objectives & deliverables

You have **three sub-tasks**, in dependency order.

### Task 1 — Object-level semantic segmentation of the 3DGS scenes
Assign every Gaussian in each scene an **object-level label** (e.g., `car_01`, `tree_03`, `building_02`, `ground`, `sky`). Instance-level where feasible; at minimum a consistent semantic class per Gaussian.

**Deliverable D1:** For each of the 12 scenes, an augmented `.ply` with one extra per-Gaussian attribute column `label` (integer id), plus a `label_map.json` mapping id → human-readable name/class. Aim for all 12; **at least 4 scenes fully done and validated** is the hard minimum.

### Task 2 — Gaze-target auto-annotation
For every frame in the EyeNavGS logs, cast the recorded gaze ray into the labeled scene, find the first-hit Gaussian(s), and emit the label of the hit object.

**Deliverable D2:** An augmented per-frame table (CSV/Parquet) that reproduces the original EyeNavGS fields **plus**: `gaze_target_id`, `gaze_target_name`, `hit_point_xyz`, `hit_distance`, `hit_confidence`, and a `no_hit` flag for frames where the ray misses all geometry.

### Task 3 — Annotation quality validation
Quantify how trustworthy the labels are, and apply light post-processing.

**Deliverable D3:** A short quality report (2–4 pages / notebook) with:
- **Gaze-to-object hit rate** (fraction of valid frames that hit a labeled object).
- **Manual spot-check accuracy** on ≥300 randomly sampled frames (does the labeled target match what the rendered view shows the user looking at?).
- **Angular / geometric error** analysis where measurable.
- Effect of **semantic snapping** (see §5, Task 3) before vs. after.

---

## 4. Scope

**In scope**
- Work entirely on the **public EyeNavGS dataset + software** (no VR hardware, no new data collection needed).
- Semantic labeling, ray–object intersection, quality validation, and clean reproducible scripts.

**Out of scope (do NOT do these — other people / later phases own them)**
- New participant recruitment or VR data capture.
- Building a dynamic scene graph or training prediction models.
- Eye-tracker calibration on our own recording hardware.

If you finish early, the best "stretch" is to improve semantic quality on the hardest scenes and to widen the manual validation set — **not** to start the modeling work.

---

## 5. Recommended technical approach

You do not need to invent anything. Follow this staged pipeline; escalate in cost only where quality demands it.

### Task 1 — Semantics (cheap → refined)

| Stage | Method | Cost | Use for |
|---|---|---|---|
| **A · fast baseline** | **FlashSplat** (training-free, linear-programming label assignment; ~30 s/scene) | none | Get all 12 scenes labeled quickly |
| **B · interactive refine** | **SAGA** (Segment Any 3D Gaussians — click/box prompt + cross-view consistency) | low | Fix key scenes / important objects |
| **C · feature distillation (optional)** | **Feature 3DGS / LangSplat** — distill DINOv2/CLIP features onto Gaussians, then cluster / open-vocabulary query | medium (GPU) | Only if A+B are insufficient |

Practical recipe:
1. Render N ≈ 100–200 viewpoints per scene from the 3DGS model.
2. Run 2D segmentation (SAM 2) + a classifier or CLIP zero-shot on the renders, **OR** use FlashSplat directly on the Gaussians.
3. Back-project / propagate to 3D so **each Gaussian gets one `label`**.
4. Sanity-check by re-rendering the label field as a color overlay from several viewpoints.

**Priority object classes** (from visual-attention priors): high = people, animals, vehicles, doors, signs; medium = buildings, trees, furniture, screens; low = ground, sky, walls, road. Instance-level (`car_01` vs `car_02`) is valuable where objects are countable and fewer than ~50 per scene.

### Task 2 — Gaze-target annotation (the core mechanic)

The mechanic is a **ray–Gaussian intersection ("ray-march")**: the ray origin is the eye position, the direction is the gaze direction, and you return the **first labeled Gaussian hit** along the ray. Concretely, for each frame:

- Ray **origin** = eye position (`PositionX/Y/Z`, which already includes the head + IPD offset).
- Ray **direction** = gaze direction (derive from `GazeQX/Y/Z/W`, or use `GazePos − Position`).
- **Coordinate-system alignment is critical:** OpenXR (recording) and COLMAP/3DGS (scene) differ by roughly a 180° rotation. EyeNavGS ships alignment utilities — use them, and **verify** by overlaying the ray in a render *before* mass-annotating. Getting this wrong points every ray the wrong way, silently.
- **Intersection:** build a KD-tree over Gaussian centers and march along the ray (or use a voxelized occupancy grid); take the **nearest hit within a small angular/spatial tolerance**; derive `hit_confidence` from local hit density.
- Emit the hit Gaussian's `label` as `gaze_target_id` / `gaze_target_name`, plus `hit_point_xyz`, `hit_distance`, and set `no_hit` when the ray reaches no labeled geometry.

Once semantics exist (Task 1), this step is small: the intersection returns not just a 3D *point* but the *label* of the Gaussian it hit — which is precisely the "user is looking at car_01" ground truth.

### Task 3 — Validation & semantic snapping

- **Semantic snapping:** people tend to fixate object *centers*, and raw rays are noisy, so when a ray passes near an object (within a threshold), snap the label to that object. Report metrics **before vs. after** snapping.
- **Manual spot-check:** sample ≥300 frames, render the user's view with the gaze point overlaid, and record whether the auto-label matches human judgment. Report per-class accuracy.
- **Hit-rate & misses:** report the `no_hit` frame fraction and investigate causes (sky/ground gaps, calibration drift, holes in the scene).

---

## 6. Environment, data & resources

- **Dataset:** EyeNavGS — Rutgers repo `github.com/symmru/EyeNavGS_Rutgers_Dataset`, NTHU repo `github.com/sawalee0811/EyeNavGS_NTHU_Dataset`.
- **Software (record-n-replay + utilities):** `github.com/symmru/EyeNavGS_Software`. Project site: `symmru.github.io/EyeNavGS`.
- **Paper:** Ding et al., *EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR*, arXiv:2506.02380 (ACM MM 2025). Licenses: CC BY 4.0 (paper) / Apache 2.0 (code).
- **Semantics toolchain (all public):** SAM 2 (Meta), DINOv2 (Meta), CLIP (OpenAI), FlashSplat, SAGA (Segment Any 3D Gaussians), Feature 3DGS, LangSplat, Gaussian Grouping — reuse and cite their public code.
- **Compute:** a single modern GPU (RTX 4090 / A100 class) is enough. FlashSplat needs little; feature distillation needs the most.
- **EyeNavGS per-frame CSV schema** (one row per frame; left/right eye rows alternate):

  | Field | Meaning |
  |---|---|
  | `ViewIndex` | 0 = left eye, 1 = right eye |
  | `FOV1–FOV4` | field-of-view (left / right / up / down, radians) |
  | `PositionX/Y/Z` | eye 3D position (world coords; head + IPD offset) |
  | `QuaternionX/Y/Z/W` | head orientation (world coords) |
  | `GazePosX/Y/Z` | gaze position (world coords) |
  | `GazeQX/Y/Z/W` | gaze orientation (world coords) |
  | `Timestamp` | millisecond offset |

---

## 7. Timeline & milestones (3 weeks)

| Week | Focus | Exit criterion |
|---|---|---|
| **W1 (7/20–7/26)** | Load & parse EyeNavGS; verify coordinate alignment by overlaying a gaze ray on renders; FlashSplat baseline on **1 scene**; draft the ray-intersection annotator | 1 scene labeled + 1 scene auto-annotated end-to-end (proof of pipeline) |
| **W2 (7/27–8/2)** | Scale semantics to all 12 scenes (SAGA-refine the important ones); run the annotator across the full dataset | All scenes labeled (≥4 validated); full per-frame gaze-target table produced |
| **W3 (8/3–8/9)** | Quality validation: hit-rate, ≥300-frame manual spot-check, angular error, semantic-snapping ablation; write report; clean & document code | **D1 + D2 + D3 delivered** |

> **Key milestone (early August):** every EyeNavGS frame has a gaze-target label → training data is ready → the method paper has its foundation.

---

## 8. Definition of done (acceptance criteria)

1. **D1** — ≥ 4 scenes fully labeled and visually validated (label-overlay renders look correct); ideally all 12. Each ships as an augmented `.ply` + `label_map.json`.
2. **D2** — A reproducible script turns any EyeNavGS log + labeled scene into the augmented per-frame table with `gaze_target_*`, `hit_*`, and `no_hit`. Runs end-to-end from a single command.
3. **D3** — Quality report with hit-rate, ≥300-frame manual accuracy (per-class), angular error, and before/after semantic-snapping numbers.
4. **Reproducibility** — Code in a clean repo with a README, pinned dependencies, and one-command reproduction of D1→D2→D3 on at least one scene.

---

## 9. Ways of working

- **Weekly check-in** (30 min) + a short written status each Friday: what shipped, blockers, plan for next week.
- **Ask early** on two things especially: (a) coordinate-system alignment sanity, and (b) whether instance-level vs. class-level labeling is worth the extra effort per scene. Getting these wrong silently is the main risk.
- Keep everything **scripted and reproducible** — no manual one-off steps that can't be rerun.
- Commit small and often; document assumptions in the repo README as you go.

---

## 10. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Coordinate misalignment → rays point the wrong way | Use EyeNavGS alignment utils; validate by overlaying the ray in a render **before** mass-annotating |
| Poor semantics → noisy gaze labels | FlashSplat baseline everywhere, SAGA-refine key scenes, manual spot-check, semantic snapping |
| Small / thin objects mislabeled | Instance refine with SAGA; report per-class accuracy so weak classes are visible |
| High `no_hit` rate | Analyze causes (sky/ground, scene holes, drift); document rather than hide |
