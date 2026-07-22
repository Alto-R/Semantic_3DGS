---
title: EyeNavGS Internship Schedule 
date: 2026-07-08
---

# Revised Internship Schedule

This schedule prioritizes completing the **semantic annotation on the open-source EyeNavGS dataset first**.

## Phase 0 — Environment & Preparation (Jul 8 – Jul 10)

### Objectives
- Read the EyeNavGS paper and documentation.
- Download and explore the dataset.
- Install and verify the EyeNavGS software.
- Understand the `.ply` Gaussian representation.
- Learn the basics of FlashSplat and SAGA.

### Deliverables
- Working environment.
- Dataset loaded successfully.
- Able to visualize at least one scene.

---

## Phase 1 — Semantic Annotation & Refinement (Jul 11 – Jul 21)

### Objectives
- Generate baseline semantic labels using FlashSplat.
- Refine labels using SAGA where needed.
- Validate labels with semantic overlay renders.
- Standardize object names and IDs.
- Complete semantic labels for at least four scenes (goal: all twelve).

### Deliverables
- Augmented `.ply` files with semantic labels.
- `label_map.json` for each scene.
- Stable semantic EyeNavGS dataset.

> **This phase is the highest priority before any gaze-target annotation begins.**

---

## Phase 2 — Gaze Target Auto-Annotation (Jul 22 – Jul 30)

### Objectives
- Verify coordinate alignment.
- Generate gaze rays.
- Build KD-tree / spatial acceleration.
- Perform ray–Gaussian intersection.
- Produce:
  - `gaze_target_id`
  - `gaze_target_name`
  - `hit_point_xyz`
  - `hit_distance`
  - `hit_confidence`
  - `no_hit`

### Deliverables
- End-to-end gaze annotation pipeline.
- Augmented per-frame dataset.

---

## Phase 3 — Validation & Finalization (Jul 31 – Aug 9)

### Objectives
- Manual validation (≥300 frames).
- Compute hit rate and analyze `no_hit` cases.
- Evaluate semantic snapping.
- Write README and final report.
- Clean repository and ensure reproducibility.

### Final Deliverables
- **D1:** Semantic EyeNavGS dataset.
- **D2:** Gaze-target annotated dataset.
- **D3:** Validation report.
- Reproducible code repository.

---

# Timeline Summary

| Dates | Focus | Expected Output |
|-------|-------|-----------------|
| Jul 8–10 | Environment setup | Working EyeNavGS environment |
| Jul 11–21 | Semantic annotation & refinement | ≥4 validated scenes (goal: all 12 baseline-labelled) |
| Jul 22–30 | Gaze-target annotation | Complete annotation pipeline and dataset |
| Jul 31–Aug 9 | Validation & finalization | D1 + D2 + D3 + documentation |
