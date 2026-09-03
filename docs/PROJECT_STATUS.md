# Project status

This file records the maintained semantic pipeline, the result used to verify
it, and the limits of that verification.

## Maintained method

The production route starts from a Graphdeco reconstruction and runs these
stages in order:

1. Render every reconstruction camera through FlashSplat.
2. Run the pinned DINOv3 ViT-7B ADE20K Mask2Former model on every view.
3. Lift the per-pixel predictions onto the Gaussians.
4. Build an equal-camera strict-majority base with explicit abstention.
5. Recover supported ties and weak majorities without filling unseen
   Gaussians.
6. Run competitive dino.txt classification and automatic SAM masks for a
   configured out-of-vocabulary target.
7. Fuse the OOV evidence against the immutable DINOv3 base.
8. Write semantic and SuperSplat PLYs, then render class-colored PNGs for every
   camera.

The [complete quickstart](DINOV3_DINOTXT_SAM_QUICKSTART.md) runs the whole
sequence. The [detailed guide](DINOV3_DINOTXT_SAM_END_TO_END_PIPELINE.md)
defines the intermediate contracts and fusion rules.

## Verified reference run

The complete quickstart was verified on the Dr. Johnson Graphdeco scene with
`window_shutter` as the dino.txt target.

- All 263 reconstruction cameras completed every stage.
- The base and final semantic and SuperSplat PLYs each contained 3,405,153
  Gaussians.
- The OOV stage changed 151,863 Gaussians to `window_shutter`.
- Of those changes, 89,738 started as unlabeled and 62,125 replaced a base
  label. The largest replaced classes were `door`, `wall`, and `windowpane`.
- Visual review of cameras 0103, 0137, and 0239 found strong shutter coverage
  without the earlier door and window streaks.
- Difficult views still contain incomplete coverage and unlabeled holes. The
  result is the best verified run, not perfect ground truth.

This verifies the implementation and its output contract for one scene and
one OOV target. It does not establish accuracy on every scene or class.

## Generalization boundary

The pipeline does not contain scene coordinates, per-Gaussian allowlists, or
hand-selected camera lists. A new OOV class does require a declarative target
configuration with target prompts, plausible competitors, and global mask
selection thresholds.

The default 3D OOV gates require:

- visibility in at least three cameras;
- an OOV win in at least two cameras;
- a winner share of at least 0.50;
- an OOV mass share of at least 0.35;
- positive mass in at least two cameras;
- a 0.05 advantage over the incumbent label.

These defaults are global. Lower-view experiments remain useful for diagnosis,
but they are not the maintained default.

## Legacy alternatives

The repository still contains the earlier DINOv2 multiview base, adaptive ADE
v5 refinement, GroundingDINO source generation, and reviewed-extension merge.
They remain available for reproducing older results. They are not stages of the
maintained DINOv3 and dino.txt pipeline.

Retired DINOv3 3D-first, hybrid-refinement, and post-v5 singleton experiments
are under `archive/`. Archived files are historical and must not be presented
as maintained entry points.

## Output policy

Every run uses a new output name. No stage overwrites an accepted base or an
existing result. A successful process exit verifies execution, file structure,
and internal consistency. Semantic acceptance still requires inspection of the
SuperSplat PLY or render-back PNGs.
