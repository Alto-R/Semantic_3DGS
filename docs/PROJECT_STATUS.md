# Project Status

This file records the current maintained method, scene coverage, accepted
interpretations, and unresolved limitations. It is intentionally separate from
the pipeline guide so historical experiments do not obscure the active workflow.

## Maintained method

- Base: DINOv2 ViT-L/14 ADE20K, all real cameras, exact multiview
  `separate_abstain` fusion.
- Automatic refinement: adaptive instance-guard v5.
- Missing ontology classes: separate GroundingDINO+SAM source jobs followed by
  explicit visual review and a versioned reviewed merge.
- Superseded singleton methods: v6-v10 are rejected and archived.

The v5 base and every published predecessor remain immutable. A review result is
not equivalent to a final merge unless a new output was deliberately published.

## Scene cohorts

The original semantic/refinement cohort is:

```text
bicycle, train, room, truck, drjohnson, playroom, stump, treehill
```

The later DINOv2-base cohort is:

```text
bonsai, counter, flowers, garden, kitchen
```

These lists describe the scenes currently evaluated by this repository; they do
not claim that every available EyeNavGS scene has been annotated.

## Known v5 limitations

The independent multiview audit found no global, class-neutral method that both
recovers every missing instance and preserves correct existing semantics.
Specifically:

- some intended Playroom door instances remain missing;
- additional DrJohnson window instances remain missing;
- the v6-v10 singleton/evidence variants introduced unacceptable false merges or
  lacked sufficient evidence and were therefore rejected.

These omissions require manual completion if exhaustive ground truth is needed.
They are not silently encoded as scene-specific exceptions.

## Custom-extension review status (In Progress)

| Scene | Class or proposal | Current decision |
|---|---|---|
| Room | `piano`, `speaker` | Reviewed source candidates; keep separate from immutable base until an explicit merge is accepted |
| Room | `media_console` | Rejected |
| Train | `railroad_track` | Treat prior track and platform source groups as one railroad-track identity; union merge has preflight support, not a separate platform class |
| Playroom | `stroller` | Candidate only; known wall/floor replacement risk requires strict review |
| Stump | `tree_stump`, `log` | Retired; treat the ambiguous object as tree rather than adding noisier custom labels |
| Bonsai | `bonsai_tree` | v1 source evidence reviewed as coherent |
| Bonsai | `electronic_keyboard` | v1 rejected because the stand/base was absorbed; tighter v2 source config is prepared but not yet accepted |
| Counter | `mixing_bowl` | v1 source evidence reviewed as coherent |
| Counter | `oven_mitt` | v1 rejected because one detected object was not an oven mitt; tighter v2 config is prepared |
| Counter | `water_filter_pitcher` | Rejected because an unrelated container was included |
| Counter | `onion` | Rejected because no coherent fused group was produced |
| Counter | `cutlery` | v2 source config is prepared and pending review |
| Flowers | none | No custom extension required in the prominent-object pass |
| Garden | `garden_statue` | Rejected because the proposal absorbed a table |
| Garden | bench, plant pot | Ignored in this pass because the identities already exist in ADE20K |
| Kitchen | `toy_bulldozer` | v1 rejected because masks leaked into the table/placemats; tighter v2 source config is prepared |
| Kitchen | `oven_mitt` | v2 source config is prepared and pending review |
| Kitchen | chair | Ignored because `chair` already exists in ADE20K |

The v2 custom-class configs are source-evidence experiments. Their hybrid
defaults remain empty until the new evidence is reviewed.

## Interpretation boundary

The project targets prominent missing object identities, not exhaustive inventory
annotation. Broad corrections to surfaces such as countertops, tablecloths,
paving, or placemats are outside the current custom-extension pass. Likewise,
generic food or produce prompts are not added when the ADE20K ontology already
contains an appropriate class.

## Acceptance standard

A class can move from source evidence to a reviewed merge only when:

1. its prompts do not collide with an ADE20K class;
2. its fused groups are coherent across views;
3. overlays remain localized to the intended physical object;
4. base-label transitions do not remove more reliable semantics; and
5. the resulting output passes label, map, provenance, and PLY validation.
