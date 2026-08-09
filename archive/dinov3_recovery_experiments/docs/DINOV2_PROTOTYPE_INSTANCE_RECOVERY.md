# DINOv2 Prototype-Based Instance Recovery

Status: deferred design proposal. Do not implement or schedule without a new
review checkpoint.

## Objective

Reduce manual annotation while avoiding GroundingDINO as an authoritative
semantic identity source. The intended interaction is candidate review rather
than drawing masks in every view.

Model training and fine-tuning are out of scope.

## Proposed pipeline

```text
trusted immutable-v5 components
    -> extract high-confidence DINOv2 feature prototypes
    -> generate class-agnostic Mask2Former regions
    -> score each region against target and competing prototypes
    -> lift high-margin candidates as separate 3D components
    -> require multiview feature and projection consistency
    -> apply immutable-v5 protection
    -> rank candidates for user approval
    -> refine accepted candidates with SAM
    -> review before any versioned merge
```

## Identity evidence

For every trusted v5 class, extract DINOv2 features only from eroded,
high-confidence, multiview-supported component cores. Build both:

- positive prototypes for the target class;
- competing prototypes from other trusted scene classes.

Score a candidate by its margin over the strongest competing class, not by
target similarity alone:

```text
target similarity - maximum competing-class similarity
```

Low-margin candidates abstain. Declarative ontology aliases must normalize
equivalent names before comparison.

## Candidate and 3D policy

- Mask2Former supplies class-agnostic candidate regions only.
- Mask2Former semantic predictions do not assign identity.
- Each candidate is lifted and evaluated as a separate physical 3D component.
- Different instances of one class are never merged into a class-wide seed.
- A component must have consistent DINOv2 features and projections in
  multiple distinct views.
- Strong unrelated v5 evidence is protected; only unlabeled or explicitly
  uncertain evidence may be reconsidered automatically.
- Cross-class, cross-component, or excessive-growth conflicts abstain.

SAM refines boundaries only after a component identity has passed these
independent gates.

## Manual fallback levels

1. **Zero-click:** trusted v5 prototypes find and rank similar missing
   components.
2. **One exemplar per class:** one user-selected object supplies a stronger
   prototype for automatic search.
3. **One prompt per difficult instance:** a point or box supplies identity
   and location when automatic matching remains ambiguous.

A manual prompt creates a candidate mask. It never directly modifies v5.

## Limitations

- A visually novel instance may not match any trusted prototype.
- Similar doors, cabinets, shutters, and wall panels may still produce a low
  identity margin.
- Objects with no trustworthy automatic evidence may still require one point
  or box.
- Geometry and connected components validate consistency, not semantic
  identity.

Abstention and a one-prompt fallback are preferred to automatic leakage.

## Acceptance sequence

1. Report-only 2D prototype-ranking audit.
2. Report-only component-level multiview audit.
3. Immutable-v5 overlap and protected-evidence audit.
4. Visual approval of ranked candidates and SAM refinements.
5. No-PLY merge preflight into a new output.
6. Versioned merge only after explicit approval.

The immutable v5 output is never modified in place.
