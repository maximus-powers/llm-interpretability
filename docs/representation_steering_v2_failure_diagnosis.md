# Four-Behavior Representation Steering V2 Failure Diagnosis

Date: 2026-06-11

## Scope

This diagnosis covers the accepted V2 centroid-delta representation-steering
development artifact:

`runs/four_behavior_representation_steering_v2_centroid_delta/development_results.json`

It does not inspect or use any raw final pool. It is not a new proof result.
It explains why the frozen V2 development run failed and what this implies for
future preregistrations.

## V2 Result

The V2 centroid-delta development run failed:

- `passed: false`;
- development records: `288`;
- individual all-gate pass count: `142/288`;
- individual all-gate pass rate: `0.4930555556`;
- mean matched primary target margin: `47.6160739958`;
- mean matched-minus-best-control primary target margin: `24.2469312698`;
- mean matched centroid improvement: `1.5253242254`;
- mean matched-minus-best-control centroid improvement: `0.4016633564`;
- mean source primary margin change: `-116.9670098159`.

Final evaluation is blocked under the V2 preregistration. The V2 final raw pool
must remain sealed for this method.

## Failure Pattern

Aggregate movement was strong, but reliability was not:

- aggregate all-gate pass rate was `142/288 = 0.4930555556`, below the `0.90`
  gate;
- every target missed the `0.80` pass-rate gate;
- every ordered direction missed the `0.90` pass-rate gate.

The strongest direction-level pass rate was:

- `has_majority -> sorted_ascending`: `18/24 = 0.75`.

The weakest direction-level pass rates were:

- `has_majority -> mountain_pattern`: `3/24 = 0.125`;
- `mountain_pattern -> sorted_descending`: `5/24 = 0.2083333333`.

Matched vectors often reached the target class, so prediction alone was not the
main bottleneck. Per-direction primary-target prediction counts ranged from
`19/24` to `24/24`, and nearest-centroid target prediction counts ranged from
`18/24` to `23/24`.

Individual gate failures were dominated by control-specificity and reliability
checks:

- `matched_minus_best_control_centroid_improvement`: `90`;
- `matched_minus_best_control_primary_target_margin`: `70`;
- `matched_centroid_improvement`: `40`;
- `centroid_predicted_behavior`: `37`;
- `matched_primary_target_margin`: `28`;
- `primary_predicted_behavior`: `28`.

## Control Pattern

The strongest controls were usually centroid deltas pointing to the same target
from a different source behavior:

- best primary-control type:
  - `same_target_other_source_centroid_delta`: `278/288`;
  - `same_source_other_target_centroid_delta`: `10/288`;
- best centroid-control type:
  - `same_target_other_source_centroid_delta`: `140/288`;
  - `same_source_other_target_centroid_delta`: `102/288`;
  - `no_edit`: `46/288`.

This means exact train centroid deltas often move representations toward the
target region, but source-specific matched deltas are not reliably better than
other source-to-same-target deltas. The failure is therefore not that centroid
directions contain no behavior information. The failure is that the V2 matched
direction is not reliably source-specific under the preregistered controls.

## Direction-Level Specificity

Mean matched-minus-best-control primary target margin was negative for:

- `has_majority -> mountain_pattern`: `-13.3889829715`;
- `mountain_pattern -> sorted_descending`: `-4.7371200323`.

Mean matched-minus-best-control centroid improvement was negative for:

- `mountain_pattern -> sorted_ascending`: `-0.2407502333`;
- `mountain_pattern -> sorted_descending`: `-0.1792594592`.

All other directions had positive mean specificity on these aggregate metrics,
but none cleared the preregistered individual pass-rate gate.

## Interpretation

V2 is a valid negative for the frozen centroid-delta protocol.

It supports only a weak development observation: train-only centroid deltas can
produce strong average target movement in fixed-probe representation space.
It does not support proof-grade four-behavior representation steering because
the movement is not reliable enough at the individual record, target, and
direction levels.

The most important next-method constraint is source-specificity. A future V3
should not merely ask whether a vector points toward the target. It should test
whether source-conditioned transformations beat same-target other-source
centroid deltas without relaxing the heldout final-access policy.

Any V3 method requires a new preregistration and reviewer acceptance before
development or final-pool access.

