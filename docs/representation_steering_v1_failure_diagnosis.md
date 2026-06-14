# Four-Behavior Representation Steering V1 Failure Diagnosis

Date: 2026-06-11

## Scope

This diagnosis covers the corrected V1 representation-steering development
artifact:

`runs/four_behavior_representation_steering_v1/development_results.json`

It does not inspect or use either raw final pool:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`.

It is not a new proof result. It explains why the frozen V1 method failed and
what this implies for a possible V2 preregistration.

## Corrected V1 Result

The corrected no-edit-relative V1 development run failed:

- `passed: false`;
- development records: `288`;
- individual all-gate pass count: `16/288`;
- individual all-gate pass rate: `0.0555555556`;
- mean matched primary target margin: `101.2225835986`;
- mean matched-minus-best-control primary target margin: `60.9470050931`;
- mean matched centroid improvement: `0.2559432056`;
- mean matched-minus-best-control centroid improvement: `-1.0535927349`;
- mean source primary margin change: `-177.3604026188`;
- selected edit-vector epoch: `140`.

The accepted evidence package treats the earlier `55/288` artifact as
superseded because its training objective did not implement no-edit-relative
centroid improvement.

## Failure Pattern

The primary linear evaluator gates were not the bottleneck:

- best primary-margin control type was `target_source_centroid_delta` for
  `285/288` records;
- matched steering beat the centroid-delta control on primary target margin for
  every record;
- mean matched-minus-best-control primary target margin was strongly positive:
  `60.9470050931`.

The bottleneck was centroid-control specificity:

- best centroid-improvement control type was `target_source_centroid_delta` for
  `244/288` records;
- best centroid-improvement control type was `no_edit` for `41/288` records;
- best centroid-improvement control type was `same_source_other_target_edit_vector`
  for `3/288` records;
- matched steering beat the centroid-delta control on centroid improvement for
  only `21/288` records;
- mean matched-minus-centroid-delta centroid improvement was `-0.9224472046`;
- mean matched-minus-best-control centroid improvement was `-1.0535927349`.

Individual gate failures were dominated by this single check:

- `matched_minus_best_control_centroid_improvement`: `272` failed records;
- `matched_centroid_improvement`: `106` failed records;
- `matched_primary_target_margin`: `4` failed records.

## Direction-Level Pattern

Every ordered source-target direction failed the preregistered `0.90`
individual pass-rate gate.

Mean matched-minus-centroid-delta centroid improvement was negative for all
twelve directions:

- `has_majority -> mountain_pattern`: `-1.4524783691`;
- `has_majority -> sorted_ascending`: `-1.1453957160`;
- `has_majority -> sorted_descending`: `-1.2938990593`;
- `mountain_pattern -> has_majority`: `-1.9832740625`;
- `mountain_pattern -> sorted_ascending`: `-0.5151786009`;
- `mountain_pattern -> sorted_descending`: `-1.4272315502`;
- `sorted_ascending -> has_majority`: `-0.5631402334`;
- `sorted_ascending -> mountain_pattern`: `-0.4875353177`;
- `sorted_ascending -> sorted_descending`: `-0.7304630280`;
- `sorted_descending -> has_majority`: `-0.4979002476`;
- `sorted_descending -> mountain_pattern`: `-0.3996926149`;
- `sorted_descending -> sorted_ascending`: `-0.5731776555`.

## Root Cause Interpretation

V1 used the train centroid delta as both:

1. the edit-vector initialization, and
2. a proof-critical centroid-improvement control that the learned vector had to
   beat per record.

For a global source-target vector and a centroid-distance objective, the vector
`centroid[target] - centroid[source]` is a very strong baseline. Under squared
Euclidean distance it is the mean-optimal translation that moves the source
cluster toward the target centroid. Under Euclidean distance it is still a
strong near-oracle translation baseline.

The corrected V1 training moved vectors away from this baseline to satisfy the
primary linear-evaluator target-margin and source-suppression objectives. That
produced very large primary-margin wins, but usually reduced target-centroid
distance improvement relative to the centroid-delta control.

Therefore V1 failed because its proof gate required one global vector to both:

- outperform a train-only primary classifier target-margin control; and
- outperform a near-oracle centroid translation on centroid-distance
  improvement.

The result is a valid negative for the frozen V1 protocol, but it does not show
that fixed-probe representation steering is impossible.

## Implication For V2

A V2 representation-steering preregistration should avoid treating
`centroid[target] - centroid[source]` as an adversarial control that must be
beaten on the same centroid-distance metric it is designed to optimize.

Conservative options:

- Treat centroid-delta as an oracle/reference baseline and require the learned
  vector to retain a preregistered fraction of its centroid improvement while
  beating it on primary target margin and source suppression.
- Use centroid-delta as the V2 steering vector itself, then evaluate whether a
  transparent train-only centroid translation can pass representation-level
  steering gates against no-edit, reverse, random, shuffled, and same-source
  other-target controls.
- Keep centroid-delta as an adversarial control only for metrics it is not
  directly optimized to maximize, such as heldout primary classifier target
  margin or source suppression.

Any V2 method or gate change after this diagnosis requires a new
preregistration and reviewer acceptance before development or final-pool access.
