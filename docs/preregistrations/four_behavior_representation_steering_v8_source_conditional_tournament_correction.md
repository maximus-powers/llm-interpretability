# Four-Behavior Representation Steering V8 Source-Conditional Tournament Correction Preregistration

Date: 2026-06-11

## Purpose

V7 is a negative development result. It achieved `245/288` individual passes and
`257/288` Pareto-undominated records, but failed preregistered development gates.
The dominant failure mode was source-conditional specificity: selected matched
candidates were Pareto-dominated mostly by same-target other-source V7 frontier
controls.

V8 tests one narrower follow-up claim:

For fixed stored-probe signatures, a deterministic train-only
source-conditional tournament correction can produce a small candidate set that
more reliably contains a target-behavior steering candidate that is not
Pareto-dominated by prior-method, non-matched V8, shuffled V8, and random
controls on primary target margin and target-centroid improvement.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model or broad MUAT generality.

## Contamination Policy

V1-V7 development artifacts were inspected. V8 must not evaluate development on
any prior steering development pool. V8 must generate fresh train, development,
and final source pools.

The V8 final raw pool remains sealed unless:

1. this preregistration is accepted by reviewer at `5/5`;
2. V8 implementation is accepted by reviewer at `5/5`;
3. V8 source-pool construction is accepted by reviewer at `5/5`;
4. V8 development evaluation passes all gates below;
5. reviewer accepts the V8 development result at `5/5`.

If V8 development fails, final evaluation is blocked.

## Source Pools

V8 uses the same subject architecture, stored probes, behavior suite, and source
generation settings as V7.

V8 source-pool output directory:

`runs/four_behavior_representation_steering_v8_pools`

Required V8 claim scopes:

- raw train/development/final pools:
  `four_behavior_representation_steering_v8_source_pool`;
- combined audit:
  `four_behavior_representation_steering_v8_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v8_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `54300000`;
- development base seed: `55300000`;
- final base seed: `56300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

Before development, V8 must validate V8 scopes, accepted counts, zero cross-pool
accepted overlaps by seed/subject id/weight hash/signature hash, source heldout
margin gates, file hashes, and final redaction. Development may read only train
raw, development raw, combined audit, and final redacted audit.

Allowed final redacted audit fields before final evaluation:

- pass/fail status;
- accepted counts by behavior;
- selected-training-vs-heldout overlap pass/fail or max overlap count;
- cross-pool overlap counts;
- final pool file hash;
- redacted payload hash;
- stored-probe hash;
- behavior-suite hashes;
- source-generation config hash;
- seed-range preflight pass/fail and configured seed ranges.

Allowed combined-audit final summary fields before final evaluation:

- accepted counts by behavior;
- final pool file hash;
- final redacted payload hash.

Forbidden final-detail fields before final evaluation:

- final per-subject records;
- final subject IDs;
- final behavior labels;
- final seeds;
- final attempt indices;
- final signatures;
- final signature hashes;
- final weights;
- final weight hashes;
- final source margins;
- final support or heldout margins;
- final attempt counts;
- final rejection counts;
- final acceptance rates;
- final accepted attempt indices;
- final rejected attempt indices;
- final accepted or rejected subject IDs;
- final per-subject metrics.

Any exposure of forbidden final raw or final-detail fields before final
evaluation invalidates the V8 final pool for proof use.

Forbidden before final evaluation:

- every prior raw final steering pool `final_subjects.json`;
- V8 raw final `runs/four_behavior_representation_steering_v8_pools/final_subjects.json`;
- any other `final_subjects.json`, except during the explicit V8 final command
  after a hash-bound passing V8 development artifact has been validated.

Before opening the V8 final raw pool, the final-evaluation command must validate
a current passing V8 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_representation_steering_v8_source_conditional_tournament_correction_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `transport_method`: `train_source_conditional_tournament_correction`;
- `train_pool_sha256`: current V8 train raw pool SHA-256;
- `eval_pool_sha256`: current V8 development raw pool SHA-256;
- `combined_audit_sha256`: current V8 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V8 final-redacted-audit SHA-256.

If any value differs, final evaluation must fail before opening the V8 final raw
pool.

## Train-Only Statistics

V8 computes all representation statistics from accepted V8 train subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only as a frozen baseline/control and primary
  floor reference.

Fixed train-only configuration:

- primary evaluator: linear signature evaluator, classifier seed `20261021`,
  AdamW learning rate `0.1`, weight decay `0.0001`, epochs `1000`;
- V5 calibration baseline: seed `20260801`, AdamW learning rate `0.03`,
  weight decay `0.001`, epochs `500`, primary rank margin `0.50`, target
  centroid rank margin `0.05`;
- PCA rank `48`;
- displacement norm cap `200.0`;
- covariance eigenvalue floor `0.0001`;
- behavior/global covariance shrinkage weights `0.75` and `0.25`;
- diagonal ratio clip `[0.25, 4.0]`;
- global normalization standard deviations below `1e-6` are clamped to `1.0`;
- V6 baseline control is the accepted V6 centroid-constrained primary
  correction logic recomputed from V8 train-only statistics with correction
  seed `20260922`, `120` steps, and learning rate `0.05`;
- V7 baseline frontier is the accepted V7 Pareto-frontier logic recomputed from
  V8 train-only statistics with correction seed `20260922`, `120` steps, and
  learning rate `0.05`.

## V8 Candidate Set

For each source-target record, V8 constructs five matched candidates with the
same registered radius budgets as V7:

1. `max(v4_distance - 0.05, 0)`;
2. `max(v4_distance + 0.50, 0)`;
3. `max(v4_distance + 1.50, 0)`;
4. `max(min(v5_distance, source_distance - 0.15), 0)`;
5. `max(source_distance - 0.15, 0)`.

Every budget is clamped to `max(source_distance - 0.15, 0)`.

Each candidate optimizes a PCA-subspace correction `q` for `120` SGD steps with
correction seed `20261022`, learning rate `0.05`, no momentum, and no weight decay. After every step, the
candidate is capped by the global displacement cap and projected back into the
target-centroid radius budget; `q` is overwritten from the projected candidate.

The V8 per-candidate loss is:

`target_ce + source_hinge + 0.001 * ||q||^2 + 0.25 * v5_primary_floor + 0.50 * v6_centroid_floor + 0.20 * same_target_primary_tournament + 0.50 * same_target_centroid_tournament`

where:

- `source_hinge = relu(primary_source_margin(candidate) + 0.05)`;
- `v5_primary_floor = relu(primary_target_margin(v5) - primary_target_margin(candidate) - 0.10)`;
- `v6_centroid_floor = relu(centroid_improvement(v6) + 0.06 - centroid_improvement(candidate))`;
- same-target tournament competitors are fixed V7 frontier candidates generated
  on the same record for the two non-source behaviors paired with the same
  target;
- the tournament competitor source set is:
  `{behavior for behavior in behavior_suite if behavior not in {source, target}}`;
- there are exactly ten same-target tournament competitors per record:
  two other sources times five V7 frontier candidates each;
- tournament competitors are computed once before each V8 candidate optimization
  using train-only V7 logic, detached, and never updated by V8 optimization;
- the matched source-target V7 frontier is not included in the tournament loss;
- `same_target_primary_tournament` is the mean
  `relu(primary_target_margin(competitor) - primary_target_margin(candidate) + 0.05)`;
- `same_target_centroid_tournament` is the mean
  `relu(centroid_improvement(competitor) - centroid_improvement(candidate) + 0.05)`.

The same V8 procedure is used for non-matched V8 directional controls.

## Controls

Per record, controls are:

- no edit;
- null vector;
- V2 centroid delta;
- V3 diagonal transport;
- V4 low-rank residual transport;
- V5 contrastive residual calibration;
- V6 centroid-constrained primary correction;
- V7 matched Pareto-frontier candidates for the same source-target direction;
- V8 reverse frontier candidates;
- V8 same-source other-target frontier candidates;
- V8 same-target other-source frontier candidates;
- V8 shuffled-direction frontier candidates;
- `32` random norm-matched vectors using the maximum matched V8 displacement
  norm.

The deterministic shuffled direction is selected by:

- enumerate all ordered behavior pairs with unequal source/target;
- exclude the matched `(source, target)` and reverse `(target, source)` pairs;
- sort lexicographically;
- compute `stable_hash_json([subject_id, source, target, "representation_steering_v8_source_conditional_tournament_correction_shuffled_direction"])`;
- take the first 16 hex characters modulo the candidate count.

Random norm-matched controls are generated by:

- compute `stable_hash_json([subject_id, source, target, "representation_steering_v8_source_conditional_tournament_correction_random"])`;
- seed a `torch.Generator` with the first 16 hex characters modulo `2**31`;
- draw `32` sequential standard-normal vectors with the same shape/dtype as the
  normalized signature;
- normalize each by its norm clamped below at `1e-12`;
- scale every vector to the maximum matched V8 frontier displacement norm
  clamped below at `1e-12`.

## Individual Record Pass

A record passes if at least one matched V8 candidate:

- primary classifier predicts the target behavior;
- centroid-nearest classifier predicts the target behavior;
- primary target margin is `> 0.25`;
- target-centroid improvement is `> 0.15`;
- source primary margin change is `< -0.05`;
- no control candidate Pareto-dominates it on primary target margin and
  target-centroid improvement.

Pareto-undominated record rate is reported separately from full individual pass
rate.

A record is Pareto-undominated if at least one matched V8 candidate is not
Pareto-dominated by any control candidate, regardless of absolute threshold,
prediction, or source-suppression gates. Aggregate, per-target, and
per-direction Pareto-undominated rates use denominators `288`, `72`, and `24`,
respectively.

## Required Reporting

Development and final result artifacts must include:

- claim scope, phase, pass/fail, next action, transport method, thresholds, and
  training config;
- train/eval pool paths and SHA-256 hashes;
- combined audit path/hash and final redacted audit path/hash;
- train-only statistics hash;
- V5 calibration hash and baseline training summary, labeled as baseline/control
  state rather than V8 evidence;
- aggregate, by-target, by-direction, and individual-gate summaries;
- every record's matched V8 candidates, controls, selected candidate index,
  selection reason, selected candidate metrics, Pareto dominator count/types,
  and random control count;
- failure list.

Development artifacts must not include a raw final pool path and must not expose
final per-subject records or final raw fields.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- aggregate individual pass rate `>= 0.90`;
- aggregate Pareto-undominated record rate `>= 0.90`;
- aggregate mean selected primary target margin `> 0.25`;
- aggregate mean selected centroid improvement `> 0.15`;
- aggregate mean source primary margin change `< -0.05`;
- aggregate mean selected-minus-V2 primary margin `> 0.10`;
- aggregate mean selected-minus-V2 centroid improvement `> 0.05`;
- aggregate mean selected-minus-V3 primary margin `> 0.10`;
- aggregate mean selected-minus-V3 centroid improvement `> 0.05`;
- aggregate mean selected-minus-V4 primary margin `> 0.10`;
- aggregate mean selected-minus-V4 centroid improvement `> 0.05`;
- aggregate mean selected-minus-V5 centroid improvement `> 0.05`;
- aggregate mean selected-minus-V6 primary margin `> 0.10`;
- aggregate mean selected-minus-V6 centroid improvement `> 0.05`;
- per target: `n == 72`, individual pass rate `>= 0.80`,
  Pareto-undominated rate `>= 0.85`;
- per ordered direction: `n == 24`, individual pass rate `>= 0.80`,
  Pareto-undominated rate `>= 0.80`, target-prediction count `>= 20`.

No final raw evaluation is permitted unless V8 development passes all gates and
the reviewer accepts the development result at `5/5`.
