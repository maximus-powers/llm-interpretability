# Four-Behavior Representation Steering V9 Source-Invariant Target Attractor Preregistration

Date: 2026-06-11

## Purpose

V7 and V8 are negative development results for source-conditional
non-domination. Their dominant failure mode was that same-target other-source
edits often Pareto-dominated the matched source-target edit. That means the
registered V7/V8 controls were testing source-conditional uniqueness, not only
whether fixed-probe signatures can be steered toward a target behavior.

V9 tests a narrower and more faithful claim:

For fixed stored-probe activation signatures, a deterministic train-only target
attractor can steer heldout subject representations toward each of four target
behaviors with high reliability, even when the edit is allowed to be largely
source-invariant. Same-target other-source edits are therefore evaluated as
positive transfer probes, not as negative controls.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model or broad MUAT generality.

## Contamination Policy

V1-V8 development artifacts were inspected. V9 must not evaluate development on
any prior steering development pool. V9 must generate fresh train,
development, and final source pools.

The V9 final raw pool remains sealed unless:

1. this preregistration is accepted by reviewer at `5/5`;
2. V9 implementation is accepted by reviewer at `5/5`;
3. V9 source-pool construction is accepted by reviewer at `5/5`;
4. V9 development evaluation passes all gates below;
5. reviewer accepts the V9 development result at `5/5`.

If V9 development fails, final evaluation is blocked.

## Source Pools

V9 uses the same subject architecture, stored probes, behavior suite, and source
generation settings as V7/V8.

V9 source-pool output directory:

`runs/four_behavior_representation_steering_v9_pools`

Required V9 claim scopes:

- raw train/development/final pools:
  `four_behavior_representation_steering_v9_source_pool`;
- combined audit:
  `four_behavior_representation_steering_v9_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v9_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `57300000`;
- development base seed: `58300000`;
- final base seed: `59300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

Before development, V9 must validate V9 scopes, accepted counts, zero
cross-pool accepted overlaps by seed/subject id/weight hash/signature hash,
source heldout margin gates, file hashes, and final redaction. Development may
read only train raw, development raw, combined audit, and final redacted audit.

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
evaluation invalidates the V9 final pool for proof use.

Before opening the V9 final raw pool, the final-evaluation command must validate
a current passing V9 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_representation_steering_v9_source_invariant_target_attractor_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `transport_method`: `train_source_invariant_target_attractor`;
- `train_pool_sha256`: current V9 train raw pool SHA-256;
- `eval_pool_sha256`: current V9 development raw pool SHA-256;
- `combined_audit_sha256`: current V9 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V9 final-redacted-audit SHA-256.

If any value differs, final evaluation must fail before opening the V9 final raw
pool.

## Train-Only Statistics

V9 computes all representation statistics from accepted V9 train subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only as a frozen baseline/control and primary
  floor reference.

Fixed train-only configuration:

- primary evaluator: linear signature evaluator, classifier seed `20261121`,
  AdamW learning rate `0.1`, weight decay `0.0001`, epochs `1000`;
- V5 calibration baseline: seed `20260801`, AdamW learning rate `0.03`,
  weight decay `0.001`, epochs `500`;
- PCA rank `48`;
- displacement norm cap `200.0`;
- covariance eigenvalue floor `0.0001`;
- behavior/global covariance shrinkage weights `0.75` and `0.25`;
- diagonal ratio clip `[0.25, 4.0]`;
- global normalization standard deviations below `1e-6` are clamped to `1.0`;
- V8 source-conditional tournament logic is recomputed from V9 train-only
  statistics and used as the matched V9 target-attractor candidate generator.

## Candidate Set

For each source-target record, V9 uses the accepted V8 candidate generator
recomputed from V9 train-only statistics:

- five radius-budgeted candidates;
- V8 correction seed `20261022`;
- `120` SGD steps;
- learning rate `0.05`;
- no momentum;
- no weight decay;
- V5 primary floor;
- V6 centroid floor;
- fixed detached V7 same-target tournament competitors.

The selected candidate is the valid matched candidate with highest primary
target margin, then highest centroid improvement, then lowest candidate index.
If no valid candidate exists, the result records a fallback selection for
diagnosis only and the record fails.

## Controls and Positive Transfer Probes

Negative controls per record:

- no edit;
- null vector;
- V2 centroid delta for the matched source-target pair;
- V3 diagonal transport for the matched source-target pair;
- V4 low-rank residual transport for the matched source-target pair;
- V5 contrastive residual calibration for the matched source-target pair;
- V6 centroid-constrained primary correction for the matched source-target pair;
- reverse V9 target-attractor candidates;
- same-source other-target V9 target-attractor candidates;
- shuffled-direction V9 target-attractor candidates;
- `32` random norm-matched vectors using the maximum matched V9 displacement
  norm.

Same-target other-source V9 candidates are not negative controls. They are
positive transfer probes. V9 reports their target-prediction and gate-pass rates
to test source-invariant target attraction.

The deterministic shuffled direction is selected by:

- enumerate all ordered behavior pairs with unequal source/target;
- exclude the matched `(source, target)` and reverse `(target, source)` pairs;
- sort lexicographically;
- compute `stable_hash_json([subject_id, source, target, "representation_steering_v9_source_invariant_target_attractor_shuffled_direction"])`;
- take the first 16 hex characters modulo the candidate count.

Random norm-matched controls are generated by:

- compute `stable_hash_json([subject_id, source, target, "representation_steering_v9_source_invariant_target_attractor_random"])`;
- seed a `torch.Generator` with the first 16 hex characters modulo `2**31`;
- draw `32` sequential standard-normal vectors with the same shape/dtype as the
  normalized signature;
- normalize each by its norm clamped below at `1e-12`;
- scale every vector to the maximum matched V9 frontier displacement norm
  clamped below at `1e-12`.

## Individual Record Pass

A record passes if at least one matched V9 candidate:

- primary classifier predicts the target behavior;
- centroid-nearest classifier predicts the target behavior;
- primary target margin is `> 0.25`;
- target-centroid improvement is `> 0.15`;
- source primary margin change is `< -0.05`;
- no negative control candidate Pareto-dominates it on primary target margin and
  target-centroid improvement.

Same-target transfer probes are excluded from this dominance set by design and
are reported separately.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- aggregate individual pass rate `>= 0.90`;
- aggregate negative-control Pareto-undominated record rate `>= 0.90`;
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
- aggregate same-target transfer candidate target-prediction rate `>= 0.90`;
- aggregate same-target transfer candidate gate-pass rate `>= 0.80`;
- per target: `n == 72`, individual pass rate `>= 0.80`,
  negative-control Pareto-undominated rate `>= 0.85`;
- per ordered direction: `n == 24`, individual pass rate `>= 0.80`,
  negative-control Pareto-undominated rate `>= 0.80`, target-prediction count
  `>= 20`.

No final raw evaluation is permitted unless V9 development passes all gates and
the reviewer accepts the development result at `5/5`.

## Required Reporting

Development and final result artifacts must include:

- claim scope, phase, pass/fail, next action, transport method, thresholds, and
  training config;
- train/eval pool paths and SHA-256 hashes;
- combined audit path/hash and final redacted audit path/hash;
- train-only statistics hash;
- V5 calibration hash and baseline training summary, labeled as baseline/control
  state rather than V9 evidence;
- aggregate, by-target, by-direction, transfer-probe, and individual-gate
  summaries;
- every record's matched V9 candidates, negative controls, same-target transfer
  probes, selected candidate index, selection reason, selected candidate
  metrics, Pareto dominator count/types, and random control count;
- failure list.

Development artifacts must not include a raw final pool path and must not expose
final per-subject records or final raw fields.
