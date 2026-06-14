# Four-Behavior Functional Weight Editing V10 V9-Conditioned Delta Preregistration

Date: 2026-06-12

## Purpose

V9 established a narrow representation-space result: fixed stored-probe
activation signatures can be moved toward train-only target-behavior attractors
on fresh heldout subjects. It did not edit subject weights or show changed model
behavior.

V10 tests the next smallest functional claim:

For the same four clean behaviors and subject architecture, a deterministic
train-only editor can use a V9-style target-attractor representation as
conditioning to edit heldout source weights with known source-behavior label and
requested target-behavior label so that the edited weights execute the requested
target behavior on heldout input cases.

This is a source-label-known functional weight-editing experiment on small
synthetic subjects. It is not a source-label-inference result, not a larger-model
result, not broad MUAT proof, and not evidence that the edited model preserves
all non-target capabilities.

## Prior Results And Motivation

Earlier four-behavior decoder attempts did not support a four-behavior
functional-decoding claim. V3 signature inversion was useful as infrastructure
but failed its specificity/control gates. V9 then showed reliable
representation-space target attraction, including same-target transfer probes,
but explicitly did not decode or edit weights.

V10 is therefore allowed to build on V9's representation machinery, but it must
evaluate actual edited weights as executable subject models.

## Contamination Policy

V1-V9 preregistrations, development artifacts, final summaries, and evidence
reports have been inspected. V10 must not reuse any prior final raw pool as a
V10 train or development input. V10 must generate fresh train, development, and
final source pools with V10-specific claim scopes and seeds.

V10 development may read:

- V10 train raw pool;
- V10 development raw pool;
- V10 combined source-pool audit;
- V10 final redacted source-pool audit.

V10 development must not read, parse, summarize, hash through loaded content, or
evaluate the V10 final raw pool. The final raw path may appear only as a literal
blocked path in guard code and documentation before final authorization.

The V10 final raw pool remains sealed unless:

1. this preregistration is accepted by reviewer at `5/5`;
2. V10 implementation is accepted by reviewer at `5/5`;
3. V10 source-pool construction is accepted by reviewer at `5/5`;
4. V10 development evaluation passes all gates below;
5. reviewer accepts the V10 development result at `5/5`.

If V10 development fails, final evaluation is blocked. Any method change after
reviewed development success requires a new V10 preregistration suffix and
invalidates final eligibility for this preregistration.

## Source Pools

V10 uses the same subject architecture, stored probes, behavior suite, and
source-generation acceptance criteria as V9.

V10 source-pool output directory:

`runs/four_behavior_functional_weight_editing_v10_pools`

Required V10 claim scopes:

- raw train/development/final pools:
  `four_behavior_functional_weight_editing_v10_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v10_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v10_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `60300000`;
- development base seed: `61300000`;
- final base seed: `62300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

Before development, V10 must validate V10 scopes, accepted counts, zero
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
evaluation invalidates the V10 final pool for proof use.

Before opening the V10 final raw pool, the final-evaluation command must
validate a current passing V10 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_functional_weight_editing_v10_v9_conditioned_delta_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `editor_method`: `v9_conditioned_train_only_ridge_weight_delta`;
- `train_pool_sha256`: current V10 train raw pool SHA-256;
- `eval_pool_sha256`: current V10 development raw pool SHA-256;
- `combined_audit_sha256`: current V10 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V10 final-redacted-audit SHA-256;
- `implementation_sha256`: current V10 experiment script SHA-256;
- `preregistration_sha256`: current V10 preregistration SHA-256.

If any value differs, final evaluation must fail before opening the V10 final
raw pool.

## Train-Only Representation Statistics

V10 recomputes the V9 train-only representation statistics from accepted V10
train subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only as a frozen baseline/control and primary
  floor reference.

The V9 target-attractor candidate generator is recomputed from V10 train-only
statistics. No V9 train, development, or final raw subject can be used as a V10
training example.

Fixed representation configuration:

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
- V8 source-conditional tournament logic is recomputed from V10 train-only
  statistics and used as the V10 target-attractor conditioning generator.

## Functional Editor Method

V10 trains a closed-form ridge regressor from train-only pair examples. The
editor is deterministic and has no access to development or final weights,
signatures, heldout metrics, or subject IDs during fitting. Evaluation is
explicitly source-label-known: the heldout subject's registered source behavior
label and the requested target behavior label are allowed conditioning inputs.
The method does not infer the source label and cannot be claimed as a
source-label-free editor.

For each ordered pair of accepted V10 train subjects `(source_subject,
target_subject)` with different behavior labels:

1. Compute the V9-style target-attractor selected representation candidate for
   `(source_behavior, target_behavior)` using only V10 train statistics.
2. Let `source_signature_norm` be the source stored-probe signature normalized
   by V10 train signature mean/std.
3. Let `selected_signature_norm` be the selected V9-style target-attractor
   candidate.
4. Build one feature vector containing:
   - `source_signature_norm`;
   - `selected_signature_norm`;
   - `selected_signature_norm - source_signature_norm`;
   - one-hot source behavior label;
   - one-hot target behavior label;
   - scalar selected primary target margin;
   - scalar selected centroid improvement;
   - scalar selected displacement norm.
5. Build the target vector as
   `target_weight_norm - source_weight_norm`, using V10 train weight mean/std.

The ridge solution is:

`coef = solve(X_aug.T @ X_aug + lambda * I_penalty, X_aug.T @ Y)`

where:

- `X_aug` is the feature matrix with a final intercept column;
- the intercept column is not ridge-penalized;
- `lambda = 10.0`;
- standard deviations below `1e-6` are clamped to `1.0`;
- all tensors use float32 inputs, float64 ridge solve, and float32 decoded
  weights for subject evaluation.

At evaluation time, the editor receives exactly the heldout source weights,
heldout source stored-probe signature, registered source behavior label, and
requested target behavior label. It computes the same V9-style selected
representation candidate, builds the feature vector, predicts a normalized
weight delta, adds it to the normalized source weights, and denormalizes the
edited flat weights.

No gradient optimization may be run on development or final examples.

## Evaluation Cases

Each accepted evaluation source subject is tested against every target behavior
different from its source behavior, for `96 * 3 = 288` source-target records per
phase.

For each edited weight vector, V10 evaluates:

- target behavior heldout BCE;
- target behavior heldout margin;
- source behavior heldout margin;
- target-vs-source margin difference;
- behavioral prediction under all four behavior evaluators;
- output MSE to the original source subject on source-behavior heldout cases;
- output MSE to a deterministic same-target train exemplar on target-behavior
  heldout cases.

The primary behavior evaluator is the actual subject forward pass on heldout
input cases. Representation-space metrics may be reported as diagnostics only.

## Controls

Every source-target record must evaluate these controls. Unless stated
otherwise, each control is evaluated on the matched requested target behavior and
the original source-behavior heldout cases:

- no edit source weights;
- null predicted delta, equivalent to no edit after normalization;
- reverse behavior-pair editor output: use the original source weights and
  source signature, but build the editor feature with `source_label =
  target_behavior`, `target_label = source_behavior`, and the V9-style selected
  representation candidate computed from the original source signature under
  that reversed behavior-pair request; apply the resulting delta to the original
  source weights;
- same-source other-target editor outputs for the two non-matched target
  behaviors, built with the true source label and each other target label, then
  applied to the original source weights;
- shuffled target request selected deterministically from the two behaviors
  neither source nor matched target, built with the true source label and the
  shuffled target label, then applied to the original source weights;
- deterministic nearest-train target-subject retrieval by V9 selected
  representation distance among accepted V10 train subjects with the target
  behavior;
- deterministic train target-behavior centroid weights, computed as the mean of
  normalized train weights for the target behavior and denormalized;
- deterministic train global centroid weights, computed as the mean of all
  normalized train weights and denormalized;
- `32` random norm-matched weight deltas.

Random norm-matched controls are generated by:

1. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v10_random_weight_delta"])`.
2. Seed `torch.Generator` with the first 16 hex characters modulo `2**31`.
3. Draw `32` sequential standard-normal vectors with the same shape/dtype as
   the normalized source weights.
4. Normalize each by its norm clamped below at `1e-12`.
5. Scale every vector to the predicted editor delta norm clamped below at
   `1e-12`.
6. Add each random delta to the normalized source weights and denormalize.

The nearest-train target-subject retrieval control is mandatory. It tests
whether V10 edits the source weights rather than merely achieving target
behavior by behaving like a retrieved target model.

The nearest-train target-subject retrieval control is selected exactly as
follows:

1. Use the matched record's `selected_signature_norm`, the V9-style
   target-attractor candidate in V10 train-normalized signature space.
2. Candidate train subjects are accepted V10 train subjects whose registered
   behavior label equals the requested target behavior.
3. For each candidate, compute squared Euclidean distance between
   `selected_signature_norm` and the candidate's train-normalized stored-probe
   signature.
4. Sort candidates by `(distance, weights_hash, signature_hash, subject_id)` and
   select the first candidate.
5. Use that candidate's actual flat train weights as the control weights. Do not
   average, optimize, or rescale them.

For shuffled target request selection:

1. Enumerate behaviors in `PATTERNS` order.
2. Keep behaviors not equal to the source behavior and not equal to the matched
   requested target behavior.
3. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v10_shuffled_target"])`.
4. Take the first 16 hex characters modulo the remaining behavior count.

The train target-behavior centroid and train global centroid controls are
computed in normalized weight space with V10 train weight mean/std, then
denormalized before functional evaluation.

All controls listed in this section are negative controls for individual
Pareto-dominance checks, aggregate Pareto-undominated rate, and
best-control target-margin aggregates. Random norm-matched controls are included
as 32 separate negative controls per record. The nearest-train target-subject
retrieval control is included in both the best-control target-margin aggregate
and the Pareto-dominance set.

## Individual Record Pass

A record passes only if the matched edited weights satisfy all gates:

- primary behavior prediction is the requested target behavior;
- target heldout margin is `> 0.20`;
- target-vs-source margin difference is `> 0.20`;
- source behavior heldout margin change relative to no-edit source is `< -0.05`;
- output MSE to the original source subject on source-behavior heldout cases is
  lower than the nearest-train target-subject retrieval control;
- no negative control Pareto-dominates the matched edit on the pair:
  `(target heldout margin, -source-output MSE)`.

The no-edit and null-delta controls may tie each other, but neither may be
counted as passing matched edits.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- aggregate individual pass rate `>= 0.85`;
- every ordered source-target direction has individual pass rate `>= 0.70`;
- aggregate target-behavior prediction rate `>= 0.90`;
- every ordered source-target direction has target-behavior prediction rate
  `>= 0.80`;
- aggregate mean matched target heldout margin `> 0.20`;
- aggregate mean matched target-vs-source margin difference `> 0.20`;
- aggregate mean source behavior margin change relative to no-edit source
  `< -0.05`;
- aggregate mean matched-minus-no-edit target margin `> 0.20`;
- aggregate mean matched-minus-nearest-train target-retrieval target margin
  `> -0.05`;
- aggregate mean matched-minus-best-control target margin `> 0.00`;
- aggregate mean nearest-train-minus-matched source-output MSE `> 0.00`;
- aggregate negative-control Pareto-undominated record rate `>= 0.85`;
- random norm-matched controls include exactly `32` controls per record;
- every record includes the nearest-train target-subject retrieval control.

The `matched-minus-nearest-train target-retrieval target margin` gate allows the
retrieved target subject to have slightly higher target margin, because retrieval
is a strong target-behavior control. V10 must still beat retrieval on preserving
source-output similarity, and it must beat the best negative control on average
target margin.

If any development gate fails, the result must be recorded as a negative
development result and final evaluation is not authorized.

## Final Gates

The final evaluation is one-shot and uses the exact reviewed method and
authorization artifact from development. It must pass the same gates as
development, except no thresholds may be weakened after seeing final results.

Final output must include:

- all aggregate gates;
- per-direction pass and prediction rates;
- per-record matched metrics;
- all control metrics;
- source-pool audit hashes;
- implementation and preregistration hashes;
- explicit `limitations` text saying the result is small-subject functional
  editing evidence only, not broad MUAT proof.

If final fails, it is recorded as a failed final result and no additional V10
final rerun may be used as proof without a new preregistered experiment.

## Reviewer Checkpoints

V10 requires a reviewer confidence score of `5/5` after each result-producing
step before proceeding:

1. preregistration review;
2. implementation and helper-test review;
3. source-pool audit review;
4. development-result review;
5. final-result review if final is authorized.

Reviewer prompts must ask specifically about data leakage, final-pool exposure,
control adequacy, metric/gate mismatch, target-label leakage, hidden adaptive
choice, and whether the claim is narrower than the evidence.
