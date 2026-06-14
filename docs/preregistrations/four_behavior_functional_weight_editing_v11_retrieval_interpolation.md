# Four-Behavior Functional Weight Editing V11 Retrieval-Interpolation Preregistration

Date: 2026-06-12

## Purpose

V10 tested a V9-conditioned ridge weight-delta editor and failed development.
A posthoc V10 development-only diagnosis, explicitly marked not proof, found
that raw layer splicing remained weak but simple interpolation between heldout
source weights and the nearest train target-retrieval weights was much stronger.

V11 tests that observation on fresh pools with all choices fixed before V11
development:

For the same four clean behaviors and subject architecture, a deterministic
source-label-known editor can use fixed stored-probe signatures to retrieve a
train target model and produce edited heldout source weights by fixed raw
weight-space interpolation. The edited weights should execute the requested
target behavior in every ordered source-target direction and, in aggregate, be
closer to the original source function than full target retrieval.

This is a narrow small-subject functional editing experiment. It is not
source-label inference, not source-free decoding, not larger-model evidence, not
broad MUAT proof, and not evidence that non-target capabilities are preserved.

## Prior Development Inputs

V11 is motivated by:

- positive V9 representation-space source-invariant target-attractor final
  evidence;
- negative V10 functional ridge-edit development evidence;
- V10 posthoc development-only interpolation diagnosis.

The V10 posthoc diagnosis may motivate V11 design but is not evidence for the
V11 claim. V11 must use fresh V11 pools for proof use.

## Contamination Policy

V1-V10 preregistrations, development artifacts, final summaries, and evidence
reports have been inspected. V11 must not reuse any prior final raw pool as a
V11 train or development input. V11 must generate fresh train, development, and
final source pools with V11-specific claim scopes and seeds.

V11 development may read:

- V11 train raw pool;
- V11 development raw pool;
- V11 combined source-pool audit;
- V11 final redacted source-pool audit.

V11 development must not read, parse, summarize, hash through loaded content, or
evaluate the V11 final raw pool. The final raw path may appear only as a literal
blocked path in guard code and documentation before final authorization.

The V11 final raw pool remains sealed unless:

1. this preregistration is accepted by reviewer at `5/5`;
2. V11 implementation is accepted by reviewer at `5/5`;
3. V11 source-pool construction is accepted by reviewer at `5/5`;
4. V11 development evaluation passes all gates below;
5. reviewer accepts the V11 development result at `5/5`.

If V11 development fails, final evaluation is blocked. Any method change after
reviewed development success requires a new preregistration suffix and
invalidates final eligibility for this preregistration.

## Source Pools

V11 uses the same subject architecture, stored probes, behavior suite, and
source-generation acceptance criteria as V9/V10.

V11 source-pool output directory:

`runs/four_behavior_functional_weight_editing_v11_pools`

Required V11 claim scopes:

- raw train/development/final pools:
  `four_behavior_functional_weight_editing_v11_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v11_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v11_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `63300000`;
- development base seed: `64300000`;
- final base seed: `65300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

Before development, V11 must validate V11 scopes, accepted counts, zero
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
evaluation invalidates the V11 final pool for proof use.

Before opening the V11 final raw pool, the final-evaluation command must
validate a current passing V11 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_functional_weight_editing_v11_retrieval_interpolation_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `editor_method`: `v9_selected_target_retrieval_raw_weight_interpolation_alpha_0_95`;
- `train_pool_sha256`: current V11 train raw pool SHA-256;
- `eval_pool_sha256`: current V11 development raw pool SHA-256;
- `combined_audit_sha256`: current V11 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V11 final-redacted-audit SHA-256;
- `implementation_sha256`: current V11 experiment script SHA-256;
- `preregistration_sha256`: current V11 preregistration SHA-256.

If any value differs, final evaluation must fail before opening the V11 final
raw pool.

## Train-Only Representation Statistics

V11 recomputes the V9 target-attractor representation machinery from accepted
V11 train subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only inside the frozen V9-style selected
  target-attractor candidate generator.

No V1-V10 raw final subject may be used as a V11 train, development, or
retrieval candidate.

Fixed representation configuration is identical to V9/V10:

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
- V8 source-conditional tournament logic is recomputed from V11 train-only
  statistics and used as the selected target-attractor conditioning generator.

## Editor Method

V11 is deterministic and has no fitted weight-delta regressor.

At evaluation time, the editor receives exactly:

- heldout source weights;
- heldout source stored-probe signature;
- registered source behavior label;
- requested target behavior label.

For each source-target record:

1. Normalize the source signature with V11 train signature mean/std.
2. Compute the V9-style selected target-attractor representation candidate for
   `(source_behavior, target_behavior)` using V11 train-only statistics.
3. Retrieve the nearest accepted V11 train subject with the requested target
   behavior by squared Euclidean distance between the selected target-attractor
   candidate and each target train subject's train-normalized stored-probe
   signature.
4. Break retrieval ties by `(distance, weights_hash, signature_hash,
   subject_id)`.
5. Let `alpha = 0.95`.
6. Produce edited flat weights:

   `edited_weights = (1 - alpha) * source_weights + alpha * retrieved_target_weights`

The interpolation is in raw flat-weight space, not normalized weight space. No
gradient optimization, behavioral fine-tuning, or development/final label
adaptation is allowed.

## Evaluation Cases

Each accepted evaluation source subject is tested against every target behavior
different from its source behavior, for `96 * 3 = 288` source-target records per
phase.

For each edited weight vector, V11 evaluates:

- target behavior heldout margin;
- source behavior heldout margin;
- target-vs-source margin difference;
- behavioral prediction under all four behavior evaluators;
- output MSE to the original source subject on source-behavior heldout cases;
- target-margin difference from full nearest-target retrieval;
- source-output-MSE improvement over full nearest-target retrieval.

The primary behavior evaluator is the actual subject forward pass on heldout
input cases. Representation-space metrics may be reported as diagnostics only.

## Controls

Every source-target record must evaluate these controls:

- no edit source weights;
- full nearest-target retrieval weights, equivalent to `alpha = 1.0`;
- interpolation controls with fixed alphas `0.25`, `0.50`, `0.75`, `0.85`, and
  `0.90` using the same retrieved target subject;
- same-source other-target interpolation at `alpha = 0.95` for the two
  non-matched target behaviors;
- shuffled target interpolation at `alpha = 0.95`, where the shuffled target is
  selected from the two behaviors neither source nor matched target by
  `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v11_shuffled_target"])`;
- train target-behavior centroid weights, computed in raw weight space as the
  mean of accepted train weights for the requested target behavior;
- train global centroid weights, computed in raw weight space as the mean of all
  accepted train weights;
- `32` random norm-matched weight deltas around the source weights, scaled to
  the V11 edited delta norm.

Shuffled target selection is deterministic:

1. Enumerate behaviors in lexicographic order.
2. Keep behaviors not equal to the source behavior and not equal to the matched
   requested target behavior.
3. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v11_shuffled_target"])`.
4. Convert the first 16 hex characters to an integer and take modulo the
   remaining behavior count.
5. Record the selected shuffled target behavior in the control metrics.

Random norm-matched controls are deterministic:

1. Work in raw flat-weight space with the same dtype and shape as
   `source_weights`.
2. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v11_random_weight_delta"])`.
3. Seed `torch.Generator` with the first 16 hex characters modulo `2**31`.
4. For control indices `0..31`, draw sequential standard-normal vectors with
   `torch.randn(source_weights.shape, generator=generator, dtype=source_weights.dtype)`.
5. Normalize each vector by its L2 norm clamped below at `1e-12`.
6. Scale every vector to the V11 edited raw delta norm
   `((edited_weights - source_weights).norm())`, clamped below at `1e-12`.
7. Add the scaled delta to `source_weights`.
8. Record the seed, index, and delta norm for every random control.

All controls are included in best-control target-margin aggregates and
Pareto-dominance checks. Full nearest-target retrieval is a required strong
control. It may have higher target margin than the edited interpolation, but it
must not dominate unless it is also at least as source-similar.

## Individual Record Pass

A record passes only if the matched edited weights satisfy all gates:

- primary behavior prediction is the requested target behavior;
- target heldout margin is `> 0.20`;
- target-vs-source margin difference is `> 0.20`;
- output MSE to the original source subject on source-behavior heldout cases is
  lower than the full nearest-target retrieval control;
- no negative control Pareto-dominates the matched edit on the pair:
  `(target heldout margin, -source-output MSE)`.

This individual pass is a joint target-behavior plus source-similarity gate. It
is expected to be stricter than target behavior alone.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- aggregate individual pass rate `>= 0.85`;
- every ordered source-target direction has individual pass rate `>= 0.70`;
- every ordered source-target direction has negative-control Pareto-undominated
  record rate `>= 0.70`;
- every ordered source-target direction has mean full-retrieval-minus-matched
  source-output MSE `> 0.0`;
- every ordered source-target direction has target-behavior prediction rate
  `>= 0.90`;
- every ordered source-target direction has mean target heldout margin `> 0.20`;
- aggregate target-behavior prediction rate `>= 0.95`;
- aggregate mean matched target heldout margin `> 0.50`;
- aggregate mean matched target-vs-source margin difference `> 0.50`;
- aggregate mean matched-minus-full-retrieval target margin `> -0.25`;
- aggregate mean full-retrieval-minus-matched source-output MSE `> 20.0`;
- aggregate negative-control Pareto-undominated record rate `>= 0.85`;
- every record includes the full nearest-target retrieval control;
- random norm-matched controls include exactly `32` controls per record.

The source-similarity claim includes both aggregate and per-direction protection
against the known V10 posthoc weak direction. If any ordered direction fails the
mean full-retrieval-minus-matched source-output MSE gate, final evaluation is
blocked even if aggregate target prediction is strong.

If any development gate fails, the result must be recorded as a negative
development result and final evaluation is not authorized.

## Final Gates

The final evaluation is one-shot and uses the exact reviewed method and
authorization artifact from development. It must pass the same gates as
development, with no threshold weakening after seeing final results.

Final output must include:

- all aggregate gates;
- per-direction target-prediction, target-margin, pass, and source-similarity
  summaries;
- per-record matched metrics;
- all control metrics;
- source-pool audit hashes;
- implementation and preregistration hashes;
- explicit `limitations` text saying the result is small-subject,
  source-label-known, retrieval-anchored interpolation evidence only.

If final fails, it is recorded as a failed final result and no additional V11
final rerun may be used as proof without a new preregistered experiment.

## Reviewer Checkpoints

V11 requires reviewer confidence `5/5` after each result-producing step before
proceeding:

1. preregistration review;
2. implementation and helper-test review;
3. source-pool audit review;
4. development-result review;
5. final-result review if final is authorized.

Reviewer prompts must ask specifically about data leakage, final-pool exposure,
control adequacy, source-label-known scope, retrieval-vs-edit overclaim,
metric/gate mismatch, hidden adaptive choice, and whether the claim is narrower
than the evidence.
