# Four-Behavior Functional Weight Editing V13 Hybrid Signature-Support Optimization Preregistration

Date: 2026-06-12

## Purpose

V11 and V12 produced valid negative development results. They showed strong
target transfer, but failed proof-grade preservation/Pareto gates. A V12
development-only diagnostic showed that source-initialized support optimization
can repair several weak directions, while a signature-only differentiable
optimization does not reliably induce target behavior from source weights.

V13 tests a narrower and more falsifiable claim:

For the same four clean behaviors and subject architecture, fixed stored-probe
signatures can provide a useful target-attractor regularizer inside a
source-initialized functional edit. The matched editor must not only pass
functional target/preservation gates; it must also outperform no-signature and
shuffled-signature optimization controls. If it does not, V13 is negative
evidence for fixed-probe signatures adding functional editing value beyond
ordinary supervised support-set optimization.

This is a narrow small-subject experiment. It is source-label-known and
target-label-requested. It is not source-label inference, not source-free
decoding, not larger-model evidence, not broad MUAT proof, and not evidence that
unrelated capabilities are preserved.

## Prior Development Inputs

V13 is motivated by:

- positive V9 representation-space source-invariant target-attractor final
  evidence;
- negative V10 ridge-edit development evidence;
- negative V11 retrieval-interpolation development evidence;
- negative V12 conflict-aware aligned interpolation development evidence;
- V12 development-only diagnostics showing:
  - support optimization can improve weak directions;
  - signature-only optimization is not sufficient for reliable target behavior.

These prior diagnostics may motivate V13 design but are not evidence for the V13
claim. V13 must use fresh V13 pools for proof use.

## Contamination Policy

V1-V12 preregistrations, development artifacts, final summaries, and evidence
reports have been inspected. V13 must not reuse any prior final raw pool as a
V13 train or development input. V13 must generate fresh train, development, and
final source pools with V13-specific claim scopes and seeds.

V13 development may read:

- V13 train raw pool;
- V13 development raw pool;
- V13 combined source-pool audit;
- V13 final redacted source-pool audit.

V13 development must not read, parse, summarize, hash through loaded content, or
evaluate the V13 final raw pool. The final raw path may appear only as a literal
blocked path in guard code and documentation before final authorization.

The V13 final raw pool remains sealed unless:

1. this preregistration is accepted by reviewer at `5/5`;
2. V13 implementation is accepted by reviewer at `5/5`;
3. V13 source-pool construction is accepted by reviewer at `5/5`;
4. V13 development evaluation passes all gates below;
5. reviewer accepts the V13 development result at `5/5`.

If V13 development fails, final evaluation is blocked. Any method change after
reviewed development success requires a new preregistration suffix and
invalidates final eligibility for this preregistration.

## Source Pools

V13 uses the same subject architecture, stored probes, behavior suite, and
source-generation acceptance criteria as V9-V12.

V13 source-pool output directory:

`runs/four_behavior_functional_weight_editing_v13_pools`

Required V13 claim scopes:

- raw train/development/final pools:
  `four_behavior_functional_weight_editing_v13_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v13_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v13_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `69300000`;
- development base seed: `70300000`;
- final base seed: `71300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

Before development, V13 must validate V13 scopes, accepted counts, zero
cross-pool accepted overlaps by seed/subject id/weight hash/signature hash,
source heldout margin gates, file hashes, and final redaction. Development may
read only train raw, development raw, combined audit, and final redacted audit.

Allowed `combined_audit.pool_summaries.final` fields before final evaluation:

- `accepted_counts_by_behavior`;
- `pool_file_sha256`;
- `pool_redacted_payload_sha256`.

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

Forbidden final-detail fields before final evaluation are the same as V12:
per-subject records, subject IDs, behavior labels, seeds, attempt indices,
signatures, signature hashes, weights, weight hashes, source/support/heldout
margins, attempt/rejection counts, accepted/rejected subject IDs, and per-subject
metrics. Any exposure of forbidden final raw or final-detail fields before final
evaluation invalidates the V13 final pool for proof use.

Before opening the V13 final raw pool, the final-evaluation command must
validate a current passing V13 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_functional_weight_editing_v13_hybrid_signature_support_optimization_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `editor_method`:
  `v9_selected_target_signature_source_initialized_support_optimization_v13`;
- `train_pool_sha256`: current V13 train raw pool SHA-256;
- `eval_pool_sha256`: current V13 development raw pool SHA-256;
- `combined_audit_sha256`: current V13 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V13 final-redacted-audit SHA-256;
- `implementation_sha256`: current V13 experiment script SHA-256;
- `preregistration_sha256`: current V13 preregistration SHA-256.

If any value differs, final evaluation must fail before opening the V13 final
raw pool.

## Fixed Suite And Probe Inputs

V13 uses the deterministic clean behavior suite from
`hypernet.behavior_suite.build_clean_behavior_suite` with exactly:

- patterns:
  `["sorted_ascending", "sorted_descending", "has_majority", "mountain_pattern"]`;
- `support_per_class = 160`;
- `heldout_per_class = 64`;
- `seed = 20260609`;
- `seq_len = 5`;
- `base = 10`;
- support hash:
  `8f75b98bd5ba8c7fff78289de3c9d40c1621d5820709cb8bb02fd58cf59500f3`;
- heldout hash:
  `c65c0c32aba2dc62db3ea7af3999859027a236ce98e21c091f50fab503a07204`.

Support cases are the only examples allowed in optimizer losses. Heldout cases
are proof metrics only.

V13 uses deterministic stored probes from
`hypernet.paired_contrast.build_digit_probe_examples` with exactly:

- `n_examples = 256`;
- `seed = 20260610`;
- `seq_len = 5`;
- `base = 10`;
- probe examples hash:
  `b156dabece5a9eb58a966271388c8e5479fd308712dcca7b373e0f253e670279`.

Stored probes are used only for signature extraction. They are not support cases,
not heldout cases, and not sampled from train/development/final pools.

## Train-Only Representation Statistics

V13 recomputes the V9 target-attractor representation machinery from accepted
V13 train subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only inside the frozen V9-style selected
  target-attractor candidate generator.

No V1-V12 raw final subject may be used as a V13 train, development, optimizer,
retrieval candidate, or control input.

## Editor Method

V13 is deterministic and source-initialized. It has no learned weight-delta
regressor.

At evaluation time, the matched editor receives exactly:

- heldout source weights;
- heldout source stored-probe signature;
- registered source behavior label;
- requested target behavior label.

For each source-target record:

1. Normalize the source signature with V13 train signature mean/std.
2. Compute the V9-style selected target-attractor representation candidate for
   `(source_behavior, target_behavior)` using V13 train-only statistics.
3. Convert the selected normalized candidate to raw signature units using V13
   train signature mean/std.
4. Initialize editable flat weights to the heldout source weights.
5. Optimize the flat weights for exactly `130` AdamW steps with learning rate
   `0.03`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.0`,
   `amsgrad=False`, and full-batch deterministic losses.
6. Use fixed support-set losses only:
   - target support binary cross-entropy on requested target positive/negative
     support examples, weight `4.0`;
   - source support conflict binary cross-entropy on source support conflict
     examples using target predicate labels, weight `2.0`;
   - compatible source support logit MSE to the original source model on
     compatible source support cases, weight `0.01`;
   - differentiable stored-probe signature MSE between the edited normalized
     signature and selected target-attractor normalized signature, weight `0.01`;
   - source-weight L2 MSE, weight `0.0005`.
7. After `loss.backward()`, clip global gradient norm across the editable flat
   weight tensor to `10.0`, then run exactly one AdamW optimizer step.
8. Use the final weights after optimizer step `129`, meaning after exactly
   `130` optimizer updates indexed `0..129`. No intermediate checkpoint
   is selected, saved as the matched edit, or evaluated for method choice.

No development/final heldout cases may be used in the optimization loss or
checkpoint selection except through the fixed source-generation acceptance
process that created the pools. Because V13 has no checkpoint selection, heldout
cases are used only for evaluation.

## Support-Loss Semantics

For every source-target record, build support tensors only from
`suite["support"]`.

Target support BCE:

- inputs are
  `suite["support"][target]["positive"] + suite["support"][target]["negative"]`;
- labels are `1.0` for the first `160` positive cases and `0.0` for the next
  `160` negative cases;
- compute edited logits, not sigmoid probabilities;
- loss is `torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)`
  with PyTorch default `reduction="mean"` across all `320` target support cases.

Source support compatible/conflict split:

- inputs are
  `suite["support"][source]["positive"] + suite["support"][source]["negative"]`;
- source and target labels are computed directly with fixed `PREDICATES`;
- compatible cases have equal source and target predicate labels;
- conflict cases have different source and target predicate labels;
- any support compatible/conflict count mismatch with the table below fails
  development/final.

| Direction | Support Compatible | Support Conflict |
| --- | ---: | ---: |
| `sorted_ascending_to_sorted_descending` | 158 | 162 |
| `sorted_ascending_to_has_majority` | 105 | 215 |
| `sorted_ascending_to_mountain_pattern` | 137 | 183 |
| `sorted_descending_to_sorted_ascending` | 160 | 160 |
| `sorted_descending_to_has_majority` | 101 | 219 |
| `sorted_descending_to_mountain_pattern` | 139 | 181 |
| `has_majority_to_sorted_ascending` | 160 | 160 |
| `has_majority_to_sorted_descending` | 160 | 160 |
| `has_majority_to_mountain_pattern` | 80 | 240 |
| `mountain_pattern_to_sorted_ascending` | 160 | 160 |
| `mountain_pattern_to_sorted_descending` | 158 | 162 |
| `mountain_pattern_to_has_majority` | 82 | 238 |

Source support conflict BCE:

- inputs are the conflict cases from the source support split;
- labels are the target predicate labels for those conflict cases, cast to
  float;
- compute edited logits, not sigmoid probabilities;
- loss is `binary_cross_entropy_with_logits` with default `reduction="mean"`
  across conflict cases.

Compatible source support logit MSE:

- inputs are compatible cases from the source support split;
- target values are original source-model logits on the same compatible support
  cases, detached before optimization;
- edited values are edited-model logits;
- loss is `torch.nn.functional.mse_loss(edited_logits, source_logits)` with
  default `reduction="mean"` across compatible cases.

Source-weight L2 MSE is `torch.nn.functional.mse_loss(edited_flat_weights,
source_flat_weights.detach())` with default `reduction="mean"` over raw flat
weights.

## Differentiable Signature-Loss Semantics

The differentiable signature extractor mirrors the stored-probe feature layout
used for V13 pool signatures, but keeps gradients through the edited flat
weights.

For each edited flat-weight vector:

1. Build probe input tensor from the fixed 256 stored probes above as
   `float32` digit sequences of shape `[256, 5]`.
2. Run the subject architecture as five hidden layers with shapes
   `(8,5), (8,8), (8,8), (8,8), (8,8)` and GELU activation after each layer.
3. For every layer in order and every neuron index `0..7`, append exactly:
   mean activation, population std (`unbiased=False`), first five FFT magnitude
   values over the 256 probe activations, five safe Pearson correlations with
   probe input columns `0..4`, the same mean again, and the same std again.
4. Stack the features in this order into a `560`-dimensional signature vector.
5. Normalize the edited signature as
   `(edited_signature - train_signature_mean) / train_signature_std`, where both
   train tensors are computed from V13 train raw subjects only.
6. Compare to a detached, fixed normalized target signature vector. For the
   matched editor this target is the selected V9-style target-attractor
   candidate. For source-signature and shuffled-signature controls, the target is
   replaced exactly as described in `Controls`.
7. Signature loss is `torch.nn.functional.mse_loss(edited_signature_norm,
   target_signature_norm.detach())` with default `reduction="mean"` across all
   560 features.

Signature extraction must not use support examples, heldout examples, final raw
pool records, final redacted per-subject data, or V1-V12 raw pool records.

## Compatible/Conflict Evaluation

V13 uses the same conflict-aware heldout evaluation semantics as V12.

For each source behavior's heldout positive and negative cases, compute source
and target predicate labels directly with fixed `PREDICATES`. Compatible source
cases have equal source and target labels; conflict source cases have different
labels. Model labels for conflict accuracy use `sigmoid(logit) >= 0.5`.

Compatible source-output MSE is mean squared error between edited logits and
original source logits on compatible heldout source cases. Conflict target-label
accuracy is the fraction of conflict heldout source cases whose edited model
label equals the target predicate label. Conflict improvement is edited conflict
accuracy minus original source model conflict accuracy.

V13 must use this exact compatible/conflict count table for all 12 ordered
directions; any count mismatch fails development/final:

| Direction | Compatible | Conflict |
| --- | ---: | ---: |
| `sorted_ascending_to_sorted_descending` | 63 | 65 |
| `sorted_ascending_to_has_majority` | 38 | 90 |
| `sorted_ascending_to_mountain_pattern` | 59 | 69 |
| `sorted_descending_to_sorted_ascending` | 64 | 64 |
| `sorted_descending_to_has_majority` | 50 | 78 |
| `sorted_descending_to_mountain_pattern` | 46 | 82 |
| `has_majority_to_sorted_ascending` | 63 | 65 |
| `has_majority_to_sorted_descending` | 64 | 64 |
| `has_majority_to_mountain_pattern` | 33 | 95 |
| `mountain_pattern_to_sorted_ascending` | 64 | 64 |
| `mountain_pattern_to_sorted_descending` | 64 | 64 |
| `mountain_pattern_to_has_majority` | 32 | 96 |

## Controls

All controls must be constructed from V13 inputs only. “V12 aligned” means the
V12-style layerwise Hungarian hidden-unit alignment and interpolation logic
reimplemented/reused as code, recomputed from V13 train subjects and the current
V13 eval source record. It must not read any V12 raw pool, V12 final artifact, or
V12 selected subject record.

Every source-target record must output the matched optimizer edit and evaluate
these controls:

- no edit source weights;
- V12 aligned full nearest-target retrieval;
- V12 aligned interpolation at `alpha = 0.975`;
- no-signature optimizer: identical optimizer, target support BCE weight `4.0`,
  source support conflict BCE weight `2.0`, compatible source support logit MSE
  weight `0.01`, and source-weight L2 MSE weight `0.0005`, but signature loss
  weight fixed to `0.0`;
- source-signature optimizer: identical optimizer and loss weights to the
  matched editor, but signature target is the original source signature
  normalized with V13 train signature mean/std;
- shuffled-signature optimizer: identical optimizer and loss weights to the
  matched editor, but signature target is the selected target-attractor signature
  for the
  deterministic shuffled target selected from the two behaviors neither source
  nor requested target;
- signature-only optimizer: selected target-attractor signature loss weight
  `0.01` plus source-weight L2 MSE weight `0.0005`, with all support-set loss
  weights fixed to `0.0`;
- target-only-support optimizer: target support BCE weight `4.0` plus
  source-weight L2 MSE weight `0.0005`, with conflict, compatible, and signature
  loss weights fixed to `0.0`;
- `16` deterministic random norm-matched weight deltas around the source weights,
  scaled to the matched optimizer raw delta norm.

The control count is exactly `24`: `8` non-random controls plus `16`
random norm-matched controls. The matched optimizer edit is the candidate being
tested and is not counted as a control.

Shuffled target selection is deterministic:

1. Enumerate behaviors in lexicographic order.
2. Keep behaviors not equal to the source behavior and not equal to the matched
   requested target behavior.
3. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v13_shuffled_signature_target"])`.
4. Convert the first 16 hex characters to an integer and take modulo the
   remaining behavior count.
5. Record the selected shuffled target behavior in the control metrics.

Random norm-matched controls are deterministic:

1. Work in raw flat-weight space with the same dtype and shape as
   `source_weights`.
2. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v13_random_weight_delta"])`.
3. Seed `torch.Generator` with the first 16 hex characters modulo `2**31`.
4. For control indices `0..15`, draw sequential standard-normal vectors with
   `torch.randn(source_weights.shape, generator=generator, dtype=source_weights.dtype)`.
5. Normalize each vector by its L2 norm clamped below at `1e-12`.
6. Scale every vector to the matched editor raw delta norm
   `((edited_weights - source_weights).norm())`, clamped below at `1e-12`.
7. Add the scaled delta to `source_weights`.
8. Record the seed, index, and delta norm for every random control.

All controls are included in best-control target-margin aggregates and
conflict-aware Pareto checks. A control Pareto-dominates the matched edit only if
it is weakly better on target heldout margin and compatible source-output MSE,
with at least one strict improvement.

## Individual Record Pass

A record passes only if the matched edited weights satisfy all gates:

- primary behavior prediction is the requested target behavior;
- target heldout margin is `> 0.20`;
- compatible source-output MSE is lower than V12 aligned full nearest-target
  retrieval;
- conflict target-label accuracy is `>= 0.70`;
- conflict target-label accuracy improves over the original source model by
  `>= 0.20`;
- no control Pareto-dominates the matched edit on
  `(target heldout margin, -compatible source-output MSE)`;
- matched target margin is at least `0.02` greater than the no-signature
  optimizer's target margin or matched compatible source-output MSE is at least
  `5.0` lower than the no-signature optimizer's compatible source-output MSE;
- matched target margin is at least `0.05` greater than the shuffled-signature
  optimizer's target margin or matched compatible source-output MSE is at least
  `5.0` lower than the shuffled-signature optimizer's compatible source-output
  MSE.

The last two gates are anti-overclaim controls. If no-signature or
shuffled-signature optimization performs as well as the matched signature
optimizer, V13 does not support a fixed-probe-signature-specific editing claim.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- every ordered source-target direction has `n == 24`;
- aggregate individual pass rate `>= 0.85`;
- every ordered source-target direction has individual pass rate `>= 0.70`;
- every ordered source-target direction has negative-control Pareto-undominated
  record rate `>= 0.70`;
- every ordered source-target direction has target-behavior prediction rate
  `>= 0.90`;
- every ordered source-target direction has mean target heldout margin `> 0.20`;
- every ordered source-target direction has mean conflict target-label accuracy
  `>= 0.70`;
- every ordered source-target direction has mean conflict target-label accuracy
  improvement over source `>= 0.20`;
- every ordered source-target direction has mean aligned-full-minus-matched
  compatible source-output MSE `> 0.0`;
- aggregate target-behavior prediction rate `>= 0.95`;
- aggregate mean matched target heldout margin `> 0.50`;
- aggregate mean conflict target-label accuracy `>= 0.85`;
- aggregate mean conflict target-label accuracy improvement `>= 0.50`;
- aggregate mean aligned-full-minus-matched compatible source-output MSE
  `> 10.0`;
- aggregate negative-control Pareto-undominated record rate `>= 0.85`;
- aggregate matched-minus-no-signature target margin mean `> 0.0` or aggregate
  no-signature-minus-matched compatible source-output MSE mean `> 2.0`;
- aggregate matched-minus-shuffled-signature target margin mean `> 0.05` or
  aggregate shuffled-signature-minus-matched compatible source-output MSE mean
  `> 2.0`;
- every record includes exactly `24` controls;
- every record includes exactly `16` random norm-matched controls;
- every record includes the no-signature, source-signature, shuffled-signature,
  signature-only, and target-only-support optimizer controls.

If any development gate fails, the result must be recorded as a negative
development result and final evaluation is not authorized.

## Final Gates

The final evaluation is one-shot and uses the exact reviewed method and
authorization artifact from development. It must pass the same gates as
development, with no threshold weakening after seeing final results.

Final output must include:

- all aggregate gates;
- per-direction target-prediction, target-margin, pass, compatible-preservation,
  conflict-relabeling, signature-control advantage, and Pareto summaries;
- per-record matched metrics;
- all control metrics;
- source-pool audit hashes;
- implementation and preregistration hashes;
- explicit `limitations` text saying the result is small-subject,
  source-label-known, target-label-requested, hybrid signature-support
  optimization evidence only.

If final fails, it is recorded as a failed final result and no additional V13
final rerun may be used as proof without a new preregistered experiment.

## Reviewer Checkpoints

V13 requires reviewer confidence `5/5` after each result-producing step before
proceeding:

1. preregistration review;
2. implementation and helper-test review;
3. source-pool audit review;
4. development-result review;
5. final-result review if final is authorized.

Reviewer prompts must ask specifically about data leakage, final-pool exposure,
support-vs-heldout separation, no-signature and shuffled-signature controls,
source-label-known scope, target-label-requested scope, metric/gate mismatch,
hidden adaptive choice, and whether the claim is narrower than the evidence.
