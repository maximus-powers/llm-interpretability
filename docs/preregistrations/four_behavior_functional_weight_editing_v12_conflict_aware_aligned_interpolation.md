# Four-Behavior Functional Weight Editing V12 Conflict-Aware Aligned Interpolation Preregistration

Date: 2026-06-12

## Purpose

V11 produced a valid negative development result: target transfer was strong, but
the proof-grade source-similarity/Pareto gates failed, especially
`sorted_descending_to_has_majority`. A development-only diagnostic showed that
the V11 source-preservation metric was semantically overbroad: for these four
predicate behaviors, source heldout positives are target negatives for every
ordered source-target pair. Preserving all source outputs therefore penalizes
correct target editing on logically conflicting examples.

V12 tests a narrower and better-specified claim on fresh pools:

For the same four clean behaviors and subject architecture, a deterministic
source-label-known editor can use fixed stored-probe signatures to retrieve a
train target model, align hidden-unit permutations, and produce edited heldout
source weights that execute the requested target behavior while preserving source
outputs only on source heldout cases whose source and target predicate labels
agree. On source heldout cases whose labels conflict, the edited model must move
toward the requested target labels rather than preserve the source outputs.

This is a narrow small-subject functional editing experiment. It is not
source-label inference, not source-free decoding, not larger-model evidence, not
broad MUAT proof, and not evidence that unrelated capabilities are preserved.

## Prior Development Inputs

V12 is motivated by:

- positive V9 representation-space source-invariant target-attractor final
  evidence;
- negative V10 functional ridge-edit development evidence;
- negative V11 fixed retrieval-interpolation development evidence;
- V11 development-only conflict analysis showing source-output preservation on
  all source heldout examples is not a valid preservation target when predicates
  disagree.

These prior results may motivate V12 design but are not evidence for the V12
claim. V12 must use fresh V12 pools for proof use.

## Contamination Policy

V1-V11 preregistrations, development artifacts, final summaries, and evidence
reports have been inspected. V12 must not reuse any prior final raw pool as a
V12 train or development input. V12 must generate fresh train, development, and
final source pools with V12-specific claim scopes and seeds.

V12 development may read:

- V12 train raw pool;
- V12 development raw pool;
- V12 combined source-pool audit;
- V12 final redacted source-pool audit.

V12 development must not read, parse, summarize, hash through loaded content, or
evaluate the V12 final raw pool. The final raw path may appear only as a literal
blocked path in guard code and documentation before final authorization.

The V12 final raw pool remains sealed unless:

1. this preregistration is accepted by reviewer at `5/5`;
2. V12 implementation is accepted by reviewer at `5/5`;
3. V12 source-pool construction is accepted by reviewer at `5/5`;
4. V12 development evaluation passes all gates below;
5. reviewer accepts the V12 development result at `5/5`.

If V12 development fails, final evaluation is blocked. Any method change after
reviewed development success requires a new preregistration suffix and
invalidates final eligibility for this preregistration.

## Source Pools

V12 uses the same subject architecture, stored probes, behavior suite, and
source-generation acceptance criteria as V9-V11.

V12 source-pool output directory:

`runs/four_behavior_functional_weight_editing_v12_pools`

Required V12 claim scopes:

- raw train/development/final pools:
  `four_behavior_functional_weight_editing_v12_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v12_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v12_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `66300000`;
- development base seed: `67300000`;
- final base seed: `68300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

Before development, V12 must validate V12 scopes, accepted counts, zero
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

Allowed `combined_audit.pool_summaries.final` fields before final evaluation:

- `accepted_counts_by_behavior`;
- `pool_file_sha256`;
- `pool_redacted_payload_sha256`.

Forbidden final-detail fields before final evaluation are the same as V11:
per-subject records, subject IDs, behavior labels, seeds, attempt indices,
signatures, signature hashes, weights, weight hashes, source/support/heldout
margins, attempt/rejection counts, accepted/rejected subject IDs, and per-subject
metrics. Any exposure of forbidden final raw or final-detail fields before final
evaluation invalidates the V12 final pool for proof use.

Before opening the V12 final raw pool, the final-evaluation command must
validate a current passing V12 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_functional_weight_editing_v12_conflict_aware_aligned_interpolation_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `editor_method`:
  `v9_selected_target_retrieval_layerwise_hungarian_aligned_raw_weight_interpolation_alpha_0_975_conflict_aware`;
- `train_pool_sha256`: current V12 train raw pool SHA-256;
- `eval_pool_sha256`: current V12 development raw pool SHA-256;
- `combined_audit_sha256`: current V12 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V12 final-redacted-audit SHA-256;
- `implementation_sha256`: current V12 experiment script SHA-256;
- `preregistration_sha256`: current V12 preregistration SHA-256.

If any value differs, final evaluation must fail before opening the V12 final
raw pool.

## Train-Only Representation Statistics

V12 recomputes the V9 target-attractor representation machinery from accepted
V12 train subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only inside the frozen V9-style selected
  target-attractor candidate generator.

No V1-V11 raw final subject may be used as a V12 train, development, or retrieval
candidate.

Fixed representation configuration is identical to V9-V11.

## Editor Method

V12 is deterministic and has no fitted weight-delta regressor.

At evaluation time, the editor receives exactly:

- heldout source weights;
- heldout source stored-probe signature;
- registered source behavior label;
- requested target behavior label.

For each source-target record:

1. Normalize the source signature with V12 train signature mean/std.
2. Compute the V9-style selected target-attractor representation candidate for
   `(source_behavior, target_behavior)` using V12 train-only statistics.
3. Retrieve the nearest accepted V12 train subject with the requested target
   behavior by squared Euclidean distance between the selected target-attractor
   candidate and each target train subject's train-normalized stored-probe
   signature.
4. Break retrieval ties by `(distance, weights_hash, signature_hash,
   subject_id)`.
5. Align the retrieved target model's hidden-unit permutations to the source
   model using deterministic layer-wise Hungarian matching:
   - unpack flat weights into five hidden layers of width `8` and one scalar
     output layer;
   - for hidden layer `l`, after applying all previous layer alignments, build
     each hidden neuron's feature vector from its incoming weights and bias;
   - compute pairwise squared Euclidean costs from source hidden neurons to
     retrieved-target hidden neurons;
   - add deterministic assignment-dependent tie-break cost
     `1e-9 * (source_neuron_index + 1) * (target_neuron_index + 1)` to each
     pairwise cost; this perturbation is used only for deterministic tie
     resolution and is small relative to real squared-distance costs;
   - solve the assignment with `linear_sum_assignment`;
   - reorder the target hidden layer rows and biases to source order;
   - apply the same permutation to the outgoing columns of the next layer,
     preserving the retrieved target function exactly up to floating-point
     ordering.
6. Let `alpha = 0.975`.
7. Produce edited flat weights:

   `edited_weights = (1 - alpha) * source_weights + alpha * aligned_target_weights`

The interpolation is in raw flat-weight space, not normalized weight space. No
gradient optimization, behavioral fine-tuning, or development/final label
adaptation is allowed.

## Conflict-Aware Evaluation

Each accepted evaluation source subject is tested against every target behavior
different from its source behavior, for `96 * 3 = 288` source-target records per
phase.

For target behavior, V12 evaluates the actual subject forward pass on the
standard target heldout positive/negative cases:

- target behavior heldout margin;
- behavioral prediction under all four behavior evaluators.

For source preservation, V12 partitions the source behavior's heldout cases by
the source and requested target predicates:

- compatible source cases: source label equals target label;
- conflict source cases: source label differs from target label.

Predicate labels are computed directly with the fixed `PREDICATES` functions
from `hypernet.behavior_suite` on the source behavior's heldout positive and
negative sequences. The source label is the source predicate output, not the
heldout positive/negative list membership inferred indirectly. Model labels for
conflict accuracy are computed as `sigmoid(logit) >= 0.5`.

Compatible source-output MSE is mean squared error between edited logits and
original source logits on compatible source cases. Conflict target-label accuracy
is the fraction of conflict source cases whose edited model label equals the
target predicate label. Conflict target-label accuracy improvement is edited
conflict accuracy minus original source model conflict accuracy.

The compatible/conflict split has fixed expected counts for this suite:

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

Every record must report the compatible and conflict counts for its direction.
If any direction has a count different from this table, development/final fails.

V12 evaluates:

- output MSE to the original source subject on compatible source cases;
- compatible-source-output-MSE improvement over full aligned target retrieval;
- target-label accuracy on conflict source cases;
- conflict target-label accuracy improvement over the original source model;
- compatible and conflict case counts per direction.

The compatibility split is fixed by the behavior predicates and heldout suite
before any model result is inspected. It is not chosen per record.

## Controls

Every source-target record must evaluate these controls:

- no edit source weights;
- unaligned full nearest-target retrieval weights;
- aligned full nearest-target retrieval weights;
- aligned interpolation controls with fixed alphas `0.75`, `0.85`, `0.90`,
  `0.95`, and `1.00` using the same aligned retrieved target subject;
- unaligned interpolation at `alpha = 0.975`;
- same-source other-target aligned interpolation at `alpha = 0.975` for the two
  non-matched target behaviors;
- shuffled target aligned interpolation at `alpha = 0.975`, where the shuffled
  target is selected from the two behaviors neither source nor matched target by
  `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v12_shuffled_target"])`;
- train target-behavior centroid weights, computed in raw weight space as the
  mean of accepted train weights for the requested target behavior;
- train global centroid weights, computed in raw weight space as the mean of all
  accepted train weights;
- `32` random norm-matched weight deltas around the source weights, scaled to
  the V12 edited raw delta norm.

Shuffled target selection is deterministic:

1. Enumerate behaviors in lexicographic order.
2. Keep behaviors not equal to the source behavior and not equal to the matched
   requested target behavior.
3. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v12_shuffled_target"])`.
4. Convert the first 16 hex characters to an integer and take modulo the
   remaining behavior count.
5. Record the selected shuffled target behavior in the control metrics.

Random norm-matched controls are deterministic:

1. Work in raw flat-weight space with the same dtype and shape as
   `source_weights`.
2. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "functional_weight_editing_v12_random_weight_delta"])`.
3. Seed `torch.Generator` with the first 16 hex characters modulo `2**31`.
4. For control indices `0..31`, draw sequential standard-normal vectors with
   `torch.randn(source_weights.shape, generator=generator, dtype=source_weights.dtype)`.
5. Normalize each vector by its L2 norm clamped below at `1e-12`.
6. Scale every vector to the V12 edited raw delta norm
   `((edited_weights - source_weights).norm())`, clamped below at `1e-12`.
7. Add the scaled delta to `source_weights`.
8. Record the seed, index, and delta norm for every random control.

All controls are included in best-control target-margin aggregates and
conflict-aware Pareto-dominance checks. A control Pareto-dominates the matched
edit only if it is weakly better on target heldout margin and weakly better on
compatible source-output MSE, with at least one strict improvement.

## Individual Record Pass

A record passes only if the matched edited weights satisfy all gates:

- primary behavior prediction is the requested target behavior;
- target heldout margin is `> 0.20`;
- compatible source-output MSE is lower than the aligned full nearest-target
  retrieval control;
- conflict target-label accuracy is `>= 0.70`;
- conflict target-label accuracy improves over the original source model by
  `>= 0.20`;
- no negative control Pareto-dominates the matched edit on the pair:
  `(target heldout margin, -compatible source-output MSE)`.

This individual pass separates preservation from relabeling: compatible source
cases should be preserved, while conflict source cases should change.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- every ordered source-target direction has `n == 24`;
- aggregate individual pass rate `>= 0.85`;
- every ordered source-target direction has individual pass rate `>= 0.70`;
- every ordered source-target direction has negative-control Pareto-undominated
  record rate `>= 0.70`;
- every ordered source-target direction has mean aligned-full-minus-matched
  compatible source-output MSE `> 0.0`;
- every ordered source-target direction has target-behavior prediction rate
  `>= 0.90`;
- every ordered source-target direction has mean target heldout margin `> 0.20`;
- every ordered source-target direction has mean conflict target-label accuracy
  `>= 0.70`;
- every ordered source-target direction has mean conflict target-label accuracy
  improvement over source `>= 0.20`;
- aggregate target-behavior prediction rate `>= 0.95`;
- aggregate mean matched target heldout margin `> 0.50`;
- aggregate mean aligned-full-minus-matched compatible source-output MSE
  `> 10.0`;
- aggregate negative-control Pareto-undominated record rate `>= 0.85`;
- every record includes exactly `46` controls;
- every record includes the aligned full nearest-target retrieval control;
- every record includes the unaligned full nearest-target retrieval control;
- random norm-matched controls include exactly `32` controls per record.

If any development gate fails, the result must be recorded as a negative
development result and final evaluation is not authorized.

## Final Gates

The final evaluation is one-shot and uses the exact reviewed method and
authorization artifact from development. It must pass the same gates as
development, with no threshold weakening after seeing final results.

Final output must include:

- all aggregate gates;
- per-direction target-prediction, target-margin, pass, compatible-preservation,
  conflict-relabeling, and Pareto summaries;
- per-record matched metrics;
- all control metrics;
- source-pool audit hashes;
- implementation and preregistration hashes;
- explicit `limitations` text saying the result is small-subject,
  source-label-known, retrieval-anchored, conflict-aware aligned interpolation
  evidence only.

If final fails, it is recorded as a failed final result and no additional V12
final rerun may be used as proof without a new preregistered experiment.

## Reviewer Checkpoints

V12 requires reviewer confidence `5/5` after each result-producing step before
proceeding:

1. preregistration review;
2. implementation and helper-test review;
3. source-pool audit review;
4. development-result review;
5. final-result review if final is authorized.

Reviewer prompts must ask specifically about data leakage, final-pool exposure,
control adequacy, source-label-known scope, retrieval-vs-edit overclaim,
metric/gate mismatch, hidden adaptive choice, conflict-aware metric validity, and
whether the claim is narrower than the evidence.
