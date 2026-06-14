# Four-Behavior Representation Steering V1 Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a four-behavior representation-steering proof
attempt for fixed stored-probe activation signatures.

The experiment tests whether train-only edit vectors in normalized stored-probe
signature space can move heldout subject signatures toward each of four clean
target behaviors while suppressing the source behavior and beating strong
controls.

This is a representation-steering experiment. It is not a four-behavior
functional-decoder proof. The result may support a functional claim only for
directions where a separately valid locked decoder already exists.

## Prior Evidence And Motivation

Accepted current evidence supports:

- four-behavior stored-probe signature interpretability under heldout
  logistic/RF classifiers and shuffled-label controls;
- restricted two-behavior functional decoding for
  `sorted_ascending <-> sorted_descending`;
- restricted two-behavior robust steering through the locked decoder for
  `sorted_ascending <-> sorted_descending`.

Accepted current evidence does not prove:

- four-behavior functional decoding;
- steering for `has_majority` or `mountain_pattern`;
- larger-model generality;
- broad MUAT generality.

Three four-behavior decoder development attempts failed on train/development
controls:

- V1 direct MLP decoder: `0/96` individual passes;
- V2 functional-distillation decoder: `0/96` individual passes;
- V3 signature-inversion decoder: `0/96` individual passes.

Therefore this experiment targets the more modest but proof-critical claim that
the fixed-probe representation itself can be directionally steered across the
four clean behaviors under heldout representation-level controls.

## Fixed Scope

Subject architecture:

- `SubjectNetwork`;
- sequence length: `5`;
- input base: `10`;
- hidden layers: `5`;
- neurons per hidden layer: `8`;
- flat weight dimension: `345`.

Stored-probe setup:

- probe count: `256`;
- probe seed: `20260610`;
- probe id: `stored_digit_probe_v1_seed_20260610_n256`;
- signature extractor:
  `paired_contrast.extract_signature_with_stored_probes.v1`;
- signature dimension: `560`.

Behavior suite:

- `sorted_ascending`;
- `sorted_descending`;
- `has_majority`;
- `mountain_pattern`.

This experiment does not claim evidence for larger models, natural-language
models, behaviors outside this behavior suite, or mechanistic equivalence beyond
the measured representation-level criteria.

## Steering-Specific Subject Pools

Generate fresh steering-specific source pools instead of reusing the sealed
four-behavior decoder final pool.

Allowed steering raw pool inputs:

- train pool for edit-vector fitting and train-only statistics;
- development pool for go/no-go evaluation of the frozen config below;
- final pool for one-shot representation-steering evaluation only.

The existing sealed decoder final raw pool
`runs/four_behavior_decoder_source_pools_v2/final_subjects.json` must not be
opened, passed as an argument, copied, summarized, or used by any V1 steering
script.

Generate steering pools with the same source-generation method as the accepted
four-behavior decoder source-pool V2 construction:

- training mode: `heldout_excluded_full_pool`;
- source heldout margin gate: `>= 0.40`;
- support cases per class: `160`;
- heldout cases per class: `64`;
- positive cap: `2048`;
- hard-negative cap: `1024`;
- generic-negative cap: `1024`;
- train epochs: `350`;
- learning rate: `0.003`;
- stored probe count: `256`;
- stored probe seed: `20260610`.

Steering pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Steering seed schedule:

- train base seed: `30300000`;
- development base seed: `31300000`;
- final base seed: `32300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every pool/behavior range and fail if any configured ranges overlap.

## Final-Pool Access Policy

The steering final pool is generated for this representation-steering proof
only. It is separate from the decoder final pool.

Before final steering evaluation:

- training/development scripts may read train and development raw pools;
- training/development scripts may inspect only the steering final redacted
  audit;
- training/development scripts must reject any raw final-pool path;
- training/development scripts must reject the sealed decoder final raw path.

The final steering raw pool may be opened only by the final steering evaluation
command after all edit-vector hyperparameters, thresholds, controls, and
selection rules are frozen by this document and after the frozen method passes
development go/no-go evaluation.

Any accidental use of the steering final raw contents for method selection
invalidates the final steering proof.

Allowed steering final redacted audit fields before final evaluation:

- pass/fail status;
- accepted counts by behavior;
- selected-training-vs-heldout overlap pass/fail or max overlap count;
- cross-pool overlap counts;
- file hash;
- redacted-payload hash;
- stored-probe hash;
- behavior-suite hashes;
- source-generation config hash;
- seed-range preflight pass/fail and configured seed ranges.

Before final evaluation, steering public audit artifacts must not expose final
per-subject records, subject IDs, labels, signatures, weights, source margins,
attempt counts, rejection counts, accepted attempt indices, rejected attempt
indices, acceptance rates, or per-subject metrics.

## Train-Only Statistics

Compute all representation statistics from accepted train subjects only:

- signature mean and standard deviation;
- behavior centroids in normalized signature space;
- train classifier for representation-level behavior prediction.

Clamp standard deviations below `1e-6` to `1.0`.

Development and final signatures must be normalized with these train-only
statistics.

## Representation Evaluators

Fit representation-level behavior evaluators on accepted train subjects only:

1. nearest centroid in normalized signature space;
2. multinomial logistic regression, or an equivalent PyTorch linear classifier
   trained only on normalized train signatures.

Frozen primary linear evaluator config:

- model: single affine layer from normalized signature dimension `560` to `4`
  behavior logits;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260621`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only;
- development/final data must not select, tune, calibrate, or refit this
  evaluator.

The final artifact must report both evaluator families. The logistic/linear
classifier is the primary evaluator. The centroid evaluator is a mandatory
robustness check.

Development and final true behavior labels may be used only for evaluation, not
for edit-vector fitting except as source/target labels on train subjects.

## Edit-Vector Training

Learn one normalized-signature edit vector for every ordered source-target
behavior pair where source and target differ.

There are `4 * 3 = 12` directions.

For a train subject with source behavior `s` and target behavior `t`, the
steered normalized signature is:

`z_steered = z_source + v[s -> t]`

The training objective may use only train subjects, train behavior labels, and
train-only representation evaluators/statistics.

Initialize every edit vector as the train-centroid delta for its direction:

`v[s -> t] = centroid[t] - centroid[s]`

This initialization is train-only and must be recorded in the result artifact.

The objective must encourage:

- primary evaluator target logit margin above the strongest non-target logit;
- centroid target-distance improvement relative to no edit;
- source behavior logit suppression;
- improvement over reverse-direction and random norm-matched controls during
  train-time evaluation;
- bounded vector norm through an L2 penalty.

Frozen optimizer and training config:

- optimizer: `AdamW`;
- learning rate: `0.05`;
- weight decay: `0.0`;
- epochs: `500`;
- deterministic seed: `20260620`;
- gradient clipping norm: `10.0`;
- train random controls per direction per epoch: `8`;
- no dropout, minibatching, stochastic data sampling, or early stopping.

Frozen objective terms and weights:

- primary target-margin hinge:
  `relu(0.35 - primary_target_margin)`, weight `2.0`;
- primary target-improvement hinge over no edit:
  `relu(0.25 - (matched_target_margin - no_edit_target_margin))`,
  weight `1.0`;
- primary source-suppression hinge:
  `relu(source_margin_change + 0.10)`, weight `1.0`;
- centroid target-distance improvement hinge:
  `relu(0.25 - centroid_target_distance_improvement)`, weight `1.0`;
- random-control target-margin hinge against worst train random control:
  `relu(0.20 - (matched_target_margin - worst_random_target_margin))`,
  weight `1.0`;
- vector L2 penalty: `mean(vector ** 2)`, weight `0.0001`.

Checkpoint selection:

- train all `500` epochs;
- after every epoch evaluate all train records under the train-time objective;
- select the epoch with the lowest total train objective;
- ties choose the earliest epoch;
- development metrics must not choose the checkpoint.

The implementation must report vector norms for all 12 directions and the
selected epoch.

## Development Evaluation

Development evaluation uses accepted development subjects only and may be used
only as a go/no-go diagnostic for the fixed method above. The final gates in this
preregistration are frozen. Thresholds, controls, objective family, objective
weights, optimizer settings, checkpoint-selection rules, and reporting fields
must not be changed after viewing development results unless a new
preregistration is written and accepted before final evaluation.

For each development subject and each non-source target behavior, evaluate:

- no edit;
- matched edit vector;
- reverse edit vector where available;
- null vector, defined exactly as the zero vector added to the normalized source
  signature;
- source-to-each-other-target edit vector;
- target-source centroid delta control, defined exactly as
  `centroid[target] - centroid[source]` in train-normalized signature space;
- worst-of-32 norm-matched random vectors.

For every matched/control representation, report:

- primary evaluator predicted behavior;
- primary evaluator target margin;
- centroid predicted behavior;
- centroid target-distance improvement relative to no edit;
- source-margin change relative to no edit;
- matched-minus-control target margin for each control;
- matched-minus-control centroid improvement for each control.

The best-control target-margin gate uses the control with the maximum target
margin. The best-control centroid gate uses the control with the maximum centroid
target-distance improvement. These controls may differ and both identities must
be reported.

Development may diagnose failures, but a development pass does not authorize
final evaluation if any implementation detail differs from this preregistration.
If development motivates any method, threshold, control, or selection change,
that changed method requires a new preregistration before final-pool access.

Development go/no-go gates are identical to the final proof gates, except that
the development pool replaces the final pool in all count checks. If development
does not pass those gates, the V1 method must be logged as a negative
development result and must not open the steering final raw pool.

## Final Evaluation Controls

The final evaluation uses accepted final steering subjects exactly once.

For each final subject and each non-source target behavior, compare matched
steering against:

- no edit;
- reverse edit vector where available;
- null vector, defined exactly as the zero vector added to the normalized source
  signature;
- all other non-matched learned edit vectors from the same source;
- target-source centroid delta control, defined exactly as
  `centroid[target] - centroid[source]` in train-normalized signature space;
- worst-of-32 norm-matched random vectors;
- shuffled-target edit vector selected deterministically from a different source
  and different target.

Random controls must be sampled with deterministic per-record seeds derived from:

`stable_hash_json([subject_id, source_behavior, target_behavior, "representation_steering_v1_random", 20260619])`

The final artifact must retain all 32 random-control summaries per record, not
only the adversarial maximum.

The shuffled-target edit vector selection rule is:

1. Build candidate edit-vector keys `(candidate_source, candidate_target)` where
   `candidate_source != source_behavior`, `candidate_target != target_behavior`,
   and `candidate_source != candidate_target`.
2. Sort candidate keys lexicographically by
   `(candidate_source, candidate_target)`.
3. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior, "representation_steering_v1_shuffled_vector", 20260619])`.
4. Interpret the first 16 hex characters as an integer and take modulo the
   candidate count.
5. Select that candidate vector.

This selection must not depend on evaluator scores, margins, distances, source
margins, or post-hoc inspection.

## Final Proof Gates

Let a final record mean one ordered `(subject, source_behavior, target_behavior)`
evaluation. With `24` final subjects per behavior and `3` targets per source,
the final expected record count is `288`.

The final result passes only if every gate below passes.

Aggregate gates:

- final record count is `288`;
- mean matched primary target margin is at least `0.20`;
- mean matched-minus-best-control primary target margin is at least `0.15`;
- mean matched centroid target-distance improvement is at least `0.15`;
- mean matched-minus-best-control centroid improvement is at least `0.10`;
- mean source primary-margin change is at most `-0.05`;
- individual all-gate pass rate is at least `0.90`.

Per-target gates:

- each target behavior has `72` records;
- mean matched primary target margin is at least `0.15`;
- mean matched-minus-best-control primary target margin is at least `0.10`;
- mean matched centroid target-distance improvement is at least `0.10`;
- individual all-gate pass rate is at least `0.80`.

Per-source-target direction gates:

- each ordered source-target direction has `24` records;
- mean matched primary target margin is at least `0.10`;
- mean matched-minus-best-control primary target margin is greater than `0.00`;
- individual all-gate pass rate is at least `0.90`.

Per-record individual gates:

- matched primary target margin is at least `0.10`;
- matched-minus-best-control primary target margin is greater than `0.00`;
- matched centroid target-distance improvement is greater than `0.00`;
- matched-minus-best-control centroid improvement is greater than `0.00`;
- source primary-margin change is less than `0.00`.

Passing aggregate metrics cannot override a failed per-target or per-direction
gate.

## Functional Interpretation

This experiment may include an auxiliary downstream functional evaluation only
for source-target directions covered by an already accepted locked decoder.

For the current evidence package, that means only:

- `sorted_ascending -> sorted_descending`;
- `sorted_descending -> sorted_ascending`.

Auxiliary functional metrics for those two directions must be reported
separately from the four-behavior representation-steering proof. They must not
be used to claim four-behavior functional decoding.

No functional claim may be made for `has_majority` or `mountain_pattern` unless
a separate locked decoder for those behaviors has already passed its own final
proof gates.

## Leakage Audit

The final result must report:

- train/development/final steering subject counts by behavior;
- steering pool file hashes;
- redacted audit hashes;
- accepted train/development/final overlap counts for seed, subject id, weight
  hash, and signature hash;
- proof that the decoder final raw path was not opened;
- proof that the steering final raw path was opened only by the final evaluation
  command;
- train-only statistics hashes;
- behavior-suite hashes;
- probe examples hash;
- code version or explicit dirty-worktree caveat;
- full config and threshold dictionary;
- all vector norms.

Any train/development/final overlap in accepted seeds, subject ids, weight
hashes, or signature hashes makes the final proof fail.

Any result artifact that names the decoder final raw pool as an opened path
makes the final proof fail.

## Result Interpretation

Allowed positive claim if all gates pass:

> Under this fixed small-network setup, stored-probe activation signatures can
> be directionally steered across four clean behaviors in representation space:
> train-only edit vectors move fresh final heldout signatures toward target
> behavior regions while suppressing source behavior and beating no-edit,
> reverse, random, centroid, and shuffled-vector controls.

Required limitations even if all gates pass:

- this is representation-space steering, not four-behavior functional decoding;
- no larger-model claim;
- no broad MUAT generality claim;
- no claim beyond these four clean behaviors;
- no mechanistic-equivalence claim;
- functional interpretation remains restricted to directions with separate
  accepted decoder evidence.

If any final gate fails, the result must be logged as a negative or limited
representation-steering result. Any follow-up method change requires a new
development cycle and a new preregistration before another final evaluation.
