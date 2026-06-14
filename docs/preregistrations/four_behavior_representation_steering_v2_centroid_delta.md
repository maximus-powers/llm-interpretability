# Four-Behavior Representation Steering V2 Centroid-Delta Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a second four-behavior representation-steering
proof attempt for fixed stored-probe activation signatures.

V1 failed under its frozen protocol. The accepted V1 failure diagnosis found
that the train centroid delta was used both as edit-vector initialization and
as a proof-critical centroid-improvement control that learned vectors had to
beat on the same centroid-distance metric. That made the centroid-specificity
gate ill-posed for the V1 method.

V2 tests a narrower and more transparent claim:

Train-only behavior centroids in fixed stored-probe signature space define
behavior-direction vectors that can move fresh heldout source signatures toward
target behavior regions while beating non-matched controls.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model, natural-language-model, or broad MUAT
generality.

## Prior Development Contamination Policy

The V1 development pool was inspected during failure analysis. Therefore V2
must not report a V2 development result on the V1 development pool.

V2 must generate fresh V2 train, development, and final source pools before any
V2 development evaluation. The V2 final raw pool must remain sealed until the
frozen V2 method passes development.

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

## V2 Source Pools

Generate V2-specific source pools with the same source-generation method as
the accepted V1 steering pool construction:

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

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `33300000`;
- development base seed: `34300000`;
- final base seed: `35300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every V2 pool/behavior range and fail if any configured ranges overlap.

Before development evaluation, the combined source-pool audit must show:

- accepted counts meet the configured per-behavior targets for train,
  development, and final;
- accepted train/development/final overlaps are zero for seed, subject id,
  weight hash, and signature hash;
- every accepted subject clears the source heldout margin gate;
- only the final redacted audit surface is used for final-pool metadata.

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
- final signatures;
- final weights;
- final source margins;
- final attempt counts;
- final rejection counts;
- final acceptance rates;
- final accepted attempt indices;
- final rejected attempt indices;
- final per-subject metrics.

Any exposure of forbidden final raw or final-detail fields before final
evaluation invalidates the V2 final pool for proof use.

## Final-Pool Access Policy

Before final evaluation, V2 scripts may read only:

- V2 train raw pool;
- V2 development raw pool;
- V2 combined source-pool audit;
- V2 final redacted audit.

Before final evaluation, V2 scripts must reject:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v2_pools/final_subjects.json`;
- any other path named `final_subjects.json` unless the command is explicitly
  the V2 final-evaluation command.

The V2 final raw pool may be opened only after:

1. this preregistration is accepted by reviewer at `5/5`;
2. V2 source-pool construction is accepted by reviewer at `5/5`;
3. V2 development evaluation passes all gates below;
4. reviewer accepts the V2 development result at `5/5`.

If V2 development fails, final evaluation is blocked and the final raw pool
must remain sealed.

## Train-Only Statistics

Compute all representation statistics from accepted V2 train subjects only:

- signature mean and standard deviation;
- behavior centroids in normalized signature space;
- representation-level primary behavior evaluator.

Clamp standard deviations below `1e-6` to `1.0`.

Development and final signatures must be normalized with these train-only
statistics.

## Primary Representation Evaluator

Fit a single affine classifier from normalized stored-probe signatures to the
four behavior labels using accepted V2 train subjects only.

Frozen config:

- model: one affine layer from dimension `560` to `4`;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260630`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only.

No development or final data may select, tune, calibrate, or refit this
evaluator.

The centroid evaluator is a mandatory secondary evaluator. Nearest-centroid
prediction uses train-only behavior centroids in normalized signature space.

## V2 Edit Vectors

V2 has no learned edit-vector optimizer.

For every ordered source-target pair where source `s` differs from target `t`,
the V2 edit vector is exactly:

`v[s -> t] = centroid[t] - centroid[s]`

where centroids are computed only from accepted V2 train subjects after
train-only normalization.

There are `4 * 3 = 12` fixed vectors. Development data may not change vector
norms, directions, thresholds, controls, or evaluator settings.

This makes centroid delta the method under test. It is not included as an
adversarial control that must be beaten on the centroid-distance metric it is
directly designed to improve.

## Controls

For each development/final source subject and target behavior, evaluate the
matched V2 vector against these controls:

- no edit;
- null vector;
- reverse centroid delta: `centroid[source] - centroid[target]`;
- same-source other-target centroid deltas:
  `centroid[other_target] - centroid[source]`;
- same-target other-source centroid deltas:
  `centroid[target] - centroid[other_source]`;
- deterministic shuffled-direction centroid delta, selected from ordered
  source-target pairs excluding the matched pair and reverse pair;
- `32` deterministic random norm-matched vectors with the same L2 norm as the
  matched V2 vector.

Random and shuffled controls must be deterministic functions of:

- subject id;
- source behavior;
- target behavior;
- control index where applicable;
- fixed salt `representation_steering_v2_centroid_delta`.

All controls use train-only statistics. No control may use development or final
labels except the evaluation label of the current source/target pair.

## Metrics

For every source subject and ordered target behavior, report:

- primary target margin:
  target logit minus strongest non-target logit after editing;
- primary source margin change:
  edited source-vs-non-source margin minus no-edit source-vs-non-source margin;
- centroid target-distance improvement:
  no-edit target-centroid distance minus edited target-centroid distance;
- primary predicted behavior;
- nearest-centroid predicted behavior;
- matched-minus-best-control primary target margin;
- matched-minus-best-control centroid improvement.

For controls, report the same primary margin, source-margin change, centroid
improvement, and predicted behaviors, plus matched-minus-control differences.

## Development And Final Gates

Expected evaluation count:

- `4` source behaviors;
- `3` non-source targets per source;
- `24` accepted subjects per source behavior;
- total `288` individual evaluations.

Per-record matched gates:

- primary predicted behavior must equal target;
- nearest-centroid predicted behavior must equal target;
- primary target margin must be `> 0.10`;
- centroid target-distance improvement must be `> 0.0`;
- primary source margin change must be `< 0.0`;
- matched-minus-best-control primary target margin must be `> 0.0`;
- matched-minus-best-control centroid improvement must be `> 0.0`.

Aggregate gates:

- total individual all-gate pass rate must be `>= 0.90`;
- per-target individual all-gate pass rate must be `>= 0.80`;
- per-direction individual all-gate pass rate must be `>= 0.90`;
- mean primary target margin must be `> 0.20`;
- mean centroid target-distance improvement must be `> 0.15`;
- mean matched-minus-best-control primary target margin must be `> 0.10`;
- mean matched-minus-best-control centroid improvement must be `> 0.10`;
- mean primary source margin change must be `< -0.05`.

Development and final use identical gates. Development is a go/no-go check
only. If development fails, do not run final evaluation.

## Pass/Fail Interpretation

If V2 passes development and final:

- evidence supports the narrow claim that fixed stored-probe signature centroids
  encode behavior-direction vectors that steer heldout representations across
  this four-behavior suite under the registered controls;
- evidence does not support functional model editing unless paired with a
  separately locked valid decoder;
- evidence does not prove larger-model, natural-language-model, or broad MUAT
  generality.

If V2 fails development:

- log a negative V2 development result;
- keep the V2 final raw pool sealed;
- do not reinterpret partial metrics as proof of steering.

If V2 passes development but fails final:

- log a failed final result;
- treat the development result as overfit or non-generalizing for this protocol.

## Reviewer Checkpoints

Required reviewer checkpoints, each with confidence `5/5` before the next step:

1. V2 preregistration acceptance.
2. V2 source-pool construction acceptance.
3. V2 development result acceptance.
4. V2 final result acceptance, only if development passes.
