# Four-Behavior Representation Steering V3 Diagonal Transport Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a third four-behavior representation-steering
proof attempt for fixed stored-probe activation signatures.

V2 showed strong aggregate target movement from exact train centroid deltas, but
failed proof-grade reliability. The accepted V2 failure diagnosis identified
source-specificity as the main bottleneck: same-target other-source centroid
deltas often beat the matched source-target centroid delta.

V3 tests a narrower source-conditioned claim:

Train-only source-conditioned diagonal affine transports in fixed stored-probe
signature space can move fresh heldout source signatures toward target behavior
regions while beating centroid-delta and non-matched source-conditioned
controls.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model, natural-language-model, or broad MUAT
generality.

## Prior Development Contamination Policy

The V1 and V2 development pools were inspected during failure analysis. V3 must
not report a V3 development result on V1 or V2 development pools.

V3 must generate fresh V3 train, development, and final source pools before any
V3 development evaluation. The V3 final raw pool must remain sealed until the
frozen V3 method passes development.

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

## V3 Source Pools

Generate V3-specific source pools with the same source-generation method as the
accepted V1/V2 steering pool construction:

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

V3 source-pool output directory:

`runs/four_behavior_representation_steering_v3_pools`

Required V3 source-pool artifact claim scopes:

- train/development/final raw pool payloads:
  `four_behavior_representation_steering_v3_source_pool`;
- combined source-pool audit:
  `four_behavior_representation_steering_v3_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v3_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `36300000`;
- development base seed: `37300000`;
- final base seed: `38300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every V3 pool/behavior range and fail if any configured ranges overlap.

Before development evaluation, the combined source-pool audit must show:

- accepted counts meet the configured per-behavior targets for train,
  development, and final;
- accepted train/development/final overlaps are zero for seed, subject id,
  weight hash, and signature hash;
- every accepted subject clears the source heldout margin gate;
- only the final redacted audit surface is used for final-pool metadata.

Before development or final evaluation, V3 scripts must fail unless:

- pool directory is exactly
  `runs/four_behavior_representation_steering_v3_pools`, unless a reviewer
  explicitly accepts a same-contents relocation before generation;
- pool directory is not under
  `runs/four_behavior_representation_steering_v1_pools`;
- pool directory is not under
  `runs/four_behavior_representation_steering_v2_pools`;
- train raw pool payload has V3 source-pool claim scope and pool name `train`;
- development evaluation payload has V3 source-pool claim scope and pool name
  `development`;
- final evaluation payload has V3 source-pool claim scope and pool name
  `final`;
- combined audit has V3 source-pool-construction claim scope;
- final redacted audit has V3 final-redacted claim scope;
- accepted counts and overlap counts match this preregistration.

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
evaluation invalidates the V3 final pool for proof use.

## Final-Pool Access Policy

Before final evaluation, V3 scripts may read only:

- V3 train raw pool;
- V3 development raw pool;
- V3 combined source-pool audit;
- V3 final redacted audit.

Before final evaluation, V3 scripts must reject:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v2_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v3_pools/final_subjects.json`;
- any other path named `final_subjects.json` unless the command is explicitly
  the V3 final-evaluation command.

The V3 final raw pool may be opened only after:

1. this preregistration is accepted by reviewer at `5/5`;
2. V3 source-pool construction is accepted by reviewer at `5/5`;
3. V3 development evaluation passes all gates below;
4. reviewer accepts the V3 development result at `5/5`.

If V3 development fails, final evaluation is blocked and the final raw pool
must remain sealed.

## Train-Only Statistics

Compute all representation statistics from accepted V3 train subjects only:

- global signature mean and standard deviation for normalization;
- behavior centroids in normalized signature space;
- behavior diagonal variances and standard deviations in normalized signature
  space;
- representation-level primary behavior evaluator.

Clamp global normalization standard deviations below `1e-6` to `1.0`.

For behavior diagonal transport statistics:

- compute per-behavior diagonal variance with unbiased variance disabled;
- compute global train diagonal variance across all accepted train signatures;
- use shrinkage variance:
  `var_shrunk[b] = 0.75 * var_behavior[b] + 0.25 * var_global`;
- clamp `var_shrunk[b]` below `1e-4` to `1e-4`;
- use `std_shrunk[b] = sqrt(var_shrunk[b])`.

Development and final signatures must be normalized with train-only global
statistics.

## Primary Representation Evaluator

Fit a single affine classifier from normalized stored-probe signatures to the
four behavior labels using accepted V3 train subjects only.

Frozen config:

- model: one affine layer from dimension `560` to `4`;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260703`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only.

No development or final data may select, tune, calibrate, or refit this
evaluator.

The centroid evaluator is a mandatory secondary evaluator. Nearest-centroid
prediction uses train-only behavior centroids in normalized signature space.

## V3 Source-Conditioned Transport

V3 has no learned edit-vector optimizer.

For every ordered source-target pair where source `s` differs from target `t`,
the V3 matched transport is the diagonal affine map:

`transport[s -> t](z) = centroid[t] + ratio[s -> t] * (z - centroid[s])`

where:

- `z` is the normalized source signature;
- `ratio[s -> t] = std_shrunk[t] / std_shrunk[s]`;
- each ratio element is clipped to `[0.25, 4.0]`;
- centroids and shrunk standard deviations are computed only from accepted V3
  train subjects.

This map exactly sends the train source centroid to the train target centroid,
but also applies a source-conditioned diagonal covariance transport around the
centroid. It is designed to test whether source-conditioned distribution
alignment improves over V2 centroid deltas.

There are `4 * 3 = 12` fixed transports. Development data may not change
ratios, clipping bounds, thresholds, controls, or evaluator settings.

## Controls

For each development/final source subject and target behavior, evaluate the
matched V3 transport against these controls:

- no edit;
- null vector;
- V2 matched centroid delta:
  `z + centroid[target] - centroid[source]`;
- reverse diagonal transport:
  `transport[target -> source](z)`;
- same-source other-target diagonal transports:
  `transport[source -> other_target](z)`;
- same-target other-source diagonal transports:
  `transport[other_source -> target](z)`;
- deterministic shuffled diagonal transport, selected from ordered
  source-target pairs excluding the matched pair and reverse pair;
- `32` deterministic random norm-matched vectors, where the norm is matched to
  the V3 matched displacement `transport[source -> target](z) - z`.

All source-conditioned controls must report:

- control type;
- control source behavior;
- control target behavior;
- control transport key.

Random and shuffled controls must be deterministic functions of:

- subject id;
- source behavior;
- target behavior;
- control index where applicable;
- fixed salt `representation_steering_v3_diagonal_transport`.

Exact shuffled-direction selection:

- candidate directions are all ordered behavior pairs
  `(candidate_source, candidate_target)` where source and target differ;
- exclude the matched pair `(source, target)`;
- exclude the reverse pair `(target, source)`;
- sort candidates lexicographically by `(candidate_source, candidate_target)`;
- compute `digest = stable_hash_json([subject_id, source, target,
  "representation_steering_v3_diagonal_transport_shuffled_direction"])`;
- select candidate index `int(digest[:16], 16) % len(candidates)`;
- apply the selected candidate's diagonal transport to the current source
  subject.

Exact random norm-matched vector generation:

- compute `seed_digest = stable_hash_json([subject_id, source, target,
  "representation_steering_v3_diagonal_transport_random"])`;
- seed a `torch.Generator` with `int(seed_digest[:16], 16) % (2 ** 31)`;
- for each random control index from `0` to `31`, draw one Gaussian vector with
  `torch.randn` using the seeded generator and the normalized-signature shape;
- normalize the Gaussian vector by its L2 norm clamped below at `1e-12`;
- multiply by the matched V3 displacement norm
  `||transport[source -> target](z) - z||`, also clamped below at `1e-12`;
- the random control candidate is `z + random_vector`.

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
- matched-minus-best-control centroid improvement;
- matched-minus-V2-centroid-delta primary target margin;
- matched-minus-V2-centroid-delta centroid improvement.

For controls, report the same primary margin, source-margin change, centroid
improvement, predicted behaviors, and matched-minus-control differences.

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
- matched-minus-best-control centroid improvement must be `> 0.0`;
- matched-minus-V2-centroid-delta primary target margin must be `> 0.0`;
- matched-minus-V2-centroid-delta centroid improvement must be `> 0.0`.

Aggregate gates:

- total individual all-gate pass rate must be `>= 0.90`;
- per-target individual all-gate pass rate must be `>= 0.80`;
- per-direction individual all-gate pass rate must be `>= 0.90`;
- mean primary target margin must be `> 0.20`;
- mean centroid target-distance improvement must be `> 0.15`;
- mean matched-minus-best-control primary target margin must be `> 0.10`;
- mean matched-minus-best-control centroid improvement must be `> 0.10`;
- mean matched-minus-V2-centroid-delta primary target margin must be `> 0.10`;
- mean matched-minus-V2-centroid-delta centroid improvement must be `> 0.10`;
- mean primary source margin change must be `< -0.05`.

Development and final use identical gates. Development is a go/no-go check
only. If development fails, do not run final evaluation.

## Pass/Fail Interpretation

If V3 passes development and final:

- evidence supports the narrow claim that source-conditioned train-only
  diagonal transports in fixed stored-probe signature space can steer heldout
  representations across this four-behavior suite under the registered
  controls;
- evidence supports source-conditioned improvement over V2 centroid deltas
  only for this small representation-space setting;
- evidence does not support functional model editing unless paired with a
  separately locked valid decoder;
- evidence does not prove larger-model, natural-language-model, or broad MUAT
  generality.

If V3 fails development:

- log a negative V3 development result;
- keep the V3 final raw pool sealed;
- do not reinterpret partial metrics as proof of steering.

If V3 passes development but fails final:

- log a failed final result;
- treat the development result as overfit or non-generalizing for this protocol.

## Reviewer Checkpoints

Required reviewer checkpoints, each with confidence `5/5` before the next step:

1. V3 preregistration acceptance.
2. V3 source-pool construction acceptance.
3. V3 development result acceptance.
4. V3 final result acceptance, only if development passes.
