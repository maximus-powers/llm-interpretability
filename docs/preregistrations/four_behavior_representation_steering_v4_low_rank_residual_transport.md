# Four-Behavior Representation Steering V4 Low-Rank Residual Transport Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a fourth four-behavior representation-steering
proof attempt for fixed stored-probe activation signatures.

V2 showed that train-only centroid deltas produce strong aggregate target
movement, but failed proof-grade reliability and source-specificity. V3 showed
that train-only diagonal covariance transport can improve over V2 on average,
but still failed the full best-control specificity gates. In V3, best controls
were usually either V2 centroid delta or non-matched diagonal transports.

V4 tests the next narrow source-conditioned claim:

Train-only low-rank residual covariance transport in fixed stored-probe
signature space can steer fresh heldout source signatures toward target behavior
regions while beating V2 centroid deltas, V3 diagonal transports, non-matched
source-conditioned transports, shuffled transports, and random norm-matched
controls.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model, natural-language-model, or broad MUAT
generality.

## Prior Development Contamination Policy

The V1, V2, and V3 development pools were inspected during failure analysis.
V4 must not report a V4 development result on V1, V2, or V3 development pools.

V4 must generate fresh V4 train, development, and final source pools before any
V4 development evaluation. The V4 final raw pool must remain sealed until the
frozen V4 method passes development and reviewer accepts the development
checkpoint.

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

## V4 Source Pools

Generate V4-specific source pools with the same accepted source-generation
method used by V1, V2, and V3 steering pool construction:

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

V4 source-pool output directory:

`runs/four_behavior_representation_steering_v4_pools`

Required V4 source-pool artifact claim scopes:

- train/development/final raw pool payloads:
  `four_behavior_representation_steering_v4_source_pool`;
- combined source-pool audit:
  `four_behavior_representation_steering_v4_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v4_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `39300000`;
- development base seed: `40300000`;
- final base seed: `41300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every V4 pool/behavior range and fail if any configured ranges overlap.

Before development evaluation, the combined source-pool audit must show:

- accepted counts meet the configured per-behavior targets for train,
  development, and final;
- accepted train/development/final overlaps are zero for seed, subject id,
  weight hash, and signature hash;
- every accepted subject clears the source heldout margin gate;
- only the final redacted audit surface is used for final-pool metadata.

Before development or final evaluation, V4 scripts must fail unless:

- pool directory is exactly
  `runs/four_behavior_representation_steering_v4_pools`, unless a reviewer
  explicitly accepts a same-contents relocation before generation;
- pool directory is not under any of:
  - `runs/four_behavior_representation_steering_v1_pools`;
  - `runs/four_behavior_representation_steering_v2_pools`;
  - `runs/four_behavior_representation_steering_v3_pools`;
- train raw pool payload has V4 source-pool claim scope and pool name `train`;
- development raw pool payload has V4 source-pool claim scope and pool name
  `development`;
- combined audit has V4 source-pool-construction claim scope;
- final redacted audit has V4 final-redacted claim scope;
- accepted counts and overlap counts match this preregistration.

Before development evaluation, the script must not open or validate the raw
final payload. Development may validate final-pool metadata only through the V4
combined audit and V4 final redacted audit.

During final evaluation only, after development passes and reviewer accepts the
development result at `5/5`, the script must validate that the final raw payload
has V4 source-pool claim scope and pool name `final` before evaluating it.

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
evaluation invalidates the V4 final pool for proof use.

## Final-Pool Access Policy

Before final evaluation, V4 scripts may read only:

- V4 train raw pool;
- V4 development raw pool;
- V4 combined source-pool audit;
- V4 final redacted audit.

Before final evaluation, V4 scripts must reject:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v2_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v3_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v4_pools/final_subjects.json`;
- any other path named `final_subjects.json` unless the command is explicitly
  the V4 final-evaluation command.

The V4 final raw pool may be opened only after:

1. this preregistration is accepted by reviewer at `5/5`;
2. V4 source-pool construction is accepted by reviewer at `5/5`;
3. V4 development evaluation passes all gates below;
4. reviewer accepts the V4 development result at `5/5`.

If V4 development fails, final evaluation is blocked and the final raw pool
must remain sealed.

## Train-Only Statistics

Compute all representation statistics from accepted V4 train subjects only:

- global signature mean and standard deviation for normalization;
- behavior centroids in normalized signature space;
- global principal-component basis in normalized signature space;
- behavior residual covariances in the principal-component subspace;
- representation-level primary behavior evaluator.

Clamp global normalization standard deviations below `1e-6` to `1.0`.

Development and final signatures must be normalized with train-only global
statistics.

## Primary Representation Evaluator

Fit a single affine classifier from normalized stored-probe signatures to the
four behavior labels using accepted V4 train subjects only.

Frozen config:

- model: one affine layer from dimension `560` to `4`;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260711`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only.

No development or final data may select, tune, calibrate, or refit this
evaluator.

The centroid evaluator is a mandatory secondary evaluator. Nearest-centroid
prediction uses train-only behavior centroids in normalized signature space.

## V4 Low-Rank Residual Transport

V4 has no learned neural optimizer and no development-tuned hyperparameters.

V4 is a closed-form source-conditioned residual transport. It maps the source
signature to the target centroid and transforms the source residual through a
train-only low-rank covariance alignment.

Fixed hyperparameters:

- principal-component rank: `48`;
- covariance shrinkage behavior weight: `0.75`;
- covariance shrinkage global weight: `0.25`;
- covariance eigenvalue floor: `1e-4`;
- orthogonal residual carry weight: `0.0`;
- displacement norm cap: `200.0`.

Train-only principal components:

1. Normalize all accepted V4 train signatures.
2. Compute the global train mean of normalized signatures.
3. Center all accepted train signatures by the global train mean.
4. Compute deterministic SVD with `torch.linalg.svd`.
5. Use the first `48` right singular vectors as columns of matrix `U`.

The script must fail if fewer than `48` principal components are available.

For each behavior `b`:

1. Compute normalized behavior centroid `c[b]`.
2. Compute residuals `r_i = z_i - c[b]` for accepted train subjects with
   behavior `b`.
3. Project residuals into the principal-component subspace:
   `p_i = U.T @ r_i`.
4. Compute behavior covariance with unbiased variance disabled:
   `C_behavior[b] = mean_i(p_i p_i.T)`.

Compute global residual covariance `C_global` from all accepted train residuals
projected into the same subspace, where each residual is relative to its own
behavior centroid.

Use shrinkage covariance:

`C_shrunk[b] = 0.75 * C_behavior[b] + 0.25 * C_global`

Then eigendecompose each `C_shrunk[b]`, clamp eigenvalues below `1e-4` to
`1e-4`, and compute:

- `sqrt_cov[b]`;
- `inv_sqrt_cov[b]`.

For every ordered source-target pair where source `s` differs from target `t`,
the V4 matched transport is:

`transport_v4[s -> t](z) = c[t] + U @ sqrt_cov[t] @ inv_sqrt_cov[s] @ U.T @ (z - c[s])`

Because orthogonal residual carry weight is fixed to `0.0`, components outside
the train principal-component subspace are not carried into the edited
signature.

If the displacement norm `||transport_v4[s -> t](z) - z||` exceeds `200.0`,
scale the displacement to norm `200.0`:

`z + 200.0 * (transport_v4[s -> t](z) - z) / ||transport_v4[s -> t](z) - z||`

This cap is fixed before development and applies equally to matched V4 and
every V4 transport control.

There are `4 * 3 = 12` fixed V4 transports. Development data may not change
rank, shrinkage, eigenvalue floor, norm cap, thresholds, controls, or evaluator
settings.

## Baseline Transports

V2 centroid-delta baseline:

`transport_v2[s -> t](z) = z + c[t] - c[s]`

V3 diagonal transport baseline:

- compute behavior diagonal variances and global diagonal variance from V4
  train normalized signatures only;
- use V3 shrinkage weights `0.75` behavior and `0.25` global;
- clamp shrunk variances below `1e-4`;
- use ratio `std_shrunk[t] / std_shrunk[s]`;
- clip each ratio element to `[0.25, 4.0]`;
- apply `c[t] + ratio[s -> t] * (z - c[s])`.

The V2 and V3 baselines in V4 are recomputed from V4 train subjects only. They
do not reuse V2 or V3 train subjects, development subjects, final subjects, or
saved vectors.

## Controls

For each development/final source subject and target behavior, evaluate the
matched V4 transport against these controls:

- no edit;
- null vector;
- V2 matched centroid delta;
- V3 matched diagonal transport;
- reverse V4 transport:
  `transport_v4[target -> source](z)`;
- same-source other-target V4 transports:
  `transport_v4[source -> other_target](z)`;
- same-target other-source V4 transports:
  `transport_v4[other_source -> target](z)`;
- deterministic shuffled V4 transport, selected from ordered source-target
  pairs excluding the matched pair and reverse pair;
- `32` deterministic random norm-matched vectors, where the norm is matched to
  the V4 matched displacement `transport_v4[source -> target](z) - z`.

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
- fixed salt `representation_steering_v4_low_rank_residual_transport`.

Exact shuffled-direction selection:

- candidate directions are all ordered behavior pairs
  `(candidate_source, candidate_target)` where source and target differ;
- exclude the matched pair `(source, target)`;
- exclude the reverse pair `(target, source)`;
- sort candidates lexicographically by `(candidate_source, candidate_target)`;
- compute `digest = stable_hash_json([subject_id, source, target,
  "representation_steering_v4_low_rank_residual_transport_shuffled_direction"])`;
- select candidate index `int(digest[:16], 16) % len(candidates)`;
- apply the selected candidate's V4 transport to the current source subject.

Exact random norm-matched vector generation:

- compute `seed_digest = stable_hash_json([subject_id, source, target,
  "representation_steering_v4_low_rank_residual_transport_random"])`;
- seed a `torch.Generator` with `int(seed_digest[:16], 16) % (2 ** 31)`;
- for each random control index from `0` to `31`, draw one Gaussian vector with
  `torch.randn` using the seeded generator and the normalized-signature shape;
- normalize the Gaussian vector by its L2 norm clamped below at `1e-12`;
- multiply by the matched V4 displacement norm
  `||transport_v4[source -> target](z) - z||`, also clamped below at `1e-12`;
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
- matched-minus-V2-centroid-delta centroid improvement;
- matched-minus-V3-diagonal-transport primary target margin;
- matched-minus-V3-diagonal-transport centroid improvement.

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
- matched-minus-V2-centroid-delta centroid improvement must be `> 0.0`;
- matched-minus-V3-diagonal-transport primary target margin must be `> 0.0`;
- matched-minus-V3-diagonal-transport centroid improvement must be `> 0.0`.

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
- mean matched-minus-V3-diagonal-transport primary target margin must be
  `> 0.10`;
- mean matched-minus-V3-diagonal-transport centroid improvement must be
  `> 0.10`;
- mean primary source margin change must be `< -0.05`.

Development and final use identical gates. Development is a go/no-go check
only. If development fails, do not run final evaluation.

## Pass/Fail Interpretation

If V4 passes development and final:

- evidence supports the narrow claim that train-only low-rank residual
  covariance transports in fixed stored-probe signature space can steer heldout
  representations across this four-behavior suite under the registered
  controls;
- evidence supports source-conditioned improvement over V2 centroid deltas and
  V3 diagonal transports only for this small representation-space setting;
- evidence does not support functional model editing unless paired with a
  separately locked valid decoder;
- evidence does not prove larger-model, natural-language-model, or broad MUAT
  generality.

If V4 fails development:

- log a negative V4 development result;
- keep the V4 final raw pool sealed;
- do not reinterpret partial metrics as proof of steering.

If V4 passes development but fails final:

- log a failed final result;
- treat the development result as overfit or non-generalizing for this protocol.

## Reviewer Checkpoints

Required reviewer checkpoints, each with confidence `5/5` before the next step:

1. V4 preregistration acceptance.
2. V4 implementation acceptance before source-pool generation.
3. V4 source-pool construction acceptance.
4. V4 development result acceptance.
5. V4 final result acceptance, only if development passes.
