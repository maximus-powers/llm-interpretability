# Four-Behavior Representation Steering V6 Centroid-Constrained Primary Correction Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a sixth four-behavior representation-steering proof
attempt for fixed stored-probe activation signatures.

V4 low-rank residual transport preserved target-centroid geometry better than
later methods, but failed primary target-margin and reliability gates. V5
contrastive residual calibration produced very strong primary target margins
and source suppression, but damaged centroid best-control specificity: V4
remained the best centroid control for most records.

V6 tests the next narrow claim:

Train-only centroid-constrained primary correction can retain V4's
target-centroid geometry while adding enough primary classifier margin and
source suppression to beat prior-method, non-matched, shuffled, and random
controls on fresh heldout subjects.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model, natural-language-model, or broad MUAT
generality.

## Prior Development Contamination Policy

The V1, V2, V3, V4, and V5 development pools were inspected during failure
analysis. V6 must not report a V6 development result on any prior steering
development pool.

V6 must generate fresh V6 train, development, and final source pools before any
V6 development evaluation. The V6 final raw pool must remain sealed until the
frozen V6 method passes development and reviewer accepts the development
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

## V6 Source Pools

Generate V6-specific source pools with the same accepted source-generation
method used by prior steering pool construction:

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

V6 source-pool output directory:

`runs/four_behavior_representation_steering_v6_pools`

Required V6 source-pool artifact claim scopes:

- train/development/final raw pool payloads:
  `four_behavior_representation_steering_v6_source_pool`;
- combined source-pool audit:
  `four_behavior_representation_steering_v6_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v6_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `45300000`;
- development base seed: `46300000`;
- final base seed: `47300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every V6 pool/behavior range and fail if any configured ranges overlap.

Before development evaluation, the combined source-pool audit must show:

- accepted counts meet the configured per-behavior targets for train,
  development, and final;
- accepted train/development/final overlaps are zero for seed, subject id,
  weight hash, and signature hash;
- every accepted subject clears the source heldout margin gate;
- only the final redacted audit surface is used for final-pool metadata.

Before development or final evaluation, V6 scripts must fail unless:

- pool directory is exactly
  `runs/four_behavior_representation_steering_v6_pools`, unless a reviewer
  explicitly accepts a same-contents relocation before generation;
- pool directory is not under any prior steering pool directory;
- train raw pool payload has V6 source-pool claim scope and pool name `train`;
- development raw pool payload has V6 source-pool claim scope and pool name
  `development`;
- combined audit has V6 source-pool-construction claim scope;
- final redacted audit has V6 final-redacted claim scope;
- accepted counts and overlap counts match this preregistration.

Before development evaluation, the script must not open or validate the raw
final payload. Development may validate final-pool metadata only through the V6
combined audit and V6 final redacted audit.

During final evaluation only, after development passes and reviewer accepts the
development result at `5/5`, the script must validate that the final raw payload
has V6 source-pool claim scope and pool name `final` before evaluating it.

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
- final per-subject metrics.

Any exposure of forbidden final raw or final-detail fields before final
evaluation invalidates the V6 final pool for proof use.

## Final-Pool Access Policy

Before final evaluation, V6 scripts may read only:

- V6 train raw pool;
- V6 development raw pool;
- V6 combined source-pool audit;
- V6 final redacted audit.

Before final evaluation, V6 scripts must reject:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v2_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v3_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v4_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v5_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v6_pools/final_subjects.json`;
- any other path named `final_subjects.json` unless the command is explicitly
  the V6 final-evaluation command.

The V6 final raw pool may be opened only after:

1. this preregistration is accepted by reviewer at `5/5`;
2. V6 source-pool construction is accepted by reviewer at `5/5`;
3. V6 development evaluation passes all gates below;
4. reviewer accepts the V6 development result at `5/5`.

If V6 development fails, final evaluation is blocked and the final raw pool
must remain sealed.

## Train-Only Statistics

Compute all representation statistics from accepted V6 train subjects only:

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
four behavior labels using accepted V6 train subjects only.

Frozen config:

- model: one affine layer from dimension `560` to `4`;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260821`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only.

No development or final data may select, tune, calibrate, or refit this
evaluator.

The centroid evaluator is a mandatory secondary evaluator. Nearest-centroid
prediction uses train-only behavior centroids in normalized signature space.

## V6 Centroid-Constrained Primary Correction

V6 has no learned cross-subject parameters beyond the train-only classifier and
statistics above. It computes a deterministic per-subject correction at
evaluation time using only the source signature, source behavior, target
behavior, train-only centroids, train-only V4 low-rank transport, and train-only
linear evaluator.

V6 first computes the same train-only V4 low-rank residual covariance transport
as in V4/V5, recomputed from V6 train subjects:

1. Normalize train signatures with train-only mean and standard deviation.
2. Compute behavior centroids in normalized signature space.
3. Center all normalized train signatures by the global train mean.
4. Compute SVD with `torch.linalg.svd(centered, full_matrices=False)`.
5. Use the first `48` rows of `vh`, transposed, as the PCA component matrix
   `U` with shape `560 x 48`.
6. For each behavior, project residuals `z - centroid[behavior]` into the PCA
   subspace.
7. Compute covariance as `projected.T @ projected / projected.size(0)`.
8. Compute global projected covariance over concatenated behavior residuals.
9. For each behavior, compute shrunk covariance as
   `0.75 * behavior_cov + 0.25 * global_cov`.
10. Factor each shrunk covariance with `torch.linalg.eigh`, clamp eigenvalues
    below `1e-4`, and construct `sqrt_cov` and `inv_sqrt_cov`.
11. For source `s` and target `t`, compute:
    - `residual = z - centroid[s]`;
    - `projected = U.T @ residual`;
    - `transported_projected = sqrt_cov[t] @ inv_sqrt_cov[s] @ projected`;
    - `v4_uncapped = centroid[t] + U @ transported_projected`.
12. Orthogonal residual carry weight is fixed at `0.0`.
13. Cap total displacement from original `z` to `200.0` to obtain
    `v4_capped`.

V6 then optimizes an additive PCA-subspace correction `q` for each
source-target example:

`candidate = project_ball(cap(z, v4_uncapped + U @ q), center=centroid[t], radius=max(||v4_capped - centroid[t]||_2 - 0.05, 0.0))`

The target-centroid ball projection is exact:

- if `||candidate - centroid[t]||_2 <= radius`, leave candidate unchanged;
- if `||candidate - centroid[t]||_2 <= 1e-12`, return `centroid[t]`;
- otherwise return
  `centroid[t] + radius * (candidate - centroid[t]) / ||candidate - centroid[t]||_2`.

This projection is applied after every optimizer step and after the final step.
After each projection, update the optimizer state variable by overwriting:

`q = U.T @ (projected_candidate - v4_uncapped)`

inside a `torch.no_grad()` block before the next optimizer step. Because the
optimizer is plain SGD with no momentum and no weight decay, there are no
optimizer moment buffers to reset after this overwrite. The projection and
overwrite happen after each of the `80` optimizer steps, including step `80`.
The final output is recomputed from the final overwritten `q` through the same
`cap` plus `project_ball` forward path. This makes the V6 matched candidate no
farther from the target centroid than V4. When
`||v4_capped - centroid[t]||_2 >= 0.05`, the projected candidate is constrained
to be at least `0.05` closer to the target centroid than V4 before numerical
roundoff.

Fixed per-example correction config:

- correction dimension: `48`;
- initialization: `q = zeros(48)`;
- optimizer: plain `torch.optim.SGD` with no momentum;
- learning rate: `0.05`;
- weight decay: `0.0`;
- steps: `80`;
- deterministic seed: `20260822`;
- no early stopping;
- no development or final metric may change steps, learning rate, thresholds,
  projection radius, or objective weights.

For a candidate `x`, define:

- `logits = classifier(x)`;
- `primary_target_margin(x) = logits[t] - max(logits[j] for j != t)`;
- `primary_source_margin(x) = logits[s] - max(logits[j] for j != s)`;
- `centroid_distance(x) = ||x - centroid[t]||_2`.

The per-step objective is exactly:

`loss = target_ce + source_loss + correction_norm_loss`

where:

- `target_ce = cross_entropy(logits[None, :], target_index[t][None])`;
- `source_loss = 1.0 * relu(primary_source_margin(candidate) + 0.05)`;
- `correction_norm_loss = 0.001 * ||q||_2^2`.

The centroid constraint is enforced only through projection, not through a soft
loss. The final candidate is the projected candidate after step `80`.

## Baseline Controls

V2, V3, V4, and V5 baseline controls are recomputed from V6 train-only
statistics:

- V2 centroid delta:
  `v2[s -> t](z) = z + centroid[t] - centroid[s]`.
- V3 diagonal transport:
  - compute train-only global and per-behavior normalized-signature variances;
  - shrink variance as `0.75 * behavior_var + 0.25 * global_var`;
  - clamp below `1e-4`;
  - `std[behavior] = sqrt(shrunk_var)`;
  - `ratio = clamp(std[t] / std[s], min=0.25, max=4.0)`;
  - `v3[s -> t](z) = centroid[t] + ratio * (z - centroid[s])`.
- V4 low-rank residual transport:
  exact V6 train-only V4 computation above with no V6 correction.
- V5 contrastive residual calibration baseline:
  - train `12 * 48` V5 calibration coefficients on V6 train subjects only;
  - use the exact accepted V5 full-batch loss, detached train-time controls,
    optimizer, epochs, and salts, but recomputed from V6 train-only statistics;
  - V5 is a baseline control only, not the V6 method under test.

## Evaluation Controls

For every development/final source subject and target behavior, evaluate the
matched V6 correction against:

- no edit;
- null vector, defined as zero displacement from source signature, candidate
  exactly `z`;
- V2 centroid delta;
- V3 diagonal transport;
- uncorrected V4 low-rank residual transport;
- V5 contrastive residual calibration;
- reverse V6 correction;
- same-source other-target V6 corrections;
- same-target other-source V6 corrections;
- deterministic shuffled V6 correction;
- `32` random norm-matched vectors.

Deterministic shuffled direction selection:

- candidate directions are all `(candidate_source, candidate_target)` pairs in
  `PATTERNS x PATTERNS` where source differs from target;
- exclude the matched direction `(source, target)`;
- exclude the reverse direction `(target, source)`;
- sort remaining candidate tuples lexicographically;
- compute `stable_hash_json([subject_id, source, target, "representation_steering_v6_centroid_constrained_primary_correction_shuffled_direction"])`;
- use the first `16` hex characters as an integer modulo the candidate count.

Deterministic random evaluation controls:

- compute `stable_hash_json([subject_id, source, target, "representation_steering_v6_centroid_constrained_primary_correction_random"])`;
- seed `torch.Generator().manual_seed(int(first_16_hex, 16) % (2 ** 31))`;
- for random indices `0..31`, sequentially sample
  `torch.randn(z.shape, dtype=z.dtype, generator=generator)` from that single
  generator;
- divide by `random_vector.norm().clamp_min(1e-12)`;
- multiply by the matched V6 displacement norm from `z`, clamped to at least
  `1e-12`;
- add the resulting vector to `z`.

Every control record must report:

- control type;
- control source and target behavior when applicable;
- primary target margin;
- primary source margin;
- centroid target distance before and after;
- centroid improvement;
- matched-minus-control primary target margin;
- matched-minus-control centroid improvement.

## Development Go/No-Go Gates

V6 development passes only if all gates below pass.

Aggregate gates over all `24 * 4 * 3 = 288` development records:

- `n == 288`;
- individual all-gate pass rate `>= 0.90`;
- mean matched primary target margin `> 0.25`;
- mean matched centroid improvement `> 0.15`;
- mean matched-minus-best-control primary target margin `> 0.10`;
- mean matched-minus-best-control centroid improvement `> 0.05`;
- mean matched-minus-V2-centroid-delta primary target margin `> 0.10`;
- mean matched-minus-V2-centroid-delta centroid improvement `> 0.05`;
- mean matched-minus-V3-diagonal-transport primary target margin `> 0.10`;
- mean matched-minus-V3-diagonal-transport centroid improvement `> 0.05`;
- mean matched-minus-V4-low-rank primary target margin `> 0.10`;
- mean matched-minus-V4-low-rank centroid improvement `> 0.05`;
- mean matched-minus-V5-calibrated primary target margin `> 0.10`;
- mean matched-minus-V5-calibrated centroid improvement `> 0.05`;
- mean source primary margin change `< -0.05`.

Per-target gates:

- each target has `72` records;
- each target individual all-gate pass rate `>= 0.80`.

Per-direction gates:

- each ordered source-target direction has `24` records;
- each direction individual all-gate pass rate `>= 0.90`.

Per-record all-gate pass requires:

- primary predicted behavior equals target behavior;
- centroid predicted behavior equals target behavior;
- matched primary target margin `> 0.25`;
- matched centroid improvement `> 0.15`;
- matched-minus-best-control primary target margin `> 0.0`;
- matched-minus-best-control centroid improvement `> 0.0`;
- matched-minus-V2-centroid-delta primary target margin `> 0.0`;
- matched-minus-V2-centroid-delta centroid improvement `> 0.0`;
- matched-minus-V3-diagonal-transport primary target margin `> 0.0`;
- matched-minus-V3-diagonal-transport centroid improvement `> 0.0`;
- matched-minus-V4-low-rank primary target margin `> 0.0`;
- matched-minus-V4-low-rank centroid improvement `> 0.0`;
- matched-minus-V5-calibrated primary target margin `> 0.0`;
- matched-minus-V5-calibrated centroid improvement `> 0.0`;
- source primary margin change `< -0.05`.

If development fails any gate, V6 is a negative development checkpoint and final
raw evaluation is blocked.

If development passes all gates and reviewer accepts the checkpoint at `5/5`,
the exact same frozen code, statistics, controls, and gates are run once on the
V6 final raw pool.

## Reporting Requirements

The development and final result artifacts must include:

- claim scope;
- phase;
- source-pool audit hashes;
- train and eval pool file hashes;
- final redacted audit hash;
- train-only statistics hash;
- V5 baseline calibration hash;
- aggregate, by-target, and by-direction summaries;
- individual gate audit;
- all per-record matched and control metrics;
- explicit pass/fail and next action.

Development result artifacts must not include any raw final pool path or final
raw per-subject field.

## Valid Claim If Development And Final Pass

If and only if both development and final pass all gates, the valid claim is:

For this small subject architecture and four clean synthetic behaviors, fixed
stored-probe activation signatures support train-only centroid-constrained
source-conditioned representation steering that generalizes to fresh heldout
subjects and beats strong non-matched, prior-method, shuffled, and random
controls in representation space.

This would still not prove:

- four-behavior functional decoding;
- larger model behavior;
- natural-language-model behavior;
- broad MUAT generality;
- mechanistic equivalence between representation-space movement and weight
  edits.
