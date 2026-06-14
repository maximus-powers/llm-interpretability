# Four-Behavior Representation Steering V5 Contrastive Residual Calibration Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a fifth four-behavior representation-steering proof
attempt for fixed stored-probe activation signatures.

V2 centroid deltas, V3 diagonal transport, and V4 low-rank residual covariance
transport all produced aggregate target movement in normalized stored-probe
signature space, but failed proof-grade reliability and source-specificity. V4
improved centroid movement over V2 and V3, yet its matched transport was often
beaten by same-source other-target, same-target other-source, V2, or V3
controls.

V5 tests the next narrow source-specificity claim:

Train-only contrastive residual calibration on top of V4 low-rank transport can
steer fresh heldout source signatures toward target behavior regions while
beating V2 centroid deltas, V3 diagonal transports, uncalibrated V4 low-rank
transports, non-matched calibrated transports, shuffled calibrated transports,
and random norm-matched controls.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model, natural-language-model, or broad MUAT
generality.

## Prior Development Contamination Policy

The V1, V2, V3, and V4 development pools were inspected during failure analysis.
V5 must not report a V5 development result on any prior steering development
pool.

V5 must generate fresh V5 train, development, and final source pools before any
V5 development evaluation. The V5 final raw pool must remain sealed until the
frozen V5 method passes development and reviewer accepts the development
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

## V5 Source Pools

Generate V5-specific source pools with the same accepted source-generation
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

V5 source-pool output directory:

`runs/four_behavior_representation_steering_v5_pools`

Required V5 source-pool artifact claim scopes:

- train/development/final raw pool payloads:
  `four_behavior_representation_steering_v5_source_pool`;
- combined source-pool audit:
  `four_behavior_representation_steering_v5_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v5_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `42300000`;
- development base seed: `43300000`;
- final base seed: `44300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every V5 pool/behavior range and fail if any configured ranges overlap.

Before development evaluation, the combined source-pool audit must show:

- accepted counts meet the configured per-behavior targets for train,
  development, and final;
- accepted train/development/final overlaps are zero for seed, subject id,
  weight hash, and signature hash;
- every accepted subject clears the source heldout margin gate;
- only the final redacted audit surface is used for final-pool metadata.

Before development or final evaluation, V5 scripts must fail unless:

- pool directory is exactly
  `runs/four_behavior_representation_steering_v5_pools`, unless a reviewer
  explicitly accepts a same-contents relocation before generation;
- pool directory is not under any prior steering pool directory;
- train raw pool payload has V5 source-pool claim scope and pool name `train`;
- development raw pool payload has V5 source-pool claim scope and pool name
  `development`;
- combined audit has V5 source-pool-construction claim scope;
- final redacted audit has V5 final-redacted claim scope;
- accepted counts and overlap counts match this preregistration.

Before development evaluation, the script must not open or validate the raw
final payload. Development may validate final-pool metadata only through the V5
combined audit and V5 final redacted audit.

During final evaluation only, after development passes and reviewer accepts the
development result at `5/5`, the script must validate that the final raw payload
has V5 source-pool claim scope and pool name `final` before evaluating it.

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
evaluation invalidates the V5 final pool for proof use.

## Final-Pool Access Policy

Before final evaluation, V5 scripts may read only:

- V5 train raw pool;
- V5 development raw pool;
- V5 combined source-pool audit;
- V5 final redacted audit.

Before final evaluation, V5 scripts must reject:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v2_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v3_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v4_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v5_pools/final_subjects.json`;
- any other path named `final_subjects.json` unless the command is explicitly
  the V5 final-evaluation command.

The V5 final raw pool may be opened only after:

1. this preregistration is accepted by reviewer at `5/5`;
2. V5 source-pool construction is accepted by reviewer at `5/5`;
3. V5 development evaluation passes all gates below;
4. reviewer accepts the V5 development result at `5/5`.

If V5 development fails, final evaluation is blocked and the final raw pool
must remain sealed.

## Train-Only Statistics

Compute all representation statistics from accepted V5 train subjects only:

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
four behavior labels using accepted V5 train subjects only.

Frozen config:

- model: one affine layer from dimension `560` to `4`;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260731`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only.

No development or final data may select, tune, calibrate, or refit this
evaluator.

The centroid evaluator is a mandatory secondary evaluator. Nearest-centroid
prediction uses train-only behavior centroids in normalized signature space.

## V5 Contrastive Residual Calibration

V5 starts from the V4 low-rank residual covariance transport, recomputed from
V5 train subjects only. It then learns one small additive calibration vector for
every ordered source-target behavior pair where source differs from target.

The V5 candidate for source `s`, target `t`, and normalized signature `z` is:

`v5[s -> t](z) = cap(z, v4[s -> t](z) + U @ a[s -> t])`

where:

- `v4[s -> t](z)` is the V4 low-rank residual covariance transport computed
  from V5 train-only statistics;
- `U` is the V5 train-only PCA component matrix;
- `a[s -> t]` is a learned rank-`48` coefficient vector;
- `cap` enforces the fixed total displacement norm cap below.

V5 must reuse the accepted V4 low-rank residual transport semantics exactly,
except all statistics are recomputed from the V5 train pool:

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
12. Because orthogonal residual carry weight is fixed at `0.0`, no orthogonal
    residual is added.
13. V5 adds `U @ a[s -> t]` to `v4_uncapped`.
14. The final post-calibration cap is applied once to total displacement from
    the original `z`: if `||candidate - z||_2 > 200.0`, return
    `z + 200.0 * (candidate - z) / ||candidate - z||_2`.

Uncalibrated V4 controls use the same V5 train-only V4 computation and the same
single total-displacement cap from `z`, but with no calibration vector.

Fixed V4 base-transport hyperparameters:

- principal-component rank: `48`;
- covariance shrinkage behavior weight: `0.75`;
- covariance shrinkage global weight: `0.25`;
- covariance eigenvalue floor: `1e-4`;
- orthogonal residual carry weight: `0.0`;
- displacement norm cap after calibration: `200.0`;
- diagonal baseline ratio clip min/max: `0.25` and `4.0`.

V2 and V3 baseline controls are recomputed from V5 train-only statistics:

- V2 centroid delta:
  `v2[s -> t](z) = z + centroid[t] - centroid[s]`.
- V3 diagonal transport:
  - compute global normalized-signature variance over V5 train subjects;
  - compute per-behavior normalized-signature variance over V5 train subjects;
  - for each behavior, compute shrunk variance as
    `0.75 * behavior_var + 0.25 * global_var`;
  - clamp shrunk variance below `1e-4`;
  - `std[behavior] = sqrt(shrunk_var)`;
  - `ratio = clamp(std[t] / std[s], min=0.25, max=4.0)`;
  - `v3[s -> t](z) = centroid[t] + ratio * (z - centroid[s])`.

Fixed calibration training config:

- parameters: only the `12 * 48` calibration coefficients `a[s -> t]`;
- initialization: all zeros;
- optimizer: `AdamW`;
- learning rate: `0.03`;
- weight decay: `0.001`;
- epochs: `500`;
- deterministic seed: `20260801`;
- checkpoint: final epoch only;
- no early stopping;
- no development or final loss, metric, label, or prediction may select any
  hyperparameter, epoch, threshold, or coefficient.

Training examples are all accepted V5 train subjects paired with every target
behavior different from the source behavior.

Training is full-batch. Each epoch evaluates all `256 * 3 = 768` train examples
in deterministic subject-file order and target order equal to `PATTERNS`,
skipping the source behavior. No minibatching or shuffling is allowed.

For each train example `(z, s, t)`, define:

- `matched = v5[s -> t](z)`;
- `logits = classifier(matched)`;
- `target_ce = cross_entropy(logits[None, :], target_index[t][None])`;
- `primary_target_margin(x) = logits_x[t] - max(logits_x[j] for j != t)`;
- `primary_source_margin(x) = logits_x[s] - max(logits_x[j] for j != s)`;
- `centroid_improvement(x) = ||z - centroid[t]||_2 - ||x - centroid[t]||_2`.

The train-time control set is:

- no edit;
- null vector, defined as zero displacement from the source signature, so its
  candidate is exactly `z`;
- V2 centroid delta;
- V3 diagonal transport;
- uncalibrated V4 low-rank transport;
- reverse V5 calibrated transport;
- same-source other-target V5 calibrated transports;
- same-target other-source V5 calibrated transports;
- deterministic shuffled V5 calibrated transport.

V5 control candidates are computed with the current calibration coefficients,
but every train-time control candidate and every control score is detached
before entering a ranking loss. Gradients from ranking losses update only the
matched direction coefficient `a[s -> t]`; they must not update reverse,
same-source, same-target, or shuffled control coefficients through the negative
side of a comparison.

Per-example loss is exactly:

`loss = target_ce + source_loss + centroid_loss + primary_rank_loss + centroid_rank_loss + coeff_loss`

where:

- `source_loss = 1.0 * relu(primary_source_margin(matched) + 0.05)`;
- `centroid_loss = 0.02 * ||matched - centroid[t]||_2^2`;
- `coeff_loss = 0.001 * ||a[s -> t]||_2^2`;
- `primary_rank_loss = 1.0 * mean(relu(0.50 - (primary_target_margin(matched) - detached_primary_target_margin(control))))`
  over all train-time controls;
- `centroid_rank_loss = 1.0 * mean(relu(0.05 - (centroid_improvement(matched) - detached_centroid_improvement(control))))`
  over all train-time controls.

The epoch objective is the arithmetic mean of per-example losses over all train
examples. The final checkpoint is the coefficient tensor after epoch `500`.

Train-time controls do not include random norm-matched vectors; random controls
are evaluation controls only.

Deterministic shuffled direction selection:

- candidate directions are all `(candidate_source, candidate_target)` pairs in
  `PATTERNS x PATTERNS` where source differs from target;
- exclude the matched direction `(source, target)`;
- exclude the reverse direction `(target, source)`;
- sort remaining candidate tuples lexicographically;
- compute `stable_hash_json([subject_id, source, target, "representation_steering_v5_contrastive_residual_calibration_shuffled_direction"])`;
- use the first `16` hex characters as an integer modulo the candidate count.

Deterministic random evaluation controls:

- compute `stable_hash_json([subject_id, source, target, "representation_steering_v5_contrastive_residual_calibration_random"])`;
- seed `torch.Generator().manual_seed(int(first_16_hex, 16) % (2 ** 31))`;
- for random indices `0..31`, sequentially sample
  `torch.randn(z.shape, dtype=z.dtype, generator=generator)` from that single
  generator; the random index is recorded in the result but is not part of the
  seed payload;
- divide by `random_vector.norm().clamp_min(1e-12)`;
- multiply by the matched V5 displacement norm from `z`, clamped to at least
  `1e-12`;
- add the resulting vector to `z`.

## Evaluation Controls

For every development/final source subject and target behavior, evaluate the
matched V5 calibrated transport against:

- no edit;
- null vector, defined as zero displacement from the source signature, so its
  candidate is exactly `z`;
- V2 centroid delta;
- V3 diagonal transport;
- uncalibrated V4 low-rank residual transport;
- reverse V5 calibrated transport;
- same-source other-target V5 calibrated transports;
- same-target other-source V5 calibrated transports;
- deterministic shuffled V5 calibrated transport;
- `32` random norm-matched vectors generated from a deterministic hash of
  subject id, source behavior, and target behavior.

Random control vectors must:

- have the same norm as the matched V5 displacement from the source signature;
- use deterministic `torch.Generator` seeds derived from SHA-256;
- be reported individually, not only as aggregates.

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

V5 development passes only if all gates below pass.

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
- source primary margin change `< -0.05`.

If development fails any gate, V5 is a negative development checkpoint and final
raw evaluation is blocked.

If development passes all gates and reviewer accepts the checkpoint at `5/5`,
the exact same frozen code, statistics, coefficients, controls, and gates are
run once on the V5 final raw pool.

## Reporting Requirements

The development and final result artifacts must include:

- claim scope;
- phase;
- source-pool audit hashes;
- train and eval pool file hashes;
- final redacted audit hash;
- train-only statistics hash;
- calibration coefficient hash;
- calibration training summary;
- aggregate, by-target, and by-direction summaries;
- individual gate audit;
- all per-record matched and control metrics;
- explicit pass/fail and next action.

Development result artifacts must not include any raw final pool path or final
raw per-subject field.

## Valid Claim If Development And Final Pass

If and only if both development and final pass all gates, the valid claim is:

For this small subject architecture and four clean synthetic behaviors, fixed
stored-probe activation signatures support train-only contrastive
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
