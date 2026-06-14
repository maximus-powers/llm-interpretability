# Four-Behavior Representation Steering V7 Pareto-Frontier Correction Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a seventh four-behavior representation-steering
development attempt for fixed stored-probe activation signatures.

V5 produced very strong primary target margins but lost centroid specificity.
V6 restored centroid movement relative to V4/V5 but failed scalar
best-control gates because V5 and other controls often dominated one metric
while losing the other. A posthoc V6 development-only Pareto diagnosis found
that V6 was Pareto-undominated on `226/288` development records, but the worst
ordered directions were only `16/24`. That diagnosis is not proof and does not
authorize opening any V6 final raw pool.

V7 tests a narrower, explicitly multiobjective claim:

For each source-target example, a deterministic train-only Pareto-frontier
correction can produce a small fixed candidate set in normalized
stored-probe-signature space that contains at least one target-behavior steering
candidate that is not Pareto-dominated by prior-method, non-matched V7,
shuffled V7, or random controls on the two registered representation metrics:
primary target margin and target-centroid improvement.

This is representation-space evidence only. It is not a functional decoder
result and does not prove larger-model, natural-language-model, or broad MUAT
generality.

## Prior Development Contamination Policy

The V1, V2, V3, V4, V5, and V6 development pools and the V6 posthoc Pareto
diagnosis were inspected during failure analysis. V7 must not report a V7
development result on any prior steering development pool.

V7 must generate fresh V7 train, development, and final source pools before any
V7 development evaluation. The V7 final raw pool must remain sealed until the
frozen V7 method passes development and reviewer accepts the development
checkpoint at `5/5`.

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

## V7 Source Pools

Generate V7-specific source pools with the same accepted source-generation
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

V7 source-pool output directory:

`runs/four_behavior_representation_steering_v7_pools`

Required V7 source-pool artifact claim scopes:

- train/development/final raw pool payloads:
  `four_behavior_representation_steering_v7_source_pool`;
- combined source-pool audit:
  `four_behavior_representation_steering_v7_source_pool_construction`;
- final redacted audit:
  `redacted_final_steering_v7_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `51300000`;
- development base seed: `52300000`;
- final base seed: `53300000`;
- behavior stride: `100000`;
- seed formula:
  `base_seed + behavior_index * behavior_stride + attempt_index`;
- max attempts per behavior:
  - train: `128`;
  - development: `64`;
  - final: `64`.

Before training any source subjects, the script must run a seed-range preflight
over every V7 pool/behavior range and fail if any configured ranges overlap.

Before development evaluation, the combined source-pool audit must show:

- accepted counts meet the configured per-behavior targets for train,
  development, and final;
- accepted train/development/final overlaps are zero for seed, subject id,
  weight hash, and signature hash;
- every accepted subject clears the source heldout margin gate;
- only the final redacted audit surface is used for final-pool metadata.

Before development or final evaluation, V7 scripts must fail unless:

- pool directory is exactly
  `runs/four_behavior_representation_steering_v7_pools`, unless a reviewer
  explicitly accepts a same-contents relocation before generation;
- pool directory is not under any prior steering pool directory;
- train raw pool payload has V7 source-pool claim scope and pool name `train`;
- development raw pool payload has V7 source-pool claim scope and pool name
  `development`;
- combined audit has V7 source-pool-construction claim scope;
- final redacted audit has V7 final-redacted claim scope;
- accepted counts and overlap counts match this preregistration.

Before development evaluation, the script must not open or validate the raw
final payload. Development may validate final-pool metadata only through the V7
combined audit and V7 final redacted audit.

During final evaluation only, after development passes and reviewer accepts the
development result at `5/5`, the script must validate that the final raw payload
has V7 source-pool claim scope and pool name `final` before evaluating it.

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

Allowed combined-audit final summary fields before final evaluation:

- accepted counts by behavior;
- final pool file hash;
- final redacted payload hash.

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
evaluation invalidates the V7 final pool for proof use.

## Final-Pool Access Policy

Before final evaluation, V7 scripts may read only:

- V7 train raw pool;
- V7 development raw pool;
- V7 combined source-pool audit;
- V7 final redacted audit.

Before final evaluation, V7 scripts must reject:

- `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`;
- `runs/four_behavior_representation_steering_v1_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v2_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v3_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v4_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v5_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v6_pools/final_subjects.json`;
- `runs/four_behavior_representation_steering_v7_pools/final_subjects.json`;
- any other path named `final_subjects.json` unless the command is explicitly
  the V7 final-evaluation command.

The V7 final raw pool may be opened only after:

1. this preregistration is accepted by reviewer at `5/5`;
2. V7 implementation is accepted by reviewer at `5/5`;
3. V7 source-pool construction is accepted by reviewer at `5/5`;
4. V7 development evaluation passes all gates below;
5. reviewer accepts the V7 development result at `5/5`.

Before opening the V7 final raw pool, the final-evaluation command must validate
a current passing V7 development artifact whose values exactly match:

- `claim_scope`:
  `four_behavior_representation_steering_v7_pareto_frontier_correction_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `transport_method`: `train_pareto_frontier_correction`;
- `train_pool_sha256`: current V7 train raw pool SHA-256;
- `eval_pool_sha256`: current V7 development raw pool SHA-256;
- `combined_audit_sha256`: current V7 combined-audit SHA-256;
- `final_redacted_audit_sha256`: current V7 final-redacted-audit SHA-256.

If any value differs, final evaluation must fail before opening the V7 final
raw pool.

If V7 development fails, final evaluation is blocked and the final raw pool
must remain sealed.

## Train-Only Statistics

Compute all representation statistics from accepted V7 train subjects only:

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
four behavior labels using accepted V7 train subjects only.

Frozen config:

- model: one affine layer from dimension `560` to `4`;
- optimizer: `AdamW`;
- learning rate: `0.10`;
- weight decay: `0.0001`;
- epochs: `1000`;
- deterministic seed: `20260921`;
- objective: full-batch cross entropy over accepted train subjects;
- checkpoint: final epoch only.

No development or final data may select, tune, calibrate, or refit this
evaluator.

The centroid evaluator is a mandatory secondary evaluator. Nearest-centroid
prediction uses train-only behavior centroids in normalized signature space.

## Shared Baseline Transports

V7 recomputes all prior-method baselines from V7 train-only statistics:

- V2 centroid delta;
- V3 diagonal transport;
- V4 low-rank residual covariance transport;
- V5 contrastive residual calibration baseline trained on V7 train subjects;
- V6 centroid-constrained primary correction baseline recomputed from V7
  train-only statistics.

The V5 baseline uses the accepted V5 full-batch loss, detached train-time
controls, optimizer, epochs, and salts, with V7 train-only statistics. V5 is a
baseline control, not the V7 method.

The V6 baseline uses the accepted V6 projected SGD procedure with V7 train-only
statistics. V6 is a baseline control, not the V7 method.

## V7 Pareto-Frontier Correction

V7 produces a fixed candidate set of `5` matched candidates per source-target
example. It has no learned cross-subject parameters beyond train-only
statistics, the primary evaluator, and the V5 baseline calibration coefficients.
V5 is used as a frozen train-only baseline/control and as the registered
primary-floor reference in the V7 objective. It is not updated during V7
per-example optimization.

For each source signature `z`, source behavior `s`, and target behavior `t`,
compute:

- V4 uncapped low-rank transport `v4_uncapped`;
- V4 capped transport `v4_capped`;
- V5 calibrated transport `v5`;
- source-to-target centroid distance before edit:
  `source_distance = ||z - centroid[t]||_2`;
- V4 target distance:
  `v4_distance = ||v4_capped - centroid[t]||_2`;
- V5 target distance:
  `v5_distance = ||v5 - centroid[t]||_2`.

The five registered target-centroid radius budgets are:

1. `max(v4_distance - 0.05, 0.0)`;
2. `max(v4_distance + 0.50, 0.0)`;
3. `max(v4_distance + 1.50, 0.0)`;
4. `max(min(v5_distance, source_distance - 0.15), 0.0)`;
5. `max(source_distance - 0.15, 0.0)`.

If any budget is larger than `source_distance - 0.15`, clamp it to
`max(source_distance - 0.15, 0.0)`. This ensures every V7 candidate is
constrained to improve target-centroid distance by at least `0.15` before
numerical roundoff.

For each radius budget, optimize an additive PCA-subspace correction `q`:

`candidate = project_ball(cap(z, v4_uncapped + U @ q), center=centroid[t], radius=radius_budget)`

The target-centroid ball projection is exact:

- if `||candidate - centroid[t]||_2 <= radius_budget`, leave candidate unchanged;
- if `||candidate - centroid[t]||_2 <= 1e-12`, return `centroid[t]`;
- otherwise return
  `centroid[t] + radius_budget * (candidate - centroid[t]) / ||candidate - centroid[t]||_2`.

Fixed per-candidate optimizer config:

- correction dimension: `48`;
- initialization: `q = zeros(48)`;
- optimizer: plain `torch.optim.SGD` with no momentum;
- learning rate: `0.05`;
- weight decay: `0.0`;
- steps: `120`;
- deterministic seed: `20260922`;
- no early stopping;
- no development or final metric may change steps, learning rate, thresholds,
  projection budgets, or objective weights.

For a candidate `x`, define:

- `logits = classifier(x)`;
- `primary_target_margin(x) = logits[t] - max(logits[j] for j != t)`;
- `primary_source_margin(x) = logits[s] - max(logits[j] for j != s)`;
- `centroid_improvement(x) = ||z - centroid[t]||_2 - ||x - centroid[t]||_2`.

The per-step objective for every radius budget is:

`loss = target_ce + source_loss + correction_norm_loss + v5_primary_floor_loss`

where:

- `target_ce = cross_entropy(logits[None, :], target_index[t][None])`;
- `source_loss = relu(primary_source_margin(candidate) + 0.05)`;
- `correction_norm_loss = 0.001 * ||q||_2^2`;
- `v5_primary_floor_loss = 0.25 * relu(primary_target_margin(v5) - primary_target_margin(candidate) - 0.10)`.

The V5 primary floor is a fixed train-only/control-derived baseline target. It
does not use development or final labels beyond the known source/target
direction and does not select hyperparameters from development outcomes.

After every optimizer step, including step `120`, project the candidate and
overwrite:

`q = U.T @ (projected_candidate - v4_uncapped)`

inside a `torch.no_grad()` block before the next step. The final output for
each radius budget is recomputed from the final overwritten `q` through the
same `cap` plus `project_ball` forward path.

The matched V7 object for a record is the ordered set of these five candidates,
not a single chosen vector.

## Fair Control Sets

Every non-matched V7 directional control receives the same `5`-candidate V7
frontier budget. A control frontier is built by applying the exact V7 algorithm
with its control source and target behaviors.

For every development/final source subject and target behavior, evaluate the
matched V7 frontier against:

- no edit;
- null vector, defined as zero displacement from source signature, candidate
  exactly `z`;
- V2 centroid delta;
- V3 diagonal transport;
- uncorrected V4 low-rank residual transport;
- V5 contrastive residual calibration;
- V6 centroid-constrained primary correction;
- reverse V7 frontier;
- same-source other-target V7 frontiers;
- same-target other-source V7 frontiers;
- deterministic shuffled V7 frontier;
- `32` random norm-matched vectors.

For scalar controls, the control candidate set has one candidate. For V7
directional controls, the control candidate set has five candidates. Pareto
dominance is computed against the union of every control candidate.

Deterministic shuffled direction selection:

- candidate directions are all `(candidate_source, candidate_target)` pairs in
  `PATTERNS x PATTERNS` where source differs from target;
- exclude the matched direction `(source, target)`;
- exclude the reverse direction `(target, source)`;
- sort remaining candidate tuples lexicographically;
- compute `stable_hash_json([subject_id, source, target, "representation_steering_v7_pareto_frontier_correction_shuffled_direction"])`;
- use the first `16` hex characters as an integer modulo the candidate count.

Deterministic random evaluation controls:

- compute `stable_hash_json([subject_id, source, target, "representation_steering_v7_pareto_frontier_correction_random"])`;
- seed `torch.Generator().manual_seed(int(first_16_hex, 16) % (2 ** 31))`;
- set the random norm to the maximum displacement norm among the five matched
  V7 candidates, clamped to at least `1e-12`;
- for random indices `0..31`, sequentially sample
  `torch.randn(z.shape, dtype=z.dtype, generator=generator)` from that single
  generator;
- divide by `random_vector.norm().clamp_min(1e-12)`;
- multiply by the random norm;
- add the resulting vector to `z`.

Every matched candidate and every control candidate must report:

- candidate set name;
- candidate index;
- source and target behavior when applicable;
- primary target margin;
- primary source margin;
- primary predicted behavior;
- centroid predicted behavior;
- centroid target distance before and after;
- centroid improvement.

## Pareto Evaluation Definitions

For each record, a matched V7 candidate is valid if:

- primary predicted behavior equals target behavior;
- centroid predicted behavior equals target behavior;
- primary target margin `> 0.25`;
- centroid improvement `> 0.15`;
- source primary margin change `< -0.05`;
- no control candidate Pareto-dominates it on
  `(primary_target_margin, centroid_improvement)`.

Control candidate `c` Pareto-dominates matched candidate `m` if:

- `primary_target_margin(c) >= primary_target_margin(m)`;
- `centroid_improvement(c) >= centroid_improvement(m)`;
- at least one of the two inequalities is strict by more than `1e-8`.

A record passes if at least one of its five matched V7 candidates is valid.

A record is Pareto-undominated if at least one of its five matched V7
candidates is not Pareto-dominated by any control candidate under the dominance
definition above, regardless of absolute primary/centroid/source thresholds and
regardless of primary or centroid predicted behavior. This diagnostic is
computed separately from record pass because it measures frontier geometry, not
whether a candidate is strong enough to count as successful steering.

Aggregate Pareto-undominated record rate is:

`number_of_pareto_undominated_records / 288`

Per-target Pareto-undominated record rate is:

`number_of_pareto_undominated_records_for_target / 72`

Per-direction Pareto-undominated record rate is:

`number_of_pareto_undominated_records_for_direction / 24`

For reporting, the record's selected matched candidate is the valid candidate
with the largest lexicographic tuple:

1. primary target margin;
2. centroid improvement;
3. negative candidate index.

If no matched candidate is valid, the selected matched candidate is the
candidate with the largest lexicographic tuple:

1. non-domination indicator (`1` if no control dominates it, else `0`);
2. primary target margin;
3. centroid improvement;
4. negative candidate index.

## Development Go/No-Go Gates

V7 development passes only if all gates below pass.

Aggregate gates over all `24 * 4 * 3 = 288` development records:

- `n == 288`;
- individual all-gate pass rate `>= 0.90`;
- Pareto-undominated record rate `>= 0.90`;
- mean selected primary target margin `> 0.25`;
- mean selected centroid improvement `> 0.15`;
- mean selected source primary margin change `< -0.05`;
- mean selected-minus-V2-centroid-delta primary target margin `> 0.10`;
- mean selected-minus-V2-centroid-delta centroid improvement `> 0.05`;
- mean selected-minus-V3-diagonal-transport primary target margin `> 0.10`;
- mean selected-minus-V3-diagonal-transport centroid improvement `> 0.05`;
- mean selected-minus-V4-low-rank primary target margin `> 0.10`;
- mean selected-minus-V4-low-rank centroid improvement `> 0.05`;
- mean selected-minus-V5-calibrated centroid improvement `> 0.05`;
- mean selected-minus-V6-correction primary target margin `> 0.10`;
- mean selected-minus-V6-correction centroid improvement `> 0.05`.

Because V7 is explicitly a Pareto-frontier method, it does not require selected
candidates to beat V5 on mean primary target margin. Instead, it requires
non-domination against the V5 candidate and all other controls per record.

Per-target gates:

- each target has `72` records;
- each target individual all-gate pass rate `>= 0.80`;
- each target Pareto-undominated record rate `>= 0.85`.

Per-direction gates:

- each ordered source-target direction has `24` records;
- each direction individual all-gate pass rate `>= 0.80`;
- each direction Pareto-undominated record rate `>= 0.80`;
- each direction has at least `20/24` records with a selected candidate whose
  primary and centroid predicted behaviors both equal the target.

Per-record all-gate pass requires:

- at least one matched V7 candidate is valid under the Pareto evaluation
  definition above.

If development fails any gate, V7 is a negative development checkpoint and final
raw evaluation is blocked.

If development passes all gates and reviewer accepts the checkpoint at `5/5`,
the exact same frozen code, statistics, controls, candidate budget, and gates
are run once on the V7 final raw pool.

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
- all per-record matched candidate and control candidate metrics;
- selected matched candidate index and selection reason;
- Pareto dominator types for failed records;
- explicit pass/fail and next action.

Development result artifacts must not include any raw final pool path or final
raw per-subject field.

## Valid Claim If Development And Final Pass

If and only if both development and final pass all gates, the valid claim is:

For this small subject architecture and four clean synthetic behaviors, fixed
stored-probe activation signatures support train-only source-conditioned
representation-space steering in the form of a small deterministic
Pareto-frontier candidate set. On fresh heldout subjects, the matched V7
frontier contains target-behavior candidates that satisfy absolute primary,
centroid, and source-suppression gates and are not Pareto-dominated by
prior-method, non-matched V7, shuffled V7, or random controls.

This would still not prove:

- a single universal steering vector per direction;
- four-behavior functional decoding;
- larger model behavior;
- natural-language-model behavior;
- broad MUAT generality;
- mechanistic equivalence between representation-space movement and weight
  edits.
