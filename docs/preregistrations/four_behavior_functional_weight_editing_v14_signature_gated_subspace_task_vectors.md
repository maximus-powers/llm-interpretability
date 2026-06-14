# Four-Behavior Functional Weight Editing V14 Signature-Gated Subspace Task Vectors Preregistration

Date: 2026-06-12

## Purpose

V13 is valid negative development evidence for a hybrid
signature-support-optimization editor. It transferred target behavior often, but
failed proof-grade pass/Pareto gates and, most importantly, showed almost no
fixed-probe-signature-specific advantage over no-signature and
shuffled-signature optimizer controls.

V14 tests a different, narrower claim:

For the same four clean behaviors and subject architecture, fixed stored-probe
signatures can help select and scale train-only, alignment-canonicalized
source-to-target task-vector edits inside a low-rank weight subspace. The claim
is not that signatures directly decode full weights and not that support-set
optimization alone can edit behavior.

This is still a small-subject, source-label-known, target-label-requested
experiment. It is not source-label inference, not source-free decoding, not
larger-model evidence, not broad MUAT proof, and not arbitrary capability
preservation.

## Literature Basis

V14 is grounded in the literature review documented in
`docs/literature_weight_space_activation_steering_notes.md`.

- Han et al. (2026), *A Survey of Weight Space Learning*,
  https://arxiv.org/abs/2603.10090, motivates separating weight-space
  understanding, representation, and generation claims. V14 tests a narrow
  aligned weight-to-weight editing operator.
- Kaushik et al. (2025), *The Universal Weight Subspace Hypothesis*,
  https://arxiv.org/abs/2512.05117, is motivational evidence for
  low-dimensional weight structure, not direct evidence for this tiny subject
  setting. V14 tests the small-subject analogue by constraining edits to
  train-only low-rank edit subspaces.
- Ilharco et al. (2023), *Editing Models with Task Arithmetic*,
  https://arxiv.org/abs/2212.04089, motivates task-vector baselines and
  target-label-only controls.
- Ainsworth et al. (2022), *Git Re-Basin*,
  https://arxiv.org/abs/2209.04836, motivates treating hidden-neuron
  permutation alignment as a first-class variable.
- Navon et al. (2023), *Equivariant Architectures for Learning in Deep Weight
  Spaces*, https://arxiv.org/abs/2301.12780, Zhou et al. (2023),
  *Permutation Equivariant Neural Functionals*,
  https://arxiv.org/abs/2302.14040, and Dayan et al. (2026), *On the
  Expressive Power of Permutation-Equivariant Weight-Space Networks*,
  https://arxiv.org/abs/2602.01083, motivate symmetry-aware treatment of weight
  spaces and narrow operator claims.
- Wortsman et al. (2022), *Model soups*, https://arxiv.org/abs/2203.05482,
  motivates interpolation and basin-compatibility diagnostics.
- Yadav et al. (2023), *TIES-Merging*,
  https://arxiv.org/abs/2306.01708, motivates sign-conflict and
  magnitude-trimming controls.
- Jordan et al. (2022), *REPAIR*,
  https://arxiv.org/abs/2211.08403, motivates activation-variance/interpolation
  diagnostics because alignment alone can leave variance-collapse barriers.
- Turner et al. (2024), *Steering Language Models With Activation
  Engineering*, https://arxiv.org/abs/2308.10248, is motivational evidence for
  activation-space steering, not direct evidence that stored-probe signatures
  select weight-space task vectors.

The V14 claim must remain narrower than this literature.

## Contamination Policy

V1-V13 preregistrations, development artifacts, final summaries, evidence
reports, and literature notes have been inspected. V14 must not reuse any prior
final raw pool as a V14 train, development, final, optimizer, retrieval,
subspace, or control input.

V14 development may read:

- V14 train raw pool;
- V14 development raw pool;
- V14 combined source-pool audit;
- V14 final redacted source-pool audit.

V14 development must not read, parse, summarize, hash through loaded content, or
evaluate the V14 final raw pool. The final raw path may appear only as a literal
blocked path in guard code and documentation before final authorization.

If V14 development fails, final evaluation is blocked. Any method change after
reviewed development success requires a new preregistration suffix and
invalidates final eligibility for this preregistration.

## Source Pools

V14 uses the same subject architecture, stored probes, behavior suite, and
source-generation acceptance criteria as V9-V13, but with fresh V14-specific
seeds and scopes.

V14 source-pool output directory:

`runs/four_behavior_functional_weight_editing_v14_pools`

Required V14 claim scopes:

- raw train/development/final pools:
  `four_behavior_functional_weight_editing_v14_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v14_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v14_source_pool_audit_surface_only`.

Pool sizes:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior.

Seed schedule:

- train base seed: `72300000`;
- development base seed: `73300000`;
- final base seed: `74300000`;
- behavior stride: `100000`;
- train max attempts per behavior: `128`;
- development/final max attempts per behavior: `64`.

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

Forbidden final-detail fields before final evaluation:
per-subject records, subject IDs, behavior labels, seeds, attempt indices,
signatures, signature hashes, weights, weight hashes, source/support/heldout
margins, attempt/rejection counts, accepted/rejected subject IDs, and per-subject
metrics. Any exposure invalidates the V14 final pool for proof use.

Before opening V14 final raw, final evaluation must validate a current passing
V14 development artifact with exact current values for:

- `claim_scope`:
  `four_behavior_functional_weight_editing_v14_signature_gated_subspace_task_vectors_development`;
- `phase`: `development`;
- `passed`: `true`;
- `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`;
- `editor_method`:
  `signature_gated_low_rank_aligned_task_vectors_v14`;
- current train/development pool hashes;
- current combined-audit and final-redacted-audit hashes;
- current implementation SHA-256;
- current preregistration SHA-256.

If any value differs, final evaluation must fail before opening V14 final raw.

## Fixed Suite And Probe Inputs

V14 uses the deterministic clean behavior suite from
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

V14 uses deterministic stored probes from
`hypernet.paired_contrast.build_digit_probe_examples` with exactly:

- `n_examples = 256`;
- `seed = 20260610`;
- `seq_len = 5`;
- `base = 10`;
- probe examples hash:
  `b156dabece5a9eb58a966271388c8e5479fd308712dcca7b373e0f253e670279`.

Support cases may be used for support-only scale selection. Heldout cases are
proof metrics only. Stored probes are signature-extraction probes only.

## Train-Only Statistics And Alignment

V14 recomputes V9-style representation statistics from accepted V14 train
subjects only:

- global signature mean/std;
- behavior centroids;
- PCA basis;
- residual covariance factors;
- diagonal transport stds;
- primary behavior evaluator;
- V5 calibration coefficients used only inside the frozen V9-style selected
  target-attractor candidate generator.

Layerwise target-to-source alignment uses the V12 Hungarian hidden-unit
alignment logic:

1. For each hidden layer `0..4`, concatenate each hidden unit's incoming weights
   and bias.
2. Build squared Euclidean cost from source units to target units.
3. Add deterministic tie-break
   `1e-9 * (source_neuron_index + 1) * (target_neuron_index + 1)`.
4. Run `scipy.optimize.linear_sum_assignment`.
5. Permute target hidden units and the next layer's incoming columns.

No activation matching, heldout metric, or development metric may affect
alignment.

## Train-Only Edit Subspaces

For every ordered source behavior `a` and target behavior `b`, `a != b`:

1. Let `S_a` be the `64` accepted V14 train subjects with source behavior `a`.
2. Let `T_b` be the `64` accepted V14 train subjects with target behavior `b`.
3. For every ordered pair `(s, t)` in `S_a x T_b`:
   - align `t.weights` to `s.weights` using the alignment rule above;
   - compute raw delta `aligned_t_weights - s.weights`.
4. Stack all `4096` deltas into a matrix `D_ab` of shape `[4096, 345]`.
5. Compute train-only delta mean `mu_ab = D_ab.mean(dim=0)`.
6. Center deltas as `D_ab - mu_ab`.
7. Compute deterministic full SVD with `torch.linalg.svd(..., full_matrices=False)`
   on CPU `float32`.
8. Use fixed rank `16`, not development-selected rank.
9. The edit basis `B_ab` is the first `16` right singular vectors.
10. Project any candidate delta `d` into this pair subspace as
    `mu_ab + B_ab.T @ (B_ab @ (d - mu_ab))`, using row-vector notation in code
    as long as the operation is algebraically equivalent.

Record train-only diagnostics:

- per-direction singular values;
- rank;
- explained variance of rank `16`;
- `mu_ab` norm;
- subspace construction hash.

## Matched V14 Editor

At evaluation time, the matched editor receives exactly:

- heldout source weights from the development/final eval pool;
- heldout source stored-probe signature;
- registered source behavior label;
- requested target behavior label.

For each source-target record:

1. Normalize source signature with V14 train signature mean/std.
2. Compute V9-style selected target-attractor normalized signature candidate for
   `(source_behavior, target_behavior)` using V14 train-only statistics.
3. For each train target subject `t` in the requested target behavior:
   - normalize `t.signature` with V14 train mean/std;
   - compute distance as mean squared difference from the selected target
     candidate over all signature dimensions.
4. Sort train target subjects by `(distance, subject_id)` ascending.
5. Keep top `8` target subjects.
6. Compute weights with softmax over `-distance / 1.0`.
7. For each selected train target subject:
   - align target weights to the eval source weights;
   - compute delta `aligned_target - eval_source_weights`;
   - project the delta into the ordered-pair edit subspace.
8. Matched raw edit direction is the weighted average of these `8` projected
   deltas.
9. Select scalar edit scale from this fixed grid:
   `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`.
10. Scale selection uses support cases only, minimizing:
    `4.0 * target_support_bce + 2.0 * source_support_conflict_bce +
    0.01 * compatible_source_support_logit_mse + 0.0005 * raw_delta_l2_mse`.
11. Ties in scale objective are broken by lower absolute scale, then lower scale.
12. Apply `edited_weights = source_weights + selected_scale * matched_direction`.
13. Evaluate only after scale selection.

No development/final heldout case may affect target-subject selection, subspace
construction, scale selection, or method choice.

## Support-Loss Semantics

Target support BCE:

- inputs are target support positive plus target support negative cases;
- labels are `1.0` for the first `160` target positives and `0.0` for the next
  `160` target negatives;
- compute logits, not sigmoid probabilities;
- use `binary_cross_entropy_with_logits` with default `reduction="mean"`.

Source support compatible/conflict split:

- inputs are source support positive plus source support negative cases;
- source and target labels are computed with fixed `PREDICATES`;
- compatible cases have equal source and target labels;
- conflict cases have different source and target labels;
- support compatible/conflict counts must match this table exactly.

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

Conflict BCE uses conflict support cases with target predicate labels.
Compatible MSE compares edited logits to original source logits on compatible
support cases. Raw-delta L2 MSE is mean squared value of
`scale * direction` over all raw flat weights.

## Compatible/Conflict Heldout Evaluation

V14 uses the same conflict-aware heldout evaluation semantics as V12/V13.

For each source behavior's heldout positive and negative cases, compute source
and target predicate labels directly with fixed `PREDICATES`. Compatible source
cases have equal source and target labels; conflict source cases have different
labels. Model labels for conflict accuracy use `sigmoid(logit) >= 0.5`.

Compatible source-output MSE is mean squared error between edited logits and
original source logits on compatible heldout source cases. Conflict target-label
accuracy is the fraction of conflict heldout source cases whose edited model
label equals the target predicate label. Conflict improvement is edited conflict
accuracy minus original source model conflict accuracy.

V14 must use this exact compatible/conflict count table for all 12 ordered
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

Every source-target record must output matched edit metrics and exactly `30`
controls: `14` non-random controls plus `16` deterministic random controls.
Matched edit is not counted as a control.

Required non-random controls:

1. `no_edit`: source weights.
2. `v13_no_signature_support_optimizer`: exact V13 no-signature optimizer
   control, with source-initialized full-weight AdamW for exactly `130` updates,
   `lr=0.03`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.0`,
   `amsgrad=False`, target support BCE weight `4.0`, source support conflict
   BCE weight `2.0`, compatible source support logit MSE weight `0.01`,
   source-weight L2 MSE weight `0.0005`, signature loss weight `0.0`, and final
   optimizer step only. No heldout case may affect this control.
3. `aligned_full_nearest_target_retrieval`: V12-style nearest target retrieval
   recomputed from V14 train subjects only, aligned to eval source.
4. `aligned_interpolation_alpha_0.975`: V12-style aligned interpolation
   recomputed from V14 train subjects only.
5. `target_label_centroid_task_vector`: `mu_ab` scaled by the same support-only
   scale objective; no signature weighting.
6. `nearest_signature_task_vector`: top-1 train target by selected target
   signature distance, projected and scale-selected.
7. `uniform_average_task_vector`: uniform average of all `64` train target
   projected deltas, scale-selected.
8. `shuffled_signature_weighted_task_vector`: same as matched, but selected
   target-attractor signature is computed for a deterministic shuffled behavior
   selected from behaviors neither source nor requested target.
9. `source_signature_weighted_task_vector`: same as matched, but weighting target
   signatures by distance to the normalized source signature instead of selected
   target-attractor signature.
10. `ties_trimmed_sign_task_vector`: uniform train target projected deltas with
    TIES-style coordinate trimming/sign election before scale selection.
11. `repair_style_aligned_interpolation`: aligned interpolation at `0.975` with
    the deterministic REPAIR-style activation-variance rescaling defined below.
12. `random_same_rank_subspace_task_vector`: matched unprojected weighted delta
    projected into a deterministic random orthonormal rank-`16` subspace,
    scale-selected.
13. `random_neuron_permutation_task_vector`: same as matched, but target train
    subjects are aligned to eval source with deterministic random hidden-unit
    permutations rather than Hungarian alignment.
14. `no_alignment_task_vector`: same as matched, but target train subjects are
    not aligned before delta computation.

Random controls:

- `16` deterministic random raw weight deltas around source weights;
- random seed is
  `stable_hash_json([subject_id, source_behavior, target_behavior,
  "functional_weight_editing_v14_random_weight_delta"])`;
- every random delta is normalized to matched raw edit norm and evaluated as a
  control.

Shuffled target selection:

1. Enumerate behaviors in lexicographic order.
2. Keep behaviors not equal to source and not equal to requested target.
3. Compute
   `stable_hash_json([subject_id, source_behavior, target_behavior,
   "functional_weight_editing_v14_shuffled_signature_target"])`.
4. Convert first 16 hex characters to integer and take modulo remaining behavior
   count.

Control Pareto-dominates matched only if it is weakly better on target heldout
margin and compatible source-output MSE, with at least one strict improvement.

## TIES And REPAIR Diagnostics

TIES-style control:

- candidate deltas are the `64` projected train target deltas for the ordered
  source-target pair and eval source;
- for each coordinate, trim values with absolute magnitude below that
  coordinate's 20th percentile absolute magnitude across the `64` deltas;
- elect sign by sign of the sum of remaining values; ties produce zero;
- average only remaining values whose sign equals elected sign;
- coordinates with no elected values become zero;
- apply support-only scale selection.

REPAIR-style control:

- evaluate hidden activation mean/std on the fixed support cases for source,
  aligned retrieved target, and aligned interpolation at `alpha=0.975`;
- support inputs are the union of source support positives, source support
  negatives, target support positives, and target support negatives, de-duplicated
  in lexicographic sequence order;
- for each hidden layer `0..4` and neuron `0..7`, compute post-GELU activation
  mean and population std (`unbiased=False`) for source, aligned target, and
  interpolation;
- desired mean is the arithmetic average of source and aligned target means;
- desired std is the arithmetic average of source and aligned target stds;
- repair ratio is `desired_std / interpolation_std.clamp_min(1e-6)`, clipped to
  `[0.25, 4.0]`;
- process hidden layers in order `0..4`; for layer `l`, multiply the outgoing
  columns from hidden layer `l` into layer `l+1` by repair ratio and adjust the
  next layer bias by `next_weight_before_scaling @ (desired_mean -
  repair_ratio * interpolation_mean)`;
- for layer `4`, apply the same outgoing-column and bias adjustment to the final
  output layer;
- no heldout activations may be used for repair statistics;
- record per-layer mean repair ratio and interpolation/source-target endpoint
  std ratios as diagnostics.

Random same-rank subspace control:

- generate a deterministic Gaussian matrix of shape `[345, 16]` with seed
  `stable_hash_json([subject_id, source_behavior, target_behavior,
  "functional_weight_editing_v14_random_same_rank_subspace"])`;
- compute a QR decomposition and use the first `16` orthonormal columns;
- project matched unprojected weighted delta into this random subspace;
- use the same support-only scale grid and tie-breaks as matched.

Random neuron-permutation control:

- for each selected train target subject and hidden layer, generate a
  deterministic random permutation with seed
  `stable_hash_json([subject_id, source_behavior, target_behavior,
  target_subject_id, layer_index,
  "functional_weight_editing_v14_random_neuron_permutation"])`;
- apply those permutations instead of Hungarian alignment before delta
  computation;
- use the same projection, weighting, and support-only scale selection as
  matched.

## Individual Record Pass

A matched record passes only if all are true:

- primary behavior prediction is the requested target behavior;
- target heldout margin is `> 0.20`;
- compatible source-output MSE is lower than
  `aligned_full_nearest_target_retrieval`;
- conflict target-label accuracy is `>= 0.70`;
- conflict target-label accuracy improves over source by `>= 0.20`;
- no control Pareto-dominates matched;
- matched target margin is at least `0.02` greater than
  `target_label_centroid_task_vector` or matched compatible MSE is at least
  `5.0` lower;
- matched target margin is at least `0.02` greater than
  `source_signature_weighted_task_vector` or matched compatible MSE is at least
  `5.0` lower;
- matched target margin is at least `0.05` greater than
  `shuffled_signature_weighted_task_vector` or matched compatible MSE is at
  least `5.0` lower;
- matched target margin is at least `0.02` greater than
  `v13_no_signature_support_optimizer` or matched compatible MSE is at least
  `5.0` lower.

These last gates are mandatory V13-lesson gates. If controls match or dominate
matched, V14 is negative for the fixed-probe-signature-specific claim even if
target behavior improves.

## Development Gates

All gates must pass to authorize final evaluation:

- aggregate `n == 288`;
- every ordered source-target direction has `n == 24`;
- aggregate individual pass rate `>= 0.85`;
- every ordered source-target direction has individual pass rate `>= 0.70`;
- aggregate target-behavior prediction rate `>= 0.95`;
- every ordered source-target direction has target-behavior prediction rate
  `>= 0.90`;
- aggregate mean matched target heldout margin `> 0.50`;
- every ordered source-target direction has mean target heldout margin `> 0.20`;
- aggregate mean conflict target-label accuracy `>= 0.85`;
- every ordered source-target direction has mean conflict target-label accuracy
  `>= 0.70`;
- aggregate mean conflict target-label accuracy improvement `>= 0.50`;
- every ordered source-target direction has mean conflict target-label accuracy
  improvement `>= 0.20`;
- aggregate mean aligned-full-minus-matched compatible source-output MSE
  `> 10.0`;
- every ordered source-target direction has mean aligned-full-minus-matched
  compatible source-output MSE `> 0.0`;
- aggregate Pareto-undominated rate `>= 0.85`;
- every ordered source-target direction has Pareto-undominated rate `>= 0.70`;
- aggregate matched-minus-target-label-centroid target margin mean `> 0.02` or
  centroid-minus-matched compatible MSE mean `> 2.0`;
- aggregate matched-minus-source-signature target margin mean `> 0.02` or
  source-signature-minus-matched compatible MSE mean `> 2.0`;
- aggregate matched-minus-shuffled-signature target margin mean `> 0.05` or
  shuffled-signature-minus-matched compatible MSE mean `> 2.0`;
- aggregate matched-minus-v13-no-signature target margin mean `> 0.02` or
  v13-no-signature-minus-matched compatible MSE mean `> 2.0`;
- every record includes exactly `30` controls;
- every record includes exactly `16` random norm-matched controls;
- every required control type appears exactly once.

If any development gate fails, result is negative development evidence and final
evaluation is not authorized.

## Final Gates

Final evaluation is one-shot and uses the exact reviewed method and passing
development authorization artifact. It must pass the same gates as development,
with no threshold weakening after seeing final results.

If final fails, it is recorded as failed final evidence and no additional V14
final rerun may be used as proof without a new preregistered experiment.

## Reviewer Checkpoints

V14 requires reviewer confidence `5/5` after each result-producing step:

1. preregistration review, including literature support;
2. implementation and helper-test review;
3. source-pool audit review;
4. development-result review;
5. final-result review if final is authorized.

Reviewer prompts must ask specifically about literature support, data leakage,
final-pool exposure, support-vs-heldout separation, hidden adaptivity,
permutation/alignment controls, target-label-only and no-signature controls,
TIES/REPAIR control validity, metric/gate mismatch, and whether the claim is
narrower than the cited theory.
