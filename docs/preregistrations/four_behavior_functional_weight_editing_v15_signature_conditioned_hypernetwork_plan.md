# V15 Plan: Signature-Conditioned Hypernetwork Functional Editing

Status: planning draft for reviewer approval. Do not implement or run source
pools until this plan is reviewed at 5/5 and converted into a formal
preregistration.

## Motivation

V13 was useful but inconclusive: support optimization often changed behavior,
yet did not cleanly prove a fixed-signature advantage over no-signature or
shuffled-signature controls. V14 was a valid negative result: its
signature-gated rank-16 task-vector method produced 0/288 individual development
passes, target prediction 25/288, and low Pareto-undominated rate. The failure
suggests the signature-to-edit map is not well modeled as a single linear
subspace/task-vector operation at this scale.

V15 will test a more expressive but still auditable mechanism: a small
train-only hypernetwork/editor that receives source weights, source fixed-probe
signature, target fixed-probe signature, and source/target labels, then emits a
weight delta. The hypothesis is narrow: fixed probe activation signatures contain
target-specific information that helps a learned editor produce functional
behavior edits on heldout development subjects beyond target label alone,
source-signature controls, shuffled-signature controls, retrieval, and
no-signature support optimization.

## Literature Support

- Ha, Dai, and Le (2016), *HyperNetworks*,
  https://arxiv.org/abs/1609.09106. Hypernetworks generate weights for another
  model, directly supporting the idea of a learned conditional weight editor.
- Klocek et al. (2023), *A Brief Review of Hypernetworks in Deep Learning*,
  https://arxiv.org/html/2306.06955v3. The review discusses data-conditioned
  and task-conditioned hypernetworks; V15 treats fixed-probe activation
  signatures as the conditioning data.
- Przewiezlikowski et al. (2022), *Few-Shot Adaptation of Deep Models with
  Hypernetworks*, https://arxiv.org/html/2205.15745v3. This motivates replacing
  per-task gradient adaptation with a learned generator conditioned on task
  information.
- Zhou et al. (2023), *Permutation Equivariant Neural Functionals*,
  https://arxiv.org/abs/2302.14040, and Dayan et al. (2026), *On the Expressive
  Power of Permutation-Equivariant Weight-Space Networks*,
  https://arxiv.org/abs/2602.01083. These support nonlinear weight-space
  functionals and highlight permutation symmetry risks. V15 will use explicit
  Hungarian alignment and random-permutation controls rather than pretending a
  flattened editor is symmetry-complete.
- Kaushik et al. (2025), *The Universal Weight Subspace Hypothesis*,
  https://arxiv.org/abs/2512.05117. This remains motivational for low-dimensional
  regularization, but V14 showed that a purely linear low-rank edit was too weak
  here. V15 will use low-rank/weight penalties as regularizers, not as the core
  mechanism.
- Postmus and Abreu (2024), *Steering Large Language Models using Conceptors*,
  https://arxiv.org/abs/2410.16314, and Turner et al. (2024), *Activation
  Addition*, https://arxiv.org/abs/2308.10248. These motivate activation-derived
  steering signals, but V15 must test the bridge to weight editing directly.

## Frozen Data Policy

V15 proof-grade evaluation requires fresh V15 pools. V14 pools may be referenced
only as prior negative evidence and must not be reused for proof-eligible V15
development or final evaluation.

Fresh V15 pool directory:

`runs/four_behavior_functional_weight_editing_v15_pools`

Pool configuration:

- train base seed `75300000`, `64` accepted subjects per behavior, maximum
  `128` attempts per behavior;
- development base seed `76300000`, `24` accepted subjects per behavior,
  maximum `64` attempts per behavior;
- final base seed `77300000`, `24` accepted subjects per behavior, maximum `64`
  attempts per behavior;
- behavior stride `100000`;
- behavior order is the project `PATTERNS` order used by the V14 script:
  `sorted_ascending`, `sorted_descending`, `has_majority`, `mountain_pattern`.

Claim scopes:

- raw source pools:
  `four_behavior_functional_weight_editing_v15_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v15_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v15_source_pool_audit_surface_only`;
- development result:
  `four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork_development`;
- final result:
  `four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork_final`.

The final redacted audit allowlist is exactly:

- `behavior_suite_hashes`;
- `candidate_pool_summary_hash`;
- `claim_scope`;
- `config_hash`;
- `pool`;
- `pool_file_sha256`;
- `pool_redacted_payload_sha256`;
- `summary`;
- `summary_payload_sha256`.

The `combined_audit.pool_summaries.final` allowlist is separately and exactly:

- `accepted_counts_by_behavior`;
- `pool_file_sha256`;
- `pool_redacted_payload_sha256`.

Any public final audit/result surface containing final `records`,
`subject_ids`, `accepted_subject_ids`, `rejected_subject_ids`, `weights`,
`weights_hash`, `signature`, `signature_hash`, per-subject margins, per-attempt
details, or other final subject-level details invalidates V15. Any additional
field in `combined_audit.pool_summaries.final` before authorized final
evaluation invalidates the V15 final pool for proof use.

Final authorization is hash-bound: the final phase may run only if the
development artifact has `passed: true`,
`next_action: eligible_for_one_shot_final_eval_without_method_changes`, and
matches the exact train pool hash, development pool hash, combined audit hash,
final redacted audit hash, implementation hash, preregistration hash, method
string, and development claim scope. Otherwise final raw remains sealed.

## Method

Train a deterministic conditional editor on train subjects only.

### Training Examples

For every train source subject and every requested target behavior different
from the source behavior:

1. Compute the source normalized signature `z_source`.
2. Use every `64` train target subjects from the requested target behavior,
   sorted by `subject_id`. There is no target selection or nearest-neighbor
   filtering in training.
3. The target conditioning signature is the individual train target subject's
   normalized fixed-probe signature, computed with train-only signature mean and
   std.
4. Align each selected target subject to the source subject with the existing
   Hungarian hidden-neuron alignment.
5. The supervised target is the aligned target weights and its fixed-probe
   signature.

No development or final subject may enter training, normalization fitting,
signature target construction, or hyperparameter selection.

Training pair count is fixed:

`4 source behaviors * 64 source subjects * 3 target behaviors * 64 target subjects = 49152`

Training pair order is lexicographic over:

`(source_behavior_index, source_subject_id, target_behavior_index, target_subject_id)`

where `target_behavior_index` skips the source behavior.

### Editor Architecture

`SignatureConditionedDeltaHypernetwork`:

- input:
  - source flat weights, 345 dims;
  - normalized source signature, 560 dims;
  - normalized target signature, 560 dims;
  - source behavior one-hot, 4 dims;
  - target behavior one-hot, 4 dims;
  - signed behavior-pair one-hot, 12 dims;
- MLP hidden sizes: `[768, 768, 384]`, GELU, LayerNorm after each hidden layer;
- output: 345-dim raw delta;
- final edit: `source_weights + scale * delta`;
- diagnostic learned scale head constrained to `[0.0, 1.5]` by sigmoid;
- primary matched edit ignores the learned scale head and uses only the fixed
  support-only scale grid `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`.
  Development records must report the diagnostic learned scale, but it cannot
  choose the primary edit.

### Training Loss

For train pairs only:

- aligned target weight MSE: `1.0`;
- target support BCE: `4.0`;
- source-support conflict BCE with target labels: `2.0`;
- compatible source support logit MSE: `0.01`;
- differentiable fixed-probe signature MSE to target signature: `0.05`;
- source-weight L2 MSE: `0.0005`;
- delta norm penalty: `0.0001`.

Train for a fixed `3000` AdamW steps, batch size `64`, learning rate `1e-3`,
betas `(0.9, 0.999)`, weight decay `1e-4`, seed `20260615`. No early stopping
on development.

Training mechanics are frozen:

- device: CPU only;
- dtype: `torch.float32`;
- all editor initialization and batch sampling use `torch.Generator` seed
  `20260615`;
- the precomputed training-pair table is sorted as specified above;
- each optimization step samples `64` integer indices from the full pair table
  with replacement via that generator;
- no epoch-level reshuffling and no dev-based early stopping;
- only the final step checkpoint is saved and evaluated;
- gradient clipping norm `10.0`;
- source flat weights are normalized in the editor input by train-only
  `weight_mean` and `weight_std.clamp_min(1e-6)`;
- source and target signatures are normalized by train-only `sig_mean` and
  `sig_std.clamp_min(1e-6)`;
- behavior one-hot and pair one-hot use the frozen behavior order above;
- train statistics hash includes signature normalization, weight normalization,
  training-pair IDs, editor hyperparameters, and control-model hyperparameters.

### Development Matched Edit

For each development subject and requested target behavior:

1. Compute the V9 target-attractor candidate signature using train-only V9
   statistics/classifier as in V13/V14.
2. Feed source weights, source normalized signature, selected target-attractor
   normalized signature, source one-hot, target one-hot, and pair one-hot into
   the editor.
3. Apply support-only scale-grid selection on the generated delta using the same
   objective as V14: `4 target BCE + 2 conflict BCE + 0.01 compatible source
   logit MSE + 0.0005 source L2`.
4. Evaluate only on heldout metrics after scale selection.

## Controls

Every record must include matched metrics and exactly `30` controls: `14`
non-random controls plus `16` random controls. Matched is not a control.

All hypernetwork controls are separately trained CPU-only models with fixed
final checkpoints. They use the same architecture, pair table, optimizer,
batch sampling, seed base, scale-grid evaluation, and checkpoint policy as the
primary editor unless specified below. Control seeds are
`20260615 + control_index`, where control index is the 1-based order in this
section.

1. `no_edit`.
2. `v13_no_signature_support_optimizer`.
3. `v14_signature_gated_task_vector`.
4. `aligned_full_nearest_target_retrieval`.
5. `aligned_interpolation_alpha_0.975`.
6. `target_label_only_hypernetwork`: replace target signature with train target
   behavior centroid signature during train and eval; signature MSE target is
   also that centroid.
7. `source_signature_hypernetwork`: replace target signature with source
   signature during train and eval; signature MSE target is source signature.
8. `shuffled_signature_hypernetwork`: replace target signature with deterministic
   shuffled behavior centroid signature during train; during eval use the
   deterministic shuffled behavior V9 target-attractor signature. Signature MSE
   target is the replacement signature.
9. `nearest_train_target_signature_hypernetwork`: replace target signature with
   nearest train target signature to the primary eval target-attractor signature
   during eval; during train this is identical to the paired target subject
   signature and therefore uses the primary training data.
10. `random_signature_hypernetwork`: deterministic random normalized signature
   generated from
   `stable_hash_json([subject_id, source, target, "v15_random_signature"])`
   during eval and from
   `stable_hash_json([source_subject_id, target_subject_id, "v15_train_random_signature"])`
   during train; signature MSE target is the random signature.
11. `random_neuron_permutation_hypernetwork`: same editor output evaluated after
   a separately trained model whose supervised aligned target weights are built
   with deterministic random hidden-neuron permutations instead of Hungarian
   alignment. Functional losses remain unchanged.
12. `target_weight_mse_only_hypernetwork`: same architecture trained with only
   aligned target weight MSE `1.0`, source L2 `0.0005`, and delta norm
   `0.0001`; all functional and signature losses are zero.
13. `functional_only_hypernetwork`: same architecture trained without target
   signature MSE; all other primary losses are unchanged.
14. `signature_only_hypernetwork`: same architecture trained without functional
   support losses; aligned target weight MSE, signature MSE, source L2, and
   delta norm remain enabled.
15. `random_norm_matched_weight_delta:00..15`.

The primary matched editor is trained with seed `20260615`. Hypernetwork control
models use seeds:

- target-label-only: `20260621`;
- source-signature: `20260622`;
- shuffled-signature: `20260623`;
- nearest-train-target-signature: `20260624`;
- random-signature: `20260625`;
- random-neuron-permutation: `20260626`;
- target-weight-MSE-only: `20260627`;
- functional-only: `20260628`;
- signature-only: `20260629`.

## Gates

Use the same core V14 gates:

- aggregate `n = 288`;
- target prediction rate `>= 0.95`;
- individual all-gate pass rate `>= 0.85`;
- Pareto-undominated rate `>= 0.85`;
- mean target margin `> 0.50`;
- aggregate conflict target accuracy `>= 0.85`;
- aggregate conflict target accuracy improvement `>= 0.50`;
- every direction has `n = 24`, target prediction `>= 0.90`, individual pass
  rate `>= 0.70`, Pareto-undominated `>= 0.70`, and mean target margin `> 0.20`.

Signature-specific gates:

- matched must beat `target_label_only_hypernetwork`, `source_signature_hypernetwork`,
  and `shuffled_signature_hypernetwork` by either target margin or compatible
  MSE at both per-record and aggregate levels;
- matched must also beat `v13_no_signature_support_optimizer` by either target
  margin or compatible MSE at both per-record and aggregate levels. This is a
  hard gate. If V15 cannot beat the no-signature optimizer, no final evaluation
  is authorized.

Formal record gates:

- primary behavior prediction equals the requested target;
- matched target margin `> 0.20`;
- matched compatible source-output MSE is lower than
  `aligned_full_nearest_target_retrieval`;
- matched conflict target accuracy `>= 0.70`;
- matched conflict target accuracy improvement over source `>= 0.20`;
- no control Pareto-dominates matched on heldout target margin and compatible
  source-output MSE;
- matched target margin is at least `0.02` greater than
  `target_label_only_hypernetwork`, or matched compatible MSE is at least `5.0`
  lower;
- matched target margin is at least `0.02` greater than
  `source_signature_hypernetwork`, or matched compatible MSE is at least `5.0`
  lower;
- matched target margin is at least `0.05` greater than
  `shuffled_signature_hypernetwork`, or matched compatible MSE is at least `5.0`
  lower;
- matched target margin is at least `0.02` greater than
  `v13_no_signature_support_optimizer`, or matched compatible MSE is at least
  `5.0` lower.

Formal aggregate gates:

- exactly `288` records;
- exactly `30` controls per record;
- exactly `16` random controls per record;
- every required non-random control type listed above is present exactly once;
- aggregate individual pass rate `>= 0.85`;
- aggregate target prediction rate `>= 0.95`;
- aggregate Pareto-undominated rate `>= 0.85`;
- mean matched target margin `> 0.50`;
- mean full-retrieval-minus-matched compatible MSE `> 10.0`;
- aggregate conflict target accuracy `>= 0.85`;
- aggregate conflict target accuracy improvement `>= 0.50`;
- aggregate matched-vs-target-label advantage: target-margin advantage `> 0.02`
  or compatible-MSE advantage `> 2.0`;
- aggregate matched-vs-source-signature advantage: target-margin advantage
  `> 0.02` or compatible-MSE advantage `> 2.0`;
- aggregate matched-vs-shuffled-signature advantage: target-margin advantage
  `> 0.05` or compatible-MSE advantage `> 2.0`;
- aggregate matched-vs-v13-no-signature advantage: target-margin advantage
  `> 0.02` or compatible-MSE advantage `> 2.0`.

Per-direction gates:

- exactly `24` records per ordered source-target direction;
- individual pass rate `>= 0.70`;
- target prediction rate `>= 0.90`;
- Pareto-undominated rate `>= 0.70`;
- mean matched target margin `> 0.20`;
- mean full-retrieval-minus-matched compatible MSE `> 0.0`;
- conflict target accuracy `>= 0.70`;
- conflict target accuracy improvement `>= 0.20`.

Final-blocking semantics:

- if any development gate fails, write
  `next_action: log_negative_development_result_do_not_open_final_raw`;
- final raw remains sealed;
- no method, threshold, control, seed, pool, or implementation change may be made
  and then evaluated on the same final raw;
- only a development artifact that passes all gates and receives reviewer
  confidence 5/5 authorizes a one-shot final run.

## Reviewer Questions

1. Does the cited literature actually justify moving from V14 linear task
   vectors to a nonlinear signature-conditioned hypernetwork, or is this too
   unconstrained for the evidence claim?
2. Should V15 reuse V14 train/development pools as a development-only iteration,
   or must it generate fresh V15 pools before implementation?
3. Is `v13_no_signature_support_optimizer` a hard gate or a reported adversarial
   benchmark? What claim is permitted under each choice?
4. Are the target-label-only, source-signature, shuffled-signature, and
   functional-only/signature-only hypernetwork controls sufficient to detect
   data leaks and label-only shortcuts?
5. Does the training-pair construction leak target behavior through aligned
   target weights in a way that invalidates the fixed-signature claim, or is it
   legitimate supervised train-only hypernetwork learning?
6. Are the proposed training losses and 3000 fixed steps too flexible relative
   to the small train pool, and should the architecture be smaller or controls
   stricter?

## Expected Interpretation

- Passing V15 development would support only a narrow claim: on small MLP
  subject models, train-only fixed-probe signatures can condition a learned
  editor that changes heldout behavior better than label-only and shuffled
  controls.
- Failing V15 would suggest that the current fixed-probe signatures are not
  enough for robust functional weight editing, or that the needed architecture
  must be permutation-equivariant rather than flattened/aligned.
