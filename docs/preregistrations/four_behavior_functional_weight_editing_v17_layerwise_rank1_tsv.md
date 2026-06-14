# V17 Formal Preregistration: Signature-Conditioned Layerwise Rank-1 / Task-Singular Weight Editing

Status: formal preregistration for implementation after reviewer approval.

## Planned Constants And Scopes

- script: `train_four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv.py`
- pool directory: `runs/four_behavior_functional_weight_editing_v17_pools`
- output directory: `runs/four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv`
- method: `signature_conditioned_layerwise_rank1_tsv_v17`
- source pool scope: `four_behavior_functional_weight_editing_v17_source_pool`
- source audit scope: `four_behavior_functional_weight_editing_v17_source_pool_construction`
- final redacted scope:
  `redacted_final_functional_weight_editing_v17_source_pool_audit_surface_only`
- development claim scope:
  `four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv_development`
- final claim scope:
  `four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv_final`
- signature top-k: `8`
- signature temperature: `1.0`
- rank grid: `[1, 2, 4, 8]`
- task scale grid: `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25]`
- activation-rank1 scale grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- layer masks: `hidden_only`, `all_layers`
- rank1 ridge: `1e-4`
- random controls per record: `16`

## Motivation

V16 was a valid negative development result: output-layer-only conceptor compilation did
not transfer enough functional behavior and was dominated by no-signature and random
controls. V17 will test a stronger but still small-scale claim: fixed probe-set activation
signatures can choose and parameterize layer-structured low-rank edits, rather than only an
output-layer rewrite.

This narrows the hypothesis to a falsifiable small model setting:

> A fixed activation signature can identify target-behavior structure well enough to
> synthesize a layerwise low-rank weight edit that changes the source model's function
> toward a target behavior while preserving source-compatible behavior better than
> signature-free and random controls.

## Literature Support

- Steer2Edit argues that activation steering vectors can be translated into component-level
  rank-1 weight edits, with the edit's output direction aligned to the steering signal and
  the input direction acting as a trigger. This directly motivates replacing V16's
  output-layer-only compilation with per-layer rank-1 edits that write where the activation
  steering signal is observed: <https://arxiv.org/html/2602.09870v2>.
- Conceptor steering motivates using activation covariance/ellipsoid structure rather than
  only mean shifts, but V16 showed that a pure output-layer conceptor was insufficient in
  this subject-model setting. V17 keeps conceptor-style activation summaries as optional
  metadata/controls, not as the only edit path: <https://arxiv.org/html/2410.16314v4>.
- Task Singular Vectors show that treating task vectors as flat parameter vectors can hide
  layer structure and task interference; layerwise matrix/SVD structure can isolate useful
  task-specific directions. V17 therefore builds per-layer low-rank task deltas rather than
  projecting the entire 345-vector at once:
  <https://openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf>.
- The Universal Weight Subspace Hypothesis supports looking for shared low-dimensional
  layerwise parametric structure across trained models, but it also warns that the useful
  subspace is architecture-specific. V17 uses fresh same-architecture pools and train-only
  layerwise bases: <https://arxiv.org/abs/2512.05117>.
- Recent weight-space learning surveys frame this as weight-space representation and
  generation: learned or structured descriptors of weights can support function prediction,
  editing, and synthesis. V17 explicitly evaluates generation/editing rather than only
  prediction: <https://arxiv.org/html/2603.10090v1>.
- HyperNet Fields motivate amortizing weight synthesis while respecting optimization
  trajectories, but V17 avoids another unconstrained full-vector hypernetwork after V15 by
  using closed-form low-rank candidates plus support-set selection:
  <https://arxiv.org/html/2412.17040v2>.

## Fresh Data Protocol

V17 must generate fresh train/development/final pools with new seeds:

- train base seed: `81300000`
- development base seed: `82300000`
- final base seed: `83300000`
- behavior stride: `100000`
- train accepted per behavior: `64`
- development accepted per behavior: `24`
- final accepted per behavior: `24`

The final raw file remains sealed. Development may read train raw, development raw,
combined audit, and final redacted audit only. Final may be opened only if development
passes all preregistered gates and a reviewer confirms authorization.

### Final Audit Redaction Contract

`combined_audit.pool_summaries.final` may contain only:

- `accepted_counts_by_behavior`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`

`final_redacted_audit.json` may contain only these top-level keys:

- `behavior_suite_hashes`
- `candidate_pool_summary_hash`
- `claim_scope`
- `config_hash`
- `pool`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`
- `summary`
- `summary_payload_sha256`

Its `summary` may contain only:

- `accepted_counts_by_behavior`
- `max_selected_train_vs_heldout_overlap_count`

Forbidden keys are inherited from V16's recursive forbidden-final-detail scanner and must
include, at minimum: `records`, `record`, `subject_id`, `subject_ids`, `seed`, `seeds`,
`weights`, `weights_hash`, `weights_hashes`, `signature`, `signature_hash`,
`signature_hashes`, `attempt_index`, `accepted_subject_ids`, per-subject metrics, and any
raw margin lists. The recursive scanner must fail closed on unknown final-detail fields.

Development authorizes final only when all of these exact fields match:

- `passed == true`
- `phase == "development"`
- `next_action == "eligible_for_one_shot_final_eval_without_method_changes"`
- `claim_scope == DEVELOPMENT_SCOPE`
- `editor_method == "signature_conditioned_layerwise_rank1_tsv_v17"`
- `train_pool_sha256 == sha256(train_subjects.json)`
- `eval_pool_sha256 == sha256(development_subjects.json)`
- `combined_audit_sha256 == sha256(combined_audit.json)`
- `final_redacted_audit_sha256 == sha256(final_redacted_audit.json)`
- `formal_prereg_sha256 == sha256(formal prereg file)`
- `implementation_sha256 == sha256(V17 script)`
- `stats_sha256 == sha256(train stats artifact)`

If any field differs, final evaluation is forbidden.

## Method

Subject flat-weight layout is fixed at 345 parameters:

- hidden layer 0: weight `0:40` as `(8,5)`, bias `40:48`
- hidden layer 1: weight `48:112` as `(8,8)`, bias `112:120`
- hidden layer 2: weight `120:184`, bias `184:192`
- hidden layer 3: weight `192:256`, bias `256:264`
- hidden layer 4: weight `264:328`, bias `328:336`
- output layer: weight `336:344` as `(1,8)`, bias `344`

V17 will reuse existing pack/unpack, support-loss, behavior-margin, and heldout metric
helpers from V14/V15/V16 where possible.

### Train-Only Statistics

For each source-target direction:

1. Normalize train signatures using train-only mean/std.
2. Sort train source records and target records by `subject_id`.
3. Build all ordered cross-behavior training pairs: `64 source records * 64 target
   records = 4096` pairs per source-target direction. No same-index or nearest-only
   pairing is used for basis construction.
4. For each pair, align the target record to the source record with the existing V14
   Hungarian neuron alignment. Alignment tie-breaks are inherited from V14's
   deterministic `linear_sum_assignment` call and sorted record order.
5. Build a full aligned delta: `aligned_target_weights - source_weights`.
6. For each layer independently, unpack the delta and flatten only that layer component:
   `weight_0` through `weight_5` and `bias_0` through `bias_5`.
7. For each direction and layer component, stack the 4096 flattened component deltas in
   pair order and compute CPU float32 SVD over the centered matrix. Store mean component delta,
   right-singular basis rows, singular values, rank, explained variance, pair count, and
   hashes.
8. SVD basis rows are sign-canonicalized for deterministic stats artifacts: for each basis
   row, find the first element with absolute value greater than `1e-8`; if that element is
   negative, multiply the row by `-1`. If no element exceeds `1e-8`, leave the row unchanged.
   Singular rows are stored in descending singular-value order as returned by
   `torch.linalg.svd`.
9. Store only train-derived bases and hashes in the stats artifact.

### Matched Edit Candidate

For each development source subject and target behavior:

1. Compute the source normalized signature.
2. Select target train records only from the requested target behavior.
3. For each target record, compute normalized signature distance:
   `mean((z_target - z_source)^2)`.
4. Sort candidates by `(distance, subject_id)`, take top `k=8`, and compute weights:
   `softmax(-distance / 1.0)`.
5. Record only `subject_id_hash`, distance, rank order, and weight in artifacts.
6. For each selected target, align target weights to source weights, compute its delta, and
   form the weighted sum delta in the deterministic top-k order.
7. Project each layer component of the weighted delta into that direction/layer's
   train-only SVD basis for a candidate rank. Component projection is:
   `mean_delta + ((delta - mean_delta) @ basis[:rank].T) @ basis[:rank]`.
8. Build an activation rank-1 candidate for each hidden layer:
   - output direction: weighted target hidden mean minus source hidden mean at that layer,
     computed on the fixed probe set.
   - input trigger direction: source previous-layer/input mean on the same probe set.
   - rank-1 weight update: `outer(output_direction, input_direction) /
     (||input_direction||^2 + ridge)`.
9. Hidden activations are post-GELU layer outputs from
   `hidden_activations_flat_batch`. Previous-layer inputs are: raw probe input for layer 0
   and post-GELU source hidden activations from layer `i-1` for layer `i`.
10. Target activation means are computed from aligned target weights after Hungarian
    alignment to source. The weighted target hidden mean uses the same top-k signature
    weights as the layerwise delta.
11. Bias edit for hidden layer `i` is `activation_scale * output_direction`; weight edit is
    `activation_scale * outer(output_direction, input_direction) / denom`. The output layer
    receives TSV projection only; no activation-rank1 update is applied to the output layer.
12. Combine candidates by the preregistered grid:
   - task-singular scale: `{0.0, 0.25, 0.5, 0.75, 1.0, 1.25}`
   - activation-rank1 scale: `{0.0, 0.25, 0.5, 0.75, 1.0}`
   - rank: `{1, 2, 4, 8}`
   - layer mask: `hidden_only`, `all_layers`
13. `hidden_only` applies TSV and activation-rank1 only to layers 0-4 and their biases.
    `all_layers` applies TSV to layers 0-5 and biases 0-5, while activation-rank1 still
    applies only to layers 0-4.
14. Select the candidate minimizing the train-support objective defined below.
15. Tie-break deterministically by objective, smaller edit norm, lower rank, smaller task
    scale, smaller activation scale, layer mask order (`hidden_only` before `all_layers`),
    and lexical metadata.

The method may use target labels and support labels in development selection, as prior
functional-editing versions do, but it must also include signature-ablation controls so any
positive result cannot be attributed only to support optimization.

## Controls

Each record must include:

- no edit
- target-label layerwise TSV centroid, no source signature
- source-signature layerwise TSV edit
- shuffled-signature layerwise TSV edit
- nearest-target layerwise TSV edit
- activation-rank1 only
- layerwise TSV only
- V14 flat-subspace task-vector baseline
- V16 output-layer conceptor baseline
- output-layer no-signature support optimizer
- 16 random norm-matched layerwise low-rank controls, named
  `random_norm_matched_layerwise_low_rank_delta:00` through `:15`

### Exact Control Construction

- `no_edit`: source weights unchanged.
- `target_label_layerwise_tsv_centroid`: uses the target-label centroid/mean layerwise
  delta from all 4096 train pairs for the direction; no source signature or source
  signature distance.
- `source_signature_layerwise_tsv`: runs the matched construction with source behavior as
  the target-signature pool while retaining the requested target label in support metrics.
- `shuffled_signature_layerwise_tsv`: deterministic within-direction cyclic shift of source
  signatures sorted by `(source_behavior, subject_id, target_behavior)`.
- `nearest_target_layerwise_tsv`: uses the single nearest target record by normalized
  signature distance with weight 1.0.
- `activation_rank1_only`: same candidate grid but task-singular scale fixed to 0.
- `layerwise_tsv_only`: same candidate grid but activation-rank1 scale fixed to 0.
- `v14_flat_subspace_task_vector`: existing V14 signature-gated flat-subspace task-vector
  candidate, reported as proof-critical. It may reuse V14 helper code but must recompute all
  train statistics from V17 train subjects only. It may not read V14 pools, V14 final raw,
  V14 results, or V14 stats artifacts.
- `v16_output_layer_conceptor`: existing V16 output-layer conceptor candidate, reported as
  proof-critical. It may reuse V16 helper code but must recompute all train statistics from
  V17 train subjects only. It may not read V16 pools, V16 final raw, V16 development
  results, or V16 stats artifacts.
- `output_layer_no_signature_support_optimizer`: V16 output-layer-only support optimizer,
  proof-critical.
- random controls: for each index, generate random per-layer components with the same layer
  mask and component rank budget as the matched selected candidate, normalize the resulting
  full delta to the matched delta norm, add to source weights, and evaluate. Seeds are
  `stable_hash_json([subject_id, source, target, index,
  "functional_weight_editing_v17_random_layerwise_low_rank"])`.

Random-control algorithm:

1. Use `torch.Generator().manual_seed(seed)` and CPU float32 tensors.
2. Use the matched candidate's selected `rank`, `layer_mask`, task scale, and activation
   scale only as metadata and norm/mask constraints; random controls do not reuse matched
   directions.
3. For each active weight matrix with shape `(out_dim, in_dim)`, draw `left` as
   `torch.randn(out_dim, rank)` and `right` as `torch.randn(rank, in_dim)`, then set
   `delta_weight = left @ right / sqrt(rank * in_dim)`.
4. For each active bias vector, draw a Gaussian vector of the same shape. Bias vectors are
   not low-rank; they are included in the full delta before normalization.
5. Inactive layers/components receive zero delta.
6. Pack component deltas into a full 345-vector. Normalize to matched full-delta norm:
   `random_delta = random_delta / random_delta.norm().clamp_min(1e-12) *
   matched_delta.norm()`.
7. If the matched delta norm is below `1e-12`, all random deltas are zero and this is
   recorded.
8. Add the normalized delta to source weights and evaluate with the same metrics as matched.

Proof-critical gating controls are all controls above except `no_edit`, which is reported
and included in best-control summaries but excluded from per-record compatible-MSE
advantage gates. No additional context or historical controls are allowed in the formal V17
artifact. Any extra diagnostic control after preregistration requires a new preregistration
or must be stored in a separate exploratory artifact with no gate impact.

### Support Objective

Candidate selection uses only train support tensors and source weights:

- target support BCE weight: `4.0`
- conflict BCE weight: `2.0`
- compatible source-output MSE weight: `0.01`
- source weight L2/MSE weight: `0.0005`

The objective is:

`4.0 * BCE(target_logits, target_labels) + 2.0 * BCE(conflict_logits,
conflict_target_labels) + 0.01 * MSE(compatible_logits, source_compatible_logits) +
0.0005 * MSE(candidate_weights, source_weights)`.

Logits are raw subject logits from `subject_forward_flat_batch`. BCE uses
`binary_cross_entropy_with_logits`. Reductions are PyTorch default means. Support split
definitions and expected compatible/conflict counts are inherited from V14
`source_target_support_split`; count mismatches fail closed.

## Metrics And Gates

Expected development record count: `288`.

Required per record:

- target prediction is the requested target behavior
- target margin >= `0.20`
- conflict target accuracy >= `0.65`
- conflict target accuracy improvement >= `0.15`
- matched target margin beats shuffled signature by >= `0.05`
- matched target margin beats target-label centroid by >= `0.02`
- matched target margin beats output-layer no-signature optimizer by >= `0.02`
- compatible-source-output MSE advantage over each proof-critical ablation/control except
  `no_edit` >= `2.0`
- Pareto-undominated against gating controls

Metric signs are fixed:

- target margin: target behavior classifier logit minus the largest non-target behavior
  classifier logit; higher is better.
- compatible-source-output MSE: MSE between edited logits and source logits on compatible
  source-support inputs; lower is better.
- target-margin advantage over a control: `matched_target_margin - control_target_margin`;
  higher is better.
- compatible-MSE advantage over a control: `control_compatible_source_output_mse -
  matched_compatible_source_output_mse`; higher is better.
- Pareto domination: a control dominates matched if
  `control.target_margin >= matched.target_margin` and
  `control.compatible_source_output_mse <= matched.compatible_source_output_mse`, with at
  least one strict inequality after epsilon `1e-8`. Matched is Pareto-undominated if no
  proof-critical gating control dominates it.

Aggregate gates:

- target prediction rate >= `0.85`
- individual all-gate pass rate >= `0.85`
- Pareto-undominated rate >= `0.85`
- mean matched target margin >= `0.25`
- mean conflict target accuracy >= `0.75`
- mean conflict target accuracy improvement >= `0.30`
- mean matched-minus-shuffled target margin >= `0.05`
- mean matched-minus-target-label target margin >= `0.02`
- mean matched-minus-output-layer-no-signature target margin >= `0.02`

Per-direction gates, for every source-target direction:

- `n == 24`
- target prediction rate >= `0.70`
- individual all-gate pass rate >= `0.70`
- Pareto-undominated rate >= `0.70`
- mean matched target margin >= `0.15`
- mean conflict target accuracy >= `0.60`
- mean matched-minus-shuffled target margin >= `0.03`
- mean matched-minus-output-layer-no-signature target margin >= `0.01`

The gates are intentionally lower than V16's development gates because V17 is a broader
layerwise edit with stronger controls and because V16 established that output-only edits are
not viable. They remain high enough to prevent a weak aggregate trend from authorizing
final.

## Leakage And Validity Guardrails

- Fresh V17 pools are mandatory.
- No prior final raw files may be opened.
- V17 final raw may not be opened before development passes and reviewer authorization.
- Final public audit surfaces must be recursively scanned for forbidden detail keys.
- Train/development/final overlap counts must be zero for seeds, subject IDs, weight hashes,
  and signature hashes.
- Development result must bind hashes for train pool, development pool, combined audit,
  final redacted audit, implementation file, formal preregistration, stats artifact, and
  result artifact.
- Per-record results may contain development subject IDs, but final public artifacts may not.
- If any startup schema fix is required, focused tests and reviewer approval are required
  before rerunning development.

## Compute Plan

- Keep V16-style process-parallel record evaluation with deterministic `spawn`, stable record
  sorting, and one PyTorch thread per worker.
- Add `--max-workers`, defaulting to `min(8, cpu_count)`.
- Add summary-only stdout by default. Full records are written only to JSON artifacts.
- Add phase timings to development output.
- Cache train-only stats by train pool hash, prereg hash, implementation hash, probe hash,
  support-suite hash, method config, and dependency code hashes for V14/V15/V16 helpers.
  Cache reuse is allowed only when all hashes match. Cache files may contain train-only
  tensors/statistics only and must never contain development or final subject records,
  development metrics, final redacted contents, or final raw contents. Cache mismatch must
  fail closed and recompute.
- Add a serial-vs-parallel equivalence test using a lightweight evaluator.

## Implementation Write Set

- `model_zoo/scripts/train_four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv.py`
- `model_zoo/scripts/test_four_behavior_functional_weight_editing_v17_helpers.py`
- formal preregistration copy after plan review:
  `docs/preregistrations/four_behavior_functional_weight_editing_v17_layerwise_rank1_tsv.md`

No lint will be run unless explicitly requested.

## Review Questions

1. Does the literature support the method change from V16 output-layer conceptors to
   layerwise rank-1/task-singular edits?
2. Are the controls sufficient to distinguish signature-conditioned editing from
   target-label, support-only, nearest-neighbor, and random low-rank editing?
3. Are the gates strong enough to avoid a misleading positive but not so strong that the
   experiment is guaranteed to be negative?
4. Are the compute and artifact-output controls sufficient for practical iteration?
5. Is the plan ready to convert into a formal preregistration and implementation?
