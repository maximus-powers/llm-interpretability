# V20 Preregistration: Signature-Conditioned Tangent Null-Space Editor

Status: formal preregistration. Implementation must follow this document.

## Planned Constants And Scopes

- script: `train_four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor.py`
- helper tests: `test_four_behavior_functional_weight_editing_v20_helpers.py`
- pool directory: `runs/four_behavior_functional_weight_editing_v20_pools`
- output directory:
  `runs/four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor`
- method: `signature_conditioned_tangent_nullspace_editor_v20`
- source pool scope: `four_behavior_functional_weight_editing_v20_source_pool`
- source audit scope: `four_behavior_functional_weight_editing_v20_source_pool_construction`
- final redacted scope:
  `redacted_final_functional_weight_editing_v20_source_pool_audit_surface_only`
- development claim scope:
  `four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor_development`
- final claim scope:
  `four_behavior_functional_weight_editing_v20_signature_conditioned_tangent_nullspace_editor_final`
- train base seed: `111400000`
- development base seed: `112400000`
- final base seed: `113400000`
- behavior stride: `100000`
- train accepted per behavior: `64`
- development accepted per behavior: `24`
- final accepted per behavior: `24`
- signature top-k: `8`
- signature temperature: `1.0`
- desired support logit magnitude: `2.0`
- compatible null-space relative SVD cutoff: `1e-4`
- compatible null-space absolute SVD cutoff: `1e-6`
- sparse sensitivity mask fractions: `[0.25, 0.5, 0.75, 1.0]`
- ridge lambda grid: `[0.01, 0.1, 1.0, 10.0]`
- signature-prior lambda grid: `[0.0, 0.01, 0.1, 1.0]`
- post-scale grid: `[0.5, 0.75, 1.0, 1.25]`
- max tangent delta norm: `8.0`
- random controls per record: `16`
- expected controls per record: `26` total, exactly `10` non-random controls plus `16`
  random controls
- eval workers default: `min(8, cpu_count - 2)` with a floor of `1`

## Motivation

V19 was a valid negative development result. It improved over V16 on target margin but did
not beat V17, output-layer no-signature, no-signature, or best-control baselines. The most
important failure pattern was not conflict accuracy: V19's mean conflict target accuracy was
above threshold. The failures were low target conversion, zero individual pass rate, weak
Pareto rate, and large losses to no-signature/output-layer controls on preservation and target
margin.

V20 therefore changes the mechanism. Instead of optimizing nonlinear decoded coefficients in a
train-derived task-vector basis, V20 solves a local tangent-space editing problem at each source
model:

> Use fixed probe signatures only to retrieve a target-behavior prior, then solve an exact
> Jacobian least-squares edit projected away from compatible-source tangent directions.

This tests whether activation signatures are useful as priors for local functional editing while
explicitly controlling tangent-space spillover.

## Literature Support

- Tangent-space task arithmetic shows that linearizing models can amplify weight
  disentanglement and improve model edits, linking successful task arithmetic to localized NTK
  eigenfunctions: <https://arxiv.org/abs/2305.12827>.
- AlphaEdit argues that model edits should be projected into a null space of preserved
  knowledge to reduce disruption; V20 directly adapts that principle to the small subject-model
  compatible-source support: <https://arxiv.org/html/2410.02355v3>.
- Recent spillover theory argues that parameter regularization alone cannot prevent off-target
  changes when tangent spaces overlap; methods need to control tangent-space overlap explicitly:
  <https://users.cs.northwestern.edu/~aravindv/finetuning-spillover.pdf>.
- TaLoS finds that task arithmetic benefits from sparse, low-sensitivity updates that promote a
  linearized regime and function localization; V20 includes per-component low-sensitivity masks
  derived from compatible-source Jacobian column norms:
  <https://arxiv.org/html/2504.02620v1>.
- Model reprogramming from an NTK perspective supports using frozen-source tangent geometry to
  adapt behavior with a small number of trainable degrees of freedom:
  <https://arxiv.org/html/2506.00620v1>.
- Multi-Subspace Representation Steering motivates separating steering attributes into
  lower-interference subspaces; V20 separates target-edit tangent rows from compatible-preserve
  tangent rows through null-space projection:
  <https://arxiv.org/abs/2508.10599>.
- Universal Weight Subspace evidence supports the broad premise that trained networks share
  low-dimensional, architecture-specific weight geometry, but V19 suggests that the next test
  should use that geometry locally rather than through a global decoder:
  <https://arxiv.org/html/2512.05117v2>.
- Subspace-boosted merging and weight-space-as-modality work support retaining explicit
  subspace structure instead of collapsing all behavior into one averaged vector:
  <https://arxiv.org/html/2506.16506v3> and <https://arxiv.org/html/2605.18632v1>.

## Fresh Data Protocol

V20 must generate fresh train/development/final pools. V16 through V19 development results have
all been observed, so V20 cannot reuse their development records.

Development may read only:

- `train_subjects.json`
- `development_subjects.json`
- `combined_audit.json`
- `final_redacted_audit.json`

Development must not read:

- `final_subjects.json`
- any prior experiment final raw file
- final raw subject IDs, seeds, signatures, weights, metrics, hashes, or raw lists

Final evaluation is allowed exactly once only if development passes all gates and a reviewer
confirms authorization.

## Final Redaction And Authorization

Use the exact V19 redaction contract with V20 scopes and paths. `combined_audit.pool_summaries.final`
may contain only:

- `accepted_counts_by_behavior`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`

`final_redacted_audit.json` top-level key set must be exactly:

- `behavior_suite_hashes`
- `candidate_pool_summary_hash`
- `claim_scope`
- `config_hash`
- `pool`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`
- `summary`
- `summary_payload_sha256`

`final_redacted_audit.summary` key set must be exactly:

- `accepted_counts_by_behavior`
- `max_selected_train_vs_heldout_overlap_count`

Any extra key, missing key, forbidden final-detail term, or unknown detailed final field
invalidates development and final authorization.

Before opening V20 final raw, the final runner must verify all of the following before opening any
`final_subjects.json` path:

- development result `phase == "development"`
- development result `claim_scope == DEVELOPMENT_SCOPE`
- development result `editor_method == "signature_conditioned_tangent_nullspace_editor_v20"`
- development result `passed == true`
- development result `next_action == "eligible_for_one_shot_final_eval_without_method_changes"`
- development result `record_count == 288`
- every record has exactly `16` random controls and exactly `26` controls total
- development result hash matches the reviewer authorization artifact
- reviewer authorization artifact has reviewer confidence `5`, reviewer name, ISO timestamp, and
  explicit authorization string `authorize_v20_one_shot_final_eval`
- train pool hash, development pool hash, combined audit hash, final redacted audit hash, formal
  preregistration hash, implementation hash, helper-test hash, stats artifact hash, constants hash,
  thresholds hash, train-statistics hash, V16 baseline statistics hash, and V17 baseline statistics
  hash all match the values embedded in the development result
- selected train-statistics hash equals the hash embedded in the stats artifact
- source-pool audit passed and all train/development/final public overlap counts are zero by seed,
  subject ID, weights hash, and signature hash

Any mismatch aborts before opening final raw.

## Method

V20 reuses reviewed primitives where possible:

- V14 deterministic Hungarian neuron alignment
- V17 signature top-k weighting and activation-rank1 residual construction
- V18/V19 redaction, source-pool validation, multiprocessing evaluation contract, and random
  basis-control audit style
- V19 fresh-pool and final-guard discipline

### Train-Only Statistics

For each source-target behavior direction:

1. Normalize train signatures with train-only mean/std.
2. Store train records by behavior.
3. Build signature centroids.
4. Build train-only signature-prior deltas from all ordered train source-target pairs:
   aligned target deltas, pair hashes, mean deltas, and per-component sensitivity summaries.
5. Fit V16 and V17 baseline statistics from V20 train subjects only.
6. Store probe examples hash, constants hash, thresholds hash, pair hashes, signature
   normalization tensors, baseline hashes, and train-statistics hash.

### Signature Prior

For a development source-target record:

1. Normalize the source signature as `(signature - train_sig_mean) / train_sig_std`, with
   train standard deviation clamped to `1e-6`.
2. Select top-8 train records from the target behavior by mean squared normalized signature
   distance, sorted by `(distance, subject_id)`.
3. Softmax weights are `softmax(-distance / 1.0)`.
4. Align selected train target weights to the source with V14 Hungarian alignment.
5. Compute the weighted aligned delta `delta_signature`.
6. Compute the V17 activation-rank1 residual from the same selected targets and weights.
7. Artifact metadata records only rank order, distance, softmax weight, and SHA256 of selected
   target subject IDs; no raw selected target IDs.

### Tangent Null-Space Editor

For each source-target record, compute Jacobians at the source weights in CPU float32. Flat
parameter order is exactly the existing 345-vector order used by `subject_forward_flat_batch` and
V17 layer-component specs.

- `J_target`: logits for V17 target support inputs with desired logits `+2/-2` from target labels.
- `J_conflict`: logits for V17 conflict inputs with desired logits `+2/-2` from conflict target labels.
- `J_preserve`: logits for compatible-source inputs with desired residual zero relative to source logits.

Let `J_edit = concat(J_target, J_conflict)` and `b_edit = desired_logits - source_logits`.
Do not center or normalize `J_preserve`; use the raw Jacobian rows of compatible logits with
respect to flat weights.

Exact support and Jacobian construction:

1. Build support tensors with `v16.v15.v14.prepare_support_tensors_with_source_logits`.
2. Row order is:
   - all target support examples in their returned tensor order,
   - then all conflict support examples in their returned tensor order,
   - compatible-source rows are separate and use compatible tensor order.
3. `target_labels`, `conflict_target_labels`, and `compatible_source_logits` are exactly the tensors
   returned by the support helper.
4. Desired target/conflict logits are `torch.where(label > 0.5, +2.0, -2.0)` in float32.
5. Source target/conflict logits are computed with the unedited source weights and subtracted from
   desired logits to form `b_edit`.
6. `J_target`, `J_conflict`, and `J_preserve` contain one row per scalar logit. For each row, use
   `torch.autograd.grad(logit, flat_source_weights, retain_graph=True)` with
   `flat_source_weights` as a detached CPU float32 leaf tensor requiring grad.
7. No minibatching, stochastic sampling, CUDA, or MPS is used. If autograd returns `None`, the run
   fails closed.

Candidate loop order is fixed and defines `candidate_index`: iterate mask fraction outermost, then
ridge lambda, then prior lambda, then activation scale, then post-scale innermost. The first
candidate has index `0`, and the index increments once per post-scale candidate, including invalid
candidates, so metadata can reconstruct the complete grid.

For each sparse sensitivity mask fraction:

1. Compute compatible sensitivity per parameter as `sqrt(mean(J_preserve ** 2, dim=0))`.
2. Within each V17 layer component, select the bottom fraction of parameters by
   `(sensitivity, flat_parameter_index)`, with at least one parameter per component.
3. Build a masked coordinate basis `E_mask` with shape `[345, mask_dim]`, whose columns are
   standard basis vectors in ascending flat-parameter index order.
4. Compute `M = J_preserve @ E_mask`, shape `[preserve_rows, mask_dim]`.
5. Compute `U, S, Vh = torch.linalg.svd(M, full_matrices=True)`.
6. Sign canonicalization: for every row vector in `Vh`, find the first element whose absolute value
   is greater than `1e-12`; if that element is negative, multiply the entire row by `-1`. Rows with
   no such element are left unchanged.
7. Let rank be the count of singular values satisfying
   `s > max(max_singular * 1e-4, 1e-6)`.
8. Let `N_mask = Vh[rank:].T`, shape `[mask_dim, mask_dim - rank]`. With
   `full_matrices=True`, this includes the full coordinate null basis when `preserve_rows <
   mask_dim`. If `rank == mask_dim`, the null dimension is zero, the candidate is invalid, and
   metadata records `empty_null_basis=true`.
9. The feasible edit basis in flat weight space is `B = E_mask @ N_mask`, shape
   `[345, null_dim]`. Columns are already orthonormal up to numerical tolerance. If
   `torch.max(abs(B.T @ B - I)) > 1e-4`, the candidate fails closed.

For each valid `B`, solve a closed-form ridge problem:

`argmin_z ||J_edit @ B @ z - b_edit||^2 + ridge_lambda * ||z||^2 + prior_lambda * ||B @ z - P_B(delta_signature)||^2`

where `P_B(delta_signature) = B @ (B.T @ delta_signature)` because `B` is orthonormal from the
SVD construction. In coefficient form, solve:

- `A = J_edit @ B`
- `z_prior = B.T @ delta_signature`
- `lhs = A.T @ A + (ridge_lambda + prior_lambda) * I`
- `rhs = A.T @ b_edit + prior_lambda * z_prior`

Solve with `torch.linalg.solve`; if the system is singular, add `1e-6 * I` once. If it still fails,
mark that candidate invalid.

The candidate delta is `B @ z + activation_scale * activation_rank1_delta`, where
`activation_scale` is selected from `[0.0, 0.5, 1.0]`. Clip candidate delta by global norm if
`norm(delta) > 8.0`, scaling to norm `8.0` and recording `trust_region_clipped=true`.

After solving, evaluate post-scales `[0.5, 0.75, 1.0, 1.25]` with the V17 support objective and
select by exact tuple:

`(support_objective, delta_norm, mask_fraction, ridge_lambda, prior_lambda, activation_scale, post_scale, candidate_index)`

ascending. The selected candidate becomes the matched V20 edit.

### Controls

Proof-critical controls:

- `no_edit`
- `target_label_tangent_nullspace`: same tangent/null-space solve, but `delta_signature` is the
  train-only target-behavior centroid delta for the requested direction.
- `no_signature_zero_tangent_nullspace`: same solve with zero signature prior and zero activation
  residual.
- `shuffled_signature_tangent_nullspace`: cyclically shifted source signature within each
  source-target direction, recomputing every signature-derived field.
- `source_signature_tangent_nullspace`: retrieval pool is train records from the source behavior.
- `no_nullspace_signature_tangent`: same signature prior but solve in masked coordinate basis
  without null-space projection.
- `v17_layerwise_rank1_tsv`: V17 recomputed on V20 train statistics.
- `v16_output_layer_conceptor`: V16 recomputed on V20 train statistics.
- `output_layer_no_signature_support_optimizer`: V16 output-layer no-signature optimizer.
- `nearest_target_layerwise_tsv`: nearest train target through V17 TSV.
- `random_norm_matched_tangent_delta_00` through `_15`: deterministic norm-matched random
  controls sampled in the selected matched feasible basis.

Random controls participate in Pareto checks. Named non-random controls participate in target
margin and compatible-source preservation advantage gates.

Exact random-control algorithm:

1. Use the selected matched candidate's feasible basis `B`. If the matched candidate has no valid
   feasible basis, no matched edit can be selected and the record fails closed before random control
   generation.
2. Build seed payload with method string, train-statistics hash, source behavior, target behavior,
   source subject ID, selected matched delta norm converted with `tensor_to_hashable`, selected
   feasible basis hash, and random index.
3. Compute `stable_hash_json(seed_payload)`.
4. Convert the first 16 hex characters to an unsigned integer modulo `2**63 - 1`.
5. Seed CPU `torch.Generator(device="cpu")`.
6. Draw `torch.randn(null_dim, generator=generator, dtype=torch.float32)`.
7. Decode random delta as `B @ coeff`.
8. If matched delta norm or raw random norm is below `1e-12`, return zero delta and record
   `zero_norm_fallback=true`; otherwise scale random delta to exactly matched delta norm in
   float32.
9. Metadata records random index, seed hash, raw norm, matched norm, final norm, coefficient hash,
   feasible basis hash, selected mask fraction, selected ridge lambda, selected prior lambda,
   selected activation scale, selected post-scale, and zero-norm fallback.

## Metrics And Gates

Use V19 gates unchanged:

- record count exactly `288`
- random controls exactly `16` per record
- aggregate target prediction rate at least `0.85`
- aggregate individual all-gates pass rate at least `0.85`
- aggregate Pareto-undominated rate at least `0.85`
- mean target margin at least `0.25`
- mean conflict target accuracy at least `0.75`
- mean conflict target accuracy improvement at least `0.25`
- matched-minus-best-control target margin at least `0.02`
- matched-minus-target-label target margin at least `0.02`
- matched-minus-shuffled-signature target margin at least `0.05`
- matched-minus-output-layer-no-signature target margin at least `0.02`
- matched-minus-V17 target margin at least `0.02`
- all source/development/final overlap counts zero by seed, subject ID, weight hash, and
  signature hash

Direction-level gates:

- target prediction rate at least `0.65`
- individual all-gates pass rate at least `0.65`
- Pareto-undominated rate at least `0.75`
- mean target margin at least `0.15`
- matched-minus-output-layer-no-signature target margin at least `0.01`
- matched-minus-V17 target margin at least `0.01`

Failure of any gate makes the result negative/inconclusive and forbids final raw access.

Exact per-record pass criteria:

- `target_prediction_pass` is true iff `functional_metrics(...).predicted_behavior == target`.
- `pareto_undominated` is true iff no Pareto control dominates matched. A control dominates matched
  when `control.target_margin >= matched.target_margin - 1e-9` and
  `control.compatible_source_output_mse <= matched.compatible_source_output_mse + 1e-9`, with at
  least one strict improvement by `1e-9`.
- `min_proof_critical_compatible_mse_advantage` is the minimum over proof-critical non-random
  controls of `control.compatible_source_output_mse - matched.compatible_source_output_mse`.
- `individual_all_gates_passed` requires target prediction, target margin, conflict target
  accuracy, conflict target accuracy improvement, Pareto-undominated, every named advantage
  target-margin threshold, and
  `min_proof_critical_compatible_mse_advantage >= -0.02`.

Aggregate and direction preservation are summarized for every named non-random control as
`mean_<control>_minus_matched_compatible_source_output_mse`, but the only hard compatible-MSE gate
is the per-record minimum proof-critical threshold above. This is unchanged from V19.

## Required Tests

Before pool generation, targeted pytest must cover:

- V20 scopes, seeds, and paths differ from V19.
- generic final raw guard rejects all `runs/**/final_subjects.json` paths.
- exact final redaction allowlists fail closed.
- train statistics hash binds signatures, pair hashes, probe examples, constants, thresholds,
  tangent constants, and baseline hashes.
- signature prior uses train target records only.
- compatible null-space basis construction uses raw compatible Jacobian rows and exact rank cutoff.
- per-component sparse sensitivity masks select bottom sensitivity values with deterministic
  tie-breaks and at least one parameter per component.
- ridge solve is deterministic and uses the signature prior term exactly.
- candidate selection tie-break is deterministic.
- target-label, no-signature, shuffled-signature, source-signature, and no-nullspace controls alter
  only preregistered initialization/projection sources.
- random controls are deterministic, norm-matched, and included in Pareto checks.
- shuffled signatures cycle within each source-target direction.
- serial and multiprocessing evaluation are byte-identical under a deterministic evaluator.
- summary stdout redacts verbose pool and final details.

Run only targeted pytest and `py_compile`; do not run lint automatically.

## Compute Plan

- Pool generation may use PyTorch internal parallelism.
- Train statistics are computed once per phase and cached.
- Per-record tangent solves are embarrassingly parallel and must run with the existing process pool,
  defaulting to 8 workers and one torch thread per worker.
- Long-running phases print summary-only progress and never print raw final details.

## Review Gates

1. Reviewer must approve this plan at confidence `5/5` before formal preregistration.
2. Reviewer must approve the formal preregistration copy at `5/5`.
3. Reviewer must approve implementation/tests at `5/5` before pool generation.
4. Reviewer must approve source-pool construction at `5/5` before development.
5. Reviewer must approve development results at `5/5`.
6. Final raw remains sealed unless development passes and reviewer explicitly authorizes final
   evaluation.
