# V23 Plan: Probe-Routed Sparse Functional Subspace Editor

## Status

Draft plan for reviewer approval. Do not implement or generate pools until review returns 5/5.

## Motivation from V22

V22 is a valid negative development result:

- development artifact:
  `runs/four_behavior_functional_weight_editing_v22_component_activation_rank1_editor/development_results.json`
- artifact hash:
  `a2d773e9f157468ae6ab3c9d8917a3eb41d861e55e58123492c42fee6b91ecf2`
- `passed=false`
- `next_action=log_negative_development_result_do_not_open_final_raw`
- target prediction rate: `0.6284722222222222`
- Pareto-undominated rate: `0.4548611111111111`
- individual all-gate pass rate: `0.0`
- mean matched minus best-control target margin: `-0.09864032715178912`
- scale-0 rate: `0.1875`
- selected layer counts: `{"0":63,"1":42,"2":48,"3":49,"4":86}`

Interpretation: the fixed-probe signal is not absent. V22 beats no-signature/output-layer and
V16/V17/V20/V21 averages on target margin, and scale-0 collapse is much lower than V21.
However, it does not prove the hypothesis because the matched editor loses to the best
per-record control, often a shuffled/source/target-label component control. The next experiment
must therefore test whether probe signatures can route a sparse functional edit that beats
strong controls under an inner-validation objective that explicitly penalizes control-like
success.

## Recent Literature Support

1. [Steer2Edit: From Activation Steering to Component-Level Editing](https://arxiv.org/html/2602.09870v2)
   motivates translating activation steering vectors into permanent component-level weight
   edits. V22 tested the single-component rank-1 version; V23 keeps component-level editing but
   adds sparse multi-component routing and inner-validation control penalties.
2. [A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering](https://arxiv.org/html/2510.01246v1)
   reports that sparse, relevance-filtered activation features can outperform mean activation
   differences and that constant steering can be unstable. V23 uses sparse feature/component
   selection and validates scales rather than assuming one constant rank-1 edit is enough.
3. [Efficient Model Editing with Task-Localized Sparse Fine-Tuning](https://arxiv.org/html/2504.02620v1)
   motivates sparse task-localized updates to reduce interference. V23 selects low-interference
   component subspaces and explicitly scores compatible-probe preservation.
4. [Model Merging in the Era of Large Language Models](https://arxiv.org/html/2603.09938v1)
   surveys task vectors, model merging, loss-landscape structure, and interference issues.
   V23 treats V22's best-control loss as an interference/control-dominance problem rather than
   only a weak-target problem.
5. [Structure Is Not Enough: Leveraging Behavior for Neural Network Weight Reconstruction](https://arxiv.org/html/2503.17138v1)
   argues that structural reconstruction alone can miss functionally critical behavior. V23
   selects subspace edits by behavioral losses on support/probe examples, not by weight-space
   closeness alone.
6. [A Survey of Weight Space Learning: Understanding, Representation, and Generation](https://arxiv.org/html/2603.10090v1)
   frames weight-space representation and generation as first-class learning problems with
   symmetry and functional-equivalence concerns. V23 keeps alignment inherited from V16/V17 and
   evaluates functional behavior under strict heldout-pool controls.
7. [The Anatomy of an Edit: Mechanism-Guided Activation Steering for Knowledge Editing](https://arxiv.org/html/2603.20795v1)
   motivates mechanism-guided routing to attribution-aligned regions. V23 routes edits through
   train-only component attribution/sensitivity estimates instead of choosing a hidden layer only
   by post-hoc candidate loss.

## Hypothesis

Fixed-probe activation signatures contain enough information to route a sparse, low-interference
functional weight edit across a small number of hidden components. Compared with V22's single
rank-1 hidden component, a train-only routed sparse subspace should improve target prediction and
Pareto reliability while beating matched controls that use no probe, source probe, shuffled probe,
target label only, or nearest target descriptors.

## Fresh V23 Pools

V23 uses fresh pools. V22 development artifacts may be used only as negative prior evidence and
to define diagnostics; V22 train/development/final raw records must not be reused.

Pool directory:
`runs/four_behavior_functional_weight_editing_v23_pools`

Output directory:
`runs/four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor`

Pool seeds:

- train base seed: `120400000`
- development base seed: `121400000`
- final base seed: `122400000`
- behavior stride: `100000`

Counts and attempt caps match V22:

- train: 64 accepted per behavior, 128 attempts per behavior
- development: 24 accepted per behavior, 64 attempts per behavior
- final: 24 accepted per behavior, 64 attempts per behavior

Final raw remains sealed. Only `final_redacted_audit.json` and the allowlisted final summary
inside `combined_audit.json` may be read before final authorization.

## Editor Overview

Editor method:
`probe_routed_sparse_subspace_editor_v23`

Base edit:
start from `v16.output_layer_no_signature_support_optimizer`, exactly as V22 did. Hidden edits
are applied on top of that base.

Candidate hidden components:

- hidden layers: `[0, 1, 2, 3, 4]`
- each component candidate edits both `weight_l` and `bias_l` using V17 component specs
- output layer 5 is not edited by the V23 hidden editor

Descriptor per subject and layer:

- `hbar_l`: mean post-GELU hidden activation over fixed probe examples
- `xbar_l`: mean pre-layer input over fixed probe examples
- `g_target_l`: gradient/sensitivity of target support loss and target probe-centroid loss with
  respect to the rank-1 direction at zero hidden edit
- `g_preserve_l`: gradient/sensitivity of compatible support/probe utility with respect to the
  rank-1 direction at zero hidden edit

Train statistics:

- signature mean/std and behavior centroids
- hidden `hbar`/`xbar` descriptors per train subject
- target hidden centroids by behavior
- target probe-logit centroids by behavior
- component attribution/sensitivity summaries by source-target direction
- inner train/validation split: 51/13 records per behavior, hash-bound and disjoint

## Sparse Subspace Candidate Construction

For each development record and target:

1. Compute matched descriptor directions:
   `d_l = target_hidden_centroid_l(target) - hbar_l(source_subject)`.
2. Compute control descriptor directions using the same modes as V22:
   `no_probe`, `source_probe`, `shuffled_probe`, `target_label_only`, `nearest_target`.
3. Score every hidden layer by the exact train-only routed relevance formula below.
4. Candidate component sets are the top `k` layers by `relevance_l`, with
   `k in [1, 2, 3]`.
5. For each selected layer, use the V22 rank-1 basis:
   `Delta W_l = alpha_l * outer(d_l, xbar_l) / (||xbar_l||^2 + lambda_rank1)`;
   `Delta b_l = alpha_l * d_l`.
6. Solve coefficients `alpha` by a small ridge problem over selected components, using
   linearized logits on support and fixed-probe examples:
   - target support logits move toward desired target labels
   - conflict logits move toward target labels
   - compatible logits preserve source logits
   - fixed-probe logits move toward target behavior centroid only on non-compatible probes
7. Clip total hidden edit norm to
   `cap_multiplier * max(||theta_output_base - theta_source||, 1e-12)`.
8. Evaluate the exact edited model, not only the linear approximation.

### Exact Layer Relevance Formula

Layer relevance is computed separately for each hyperparameter configuration because
`lambda_rank1`, `compatible_weight`, `probe_centroid_weight`, and `control_penalty_weight`
change the score.

For a descriptor mode `m`, layer `l`, and rank parameter `lambda_rank1`, define the raw
component basis vector `b_l(m)` as a flat 345-dimensional vector with nonzero entries only in
`weight_l` and `bias_l`:

- `Delta W_l(m) = outer(d_l(m), xbar_l(source_subject)) / (||xbar_l(source_subject)||^2 + lambda_rank1)`
- `Delta bias_l(m) = d_l(m)`

The matched basis is `b_l = b_l(matched_probe)`.

Let `J_base(inputs)` be the exact autograd Jacobian of scalar logits with respect to the flat
345-dimensional weight vector at `base_weights`, where `base_weights` is the output-layer
no-signature support-optimized model. For each row group below, the scalar linearized feature for
layer `l` is `x_l = J_base(row_input) @ b_l`.

Row groups are ordered exactly:

1. `target_support`
2. `conflict_support`
3. `compatible_support`
4. `probe_target`
5. `probe_compatible`

Targets are residuals from the base logits:

- `target_support`: `desired_target_logit - base_logit`, where desired target logit is
  `+2.0` for positive target label and `-2.0` otherwise.
- `conflict_support`: same desired target-logit rule using conflict target labels.
- `compatible_support`: `source_logit - base_logit`, preserving source behavior.
- `probe_target`: `target_probe_logit_centroid[target] - base_probe_logit` on fixed-probe
  examples where `predicate_source(sequence) != predicate_target(sequence)`.
- `probe_compatible`: `source_probe_logit - base_probe_logit` on fixed-probe examples where
  `predicate_source(sequence) == predicate_target(sequence)`.

Per-row weights are the square root of block weight divided by row count:

- `target_support`: `sqrt(1.0 / n_target_support)`
- `conflict_support`: `sqrt(1.0 / n_conflict_support)`
- `compatible_support`: `sqrt(compatible_weight / n_compatible_support)`
- `probe_target`: `sqrt(probe_centroid_weight / max(1, n_probe_target))`; omitted if
  `probe_centroid_weight == 0` or `n_probe_target == 0`
- `probe_compatible`: `sqrt(compatible_weight / n_probe_compatible)`

Let `x_edit_l` and `y_edit` be the weighted rows/targets from `target_support`,
`conflict_support`, and `probe_target`. Let `x_preserve_l` be the weighted rows from
`compatible_support` and `probe_compatible`.

Use `eps = 1e-12`.

- `target_gain_l = (dot(x_edit_l, y_edit)^2) / ((sum(x_edit_l^2) * sum(y_edit^2)) + eps)`
- `preserve_cost_l = mean(x_preserve_l^2)`; if there are no preserve rows, use `inf`
- For each nonzero control mode in `{source_probe, shuffled_probe, target_label_only, nearest_target_probe}`,
  `cos2_l(control) = dot(d_l(matched), d_l(control))^2 / ((||d_l(matched)||^2 * ||d_l(control)||^2) + eps)`.
  If either direction norm is below `eps`, that control contributes `0.0`.
- `control_similarity_penalty_l = max(cos2_l(control))`
- `relevance_l = target_gain_l - compatible_weight * preserve_cost_l - control_penalty_weight * control_similarity_penalty_l`

Sort layers by:

1. descending `relevance_l`
2. descending `target_gain_l`
3. ascending `preserve_cost_l`
4. ascending `control_similarity_penalty_l`
5. ascending `layer_index`

The top `k` layers form the sparse set. If any selected layer has nonfinite relevance, the
configuration is invalid for that record.

### Exact Sparse Ridge Solve

For a selected layer set `S = [l_0, ..., l_{k-1}]`, build basis matrix
`B = [b_l0, ..., b_l{k-1}]` using matched descriptor directions only. The same row groups,
targets, and row weights from the relevance calculation define a weighted design matrix
`X = J_base @ B` and weighted target vector `y`, with rows ordered exactly by group order above
and original input order inside each group.

Rows with zero block weight are omitted. If `X` has zero rows, the configuration is invalid.

Solve:

`alpha = solve(X.T @ X + lambda_solve * I, X.T @ y)`

All tensors are `torch.float32` on CPU. If `torch.linalg.solve` raises, retry exactly once with
`(lambda_solve + 1e-6) * I`. If the retry raises, the configuration is invalid for that record.

For each `post_scale` in the grid, compute:

`delta = post_scale * (B @ alpha)`

Clip `delta` if its hidden-component norm exceeds:

`cap_multiplier * max(||theta_output_base - theta_source||, 1e-12)`

where the hidden-component norm includes only selected `weight_l` and `bias_l` entries. If cap is
`0` and norm is nonzero, use zero delta. Apply `delta` to `base_weights`, evaluate the exact edited
model, and select the record-level post-scale candidate by:

1. lower support objective
2. lower target probe-centroid loss on `probe_target`
3. lower compatible probe utility loss on `probe_compatible`
4. lower hidden edit norm
5. lower post-scale
6. lower deterministic candidate index

The selected post-scale is stored in metadata. The hidden coefficient vector, basis matrix, and
raw private vectors are not serialized into public result records.

Candidate grid:

- `k`: `[1, 2, 3]`
- `lambda_rank1`: `[0.01, 0.1, 1.0, 10.0]`
- `lambda_solve`: `[0.01, 0.1, 1.0, 10.0]`
- `compatible_weight`: `[0.5, 1.0, 2.0, 4.0]`
- `probe_centroid_weight`: `[0.0, 0.25, 0.5, 1.0]`
- `control_penalty_weight`: `[0.0, 0.5, 1.0, 2.0]`
- `cap_multiplier`: `[0.25, 0.5, 1.0]`
- post-solve scale: `[0.25, 0.5, 0.75, 1.0]`

## Inner-Validation Selection

V23 must not select hyperparameters by development performance. It uses train-only inner
validation.

Inner split:

- per behavior: first 51 by deterministic split hash are inner train, remaining 13 are inner
  validation
- split hash payload is exactly:
  `{"scope":"four_behavior_functional_weight_editing_v23_inner_split","behavior":behavior,"subject_id":subject_id}`
- sort tuple is `(split_hash, subject_id)`
- no subject can appear in both inner train and inner validation

Inner-validation jobs are ordered by:

1. source behavior in `PATTERNS` order
2. subject ID ascending
3. target behavior in `PATTERNS` order, excluding source

Expected denominator: `52 inner-validation subjects * 3 targets = 156` records.

For every hyperparameter configuration, fit train statistics on inner-train subjects only, then
evaluate inner-validation records against all proof-critical non-random controls. Controls whose
behavior depends on the selected V23 config use that same config. Historical controls use their
own inner-train-only baseline statistics.

Invalid-config handling:

- if any inner-validation record raises, produces nonfinite public metrics, produces a wrong
  control count, or duplicates/misses a proof-critical control, mark the whole config invalid;
- invalid configs sort after valid configs;
- if all configs are invalid, training fails closed before development evaluation.

For valid configs, select lexicographically by:

1. higher inner-validation target prediction rate
2. higher inner-validation Pareto-undominated rate
3. higher mean matched-minus-best-control target margin
4. higher mean matched-minus-shuffled-signature target margin
5. higher mean target margin
6. lower mean compatible source-output MSE
7. lower scale-0/effectively-zero coefficient rate
8. lower total hidden edit norm
9. deterministic config index

The selected config hash is part of `train_statistics_hash`.

## Controls

Expected controls per record: 32.

Proof-critical non-random controls:

1. `no_edit`
2. `output_layer_no_signature_support_optimizer`
3. `v16_output_layer_conceptor`
4. `v17_layerwise_rank1_tsv`
5. `v20_tangent_nullspace_editor_recomputed`
6. `v21_behavioral_probe_residual_output_editor_recomputed`
7. `v22_component_activation_rank1_editor_recomputed`
8. `no_probe_sparse_subspace_editor`
9. `source_probe_sparse_subspace_editor`
10. `shuffled_probe_sparse_subspace_editor`
11. `target_label_only_sparse_subspace_editor`
12. `nearest_target_sparse_subspace_editor`

Random controls:

- 20 random norm-matched sparse-subspace controls
- use the selected matched component set and norm, but randomize coefficient direction in the
  selected sparse basis
- deterministic seed hash includes subject hash, direction, selected config hash,
  train-statistics hash, and random index

Exact random control procedure:

1. Let `B_selected` be the matched selected sparse basis matrix after applying the selected
   `lambda_rank1`, with columns ordered by selected layer order.
2. Let `matched_hidden_delta = edited_matched_weights - base_weights`, restricted to selected
   hidden component entries.
3. Let `matched_norm = ||matched_hidden_delta||_2`.
4. Seed payload is exactly:
   `{"scope":"four_behavior_functional_weight_editing_v23_random_sparse_control","subject_hash":subject_hash,"source":source,"target":target,"selected_config_hash":selected_config_hash,"train_statistics_hash":train_statistics_hash,"index":index,"selected_layers":selected_layers}`
5. `seed_hash = stable_hash_json(seed_payload)`.
6. `seed = int(seed_hash[:16], 16) % (2**63 - 1)`.
7. Use `torch.Generator(device="cpu").manual_seed(seed)` and sample
   `z = torch.randn(k, dtype=torch.float32, generator=generator)`.
8. `raw_delta = B_selected @ z`.
9. If `matched_norm < 1e-12` or `||raw_delta|| < 1e-12`, random delta is zero and
   `zero_norm_fallback=true`; otherwise
   `random_delta = raw_delta / ||raw_delta|| * matched_norm`.
10. Apply `random_delta` to `base_weights`.
11. Metadata fields: `control_type`, `index`, `seed_hash`, `random_seed`,
    `selected_layers`, `matched_hidden_delta_norm`, `random_hidden_delta_norm`,
    `zero_norm_fallback`, `selected_config_hash`, and `train_statistics_hash`.

Proof-critical controls participate in all per-record gates. Random controls participate in Pareto
domination but not in best-control aggregate target-margin threshold.

## Gates

Use the V22 aggregate/direction/per-record gates with these changes:

- expected controls per record: 32
- expected random controls per record: 20
- add aggregate matched-minus-V22 target-margin advantage >= `0.02`
- add direction matched-minus-V22 target-margin advantage >= `0.01`
- require aggregate matched-minus-shuffled-signature target-margin advantage >= `0.05`
- require mean matched-minus-best-control target margin >= `0.02`

Diagnostic downgrades:

- If scale/effective-zero coefficient rate >= V22's `0.1875`, a passing result is downgraded
  to diagnostic-only pending reviewer approval.
- If selected `k=1` on more than 80% of records, result is diagnostic-only because it did not
  really test sparse multi-component routing.
- If any proof-critical control count is missing or duplicated, result fails closed.

## Leakage Controls

- V23 source pools must be fresh and disjoint from V22 pools by seed, subject ID, signature hash,
  and weights hash.
- Development may load train/development pools, combined audit, and final redacted audit only.
- Development must call `assert_no_forbidden_final_raw_paths(... allow_v23_final=False)` on all
  public paths.
- `final_subjects.json` is forbidden before hash-bound final authorization.
- Final redacted top-level allowlist exactly matches V22:
  `behavior_suite_hashes`, `candidate_pool_summary_hash`, `claim_scope`, `config_hash`, `pool`,
  `pool_file_sha256`, `pool_redacted_payload_sha256`, `summary`, `summary_payload_sha256`.
- Final redacted summary allowlist exactly:
  `accepted_counts_by_behavior`, `max_selected_train_vs_heldout_overlap_count`.
- Combined audit `pool_summaries.final` allowlist exactly:
  `accepted_counts_by_behavior`, `pool_file_sha256`, `pool_redacted_payload_sha256`.
- Any extra or missing key in `combined_audit.pool_summaries.final`,
  `final_redacted_audit`, or `final_redacted_audit.summary` before authorized final access
  invalidates final proof use and fails development/final contract validation closed.

## Output Artifacts

Development result must include:

- `claim_scope`
- implementation, helper-test, formal-prereg, and plan hashes
- train/development pool hashes
- combined audit hash
- final redacted audit hash
- train statistics hash
- selected config and selected config hash
- aggregate and by-direction summaries
- scale/effective-zero coefficient rate
- selected k counts
- selected layer-set counts
- all gate failures
- `next_action`

If development fails:
`next_action=log_negative_development_result_do_not_open_final_raw`

If development passes:
`next_action=run_hash_bound_final_after_reviewer_authorization`

Final evaluation is not implemented until development passes and reviewer authorization is bound
to exact development result hash, implementation hash, test hash, formal-prereg hash, plan hash,
train/development/final pool hashes, combined audit hash, and final redacted audit hash.

## Implementation Tasks

### Task 1: Create V23 Scaffold and Failing Contract Tests

Files:

- Create: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor.py`
- Create: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v23_helpers.py`

Steps:

1. Copy V22 script and tests as a starting point.
2. Write failing tests for V23 paths, seeds, `EDITOR_METHOD`, expected control counts, final raw
   guards including V22 final raw, and next-action constants.
3. Run:
   `python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v23_helpers.py`
   Expected: fail on missing V23 constants/paths.
4. Patch constants and path scopes.
5. Run the same pytest until those contract tests pass.

### Task 2: Add Sparse Subspace Feature Construction

Files:

- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor.py`
- Modify: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v23_helpers.py`

Tests:

- `test_v23_component_relevance_penalizes_control_similarity`
- `test_v23_sparse_component_set_is_deterministic`
- `test_v23_hidden_edit_touches_only_selected_components`

Implementation:

- add hidden descriptor extraction from V22
- add component relevance scoring
- add top-k deterministic sparse set selection
- add multi-component edit application with norm clipping

### Task 3: Add Linearized Sparse Coefficient Solver

Tests:

- `test_v23_solver_preserves_compatible_logits_when_weight_high`
- `test_v23_solver_moves_target_probe_logits_when_probe_weight_high`
- `test_v23_solver_fails_closed_on_singular_system`

Implementation:

- build support/probe design rows for selected component basis
- solve ridge coefficients with one jitter retry
- clip total hidden edit norm
- evaluate exact edited weights after solve

### Task 4: Add Inner-Validation Hyperparameter Selection

Tests:

- `test_v23_inner_split_is_51_13_per_behavior_and_disjoint`
- `test_v23_config_selection_prefers_best_control_advantage_after_prediction_and_pareto`
- `test_v23_selected_config_hash_changes_when_grid_changes`

Implementation:

- deterministic split
- evaluate candidate configs on inner validation
- select config lexicographically as specified
- bind selected config hash into train-statistics hash

### Task 5: Add Controls and Gates

Tests:

- `test_v23_build_controls_has_32_controls_and_20_random_controls`
- `test_v23_random_sparse_controls_are_deterministic_and_norm_matched`
- `test_v23_gate_failures_enforce_v22_advantage_and_control_counts`

Implementation:

- recompute V16/V17/V20/V21/V22 historical controls
- add five sparse-subspace descriptor controls
- add 20 random sparse controls
- update advantage maps and gate thresholds

### Task 6: Pool Generation and Development Runner

Tests:

- `test_v23_final_redaction_allowlists_fail_closed`
- `test_v23_source_pool_contract_rejects_any_final_raw_before_authorization`
- `test_v23_stats_artifact_excludes_raw_subjects_and_private_vectors`

Implementation:

- fresh V23 pool seeds
- redacted final audit and combined final summary allowlists
- development runner writes stats and results
- final remains fail-closed

### Task 7: Verification and Review

Commands:

- `python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v23_helpers.py`
- `python -m py_compile model_zoo/scripts/train_four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor.py model_zoo/scripts/test_four_behavior_functional_weight_editing_v23_helpers.py`
- stale-surface scan for V22-only strings, accidental final raw reads, and private-key exposure

After passing local checks, request reviewer confidence 5/5 before pool generation.

## Reviewer Questions

1. Does V23 address the V22 failure mode without leaking development or final information into
   method selection?
2. Are the literature-supported changes specific enough to justify another run?
3. Are the controls strong enough to prevent target-label, shuffled-signature, or source-signature
   confounds?
4. Are the final raw sealing and redacted allowlists complete?
5. Is the compute plan reasonable for 12 logical CPUs with `--max-workers 8` during evaluation?
