# V22 Plan: Component Activation Rank-1 Editor

Status: revised draft for reviewer approval; do not implement or generate pools until review is
5/5.

## Motivation

V21 is a clean negative development result. It failed the preregistered gates and selected scale
0 on 199/288 records. V22 therefore stops predicting an output-layer residual and instead tests
whether fixed-probe activation signatures can select a hidden component whose local rank-1 edit
changes the representation path while preserving utility.

## Literature Support

- Steer2Edit (Sun, Yan, Wang, Weng, 2026) directly motivates translating activation steering
  vectors into selective component-level rank-1 weight edits:
  https://arxiv.org/abs/2602.09870
- Why Steering Works (Xu et al., 2026) motivates explicit preference/utility scoring and
  off-manifold penalties for interventions that can trade target control against validity:
  https://arxiv.org/abs/2602.02343
- Structure Is Not Enough (Falk, Schurholt, Borth, 2025) motivates scoring generated/edited
  weights by behavioral losses, not Euclidean weight distance alone:
  https://arxiv.org/abs/2503.17138
- Reinforcement Learning for Neural Model Editing (Malik, 2026) is used only as weak
  motivation for reward-guided model-edit search. V22 uses deterministic candidate search,
  not RL:
  https://arxiv.org/abs/2606.13461
- Spectral Over-Accumulation in Model Merging (Li et al., 2026) is used only as weak
  motivation for avoiding additive multi-component accumulation. V22 selects one hidden layer
  and applies an explicit hidden-delta norm cap:
  https://arxiv.org/abs/2602.05536

## Fresh V22 Pools

V22 must use fresh pools for any proof-grade development result.

Pool scope: `four_behavior_functional_weight_editing_v22_source_pool`.
Audit scope: `four_behavior_functional_weight_editing_v22_source_pool_construction`.
Development scope:
`four_behavior_functional_weight_editing_v22_component_activation_rank1_editor_development`.
Final scope:
`four_behavior_functional_weight_editing_v22_component_activation_rank1_editor_final`.
Final redacted scope:
`redacted_final_functional_weight_editing_v22_source_pool_audit_surface_only`.

Pool directory:
`runs/four_behavior_functional_weight_editing_v22_pools`.

Development/final output directory:
`runs/four_behavior_functional_weight_editing_v22_component_activation_rank1_editor`.

Seed bases:

- train: `117400000`
- development: `118400000`
- final: `119400000`

Pattern stride remains `100000`. Counts and attempt caps remain:

- train: 64 accepted per behavior, max 128 attempts per behavior
- development: 24 accepted per behavior, max 64 attempts per behavior
- final: 24 accepted per behavior, max 64 attempts per behavior

Final raw handling:

- Never open `runs/**/final_subjects.json`.
- Final raw path is forbidden in all development reads.
- Final redacted audit may expose only:
  `behavior_suite_hashes`, `candidate_pool_summary_hash`, `claim_scope`, `config_hash`,
  `pool`, `pool_file_sha256`, `pool_redacted_payload_sha256`, `summary`,
  `summary_payload_sha256`.
- Final redacted `summary` may expose only:
  `accepted_counts_by_behavior`, `max_selected_train_vs_heldout_overlap_count`.
- `combined_audit.pool_summaries.final` may expose only:
  `accepted_counts_by_behavior`, `pool_file_sha256`, `pool_redacted_payload_sha256`.
  Any extra key in this combined-audit final summary before authorized final evaluation
  invalidates the final pool for proof use.

## Final-Evaluation Authorization

Final evaluation is not authorized by a passing development run alone. Final raw may be opened
only after reviewer authorization that is explicitly bound to a passing V22 development artifact.

Before opening final raw, the final runner must load the development result and verify exact
matches for all of the following:

- `claim_scope` equals
  `four_behavior_functional_weight_editing_v22_component_activation_rank1_editor_development`
- `phase` equals `development`
- `passed` is `true`
- `next_action` equals `run_hash_bound_final_after_reviewer_authorization`
- `editor_method` equals `component_activation_rank1_editor_v22`
- train pool SHA equals the hash recorded in the authorized development result
- development pool SHA equals the hash recorded in the authorized development result
- combined audit SHA equals the hash recorded in the authorized development result
- final redacted audit SHA equals the hash recorded in the authorized development result
- implementation SHA equals the hash reviewed for final authorization
- formal prereg SHA equals the hash reviewed for final authorization
- train statistics hash equals the hash reviewed for final authorization
- reviewer authorization artifact contains the matching development result SHA and the exact
  phrase `V22 final evaluation authorized`

Any mismatch must fail closed before opening `final_subjects.json`. If development fails, or if
`next_action` is anything else, final raw remains sealed.

## Architecture and Tensor Definitions

Subject flat weight dimension is exactly 345. Hidden layers are the first five GELU layers
exposed by `v17.hidden_inputs_and_outputs_flat_batch`:

- layer 0: weight shape `(8, 5)`, bias shape `(8,)`, input shape `(n_probe, 5)`,
  post-GELU output shape `(n_probe, 8)`
- layers 1..4: weight shape `(8, 8)`, bias shape `(8,)`, input shape `(n_probe, 8)`,
  post-GELU output shape `(n_probe, 8)`
- output layer 5 is not edited by the hidden component editor; its output theta is inherited
  from the deterministic output-layer base.

All tensors are `torch.float32` on CPU for artifact determinism.

Fixed probe examples are exactly V21's `build_probe_examples()`:
256 examples, seed `20260610`, sequence length 5, base 10.

For each train subject and layer `l`, compute:

- `x_l(record)`: layer input activations on fixed probes, shape above.
- `h_l(record)`: post-GELU layer output activations on fixed probes.
- `xbar_l(record) = mean_probe x_l(record)`.
- `hbar_l(record) = mean_probe h_l(record)`.

Train-only centroids:

- `target_centroid_l(behavior) = mean_records hbar_l(record)` over accepted V22 train records
  for that behavior.
- `global_centroid_l = mean_records hbar_l(record)` over all accepted V22 train records.
- `target_probe_logit_centroid(behavior) = mean_records logits(record, fixed_probe)` over
  accepted V22 train records for that behavior.

No development or final records may contribute to these statistics.

## Matched Editor

Editor method: `component_activation_rank1_editor_v22`.

For each development record and requested source-to-target direction:

1. Compute `base_weights` with `output_layer_no_signature_support_optimizer(source, target)`.
2. Copy `base_weights` into each candidate. Only one hidden layer component may be edited per
   candidate; output-layer theta remains exactly `output_layer_theta(base_weights)`.
3. For each hidden layer `l`, compute current-source means `xbar_l(source_subject)` and
   `hbar_l(source_subject)` from the source subject's fixed-probe activations.
4. Matched steering direction:
   `d_l = target_centroid_l(target) - hbar_l(source_subject)`.
5. Rank-1 edit for lambda `lambda_ridge` and scale `scale`:
   `denom = ||xbar_l||_2^2 + lambda_ridge`.
   If `denom <= 1e-12`, the candidate is invalid unless `scale == 0`.
   `Delta W_l = scale * outer(d_l, xbar_l) / denom`.
   `Delta b_l = scale * d_l`.
6. Hidden delta norm:
   `hidden_delta_norm = sqrt(||Delta W_l||_F^2 + ||Delta b_l||_2^2)`.
7. Norm cap:
   `cap = 0.5 * max(||output_layer_theta(base_weights) - output_layer_theta(source_weights)||_2, 1e-12)`.
   If `hidden_delta_norm > cap`, rescale both `Delta W_l` and `Delta b_l` by
   `cap / hidden_delta_norm`. Record `hidden_delta_clipped = true`.
8. Insert `Delta W_l` and `Delta b_l` into layer `l` using V17 component specs and evaluate.

This is a first-order post-GELU steering-to-pre-GELU-parameter approximation. The approximation
is explicit and part of the hypothesis test.

## Candidate Grid and Selection

Grid:

- layers: `[0, 1, 2, 3, 4]`
- ridge lambdas: `[0.01, 0.1, 1.0, 10.0]`
- scales: `[0.0, 0.125, 0.25, 0.5, 0.75, 1.0]`

Candidate index increments in nested-loop order: layer, lambda, scale.

Losses:

- `support_objective`: exactly `v17.support_objective_for_weights(candidate, source_weights,
  source, target)["objective"]`.
- `target_probe_centroid_loss`: mean squared error over all fixed probe logits between
  `candidate_logits` and `target_probe_logit_centroid(target)`.
- `compatible_probe_utility_loss`: mean squared error over fixed probe examples where
  `source_label(sequence) == target_label(sequence)`, comparing `candidate_logits` to
  `source_logits`. If there are zero compatible fixed-probe examples, fail closed.
- `hidden_off_manifold_loss`: for the selected candidate layer only,
  `min_behavior mean((candidate_hbar_l - target_centroid_l(behavior))^2)`. This is a
  layer-local distance to the train hidden-centroid manifold.
- `hidden_edit_norm`: hidden delta norm after clipping.

Selection tuple:

1. `support_objective`
2. `target_probe_centroid_loss`
3. `compatible_probe_utility_loss`
4. `hidden_off_manifold_loss`
5. `hidden_edit_norm`
6. `layer_index`
7. `lambda_ridge`
8. `scale`
9. `candidate_index`

The tuple is minimized exactly. No development aggregate metric may alter this grid or tuple.

## Descriptor Ablation Controls

Controls select candidates through the same grid and tuple unless stated otherwise.

- `no_probe_component_rank1_editor`: `d_l = 0` for all layers. This should usually reproduce
  the output-layer base and is proof-critical for detecting base-only wins.
- `source_probe_component_rank1_editor`: replace `target_centroid_l(target)` with
  `target_centroid_l(source)`, so the direction tests source-behavior retention rather than
  target steering.
- `shuffled_probe_component_rank1_editor`: within each source-target direction, sort jobs by
  `(source, subject_id, target)` and use the next job's full `hbar_l` values cyclically as the
  source descriptor in `d_l = target_centroid_l(target) - shuffled_hbar_l`. Use the current
  subject's `xbar_l` for the rank-1 denominator because the edit is applied to the current
  subject's parameters.
- `target_label_only_component_rank1_editor`: source-specific hidden descriptor is removed;
  `d_l = target_centroid_l(target) - global_centroid_l`.
- `nearest_target_component_rank1_editor`: use the train target record with minimum
  `sum_l ||hbar_l(source_subject) - hbar_l(target_record)||_2^2`; define
  `d_l = hbar_l(nearest_target_record) - hbar_l(source_subject)`.

## Random Controls

There are exactly 16 random controls per record:
`random_norm_matched_component_rank1_00` through `random_norm_matched_component_rank1_15`.

Random controls reuse the matched editor's selected layer, selected lambda, current subject
`xbar_l`, output-layer base theta, and final clipped hidden edit norm. They do not reuse the
matched direction.

For random control index `i`:

1. Seed payload includes: method, subject hash, source, target, index, train_statistics_hash,
   selected layer, selected lambda, selected clipped hidden edit norm hash, and script hash.
2. Seed is `int(sha256(payload)[:16], 16) % (2**63 - 1)`.
3. Use `torch.Generator(device="cpu").manual_seed(seed)` and sample `random_d_l ~ N(0, I_8)`.
4. Convert to `Delta W_l`, `Delta b_l` using the same rank-1 formula with scale 1.
5. Rescale the resulting hidden delta to exactly the matched clipped hidden edit norm unless
   either norm is below `1e-12`, in which case use the zero hidden delta.
6. Apply to the output-layer base weights, not to matched weights.

Random controls participate in Pareto domination but are excluded from best proof-critical
target-margin control selection, matching V21 semantics.

## Proof-Critical Controls and Counts

Expected controls per record: 27.

Non-random controls:

1. `no_edit`
2. `output_layer_no_signature_support_optimizer`
3. `v16_output_layer_conceptor`
4. `v17_layerwise_rank1_tsv`
5. `v20_tangent_nullspace_editor_recomputed`
6. `v21_behavioral_probe_residual_output_editor_recomputed`
7. `no_probe_component_rank1_editor`
8. `source_probe_component_rank1_editor`
9. `shuffled_probe_component_rank1_editor`
10. `target_label_only_component_rank1_editor`
11. `nearest_target_component_rank1_editor`

Random controls: 16.

Proof-critical controls are the 11 non-random controls. Pareto controls are proof-critical
controls plus all random controls. `best_control` for target-margin advantage is computed only
over proof-critical controls, not random controls.

Pareto domination epsilon: `1e-9`, same signs as V21: higher target margin and conflict target
accuracy are better; lower compatible source output MSE and delta norm are better.

## Gates

Use exact V21 gate values plus explicit V22 historical-control gates.

Aggregate gates:

- expected record count: 288
- expected controls per record: 27
- expected random controls per record: 16
- target prediction rate >= 0.85
- individual all-gate pass rate >= 0.85
- Pareto-undominated rate >= 0.85
- mean target margin >= 0.25
- mean conflict target accuracy >= 0.75
- mean conflict target accuracy improvement >= 0.25
- mean matched minus best proof-critical control target margin >= 0.02
- mean matched minus target-label target margin >= 0.02
- mean matched minus shuffled-probe target margin >= 0.05
- mean matched minus output-layer-no-signature target margin >= 0.02
- mean matched minus v16 target margin >= 0.02
- mean matched minus v17 target margin >= 0.02
- mean matched minus v20 target margin >= 0.02
- mean matched minus v21 target margin >= 0.02

Per-direction gates:

- target prediction rate >= 0.65
- individual all-gate pass rate >= 0.65
- Pareto-undominated rate >= 0.75
- mean target margin >= 0.15
- mean matched minus output-layer-no-signature target margin >= 0.01
- mean matched minus v16/v17/v20/v21 target margin >= 0.01 each

Per-record individual gates:

- predicted behavior equals target
- target margin >= 0.25
- conflict target accuracy >= 0.75
- conflict target accuracy improvement >= 0.25
- Pareto undominated
- min proof-critical compatible MSE advantage >= -0.02
- matched minus every proof-critical control target margin >= 0.02

Diagnostic negative-interpretation rules:

- If scale-0 selection rate >= V21's `0.6909722222`, any passing result is downgraded to
  inconclusive pending a reviewer decision.
- If selected-layer entropy is < 0.5 bits, report layer collapse and treat evidence as weaker,
  but this is not an automatic failure unless other gates fail.

## Artifacts

Development results must include:

- plan SHA
- formal prereg SHA
- script SHA
- helper test SHA
- source pool audit SHA
- train/development pool SHAs
- train statistics hash
- selected-layer counts and entropy
- scale-0 selection rate
- all gate failures
- no private metadata keys

## Expected Outcomes

Passing outcome: V22 beats output-layer and historical controls on the exact gates above, with
lower scale-0 collapse than V21 and no final raw access.

Negative outcome: V22 fails target prediction/Pareto/best-control gates, implying that
single-component fixed-probe hidden rank-1 edits are insufficient at this scale.

Either outcome is useful. A clean negative would argue that fixed probes diagnose activation
structure but still do not provide enough information for reliable one-step hidden editing.
