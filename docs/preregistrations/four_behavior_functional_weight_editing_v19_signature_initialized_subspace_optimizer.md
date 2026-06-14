# V19 Preregistration: Signature-Initialized Subspace Support Optimizer

Status: formal preregistration. Implementation must follow this document.

## Planned Constants And Scopes

- script: `train_four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer.py`
- helper tests: `test_four_behavior_functional_weight_editing_v19_helpers.py`
- pool directory: `runs/four_behavior_functional_weight_editing_v19_pools`
- output directory: `runs/four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer`
- method: `signature_initialized_subspace_support_optimizer_v19`
- source pool scope: `four_behavior_functional_weight_editing_v19_source_pool`
- source audit scope: `four_behavior_functional_weight_editing_v19_source_pool_construction`
- final redacted scope:
  `redacted_final_functional_weight_editing_v19_source_pool_audit_surface_only`
- development claim scope:
  `four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer_development`
- final claim scope:
  `four_behavior_functional_weight_editing_v19_signature_initialized_subspace_optimizer_final`
- train base seed: `101400000`
- development base seed: `102400000`
- final base seed: `103400000`
- behavior stride: `100000`
- train accepted per behavior: `64`
- development accepted per behavior: `24`
- final accepted per behavior: `24`
- basis rank per layer component: `8`
- optimizer steps: `80`
- optimizer: Adam, learning rate `0.08`, betas `(0.9, 0.999)`, eps `1e-8`,
  weight decay `0.0`, amsgrad `false`
- gradient clip norm: `5.0`
- coefficient L2 to signature initialization: `0.01`
- source-weight L2: `0.0005`
- delta L2: `0.0005`
- gate L1: `0.002`
- post-scale grid: `[0.5, 0.75, 1.0, 1.25]`
- random controls per record: `16`
- eval workers default: `min(8, cpu_count - 2)` with a floor of `1`

## Motivation

V18 was a valid negative development result. Its learned amortized low-rank hypernetwork
failed badly: target prediction was `0.038`, individual pass rate was `0`, target margin was
negative, and V17 outperformed it on target-margin advantage. This suggests that the fixed
probe signatures are more useful as local retrieval/initialization signals than as inputs to
a small global decoder trained from limited source pools.

V19 tests a narrower and more promising claim:

> Fixed probe signatures can initialize and regularize a constrained per-record optimizer in
> a train-only task-vector basis, improving functional source-to-target edits beyond
> target-label, no-signature, shuffled-signature, V16, V17, output-layer, and random controls.

This is intentionally not a source-free decoder claim. It is a small-scale test of whether
signatures provide useful optimization geometry for functional weight edits.

## Literature Support

- Task-vector theory and task-vector-basis work argue that storing or optimizing in a
  task-vector basis can be more flexible than committing to one merged vector, while still
  saving memory relative to full task-vector banks: <https://arxiv.org/html/2502.01015v3>.
- Task-localized sparse fine-tuning identifies that task arithmetic needs localized,
  low-interference updates; V19 therefore optimizes only a train-derived low-rank basis and
  keeps source-compatible preservation as a proof gate:
  <https://arxiv.org/html/2504.02620v1>.
- LoRI and orthogonal/low-rank adaptation work show that constraining adaptation subspaces
  can reduce cross-task interference and forgetting. V19 uses low-rank basis-constrained
  updates rather than unconstrained full-vector edits:
  <https://arxiv.org/html/2504.07448v2>.
- Subspace Control frames steering as constrained spectral optimization. V19 uses this
  principle in the small subject-model setting: optimize coefficients in a fixed subspace
  rather than learning a global decoder: <https://arxiv.org/html/2604.04231v1>.
- In-place test-time training motivates fast per-instance adaptation without retraining the
  whole model. V19 is an offline small-model analogue: optimize low-dimensional edit
  coefficients for the requested source-target record:
  <https://arxiv.org/html/2604.06169v1>.
- Model-editing studies on ROME/MEMIT warn that edits can bleed into unrelated behavior and
  cause forgetting, so V19 keeps compatible-source MSE, Pareto domination, random controls,
  and output-layer/no-signature controls as hard gates:
  <https://arxiv.org/html/2401.07453v3>.
- Steer2Edit and TSV continue to motivate component-level and layerwise low-rank updates,
  but V18 showed that amortizing such updates was not enough here. V19 keeps layerwise
  bases but moves the adaptation to per-record coefficient optimization:
  <https://arxiv.org/html/2602.09870v2> and <https://arxiv.org/abs/2412.00081>.

## Fresh Data Protocol

V19 must generate fresh train/development/final pools. V16, V17, and V18 development results
have all been observed, so V19 cannot reuse their development records without
researcher-overfitting risk.

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

`combined_audit.pool_summaries.final` key set must be exactly:

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
invalidates development and final authorization. Forbidden terms include `records`,
`record`, `subject_id`, `subject_ids`, `seed`, `seeds`, `weights`, `weights_hash`,
`weights_hashes`, `signature`, `signature_hash`, `signature_hashes`, `attempt_index`,
`accepted_subject_ids`, raw metric lists, and per-subject metric values.

Before opening V19 final raw, the final runner must verify:

- `phase == "development"`
- `passed == true`
- `next_action == "eligible_for_one_shot_final_eval_without_method_changes"`
- `claim_scope == DEVELOPMENT_SCOPE`
- `editor_method == "signature_initialized_subspace_support_optimizer_v19"`
- record count `288`
- every record has `16` random controls
- train, development, combined audit, final redacted audit, formal prereg, implementation,
  helper test, stats, constants, and thresholds hashes match current files
- selected train-statistics hash equals the hash embedded in the stats artifact
- V16 and V17 baseline statistics hashes match the stats artifact
- reviewer authorization artifact exists and references the exact development result hash
  with reviewer confidence `5`, reviewer name, ISO timestamp, and explicit authorization
  string `authorize_v19_one_shot_final_eval`

Any mismatch aborts before opening final raw.

## Method

V19 reuses reviewed V17/V18 primitives where possible:

- V14 deterministic Hungarian neuron alignment
- V17 layer-component layout and train-only SVD basis construction
- V17 activation-rank1 residual
- V16 output-layer conceptor and no-signature output optimizer controls
- V18 exact redaction, source-pool validation, random basis-control algorithm, and
  multiprocessing evaluation contract

### Train-Only Statistics

For each source-target behavior direction:

1. Normalize train signatures with train-only mean/std.
2. Build all ordered train source-target pairs.
3. Align each target model to each source model with V14 Hungarian alignment.
4. Compute aligned deltas.
5. Fit per-layer-component rank-8 SVD bases over centered aligned deltas.
6. Store mean deltas, basis rows, singular values, explained variance, pair hashes, signature
   centroids, probe examples hash, and baseline V16/V17 statistics.

The train statistics hash must bind all stored bases, signature normalization tensors,
centroids, probe examples, thresholds, constants, pair hashes, and baseline hashes.

### Signature Initialization

For a development source-target record:

1. Normalize the source signature.
2. Select the top-8 train target records by normalized signature distance within the target
   behavior.
3. Align each selected target to the source.
4. Compute a softmax-weighted aligned delta.
5. Project that delta into the train-only layerwise basis and extract initial coefficients.
6. Build the V17 activation-rank1 residual from the same selected targets and weights.

This initialization is the only signature-conditioned advantage. The optimizer then gets the
same support objective budget as the controls.

Exact signature selection:

1. Source signature is normalized as `(signature - train_sig_mean) / train_sig_std`, with
   train standard deviation clamped to `1e-6`.
2. Each candidate train target signature uses the same train-only normalization.
3. Distance is `mean((z_target - z_source) ** 2)` in float32.
4. Candidates are sorted by `(distance, subject_id)`.
5. Select `min(8, candidate_count)` records.
6. Weights are `softmax(-distance / 1.0)` over the selected distances.
7. Artifact metadata records only rank order, distance, softmax weight, and SHA256 of the
   selected target `subject_id`.
8. No development or final target records may enter this selection.

Exact coefficient initialization:

1. For each selected train target, align target weights to source weights with V14 Hungarian
   alignment.
2. Weighted aligned delta is the softmax-weighted sum of aligned target deltas.
3. For each layer component, subtract the train-only component mean and project into the
   component's rank-8 basis to obtain `c_signature_init`.
4. The activation-rank1 residual is computed from the same selected aligned target weights
   and softmax weights.
5. `c_signature_init`, activation residual hash, selected-target metadata, basis hash, and
   weighted-delta hash are recorded.

### Matched Optimizer

Optimize coefficient vector `c`, component gates `g`, activation scale `a`, and global scale
`s` for 80 Adam steps. Decode:

`weights = source + s * (basis_delta(c, sigmoid(g)) + sigmoid(a) * activation_rank1_delta)`

Loss per step:

- V17 support objective target BCE
- V17 support objective conflict BCE
- V17 compatible-source MSE
- source-weight L2
- `0.01 * ||c - c_signature_init||^2`
- `0.0005 * ||decoded_delta||^2`
- `0.002 * mean(sigmoid(g))`

Initialization:

- `c = c_signature_init`
- `g = 2.0` for every component
- `a = logit(0.5)`
- `s = logit(1.0 / 1.5)`

Optimizer semantics:

- CPU `torch.float32` only; CUDA/MPS are not used.
- Variables are leaf tensors `c_raw`, `g_raw`, `a_raw`, `s_raw`.
- `c = c_raw`.
- component gates are `sigmoid(g_raw)`.
- activation scale is `sigmoid(a_raw)`.
- global scale is `1.5 * sigmoid(s_raw)`.
- Adam optimizes exactly `[c_raw, g_raw, a_raw, s_raw]` with the constants above.
- For each step `0..79`, compute full support loss over all V17 support tensors for that
  source-target direction. Reductions are arithmetic means over all logits/examples.
- Loss is:
  - `1.0 * target_bce`
  - `1.0 * conflict_bce`
  - `1.0 * compatible_source_mse`
- `0.0005 * source_weight_l2`
  - `0.01 * mean((c - c_init) ** 2)`
  - `0.0005 * mean(decoded_delta ** 2)`
  - `0.002 * mean(sigmoid(g_raw))`
- Gradients are clipped after `loss.backward()` and before `optimizer.step()`.
- There is no early stopping, no intermediate checkpoint selection, and no stochastic
  sampling.
- The final step-80 variables are the only optimizer state eligible for post-scale
  selection.
- `source_weight_l2` is `mean((weights - source_weights) ** 2)` over all 345 flat
  parameters. `decoded_delta_l2` is `mean(decoded_delta ** 2)` over the same 345-vector;
  these are equal for the matched decode but both are recorded separately so future decoder
  variants cannot silently change semantics.

After optimization, evaluate fixed post-scales `[0.5, 0.75, 1.0, 1.25]` and select by the
same support objective. Select by the exact tuple
`(support_objective, delta_norm, post_scale, candidate_index)` ascending, where
`candidate_index` is the zero-based index in `POST_SCALE_GRID`.

### Controls

Proof-critical controls:

- `no_edit`
- `target_label_subspace_optimizer`: same optimizer and budget, initialized from target
  behavior centroid delta, no source signature.
- `no_signature_zero_subspace_optimizer`: same optimizer and budget, initialized at zero
  coefficients and no activation-rank1 residual.
- `shuffled_signature_subspace_optimizer`: same optimizer and budget, initialized from a
  cyclically shifted source signature within each source-target direction.
- `source_signature_subspace_optimizer`: same optimizer and budget, initialized from source
  behavior train records rather than target behavior train records.
- `v17_layerwise_rank1_tsv`: V17 recomputed on V19 train statistics.
- `v16_output_layer_conceptor`: V16 recomputed on V19 train statistics.
- `output_layer_no_signature_support_optimizer`: V16 output-layer no-signature optimizer.
- `nearest_target_layerwise_tsv`: nearest train target through V17 TSV.
- `random_norm_matched_lowrank_delta_00` through `_15`: deterministic V18-style
  basis-constrained random controls, norm-matched to matched delta.

Random controls participate in Pareto checks. Named non-random controls participate in target
margin and compatible-source preservation advantage gates.

Exact control initialization:

- `target_label_subspace_optimizer`:
  - `c_init` is the projection of the train-only target-behavior centroid delta for the
    requested source-target direction.
  - centroid delta is the mean aligned delta over all train source-target pairs in that
    direction.
  - activation residual is zero.
  - coefficient regularization anchor is this centroid `c_init`.
- `no_signature_zero_subspace_optimizer`:
  - `c_init` is all zeros.
  - activation residual is zero.
  - coefficient regularization anchor is zeros.
- `shuffled_signature_subspace_optimizer`:
  - source signatures are cyclically shifted within each `(source_behavior, target_behavior)`
    group sorted by `(source_behavior, subject_id, target_behavior)`.
  - every signature-derived field is recomputed from the shifted signature.
  - `c_init`, activation residual, and coefficient anchor are built by the exact matched
    signature-initialization algorithm using the shifted signature.
- `source_signature_subspace_optimizer`:
  - target retrieval pool is train records from the source behavior, not the target behavior.
  - distance, sorting, softmax, projection, activation residual, and coefficient anchor use
    the exact matched algorithm with that source-behavior pool.
- All controls use the same optimizer settings, 80 steps, post-scale grid, and exact
  `(support_objective, delta_norm, post_scale, candidate_index)` tie-break as matched.

### Random Control Algorithm

For random control index `i`:

1. Build seed payload with method string
   `signature_initialized_subspace_support_optimizer_v19`, train statistics hash, source
   behavior, target behavior, source subject ID, matched delta norm converted with
   `tensor_to_hashable`, and random index `i`.
2. Compute `stable_hash_json(seed_payload)`.
3. Convert first 16 hex characters to unsigned integer modulo `2**63 - 1`.
4. Seed CPU `torch.Generator` with that value.
5. Draw one standard-normal coefficient for every rank-8 basis row of every layer component
   in the source-target direction.
6. Decode with zero component means, all gates equal to `1`, global scale `1`, and no
   activation-rank1 residual.
7. If matched delta norm or random raw norm is below `1e-12`, return zero delta and record
   `zero_norm_fallback=true`; otherwise scale random delta to exactly matched delta norm in
   float32.
8. Metadata records random index, seed hash, raw norm, matched norm, final norm, coefficient
   hash, basis hash, and zero-norm fallback.

## Metrics And Gates

Use V18 gates unchanged:

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

## Required Tests

Before pool generation, targeted pytest must cover:

- V19 scopes, seeds, and paths differ from V18.
- generic final raw guard rejects all `runs/**/final_subjects.json` paths.
- exact final redaction allowlists fail closed.
- train statistics hash binds bases, signatures, centroids, probe examples, constants,
  thresholds, and baseline hashes.
- signature initialization uses train target records only.
- target-label, no-signature, shuffled-signature, and source-signature controls alter only
  preregistered initialization sources.
- optimizer tie-breaks are deterministic.
- random controls are deterministic, basis-constrained, and norm-matched.
- random controls enter Pareto checks.
- shuffled signatures cycle within each source-target direction.
- serial and multiprocessing evaluation are byte-identical under a deterministic evaluator.
- summary stdout redacts verbose pool and final details.

Run only targeted pytest and `py_compile`; do not run lint automatically.

## Compute Plan

- Pool generation may use PyTorch internal parallelism.
- Train statistics are computed once per phase and cached.
- Per-record coefficient optimization is embarrassingly parallel and must run with the
  existing process pool, defaulting to 8 workers and one torch thread per worker.
- Long-running phases print summary-only progress and never print raw final details.

## Review Gates

1. Reviewer must approve this plan at confidence `5/5` before formal preregistration.
2. Reviewer must approve the formal preregistration copy at `5/5`.
3. Reviewer must approve implementation/tests at `5/5` before pool generation.
4. Reviewer must approve source-pool construction at `5/5` before development.
5. Reviewer must approve development results at `5/5`.
6. Final raw remains sealed unless development passes and reviewer explicitly authorizes
   final evaluation.
