# V18 Plan: Probe-Conditioned Low-Rank Hypernetwork Weight Editing

Status: planning draft for reviewer approval. This is not a final preregistration.

## Planned Constants And Scopes

- script: `train_four_behavior_functional_weight_editing_v18_probe_conditioned_lowrank_hypernetwork.py`
- helper tests:
  `test_four_behavior_functional_weight_editing_v18_helpers.py`
- pool directory:
  `runs/four_behavior_functional_weight_editing_v18_pools`
- output directory:
  `runs/four_behavior_functional_weight_editing_v18_probe_conditioned_lowrank_hypernetwork`
- method:
  `probe_conditioned_lowrank_hypernetwork_v18`
- source pool scope:
  `four_behavior_functional_weight_editing_v18_source_pool`
- source audit scope:
  `four_behavior_functional_weight_editing_v18_source_pool_construction`
- final redacted scope:
  `redacted_final_functional_weight_editing_v18_source_pool_audit_surface_only`
- development claim scope:
  `four_behavior_functional_weight_editing_v18_probe_conditioned_lowrank_hypernetwork_development`
- final claim scope:
  `four_behavior_functional_weight_editing_v18_probe_conditioned_lowrank_hypernetwork_final`
- train base seed: `91400000`
- development base seed: `92400000`
- final base seed: `93400000`
- behavior stride: `100000`
- train accepted per behavior: `64`
- development accepted per behavior: `24`
- final accepted per behavior: `24`
- SVD rank per layer component: `8`
- hidden units: `[512, 512, 256]`
- training steps: `400`
- batch size: `64`
- optimizer: AdamW, learning rate `7e-4`, weight decay `1e-4`
- gradient clip norm: `5.0`
- model seeds: `[20260718, 20260719]`
- scale grid after hypernetwork prediction: `[0.5, 0.75, 1.0, 1.25]`
- random controls per record: `16`
- eval workers default: `min(8, cpu_count - 2)` with a floor of `1`

## Motivation

V17 was a valid negative development result. It improved over V16 on target prediction and
target margin, but it failed all proof gates and was often dominated by output-layer or
target-label controls. The next attempt should test whether fixed probe signatures can
condition a learned edit generator, while keeping the generator constrained enough that a
positive result would not be explained by unconstrained memorization or generic target-label
rewriting.

V18 therefore moves from a handcrafted nearest-target TSV selector to a train-only learned
coefficient generator:

> A fixed probe-set activation signature can condition a small hypernetwork that predicts
> low-rank layerwise edit coefficients. The decoded edit should move a source model toward a
> target behavior while preserving source-compatible behavior, and it must beat target-label,
> no-signature, shuffled-signature, output-layer, V16, V17, and random controls.

## Literature Support

- ProbeGen shows that representing a model by responses to fixed probes avoids neuron
  permutation ambiguity and can outperform heavier weight-space learners with much lower
  compute. This supports treating V18 signatures as the primary conditioning signal rather
  than raw flattened weights alone: <https://arxiv.org/html/2410.10811v2>.
- ProbeLog computes logit-level descriptors by observing model responses on fixed probes and
  uses them for model search without training metadata. This supports fixed ordered probes as
  standardized behavioral questions for a model: <https://arxiv.org/abs/2502.09619>.
- Multi-View Probing argues that first-order probe responses can miss higher-order
  row-column interactions, motivating V18's explicit controls and a low-rank coefficient
  decoder rather than claiming that probe signatures are complete:
  <https://arxiv.org/html/2605.23410v1>.
- HyperSteer trains a hypernetwork to generate steering vectors conditioned on task prompts
  and model internals, outperforming per-task supervised steering baselines on held-out
  steering prompts. V18 adapts this idea to small subject models by generating weight-edit
  coefficients from fixed probe signatures: <https://arxiv.org/html/2506.03292v1>.
- A recent weight-space learning survey describes hypernetworks as mappings from a
  conditioning signal to target-model parameters, with downstream task loss backpropagated
  through generated weights. V18 follows this pattern but emits coefficients over a
  train-only basis instead of unconstrained full weights:
  <https://arxiv.org/html/2603.10090v1>.
- "Structure Is Not Enough" shows that low structural reconstruction error in weight-space
  autoencoders can still yield functionally degraded models, and that adding behavioral
  query loss improves reconstruction/generation. V18 therefore trains with both structural
  aligned-delta loss and behavioral query loss:
  <https://arxiv.org/html/2503.17138v1>.
- Task Arithmetic shows that weight deltas can steer model behavior, while Task Singular
  Vectors show that layerwise SVD structure can reduce task interference. V18 uses
  train-only layerwise task-vector bases and learns coefficients over them:
  <https://arxiv.org/abs/2212.04089> and <https://arxiv.org/abs/2412.00081>.
- The Universal Weight Subspace Hypothesis supports the idea that same-architecture models
  reuse low-dimensional parametric subspaces, but it also cautions that subspaces must be
  learned from same-architecture train data. V18 uses fresh same-architecture train pools
  only: <https://arxiv.org/html/2512.05117v2>.
- Steer2Edit motivates the activation-to-weight bridge by converting steering signals into
  component-level rank-1 edits. V18 keeps a V17/Steer2Edit-style component edit as a direct
  control and focuses the matched method on learned coefficients:
  <https://arxiv.org/html/2602.09870v2>.

## Fresh Data Protocol

V18 must generate fresh train/development/final pools. V17 development has already been
observed, so reusing V17 development records would introduce researcher-overfitting risk.
All V18 model selection, basis fitting, hypernetwork training, and threshold decisions must
use only V18 train data and this preregistered plan.

Development may read:

- `train_subjects.json`
- `development_subjects.json`
- `combined_audit.json`
- `final_redacted_audit.json`

Development must not read:

- `final_subjects.json`
- any prior experiment final raw file
- any raw per-subject final seed, subject ID, signature, weight, metric, or hash detail

Final evaluation may run exactly once only if development passes every gate and a reviewer
confirms the artifact authorizes final. If development fails, the final raw file remains
sealed and the next action is to log the negative result.

## Final Audit Redaction Contract

`combined_audit.pool_summaries.final` may contain only:

- `accepted_counts_by_behavior`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`

The key set must match exactly. Any additional field in
`combined_audit.pool_summaries.final` invalidates development and final authorization.

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

Both the top-level `final_redacted_audit.json` key set and the nested `summary` key set
must match these allowlists exactly. Any extra field invalidates development and final
authorization even if the field name is not in the forbidden-term list.

The recursive forbidden-final-detail scanner must fail if any final redacted object contains:
`records`, `record`, `subject_id`, `subject_ids`, `seed`, `seeds`, `weights`,
`weights_hash`, `weights_hashes`, `signature`, `signature_hash`, `signature_hashes`,
`attempt_index`, `accepted_subject_ids`, raw metric lists, per-subject metric values, or
unknown detailed final fields.

## Exact Final Authorization Contract

Before any code path may open V18 `final_subjects.json`, it must load
`development_results.json` and verify all exact fields below. Any missing field, mismatched
value, mismatched file hash, or extra final redaction field aborts before opening final raw:

- `phase == "development"`
- `passed == true`
- `next_action == "eligible_for_one_shot_final_eval_without_method_changes"`
- `claim_scope == "four_behavior_functional_weight_editing_v18_probe_conditioned_lowrank_hypernetwork_development"`
- `editor_method == "probe_conditioned_lowrank_hypernetwork_v18"`
- `record_count == 288`
- every record has `random_control_count == 16`
- `train_pool_sha256 == sha256(train_subjects.json)`
- `eval_pool_sha256 == sha256(development_subjects.json)`
- `combined_audit_sha256 == sha256(combined_audit.json)`
- `final_redacted_audit_sha256 == sha256(final_redacted_audit.json)`
- `formal_prereg_sha256 == sha256(formal preregistration file)`
- `implementation_sha256 == sha256(V18 implementation script)`
- `helper_tests_sha256 == sha256(V18 helper test file)`
- `stats_sha256 == sha256(train statistics artifact)`
- `selected_hypernetwork_state_sha256 == sha256(selected matched model state)`
- `matched_seed_selection_sha256 == sha256(matched seed-selection artifact)`
- `target_label_control_state_sha256 == sha256(target-label control state)`
- `source_signature_control_state_sha256 == sha256(source-signature control state)`
- `v16_baseline_train_statistics_hash` equals the V16 baseline hash inside the stats artifact
- `v17_baseline_train_statistics_hash` equals the V17 baseline hash inside the stats artifact
- `thresholds_sha256 == stable_hash_json(thresholds embedded in artifact)`
- `constants_sha256 == stable_hash_json(constants embedded in artifact)`

The final runner must also require reviewer authorization to be represented by a local
`final_authorization.json` containing the development result SHA256, reviewer name, reviewer
confidence `5`, and an explicit authorization string. This file must not include or derive
from final raw data.

## Method

V18 reuses V17's 345-parameter subject layout, V14 Hungarian neuron alignment, V17
layer-component specifications, support objective, source-pool validation, redaction
scanner, multiprocessing evaluation harness, and summary/gating format wherever possible.

### Train-Only Basis And Training Examples

1. Normalize signatures with train-only mean and standard deviation.
2. For each source-target behavior direction, build all ordered train pairs:
   `64 source records * 64 target records = 4096` pairs.
3. Align each target record to the source record using the existing V14 deterministic
   Hungarian alignment.
4. Compute aligned full delta: `aligned_target_weights - source_weights`.
5. For each direction and each layer component, fit a centered CPU float32 SVD basis over
   train-pair component deltas. Store mean delta, rank-8 basis rows, singular values,
   explained variance, pair count, pair hash, and sign-canonicalized basis rows.
6. Project each train aligned delta into the rank-8 component bases to obtain supervised
   coefficient targets. Coefficients are ordered by the deterministic component order and
   basis-row order.
7. Store only train-derived tensors, hashes, and aggregate metadata in the stats artifact.

### Hypernetwork Inputs

For every train pair and every evaluation source-target record, the matched V18 input is:

- normalized source signature, 560 dimensions
- source behavior one-hot, 4 dimensions
- target behavior one-hot, 4 dimensions
- direction one-hot, 12 dimensions
- train-only target signature centroid for target behavior, 560 dimensions
- source signature minus train-only source-behavior centroid, 560 dimensions
- train-only target centroid minus source signature, 560 dimensions
- compact source weight statistics by component:
  mean, standard deviation, L2 norm, minimum, maximum for each V17 component

The matched input intentionally does not include development target records, final records,
or nearest development target examples.

### Hypernetwork Output And Decode

The hypernetwork outputs:

- coefficient vector over rank-8 component bases for the requested direction
- per-component nonnegative gate values in `[0, 1]`
- global edit scale in `[0, 1.5]`
- hidden-only activation-rank1 scale in `[0, 1.0]`

Decoded edit:

1. For each layer component, reconstruct:
   `component_delta = mean_delta + (gate * coefficients @ basis)`.
2. Concatenate component deltas into a 345-vector.
3. Add the V17 activation-rank1 hidden-only delta multiplied by the learned activation
   scale.
4. Multiply the full decoded delta by the learned global edit scale.
5. At evaluation, additionally try fixed post-scales `[0.5, 0.75, 1.0, 1.25]` and select
   by the preregistered source/target support objective. The selected post-scale is recorded.

### Training Loss

For each train batch, generate edited weights from train source records and train target
records only. The loss is:

- `1.0 * coefficient_mse`: MSE between predicted and projected train-pair coefficients.
- `0.5 * structural_delta_mse`: MSE between decoded delta and aligned train-pair delta.
- `2.0 * target_behavior_bce`: BCE/logit loss on train target support examples.
- `0.5 * behavioral_probe_mse`: MSE between edited-source outputs and aligned-target
  outputs on train fixed probe examples.
- `0.5 * compatible_source_mse`: source-output MSE on examples compatible with both source
  and target behavior.
- `0.02 * delta_norm_penalty`: squared decoded delta norm.
- `0.01 * gate_l1`: average gate magnitude.

The behavioral query loss uses only train probe examples and train aligned target outputs.
Development and final probe outputs are not used for training or model selection.

### Hypernetwork Architecture And Determinism

The matched and separately trained learned-control hypernetworks use the same architecture:

1. Input tensor is CPU `torch.float32`.
2. Input features are standardized by train-only mean and standard deviation computed over
   the exact train-pair feature matrix used by that model. Standard deviation is clamped to
   at least `1e-6`.
3. MLP trunk:
   `Linear(input_dim, 512) -> GELU -> Linear(512, 512) -> GELU -> Linear(512, 256) -> GELU`.
4. Output heads are separate linear layers from the 256-dimensional trunk:
   coefficient head, component-gate head, global-scale head, activation-scale head.
5. Component gates use `sigmoid`.
6. Global scale is `1.5 * sigmoid(raw_global_scale)`.
7. Activation scale is `sigmoid(raw_activation_scale)`.
8. No dropout, batch normalization, layer normalization, stochastic depth, or data
   augmentation is used.
9. Initialization is deterministic per model seed: call `torch.manual_seed(seed)`, then use
   `torch.nn.init.xavier_uniform_` for every linear weight and zero for every linear bias in
   module traversal order.
10. Training runs on CPU float32. CUDA/MPS is not used for this experiment.

Train pair rows are sorted by:
`(source_behavior, target_behavior, source_subject_id, target_subject_id)`.

For each training step `step` from `0` to `399`, batch indices are sampled with
`torch.randint(0, pair_count, (64,), generator=generator)`, where `generator` is a
CPU `torch.Generator` seeded once with the model seed before step 0. Loss reductions are
plain arithmetic means over batch examples and output dimensions. The optimizer is AdamW
with the constants listed above. Gradient norm is clipped to `5.0` after backpropagation and
before `optimizer.step()`.

Only the final step-400 model state is eligible for seed selection. There are no intermediate
checkpoint candidates.

### Train-Only Model Selection

V18 trains two matched seeds: `[20260718, 20260719]`. Selection is train-only:

1. Split train subjects deterministically by subject ID hash into inner train 75 percent and
   inner validation 25 percent, stratified by behavior.
2. For seed/checkpoint selection, fit temporary bases on inner-train subjects only and train
   hypernetwork weights only on inner-train pair examples.
3. Select the seed/checkpoint with the lowest inner-validation support objective. This
   objective uses only inner-validation subjects and the inner-train-derived temporary
   bases.
4. After selecting seed/checkpoint, refit final bases on all V18 train subjects, then refit
   that seed for the same number of steps on all V18 train pairs. This final matched model
   and full-train basis are frozen before development evaluation.
5. The stats hash must bind both the inner-selection artifact and the final refit artifact.

No development metric may affect checkpoint, seed, threshold, architecture, rank, or scale
selection.

Exact inner split:

1. For each behavior, collect its 64 V18 train subjects.
2. For each subject, compute `stable_hash_json({"scope":
   "four_behavior_functional_weight_editing_v18_inner_split", "behavior": behavior,
   "subject_id": subject_id})`.
3. Sort by `(split_hash, subject_id)`.
4. The first 48 subjects are inner-train and the final 16 subjects are inner-validation.
5. Inner-train and inner-validation subject ID sets must be disjoint within each behavior
   and across the full train pool.

Exact seed-selection objective:

1. For each candidate seed, train only the final step-400 checkpoint on inner-train pair
   examples using inner-train temporary bases.
2. Evaluate that checkpoint on all inner-validation subjects and all non-source target
   behaviors: `64 inner-validation subjects * 3 target behaviors = 192` records.
3. For each inner-validation record, try post-scales `[0.5, 0.75, 1.0, 1.25]`; select the
   post-scale with minimum inherited V17 support objective. Tie-break by smaller post-scale,
   then lexical metadata.
4. The seed score is the arithmetic mean of the selected support objective over the 192
   records.
5. Select the candidate by `(seed_score, seed, candidate_state_sha256)` ascending.
6. The seed-selection artifact records per-seed score, selected seed, selected state hash,
   inner split hash, and the 192-record aggregate only. It does not include development or
   final records.

## Controls

Each evaluated source-target record must include all proof-critical controls below. Controls
must use the same V18 train pool, the same V18 basis family where applicable, and the same
development source record.

- `no_edit`
- `target_label_lowrank_hypernetwork`: separately trained with the same architecture,
  seeds, step count, optimizer, basis construction, seed-selection protocol, coefficient
  targets, and loss weights as the matched model. Its input transform sets normalized source
  signature, source-behavior residual, and target-minus-source signature fields to all
  zeros. It keeps source behavior one-hot, target behavior one-hot, direction one-hot,
  target centroid, and compact source weight statistics unchanged.
- `source_signature_lowrank_hypernetwork`: separately trained with the same architecture,
  seeds, step count, optimizer, basis construction, seed-selection protocol, coefficient
  targets, and loss weights as the matched model. Its input transform keeps normalized
  source signature, source behavior one-hot, source-behavior residual, and compact source
  weight statistics unchanged. It replaces target behavior one-hot with source behavior
  one-hot, sets direction one-hot to all zeros, replaces target centroid with source
  behavior centroid, and sets target-minus-source signature to all zeros.
- `shuffled_signature_lowrank_hypernetwork`: same architecture and trained matched weights,
  but evaluation source signatures are cyclically shifted within each `(source_behavior,
  target_behavior)` group sorted by `(source_behavior, subject_id, target_behavior)`.
  Normalized source signature, source-behavior residual, target-minus-source signature, and
  every other source-signature-derived input field are recomputed from the shifted
  signature.
- `target_centroid_lowrank_hypernetwork`: same trained matched weights, but evaluation uses
  the train-only target behavior centroid signature instead of the source signature; all
  dependent residual fields are recomputed from that substituted signature.
- `v17_layerwise_rank1_tsv`: the V17 selector recomputed from V18 train statistics only.
- `v16_output_layer_conceptor`: the V16 baseline recomputed from V18 train statistics only.
- `output_layer_no_signature_support_optimizer`: V16 output-layer support optimizer with no
  source signature.
- `nearest_target_layerwise_tsv`: nearest target signature train record decoded through
  V17 layerwise TSV.
- `random_norm_matched_lowrank_delta_00` through
  `random_norm_matched_lowrank_delta_15`: deterministic random controls constrained to the
  same train-only basis family and norm-matched to the V18 matched delta.

The matched V18 edit must be compared against every non-random proof-critical control for
target margin and compatible-source preservation. Random controls are included for Pareto
and sanity checks.

### Random Control Algorithm

For each record and random control index `i`:

1. Build a seed payload containing V18 method name, train statistics hash, source behavior,
   target behavior, source subject ID, matched delta norm rounded by `tensor_to_hashable`,
   and random index `i`.
2. Compute `stable_hash_json(seed_payload)`, convert the first 16 hex characters to an
   unsigned 64-bit integer, and seed a CPU `torch.Generator` with that integer modulo
   `2**63 - 1`.
3. Draw one standard-normal coefficient for every active rank-8 basis row of every component
   in the requested source-target direction.
4. Decode a random basis delta with zero component means, all gates equal to `1`, no
   activation-rank1 residual, and all components active.
5. If either matched delta norm or raw random delta norm is below `1e-12`, return a zero
   delta. Otherwise scale the random delta to exactly the matched delta norm in float32.
6. Metadata records random index, seed hash, raw norm, matched norm, final norm,
   coefficient hash, basis hash, and `zero_norm_fallback`.

## Metrics And Gates

Per record:

- target prediction must equal target behavior.
- target margin must be at least `0.25`.
- conflict target accuracy must be at least `0.75`.
- conflict target accuracy improvement over source must be at least `0.25`.
- matched edit must be Pareto-undominated by proof-critical controls over
  `(higher target_margin, lower compatible_source_output_mse)`.
- matched target margin must exceed each named proof-critical non-random control by at
  least `0.02`.
- matched compatible-source MSE must be no worse than every named proof-critical non-random
  control by more than `0.02`.

Aggregate development gates:

- record count exactly `288`.
- random control count exactly `16` for every record.
- target prediction rate at least `0.85`.
- individual all-gates pass rate at least `0.85`.
- Pareto-undominated rate at least `0.85`.
- mean matched target margin at least `0.25`.
- mean conflict target accuracy at least `0.75`.
- mean conflict target accuracy improvement at least `0.25`.
- mean matched-minus-best-control target margin at least `0.02`.
- mean matched-minus-target-label target margin at least `0.02`.
- mean matched-minus-shuffled-signature target margin at least `0.05`.
- mean matched-minus-output-layer-no-signature target margin at least `0.02`.
- mean matched-minus-V17 target margin at least `0.02`.
- source, development, and final redacted pool overlap counts all zero by seed,
  subject ID, weights hash, and signature hash.

Direction-level gates for all 12 source-target directions:

- target prediction rate at least `0.65`.
- individual all-gates pass rate at least `0.65`.
- Pareto-undominated rate at least `0.75`.
- mean matched target margin at least `0.15`.
- mean matched-minus-output-layer-no-signature target margin at least `0.01`.
- mean matched-minus-V17 target margin at least `0.01`.

Failure of any gate makes the result negative/inconclusive and forbids final raw access.

## Required Artifact Hashes

Development and final result artifacts must bind:

- formal preregistration SHA256
- implementation script SHA256
- helper test file SHA256
- train pool SHA256
- eval pool SHA256
- combined audit SHA256
- final redacted audit SHA256
- train statistics SHA256
- selected hypernetwork state SHA256
- matched seed-selection artifact SHA256
- V16 recomputed baseline statistics SHA256
- V17 recomputed baseline statistics SHA256
- exact thresholds and constants
- dirty-worktree caveat

The development result file may include development subject IDs and metrics. It must not
include final raw subject IDs, seeds, signatures, weights, per-subject metrics, or raw final
hash lists. The result text itself must not contain `final_subjects.json`.

## Required Tests Before Pool Generation

Add pytest coverage for:

- V18 scopes, paths, and seeds differ from V15, V16, and V17.
- V18 final raw guard rejects all `runs/**/final_subjects.json` paths.
- final redacted audit scanner fails closed on forbidden keys and unknown detailed final
  fields.
- train statistics hash changes when train weights, train signatures, basis tensors,
  probe examples, hypernetwork state, or selected seed metadata changes.
- all train-pair tables contain train subject IDs only.
- inner-validation split is deterministic, behavior-stratified, and disjoint from
  inner-train.
- hypernetwork input builders do not accept development or final target records.
- matched, target-label, source-signature, and shuffled-signature input ablations alter only
  preregistered fields.
- decoded low-rank delta lies in the train-only basis family plus the explicit
  activation-rank1 residual.
- random low-rank controls are deterministic, norm-matched, and basis-constrained.
- shuffled signatures are cyclically shifted within source-target direction groups.
- serial and multiprocessing evaluation produce byte-identical sorted records.
- gate failures fire when V18 is Pareto-dominated by any proof-critical control.
- summary stdout redacts verbose pool details and final audit details.

Run only targeted pytest and `py_compile`; do not run lint automatically.

## Compute Plan

- Pool generation may use existing PyTorch internal parallelism.
- Hypernetwork training runs serially per seed to avoid CPU oversubscription, with
  `torch.set_num_threads(max(1, cpu_count // 2))`.
- Inner-validation evaluation for seed selection may use process parallelism only after each
  seed has finished training.
- Development evaluation uses the existing process pool with at most 8 workers and one
  PyTorch thread per worker.
- Long-running phases must print summary-only progress and never print raw final details.

## Review And Stopping Rules

1. Reviewer must approve this plan at confidence `5/5` before implementation starts.
2. Reviewer must approve implementation and helper tests at confidence `5/5` before pool
   generation.
3. Reviewer must approve V18 source-pool construction at confidence `5/5` before
   development.
4. Reviewer must approve V18 development results at confidence `5/5`.
5. Final raw access is allowed only if development passes every gate and reviewer confirms
   final authorization. Otherwise, final remains sealed.
