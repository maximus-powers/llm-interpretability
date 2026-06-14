# V21 Plan: Behavioral-Probe Residual Output Editor

Status: planning draft for reviewer critique. Do not implement until reviewer confidence is 5/5.

## Claim Scope

This plan tests whether fixed probe responses can be decoded into a functional weight edit when the
edit is anchored on the strongest observed no-signature baseline rather than replacing that baseline.
V21 is a development-only method until it passes all preregistered gates and receives reviewer
approval. V21 must generate fresh train/development/final source pools before development
evaluation. V20 final raw remains sealed.

## Prior Result

V20 is a valid negative/inconclusive development result. It produced:

- target prediction `127/288 = 0.4409722222`
- individual all-gate pass `0/288 = 0.0`
- Pareto-undominated `69/288 = 0.2395833333`
- mean target margin `0.1201667197`
- mean matched minus best-control target margin `-0.2321420627`
- mean matched minus output-layer no-signature target margin `-0.0856206854`
- mean matched minus V17 target margin `0.0556163218`
- mean matched minus V16 target margin `0.1770446601`

The important diagnostic is not that fixed-probe information is useless. V20 beat several
signature controls and V16/V17 on target margin, but it lost badly to the best and output-layer
controls. V21 therefore treats the no-signature output-layer support optimizer as the base edit and
asks whether probe-derived behavioral descriptors can add a residual that improves target behavior
without breaking compatibility.

## Literature Support

Research pass performed on June 13, 2026.

- Kahana et al., "Deep Linear Probe Generators for Weight Space Learning"
  ([arXiv:2410.10811](https://arxiv.org/abs/2410.10811)). ProbeGen shows that probing can be a
  strong standalone route for weight-space learning and that structured probe generators reduce
  overfitting while requiring far fewer FLOPs than heavier weight-space architectures. V21 adopts
  the central lesson: use fixed probe responses as behavioral descriptors, not merely as weak priors.
- Kahana et al., "Can this Model Also Recognize Dogs? Zero-Shot Model Search from Weights"
  ([arXiv:2502.09619](https://arxiv.org/abs/2502.09619)). ProbeLog represents individual logits by
  their responses on a fixed probe set. V21 adapts this logit-level descriptor idea to the small
  four-behavior subject models by using output and hidden probe-response descriptors to condition
  residual output-layer edits.
- Horwitz et al., "Learning on Model Weights using Tree Experts"
  ([arXiv:2410.13569](https://arxiv.org/abs/2410.13569)). ProbeX shows that hidden-layer probing
  can be lightweight and effective within related model families. V21 uses the fact that all subject
  models share architecture and training procedure by adding hidden activation views to the probe
  descriptor.
- Heo et al., "What Linear Probes Miss: Multi-View Probing for Weight-Space Learning"
  ([arXiv:2605.23410](https://arxiv.org/abs/2605.23410)). MVProbe argues that single-view probing
  misses higher-order row-column interactions and adds interaction-aware Gram views. V21 includes
  probe activation means plus Gram/correlation summaries rather than output signatures alone.
- Meynent et al., "Structure Is Not Enough: Leveraging Behavior for Neural Network Weight
  Reconstruction" ([arXiv:2503.17138](https://arxiv.org/abs/2503.17138)). This paper shows that
  low structural weight error does not guarantee functional reconstruction, and behavioral losses
  improve weight generation. V21 trains residual targets against behavior/probe losses rather than
  Euclidean deltas alone.
- "Generative Modeling of Weights: Generalization or Memorization?"
  ([arXiv:2506.07998](https://arxiv.org/html/2506.07998v1)) reports memorization patterns in
  generated checkpoints. V21 avoids full-model generation and instead learns small residual edits
  with explicit train-only splits, norm controls, random residual controls, and fresh source pools.
- "Weight Space Should Be a First-Class Generative AI Modality"
  ([arXiv:2605.18632](https://arxiv.org/html/2605.18632v1)) frames weight-space generation as a
  useful but governance-sensitive modality requiring rigorous provenance and evaluation protocols.
  V21 keeps provenance via hashes, redaction, source-pool audits, and reviewer gates.

## High-Level Method

V21 is a residual editor:

```text
base_weights = output_layer_no_signature_support_optimizer(source_weights, source, target)
probe_residual = decoder(probe_descriptor(source_model), target_behavior)
candidate_weights = source_weights with output layer from base_weights + scale * probe_residual
```

The matched V21 edit may change only the output layer. This is intentional: V20 showed that
full-weight tangent edits were not competitive with output-layer controls. V21 asks whether fixed
probe descriptors add useful target-specific residual information on top of the strong output-layer
baseline.

## Fresh Pools

V21 must not reuse V16 through V20 development or final raw records.

Use fresh seeds:

- train base seed: `114400000`
- development base seed: `115400000`
- final base seed: `116400000`
- behavior stride: `100000`

Accepted counts:

- train: `64` per behavior, `256` total
- development: `24` per behavior, `96` total
- final: `24` per behavior, redacted until final authorization

The source-pool contract must include:

- no seed, subject ID, weights hash, or signature hash overlap across train/development/final
- final redacted audit with only allowlisted count/hash fields
- combined audit final summary with only allowlisted final count/hash fields
- reviewer confidence `5/5` before any development evaluation

## Subject Weight Layout

V21 uses the existing 345-parameter subject-model layout:

- non-output weights: `weights[0:336]`
- output weight: `weights[336:344]`, reshaped to `(1, 8)`
- output bias: `weights[344]`
- output-layer parameter vector `theta_out`: concatenate `weights[336:344]` and `weights[344]`,
  producing length `9`

The matched V21 edit and every V21 residual control may change only `theta_out`. All hidden-layer
weights remain exactly equal to the source subject weights.

## Probe Descriptor

All descriptor statistics are computed from train-pool fitted normalization only.

For a subject model and the fixed V21 probe examples:

Probe examples are frozen exactly as:

```python
v16.v15.build_digit_probe_examples(
    n_examples=256,
    seed=20260610,
    seq_len=5,
    base=10,
)
```

Descriptor extraction uses float32 tensors and this exact order:

1. `output_signature`: existing fixed-probe output vector from the inherited signature function,
   normalized by train mean/std.
2. `penultimate_mean`: mean of the post-GELU penultimate hidden activation over the 256 probes,
   length `8`.
3. `penultimate_std`: population standard deviation (`unbiased=False`) of the post-GELU
   penultimate hidden activation over probes, length `8`.
4. `penultimate_gram_upper`: compute `G = H.T @ H / 256` for post-GELU penultimate activations
   `H` with shape `(256, 8)`, then flatten the upper triangle in row-major order
   `(0,0), (0,1), ..., (0,7), (1,1), ..., (7,7)`, length `36`.
5. `output_logit_stats`: raw source logits over probes, summarized as
   `[mean, std(unbiased=False), min, max]`, length `4`.
6. `output_logit_gram`: scalar `mean(logits ** 2)`, length `1`.
7. `target_centroid_delta`: train-only difference between target and source behavior descriptor
   centroids, using the normalized descriptor components above.
8. `target_one_hot`: one-hot target behavior in `PATTERNS` order.

Normalization:

- Fit componentwise mean/std from accepted V21 train subjects only.
- If train std is `< 1e-6`, replace std with `1.0` and record the zero-variance component count
  in the train statistics artifact.
- Clamp normalized descriptor components to `[-10.0, 10.0]`.
- Descriptor concatenation order is exactly the numbered order above.

The descriptor vector must not include development labels, final labels, final records, source-pool
candidate metadata outside the accepted train payload, or raw subject IDs. Hash payloads may include
train-only descriptor statistics and train-only centroid hashes.

## Train Residual Targets

For each ordered train pair `(source_subject, target_behavior)`:

1. Compute `base_weights` with the existing no-signature output-layer support optimizer using only
   public support examples for `source` and `target`.
2. Freeze the source hidden layers and build a linear design matrix from penultimate post-GELU
   activations with a bias column. For an input batch `x`, each row is
   `concat(penultimate_activation(source_hidden, x), 1.0)`, length `9`.
3. Compute `oracle_probe_output_layer_weights` with deterministic weighted ridge least squares.
   The target vector uses logits, not probabilities:
   - target support rows use desired logits `+2.0` for positive target examples and `-2.0` for
     negative target examples.
   - compatible source-preservation rows use the original source model logits on compatible
     support examples.
   - probe rows use the train-only target behavior output-signature centroid converted to logits
     by clamping probabilities to `[1e-4, 1 - 1e-4]` and applying `log(p / (1 - p))`.
4. Solve this objective:

```text
min_theta
  w_target * mean((X_target theta - y_target)^2)
+ w_preserve * mean((X_preserve theta - y_preserve)^2)
+ w_probe * mean((X_probe theta - y_probe_centroid)^2)
+ lambda_base * mean((theta - theta_base)^2)
```

Exact grids:

- `w_target = 1.0`
- `w_preserve = 1.0`
- `w_probe in [0.25, 0.5, 1.0]`
- `lambda_base in [0.01, 0.1, 1.0, 10.0]`

Closed-form solve:

- Convert means to row weights by dividing each block weight by its row count.
- Let `A = X_weighted.T @ X_weighted + lambda_base / 9 * I`.
- Let `b = X_weighted.T @ y_weighted + lambda_base / 9 * theta_base`.
- Use `torch.linalg.solve(A, b)`.
- If solve fails, retry once with `A + 1e-6 * I`.
- If the retry fails, mark that oracle hyperparameter candidate invalid.
- If all oracle candidates fail for a pair, omit that pair from decoder training and record
  `oracle_pair_failed=true` in train statistics.

5. Select the oracle candidate for the pair by ascending tuple:
   `(support_objective, probe_centroid_mse, compatible_source_output_mse, output_delta_norm,
   w_probe, lambda_base)`.
6. Define `residual_target = theta_oracle - theta_base`.

The residual target is train-only. No development records may be used to fit residual targets,
normalization, hyperparameters, or decoder weights.

## Decoder

Fit a deterministic ridge residual decoder from train pairs only. V21 intentionally removes the MLP
option from the plan to avoid optimizer nondeterminism at this stage.

Inputs:

- source probe descriptor
- target behavior one-hot
- target-minus-source descriptor centroid
- base edit summary metrics computed on public support only, in the exact order below

Outputs:

- output-layer residual delta, length `9`

Base edit summary vector:

Compute `base_weights = output_layer_no_signature_support_optimizer(...)`, then compute:

1. `base_support_objective`: `v17.support_objective_for_weights(... )["objective"]`
2. `base_target_bce`: same function, key `"target_bce"`
3. `base_conflict_bce`: same function, key `"conflict_bce"`
4. `base_compatible_mse`: same function, key `"compatible_mse"`
5. `base_source_l2`: same function, key `"source_l2"`
6. `base_target_margin`: `v16.v15.v14.functional_metrics(base_weights, source, target, source_weights)["target_margin"]`
7. `base_conflict_target_accuracy`: same metrics dict, key `"conflict_target_accuracy"`
8. `base_output_delta_norm`: L2 norm of `theta_base - theta_source`

These eight values are appended after target one-hot. They are normalized with the same inner-train
input standardization as the rest of the decoder input. Their statistics are fit from inner-train
pairs only during hyperparameter selection, and from all accepted train pairs only after the selected
hyperparameters are frozen for development.

Hyperparameters:

- ridge lambda: `[0.01, 0.1, 1.0, 10.0, 100.0]`
- residual target normalization: `[none, per_component_train_std]`
- residual norm cap multiplier relative to base output-layer delta: `[0.25, 0.5, 1.0]`

Ridge details:

- Add an intercept column of ones to the decoder input matrix.
- Fit input componentwise mean/std on inner-train pairs only. If std `< 1e-6`, set std to `1.0`.
- Fit output residual normalization on inner-train pairs only when
  `residual target normalization == per_component_train_std`; if output std `< 1e-6`, set std to
  `1.0`.
- Solve `(X.T @ X + lambda * I) W = X.T @ Y` with the intercept column excluded from L2 penalty.
- Use `torch.linalg.solve`; retry once with `+1e-6 * I`; if retry fails, mark the hyperparameter
  invalid.
- Prediction de-normalizes outputs if output residual normalization was enabled.
- No stochastic training, no minibatches, no optimizer, no checkpoint selection.

Select hyperparameters by deterministic inner train split only:

- split payload: `{"method": "v21_inner_split", "subject_id": subject_id}`
- split hash: SHA256 of canonical JSON payload
- split is per source behavior, not global
- sort each behavior's 64 accepted train subjects by `(split_hash, subject_id)`
- inner train: first `51` subjects per behavior
- inner validation: remaining `13` subjects per behavior
- train residual decoder pairs include only source subjects in the inner-train split; target behavior
  may be any of the other three behaviors because target centroids are aggregate train-only behavior
  statistics, not target subject records
- inner validation pairs include only source subjects in the inner-validation split
- selection tuple ascending:
  `(inner_validation_support_objective, inner_validation_probe_centroid_loss, residual_norm,
  ridge_lambda, residual_target_normalization_order, norm_cap_multiplier)`

Inner-validation scoring:

For each decoder hyperparameter candidate:

1. Fit descriptor/input normalizers, output residual normalizers, oracle residual targets, and ridge
   decoder weights on inner-train source subjects only.
2. For each inner-validation source subject and ordered target behavior, run the same candidate
   algorithm defined in "Development Candidate Selection":
   - compute base output-layer edit
   - compute descriptor and base summary vector using inner-train normalizers
   - predict residual
   - clip residual before scaling to `norm_cap_multiplier * max(||theta_base - theta_source||, 1e-12)`
   - evaluate scales `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25]`
   - select by the same candidate tuple
3. A validation pair is invalid only if base edit construction, descriptor extraction, or every scale
   candidate fails. If any pair is invalid, mark the hyperparameter candidate invalid and exclude it
   from selection.
4. Reduce validation metrics by arithmetic mean over all valid inner-validation pairs.
5. `inner_validation_support_objective` is the mean selected `v17.support_objective_for_weights`
   objective.
6. `inner_validation_probe_centroid_loss` is the mean squared error between selected probe logits
   and the train-only target behavior probe-logit centroid.
7. `residual_norm` is the mean selected clipped-and-scaled residual norm.
8. If all hyperparameter candidates are invalid, fail closed before development evaluation.

## Development Candidate Selection

For each development record and ordered target:

1. Compute the no-signature output-layer base edit.
2. Compute V21 descriptor using train-fitted descriptor normalizers.
3. Predict residual.
4. Apply residual candidate scales `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25]`.
5. Clip residual norm to the selected train-only cap.
6. Evaluate candidates on public support/probe objectives only.
7. Select matched V21 candidate with ascending tuple:
   `(support_objective, probe_centroid_loss, compatible_source_output_mse, target_margin_negative,
   residual_norm, scale, candidate_index)`.

`scale=0.0` is the base no-signature output-layer edit and must be included as a candidate, but if
selected it still counts as the matched V21 method. The controls separately record the base edit so
the result can show whether V21 actually adds value.

## Controls

Each development record must include exactly `27` controls:

Non-random controls (`11`):

1. `no_edit`
2. `output_layer_no_signature_support_optimizer`
3. `v16_output_layer_conceptor`
4. `v17_layerwise_rank1_tsv`
5. `v20_tangent_nullspace_editor_recomputed`
6. `no_probe_residual_output_editor`
7. `source_probe_residual_output_editor`
8. `shuffled_probe_residual_output_editor`
9. `target_label_only_residual_output_editor`
10. `nearest_target_probe_residual_output_editor`
11. `oracle_train_centroid_probe_residual_output_editor`

Control descriptor substitutions:

- `no_probe_residual_output_editor`: descriptor vector is all zeros except target one-hot and public
  support base metrics; uses the same selected decoder and norm cap.
- `source_probe_residual_output_editor`: replace `target_centroid_delta` with the source behavior
  centroid minus itself, i.e. zero target delta; keep the source subject descriptor.
- `shuffled_probe_residual_output_editor`: within each ordered `(source, target)` development job
  group, sort jobs by `(source, target, subject_id)` and assign the next job's source descriptor
  cyclically; target one-hot is unchanged.
- `target_label_only_residual_output_editor`: zero all source descriptor components and
  target-centroid-delta components; keep only target one-hot and base support metrics.
- `nearest_target_probe_residual_output_editor`: choose the accepted V21 train subject of the target
  behavior with smallest Euclidean distance between normalized descriptor vectors; tie-break by
  `(distance, train_subject_id)`. Use that train subject's descriptor as the source descriptor
  substitute while keeping the development source base edit.
- `oracle_train_centroid_probe_residual_output_editor`: replace the source descriptor with the
  train-only target behavior descriptor centroid; keep target one-hot and support base metrics.

All descriptor-substitution controls use the same selected decoder, same selected residual
normalization, same selected norm cap, and same candidate scale grid as matched V21. None may train
or select separate hyperparameters.

Random controls (`16`):

- `random_norm_matched_probe_residual_00` through `random_norm_matched_probe_residual_15`
- each random residual is output-layer-only, norm-matched to the selected V21 residual, generated
  from a CPU RNG seed payload containing method, train statistics hash, source behavior, target
  behavior, subject stable hash, matched residual norm hash, decoder config hash, and random index.

Random residual algorithm:

1. Seed payload canonical JSON:

```json
{
  "decoder_config_hash": "<selected_decoder_config_hash>",
  "index": <0_to_15>,
  "matched_residual_norm_hash": "<sha256_tensor_hash>",
  "method": "behavioral_probe_residual_output_editor_v21",
  "source": "<source_behavior>",
  "subject_hash": "<stable_subject_hash_without_raw_id_in_result>",
  "target": "<target_behavior>",
  "train_statistics_hash": "<train_statistics_hash>"
}
```

2. Compute `seed_hash = sha256(canonical_json).hexdigest()`.
3. Convert to CPU RNG seed with `int(seed_hash[:16], 16) % (2**63 - 1)`.
4. Create `torch.Generator(device="cpu").manual_seed(seed)`.
5. Draw `raw = torch.randn(9, generator=generator, dtype=torch.float32)`.
6. If `raw.norm() < 1e-12`, use the first standard basis vector `[1, 0, ..., 0]`.
7. If selected V21 residual norm `< 1e-12`, random residual is all zeros.
8. Otherwise `random_residual = raw / raw.norm() * selected_residual_norm`.
9. Add the random residual to `theta_base`, leaving hidden weights unchanged.
10. Metadata fields: `control_type`, `seed_hash`, `random_seed`, `matched_residual_norm`,
    `random_residual_norm`, `decoder_config_hash`, `train_statistics_hash`, `index`.

The `oracle_train_centroid_probe_residual_output_editor` may use train-only target centroid
descriptors but must not use development target records. It is a control because it does not use the
source subject's fixed-probe descriptor.

## Gates

Use V20 aggregate and direction gates unchanged except for the expected total control count:

- expected records: `288`
- expected controls per record: `27`
- expected random controls per record: `16`
- aggregate target prediction rate: `>= 0.85`
- aggregate individual all-gate pass rate: `>= 0.85`
- aggregate Pareto-undominated rate: `>= 0.85`
- mean target margin: `>= 0.25`
- mean conflict target accuracy: `>= 0.75`
- mean conflict target accuracy improvement: `>= 0.25`
- mean matched minus best-control target margin: `>= 0.02`
- mean matched minus target-label target margin: `>= 0.02`
- mean matched minus shuffled-signature target margin: `>= 0.05`
- mean matched minus output-layer-no-signature target margin: `>= 0.02`
- mean matched minus V17 target margin: `>= 0.02`
- per-direction target prediction rate: `>= 0.65`
- per-direction individual pass rate: `>= 0.65`
- per-direction Pareto-undominated rate: `>= 0.75`
- per-direction target margin: `>= 0.15`
- per-direction output-layer-no-signature target margin advantage: `>= 0.01`
- per-direction V17 target margin advantage: `>= 0.01`

Gate/control mapping:

| Gate metric name | Exact control type |
| --- | --- |
| `best_control` | max over all non-random controls listed above |
| `target_label` | `target_label_only_residual_output_editor` |
| `shuffled_signature` | `shuffled_probe_residual_output_editor` |
| `output_layer_no_signature` | `output_layer_no_signature_support_optimizer` |
| `v17` | `v17_layerwise_rank1_tsv` |
| `v16` | `v16_output_layer_conceptor` |
| `no_signature` | `no_probe_residual_output_editor` |
| `source_signature` | `source_probe_residual_output_editor` |
| `nearest_target` | `nearest_target_probe_residual_output_editor` |
| `v20` | `v20_tangent_nullspace_editor_recomputed` |

Target-margin and compatible-preservation advantage gates use the named non-random controls in the
table. Pareto dominance uses every control, including all 16 random controls, `no_edit`, and all
non-random named controls. `best_control` excludes random controls so random controls cannot alone
make the best-control margin gate impossible, but random controls can still dominate in Pareto.

If any gate fails, V21 is a valid negative/inconclusive development result and final evaluation is
not authorized.

## Leakage and Misleading-Result Defenses

- V21 train/development/final pools are fresh.
- Final raw may be written only by source-pool generation and remains sealed after generation.
- Development code may read train raw, development raw, combined audit, and final redacted audit
  only.
- Final runner must verify the development result hash, source-pool hashes, train statistics hash,
  implementation hash, formal prereg hash, and reviewer authorization before opening final raw.
- Development result must include `next_action = "log_negative_development_result_do_not_open_final_raw"`
  unless all gates pass.
- The result must report total control counts and random control counts per record.
- Matched metadata must strip private tensors and selected bases before JSON serialization.
- Hyperparameter selection must be completed from train inner validation before any development
  metrics are computed.
- Random controls must be generated after the matched residual is selected and must be norm-matched.
- If the matched V21 candidate selects `scale=0.0` often and does not beat
  `output_layer_no_signature_support_optimizer` on the preregistered advantage gates, the result is
  negative for the probe-residual claim even if raw target prediction is high. The result summary
  must report `scale_0_selection_count` and `scale_0_selection_rate`.

## Final Redaction and Authorization

Combined audit final summary allowlist:

- `accepted_counts_by_behavior`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`

Final redacted audit top-level allowlist:

- `behavior_suite_hashes`
- `candidate_pool_summary_hash`
- `claim_scope`
- `config_hash`
- `pool`
- `pool_file_sha256`
- `pool_redacted_payload_sha256`
- `summary`
- `summary_payload_sha256`

Final redacted audit `summary` allowlist:

- `accepted_counts_by_behavior`
- `max_selected_train_vs_heldout_overlap_count`

Recursive forbidden final-detail terms in final-redacted and combined-final surfaces:

- `records`
- `weights`
- `signature`
- `subject_id`
- `seed`
- `attempt_index`
- `accepted_index`
- `train_info`
- `weights_hash`
- `signature_hash`
- any key ending in `_logits`
- any key ending in `_sequence`

Final authorization must be a JSON object with exactly these fields:

- `authorization_scope`
- `reviewer`
- `reviewer_confidence`
- `reviewer_authorization_string`
- `formal_prereg_sha256`
- `implementation_sha256`
- `helper_tests_sha256`
- `development_result_sha256`
- `train_pool_sha256`
- `development_pool_sha256`
- `combined_audit_sha256`
- `final_redacted_audit_sha256`
- `train_statistics_sha256`
- `phase`
- `claim_scope`
- `editor_method`
- `next_action`
- `record_count`
- `passed`

The final runner may open V21 final raw only if:

- `reviewer_confidence == "5/5"`
- `reviewer_authorization_string == "V21 final evaluation authorized"`
- `phase == "development"`
- `claim_scope == "four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor_development"`
- `editor_method == "behavioral_probe_residual_output_editor_v21"`
- `next_action == "eligible_for_one_shot_final_eval_without_method_changes"`
- `record_count == 288`
- `passed is true`
- all hashes in the authorization object match files on disk

## Reviewer Checkpoints

1. Reviewer must approve this plan at confidence `5/5`.
2. Formal preregistration must be created by copying the approved plan and changing only the title
   and status lines; reviewer must approve the formal copy at `5/5`.
3. Implementation must be test-first. Helper tests must fail before implementation and pass after.
4. Reviewer must approve implementation at `5/5` before source-pool generation.
5. Reviewer must approve source-pool generation at `5/5` before development evaluation.
6. Reviewer must approve development result at `5/5`.
7. Final evaluation is authorized only if the development result passes all gates and reviewer returns
   explicit `5/5` final authorization.
