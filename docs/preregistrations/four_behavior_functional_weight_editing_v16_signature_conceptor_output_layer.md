# V16 Preregistration: Signature-Conditioned Conceptor Steering Compiled To Output-Layer Edits

Status: formal preregistration converted from the reviewer-approved V16 plan.
Do not generate V16 pools or run development until implementation has focused
tests and a 5/5 implementation review.

## Motivation

V15 is a valid negative/inconclusive development result. It moved many heldout
subjects toward target behavior, but it failed proof-grade gates: 6/288
individual all-gate passes, target prediction rate 0.822917, Pareto-undominated
rate 0.125, and failure of the V13 no-signature hard advantage gate. The
important diagnostic is not "no signal"; mean conflict target accuracy was
0.888996 and mean target margin was 0.510108. The issue is causal specificity:
the matched hypernetwork was often beaten by retrieval, no-signature support
optimization, or target-label controls.

V16 will therefore test a narrower and more direct claim before attempting
another full-weight generator:

> Fixed-probe activation signatures identify behavior-specific activation
> steering operators that can alter a heldout subject model's behavior, preserve
> compatible source behavior, beat shuffled/label-only/no-signature
> output-layer controls, and be compiled into an actual output-layer weight edit.

This is deliberately not a claim that V16 solves arbitrary full-weight editing
or beats an unconstrained full-network optimizer. It is a bridge experiment:
activation signatures -> representation steering -> functional model with
altered behavior via a small, auditable weight edit.

## Literature Support

- Turner et al. (2024), *Steering Language Models With Activation Engineering*,
  https://arxiv.org/abs/2308.10248. The paper introduces activation engineering
  as inference-time activation modification and reports that activation
  additions can steer high-level output properties while preserving off-target
  behavior. V16 adopts the activation-intervention framing but tests it in small
  subject models with preregistered controls.
- Postmus and Abreu (2024), *Steering Large Language Models using Conceptors*,
  https://arxiv.org/abs/2410.16314. The paper motivates conceptors as soft
  projection matrices over activation sets rather than single steering vectors.
  V16 uses train-only conceptor operators over last-hidden activations.
- Ilharco et al. (2023), *Editing Models with Task Arithmetic*,
  https://arxiv.org/abs/2212.04089. Task vectors motivate parameter-space
  behavior editing, but V14/V15 showed linear/full-weight transfer is not enough
  here. V16 keeps task-vector and label-only baselines as controls rather than
  assuming task-vector success.
- Ainsworth et al. (2022), *Git Re-Basin: Merging Models modulo Permutation
  Symmetries*, https://arxiv.org/abs/2209.04836. Hidden-unit permutation
  symmetry can make raw weight operations misleading. V16 avoids most hidden
  permutation ambiguity by intervening at the source model's own last hidden
  layer and compiling only the output layer.
- Zhou et al. (2023), *Permutation Equivariant Neural Functionals*,
  https://arxiv.org/abs/2302.14040. NFNs motivate symmetry-aware processing of
  neural weights and network editing. V16 is not an NFN, but it responds to the
  same symmetry risk by avoiding flattened full-weight generation.
- Dayan et al. (2026), *On the Expressive Power of Permutation-Equivariant
  Weight-Space Networks*, https://arxiv.org/abs/2602.01083. The paper
  distinguishes function-space and weight-space maps. V16 explicitly tests a
  function-space steering map first, then compiles that map into a restricted
  weight-space edit.
- Kaushik et al. (2025), *The Universal Weight Subspace Hypothesis*,
  https://arxiv.org/abs/2512.05117. The universal-subspace result motivates
  low-dimensional structure in trained weights, but V16 treats that as
  background rather than proof. The primary V16 object is an 8-dimensional
  last-hidden activation operator, not a full 345-dimensional raw-weight delta.
- Przewiezlikowski et al. (2022), *HyperMAML: Few-Shot Adaptation of Deep Models
  with Hypernetworks*, https://arxiv.org/abs/2205.15745. V15 tested the
  hypernetwork route and failed its gates. V16 records this as a reason to test
  an interpretable operator before returning to learned full-weight generation.

## Candidate Approaches Considered

1. **Another full-weight hypernetwork with more capacity.**
   - Upside: closest to decoding signatures into weights.
   - Downside: V15 already shows capacity is not the core issue; it risks more
     compute without resolving causal specificity or control leakage.

2. **Permutation-equivariant learned functional.**
   - Upside: strongest weight-space-learning literature match.
   - Downside: heavier implementation and harder audit surface; likely too much
     machinery before proving the activation-signature bridge.

3. **Signature-conditioned activation conceptors compiled to output-layer
   edits.**
   - Upside: directly tests representation steering, has a closed-form
     compile-to-weights step, reduces permutation ambiguity, and creates
     stronger controls around signature specificity.
   - Downside: claim is narrower than full functional decoding.

Recommendation: use approach 3 for V16.

## Frozen Data Policy

V16 proof-grade evaluation requires fresh V16 pools. V13/V14/V15 pools and
development results may be referenced only as prior evidence and diagnostics.
They must not be reused for proof-eligible V16 development or final evaluation.

Fresh V16 pool directory:

`runs/four_behavior_functional_weight_editing_v16_pools`

Pool configuration:

- train base seed `78300000`, `64` accepted subjects per behavior, maximum
  `128` attempts per behavior;
- development base seed `79300000`, `24` accepted subjects per behavior,
  maximum `64` attempts per behavior;
- final base seed `80300000`, `24` accepted subjects per behavior, maximum
  `64` attempts per behavior;
- behavior stride `100000`;
- behavior order remains:
  `sorted_ascending`, `sorted_descending`, `has_majority`, `mountain_pattern`.

Claim scopes:

- raw source pools:
  `four_behavior_functional_weight_editing_v16_source_pool`;
- combined audit:
  `four_behavior_functional_weight_editing_v16_source_pool_construction`;
- final redacted audit:
  `redacted_final_functional_weight_editing_v16_source_pool_audit_surface_only`;
- development result:
  `four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer_development`;
- final result:
  `four_behavior_functional_weight_editing_v16_signature_conceptor_output_layer_final`.

The final redacted audit allowlist and
`combined_audit.pool_summaries.final` allowlist are listed explicitly in the
Leak Guards section below. Any subject-level final details on a public surface
before authorized final evaluation invalidates V16.

Final authorization is hash-bound: the final phase may run only if the
development artifact has `passed: true`,
`next_action: eligible_for_one_shot_final_eval_without_method_changes`, and
matches the exact train pool hash, development pool hash, combined audit hash,
final redacted audit hash, implementation hash, preregistration hash, method
string, and development claim scope. Otherwise final raw remains sealed.

## Method

### Train-Only Statistics

For train subjects only:

1. Build the fixed probe examples exactly as V15:
   `build_digit_probe_examples(n_examples=256, seed=20260610, seq_len=5,
   base=10)`.
2. Fit train-only signature mean/std from train subject signatures.
3. Fit the V9 target-attractor classifier/calibration from train subjects only.
4. For each train subject, compute last-hidden activations on the fixed probe
   inputs using the subject's own weights. The last hidden layer is the fifth
   GELU output, shape `[256, 8]`.
5. Store train-only per-subject last-hidden activation mean `mu_i`, covariance
   `R_i = H_i^T H_i / n`, and one conceptor per frozen aperture:
   `C_i[aperture] = R_i @ inverse(R_i + aperture^-2 * I + ridge * I)`.

Frozen conceptor parameters:

- aperture grid for primary matched candidate selection:
  `[0.5, 1.0, 2.0, 4.0, 8.0]`;
- ridge for matrix inverse: `1e-4`;
- activation dtype: `torch.float32`;
- no development/final subject may enter conceptor fitting, signature
  normalization, classifier/calibration fitting, target signature selection, or
  hyperparameter selection.

### Signature-Conditioned Target Operator

For a development source subject and requested target behavior:

1. Compute source normalized signature `z_source` using train-only stats.
2. Compute the V9 target-attractor selected target signature `z_target` using
   train-only V9 statistics/classifier, as in V15.
3. Among train subjects with the requested target behavior, compute softmax
   weights by negative signature MSE to `z_target`.
4. For each aperture value, form weighted target conceptor and mean:
   - `C_target[aperture] = sum_i w_i C_i[aperture]`;
   - `mu_target = sum_i w_i mu_i`.
5. Compute the source subject's own last-hidden activations on fixed probes and
   derive source mean/covariance and `C_source[aperture]` for every frozen
   aperture using the same conceptor formula.
6. Define the last-hidden steering operator:
   - `A = I + alpha * (C_target[aperture] - C_source[aperture])`;
   - `shift = beta * (mu_target - mu_source)`;
   - `h_steered = h @ A.T + shift`.

Frozen scale grids:

- `alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`;
- `beta_grid = [0.0, 0.25, 0.5, 0.75, 1.0]`;
- aperture grid as above.

Candidate selection uses only source support inputs and labels implied by the
requested target behavior. No heldout development metrics may select a
candidate. Matched and all conceptor/output-layer controls sweep the same
frozen aperture/alpha/beta grids under their own conditioning surface, then
select their final candidate by the same support-only objective and deterministic
tie-break: lowest support objective, then smaller aperture, then smaller alpha,
then smaller beta.

### Compile To Output-Layer Edit

Let the source output layer be `logit = h @ W_out.T + b_out`, with
`W_out` shape `[1, 8]`. The intervention
`h_steered = h @ A.T + shift` is compiled to a real edited model by changing
only the output layer:

- `W_out_edited = W_out @ A`;
- `b_out_edited = b_out + W_out @ shift`.

All hidden-layer weights remain exactly the source weights. The development
artifact must verify, per record, that the explicit activation intervention and
the compiled output-layer edit produce matching logits on support and heldout
inputs within absolute tolerance `1e-5`. A record with failed compile
equivalence cannot pass.

### Controls

Each development record must include these non-random controls:

- `no_edit`;
- `target_label_centroid_conceptor`: target behavior centroid conceptor/mean,
  no signature similarity weighting;
- `shuffled_signature_conceptor`: target behavior fixed, but the target
  signature is deterministically replaced by a signature selected from a
  different target behavior;
- `source_signature_conceptor`: uses the source signature as the target
  signature;
- `nearest_train_target_conceptor`: one nearest train target subject by
  signature, not a weighted conceptor;
- `activation_addition_mean_shift`: `A = I`, signature-weighted target/source
  mean shift only;
- `activation_conceptor_no_shift`: `beta = 0`, conceptor operator only;
- `output_layer_no_signature_support_optimizer`: output-layer-only optimizer
  using source support labels for the requested target behavior, no signature
  term;
- `output_layer_random_conceptor`: random PSD conceptor with norm matched to
  the target conceptor;
- `v13_no_signature_support_optimizer`: full-weight no-signature optimizer from
  the existing V13/V14/V15 control path, reported as context but not a required
  V16 gate because V16's claim is output-layer compiled activation steering,
  not full-network optimization supremacy.

Random controls:

- `16` norm-matched random output-layer edits per record;
- deterministic random seeds derived from `(subject_id, target_behavior,
  control_index, V16_RANDOM_CONTROL_SEED=20260630)`;
- random controls must not inspect heldout outcomes during generation.

Expected controls per record: `26` (`10` non-random controls plus `16`
random).

Record schema clarification: matched is stored as the top-level `matched`
record, not inside `controls`. Therefore `controls` contains `26` entries:
`10` non-random controls plus `16` random controls. The development artifact
must report both `expected_controls_per_record = 26` and
`expected_non_random_controls_per_record = 10`.

### No-Signature Output-Layer Optimizer Control

`output_layer_no_signature_support_optimizer` is frozen as follows:

- optimize only the output layer weight and bias initialized from the source
  model; all hidden weights remain fixed;
- optimizer: AdamW;
- steps: `250`;
- learning rate: `0.05`;
- betas: `(0.9, 0.999)`;
- weight decay: `1e-4`;
- gradient clipping norm: `10.0`;
- seed: stable hash of
  `("v16_output_layer_no_signature_support_optimizer", subject_id,
  target_behavior, 20260631)`;
- support-only objective:
  - target support BCE on source support examples relabeled with the requested
    target behavior: weight `4.0`;
  - compatible source support logit MSE to the original source logits: weight
    `0.05`;
  - output-layer delta MSE to source output layer: weight `0.0005`;
- no signature term, no train target activations, no development heldout
  metrics, and no final raw access;
- final-step policy only: evaluate the parameters after step `250`; do not
  early-stop or select by heldout metrics;
- deterministic tie-break is irrelevant because there is exactly one final
  checkpoint.

### Development Evaluation

Evaluate every development subject against every non-source target behavior:

`4 source behaviors * 24 subjects * 3 targets = 288 records`.

For each matched/control edited model, report:

- predicted behavior by the train-only classifier;
- target margin;
- compatible source output MSE;
- compatible source accuracy;
- conflict source accuracy;
- conflict target accuracy;
- conflict target accuracy improvement over source;
- compile equivalence max absolute logit difference for matched candidates;
- selected aperture, alpha, beta, support-only objective;
- signature similarity summary and selected train target IDs redacted to hashes.

Development evaluation should parallelize across records using a frozen
`ProcessPoolExecutor` contract:

- `max_workers = min(8, os.cpu_count() or 1)`;
- multiprocessing start method: `spawn`;
- each worker calls `torch.set_num_threads(1)` before evaluation;
- the parent process precomputes train-only immutable stats and passes only
  read-only serializable payloads to workers;
- no worker may read or write result files;
- per-record random seeds are derived only from stable hashes of
  `(subject_id, target_behavior, control_type, control_index)` and never from
  worker id or completion order;
- worker failure aborts the whole phase without writing a partial result;
- final records are sorted by `(source_behavior, subject_id, target_behavior)`
  before hashing/writing;
- a serial debug mode must produce byte-identical aggregate metrics and record
  order on the same implementation.

Any nondeterministic record ordering, worker-count-dependent seed, or partial
result write invalidates the development artifact.

## Gates

V16 is considered development-passing only if all gates pass.

Aggregate gates:

- `n == 288`;
- source-pool audit passed;
- all overlap counts are zero;
- exactly `26` controls per record;
- exactly `16` random controls per record;
- aggregate target prediction rate >= `0.90`;
- aggregate conflict target accuracy >= `0.80`;
- aggregate conflict target accuracy improvement >= `0.35`;
- aggregate individual all-gate pass rate >= `0.85`;
- aggregate Pareto-undominated rate among V16-gating controls >= `0.85`;
- mean matched target margin >= `0.35`;
- mean matched-minus-target-label target margin >= `0.02` OR mean
  target-label-minus-matched compatible source-output MSE >= `2.0`;
- mean matched-minus-shuffled-signature target margin >= `0.05` OR mean
  shuffled-signature-minus-matched compatible source-output MSE >= `2.0`;
- mean matched-minus-output-layer-no-signature target margin >= `0.02` OR mean
  output-layer-no-signature-minus-matched compatible source-output MSE >= `2.0`;
- compile equivalence max absolute logit difference <= `1e-5` for every matched
  record.

Per-direction gates:

- `24` records per direction;
- target prediction rate >= `0.85`;
- conflict target accuracy >= `0.65`;
- mean target margin >= `0.15`;
- individual all-gate pass rate >= `0.70`;
- Pareto-undominated rate among V16-gating controls >= `0.70`.

Per-record all-gate pass requires:

- matched predicted behavior equals requested target behavior;
- matched target margin >= `0.15`;
- conflict target accuracy >= `0.65`;
- conflict target accuracy improvement >= `0.15`;
- matched beats target-label, shuffled-signature, and output-layer-no-signature
  controls by either target margin or compatible-MSE preservation:
  - target margin advantage >= `0.02`; OR
  - compatible source-output MSE advantage >= `2.0`;
- matched is Pareto-undominated by the V16-gating controls on
  `(target_margin, compatible_source_output_mse)`;
- compile equivalence max absolute logit difference <= `1e-5`.

V16-gating controls exclude only the full `v13_no_signature_support_optimizer`.
They include all output-layer/conceptor controls listed above and all random
output-layer controls. The full V13 no-signature optimizer remains reported as
a context control and must be discussed honestly, but it does not gate V16
because it tests a broader full-network optimization method class.

## Leak Guards

- V16 must generate fresh V16 pools before development.
- No V16 final raw file may be opened before development passes and reviewer
  approves final authorization.
- V16 development may read only train subjects, development subjects, combined
  audit, and final redacted audit.
- Train-only stats hash must include:
  - fixed probe example hash;
  - train subject IDs and hashes;
  - signature mean/std hash;
  - per-subject train conceptor hashes;
  - aperture grid;
  - V9 classifier/calibration summary hash;
  - control config hash;
  - multiprocessing configuration.
- Development artifact must include:
  - implementation SHA256;
  - preregistration SHA256;
  - train/development pool SHA256;
  - combined audit SHA256;
  - final redacted audit SHA256;
  - explicit flags that prior final raw and V16 final raw were not opened.
- If `passed: false`, artifact `next_action` must be
  `log_negative_development_result_do_not_open_final_raw`.
- If `passed: true`, artifact `next_action` must be
  `eligible_for_one_shot_final_eval_without_method_changes`.

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
details, or other final subject-level details invalidates V16. Any extra field
in `combined_audit.pool_summaries.final` before authorized final evaluation
invalidates the V16 final pool for proof use.

## Tests Required Before Source Pools

Add focused pytest coverage before implementation is accepted:

- conceptor construction returns symmetric finite matrices with eigenvalues in
  `[0, 1]` within tolerance;
- signature-weighted target conceptor uses only train target-behavior subjects;
- shuffled/source/target-label controls produce distinct deterministic
  conditioning surfaces;
- output-layer compile equivalence matches explicit hidden activation
  intervention within `1e-5`;
- random controls are deterministic and norm-matched;
- record-ordering is stable under parallel evaluation;
- serial and parallel development evaluation produce identical record order and
  aggregate metrics on a small fixture;
- final redacted allowlists contain exactly the explicitly listed allowed
  fields and no extras;
- development cannot run if source-pool contract fails;
- final cannot run unless development artifact is hash-bound and passed.

Do not run Ruff/lint. Run only focused pytest and py_compile unless the user
explicitly requests linting.

## Reviewer Questions

1. Does the literature support the narrowed claim that fixed-probe signatures
   can select representation-steering operators, not broad full-weight decoding?
2. Are the output-layer compile step and equivalence gate sufficient to call the
   result a functional model edit rather than only an inference hook?
3. Are the target-label, shuffled-signature, source-signature, no-signature
   output-layer optimizer, and random controls enough to rule out the main
   misleading explanations?
4. Is it acceptable that the full V13 no-signature optimizer is context-only
   rather than a V16 gate, given the narrowed V16 claim?
5. Are the proposed thresholds strict enough to avoid turning a weak partial
   movement effect into an overstated positive result?
6. Does deterministic multiprocessing preserve auditability, or should V16
   prioritize single-process evaluation despite poorer compute use?
