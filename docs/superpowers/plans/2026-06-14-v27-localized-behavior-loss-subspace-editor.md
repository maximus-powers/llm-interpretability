# V27 Localized Behavior-Loss Subspace Editor Plan

Date: 2026-06-14

## Motivation

V26 showed that direct train-only centroid task vectors can produce weak
relative margin signals without functional target flips. The next experiment
should stop relying on a single global delta and instead solve a localized edit
against behavior loss in a small subspace, while preserving the sealed split and
the V25 proof controls.

The goal is still development-only: discover whether a localized, behavior-loss
optimized edit can produce reliable target behavior on the development split.
No final raw data may be opened or read unless a later reviewer-approved
development result justifies a hash-bound final run.

## Literature Basis

- Ilharco et al., "Editing Models with Task Arithmetic"
  (https://arxiv.org/abs/2212.04089): task vectors show that weight-space
  directions can steer model behavior, but V26 demonstrates that global
  centroid arithmetic is insufficient in this setting.
- Fierro and Roger, "Steering Language Models with Weight Arithmetic"
  (https://arxiv.org/abs/2511.05408): contrastive weight directions can
  outperform activation steering in some behavioral settings; this supports
  keeping weight edits as a serious path, but with contrastive/localized
  objectives rather than naive centroid deltas.
- Ortiz-Jimenez et al., "Task Arithmetic in the Tangent Space"
  (https://arxiv.org/abs/2305.12827): task arithmetic improves when edits are
  disentangled and localized in function space. This motivates optimizing in a
  small tangent/subspace basis instead of adding a global weight vector.
- Meng et al., "Locating and Editing Factual Associations in GPT"
  (https://arxiv.org/abs/2202.05262): ROME succeeds by identifying localized,
  directly editable computations and applying rank-one writes. The small MLP
  analog is to restrict the edit to behavior-relevant blocks/bases and solve a
  targeted behavioral write.
- Meng et al., "Mass-Editing Memory in a Transformer"
  (https://arxiv.org/abs/2210.07229): MEMIT scales direct editing by solving
  explicit parameter updates over mediating layers. The relevant lesson here is
  to solve a constrained update, not only retrieve or average deltas.
- Adila et al., "Weight Updates as Activation Shifts"
  (https://arxiv.org/html/2603.00425v1): weight updates and activation shifts
  can play complementary roles, and intervention site matters. This supports
  evaluating both output/logit behavior loss and activation-signature controls.
- Sun et al., "HyperSteer"
  (https://arxiv.org/html/2506.03292v1): supervised steering-vector generation
  can generalize when conditioned on internal state. V27 remains smaller and
  non-hypernetworked, but keeps conditioning on activation signatures and
  source/target labels.
- Kaushik et al., "The Universal Weight Subspace Hypothesis"
  (https://arxiv.org/abs/2512.05117): shared low-dimensional spectral subspaces
  appear across trained models. This supports testing spectral/train-delta
  bases as a compact editable subspace.
- "A Survey of Weight Space Learning"
  (https://arxiv.org/pdf/2603.10090): frames weight-space representation and
  generation as a coherent modality, supporting the broader MUAT research
  direction while keeping V27's claim narrow.

## Hypothesis

A behavior-loss optimized edit in a localized low-dimensional subspace will
produce stronger target behavior than V25/V26's descriptor ridge or centroid
task-vector edits, while preserving source outputs better than broad
full-vector optimization.

## Non-Claims

- V27 will not claim final performance.
- V27 will not claim large-model transfer.
- V27 will not claim universal MUAT evidence.
- V27 will not claim decoded functional models unless target prediction and
  proof-control gates pass on development, then on a later sealed final run.

## Proposed Implementation

Extend
`model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
instead of creating a disconnected runner.

Add a new matched edit source:

```python
"localized_behavior_loss_subspace"
```

The source computes a small basis `B` and optimizes coefficients `alpha`:

```text
edited_weights = source_weights + B @ alpha
```

Candidate basis options:

- `spectral_train_delta_rank4`: current V25/V26 train-delta spectral basis,
  computed from train-only edit banks and sign-canonicalized by making the
  largest-absolute coordinate positive.
- `target_source_logit_gradient_rank4`: the orthonormal basis from the first
  four right singular vectors of the support-loss gradient rows listed below.
- `combined_spectral_gradient_rank8`: concatenation of the spectral and
  gradient bases, then QR-orthonormalized with the same sign canonicalization.
- `output_layer_topk`: standard basis over output-layer coordinates selected by
  absolute value of the mean target-support gradient minus source-compatible
  preservation gradient. Ties break by lower flattened coordinate index.

All basis constructors must return a matrix shaped
`(SOURCE_WEIGHT_DIM, rank)`, a hash-only provenance payload, and no raw basis
values in progress logs. Rank fallback is allowed only when the candidate rank
exceeds the available independent directions; rank `0`, nonfinite values, or
wrong dimensions are contract-invalid.

Candidate objective:

```text
loss =
  target_bce_weight * BCE(target_logits, target_labels)
  + conflict_bce_weight * BCE(conflict_logits, conflict_target_labels)
  + source_mse_weight * MSE(compatible_logits, source_compatible_logits)
  + delta_l2_weight * ||B @ alpha||^2
  + norm_barrier_weight * max(0, ||B @ alpha|| - norm_cap)^2
```

Use train/development support tensors from existing V14/V17 helpers, not final
records. Keep the evaluation metrics and V25 proof controls unchanged.

## Optimization/Evaluation Boundary

V27 is not allowed to optimize on heldout proof rows.

Coefficient optimization may use only deterministic support tensors produced by
`prepare_support_tensors_with_source_logits(source_weights, source, target)`,
which are generated from the fixed support split in the V10/V13 helper chain:

- `target_inputs` and `target_labels` from `suite["support"][target]`;
- `conflict_inputs` and `conflict_target_labels` from source-support rows where
  source and target predicates disagree;
- `compatible_inputs` and `compatible_source_logits` from source-support rows
  where source and target predicates agree.

Proof metrics remain the existing V25/V10 heldout evaluation metrics:
`behavior_margin()`, `functional_metrics()`, and V25 proof controls. Support
objective values may be logged only as redacted scalar diagnostics and cannot be
used as proof metrics.

If a later implementation intentionally optimizes on any per-record development
heldout row, the result must be labeled transductive and cannot be used as
ordinary heldout development evidence. This V27 plan does not authorize that.

## Exact Basis Definitions

Gradient rows for `target_source_logit_gradient_rank4` are computed at the
source weights using float32 CPU tensors:

- row group `target_positive`: gradient of mean BCE target loss on
  `target_inputs`;
- row group `conflict`: gradient of mean BCE conflict loss on
  `conflict_inputs`;
- row group `compatible`: gradient of compatible-source MSE on
  `compatible_inputs`;
- row group `source_l2`: gradient of mean squared distance to source weights,
  included only as a zero row at initialization and recorded for shape/hash
  stability.

Stack nonzero finite rows, center by row mean, run `torch.linalg.svd`, take the
top `rank` right singular vectors, transpose to `(SOURCE_WEIGHT_DIM, rank)`, and
sign-canonicalize each column. If SVD fails or any value is nonfinite, mark the
candidate contract-invalid.

For `output_layer_topk`, use the known output-layer flattened slice from the
V16/V17 helpers. Select `rank=8` coordinates by descending absolute value of:

```text
mean_gradient(target BCE on target_inputs)
- source_mse_weight * mean_gradient(compatible MSE on compatible_inputs)
```

Return a standard-basis matrix with selected coordinates in sorted selection
rank order. The provenance hash includes basis type, rank, selected coordinate
hash, source/target labels, support split counts, train pool file hash, train
summary hash, and script hash.

## Leakage Controls

- Keep `assert_no_forbidden_final_raw_paths()` unchanged.
- Do not load or inspect
  `runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- Use only `train_subjects.json` and `development_subjects.json`.
- Log hashes/counts/provenance only; no raw weights, subject IDs, support
  examples, signatures, logits, Jacobians, or deltas in progress logs.
- Include train pool file SHA-256 and train summary hash in any learned/solved
  subspace provenance.
- Keep shuffled-signature, source-target ablation, no-signature, random,
  nearest-train, teacher-oracle, and prior-editor controls.

## Initial Grid

Use a bounded first grid:

- basis: `spectral_train_delta_rank4`,
  `target_source_logit_gradient_rank4`, `combined_spectral_gradient_rank8`,
  `output_layer_topk`
- steps: `25`, `75`
- learning rate: `0.05`, `0.01`
- source MSE weight: `0.5`, `1.0`
- delta L2 weight: `0.0`, `0.01`
- norm cap: `0.25`

This gives `64` configs. The first smoke run should use only the first `8`
configs with successive halving on `12,24` balanced jobs, matching the V26
compute pattern.

Loop order must be exactly:

```python
for steps in [25, 75]:
    for lr in [0.05, 0.01]:
        for source_mse_weight in [0.5, 1.0]:
            for delta_l2_weight in [0.0, 0.01]:
                for basis in [
                    "spectral_train_delta_rank4",
                    "target_source_logit_gradient_rank4",
                    "combined_spectral_gradient_rank8",
                    "output_layer_topk",
                ]:
                    ...
```

The first eight configs therefore compare all four bases at:

| Config index | Steps | LR | Source MSE | Delta L2 | Basis |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 25 | 0.05 | 0.5 | 0.0 | spectral_train_delta_rank4 |
| 1 | 25 | 0.05 | 0.5 | 0.0 | target_source_logit_gradient_rank4 |
| 2 | 25 | 0.05 | 0.5 | 0.0 | combined_spectral_gradient_rank8 |
| 3 | 25 | 0.05 | 0.5 | 0.0 | output_layer_topk |
| 4 | 25 | 0.05 | 0.5 | 0.01 | spectral_train_delta_rank4 |
| 5 | 25 | 0.05 | 0.5 | 0.01 | target_source_logit_gradient_rank4 |
| 6 | 25 | 0.05 | 0.5 | 0.01 | combined_spectral_gradient_rank8 |
| 7 | 25 | 0.05 | 0.5 | 0.01 | output_layer_topk |

Add a grid-count/order/hash test before compute.

## Optimizer Protocol

- Optimize only `alpha`, never the full weight vector.
- Initialize `alpha = zeros(rank)`.
- Optimizer: `torch.optim.AdamW([alpha], lr=lr, betas=(0.9, 0.999),
  eps=1e-8, weight_decay=0.0, amsgrad=False)`.
- Device/dtype: CPU float32.
- Gradient clipping: `clip_grad_norm_([alpha], 5.0)`.
- Step logging: record only every `max(1, steps // 5)` plus final step, with
  scalar losses, finite status, alpha norm, delta norm, and hash-only basis
  provenance. Do not log alpha values or delta values.
- Norm handling: the barrier is soft during optimization; before evaluation,
  hard-project `delta` to `norm_cap` if needed and log `hard_norm_clipped=true`.
- Tie-breaking: select the checkpoint with lowest support objective; ties break
  by lower delta norm, then earliest step.
- Nonfinite loss, gradient, alpha, or delta marks the candidate
  contract-invalid for that record and emits a redacted failure event.
- Unknown basis names fail closed with `ValueError`.

## Result

Bounded inner validation completed on 2026-06-14.

Result note:
`docs/superpowers/plans/2026-06-14-v27-localized-behavior-loss-subspace-editor-results.md`.

Verdict: accepted as a bounded negative diagnostic only. All `12` candidate
evaluations had `target_prediction_rate=0.0`; the best candidate showed
positive matched-vs-control margin movement but no functional target flips.

## Long-Run Monitoring

Use the existing long-run monitor for every run longer than a quick unit test:

- `--monitor-interval-seconds 5`
- progress log:
  `runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl`
- monitor log:
  `runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl`

During compute, verify every 30-60 seconds:

```bash
ps -p <pid> -o pid=,etime=,pcpu=,pmem=,command=
tail -n 8 runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
tail -n 40 runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl
```

If progress stops advancing and CPU is low for repeated checks, stop and debug
instead of letting the run consume compute blindly.

## Acceptance Gates

Development acceptance remains strict:

- `target_prediction_rate >= 0.85`
- `pareto_undominated_rate >= 0.85`
- `mean_target_margin >= 0.25`
- `individual_all_gate_pass_rate >= 0.85`
- positive margins over all proof-critical controls.

A bounded run with any target prediction rate below `0.65` is diagnostic only,
even if relative margins improve.

## Tasks

- [ ] Add config constants and grid entries for
  `localized_behavior_loss_subspace`.
- [ ] Implement basis construction with redacted hash-only provenance.
- [ ] Implement coefficient optimization with finite checks, norm cap, and
  progress events.
- [ ] Dispatch the new source from `evaluate_v25_development_job()`.
- [ ] Add tests for basis determinism, final-raw guard preservation, no raw log
  leakage, unknown basis failure, and bounded optimizer finite outputs.
- [ ] Add tests for exact V27 grid count/order/hash and first-eight table.
- [ ] Add tests that proof controls and `EXPECTED_CONTROLS_PER_RECORD` are
  unchanged by adding the new matched edit source.
- [ ] Add tests that support objective rows are optimization-only and heldout
  proof metrics remain the reported evaluation surface.
- [ ] Add tests that `teacher_oracle_delta` remains diagnostic-only and is not
  elevated into proof-critical acceptance.
- [ ] Run focused pytest for the helper suite.
- [ ] Run `py_compile` for the edited script and tests.
- [ ] Send implementation and launch plan to reviewer; require `5/5` before
  compute.
- [ ] Run the bounded first-8-config inner validation with monitor logs.
- [ ] Send results to reviewer; require `5/5` before recording or building on
  the result.

## Reviewer Checklist

The reviewer should specifically assess:

- whether the literature support really motivates the experiment;
- whether V27 is distinguishable from the failed V23/V25/V26 variants;
- whether any objective term can leak heldout/final labels or raw records;
- whether logs expose raw weights/signatures/IDs;
- whether the first compute budget is bounded and observable;
- whether the stated acceptance gates prevent overclaiming weak margin signals.
