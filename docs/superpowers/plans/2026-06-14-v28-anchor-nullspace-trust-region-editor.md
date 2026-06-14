# V28 Anchor Nullspace Trust-Region Editor Plan

Date: 2026-06-14

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether gradient-selected edit anchors plus source-preserving
nullspace projection allow larger, still-localized behavior edits than V27
without leaking heldout proof rows or final data.

**Architecture:** Extend the existing V25/V27 runner with a new matched edit
source that selects anchor coordinates from support gradients, projects candidate
updates away from source-compatible support Jacobians, and sweeps a small
trust-region grid. Keep V25-native proof controls unchanged.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## Motivation

V27 produced directional target-margin movement but zero functional target
flips. The best and runner-up configs both used
`target_source_logit_gradient_rank4`, suggesting that gradient-localized
directions are useful, but the soft source MSE penalty and fixed `0.25` norm cap
may be preventing boundary crossing.

V28 should not simply scale V27. It should make source preservation geometric:
project edit directions into the approximate nullspace of source-compatible
support logits, then allow a bounded trust-region sweep over larger caps.

## Literature Basis

- Fang et al., "AlphaEdit: Null-Space Constrained Knowledge Editing for
  Language Models" (https://arxiv.org/abs/2410.02355): projecting perturbations
  into the null space of preserved knowledge can reduce disruption, supporting a
  geometric preservation constraint instead of only a soft source penalty.
- Yang et al., "Fine-tuning Done Right in Model Editing"
  (https://arxiv.org/abs/2509.22072): model editing improves when localized
  tuning is restored to a breadth-first, mini-batch style rather than
  over-optimizing samples depth-first. V28 should optimize support batches, not
  tune on heldout proof records.
- "Constraining Sequential Model Editing with Editing Anchor Compression"
  (https://arxiv.org/html/2503.00035v2): weighted-gradient saliency can identify
  important edit anchors and reduce update norm while preserving editing
  performance. V28 adapts this to small MLP weight coordinates.
- Gupta et al., "A Unified Framework for Model Editing"
  (https://arxiv.org/html/2403.14236v2): ROME and MEMIT can be understood via a
  preservation-memorization objective. V28 explicitly separates memorization
  rows from preservation rows.
- Mitchell et al., "Fast Model Editing at Scale"
  (https://arxiv.org/abs/2110.11309): MEND learns to transform gradients through
  low-rank structure, supporting the use of gradient-derived edit bases even
  when not yet training a full editor network.
- Gargiulo et al., "Task Singular Vectors: Reducing Task Interference in Model
  Merging" (https://openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf):
  task weight differences are often low-rank and singular-vector interactions
  can diagnose interference. V28 uses this to compare anchor-nullspace bases
  against V27 gradient bases.
- "Steer2Edit: From Activation Steering to Component-Level Editing"
  (https://arxiv.org/abs/2602.09870): translating steering signals into
  component-level parameter updates can bridge activation and weight editing.
  V28 keeps this bridge local by using fixed support activation predicates and
  weight-coordinate anchors.

## Hypothesis

If V27 failed because source preservation was competing with target movement,
then projecting target-support updates into an approximate compatible-source
nullspace should allow a larger trust-region cap while preserving source
behavior better than naive larger-norm V27 optimization.

Success requires actual target prediction flips on heldout development proof
records. Margin improvements alone remain diagnostic only.

## Non-Claims

- V28 will not claim final performance.
- V28 will not claim large-model transfer.
- V28 will not claim universal MUAT evidence.
- V28 will not claim decoded functional models unless target prediction and
  proof-control gates pass on development, then on a later sealed final run.

## Matched Edit Source

Add one new matched edit source:

```python
"anchor_nullspace_trust_region"
```

For each source-target proof record, construct support tensors using the same
support-only boundary as V27:

- `target_inputs` and `target_labels` from `suite["support"][target]`;
- `conflict_inputs` and `conflict_target_labels` from source-support rows where
  source and target predicates disagree;
- `compatible_inputs` and `compatible_source_logits` from source-support rows
  where source and target predicates agree.

Heldout development proof rows are evaluation-only. Final raw records remain
sealed.

## Anchor Selection

Compute flattened gradients at the source weights:

- `g_target`: gradient of mean target BCE on `target_inputs`;
- `g_conflict`: gradient of mean conflict BCE on `conflict_inputs`;
- `g_compatible`: gradient of compatible-source MSE on `compatible_inputs`.

For anchor coordinate selection, define:

```text
anchor_score =
  abs(g_target + conflict_weight * g_conflict)
  * sqrt(abs(source_weights) + 1e-6)
  / (abs(g_compatible) + compatible_floor)
```

Use `conflict_weight=0.5`. The first bounded grid fixes
`compatible_floor=0.05`; a later reviewer-approved expansion may compare
`compatible_floor in [0.05, 0.1]`.
Select top `anchor_count` coordinates by descending `anchor_score`; ties break
by lower flattened coordinate index. Candidate anchor counts are `[8, 16, 32]`.

This adapts editing-anchor saliency to weight-space behavior editing: choose
coordinates that are target/conflict-sensitive, already substantively present in
the source model, and relatively insensitive on compatible-source support rows.

## Nullspace Projection

Build a compatible-source Jacobian matrix `J_compatible` from the scalar
compatible logits with respect to flattened source weights. Center rows by their
mean, then compute `torch.linalg.svd(J_compatible, full_matrices=False)`.

Let `s_max = max(singular_values)`. A normalized singular value is
`s_i / max(s_max, 1e-12)`. Let `V_preserve` be right singular vectors whose
normalized singular value is strictly above `nullspace_rtol`. Use grid
`nullspace_rtol in [1e-3, 1e-2]`.

If no singular vectors exceed `nullspace_rtol`, use the identity projection with
`preserve_rank=0`; do not fail closed. This means the support-compatible
Jacobian has no numerically active preservation direction under that threshold.
Still log `preserve_rank=0` and the compatible energy ratio so the case is
auditable.

The projection is:

```text
P_null = I - V_preserve @ V_preserve.T
```

The raw anchor basis `E_anchor` is the standard basis over selected coordinates.
The editable basis is:

```text
B = orthonormalize(sign_canonicalize(P_null @ E_anchor))
```

Fail closed if:

- SVD fails;
- any value is nonfinite;
- rank becomes `0`;
- basis shape is not `(SOURCE_WEIGHT_DIM, rank)`;
- projected compatible energy
  `||J_compatible @ B||_F / max(||J_compatible||_F, 1e-12)` is not finite.

Log only hashes, counts, ranks, scalar energy ratios, and config IDs. Do not log
raw gradients, anchors, basis values, logits, subject IDs, support examples, or
deltas.

## Objective

Optimize only coefficients `alpha`:

```text
delta = B @ alpha
edited_weights = source_weights + delta
```

Objective:

```text
loss =
  target_bce_weight * BCE(target_logits, target_labels)
  + conflict_bce_weight * BCE(conflict_logits, conflict_target_labels)
  + compatible_probe_weight * MSE(compatible_logits, source_compatible_logits)
  + delta_l2_weight * ||delta||^2
```

Use fixed weights for the first grid:

- `target_bce_weight=1.0`;
- `conflict_bce_weight=0.5`;
- `compatible_probe_weight=0.1`;
- `delta_l2_weight=0.0`.

Hard-project `delta` to `trust_norm_cap` after every optimizer step and before
evaluation. This differs from V27's soft barrier: the cap is an explicit
trust-region boundary.

## Initial Grid

Run only the first `8` configs initially, using successive halving on `12,24`
balanced development jobs. This first grid must test the core V28 hypothesis by
including both the V27 cap (`0.25`) and a larger trust-region cap (`0.5`).

Loop order:

```python
for anchor_count in [8, 16]:
    for nullspace_rtol in [1e-3, 1e-2]:
        for compatible_floor in [0.05]:
            for trust_norm_cap in [0.25, 0.5]:
                ...
```

The first eight configs are:

| Config index | Trust cap | Anchor count | Nullspace rtol | Compatible floor |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.25 | 8 | 1e-3 | 0.05 |
| 1 | 0.5 | 8 | 1e-3 | 0.05 |
| 2 | 0.25 | 8 | 1e-2 | 0.05 |
| 3 | 0.5 | 8 | 1e-2 | 0.05 |
| 4 | 0.25 | 16 | 1e-3 | 0.05 |
| 5 | 0.5 | 16 | 1e-3 | 0.05 |
| 6 | 0.25 | 16 | 1e-2 | 0.05 |
| 7 | 0.5 | 16 | 1e-2 | 0.05 |

If all first-eight configs repeat V27's zero target-prediction rate, do not run
larger caps or compatible-floor expansions unchanged. This means the
anchor/nullspace mechanism did not create enough target motion even when
allowed `0.5` norm.

## Optimizer Protocol

- Optimize only `alpha`, never the full weight vector.
- Initialize `alpha=zeros(rank)`.
- Optimizer: `torch.optim.AdamW([alpha], lr=0.05, betas=(0.9, 0.999),
  eps=1e-8, weight_decay=0.0, amsgrad=False)`.
- Steps: `50`.
- Device/dtype: CPU float32.
- Gradient clipping: `clip_grad_norm_([alpha], 5.0)`.
- After each step, hard-project `delta` to `trust_norm_cap` and map projected
  `delta` back to `alpha` via least squares in basis coordinates.
- Step logging: every `10` steps plus final step, scalar-only.
- Emit `anchor_nullspace_basis_start` and `anchor_nullspace_basis_completed`
  events around basis construction. These events may include only
  `selected_config_hash`, `record_id_hash`, support counts, anchor count,
  selected-coordinate hash, Jacobian row count, `preserve_rank`,
  `nullspace_rtol`, compatible energy ratio, finite status, elapsed seconds,
  and failure reason enum. They must not include raw coordinates, gradients,
  logits, weights, examples, anchors, basis values, or deltas.
- Tie-breaking: select the checkpoint with lowest support objective; ties break
  by lower delta norm, then earliest step.
- Any nonfinite loss, gradient, alpha, or delta marks the candidate
  contract-invalid for that record and emits a redacted failure event.

## Acceptance Gates

The first-eight bounded run is diagnostic unless strict preregistered gates are
met. A V28 config is a development success only if all are true:

- `target_prediction_rate >= 0.85`;
- `mean_target_margin > 0.0`;
- `mean_matched_minus_best_control_target_margin > 0.0`;
- `mean_matched_minus_shuffled_signature_target_margin > 0.0`;
- `pareto_undominated_rate >= 0.85`;
- `proof_gate_failure_count=0`;
- `contract_failure_count=0`;
- no forbidden final raw file access;
- reviewer confidence is `5/5`.

If `target_prediction_rate=0.0`, the result is negative even if relative margins
improve. If `0.0 < target_prediction_rate < 0.85`, the result is only a weak
diagnostic signal and cannot be framed as success or as support for final-run
escalation.

## Proof-Control Boundary

V28 matched-edit configs must never be passed into V25-native train banks or
proof controls. The implementation must use an explicit native-control bridge:

```python
V28_ANCHOR_NULLSPACE_NATIVE_CONTROL_CONFIG = V27_LOCALIZED_NATIVE_CONTROL_CONFIG
```

or an equivalent `v25_native_control_config(config)` helper that strips V28
fields and returns the fixed V25-native baseline config. Tests must assert that:

- matched-edit dispatch receives the V28 config;
- train-bank and proof-control construction receive the fixed V25-native config;
- `EXPECTED_CONTROLS_PER_RECORD` is unchanged;
- proof-control labels and hashes do not include V28-only fields.

## Implementation Checklist

- [x] Add V28 constants, variant label, config-grid builder, and exact grid hash
  test.
- [x] Add support-gradient helpers for `g_target`, `g_conflict`,
  `g_compatible`, and compatible Jacobian rows.
- [x] Add deterministic anchor selection with tie-breaking and hash-only
  provenance.
- [x] Add nullspace projection basis construction with energy-ratio diagnostics.
- [x] Add scalar-only optimizer progress logging and redaction tests.
- [x] Add `anchor_nullspace_basis_start/completed` progress events and redaction
  tests.
- [x] Add matched-edit dispatch while leaving V25-native proof controls
  unchanged.
- [x] Add tests that V28 configs are stripped to fixed V25-native configs for
  train banks and proof controls.
- [x] Add tests that V28 rejects missing train provenance and unknown grid names.
- [x] Run focused pytest for V28 tests.
- [x] Run full helper pytest suite.
- [x] Run `py_compile` for the edited script and tests.
- [x] Send implementation and bounded launch plan to reviewer; require `5/5`
  before compute.
- [x] Run only the first-8-config bounded inner validation with monitor logs.
- [x] Send results to reviewer; require `5/5` before recording or expanding.

## Reviewer Checklist

The reviewer should specifically assess:

- whether the literature supports replacing soft source MSE with nullspace
  projection;
- whether anchor selection can leak heldout or final labels;
- whether compatible-source Jacobian construction uses support-only rows;
- whether any log event exposes raw weights, gradients, anchors, signatures,
  logits, subject IDs, or examples;
- whether V25-native proof controls remain unchanged;
- whether the first compute budget is bounded and observable;
- whether the cap escalation is blocked unless first-eight diagnostics justify
  it.

## Result

Recorded separately in
`docs/superpowers/plans/2026-06-14-v28-anchor-nullspace-trust-region-editor-results.md`.

Summary: V28 is a bounded negative diagnostic. The bounded run completed with
`passed=false`; all evaluated candidates had `target_prediction_rate=0.0`.
Positive matched-control and shuffled-signature margin lifts were observed, but
they remained below the target decision boundary and were accompanied by proof
gate failures. The selected best config was selected by the preregistered
ranking tuple, mainly Pareto/aggregate ranking behavior, not because it
approached the target prediction gate. No final-run authorization follows.
