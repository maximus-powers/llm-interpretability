# V30 Margin-Gated Sparse Support Editor Implementation Plan

Date: 2026-06-14

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert V29's weak-positive target-flip signal into a more reliable
and localized development candidate by adding support-margin gates and stronger
but still bounded sparse capacity.

**Architecture:** Extend the existing V25/V27/V28/V29 runner with one new
matched edit source. Reuse V29's support-only sparse coordinate selection,
redacted logs, V25-native controls, balanced development jobs, and sealed-final
boundary.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## Motivation

V29 showed the first meaningful target flips: the best 24-job rung reached
`target_prediction_rate=0.5`, with positive best-control and shuffled-signature
lifts. It still failed development because `target_prediction_rate < 0.85` and
`proof_gate_failure_count=18`.

The best V29 final-rung config was:

```text
sparse_top_k=32
trust_norm_cap=1.0
extra_compatible_weight=0.05
target_prediction_rate=0.5
mean_target_margin=0.1888460420317036
proof_gate_failure_count=18
contract_failure_count=0
```

Direction analysis of the redacted progress records showed asymmetric
reliability. In the best final-rung config, `has_majority -> *` directions
flipped reliably, while `* -> has_majority` directions failed target prediction.
V30 should improve per-direction reliability and margin, not simply scale V29.

## Literature Basis

- Yang et al., "Fine-tuning Done Right in Model Editing"
  (https://arxiv.org/abs/2509.22072): supports breadth-first mini-batch editing
  over depth-first sample-wise updates. V30 keeps the breadth-first support
  optimizer.
- Iurada et al., "Efficient Model Editing with Task-Localized Sparse
  Fine-tuning" (https://arxiv.org/abs/2504.02620): motivates sparse
  low-interference coordinate updates because task vectors can interfere when
  important parameters for other tasks move.
- Gangadhar and Stratos, "Model Editing by Pure Fine-Tuning"
  (https://aclanthology.org/2024.findings-acl.352/): argues that regularized
  fine-tuning with locality data is a serious editing baseline, so V30 treats
  V29 as an editing baseline that needs stronger support-locality pressure.
- Tan et al., "Massive Editing for Large Language Models via Meta Learning"
  (https://arxiv.org/abs/2311.04661): motivates batch-consistent edits instead
  of naive per-record shifts, supporting V29/V30's support-batch optimizer.
- Yadav et al., "TIES-Merging" (https://arxiv.org/abs/2306.01708): emphasizes
  that task interference can come from sign conflicts and redundant parameter
  updates. V30 expands sparse capacity only modestly and keeps deterministic
  coordinate selection/provenance.
- Ortiz-Jimenez et al., "Task Arithmetic in the Tangent Space"
  (https://proceedings.neurips.cc/paper_files/paper/2023/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html):
  supports the idea that localized weight directions and disentanglement matter
  for reliable arithmetic/editing.
- Mitchell et al., "Fast Model Editing at Scale" (MEND)
  (https://arxiv.org/abs/2110.11309): supports gradient-derived low-rank/sparse
  edit transformations as a stepping stone before a learned editor.

## Hypothesis

If V29 underperformed because BCE loss allowed some support examples to remain
near the decision boundary, then adding a support target-margin hinge should
improve heldout target prediction reliability while maintaining positive proof
control lifts.

If V29 underperformed because the sparse coordinate set was too small for
harder directions such as `* -> has_majority`, then increasing `sparse_top_k`
from `32` to `64` with only a modest trust-region increase should improve
target rate without a large jump in proof failures.

## Non-Claims

- V30 will not claim final performance.
- V30 will not read or evaluate sealed final raw data.
- V30 will not claim success from support loss or margin improvement alone.
- V30 will not claim decoded functional models unless development gates pass
  and a later reviewed sealed-final protocol is run.

## Matched Edit Source

Add one matched edit source:

```python
"margin_gated_sparse_support"
```

V30 reuses V29 support tensors:

- target support rows from `suite["support"][target_behavior]`;
- conflict support rows where source and target predicates disagree;
- compatible support rows where source and target predicates agree.

Heldout development proof rows remain evaluation-only. The sealed final raw file
must not be read.

## Objective

V30 starts from V29's sparse optimizer and adds a target-margin hinge:

```text
signed_target_margin = (2 * target_label - 1) * target_logit
target_margin_hinge = mean(relu(target_margin_floor - signed_target_margin))

loss =
  target_bce_weight * BCE(target_logits, target_labels)
  + target_margin_weight * target_margin_hinge
  + conflict_bce_weight * BCE(conflict_logits, conflict_target_labels)
  + compatible_probe_weight * MSE(compatible_logits, source_compatible_logits)
  + extra_compatible_weight * MSE(compatible_logits, source_compatible_logits)
  + delta_l2_weight * ||delta||^2
```

Initial fixed weights:

- `target_bce_weight=1.0`;
- `target_margin_weight=0.5`;
- `conflict_bce_weight=0.5`;
- `compatible_probe_weight=0.2`;
- `extra_compatible_weight=0.05`;
- `delta_l2_weight=1e-4`.

Use hard trust-region projection after every optimizer step and before
evaluation. Nonfinite values fail closed.

## Initial Grid

V30 keeps the run bounded at eight configs. The cap expansion is deliberately
small: V29 already needed `trust_norm_cap=1.0`, so V30 compares `1.0` against
`1.25`, not an open-ended increase.

Loop order:

```python
for sparse_top_k in [32, 64]:
    for trust_norm_cap in [1.0, 1.25]:
        for target_margin_floor in [0.15, 0.25]:
            ...
```

Config table:

```text
0: sparse_top_k=32, trust_norm_cap=1.0, target_margin_floor=0.15
1: sparse_top_k=32, trust_norm_cap=1.0, target_margin_floor=0.25
2: sparse_top_k=32, trust_norm_cap=1.25, target_margin_floor=0.15
3: sparse_top_k=32, trust_norm_cap=1.25, target_margin_floor=0.25
4: sparse_top_k=64, trust_norm_cap=1.0, target_margin_floor=0.15
5: sparse_top_k=64, trust_norm_cap=1.0, target_margin_floor=0.25
6: sparse_top_k=64, trust_norm_cap=1.25, target_margin_floor=0.15
7: sparse_top_k=64, trust_norm_cap=1.25, target_margin_floor=0.25
```

All configs include:

```text
matched_edit_source=margin_gated_sparse_support
compatible_floor=0.05
extra_compatible_weight=0.05
```

Expected raw grid hash from `stable_hash_json(grid)`:

```text
3225c6db22149aba92f1366f23010a1e87de8cde4175cd5b37ff826071cb59cc
```

## Logging And Monitoring

Add or reuse redacted progress events:

- `v30_sparse_coordinate_selection_completed`;
- `v30_margin_gated_optimizer_progress`;
- `v30_margin_gated_optimizer_completed`.

Allowed fields are hashes, counts, scalar losses, scalar norms, finite flags,
epoch/step counters, config IDs, and variant names. Forbidden fields are raw
weights, raw deltas, raw logits, raw gradients, selected coordinate lists,
support examples, subject IDs, and final raw paths/content.

Any long-running V30 command must use:

```bash
--monitor-interval-seconds 5 --summary-only-stdout
```

Monitor during execution:

- process liveness and CPU time;
- progress log line count;
- monitor log line count;
- candidate-completion events;
- `monitor_stop` at the end.

## Implementation Tasks

- [x] Add V30 constants, matched edit source, and eight-config grid builder.
- [x] Add a V30 grid-order/hash pytest using
      `3225c6db22149aba92f1366f23010a1e87de8cde4175cd5b37ff826071cb59cc`.
- [x] Update config selection, variant mapping, CLI choices, and V25-native
      control mapping.
- [x] Add V30 redaction helper or extend the V29 helper without permitting new
      raw fields.
- [x] Reuse V29 sparse coordinate selection, but bind V30 coordinate hashes and
      provenance under the V30 event names.
- [x] Implement the margin-gated sparse support optimizer with hard trust cap,
      target-margin hinge, scalar-only logging, and nonfinite fail-closed
      behavior.
- [x] Add V30 matched-edit evaluator and dispatch branch.
- [x] Add pytest coverage for grid hash/order, provenance binding, native
      controls, redaction, margin hinge behavior, trust cap enforcement, and
      dispatch.
- [x] Run focused V30 pytest, full helper pytest, and py_compile. Do not run
      lint unless explicitly requested.
- [x] Get reviewer confidence `5/5` before compute.
- [x] Run bounded V30 development inner validation with monitor logs.
- [x] Write results and get reviewer confidence `5/5`.

## Validation Commands

Focused tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v30'
```

Full helper tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Compile check:

```bash
python -m py_compile \
  model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Bounded development run:

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v30-margin-gated-sparse \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Acceptance Gates

V30 can only be called a successful development candidate if:

- `target_prediction_rate >= 0.85`;
- `mean_target_margin > 0.0`;
- `mean_matched_minus_best_control_target_margin > 0.0`;
- `mean_matched_minus_shuffled_signature_target_margin > 0.0`;
- `pareto_undominated_rate >= 0.85`;
- proof failure count is zero;
- contract failure count is zero;
- logs contain only redacted, auditable fields;
- reviewer confidence is `5/5`.

If target prediction improves but remains below `0.85`, V30 is only a weak or
moderate positive diagnostic. It must not trigger sealed-final evaluation.

## Risk Controls

- Data leak risk: V30 uses support tensors only for optimization; heldout
  development rows are evaluation-only; final raw remains sealed.
- Misleading metric risk: target-margin hinge is a support objective, not a
  proof metric. Success still requires heldout proof gates.
- Compute risk: first run remains capped at eight configs and 12/24
  successive-halving jobs.
- Overfitting risk: all final-rung candidates must be reported, including proof
  failures and target rates.
- Logging risk: tests must reject raw tensors, selected coordinates, support
  examples, subject IDs, or final paths in progress events.

## Reviewer Checklist

- [ ] Literature support is relevant and does not overclaim.
- [ ] V30 addresses V29's observed failure pattern rather than blindly scaling
      norm.
- [ ] Data boundaries remain support-only for optimization.
- [ ] The `1.25` cap and `64` coordinates are bounded and justified by V29's
      partial target success.
- [ ] The target-margin hinge cannot be mistaken for proof success.
- [ ] Logs are useful for long-run monitoring and remain redacted.
