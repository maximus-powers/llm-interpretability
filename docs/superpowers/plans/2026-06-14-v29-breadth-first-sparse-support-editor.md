# V29 Breadth-First Sparse Support Editor Implementation Plan

Date: 2026-06-14

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether V28's zero-flip result was caused by an overly
restricted single-record/nullspace update by using a bounded breadth-first
support optimizer over sparse low-interference weight coordinates.

**Architecture:** Extend the existing V25/V27/V28 development runner with one
new matched edit source. Keep V25-native proof controls, balanced development
job selection, sealed-final restrictions, and monitor logs unchanged.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## Motivation

V26, V27, and V28 all moved margins in useful directions but produced
`target_prediction_rate=0.0` on heldout development proof rows. V28 also showed
that a geometric preservation basis can be too restrictive: all final-rung V28
configs remained below the target boundary, even with a `0.5` trust-region cap.

V29 should test a stronger but still auditable hypothesis: sparse localized
fine-tuning over support batches may cross the decision boundary where analytic
or basis-restricted one-record updates could not. This remains a development
diagnostic only; it must not touch sealed final subjects.

## Literature Basis

- Yang et al., "Fine-tuning Done Right in Model Editing"
  (https://arxiv.org/abs/2509.22072): argues that fine-tuning failures in model
  editing often come from sample-wise depth-first editing, and that localized
  breadth-first mini-batch optimization substantially improves editing.
- Iurada et al., "Efficient Model Editing with Task-Localized Sparse
  Fine-tuning" (https://arxiv.org/abs/2504.02620): motivates sparse task-vector
  construction on low-interference coordinates because linearization alone does
  not guarantee disentanglement.
- Ortiz-Jimenez et al., "Task Arithmetic in the Tangent Space"
  (https://proceedings.neurips.cc/paper_files/paper/2023/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html):
  supports the premise that useful weight-space edits depend on disentangled
  localized weight directions.
- Gangadhar and Stratos, "Model Editing by Pure Fine-Tuning"
  (https://aclanthology.org/2024.findings-acl.352/): shows that standard
  fine-tuning can be a serious editing baseline when optimized conditionally and
  regularized with unrelated facts for locality.
- Tan et al., "Massive Editing for Large Language Models via Meta Learning"
  (https://arxiv.org/abs/2311.04661): warns against naive aggregation of
  per-edit shifts and motivates batch/least-squares-style treatment of multiple
  edits.
- Mitchell et al., "Fast Model Editing at Scale" (MEND)
  (https://arxiv.org/abs/2110.11309): supports gradient-derived low-rank edit
  transformations as a scalable path beyond hand-written local optimizers.
- Ibarra et al., "Universal Weight Subspace Hypothesis"
  (https://arxiv.org/abs/2512.05117): supports the broader MUAT hypothesis that
  trained models may share low-dimensional weight subspaces, while V29 remains
  a small-scale diagnostic rather than a proof of that claim.

## Hypothesis

If V28 failed because nullspace projection removed too much target-effective
capacity, then a sparse breadth-first support optimizer should improve target
prediction rate and mean target margin over V28 while preserving V25-native
proof controls.

Success requires heldout development proof target flips, not just support loss
improvement or margin drift. The selected config can only be called a candidate
for sealed final evaluation if it passes the preregistered development gates.

## Non-Claims

- V29 will not claim final performance.
- V29 will not claim large-model transfer.
- V29 will not claim universal MUAT evidence.
- V29 will not claim decoded functional models unless development and later
  sealed-final proof gates pass.

## Matched Edit Source

Add one matched edit source:

```python
"breadth_first_sparse_support"
```

For each source-target development proof record, construct support tensors using
the same boundary as V27/V28:

- target support rows from `suite["support"][target_behavior]`;
- conflict support rows where source and target predicates disagree;
- compatible support rows where source and target predicates agree;
- unrelated/native controls from the existing V25-native proof path.

Heldout development proof rows remain evaluation-only. The sealed final raw
file remains unread.

## Sparse Coordinate Selection

Compute flattened source-weight gradients on support rows only:

- `g_target`: mean target BCE gradient;
- `g_conflict`: mean conflict BCE gradient;
- `g_compatible`: compatible-source MSE gradient;
- `g_unrelated`: exactly zero for V29. Do not add unrelated/locality gradient
  rows unless a later reviewed plan names a support-only tensor source. V29 must
  not read heldout proof rows, development proof rows, train/development subject
  records beyond the existing support tensors, or sealed final data to construct
  this term.

Score coordinates deterministically:

```text
score =
  abs(g_target + conflict_weight * g_conflict)
  / (compatible_floor + abs(g_compatible))
```

Use lower flattened coordinate index as the final tie-break. Log only
coordinate count, coordinate hash, scalar sensitivity summaries, and config ID.
Never log raw coordinate lists, gradients, logits, weights, examples, subject
IDs, or deltas.

Initial grid loop order:

```python
for sparse_top_k in [16, 32]:
    for trust_norm_cap in [0.5, 1.0]:
        for compatible_floor in [0.05]:
            for extra_compatible_weight in [0.05, 0.2]:
                ...
```

This gives eight configs. The `1.0` cap is deliberate but bounded: V28's `0.5`
cap still produced zero flips, so V29 needs one controlled escalation while
retaining proof controls and hard projection.

Expected config table:

```text
0: sparse_top_k=16, trust_norm_cap=0.5, compatible_floor=0.05, extra_compatible_weight=0.05
1: sparse_top_k=16, trust_norm_cap=0.5, compatible_floor=0.05, extra_compatible_weight=0.2
2: sparse_top_k=16, trust_norm_cap=1.0, compatible_floor=0.05, extra_compatible_weight=0.05
3: sparse_top_k=16, trust_norm_cap=1.0, compatible_floor=0.05, extra_compatible_weight=0.2
4: sparse_top_k=32, trust_norm_cap=0.5, compatible_floor=0.05, extra_compatible_weight=0.05
5: sparse_top_k=32, trust_norm_cap=0.5, compatible_floor=0.05, extra_compatible_weight=0.2
6: sparse_top_k=32, trust_norm_cap=1.0, compatible_floor=0.05, extra_compatible_weight=0.05
7: sparse_top_k=32, trust_norm_cap=1.0, compatible_floor=0.05, extra_compatible_weight=0.2
```

Expected grid hash from `stable_hash_json(grid)`:

```text
ef40cccc68f4cf08e9e8373de9a8df7555170273f6aaa4ff4d524820859aa9d0
```

## Breadth-First Optimizer

Optimize a dense `delta_values` vector over the selected sparse coordinates.
The full edit is zero everywhere else.

Use deterministic epoch/batch order over support mini-batches. This differs
from the prior one-record optimizer: every epoch sees target, conflict, and
compatible rows before the next epoch, matching the breadth-first editing
finding in LocFT-BF.

Objective:

```text
loss =
  target_bce_weight * BCE(target_logits, target_labels)
  + conflict_bce_weight * BCE(conflict_logits, conflict_target_labels)
  + compatible_probe_weight * MSE(compatible_logits, source_compatible_logits)
  + extra_compatible_weight * MSE(compatible_logits, source_compatible_logits)
  + delta_l2_weight * ||delta||^2
```

Initial fixed weights:

- `target_bce_weight=1.0`;
- `conflict_bce_weight=0.5`;
- `compatible_probe_weight=0.2`;
- `extra_compatible_weight` is the grid value and controls additional
  compatible-source preservation pressure;
- `delta_l2_weight=1e-4`.

Hard-project `delta` to `trust_norm_cap` after every optimizer step and before
evaluation. Clamp nonfinite updates closed and record a contract failure.

## Logging And Monitoring

Add redacted progress events:

- `v29_sparse_coordinate_selection_completed`;
- `v29_breadth_first_optimizer_progress`;
- `v29_breadth_first_optimizer_completed`.

Required fields are hashes, counts, scalar losses, scalar norms, finite flags,
step/epoch/batch counters, config IDs, and variant names. Forbidden fields are
raw weights, raw deltas, raw logits, raw gradients, raw selected coordinates,
support examples, subject IDs, or any final raw content/path.

Any long-running V29 command must use:

```bash
--monitor-interval-seconds 5 --summary-only-stdout
```

During the run, verify:

- process liveness and CPU time are changing;
- development progress log line count is increasing;
- monitor log line count is increasing;
- monitor events include progress, not only startup;
- no raw final data path appears in progress logs.

## Implementation Tasks

- [x] Add V29 constants, matched edit source name, and eight-config grid builder.
- [x] Add a V29 grid-order/hash pytest using
      `ef40cccc68f4cf08e9e8373de9a8df7555170273f6aaa4ff4d524820859aa9d0`.
- [x] Update inner-validation config selection and variant derivation for V29.
- [x] Keep `v25_native_control_config()` mapped to the existing V25-native
      control baseline for V29.
- [x] Add redaction helper for V29 progress events.
- [x] Implement support-only sparse coordinate scoring with deterministic
      tie-breaking and coordinate hashing.
- [x] Implement breadth-first sparse support optimizer with hard trust-region
      projection and nonfinite fail-closed behavior.
- [x] Add V29 matched-edit evaluator and wire it into
      `evaluate_v25_development_job()`.
- [x] Add CLI choice `v29-breadth-first-sparse`.
- [x] Add pytest coverage for grid stability, provenance binding, native
      controls, redaction, sparse coordinate selection, trust cap enforcement,
      deterministic batch order, and dispatch behavior.
- [x] Run focused V29 pytest, full helper pytest, and py_compile. Do not run
      lint unless explicitly requested.
- [x] Run bounded V29 development inner validation with monitor logs.
- [x] Spawn reviewer after implementation, after verification, and after
      results; continue only when reviewer confidence is 5/5.

## Validation Commands

Focused tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v29'
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
  --inner-validation-config-grid v29-breadth-first-sparse \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Acceptance Gates

V29 can only be considered a successful development candidate if:

- `target_prediction_rate >= 0.85`;
- `mean_target_margin > 0.0`;
- `mean_matched_minus_best_control_target_margin > 0.0`;
- `mean_matched_minus_shuffled_signature_target_margin > 0.0`;
- Pareto domination rate is at least `0.85`;
- proof failure count is zero;
- contract failure count is zero;
- progress and monitor logs contain only redacted, auditable fields;
- the reviewer returns confidence `5/5`.

If target prediction rate remains `0.0`, V29 is a negative diagnostic even if
margins improve. If target prediction rate is between `0.0` and `0.85`, V29 is
a weak positive diagnostic that requires a new plan and review before any
sealed-final evaluation.

## Risk Controls

- Data leak risk: coordinate selection and optimization use support rows only;
  heldout development proof rows are evaluation-only; final raw remains sealed.
- Misleading metric risk: success is not support loss and not average margin
  alone; target flips and proof controls are required.
- Compute waste risk: first run is capped at eight configs and 12/24
  successive-halving jobs with five-second monitor logs.
- Overfitting risk: V29 must report selected-config ranking criteria and all
  final-rung candidates, not only the best row.
- Logging risk: redaction tests must fail if raw tensors, coordinates, subject
  IDs, final paths, or examples are emitted.

## Reviewer Checklist

- [ ] Literature support is relevant and not overstated.
- [ ] The plan tests a distinct hypothesis from V28 rather than just tuning
      around a negative result.
- [ ] Data boundaries prevent development proof or sealed-final leakage.
- [ ] Trust-region escalation is bounded and justified by V28's zero-flip
      result.
- [ ] Logs are sufficient to monitor long runs without exposing raw data.
- [ ] Acceptance gates would classify misleading margin-only results as
      negative or weak diagnostic, not success.
