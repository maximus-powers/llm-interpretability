# V31 Orthogonal Sign-Sparse Support Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether sign-conflict-aware sparse coordinate selection plus a
compatible-gradient orthogonality penalty can convert V30's partial target flips
into a more reliable development candidate without increasing leakage risk.

**Architecture:** Add one new matched edit source to the existing V25/V29/V30
runner. Reuse V30 support-only optimization, V25-native controls, balanced
development jobs, redacted JSONL logs, and sealed-final boundary. V31 changes
only coordinate scoring and the support objective.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## Current Evidence

V30 was accepted by reviewer confidence `5/5` as a moderate positive diagnostic,
not a success. Its best final-rung candidate reached:

```text
target_prediction_rate=0.6666666666666666
mean_target_margin=0.32473520809435286
mean_matched_minus_best_control_target_margin=0.4790057574898583
mean_matched_minus_shuffled_signature_target_margin=0.4879827120810205
proof_gate_failure_count=10
contract_failure_count=0
```

The best V30 config saturated the trust cap on almost every direction. Redacted
direction aggregates for the best config showed all `*->has_majority` directions
failed target prediction, plus `mountain_pattern->sorted_descending`:

```text
mountain_pattern->has_majority        target_rate=0.000 mean_margin=-0.067236
sorted_ascending->has_majority        target_rate=0.000 mean_margin= 0.002312
sorted_descending->has_majority       target_rate=0.000 mean_margin= 0.088791
mountain_pattern->sorted_descending   target_rate=0.000 mean_margin= 0.092045
```

This argues against a blind capacity increase. V31 should test whether the
selected sparse coordinates are still carrying target/conflict sign interference
or compatibility drift.

## Literature Basis

- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models"
  (https://papers.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf):
  TIES trims low-magnitude update entries, elects signs to resolve conflicts,
  and merges only sign-aligned values. V31 adapts the sign-conflict idea to
  support-gradient coordinate selection rather than multi-model merging.
- Iurada et al., "Efficient Model Editing with Task-Localized Sparse
  Fine-tuning" (https://arxiv.org/abs/2504.02620): sparse task-localized updates
  can improve weight disentanglement and reduce interference. V31 keeps sparse
  support updates but makes the sparse mask more interference-aware.
- Porrello et al., "Dataless Weight Disentanglement in Task Arithmetic via
  Kronecker-Factored Approximate Curvature" (https://arxiv.org/abs/2602.17385):
  frames representation-drift regularization as a curvature approximation
  problem. V31 uses a cheaper first-order proxy: keep the edit delta close to
  orthogonal to the compatible-support gradient.
- Sommariva et al., "Distilling Linearized Behavior into Non-Linear Fine-Tuning
  for Effective Task Arithmetic" (https://arxiv.org/abs/2605.18993): supports
  preserving linearized/tangent behavior through activation constraints. V31
  stays in the small-scale setting by regularizing the first-order compatible
  gradient instead of adding a teacher model.
- "Understanding and Enforcing Weight Disentanglement in Task Arithmetic"
  (https://arxiv.org/html/2604.17078v1): identifies task-feature specialization
  and weight-vector orthogonality as signals tied to disentanglement, and
  proposes orthogonality pressure on updates. V31 introduces a bounded
  compatible-gradient orthogonality penalty.
- Gu and Yeung-Levy, "Foundation Models Secretly Understand Neural Network
  Weights" (https://arxiv.org/abs/2503.00838), and Zhou, "Universal
  Hypernetworks for Arbitrary Models" (https://arxiv.org/abs/2604.02215):
  support the broader hypothesis that weight-space descriptors can drive
  parameter generation/editing. V31 remains a non-hypernetwork diagnostic, but
  it produces cleaner evidence about whether fixed probe/signature information
  can localize functional edits before moving to learned editors.

## Hypothesis

If V30's remaining failures come from sign-conflicted sparse coordinates, then
penalizing target/conflict sign disagreement in coordinate selection should
raise target prediction rate without increasing proof failures.

If V30's remaining failures come from compatible-behavior drift, then adding a
small squared orthogonality penalty between the edit delta and the compatible
support gradient should improve proof gates while preserving the V30 target
signal.

If V30 simply lacks capacity, then a bounded `trust_norm_cap=1.5` arm may help,
but V31 must only interpret that as useful if controls and proof failures also
improve.

## Non-Claims

- V31 will not read or evaluate sealed-final raw data.
- V31 will not claim success unless development gates pass.
- V31 will not claim hypernetwork or universal-weight-subspace validation.
- V31 will not treat support objective improvement as proof performance.
- V31 will not run linting unless explicitly requested by the user.

## Matched Edit Source

Add one matched edit source:

```python
"orthogonal_sign_sparse_support"
```

V31 reuses the V30 support tensors and heldout proof split. All training,
coordinate selection, hardness weighting, and orthogonality terms are support
only.

## Coordinate Selection

Add:

```python
def select_v31_sign_coherent_sparse_coordinates(
    *,
    g_target: torch.Tensor,
    g_conflict: torch.Tensor,
    g_compatible: torch.Tensor,
    sparse_top_k: int,
    compatible_floor: float,
    conflict_weight: float,
    sign_conflict_penalty: float,
) -> list[int]:
```

All gradients are loss gradients, so target and conflict descent directions are
sign-compatible when `g_target * g_conflict > 0`. Score coordinates as:

```text
base = abs(g_target + conflict_weight * g_conflict)
compatible_denominator = abs(g_compatible) + compatible_floor
sign_conflict = 1 if g_target * g_conflict < 0 else 0
score = base / compatible_denominator / (1 + sign_conflict_penalty * sign_conflict)
```

Sort by descending score, then ascending index for deterministic ties. Reject
nonfinite inputs, nonpositive `sparse_top_k`, nonpositive `compatible_floor`, and
negative/nonfinite `sign_conflict_penalty`.

## Objective

Start from V30 and add two terms.

Support-hardness multiplier:

```text
initial_signed_target_margin = (2 * target_label - 1) * source_target_logit
hardness = mean(relu(target_margin_floor - initial_signed_target_margin))
target_multiplier = 1 + hard_target_margin_weight * stop_gradient(hardness)
```

Compatible-gradient orthogonality:

```text
unit_compatible_gradient = g_compatible / (norm(g_compatible) + 1e-8)
compatible_orthogonal_loss = dot(delta, unit_compatible_gradient)^2
```

Loss:

```text
loss =
  target_multiplier * (
      target_bce_weight * BCE(target_logits, target_labels)
      + target_margin_weight * target_margin_hinge
  )
  + conflict_bce_weight * BCE(conflict_logits, conflict_target_labels)
  + compatible_probe_weight * MSE(compatible_logits, source_compatible_logits)
  + extra_compatible_weight * MSE(compatible_logits, source_compatible_logits)
  + compatible_orthogonal_weight * compatible_orthogonal_loss
  + delta_l2_weight * ||delta||^2
```

Keep V30 optimizer settings and hard trust-region projection. Nonfinite values
fail closed.

## Initial Grid

Keep the first run bounded at eight configs:

```python
for trust_norm_cap in [1.25, 1.5]:
    for sign_conflict_penalty in [0.5, 1.0]:
        for compatible_orthogonal_weight in [0.05, 0.15]:
            ...
```

All configs include:

```text
matched_edit_source=orthogonal_sign_sparse_support
sparse_top_k=64
target_margin_floor=0.25
compatible_floor=0.05
extra_compatible_weight=0.05
hard_target_margin_weight=1.0
```

Config table:

```text
0: trust_norm_cap=1.25, sign_conflict_penalty=0.5, compatible_orthogonal_weight=0.05
1: trust_norm_cap=1.25, sign_conflict_penalty=0.5, compatible_orthogonal_weight=0.15
2: trust_norm_cap=1.25, sign_conflict_penalty=1.0, compatible_orthogonal_weight=0.05
3: trust_norm_cap=1.25, sign_conflict_penalty=1.0, compatible_orthogonal_weight=0.15
4: trust_norm_cap=1.50, sign_conflict_penalty=0.5, compatible_orthogonal_weight=0.05
5: trust_norm_cap=1.50, sign_conflict_penalty=0.5, compatible_orthogonal_weight=0.15
6: trust_norm_cap=1.50, sign_conflict_penalty=1.0, compatible_orthogonal_weight=0.05
7: trust_norm_cap=1.50, sign_conflict_penalty=1.0, compatible_orthogonal_weight=0.15
```

Expected raw grid hash from `stable_hash_json(grid)`:

```text
bf2336c5997b5f1258f407d13a687fd8923424d849632bf7e5673e0728f23337
```

Runtime bound-grid hashes may differ because train-pool provenance is added to
each config before evaluation.

## Logging And Monitoring

Add redacted progress events:

- `v31_sign_coherent_coordinate_selection_completed`;
- `v31_orthogonal_sign_optimizer_progress`;
- `v31_orthogonal_sign_optimizer_completed`.

Allowed fields are hashes, counts, finite flags, scalar losses, scalar norms,
epoch/step counters, config IDs, variant names, `sign_conflict_penalty`,
`compatible_orthogonal_weight`, `target_multiplier`, and
`compatible_orthogonal_loss`.

Forbidden fields remain raw weights, raw deltas, raw logits, raw gradients,
selected coordinate lists, support examples, subject IDs, and final raw
paths/content.

Any long-running V31 command must use:

```bash
--monitor-interval-seconds 5 --summary-only-stdout
```

During and after compute, verify:

- process liveness and CPU time;
- progress and monitor row counts;
- candidate-completion events;
- final `monitor_stop`;
- log SHA256 hashes;
- forbidden-field scan over progress/monitor logs.

## Files

- Modify:
  `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Update after implementation:
  `docs/superpowers/plans/2026-06-14-v31-orthogonal-sign-sparse-support-editor.md`
- Create after compute:
  `docs/superpowers/plans/2026-06-14-v31-orthogonal-sign-sparse-support-editor-results.md`

## Implementation Tasks

- [x] Add V31 constants, matched edit source, and eight-config grid builder.
- [x] Add a V31 grid-order/hash pytest using
      `bf2336c5997b5f1258f407d13a687fd8923424d849632bf7e5673e0728f23337`.
- [x] Update config selection, variant mapping, CLI choices, and V25-native
      control mapping.
- [x] Add V31 redaction helper without permitting new raw fields.
- [x] Add sign-coherent sparse coordinate selection and tests for deterministic
      order, sign-conflict penalty, compatible denominator, and invalid inputs.
- [x] Implement support-hardness multiplier and compatible-gradient
      orthogonality helper with tests for finite behavior and expected scalar
      values.
- [x] Implement the V31 optimizer with hard trust cap, V30 target margin hinge,
      target multiplier, compatible orthogonality loss, scalar-only logging, and
      nonfinite fail-closed behavior.
- [x] Add V31 matched-edit evaluator and dispatch branch.
- [x] Add pytest coverage for provenance binding, native controls, redaction,
      optimizer trust-cap enforcement, dispatch, and scalar-only logging.
- [x] Run focused V31 pytest, full helper pytest, and py_compile. Do not run
      lint unless explicitly requested.
- [x] Get reviewer confidence `5/5` before compute.
- [x] Run bounded V31 development inner validation with monitor logs.
- [x] Write results and get reviewer confidence `5/5`.

## Validation Commands

Focused tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v31'
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
  --inner-validation-config-grid v31-orthogonal-sign-sparse \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Review Gates

Plan review must confirm:

- literature basis is real and relevant;
- V31 is not just a larger-capacity retry of V30;
- all optimization inputs are support-only;
- sealed final raw data remains untouched;
- redacted logs are sufficient to detect wasted compute;
- no success claim can be made without development gates.

Results review must confirm:

- command used the bounded monitored protocol;
- progress and monitor logs ended cleanly and hash-match reported output;
- candidate metrics were parsed from logs, not copied from memory;
- forbidden-field scan over progress/monitor logs has no matches;
- interpretation is conservative relative to preregistered gates.
