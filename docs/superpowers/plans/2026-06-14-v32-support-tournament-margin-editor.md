# V32 Support Tournament Margin Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether optimizing a support-only target-vs-runner-up behavior
margin can close V31's remaining predicted-behavior failures, especially target
`has_majority`, without using heldout or final data for optimization.

**Architecture:** Add one new matched edit source to the existing V25/V31 runner.
Reuse V31 sign-coherent sparse selection, compatible-gradient orthogonality,
hardness multiplier, V25-native controls, balanced development jobs, redacted
JSONL logs, and sealed-final boundary. V32 adds a support-tournament loss that
matches the proof's `predicted_behavior == target` criterion more directly.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## Current Evidence

V31 was accepted by reviewer confidence `5/5` as a stronger positive diagnostic,
not a success. Its best final-rung candidate reached:

```text
target_prediction_rate=0.7916666666666666
mean_target_margin=0.567085697936515
mean_matched_minus_best_control_target_margin=0.7222636344971534
mean_matched_minus_shuffled_signature_target_margin=0.7303332019231826
proof_gate_failure_count=7
contract_failure_count=0
```

The proof's target-prediction gate checks whether the edited model's
`predicted_behavior` equals the target behavior, where `predicted_behavior` is
the behavior with the largest heldout behavior margin. V31 optimized the binary
target predicate margin but did not explicitly require the target behavior
margin to beat every other behavior margin. The remaining V31 target failures
are concentrated in target `has_majority`:

```text
mountain_pattern->has_majority      target_rate=0.000 mean_margin=-0.019759
sorted_ascending->has_majority      target_rate=0.000 mean_margin= 0.207635
sorted_descending->has_majority     target_rate=0.500 mean_margin= 0.083126
```

The positive mean margins with failed predicted behavior indicate that binary
target margin alone is not sufficient. V32 should optimize a support-only
multiclass/tournament analogue of the proof criterion.

## Literature Basis

- BalancEdit, "Dynamically Balancing the Generality-Locality Trade-off in Model
  Editing" (https://arxiv.org/html/2505.01343v2): motivates balancing positive
  and negative edit scope. V32 uses support positives and negatives for every
  behavior, then asks the target behavior to beat all support runners-up.
- Yu et al., "Gradient Surgery for Multi-Task Learning"
  (https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf):
  motivates resolving gradient interference by altering conflicting gradients.
  V32 keeps V31's sign-coherent selector and adds a tournament loss to reduce
  objective mismatch rather than only changing gradients.
- "Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent"
  (https://arxiv.org/html/2501.01230v1): argues that conflict handling should
  still preserve task-specific performance. V32 tests target-vs-runner-up
  behavior performance directly on support margins.
- "Multi-Task Model Merging via Adaptive Weight Disentanglement"
  (https://arxiv.org/html/2411.18729v1): frames task consistency as a response
  to task-vector interference. V32 keeps V31 disentanglement pressure while
  adding consistency with the predicted-behavior decision rule.
- "Are We Evaluating the Edit Locality of LLM Model Editing Properly?"
  (https://arxiv.org/pdf/2601.17343): emphasizes that locality/specificity
  should be measured as behavioral deviation rather than assumed from labels.
  V32 keeps heldout proof evaluation untouched and treats support tournament
  improvement only as an optimization objective, not proof.
- Steer2Edit, "From Activation Steering to Component-Level Editing"
  (https://arxiv.org/html/2602.09870v2): supports localized component-level
  interventions for behavior control. V32 remains weight-editing rather than
  activation steering, but continues the localized sparse-update trajectory.

## Hypothesis

If V31 fails because its binary target predicate objective does not match the
proof's behavior-ranking criterion, then adding a support tournament margin
should improve final-rung target prediction, especially for target
`has_majority`.

If V31's remaining failures are due to insufficient capacity or support/heldout
mismatch, the support tournament objective may improve support behavior margins
without improving heldout proof gates. That would be a useful negative
diagnostic, not success.

## Non-Claims

- V32 will not read or evaluate sealed-final raw data.
- V32 will not optimize on heldout proof rows.
- V32 will not claim success unless development gates pass.
- V32 will not treat support tournament improvement as proof performance.
- V32 will not run linting unless explicitly requested by the user.

## Matched Edit Source

Add one matched edit source:

```python
"support_tournament_margin_sparse"
```

V32 reuses V31 sparse coordinate selection and support tensors. It also builds
support-only behavior-margin tensors for all four behavior predicates from
`evaluation_suite()["support"]`.

## Support Tournament Objective

Add support behavior margins:

```text
support_behavior_margin(pattern) =
  mean(sigmoid(model(pattern_support_positive)))
  - mean(sigmoid(model(pattern_support_negative)))
```

Add target-vs-runner tournament hinge:

```text
target_support_margin = support_behavior_margin(target_behavior)
runner_margin = max(support_behavior_margin(pattern) for pattern != target_behavior)
tournament_margin_hinge =
  relu(tournament_margin_floor - (target_support_margin - runner_margin))
```

The loss starts from V31 and adds:

```text
tournament_margin_weight * tournament_margin_hinge
```

Keep V31's target BCE, target-margin hinge, conflict BCE, compatible MSE,
compatible-gradient orthogonality, delta L2, hard trust cap, and nonfinite
fail-closed behavior.

## Initial Grid

Keep the first run bounded at eight configs. Because V31's `trust_norm_cap=1.5`
arms did not reach the final rung, V32 fixes `trust_norm_cap=1.25` instead of
raising capacity.

```python
for tournament_margin_weight in [0.5, 1.0]:
    for tournament_margin_floor in [0.05, 0.15]:
        for compatible_orthogonal_weight in [0.05, 0.15]:
            ...
```

All configs include:

```text
matched_edit_source=support_tournament_margin_sparse
sparse_top_k=64
trust_norm_cap=1.25
sign_conflict_penalty=1.0
target_margin_floor=0.25
compatible_floor=0.05
extra_compatible_weight=0.05
hard_target_margin_weight=1.0
```

Config table:

```text
0: tournament_margin_weight=0.5, tournament_margin_floor=0.05, compatible_orthogonal_weight=0.05
1: tournament_margin_weight=0.5, tournament_margin_floor=0.05, compatible_orthogonal_weight=0.15
2: tournament_margin_weight=0.5, tournament_margin_floor=0.15, compatible_orthogonal_weight=0.05
3: tournament_margin_weight=0.5, tournament_margin_floor=0.15, compatible_orthogonal_weight=0.15
4: tournament_margin_weight=1.0, tournament_margin_floor=0.05, compatible_orthogonal_weight=0.05
5: tournament_margin_weight=1.0, tournament_margin_floor=0.05, compatible_orthogonal_weight=0.15
6: tournament_margin_weight=1.0, tournament_margin_floor=0.15, compatible_orthogonal_weight=0.05
7: tournament_margin_weight=1.0, tournament_margin_floor=0.15, compatible_orthogonal_weight=0.15
```

Expected raw grid hash from `stable_hash_json(grid)`:

```text
a7b866c5b8c9808a67c1aa95b063ad49288b33fc5ded869058b6cba2351eb90c
```

Runtime bound-grid hashes may differ because train-pool provenance is added to
each config before evaluation.

## Logging And Monitoring

Add redacted progress events:

- `v32_support_tournament_margin_prepared`;
- `v32_support_tournament_optimizer_progress`;
- `v32_support_tournament_optimizer_completed`.

Allowed fields are hashes, counts, finite flags, scalar losses, scalar margins,
scalar norms, epoch/step counters, config IDs, variant names,
`tournament_margin_weight`, `tournament_margin_floor`,
`tournament_margin_hinge`, and `support_tournament_margin`.

Forbidden fields remain raw weights, raw deltas, raw logits, raw gradients,
selected coordinate lists, support examples, subject IDs, and final raw
paths/content.

Any long-running V32 command must use:

```bash
--monitor-interval-seconds 5 --summary-only-stdout
```

During and after compute, verify process liveness, CPU time, progress and
monitor row counts, candidate-completion events, final `monitor_stop`, log
hashes, and the forbidden-field scan over progress/monitor logs.

## Files

- Modify:
  `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Update after implementation:
  `docs/superpowers/plans/2026-06-14-v32-support-tournament-margin-editor.md`
- Create after compute:
  `docs/superpowers/plans/2026-06-14-v32-support-tournament-margin-editor-results.md`

## Implementation Tasks

- [x] Add V32 constants, matched edit source, and eight-config grid builder.
- [x] Add a V32 grid-order/hash pytest using
      `a7b866c5b8c9808a67c1aa95b063ad49288b33fc5ded869058b6cba2351eb90c`.
- [x] Update config selection, variant mapping, CLI choices, and V25-native
      control mapping.
- [x] Add V32 redaction helper without permitting raw fields.
- [x] Add support-only behavior-margin tensor builder for all four patterns and
      tests that it returns tensors, labels, counts, and hashes without raw
      examples.
- [x] Add support tournament margin loss helper with tests for satisfied and
      unsatisfied target-vs-runner margins.
- [x] Implement the V32 optimizer by extending V31 with the tournament loss,
      scalar-only logging, hard trust cap, and nonfinite fail-closed behavior.
- [x] Add V32 matched-edit evaluator and dispatch branch.
- [x] Add pytest coverage for provenance binding, native controls, redaction,
      optimizer trust-cap enforcement, dispatch, and scalar-only logging.
- [x] Run focused V32 pytest, full helper pytest, and py_compile. Do not run
      lint unless explicitly requested.
- [x] Get reviewer confidence `5/5` before compute.
- [x] Run bounded V32 development inner validation with monitor logs.
- [x] Write results and get reviewer confidence `5/5`.

## Validation Commands

Focused tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v32'
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
  --inner-validation-config-grid v32-support-tournament-margin \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Review Gates

Plan review must confirm:

- the support-tournament objective is a justified response to V31;
- all optimization inputs are support-only;
- the plan does not optimize on heldout proof or final data;
- the grid is bounded and not a capacity increase;
- monitoring and leak-audit requirements are sufficient.

Results review must confirm:

- the bounded monitored command was used;
- progress and monitor logs ended cleanly and hash-match reported output;
- candidate metrics were parsed from logs/stdout;
- forbidden-field scan over progress/monitor logs has no matches;
- interpretation remains conservative relative to preregistered gates.
