# V36 Compatible-Nullspace Projection Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development and superpowers:executing-plans.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether an explicit support-compatible Jacobian null-space
projection can reduce compatible-MSE failures more effectively than V35 scalar
line search, without using heldout proof rows or sealed-final data.

**Architecture:** Add one new matched edit source that computes the V32 support
tournament sparse delta, projects that delta away from the row space of the
support compatible-logit Jacobian, then runs the V35 support-only alpha selector
on the projected delta. Proof evaluation remains the existing heldout/native
control path.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## V35 Evidence

V35 was accepted by reviewer confidence `5/5` as a bounded leak-free diagnostic.
It did not pass proof gates, but it changed the failure profile:

```text
V33/V34 compatible_mse_fail_count: 24/24
V35 best compatible_mse_fail_count: 18/24
V35 selected best config compatible_mse_fail_count: 19/24
V35 selected best target_prediction_rate: 0.75
V35 selected best control_margin_fail_count: 55
```

This suggests scalar shrinking has some source-preservation signal but cannot
solve locality by itself. The next experiment should alter the edit direction,
not just its magnitude.

## Literature Basis

- AlphaEdit (https://arxiv.org/abs/2410.02355): projects edit perturbations
  into the null space of preserved knowledge and reports large gains for
  locate-then-edit methods. V36 adapts the core idea to this small model by
  projecting the sparse delta away from compatible-support logit Jacobian rows.
- "Task Arithmetic in the Tangent Space"
  (https://arxiv.org/abs/2305.12827): shows weight edits work best when weight
  directions correspond to localized function-space effects. V36 uses a
  tangent/Jacobian approximation to identify directions that should not affect
  source-compatible behavior.
- "Continual Model Merging without Data: Dual Projections for Stability and
  Plasticity"
  (https://openreview.net/forum?id=zD5cUX67b9): frames stability and plasticity
  via orthogonal projections. V36 explicitly treats compatible behavior as the
  stability subspace and target/tournament margins as plasticity.
- "Are We Evaluating the Edit Locality of LLM Model Editing Properly?"
  (https://arxiv.org/abs/2601.17343): argues locality metrics must be sensitive
  to preservation strength. V36 keeps the heldout compatible-MSE proof gate as
  the real locality result, using support locality only for edit selection.
- "Model Editing Harms General Abilities of Large Language Models"
  (https://arxiv.org/abs/2401.04700): attributes side effects partly to
  excessive weight changes and motivates regularized/complexity-constrained
  updates. V36 constrains the update by removing first-order source-affecting
  components rather than simply shrinking norm.
- "Lifelong Knowledge Editing requires Better Regularization"
  (https://arxiv.org/abs/2502.01636): identifies over-optimization and norm
  growth as degradation causes in locate-then-edit methods. V36 avoids another
  heavier optimizer and instead performs a bounded projection plus line search.

## Hypothesis

If V35 failed because the base sparse delta still contains components that
first-order affect compatible-source behavior, then projecting the delta away
from the compatible-support Jacobian row space should reduce proof
`compatible_mse_fail_count` below V35's `18/24` while retaining some target
prediction.

If target prediction collapses or compatible-MSE remains high, then the support
compatible Jacobian is not the right preservation subspace for this setup, and
the next step should be a dual-objective solver in a learned basis rather than a
post-hoc projection.

## Non-Claims

- V36 will not read
  `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- V36 will not run sealed-final evaluation.
- V36 will not optimize on heldout proof rows.
- V36 will not claim success unless existing development proof gates pass.
- V36 will not run linting unless explicitly requested by the user.

## Files

- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create results later:
  `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v36-compatible-nullspace-projection-diagnostic-results.md`

## Matched Edit Source

Add:

```python
V36_COMPATIBLE_NULLSPACE_MATCHED_EDIT_SOURCE = (
    "compatible_nullspace_projected_sparse"
)
V36_EXPERIMENT_VARIANT = "v36_compatible_nullspace_projection_diagnostic"
```

The new matched edit source reuses V32 sparse coordinate selection and optimizer,
then projects the resulting dense sparse delta using support-compatible Jacobian
rows before V35 alpha selection.

## Projection Method

Reuse `v28_anchor_gradients_and_compatible_jacobian(...)` to obtain:

```text
compatible_jacobian: [compatible_support_logit_count, SOURCE_WEIGHT_DIM]
```

Build a preserve row-space basis directly from the raw compatible Jacobian. Do
not center rows: the preservation target is the first-order compatible-logit
change `Jc @ delta`, and centering would leave the mean compatible-output
direction unprotected.

```text
U, S, Vh = svd(compatible_jacobian)
preserve = Vh[S / max(S) > compatible_nullspace_rtol].T
```

Use:

```text
normalized_singular_values = S / max(max(S), 1e-12)
preserve_mask = normalized_singular_values > compatible_nullspace_rtol
```

If no singular vectors pass `compatible_nullspace_rtol`, set:

```text
preserve_rank = 0
row_component = 0
projected_delta = base_delta
```

Scalar diagnostics must still be finite in this case.

Project the base delta:

```text
row_component = preserve @ preserve.T @ base_delta
projected_delta = base_delta - projection_strength * row_component
projected_delta = apply_norm_cap(projected_delta, trust_norm_cap)
```

Record support-only scalar diagnostics:

```text
base_preservation_energy = ||Jc @ base_delta||
projected_preservation_energy = ||Jc @ projected_delta||
preservation_energy_ratio =
  projected_preservation_energy / max(base_preservation_energy, 1e-12)
projection_removed_norm = ||projection_strength * row_component||
projection_retained_norm = ||projected_delta||
preserve_rank
jacobian_row_count
```

Then evaluate alpha candidates with the V35 selector using the projected delta.

## V36 Grid

Add grid name:

```text
v36-compatible-nullspace-projection
```

Initial bounded grid, four configs:

```python
for compatible_nullspace_rtol in [1e-4, 1e-3]:
    for projection_strength in [0.75, 1.0]:
        ...
```

Fixed:

```text
matched_edit_source=compatible_nullspace_projected_sparse
sparse_top_k=64
sign_conflict_penalty=1.0
compatible_orthogonal_weight=0.15
extra_compatible_weight=0.05
tournament_margin_floor=0.15
tournament_margin_weight=1.0
target_margin_floor=0.25
compatible_floor=0.05
hard_target_margin_weight=1.0
trust_norm_cap=1.25
alpha_candidates=[1.0,0.75,0.5,0.25,0.125,0.0]
alpha_target_margin_floor=0.05
alpha_tournament_margin_floor=0.0
fallback_target_penalty=10.0
fallback_tournament_penalty=5.0
```

Train-pool provenance must be bound in `select_v25_inner_validation_configs`.

## Logging Contract

All projection and alpha logs must be redacted. Allowed projection fields:

```text
base_preservation_energy
compatible_nullspace_rtol
jacobian_row_count
preservation_energy_ratio
preserve_rank
projected_preservation_energy
projection_removed_norm
projection_retained_norm
projection_strength
finite
hashes/counts
```

Do not log raw support examples, logits, gradients, weights, selected coordinate
lists, sequences, subject IDs, full Jacobian rows, basis vectors, or raw deltas.

## Long-Run Monitoring Contract

Any V36 development command must use:

```bash
--monitor-interval-seconds 5 --summary-only-stdout
```

Accept the run only after checking:

```text
process exited
progress log row count > 0
monitor log row count > 0
monitor terminal event == monitor_stop
candidate completion events exist
projection events exist
alpha-selection events exist
proof_gate_breakdown exists in completed candidate events
progress log SHA256 recorded
monitor log SHA256 recorded
forbidden-field scan has no matches
```

Forbidden-field scan:

```bash
rg -n 'final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence|compatible_jacobian|raw_delta' \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
```

Expected: no matches.

Note: do not scan for the bare substring `basis`, because existing inherited
candidate summaries include redacted `spectral_basis_sha256` hashes. Raw basis
vectors remain forbidden and should be covered by redaction tests, not a broad
substring scan that also catches hash field names.

## Task 1: Add Failing Tests

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] Add a pure projection helper test where a row-space-aligned delta has its
  preservation energy `||Jc @ delta||` reduced by projection.
- [ ] Add a zero-compatible-Jacobian test proving `preserve_rank=0`, the
  projected delta equals the base delta, and scalar diagnostics are finite.
- [ ] Add a projection redaction test proving raw Jacobian/basis/delta/vector
  fields are omitted while scalar energy diagnostics and hashes remain.
- [ ] Add a bounded-grid/provenance test with stable grid hash and
  `v36-compatible-nullspace-projection` variant mapping.
- [ ] Add a matched-edit wrapper test that monkeypatches the expensive V32
  solver and compatible Jacobian, then verifies projected delta is used before
  V35 alpha selection.
- [ ] Add dispatcher and CLI tests for the new matched edit source/grid.

Run:

```bash
pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k v36 -q
```

Expected: fail on missing V36 helpers/branches.

## Task 2: Implement V36 Helpers and Grid

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`

- [ ] Add constants, grid builder, grid hash, variant mapping, CLI choice, and
  native-control mapping.
- [ ] Add `project_v36_delta_through_compatible_nullspace(...)`.
- [ ] Add `redact_v36_compatible_nullspace_progress_event(...)`.
- [ ] Reuse V35 alpha selector output including `candidate_metrics_hash` and
  `eligible_count`.

Run focused tests. Expected: V36 helper/grid tests pass; matched edit may still
fail until Task 3.

## Task 3: Implement Matched Edit and Dispatch

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`

- [ ] Add `evaluate_v25_compatible_nullspace_projected_matched_edit(...)`.
- [ ] Reuse V32 sparse optimizer for the base delta.
- [ ] Compute compatible support Jacobian only from support split.
- [ ] Project base delta, then call V35 support alpha selection on projected
  delta.
- [ ] Log projection completion and alpha selection with redacted fields only.
- [ ] Add `evaluate_v25_development_job(...)` dispatch branch.

Run:

```bash
pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v36 or parse_args' -q
```

Expected: pass.

## Task 4: Verification and Review Gate

Run:

```bash
pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
python -m py_compile \
  /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Do not run linting.

Send implementation to reviewer. Proceed only after reviewer confidence `5/5`.

## Task 5: Run Bounded V36 Diagnostic

Only after implementation review confidence `5/5`, run:

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v36-compatible-nullspace-projection \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

Monitor every ~30 seconds using process status plus progress/monitor row counts.

## Task 6: Results Document and Review

Create:

```text
/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v36-compatible-nullspace-projection-diagnostic-results.md
```

Include:

- command
- row counts and monitor terminal event
- progress and monitor SHA256
- forbidden-field scan result
- projection event count
- alpha-selection count
- candidate table
- comparison to V35
- conservative interpretation

Send to reviewer. Accept only at confidence `5/5`.
