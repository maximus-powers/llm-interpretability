# V35 Support Source Line Search Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether an explicit support-only source-preserving line search can
reduce proof compatible-MSE failures without using heldout proof metrics or
sealed-final data.

**Architecture:** Add one new matched edit source that first computes the V32
support-tournament sparse edit, then chooses a scalar shrink factor using only
support split metrics. The selector ranks alpha-scaled deltas by support
compatible-source MSE subject to minimum support target/tournament margins. The
proof still uses the existing heldout/native-control path and V33 proof-gate
breakdown.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## V34 Evidence

V34 was accepted by reviewer confidence `5/5` as useful negative diagnostic
evidence. Every scalar locality-pressure candidate failed compatible MSE on all
records:

```text
V33 best target_prediction_rate=0.8333333333333334
V33 compatible_mse_fail_count=24/24
V33 control_margin_fail_count=0

V34 best target_prediction_rate=0.6666666666666666
V34 compatible_mse_fail_count=24/24
V34 control_margin_fail_count=34
```

This rules out simply increasing compatible loss weight or lowering trust norm
as the next promising step.

## Literature Basis

- AlphaEdit (https://arxiv.org/abs/2410.02355): projects perturbations onto the
  null space of preserved knowledge before applying parameter edits. V35 uses a
  simpler support-side source-preserving line search as a cheaper diagnostic
  before implementing a full null-space projector.
- "Task Arithmetic in the Tangent Space" (https://arxiv.org/abs/2305.12827):
  links reliable weight-space editing to localized function-space effects and
  tangent-space linearization. V35 explicitly evaluates scaled weight deltas by
  their support functional effects.
- "A Unified Framework for Model Editing" (https://arxiv.org/abs/2403.14236):
  frames editing as a preservation-memorization objective. V35 makes that trade
  explicit by selecting the most source-preserving alpha that still satisfies
  support target constraints.
- "Continual Model Merging without Data: Dual Projections for Stability and
  Plasticity" (https://papers.neurips.cc/paper_files/paper/2025/file/37d9f19150fce07bced2a81fc87d47a6-Paper-Conference.pdf):
  motivates separating stability and plasticity objectives. V35 uses support
  compatible MSE as stability and support target/tournament margins as
  plasticity.
- "Are We Evaluating the Edit Locality of LLM Model Editing Properly?"
  (https://arxiv.org/pdf/2601.17343): motivates behavior-sensitive locality
  evaluation. V35 keeps heldout proof locality as the real result metric and
  uses support locality only for selection.

## Hypothesis

If V34 failed because the optimizer found high-target deltas with source changes
that could be reduced by shrinking after optimization, then support-selected
alpha scaling should reduce proof `compatible_mse_fail_count` while retaining
some target prediction.

If all nonzero alphas still fail compatible MSE or the selected alphas collapse
target behavior, then the next step should be a true null-space/projection edit,
not scalar line search.

## Non-Claims

- V35 will not read `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- V35 will not run sealed-final evaluation.
- V35 will not optimize on heldout proof rows.
- V35 will not claim success unless existing development gates pass.
- V35 will not run linting unless explicitly requested by the user.

## Files

- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create results later:
  `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v35-support-source-line-search-diagnostic-results.md`

## Matched Edit Source

Add:

```python
V35_SUPPORT_SOURCE_LINE_SEARCH_MATCHED_EDIT_SOURCE = (
    "support_source_line_search_sparse"
)
V35_EXPERIMENT_VARIANT = "v35_support_source_line_search_diagnostic"
```

The new matched edit source reuses V32 coordinate selection and V32 optimizer,
then evaluates alpha-scaled candidate deltas on support tensors only.

## Support Alpha Selection

Candidate alphas:

```python
[1.0, 0.75, 0.5, 0.25, 0.125, 0.0]
```

For each alpha:

```text
candidate_delta = alpha * v32_delta
support_target_margin = target support positive mean - target support negative mean
support_runner_margin = max(other behavior support margins)
support_tournament_margin = support_target_margin - support_runner_margin
support_compatible_mse = MSE(
    edited_model(compatible_support_inputs),
    source_model(compatible_support_inputs)
)
```

Selection:

```text
eligible if:
  support_target_margin >= alpha_target_margin_floor
  support_tournament_margin >= alpha_tournament_margin_floor

choose eligible candidate with lowest support_compatible_mse;
tie-break by higher support_tournament_margin, higher support_target_margin,
larger alpha.

if no eligible alpha exists:
  choose candidate minimizing:
    support_compatible_mse
    + fallback_target_penalty * relu(alpha_target_margin_floor - support_target_margin)
    + fallback_tournament_penalty * relu(alpha_tournament_margin_floor - support_tournament_margin)
```

Default floors should be modest, because V34 showed lower norms can damage
target behavior:

```text
alpha_target_margin_floor=0.10
alpha_tournament_margin_floor=0.00
fallback_target_penalty=10.0
fallback_tournament_penalty=5.0
```

All alpha evaluation logs must be redacted: alpha, support scalar metrics,
selected alpha, candidate count, hashes, and finite flags only. Do not log raw
support examples, logits, gradients, weights, selected coordinate lists, or
subject IDs.

## V35 Grid

Add grid name:

```text
v35-support-source-line-search
```

Initial bounded grid, four configs:

```python
for trust_norm_cap in [1.0, 1.25]:
    for alpha_target_margin_floor in [0.05, 0.10]:
        ...
```

Fixed:

```text
matched_edit_source=support_source_line_search_sparse
sparse_top_k=64
sign_conflict_penalty=1.0
compatible_orthogonal_weight=0.15
extra_compatible_weight=0.05
tournament_margin_floor=0.15
tournament_margin_weight=1.0
target_margin_floor=0.25
compatible_floor=0.05
hard_target_margin_weight=1.0
alpha_candidates=[1.0,0.75,0.5,0.25,0.125,0.0]
alpha_tournament_margin_floor=0.0
fallback_target_penalty=10.0
fallback_tournament_penalty=5.0
```

Train-pool provenance must be bound in `select_v25_inner_validation_configs`.

## Long-Run Monitoring Contract

Any V35 development command must use:

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
alpha-selection events exist
proof_gate_breakdown exists in completed candidate events
progress log SHA256 recorded
monitor log SHA256 recorded
forbidden-field scan has no matches
```

Forbidden-field scan:

```bash
rg -n 'final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence' \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
```

Expected: no matches.

## Task 1: Add Failing Tests

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Add alpha selector test**

Add a pure helper test that constructs three alpha candidates and verifies the
selector chooses the lowest compatible-MSE eligible candidate, not the largest
target margin:

```python
def test_v35_alpha_selector_prefers_source_preservation_among_eligible() -> None:
    result = v25.select_v35_support_source_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 0.9,
                "support_target_margin": 0.8,
                "support_tournament_margin": 0.6,
            },
            {
                "alpha": 0.5,
                "support_compatible_mse": 0.2,
                "support_target_margin": 0.2,
                "support_tournament_margin": 0.05,
            },
            {
                "alpha": 0.125,
                "support_compatible_mse": 0.01,
                "support_target_margin": 0.03,
                "support_tournament_margin": -0.02,
            },
        ],
        alpha_target_margin_floor=0.10,
        alpha_tournament_margin_floor=0.0,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
    )

    assert result["selected_alpha"] == 0.5
    assert result["eligible_count"] == 2
```

- [ ] **Step 2: Add no-eligible fallback test**

```python
def test_v35_alpha_selector_fallback_penalizes_target_and_tournament_failures() -> None:
    result = v25.select_v35_support_source_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 0.4,
                "support_target_margin": 0.09,
                "support_tournament_margin": -0.01,
            },
            {
                "alpha": 0.0,
                "support_compatible_mse": 0.0,
                "support_target_margin": -0.5,
                "support_tournament_margin": -0.5,
            },
        ],
        alpha_target_margin_floor=0.10,
        alpha_tournament_margin_floor=0.0,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
    )

    assert result["selected_alpha"] == 1.0
    assert result["eligible_count"] == 0
```

- [ ] **Step 3: Add redaction test**

Add a progress redaction test asserting V35 alpha events omit raw terms and keep
only scalar/hash fields.

- [ ] **Step 4: Add V35 grid and variant tests**

Add tests like V33/V34:

```text
build_v35_support_source_line_search_config_grid exists
grid has four configs
matched_edit_source is support_source_line_search_sparse
selection binds train_pool_file_sha256 and train_pool_summary_hash
experiment_variant_for_inner_validation_grid("v35-support-source-line-search")
  == V35_EXPERIMENT_VARIANT
candidate result/progress labels use V35 variant
```

- [ ] **Step 5: Run focused tests and verify RED**

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q -k 'v35'
```

Expected: fail because V35 helpers/grid/source do not exist yet.

## Task 2: Implement V35 Helpers And Matched Source

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`

- [ ] **Step 1: Implement selector helper**

Implement:

```python
def select_v35_support_source_alpha_candidate(...) -> dict[str, Any]:
    ...
```

Validate finite numeric values and return selected alpha, selected metrics,
eligible count, candidate count, and a candidate metrics hash.

- [ ] **Step 2: Implement support metric evaluator**

Implement a helper that evaluates a source weight vector plus a delta on support
positive/negative behavior tensors and compatible tensors. This helper must
return only scalar metrics and hashes.

- [ ] **Step 3: Implement matched edit source**

Implement:

```python
def evaluate_v35_support_source_line_search_matched_edit(...):
    ...
```

It should call the V32 sparse solver, evaluate alpha candidates on support, log
redacted alpha-selection events, scale the delta, and return the existing
`control_record_for_delta`.

- [ ] **Step 4: Wire dispatcher and metadata**

Support the new matched edit source in `evaluate_v25_development_job`,
`experiment_variant_for_config`, `v25_config_requires_spectral_basis`,
`v25_native_control_config`, and any config-source checks required by the
existing V32 path.

## Task 3: Verify Implementation Before Compute

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q -k 'v35'
```

- [ ] **Step 2: Run full helper tests**

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q
```

- [ ] **Step 3: Compile changed files**

```bash
python -m py_compile \
  /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

- [ ] **Step 4: Request reviewer approval**

Do not start V35 compute until Kepler returns confidence `5/5`.

## Task 4: Run Bounded Monitored V35 Diagnostic

- [ ] **Step 1: Run V35**

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v35-support-source-line-search \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

- [ ] **Step 2: Monitor during run**

Check row growth and latest events every ~30 seconds.

- [ ] **Step 3: Audit output**

Record row counts, event counts, hashes, terminal `monitor_stop`, candidate
table, proof-gate breakdowns, selected alpha distribution, and leak scan.

- [ ] **Step 4: Write and review results**

Write:

```text
/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v35-support-source-line-search-diagnostic-results.md
```

Require Kepler confidence `5/5` before accepting V35.

## Self-Review

- Spec coverage: V35 directly follows V34's negative result, preserves sealed
  final boundaries, uses support-only selection, and keeps monitored logs.
- Placeholder scan: no `TBD`, `TODO`, or unspecified test steps remain.
- Type consistency: matched source and grid names are consistent throughout.
- Compute discipline: the run is bounded to four configs and 24 balanced jobs
  each, with monitor/progress logs and leak scan.
