# V34 Locality Pressure Grid Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether V33's dominant compatible-MSE/locality failure can be
reduced by stronger support-compatible preservation pressure and smaller edit
norms, without adding a new editor or touching sealed-final data.

**Architecture:** Add a small V34 config grid that reuses the V32
`support_tournament_margin_sparse` matched edit source. Vary only
`trust_norm_cap`, `extra_compatible_weight`, and a fixed high
`compatible_orthogonal_weight`, then use V33's proof-gate breakdown to evaluate
the reliability/locality tradeoff on development data.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## V33 Evidence

V33 was accepted by reviewer confidence `5/5` as a diagnostic, not success.
Both replay configs reproduced V32-level target prediction and exposed the same
proof-gate blocker:

```text
target_prediction_rate=0.8333333333333334
proof_gate_failure_count=7
compatible_mse_fail_count=24/24
target_prediction_fail_count=4/24
target_margin_fail_count=4/24
pareto_fail_count=0
control_margin_fail_count=0
```

This means the edit already beats proof-critical controls on target margin and
is Pareto-undominated, but it changes compatible/source behavior too much.

## Literature Basis

- "Lifelong Knowledge Editing requires Better Regularization"
  (https://arxiv.org/html/2502.01636v2): treats locality as whether neighboring
  facts remain unaffected, motivating stronger regularization when edits are
  repeatedly applied.
- "A Unified Framework for Model Editing" (https://arxiv.org/abs/2403.14236):
  frames ROME/MEMIT under a preservation-memorization objective. V34 directly
  tests the preservation side before adding more memorization capacity.
- M-ORE, "Modality-Decoupled Online Recursive Editing"
  (https://arxiv.org/html/2605.20273v1): reports gains by balancing reliability,
  generality, and locality while preserving general capability.
- "Beyond Hard Writes and Rigid Preservation: Soft Recursive Least-Squares for
  Lifelong LLM Editing" (https://arxiv.org/html/2601.15686v1): motivates soft
  preservation constraints rather than hard writes that damage unrelated
  behavior.
- "Model Editing Harms General Abilities of Large Language Models:
  Regularization to the Rescue" (https://arxiv.org/html/2401.04700v3): supports
  regularization as a response to edit-induced degradation.
- "Continual Model Merging without Data: Dual Projections for Stability and
  Plasticity" (https://papers.neurips.cc/paper_files/paper/2025/file/37d9f19150fce07bced2a81fc87d47a6-Paper-Conference.pdf):
  motivates treating stability and plasticity as competing objectives. V34 is a
  grid diagnostic for that tradeoff before implementing projection.

## Hypothesis

If V33's compatible-MSE failures are caused mainly by excessive edit norm or too
weak support-compatible pressure, a V34 grid with smaller norm caps and larger
compatible weights should reduce `compatible_mse_fail_count`, possibly at the
cost of lower target prediction or target margin.

If compatible-MSE remains `24/24` failed even under smaller norms and stronger
support-compatible pressure, the next version should implement a more explicit
source-preserving projection or line-search method rather than further scalar
weight tuning.

## Non-Claims

- V34 will not read `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- V34 will not run sealed-final evaluation.
- V34 will not optimize on heldout proof rows.
- V34 will not introduce a new matched edit source.
- V34 will not claim success unless existing development gates pass.
- V34 will not run linting unless explicitly requested by the user.

## Files

- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create results later:
  `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v34-locality-pressure-grid-diagnostic-results.md`

## V34 Grid

Add grid name:

```text
v34-locality-pressure
```

Grid:

```python
[
    {
        "config_index": 0,
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "sparse_top_k": 64,
        "trust_norm_cap": 0.50,
        "sign_conflict_penalty": 1.0,
        "compatible_orthogonal_weight": 0.15,
        "extra_compatible_weight": 0.50,
        "tournament_margin_floor": 0.15,
        "tournament_margin_weight": 1.0,
        "target_margin_floor": 0.25,
        "compatible_floor": 0.05,
        "hard_target_margin_weight": 1.0,
    },
    # same for trust_norm_cap=0.75, 1.00
    # and extra_compatible_weight=0.50, 2.00
]
```

Total configs: `6`.

Expected raw grid hash from `stable_hash_json(grid)`:

```text
4ee212a749e6db4210ce7ac096e1d5884130d38a2693a7272a23ab354229f722
```

Train-pool provenance must be bound in `select_v25_inner_validation_configs`
exactly like V32/V33.

Variant string:

```python
V34_EXPERIMENT_VARIANT = "v34_locality_pressure_grid_diagnostic"
```

## Long-Run Monitoring Contract

Any V34 development command must use:

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

- [ ] **Step 1: Add V34 grid test**

Add near the V33 grid tests:

```python
def test_v34_locality_pressure_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v34_locality_pressure_config_grid()

    assert len(grid) == 6
    assert v25.stable_hash_json(grid) == v25.V34_LOCALITY_PRESSURE_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(6))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE
    }
    assert {item["compatible_orthogonal_weight"] for item in grid} == {0.15}
    assert sorted({item["trust_norm_cap"] for item in grid}) == [0.5, 0.75, 1.0]
    assert sorted({item["extra_compatible_weight"] for item in grid}) == [0.5, 2.0]

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v34-locality-pressure",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid("v34-locality-pressure") == (
        v25.V34_EXPERIMENT_VARIANT
    )
```

- [ ] **Step 2: Add V34 variant-label regression**

Copy the V33 variant-label regression and change the grid/variant to V34. It
must assert:

```python
assert result["experiment_variant"] == v25.V34_EXPERIMENT_VARIANT
assert result["best_candidate"]["experiment_variant"] == v25.V34_EXPERIMENT_VARIANT
assert any(
    event["event"] == "inner_validation_candidate_completed"
    and event["experiment_variant"] == v25.V34_EXPERIMENT_VARIANT
    for event in events
)
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q -k 'v34'
```

Expected: fail because V34 grid and variant do not exist yet.

## Task 2: Implement V34 Grid

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`

- [ ] **Step 1: Add constant**

```python
V34_EXPERIMENT_VARIANT = "v34_locality_pressure_grid_diagnostic"
```

- [ ] **Step 2: Add grid builder**

```python
def build_v34_locality_pressure_config_grid() -> list[dict[str, Any]]:
    grid = []
    for trust_norm_cap in [0.5, 0.75, 1.0]:
        for extra_compatible_weight in [0.5, 2.0]:
            grid.append({
                "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                "compatible_orthogonal_weight": 0.15,
                "config_index": len(grid),
                "extra_compatible_weight": float(extra_compatible_weight),
                "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
                "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                "sparse_top_k": int(V32_SPARSE_TOP_K),
                "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                "tournament_margin_floor": 0.15,
                "tournament_margin_weight": 1.0,
                "trust_norm_cap": float(trust_norm_cap),
            })
    return grid
```

- [ ] **Step 3: Wire selection, variant mapping, and CLI choices**

Support `v34-locality-pressure` in:

```text
select_v25_inner_validation_configs
experiment_variant_for_inner_validation_grid
--inner-validation-config-grid choices
```

Bind train-pool provenance exactly like V32/V33.

## Task 3: Verify Implementation Before Compute

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q -k 'v34'
```

Expected: V34 tests pass.

- [ ] **Step 2: Run full helper tests**

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q
```

Expected: all helper tests pass.

- [ ] **Step 3: Compile changed Python files**

```bash
python -m py_compile \
  /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Expected: exit code `0`.

- [ ] **Step 4: Request reviewer approval**

Send the plan, implementation summary, and verification output to Kepler. Do
not start V34 compute until the reviewer returns confidence `5/5`.

## Task 4: Run Bounded Monitored V34 Diagnostic

- [ ] **Step 1: Run V34**

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v34-locality-pressure \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

- [ ] **Step 2: Monitor during run**

Check row growth and latest events every ~30 seconds:

```bash
tail -n 5 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
tail -n 8 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl
```

- [ ] **Step 3: Audit output**

Record row counts, event counts, hashes, terminal `monitor_stop`, candidate
table, and proof-gate breakdowns. Run the forbidden-field scan.

- [ ] **Step 4: Write and review results**

Write:

```text
/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v34-locality-pressure-grid-diagnostic-results.md
```

Require Kepler confidence `5/5` before accepting the V34 diagnostic.

## Self-Review

- Spec coverage: V34 directly tests the V33 compatible-MSE/locality blocker,
  keeps sealed-final data untouched, keeps monitored logs, and requires review
  before compute/results acceptance.
- Placeholder scan: no `TBD`, `TODO`, or unspecified test steps remain.
- Type consistency: grid name is consistently `v34-locality-pressure`; variant
  constant is consistently `V34_EXPERIMENT_VARIANT`.
- Compute discipline: the run is bounded to six configs and 24 balanced jobs
  each, with monitor/progress logs and leak scan.
