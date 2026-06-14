# V33 Proof Gate Decomposition Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add redacted proof-gate decomposition so the next bounded V32 replay
identifies whether remaining failures are target-prediction, target-margin,
Pareto/locality, compatible-MSE, or control-advantage failures.

**Architecture:** Do not add a new editor or touch sealed-final data. Extend the
existing V25/V32 proof-record summary and progress logging with redacted
sub-gate booleans, counts, rates, and scalar margins; add a tiny V33 grid that
replays the two best V32 configs; then run a monitored development-only
diagnostic.

**Tech Stack:** Python, PyTorch CPU float32, pytest, JSONL progress logs.

---

## Current Evidence

V32 was accepted by reviewer confidence `5/5` as incremental positive evidence,
not a success. Its best final-rung candidate reached:

```text
target_prediction_rate=0.8333333333333334
mean_target_margin=0.5318264380718271
mean_matched_minus_best_control_target_margin=0.6864466418347016
mean_matched_minus_shuffled_signature_target_margin=0.6950739420584947
pareto_undominated_rate=1.0
proof_gate_failure_count=7
contract_failure_count=0
```

The result improved target prediction but every best-candidate direction still
had `individual_all_gate_pass_rate=0.0`. The current progress events show that a
record failed but not which proof sub-gate failed. Blindly adding capacity or a
new objective at this point risks optimizing the wrong bottleneck.

## Literature Basis

- Liu et al., "Are We Evaluating the Edit Locality of LLM Model Editing
  Properly?" (https://arxiv.org/pdf/2601.17343): argues that commonly used
  specificity/locality metrics can be insensitive to regularizer strength and
  recommends behavior-based evaluation. V33 decomposes locality/control failure
  modes instead of treating a proof failure as a single scalar.
- He et al., "Knowledge Updating? No More Model Editing! Just Selective
  Contextual Reasoning" (https://arxiv.org/html/2503.05212v1): evaluates model
  editing across reliability, generalization, locality, and portability. V33 maps
  the local proof gates onto that same idea: reliability is target prediction and
  target margin, locality is Pareto/control advantage, and compatibility is
  source-output MSE.
- Balloccu et al., "Leak, Cheat, Repeat" (https://aclanthology.org/2024.eacl-long.5/):
  documents contamination and evaluation malpractice risks including missing
  baselines and reproducibility issues. V33 keeps sealed-final data unopened,
  logs only redacted proof diagnostics, records hashes, and keeps native controls
  visible in aggregate.
- Han et al., "A Survey of Weight Space Learning" (https://arxiv.org/html/2603.10090v1):
  frames neural weights as a structured domain for understanding,
  representation, and generation. V33 keeps the evidence chain in weight space
  but focuses on whether the learned/steered weight edit has reliable functional
  effects.
- Kaushik et al., "The Universal Weight Subspace Hypothesis"
  (https://arxiv.org/abs/2512.05117): reports shared low-dimensional spectral
  subspaces across many models. V33 does not claim universality, but the result
  interpretation should distinguish spectral/weight-space plausibility from the
  stricter behavioral proof gates.

## Hypothesis

If V32 is close but blocked by a specific proof sub-gate, then a redacted replay
of its best configs should show a concentrated failure pattern. Examples:

```text
target_prediction_fail_count high        -> still a ranking/target issue
target_margin_fail_count high            -> target wins weakly or inconsistently
control_margin_fail_count high           -> edit is not beating native controls
compatible_mse_fail_count high           -> locality/source preservation issue
pareto_fail_count high                   -> controls dominate on proof metrics
```

If the failures are diffuse or vary by source/target direction, the next editor
change should be direction-aware rather than a global capacity increase.

## Non-Claims

- V33 will not read `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- V33 will not run sealed-final evaluation.
- V33 will not optimize on heldout proof rows.
- V33 will not introduce a new matched edit source.
- V33 will not claim success unless the existing development gates pass.
- V33 will not run linting unless explicitly requested by the user.

## Files

- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create results later:
  `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v33-proof-gate-decomposition-diagnostic-results.md`

## Redacted Diagnostic Schema

For each proof record, add a summary payload with only constants, booleans,
counts, hashes, and scalar metrics:

```python
{
    "target_prediction_pass": bool,
    "target_margin_pass": bool,
    "pareto_undominated": bool,
    "compatible_mse_pass": bool,
    "control_margin_pass_count": int,
    "control_margin_fail_count": int,
    "failed_control_types_hash": str,
    "min_control_margin_advantage": float,
    "mean_control_margin_advantage": float,
    "shuffled_signature_margin_pass": bool,
    "individual_all_gates_passed": bool,
}
```

Do not log raw weights, deltas, logits, gradients, selected coordinates,
support examples, subject IDs, raw sequences, or final raw paths/content.

## Bounded V33 Grid

Add grid name:

```text
v33-proof-gate-diagnostic
```

It returns the two V32 best configs from V32 results:

```python
[
    {
        "config_index": 0,
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "sparse_top_k": 64,
        "trust_norm_cap": 1.25,
        "sign_conflict_penalty": 1.0,
        "compatible_orthogonal_weight": 0.15,
        "tournament_margin_floor": 0.15,
        "tournament_margin_weight": 1.0,
        "target_margin_floor": 0.25,
        "compatible_floor": 0.05,
        "extra_compatible_weight": 0.05,
        "hard_target_margin_weight": 1.0,
    },
    {
        "config_index": 1,
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "sparse_top_k": 64,
        "trust_norm_cap": 1.25,
        "sign_conflict_penalty": 1.0,
        "compatible_orthogonal_weight": 0.05,
        "tournament_margin_floor": 0.15,
        "tournament_margin_weight": 1.0,
        "target_margin_floor": 0.25,
        "compatible_floor": 0.05,
        "extra_compatible_weight": 0.05,
        "hard_target_margin_weight": 1.0,
    },
]
```

Train-pool provenance must be bound in `select_v25_inner_validation_configs`
exactly like V32.

## Long-Run Monitoring Contract

Any V33 development command must use:

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
proof decomposition fields exist in completed-record events
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

## Task 1: Add Failing Tests For Proof-Gate Decomposition

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Add proof-record subgate test**

Add a test near `test_v25_proof_record_computes_pareto_and_gate_summaries`:

```python
def test_v25_proof_record_exposes_redacted_gate_decomposition() -> None:
    controls = make_v25_controls(shuffled_margin=0.28)
    record = v25.build_v25_proof_record(
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        matched=make_v25_matched(target_margin=0.30, compatible_mse=0.20),
        controls=controls,
    )

    diagnostics = record["summary"]["proof_gate_diagnostics"]

    assert diagnostics["target_prediction_pass"] is True
    assert diagnostics["target_margin_pass"] is True
    assert diagnostics["pareto_undominated"] is True
    assert diagnostics["compatible_mse_pass"] is False
    assert diagnostics["control_margin_fail_count"] >= 1
    assert diagnostics["control_margin_pass_count"] >= 1
    assert len(diagnostics["failed_control_types_hash"]) == 64
    assert "shuffled_signature" not in json.dumps(diagnostics, sort_keys=True)
```

- [ ] **Step 2: Add aggregate breakdown test**

Extend `test_v25_aggregate_records_compute_required_gate_metrics`:

```python
    breakdown = result["aggregate"]["proof_gate_breakdown"]
    assert breakdown["record_count"] == 2
    assert breakdown["target_prediction_fail_count"] == 1
    assert breakdown["target_margin_fail_count"] == 1
    assert breakdown["control_margin_fail_count"] >= 1
    assert "control_margin_failure_type_counts_hash" in breakdown
```

- [ ] **Step 3: Add progress redaction test**

Extend `test_v25_development_job_with_progress_logs_redacted_record_events`:

```python
    completed = events[1]
    assert "proof_gate_diagnostics" in completed
    assert "compatible_mse_pass" in completed["proof_gate_diagnostics"]
    assert len(completed["proof_gate_diagnostics"]["failed_control_types_hash"]) == 64
```

- [ ] **Step 4: Add V33 grid test**

Add near the V32 grid tests:

```python
def test_v33_diagnostic_grid_replays_two_best_v32_configs() -> None:
    grid = v25.build_v33_proof_gate_diagnostic_config_grid()

    assert len(grid) == 2
    assert [item["config_index"] for item in grid] == [0, 1]
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE
    }
    assert {item["trust_norm_cap"] for item in grid} == {1.25}
    assert {item["tournament_margin_weight"] for item in grid} == {1.0}
    assert {item["tournament_margin_floor"] for item in grid} == {0.15}
    assert [item["compatible_orthogonal_weight"] for item in grid] == [0.15, 0.05]
```

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'proof_record_exposes_redacted_gate_decomposition or aggregate_records_compute_required_gate_metrics or development_job_with_progress_logs_redacted_record_events or v33'
```

Expected: fail because V33 helpers and diagnostic fields do not exist yet.

## Task 2: Implement Proof-Gate Decomposition

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`

- [ ] **Step 1: Add helper for record diagnostics**

Create:

```python
def build_v25_proof_gate_diagnostics(
    *,
    matched_payload: Mapping[str, Any],
    margin_pass_payloads: Sequence[Mapping[str, Any]],
    compatible_mse_pass: bool,
) -> dict[str, Any]:
    advantages = [
        float(payload["advantage"])
        for payload in margin_pass_payloads
    ]
    failed_control_types = sorted(
        str(payload["control_type"])
        for payload in margin_pass_payloads
        if not bool(payload["passed"])
    )
    return {
        "compatible_mse_pass": bool(compatible_mse_pass),
        "control_margin_fail_count": len(failed_control_types),
        "control_margin_pass_count": len(margin_pass_payloads) - len(failed_control_types),
        "failed_control_types_hash": stable_hash_json(failed_control_types),
        "individual_all_gates_passed": bool(matched_payload["individual_all_gates_passed"]),
        "mean_control_margin_advantage": mean_float(advantages),
        "min_control_margin_advantage": min(advantages) if advantages else 0.0,
        "pareto_undominated": bool(matched_payload["pareto_undominated"]),
        "shuffled_signature_margin_pass": all(
            bool(payload["passed"])
            for payload in margin_pass_payloads
            if str(payload["control_type"]) == "shuffled_signature"
        ),
        "target_margin_pass": (
            float(matched_payload["target_margin"]) >= PER_RECORD_MIN_TARGET_MARGIN
        ),
        "target_prediction_pass": bool(matched_payload["target_prediction_pass"]),
    }
```

- [ ] **Step 2: Store diagnostics in proof summary**

In `build_v25_proof_record`, replace the current bare `margin_passes` list with
payloads containing `control_type`, `advantage`, `threshold`, and `passed`.
Then attach:

```python
summary["proof_gate_diagnostics"] = build_v25_proof_gate_diagnostics(
    matched_payload=matched_payload,
    margin_pass_payloads=margin_pass_payloads,
    compatible_mse_pass=compatible_mse_pass,
)
```

- [ ] **Step 3: Add aggregate breakdown helper**

Create:

```python
def summarize_v25_proof_gate_breakdown(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ...
```

It should count record-level failures for each boolean gate and hash a
`dict[str, int]` of failed control types instead of exposing the names in
candidate summaries.

- [ ] **Step 4: Include breakdown in aggregate and candidate summaries**

Set:

```python
aggregate["proof_gate_breakdown"] = summarize_v25_proof_gate_breakdown(records)
```

In `summarize_v25_inner_validation_candidate`, expose:

```python
"proof_gate_breakdown": aggregate["proof_gate_breakdown"],
```

- [ ] **Step 5: Include record diagnostics in redacted progress**

Update `redacted_v25_development_record_progress`:

```python
"proof_gate_diagnostics": dict(proof_record["summary"]["proof_gate_diagnostics"]),
```

Do not include per-control type names in progress events.

## Task 3: Add V33 Diagnostic Grid

**Files:**
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify:
  `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Add grid builder**

Add:

```python
def build_v33_proof_gate_diagnostic_config_grid() -> list[dict[str, Any]]:
    base = {
        "compatible_floor": float(V32_COMPATIBLE_FLOOR),
        "extra_compatible_weight": float(V32_EXTRA_COMPATIBLE_WEIGHT),
        "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
        "matched_edit_source": V32_SUPPORT_TOURNAMENT_MATCHED_EDIT_SOURCE,
        "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
        "sparse_top_k": int(V32_SPARSE_TOP_K),
        "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
        "tournament_margin_floor": 0.15,
        "tournament_margin_weight": 1.0,
        "trust_norm_cap": float(V32_TRUST_NORM_CAP),
    }
    return [
        {**base, "compatible_orthogonal_weight": 0.15, "config_index": 0},
        {**base, "compatible_orthogonal_weight": 0.05, "config_index": 1},
    ]
```

- [ ] **Step 2: Wire grid selection**

In `select_v25_inner_validation_configs`, support
`"v33-proof-gate-diagnostic"` and bind `train_pool_file_sha256` and
`train_pool_summary_hash` exactly as V32 does.

- [ ] **Step 3: Wire parser choices and variant mapping**

Add the grid name to `--inner-validation-config-grid` choices. If
`experiment_variant_for_inner_validation_grid` has an explicit mapping, map V33
to a diagnostic variant string such as:

```python
"v33_proof_gate_decomposition_diagnostic"
```

## Task 4: Verify Implementation Before Compute

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'proof_record_exposes_redacted_gate_decomposition or aggregate_records_compute_required_gate_metrics or development_job_with_progress_logs_redacted_record_events or v33'
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full helper tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: all helper tests pass.

- [ ] **Step 3: Compile changed Python files**

Run:

```bash
python -m py_compile \
  /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Expected: exit code `0`.

- [ ] **Step 4: Request reviewer approval**

Send the plan, diff summary, and verification output to Kepler. Do not start V33
compute until the reviewer returns confidence `5/5`.

## Task 5: Run Bounded Monitored V33 Diagnostic

- [ ] **Step 1: Run the diagnostic**

Use:

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v33-proof-gate-diagnostic \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

- [ ] **Step 2: Monitor during run**

Check the process and latest events without reading final raw data:

```bash
tail -n 5 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
tail -n 5 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl
```

- [ ] **Step 3: Audit output**

Record row counts, event counts, hashes, terminal `monitor_stop`, best candidate,
and proof-gate breakdown. Run the forbidden-field scan from the monitoring
contract.

- [ ] **Step 4: Write results doc and request review**

Write:

```text
/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v33-proof-gate-decomposition-diagnostic-results.md
```

Include citations, command, hashes, monitor evidence, leak audit, candidate
table, proof-gate breakdown, interpretation, and reviewer confidence. Require
Kepler confidence `5/5` before treating the diagnostic as accepted.

## Self-Review

- Spec coverage: the plan adds logs for long-running work, decomposes
  misleading proof failures, preserves sealed-final boundaries, includes current
  literature, and requires review before compute/results acceptance.
- Placeholder scan: no `TBD`, `TODO`, or unspecified "write tests" steps remain.
- Type consistency: the record-level key is consistently
  `proof_gate_diagnostics`; aggregate/candidate key is consistently
  `proof_gate_breakdown`.
- Compute discipline: the only planned compute is two V32 replay configs with
  24 balanced development jobs each, plus monitor logs and post-run audit.
