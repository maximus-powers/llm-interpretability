# V38 Compatible-MSE Gated Projected Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce V37's compatible-MSE locality failures while preserving its recovered target plasticity.

**Architecture:** V38 keeps the V37 target-aware projected optimizer, but adds compatible-MSE gating as a first-class support constraint in both optimizer checkpoint ranking and alpha selection. The run remains a bounded four-config development diagnostic; heldout proof remains evaluation-only.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing V25-V37 runner.

---

## Literature Support

- [BalancEdit](https://arxiv.org/abs/2505.01343) explicitly frames editing as a generality-locality tradeoff and motivates dynamic balancing rather than optimizing only edit success.
- [AlphaEdit](https://arxiv.org/abs/2410.02355) supports null-space preservation for locality, but V36/V37 show projection alone is not enough in this small setting.
- [GNSP](https://arxiv.org/html/2507.19839v1) motivates tuning the null-space/plasticity threshold; V38 keeps the V37 projection range but adds a locality gate.
- [Are We Evaluating the Edit Locality of LLM Model Editing Properly?](https://arxiv.org/pdf/2601.17343) motivates behavior-level locality checks rather than relying on parameter-space distance.
- [ENFORCE](https://arxiv.org/html/2502.06774v4) supports the broader principle of constraint enforcement through projection/gating while maintaining gradient flow.
- [Model Merging by Output-Space Projection](https://arxiv.org/abs/2605.29101) supports output-space calibration as the relevant preservation surface.

## Prior Result Constraint

V37 is accepted as a 5/5-reviewed bounded diagnostic:

- Best target-plastic config: target prediction `0.7083`, mean target margin `0.2846`.
- Main failure: compatible-MSE locality `19/24` failures on the selected best config.
- Best locality config: compatible-MSE failures `10/24`, but target prediction only `0.2917`.

V38 should not chase more target margin until compatible-MSE is part of the support objective and selection rule.

## Files

- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create after run: `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v38-compatible-mse-gated-projected-optimizer-results.md`

## Boundaries

- Never read `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- Use support tensors only for optimization and alpha selection.
- Heldout proof remains evaluation-only.
- Do not run lint.
- Long run must use monitor interval `5`, progress JSONL, monitor JSONL, terminal log redirection, PID checks, hashes, and leak scan.
- Raw forbidden JSON keys: `final_subjects`, `subject_id`, `weights`, `logits`, `gradient`, `selected_coordinates`, `support_examples`, `sequence`, `compatible_jacobian`, `raw_delta`, `projected_delta`.

## Task 1: Add V38 Grid, Routing, And Redaction Coverage

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write RED grid/routing test**

Add:

```python
def test_v38_compatible_gated_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v38_compatible_mse_gated_projected_optimizer_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V38_COMPATIBLE_GATED_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["compatible_mse_gate"] for item in grid}) == [5.0, 15.0]
    assert sorted({item["compatible_gate_weight"] for item in grid}) == [0.5, 1.5]

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v38-compatible-mse-gated-projected-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v38-compatible-mse-gated-projected-optimizer"
    ) == v25.V38_EXPERIMENT_VARIANT
```

- [ ] **Step 2: Run RED**

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v38_compatible_gated_grid' -q
```

Expected: fail on missing V38 symbols.

- [ ] **Step 3: Implement constants/grid/routing**

Add constants near V37:

```python
V38_EXPERIMENT_VARIANT = "v38_compatible_mse_gated_projected_optimizer_diagnostic"
V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE = "compatible_mse_gated_projected_optimizer_sparse"
V38_COMPATIBLE_GATED_GRID_SHA256 = (
    "17513c7f6e091466a3aa364ef4e927591bb6191c8a3cbebb3ca3fa9f2da7895e"
)
```

Add grid:

```python
def build_v38_compatible_mse_gated_projected_optimizer_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for compatible_mse_gate in [5.0, 15.0]:
        for compatible_gate_weight in [0.5, 1.5]:
            grid.append({
                **build_v37_projected_support_optimizer_config_grid()[2],
                "alpha_compatible_mse_gate": float(compatible_mse_gate),
                "compatible_gate_weight": float(compatible_gate_weight),
                "compatible_mse_gate": float(compatible_mse_gate),
                "config_index": len(grid),
                "matched_edit_source": V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE,
                "projected_optimizer_event_prefix": "v38_compatible_gated_optimizer",
            })
    return grid
```

Add selection, variant mapping, native-control mapping, spectral-basis bypass, and CLI choice for `v38-compatible-mse-gated-projected-optimizer`. The focused grid test must pass with hash `17513c7f6e091466a3aa364ef4e927591bb6191c8a3cbebb3ca3fa9f2da7895e`.

## Task 2: Add Compatible-MSE Gating To Optimizer Scoring

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write RED optimizer-gate test**

Add:

```python
def test_v38_score_defaults_match_v37_formula() -> None:
    kwargs = {
        "support_target_margin": -0.1,
        "support_tournament_margin": -0.2,
        "support_compatible_mse": 12.0,
        "loss": 30.0,
        "target_margin_floor": 0.05,
        "tournament_margin_floor": 0.15,
    }
    expected = (
        100.0 * (0.05 - (-0.1))
        + 50.0 * (0.15 - (-0.2))
        + 12.0
        + 0.01 * 30.0
    )

    assert v25.v38_projected_optimizer_support_score(
        **kwargs,
        compatible_mse_gate=float("inf"),
        compatible_gate_weight=0.0,
    ) == pytest.approx(expected)


def test_v38_optimizer_score_penalizes_compatible_mse_gate() -> None:
    low = v25.v38_projected_optimizer_support_score(
        support_target_margin=0.5,
        support_tournament_margin=0.4,
        support_compatible_mse=4.0,
        loss=10.0,
        target_margin_floor=0.05,
        tournament_margin_floor=0.15,
        compatible_mse_gate=5.0,
        compatible_gate_weight=1.5,
    )
    high = v25.v38_projected_optimizer_support_score(
        support_target_margin=0.5,
        support_tournament_margin=0.4,
        support_compatible_mse=12.0,
        loss=1.0,
        target_margin_floor=0.05,
        tournament_margin_floor=0.15,
        compatible_mse_gate=5.0,
        compatible_gate_weight=1.5,
    )

    assert high > low
```

- [ ] **Step 2: Run RED**

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v38_score_defaults_match_v37_formula or v38_optimizer_score_penalizes_compatible_mse_gate' -q
```

Expected: fail on missing scoring function.

- [ ] **Step 3: Implement scoring helper and optimizer hook**

Add:

```python
def v38_projected_optimizer_support_score(
    *,
    support_target_margin: float,
    support_tournament_margin: float,
    support_compatible_mse: float,
    loss: float,
    target_margin_floor: float,
    tournament_margin_floor: float,
    compatible_mse_gate: float,
    compatible_gate_weight: float,
) -> float:
    target_gap = max(0.0, float(target_margin_floor) - float(support_target_margin))
    tournament_gap = max(
        0.0,
        float(tournament_margin_floor) - float(support_tournament_margin),
    )
    compatible_gap = max(0.0, float(support_compatible_mse) - float(compatible_mse_gate))
    return float(
        100.0 * target_gap
        + 50.0 * tournament_gap
        + float(compatible_gate_weight) * compatible_gap * compatible_gap
        + float(support_compatible_mse)
        + 0.01 * float(loss)
    )
```

Update `solve_v37_projected_support_optimizer_edit` to read optional config keys:

```python
compatible_mse_gate = float(config.get("compatible_mse_gate", float("inf")))
compatible_gate_weight = float(config.get("compatible_gate_weight", 0.0))
projected_optimizer_event_prefix = str(
    config.get("projected_optimizer_event_prefix", "v37_projected_optimizer")
)
```

Use `v38_projected_optimizer_support_score(...)` instead of inline support score. With default `inf/0.0`, V37 behavior remains unchanged.

Use `projected_optimizer_event_prefix` for progress/completion event names:

```python
event=f"{projected_optimizer_event_prefix}_progress"
event=f"{projected_optimizer_event_prefix}_completed"
```

This preserves V37 event names by default and makes V38 logs distinguishable.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v38_score_defaults_match_v37_formula or v38_optimizer_score_penalizes_compatible_mse_gate or v37_projected_optimizer_final_loss_matches_returned_delta' -q
```

Expected: pass.

- [ ] **Step 5: Add event-prefix regression**

Add a small monkeypatched optimizer test that sets:

```python
"projected_optimizer_event_prefix": "v38_compatible_gated_optimizer"
```

and captures `record_progress_event` calls. Assert both
`v38_compatible_gated_optimizer_progress` and
`v38_compatible_gated_optimizer_completed` occur, and that the default V37 tests
still see `v37_projected_optimizer_progress`/`completed` behavior.

## Task 3: Add Compatible-Gated Alpha Selection And Matched Wrapper

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write RED alpha-selection test**

Add:

```python
def test_v38_alpha_redaction_keeps_gate_evidence_and_omits_raw_fields() -> None:
    event = v25.redact_v38_compatible_gated_alpha_progress_event({
        "alpha": 0.5,
        "alpha_candidate_count": 2,
        "alpha_candidates_hash": "a" * 64,
        "alpha_compatible_mse_gate": 5.0,
        "candidate_metrics_hash": "b" * 64,
        "compatible_gate_pass": True,
        "eligible_count": 1,
        "fallback_compatible_penalty": 2.0,
        "fallback_score": 4.0,
        "record_id_hash": "c" * 64,
        "selected_alpha_candidate_hash": "d" * 64,
        "selected_config_hash": "e" * 64,
        "selection_mode": "eligible_min_compatible_mse",
        "support_compatible_mse": 4.0,
        "support_target_margin": 0.4,
        "support_tournament_margin": 0.3,
        "compatible_jacobian": torch.ones(1, 2),
        "projected_delta": torch.ones(2),
        "subject_id": "raw",
        "weights": torch.ones(2),
    })
    text = json.dumps(event, sort_keys=True)

    assert event["compatible_gate_pass"] is True
    assert event["alpha_compatible_mse_gate"] == 5.0
    assert event["fallback_compatible_penalty"] == 2.0
    assert event["eligible_count"] == 1
    assert event["candidate_metrics_hash"] == "b" * 64
    for forbidden in ["compatible_jacobian", "projected_delta", "subject_id", "weights"]:
        assert forbidden not in event
    assert "raw" not in text


def test_v38_alpha_selection_requires_compatible_mse_gate() -> None:
    selected = v25.select_v38_compatible_gated_alpha_candidate(
        candidates=[
            {
                "alpha": 1.0,
                "support_compatible_mse": 20.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 1.0,
                "support_tournament_margin": 1.0,
            },
            {
                "alpha": 0.5,
                "support_compatible_mse": 4.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.4,
                "support_tournament_margin": 0.3,
            },
        ],
        alpha_target_margin_floor=0.05,
        alpha_tournament_margin_floor=0.0,
        alpha_compatible_mse_gate=5.0,
        fallback_target_penalty=10.0,
        fallback_tournament_penalty=5.0,
        fallback_compatible_penalty=2.0,
    )

    assert selected["alpha"] == 0.5
    assert selected["eligible"] is True
    assert selected["selection_mode"] == "eligible_min_compatible_mse"
```

- [ ] **Step 2: Run RED**

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v38_alpha_redaction_keeps_gate_evidence or v38_alpha_selection_requires_compatible_mse_gate' -q
```

Expected: fail on missing selector.

- [ ] **Step 3: Implement selector and wrapper**

Implement `select_v38_compatible_gated_alpha_candidate(...)` by extending `select_v35_support_source_alpha_candidate(...)` logic:

- eligible iff target margin passes, tournament margin passes, and `support_compatible_mse <= alpha_compatible_mse_gate`
- fallback score adds `fallback_compatible_penalty * max(0, support_compatible_mse - gate)`
- include `compatible_gate_pass`, `alpha_compatible_mse_gate`, `fallback_compatible_penalty`, `eligible_count`, and `candidate_metrics_hash`
- emit V38 alpha progress with `redact_v38_compatible_gated_alpha_progress_event(...)`, not the V35 redactor, so logs retain gate evidence

Add:

```python
def redact_v38_compatible_gated_alpha_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    redacted = redact_v35_support_source_alpha_progress_event(payload)
    for key in ["alpha_candidates_hash", "candidate_metrics_hash"]:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    if "compatible_gate_pass" in payload:
        redacted["compatible_gate_pass"] = bool(payload["compatible_gate_pass"])
    for key in ["eligible_count"]:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in ["alpha_compatible_mse_gate", "fallback_compatible_penalty"]:
        if key in payload:
            redacted[key] = float(payload[key])
    finite_values = [
        value for value in redacted.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted
```

Add `evaluate_v25_compatible_gated_projected_optimizer_matched_edit(...)` by delegating to the V37 wrapper shape, but:

- uses V38 matched source
- uses `select_v38_compatible_gated_alpha_candidate`
- metadata includes `compatible_mse_gate`, `compatible_gate_weight`, and `alpha_compatible_mse_gate`
- progress alpha event must use `redact_v38_compatible_gated_alpha_progress_event`

Add dispatcher branch for `V38_COMPATIBLE_GATED_MATCHED_EDIT_SOURCE`.

- [ ] **Step 4: Add dispatch test and run GREEN**

Add a V38 dispatch test mirroring V37 dispatch and assert V25-native controls.

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v38' -q
```

Expected: pass.

## Task 4: Verification, Review, And Bounded Run

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create after run: `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v38-compatible-mse-gated-projected-optimizer-results.md`

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v38 or v37_projected_optimizer_final_loss_matches_returned_delta or parse_args_accepts_all_inner_validation_grids' -q
```

Expected: pass.

- [ ] **Step 2: Run full helper suite**

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: pass.

- [ ] **Step 3: Run compile check**

```bash
python -m py_compile /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Expected: exit code 0.

- [ ] **Step 4: Request Kepler implementation review**

Proceed only after confidence `5/5`.

- [ ] **Step 5: Run bounded monitored V38 diagnostic**

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v38-compatible-mse-gated-projected-optimizer \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions \
  > /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v38_compatible_mse_gated_projected_optimizer_terminal.log 2>&1
```

- [ ] **Step 6: Post-run verification**

Run process check, row counts, `monitor_stop`, SHA256 hashes, strict raw-key leak scan, event counts, candidate table extraction, and result-doc creation. Send results to Kepler and accept only at `5/5`.
