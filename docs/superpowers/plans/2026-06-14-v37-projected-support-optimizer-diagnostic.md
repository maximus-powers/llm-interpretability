# V37 Projected Support Optimizer Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether target-aware support optimization inside a compatible-nullspace projection can recover target plasticity while retaining V36's source-preservation gains.

**Architecture:** V37 keeps V36's raw compatible-Jacobian preservation subspace, but does not post-hoc project a completed V32 sparse delta. Instead, it optimizes sparse coordinate parameters on support data while every objective evaluation uses the projected delta, so the target/tournament gradient sees the preservation constraint throughout optimization. Heldout proof remains evaluation-only.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing V25-V36 experiment runner.

---

## Literature Support

- [AlphaEdit](https://arxiv.org/abs/2410.02355) motivates null-space-constrained edits to preserve existing behavior, but V36 showed post-hoc projection can over-preserve in this small model.
- [Task Arithmetic in the Tangent Space](https://arxiv.org/abs/2305.12827) supports optimizing/editing in a linearized tangent regime where weight directions map more locally to function changes.
- [Model Merging by Output-Space Projection](https://arxiv.org/abs/2605.29101) frames residual updates as calibration-output projection/least-squares, supporting an output-space objective rather than raw weight-distance heuristics.
- [GNSP](https://arxiv.org/html/2507.19839v1) shows null-space projection has a stability/plasticity threshold tradeoff; V37 therefore tests lower projection strengths and looser null-space ranks instead of only maximal preservation.
- [Structure Is Not Enough](https://arxiv.org/html/2503.17138v1) argues behavioral losses are necessary for functional weight reconstruction/editing; V37 optimizes behavior probes directly.
- [Universal Weight Subspace Hypothesis](https://arxiv.org/abs/2512.05117) supports the idea that useful model edits may live in structured low-dimensional weight subspaces, but this diagnostic remains small-scale and does not claim large-model transfer.
- [Weight Space Learning Survey](https://arxiv.org/html/2603.10090v1) frames neural weights as a learnable data modality and motivates rigorous leakage-aware evaluation on model zoos.

## Prior Result Constraint

V36 is accepted as a 5/5-reviewed negative diagnostic:

- Best locality reached compatible-MSE failures `4/24`, but target prediction was `0.0`.
- Best selected config target prediction was only `0.125`, with negative mean target margin.
- Projection energy ratio mean was `0.12544`; preservation worked, plasticity collapsed.

V37 may claim success only if heldout proof gates pass. Otherwise it must be reported as diagnostic.

## Files

- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create after run: `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v37-projected-support-optimizer-diagnostic-results.md`

## Leak And Monitoring Boundaries

- Never read `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- V37 support optimization may use only support tensors and train-pool provenance hashes.
- Heldout proof data remains evaluation-only through the existing proof evaluator.
- Do not log raw weights, subject IDs, selected coordinate lists, logits, examples, gradients, compatible Jacobians, bases, or raw deltas.
- Allowed V37 progress fields: hashes, counts, scalar losses, scalar margins, scalar projection norms, scalar preservation energies, selected alpha metadata.
- Every long run must have:
  - `--monitor-interval-seconds 5`
  - `--summary-only-stdout`
  - progress log row count check
  - monitor log row count check
  - terminal `monitor_stop` check
  - PID check after completion
  - SHA256 hashes of progress and monitor logs

## Task 1: Add V37 Constants, Grid, Routing, And Redaction

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing tests for V37 grid and redaction**

Add these tests after the V36 grid/redaction tests:

```python
def test_v37_projected_optimizer_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v37_projected_support_optimizer_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V37_PROJECTED_OPTIMIZER_GRID_SHA256
    assert [item["config_index"] for item in grid] == list(range(4))
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    }
    assert sorted({item["compatible_nullspace_rtol"] for item in grid}) == [
        1e-3,
        1e-2,
    ]
    assert sorted({item["projection_strength"] for item in grid}) == [0.5, 0.75]
    assert all(item["projected_optimizer_epochs"] == 80 for item in grid)

    configs = v25.select_v25_inner_validation_configs(
        grid_name="v37-projected-support-optimizer",
        max_configs=None,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
    )

    assert all(config["train_pool_file_sha256"] == "a" * 64 for config in configs)
    assert all(config["train_pool_summary_hash"] == "b" * 64 for config in configs)
    assert v25.experiment_variant_for_inner_validation_grid(
        "v37-projected-support-optimizer"
    ) == v25.V37_EXPERIMENT_VARIANT


def test_v37_projected_optimizer_progress_redaction_omits_raw_values() -> None:
    event = v25.redact_v37_projected_optimizer_progress_event({
        "basis": torch.eye(2),
        "compatible_jacobian": torch.ones(1, 2),
        "final_subjects_path": str(v25.V25_FINAL_RAW),
        "gradient": torch.ones(2),
        "logits": torch.ones(2),
        "projected_delta": torch.ones(2),
        "raw_delta": torch.ones(2),
        "selected_coordinates": [1, 2],
        "sequence": [0, 1, 1, 0],
        "subject_id": "raw",
        "support_examples": [{"x": [1, 0], "y": 1}],
        "weights": torch.ones(2),
        "optimizer_audit_hash": "a" * 64,
        "record_id_hash": "b" * 64,
        "selected_config_hash": "c" * 64,
        "compatible_nullspace_rtol": 1e-3,
        "projection_strength": 0.5,
        "optimization_steps": 80,
        "preserve_rank": 3,
        "jacobian_row_count": 4,
        "final_loss": 0.25,
        "support_compatible_mse": 0.01,
        "support_target_margin": 0.2,
        "support_tournament_margin": 0.1,
        "preservation_energy_ratio": 0.4,
        "projected_delta_norm": 0.3,
    })
    text = json.dumps(event, sort_keys=True)

    assert event["optimization_steps"] == 80
    assert event["preserve_rank"] == 3
    assert event["finite"] is True
    for forbidden in [
        "basis",
        "compatible_jacobian",
        "final_subjects_path",
        "gradient",
        "logits",
        "projected_delta",
        "raw_delta",
        "selected_coordinates",
        "sequence",
        "subject_id",
        "support_examples",
        "weights",
    ]:
        assert forbidden not in event
    assert "final_subjects" not in text
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37_projected_optimizer_grid or v37_projected_optimizer_progress_redaction' -q
```

Expected: fail because V37 symbols/functions do not exist.

- [ ] **Step 3: Add constants, grid, routing, parser choice, and redaction**

Add constants near V36:

```python
V37_EXPERIMENT_VARIANT = "v37_projected_support_optimizer_diagnostic"
V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE = "projected_support_optimizer_sparse"
V37_PROJECTED_OPTIMIZER_GRID_SHA256 = (
    "e62e13a7ee50b407f9aa5364be2ddbf4dca9f60a385f4bbb0e36dd8a997243bf"
)
```

Add:

```python
def build_v37_projected_support_optimizer_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for compatible_nullspace_rtol in [1e-3, 1e-2]:
        for projection_strength in [0.5, 0.75]:
            grid.append({
                "alpha_candidates": [float(value) for value in V35_ALPHA_CANDIDATES],
                "alpha_target_margin_floor": 0.05,
                "alpha_tournament_margin_floor": 0.0,
                "compatible_floor": float(V32_COMPATIBLE_FLOOR),
                "compatible_nullspace_rtol": float(compatible_nullspace_rtol),
                "compatible_orthogonal_weight": 0.05,
                "config_index": len(grid),
                "extra_compatible_weight": 0.05,
                "fallback_target_penalty": 10.0,
                "fallback_tournament_penalty": 5.0,
                "hard_target_margin_weight": float(V32_HARD_TARGET_MARGIN_WEIGHT),
                "matched_edit_source": V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE,
                "projected_optimizer_epochs": 80,
                "projected_optimizer_lr": float(V29_BREADTH_FIRST_LR),
                "projection_strength": float(projection_strength),
                "sign_conflict_penalty": float(V32_SIGN_CONFLICT_PENALTY),
                "sparse_top_k": int(V32_SPARSE_TOP_K),
                "target_margin_floor": float(V32_TARGET_MARGIN_FLOOR),
                "tournament_margin_floor": 0.15,
                "tournament_margin_weight": 1.0,
                "trust_norm_cap": float(V32_TRUST_NORM_CAP),
            })
    return grid
```

Add selection/routing for `v37-projected-support-optimizer`, return `V37_EXPERIMENT_VARIANT`, map config source to `V37_EXPERIMENT_VARIANT`, use `V32_SUPPORT_TOURNAMENT_NATIVE_CONTROL_CONFIG` as the native control, and add parser choice.

Add redaction:

```python
def redact_v37_projected_optimizer_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_hash_keys = {
        "optimizer_audit_hash",
        "record_id_hash",
        "selected_config_hash",
    }
    allowed_int_keys = {
        "jacobian_row_count",
        "optimization_steps",
        "preserve_rank",
    }
    allowed_float_keys = {
        "compatible_nullspace_rtol",
        "final_loss",
        "preservation_energy_ratio",
        "projected_delta_norm",
        "projection_strength",
        "support_compatible_mse",
        "support_runner_margin",
        "support_target_margin",
        "support_tournament_margin",
    }
    redacted: dict[str, Any] = {}
    for key in allowed_hash_keys:
        if key in payload:
            redacted[key] = require_sha256_hex(payload[key], field_name=key)
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted
```

- [ ] **Step 4: Rerun Task 1 tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37_projected_optimizer_grid or v37_projected_optimizer_progress_redaction' -q
```

Expected: focused Task 1 tests pass with grid hash `e62e13a7ee50b407f9aa5364be2ddbf4dca9f60a385f4bbb0e36dd8a997243bf`.

## Task 2: Add Differentiable Projection And Projected Support Optimizer

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing tests for differentiable projection and optimizer behavior**

Add:

```python
def test_v37_differentiable_projection_removes_row_component_and_backprops() -> None:
    sparse_delta = torch.zeros(v25.SOURCE_WEIGHT_DIM, dtype=torch.float32)
    sparse_delta[0] = 1.0
    sparse_delta[1] = 2.0
    sparse_delta.requires_grad_(True)
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    projected, audit = v25.project_v37_delta_differentiably(
        sparse_delta=sparse_delta,
        compatible_jacobian=compatible_jacobian,
        compatible_nullspace_rtol=1e-4,
        projection_strength=1.0,
        trust_norm_cap=10.0,
    )
    loss = projected[1] ** 2
    loss.backward()

    assert torch.isclose(projected[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(projected[1], torch.tensor(2.0), atol=1e-6)
    assert sparse_delta.grad is not None
    assert torch.isclose(sparse_delta.grad[1], torch.tensor(4.0), atol=1e-6)
    assert audit["preserve_rank"] == 1
    assert audit["preservation_energy_ratio"] < 1e-5


def test_v37_projected_optimizer_optimizes_target_after_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    support = {
        "compatible_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "compatible_source_logits": torch.zeros(1, dtype=torch.float32),
        "conflict_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "conflict_target_labels": torch.ones(1, dtype=torch.float32),
        "target_inputs": torch.tensor([[1.0]], dtype=torch.float32),
        "target_labels": torch.ones(1, dtype=torch.float32),
    }
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian[0, 0] = 1.0

    monkeypatch.setattr(v25, "v27_support_tensors_for_source_target", lambda **_: support)
    monkeypatch.setattr(v25, "v32_support_behavior_margin_tensors", lambda: {
        "by_behavior": {},
        "counts_by_behavior": {},
        "tensor_hash": "e" * 64,
    })
    monkeypatch.setattr(
        v25,
        "v27_subject_logits_for_inputs",
        lambda flat, inputs: flat[1].expand(int(inputs.shape[0])),
    )

    def fake_margins(*, weights: torch.Tensor, tournament_tensors: Mapping[str, Any]):
        return {
            "has_majority": weights[1] - 0.30,
            "mountain_pattern": weights[1] - 0.20,
            "sorted_ascending": weights[1] - 0.10,
            "sorted_descending": weights[1],
        }

    monkeypatch.setattr(v25, "v32_support_behavior_margins", fake_margins)
    result = v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=torch.ones(v25.SOURCE_WEIGHT_DIM),
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config={
            **v25.build_v37_projected_support_optimizer_config_grid()[0],
            "projected_optimizer_epochs": 30,
            "projected_optimizer_lr": 0.2,
            "trust_norm_cap": 1.0,
        },
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    assert result["delta"][0].abs() < 1e-5
    assert result["delta"][1] > 0.2
    assert result["audit"]["support_target_margin"] > 0.0
    assert result["audit"]["preservation_energy_ratio"] < 1e-5
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37_differentiable_projection or v37_projected_optimizer_optimizes_target' -q
```

Expected: fail because V37 projection/optimizer functions do not exist.

- [ ] **Step 3: Implement differentiable projection**

Add:

```python
def project_v37_delta_differentiably(
    *,
    sparse_delta: torch.Tensor,
    compatible_jacobian: torch.Tensor,
    compatible_nullspace_rtol: float,
    projection_strength: float,
    trust_norm_cap: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    delta = sparse_delta.to(dtype=torch.float32).reshape(-1)
    if int(delta.numel()) != SOURCE_WEIGHT_DIM:
        raise ValueError("sparse_delta has wrong dimension")
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite sparse_delta")
    jacobian = compatible_jacobian.detach().clone().to(dtype=torch.float32)
    if jacobian.ndim != 2 or int(jacobian.shape[1]) != SOURCE_WEIGHT_DIM:
        raise ValueError("compatible_jacobian has wrong shape")
    if not torch.isfinite(jacobian).all():
        raise ValueError("nonfinite compatible_jacobian")
    rtol = float(compatible_nullspace_rtol)
    strength = float(projection_strength)
    norm_cap = float(trust_norm_cap)
    for name, value in [
        ("compatible_nullspace_rtol", rtol),
        ("projection_strength", strength),
        ("trust_norm_cap", norm_cap),
    ]:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    base_energy_tensor = torch.linalg.norm(jacobian @ delta)
    if int(jacobian.shape[0]) == 0:
        preserve_rank = 0
        row_component = torch.zeros_like(delta)
    else:
        _u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
        if not torch.isfinite(singular_values).all() or not torch.isfinite(vh).all():
            raise ValueError("nonfinite compatible nullspace svd")
        s_max = float(torch.max(singular_values).item()) if int(singular_values.numel()) else 0.0
        normalized = singular_values / max(s_max, 1e-12)
        preserve_mask = normalized > rtol
        preserve_rank = int(torch.count_nonzero(preserve_mask).item())
        if preserve_rank > 0:
            preserve = vh[preserve_mask].T.to(dtype=torch.float32)
            row_component = preserve @ (preserve.T @ delta)
        else:
            row_component = torch.zeros_like(delta)
    projected = delta - float(strength) * row_component
    norm = torch.linalg.norm(projected)
    scale = torch.clamp(torch.tensor(norm_cap, dtype=torch.float32) / torch.clamp(norm, min=1e-12), max=1.0)
    projected = projected * scale
    projected_energy_tensor = torch.linalg.norm(jacobian @ projected)
    audit = {
        "base_preservation_energy": float(base_energy_tensor.detach().item()),
        "compatible_nullspace_rtol": rtol,
        "finite": True,
        "jacobian_row_count": int(jacobian.shape[0]),
        "preservation_energy_ratio": float(
            (projected_energy_tensor / torch.clamp(base_energy_tensor, min=1e-12)).detach().item()
        ),
        "preserve_rank": int(preserve_rank),
        "projected_preservation_energy": float(projected_energy_tensor.detach().item()),
        "projected_delta_norm": float(torch.linalg.norm(projected).detach().item()),
        "projection_strength": strength,
    }
    finite_failures = recursive_numeric_finiteness_failures(audit)
    if finite_failures:
        raise ValueError("nonfinite V37 projection audit: " + ", ".join(finite_failures[:5]))
    return projected.to(dtype=torch.float32), audit
```

- [ ] **Step 4: Implement projected support optimizer**

Add `solve_v37_projected_support_optimizer_edit(...)` mirroring `solve_v32_support_tournament_sparse_edit`, with these differences:

- Parameters include `compatible_jacobian`.
- `delta = project_v37_delta_differentiably(...)[0]` before computing edited weights.
- Optimizer variables remain only selected sparse coordinates.
- Loss uses target BCE, target margin hinge, conflict BCE, support-compatible MSE, compatible-gradient orthogonal loss, tournament hinge, and delta L2.
- Best checkpoint ranking must use this explicit scalar, minimized at each epoch:

```python
target_gap = max(0.0, float(target_margin_floor) - scalar_losses["support_target_margin"])
tournament_gap = max(
    0.0,
    float(tournament_margin_floor) - scalar_losses["support_tournament_margin"],
)
support_score = (
    100.0 * target_gap
    + 50.0 * tournament_gap
    + scalar_losses["support_compatible_mse"]
    + 0.01 * scalar_losses["loss"]
)
```

The stored best tuple must be:

```python
best = (
    support_score,
    scalar_losses["support_compatible_mse"],
    epoch,
    current_delta.detach().clone(),
    dict(scalar_losses),
    dict(projection_audit),
)
```

This makes plasticity gates primary while still preferring source preservation among equally target-capable projected deltas.
- Audit includes only scalar redacted fields plus `coordinate_hash`, `support_tournament_tensor_hash`, and `optimizer_audit_hash`.
- If `progress_log_path` is provided, write `v37_projected_optimizer_progress` every `max(1, projected_optimizer_epochs // 5)` epochs and at the final epoch, using `redact_v37_projected_optimizer_progress_event(...)`.
- If `progress_log_path` is provided, write `v37_projected_optimizer_completed` once after best-delta selection, using the same redactor.

The per-epoch progress event payload must be scalar/hash-only:

```python
progress_stride = max(1, int(projected_optimizer_epochs) // 5)
if progress_log_path is not None and (
    epoch % progress_stride == 0 or epoch == int(projected_optimizer_epochs)
):
    progress_extra = redact_v37_projected_optimizer_progress_event({
        **scalar_losses,
        **projection_audit,
        **({"record_id_hash": record_id_hash} if record_id_hash else {}),
        "final_loss": scalar_losses["loss"],
        "optimizer_audit_hash": stable_hash_json({
            "epoch": int(epoch),
            "projection_audit": projection_audit,
            "scalar_losses": scalar_losses,
        }),
        "optimization_steps": int(epoch),
        "selected_config_hash": selected_config_hash,
    })
    record_progress_event(
        progress_log_path,
        event="v37_projected_optimizer_progress",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra=progress_extra,
    )
```

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37_differentiable_projection or v37_projected_optimizer_optimizes_target' -q
```

Expected: pass.

## Task 3: Add V37 Matched Edit Dispatch And Proof Integration

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing dispatch test**

Add:

```python
def test_v37_dispatch_uses_projected_optimizer_and_v25_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_matched(**kwargs: Any) -> dict[str, Any]:
        called["matched_config"] = dict(kwargs["config"])
        return {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.1,
            "editor": {
                "delta_sha256": "d" * 64,
                "matched_edit_source": v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE,
                "matched_spectral_projection_norm": 0.0,
            },
            "predicted_behavior": "sorted_descending",
            "source_margin": 0.1,
            "target_margin": 0.5,
        }

    monkeypatch.setattr(v25, "compute_jacobian_cache_entry", lambda *_args, **_kwargs: {
        "cache_key": "cache",
        "source_logit_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
    })
    monkeypatch.setattr(
        v25,
        "evaluate_v25_projected_support_optimizer_matched_edit",
        fake_matched,
    )
    monkeypatch.setattr(
        v25,
        "build_v25_native_controls",
        lambda **kwargs: called.setdefault("control_config", dict(kwargs["config"])) or [],
    )
    monkeypatch.setattr(v25, "build_v25_proof_record", lambda **kwargs: {
        "controls": [],
        "matched": dict(kwargs["matched"]),
        "summary": {
            "individual_all_gates_passed": True,
            "pareto_undominated": True,
            "target_prediction_pass": True,
        },
    })

    proof = v25.evaluate_v25_development_job(
        job={
            "record_id": "dev-1",
            "source_behavior": "sorted_ascending",
            "subject": {
                "pattern": "sorted_ascending",
                "subject_id": "hidden",
                "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
            },
            "target_behavior": "sorted_descending",
        },
        train_stats={},
        config={
            **v25.build_v37_projected_support_optimizer_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
        control_context={"context_hash": "e" * 64},
        selected_config_hash="c" * 64,
        script_sha256="script",
    )

    assert called["matched_config"]["matched_edit_source"] == (
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    )
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE
    )
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
```

- [ ] **Step 2: Run RED dispatch test**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37_dispatch_uses_projected_optimizer' -q
```

Expected: fail because matched edit dispatch is missing.

- [ ] **Step 3: Implement matched edit wrapper and dispatcher branch**

Add `evaluate_v25_projected_support_optimizer_matched_edit(...)` analogous to V36:

- Validate train-pool hashes.
- Compute source weights.
- Compute `v28_anchor_gradients_and_compatible_jacobian(...)`.
- Select V31 sign-coherent sparse coordinates.
- Call `solve_v37_projected_support_optimizer_edit(...)`.
- Run V35 support-source alpha selection over the optimized projected delta.
- Return `control_record_for_delta(...)`.
- Metadata must include:
  - `matched_edit_source`
  - `optimization_boundary`
  - `optimization_split="support"`
  - `proof_split="heldout"`
  - `support_objective_is_proof_metric=False`
  - `optimizer_audit`
  - `optimizer_audit_hash`
  - `selected_alpha`
  - `candidate_metrics_hash`
  - train-pool hashes
  - script hash

The metadata dictionary must set:

```python
metadata = {
    "alpha_candidate_count": len(candidates),
    "alpha_candidates_hash": alpha_candidates_hash,
    "alpha_selection": alpha_selection,
    "candidate_metrics_hash": selected_alpha["candidate_metrics_hash"],
    "eligible_count": int(selected_alpha["eligible_count"]),
    "matched_edit_source": V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE,
    "matched_spectral_projection_norm": 0.0,
    "optimization_boundary": v35_support_source_line_search_optimization_boundary(),
    "optimization_split": "support",
    "optimizer_audit": optimizer_audit,
    "optimizer_audit_hash": optimizer_audit_hash,
    "proof_split": "heldout",
    "projected_optimizer_provenance_hash": "",
    "script_sha256": str(script_sha256),
    "selected_alpha": float(selected_alpha["alpha"]),
    "selected_alpha_candidate_hash": selected_alpha_candidate_hash,
    "selected_config_hash": selected_hash,
    "source_behavior": str(source_behavior),
    "support_objective_is_proof_metric": False,
    "support_split_counts": dict(gradient_info["support_split_counts"]),
    "target_behavior": str(target_behavior),
    "train_pool_file_sha256": str(train_pool_file_sha256),
    "train_pool_summary_hash": str(train_pool_summary_hash),
}
metadata["projected_optimizer_provenance_hash"] = stable_hash_json(metadata)
```

No raw selected coordinates, raw Jacobian, raw delta, logits, weights, examples, or subject IDs may be added to metadata.

Add dispatcher branch for `V37_PROJECTED_OPTIMIZER_MATCHED_EDIT_SOURCE` requiring train-pool hashes.

- [ ] **Step 4: Run dispatch GREEN test**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37_dispatch_uses_projected_optimizer' -q
```

Expected: pass.

## Task 4: Verification, Review, And Bounded Run

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create after run: `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v37-projected-support-optimizer-diagnostic-results.md`

- [ ] **Step 1: Run focused V37 tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v37 or parse_args_accepts_all_inner_validation_grids' -q
```

Expected: pass.

- [ ] **Step 2: Run full helper tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: pass.

- [ ] **Step 3: Run py_compile**

Run:

```bash
python -m py_compile /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Expected: exit code 0.

- [ ] **Step 4: Send implementation to Kepler**

Ask Kepler to review:

- data leak boundaries
- V37 projection math
- V37 optimizer objective
- progress redaction
- tests and verification
- bounded compute plan

Proceed only after confidence `5/5`.

- [ ] **Step 5: Run bounded monitored V37 development diagnostic**

Run:

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v37-projected-support-optimizer \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

Expected: bounded run exits cleanly. It may pass or fail; interpretation depends on heldout proof metrics.

- [ ] **Step 6: Verify logs and leak scan**

Run:

```bash
pgrep -fl 'train_four_behavior_functional_weight_editing_v25|python.*muat'
wc -l /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl
wc -l /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
tail -5 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
shasum -a 256 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl
shasum -a 256 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
rg -n 'final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence|compatible_jacobian|raw_delta|projected_delta' /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
```

Expected:

- no training PID remains after completion
- progress and monitor row counts are nonzero
- monitor includes `monitor_stop`
- hashes are recorded
- raw-field leak scan returns no matches

- [ ] **Step 7: Create result doc and request results review**

Create `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v37-projected-support-optimizer-diagnostic-results.md` with:

- command
- process/log monitoring evidence
- progress/monitor hashes
- candidate table
- V37 optimizer event counts
- alpha selection mode counts
- leak scan result
- proof pass/fail status
- conservative interpretation
- non-claims

Send result doc to Kepler and proceed only after confidence `5/5`.

## Reviewer Checklist

- Does V37 keep heldout proof data out of optimization?
- Does redaction omit raw model and example fields?
- Is `spectral_basis_sha256` treated as a hash-only provenance field if present?
- Is the optimizer actually target-aware after projection, or just post-hoc projection again?
- Is the grid bounded to four configs?
- Does the plan avoid sealed-final data?
- Are long-running commands monitored and hash-verifiable?
- Are negative results explicitly allowed and conservatively interpreted?
