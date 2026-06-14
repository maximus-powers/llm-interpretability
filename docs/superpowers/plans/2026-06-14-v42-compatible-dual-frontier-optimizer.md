# V42 Compatible Dual Frontier Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether compatible-MSE locality can be preserved by enforcing it during sparse projected optimization rather than only selecting for it after optimization.

**Architecture:** V42 extends the V41 trajectory-frontier optimizer with an inequality-style compatible-MSE constraint. The optimizer adds a nonnegative dual variable and squared residual penalty for `support_compatible_mse - compatible_mse_budget`, logs the constraint state at each trajectory checkpoint, and selects frontier candidates by target feasibility plus compatible-budget feasibility.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing JSONL progress/monitor logging in `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`.

---

## Research Basis

V41 recovered target behavior but failed compatible-MSE on every proof record. The literature points to pushing preservation into the optimizer:

- AlphaEdit projects edits into a null space of preserved knowledge and reports that this protects preserved outputs in locating-then-editing methods: https://arxiv.org/abs/2410.02355
- Deep Augmented Lagrangian work argues that fixed feasibility penalties are weaker than dual/augmented methods for constrained optimization and motivates residual penalties plus dual updates: https://arxiv.org/html/2403.03454v1
- Parametric Pareto Set Learning supports retaining a frontier of tradeoff candidates instead of collapsing to one scalar objective too early: https://arxiv.org/html/2511.05815v1
- Representation engineering surveys warn that linear steering can entangle features and break locality, so locality must be measured and constrained rather than inferred from target success: https://arxiv.org/html/2502.17601v1
- Steer2Edit reports better behavior/utility tradeoffs by moving from global activation shifts to component-level rank-1 edits, supporting V42's focus on constrained sparse component edits: https://arxiv.org/html/2602.09870v2
- Weight-space learning and neural functional work supports treating model weights, gradients, and edit structures as first-class data, which matches MUAT's fixed-probe weight/signature framing: https://arxiv.org/html/2510.02096v1 and https://arxiv.org/abs/2410.04209
- Universal Weight Subspace Hypothesis argues trained networks exploit shared low-dimensional spectral subspaces, supporting the search for compact edit directions while not guaranteeing locality: https://arxiv.org/abs/2512.05117

## Files

- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create after run: `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v42-compatible-dual-frontier-results.md`

## Acceptance Criteria

- V42 must never read `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- V42 run command must redirect stdout/stderr to a terminal log and use existing structured progress and monitor JSONL logs.
- The progress log must include redacted V42 fields for compatible budget, residual, dual lambda, constraint feasibility, and selection mode.
- The selector must report whether any candidate is both target-feasible and compatible-budget-feasible.
- A negative result is acceptable if it is bounded, leak-free, logged, and interpreted conservatively.
- Do not run lint.

## Task 1: Add V42 Constraint Helpers and Frontier Selection

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing helper tests**

Add these tests near the V41 selection tests:

```python
def test_v42_compatible_constraint_penalty_zero_below_budget() -> None:
    result = v25.v42_compatible_constraint_terms(
        support_compatible_mse=torch.tensor(8.0),
        compatible_mse_budget=10.0,
        compatible_dual_lambda=3.0,
        compatible_augmented_weight=2.0,
    )

    assert result["compatible_constraint_residual"].item() == pytest.approx(0.0)
    assert result["compatible_constraint_penalty"].item() == pytest.approx(0.0)


def test_v42_compatible_constraint_penalty_positive_above_budget() -> None:
    result = v25.v42_compatible_constraint_terms(
        support_compatible_mse=torch.tensor(13.0),
        compatible_mse_budget=10.0,
        compatible_dual_lambda=2.0,
        compatible_augmented_weight=0.5,
    )

    assert result["compatible_constraint_residual"].item() == pytest.approx(3.0)
    assert result["compatible_constraint_penalty"].item() == pytest.approx(10.5)
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v42_compatible_constraint_penalty_zero_below_budget /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v42_compatible_constraint_penalty_positive_above_budget -q
```

Expected: both fail because `v42_compatible_constraint_terms` is not defined.

- [ ] **Step 3: Implement constraint helper**

Add near `v38_projected_optimizer_support_score`:

```python
def v42_compatible_constraint_terms(
    *,
    support_compatible_mse: torch.Tensor,
    compatible_mse_budget: float,
    compatible_dual_lambda: float,
    compatible_augmented_weight: float,
) -> dict[str, torch.Tensor]:
    budget = float(compatible_mse_budget)
    dual_lambda = float(compatible_dual_lambda)
    augmented_weight = float(compatible_augmented_weight)
    for name, value in [
        ("compatible_mse_budget", budget),
        ("compatible_dual_lambda", dual_lambda),
        ("compatible_augmented_weight", augmented_weight),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if budget < 0.0 or dual_lambda < 0.0 or augmented_weight < 0.0:
        raise ValueError("compatible constraint parameters must be nonnegative")
    residual = torch.clamp(
        support_compatible_mse - support_compatible_mse.new_tensor(budget),
        min=0.0,
    )
    penalty = dual_lambda * residual + augmented_weight * residual.pow(2)
    return {
        "compatible_constraint_penalty": penalty,
        "compatible_constraint_residual": residual,
    }
```

- [ ] **Step 4: Add failing selector tests**

Add:

```python
def test_v42_frontier_selection_prefers_target_and_compatible_feasible() -> None:
    selected = v25.select_v42_compatible_dual_frontier_candidate(
        candidates=[
            {
                "epoch": 4,
                "loss": 20.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 1.0,
                "support_compatible_mse": 30.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.20,
                "support_tournament_margin": 0.10,
                "compatible_constraint_residual": 20.0,
                "compatible_dual_lambda": 4.0,
            },
            {
                "epoch": 8,
                "loss": 25.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 0.8,
                "support_compatible_mse": 8.0,
                "support_runner_margin": 0.0,
                "support_target_margin": 0.08,
                "support_tournament_margin": 0.04,
                "compatible_constraint_residual": 0.0,
                "compatible_dual_lambda": 4.0,
            },
        ],
        compatible_mse_budget=10.0,
        target_margin_floor=0.05,
        target_rank_score_tolerance=0.05,
        tournament_margin_floor=0.0,
    )

    assert selected["epoch"] == 8
    assert selected["compatible_constraint_feasible"] is True
    assert selected["target_feasible"] is True
    assert selected["selection_mode"] == "frontier_target_and_compatible_feasible"


def test_v42_frontier_selection_reports_no_localized_candidate_when_budget_fails() -> None:
    selected = v25.select_v42_compatible_dual_frontier_candidate(
        candidates=[
            {
                "epoch": 5,
                "loss": 20.0,
                "preservation_energy_ratio": 0.5,
                "projected_delta_norm": 0.9,
                "support_compatible_mse": 12.0,
                "support_runner_margin": -0.1,
                "support_target_margin": 0.10,
                "support_tournament_margin": 0.10,
                "compatible_constraint_residual": 2.0,
                "compatible_dual_lambda": 6.0,
            },
        ],
        compatible_mse_budget=10.0,
        target_margin_floor=0.05,
        target_rank_score_tolerance=0.05,
        tournament_margin_floor=0.0,
    )

    assert selected["epoch"] == 5
    assert selected["target_feasible"] is True
    assert selected["compatible_constraint_feasible"] is False
    assert selected["localized_feasible_count"] == 0
    assert selected["selection_mode"] == "frontier_target_feasible_min_compatible_residual"
```

- [ ] **Step 5: Implement selector**

Add after `select_v41_trajectory_frontier_candidate`:

```python
def select_v42_compatible_dual_frontier_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    compatible_mse_budget: float,
    target_margin_floor: float,
    target_rank_score_tolerance: float,
    tournament_margin_floor: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("V42 frontier candidates must be nonempty")
    budget = float(compatible_mse_budget)
    target_floor = float(target_margin_floor)
    tolerance = float(target_rank_score_tolerance)
    tournament_floor = float(tournament_margin_floor)
    for name, value in [
        ("compatible_mse_budget", budget),
        ("target_margin_floor", target_floor),
        ("target_rank_score_tolerance", tolerance),
        ("tournament_margin_floor", tournament_floor),
    ]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if budget < 0.0 or tolerance < 0.0:
        raise ValueError("compatible budget and target tolerance must be nonnegative")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = {
            "candidate_index": int(index),
            "epoch": int(candidate["epoch"]),
            "loss": float(candidate["loss"]),
            "preservation_energy_ratio": float(candidate["preservation_energy_ratio"]),
            "projected_delta_norm": float(candidate["projected_delta_norm"]),
            "support_compatible_mse": float(candidate["support_compatible_mse"]),
            "support_runner_margin": float(candidate.get("support_runner_margin", 0.0)),
            "support_target_margin": float(candidate["support_target_margin"]),
            "support_tournament_margin": float(candidate["support_tournament_margin"]),
            "compatible_constraint_residual": float(
                candidate["compatible_constraint_residual"]
            ),
            "compatible_dual_lambda": float(candidate["compatible_dual_lambda"]),
            "compatible_mse_budget": budget,
            "target_rank_score_tolerance": tolerance,
        }
        if not all(math.isfinite(float(value)) for value in item.values()):
            raise ValueError("nonfinite V42 frontier candidate")
        if item["epoch"] <= 0:
            raise ValueError("V42 frontier candidate epoch must be positive")
        item["target_gap"] = max(0.0, target_floor - item["support_target_margin"])
        item["tournament_gap"] = max(
            0.0,
            tournament_floor - item["support_tournament_margin"],
        )
        item["compatible_gap"] = max(
            0.0,
            item["support_compatible_mse"] - budget,
        )
        item["target_rank_score"] = item["target_gap"] + item["tournament_gap"]
        item["target_feasible"] = bool(item["target_rank_score"] == 0.0)
        item["compatible_constraint_feasible"] = bool(item["compatible_gap"] == 0.0)
        normalized.append(item)

    best_target_rank_score = min(item["target_rank_score"] for item in normalized)
    localized = [
        item for item in normalized
        if item["target_feasible"] and item["compatible_constraint_feasible"]
    ]
    target_feasible = [item for item in normalized if item["target_feasible"]]
    if localized:
        pool = localized
        selection_mode = "frontier_target_and_compatible_feasible"
    elif target_feasible:
        pool = target_feasible
        selection_mode = "frontier_target_feasible_min_compatible_residual"
    else:
        pool = [
            item for item in normalized
            if item["target_rank_score"] <= best_target_rank_score + tolerance
        ]
        selection_mode = "frontier_target_tolerance_min_compatible_residual"

    selected = min(
        pool,
        key=lambda item: (
            item["compatible_gap"],
            item["support_compatible_mse"],
            item["target_rank_score"],
            -item["support_tournament_margin"],
            -item["support_target_margin"],
            item["projected_delta_norm"],
            item["epoch"],
        ),
    )
    candidate_metrics_hash = stable_hash_json([
        {
            "epoch": item["epoch"],
            "compatible_constraint_feasible": item["compatible_constraint_feasible"],
            "compatible_constraint_residual": item["compatible_constraint_residual"],
            "compatible_dual_lambda": item["compatible_dual_lambda"],
            "compatible_gap": item["compatible_gap"],
            "compatible_mse_budget": item["compatible_mse_budget"],
            "loss": item["loss"],
            "preservation_energy_ratio": item["preservation_energy_ratio"],
            "projected_delta_norm": item["projected_delta_norm"],
            "support_compatible_mse": item["support_compatible_mse"],
            "support_runner_margin": item["support_runner_margin"],
            "support_target_margin": item["support_target_margin"],
            "support_tournament_margin": item["support_tournament_margin"],
            "target_feasible": item["target_feasible"],
            "target_gap": item["target_gap"],
            "target_rank_score": item["target_rank_score"],
            "target_rank_score_tolerance": item["target_rank_score_tolerance"],
            "tournament_gap": item["tournament_gap"],
        }
        for item in normalized
    ])
    result = dict(selected)
    result["best_target_rank_score"] = float(best_target_rank_score)
    result["candidate_metrics_hash"] = candidate_metrics_hash
    result["frontier_candidate_count"] = len(normalized)
    result["localized_feasible_count"] = len(localized)
    result["selection_mode"] = selection_mode
    result["within_target_tolerance_count"] = len(pool)
    return result
```

- [ ] **Step 6: Verify helper tests pass**

Run:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v42_compatible_constraint_penalty_zero_below_budget /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v42_compatible_constraint_penalty_positive_above_budget /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v42_frontier_selection_prefers_target_and_compatible_feasible /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v42_frontier_selection_reports_no_localized_candidate_when_budget_fails -q
```

Expected: `4 passed`.

## Task 2: Wire V42 into the Optimizer Loop and Logs

**Files:**
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Add V42 constants and config grid**

Add constants near V41:

```python
V42_EXPERIMENT_VARIANT = "v42_compatible_dual_frontier_projected_optimizer_diagnostic"
V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE = (
    "compatible_dual_frontier_projected_optimizer_sparse"
)
```

Add grid builder after `build_v41_trajectory_frontier_config_grid`:

```python
def build_v42_compatible_dual_frontier_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base = build_v41_trajectory_frontier_config_grid()[0]
    for compatible_mse_budget in [10.0, 20.0]:
        for compatible_augmented_weight in [0.5, 2.0]:
            grid.append({
                **dict(base),
                "alpha_compatible_mse_soft_gate": float(compatible_mse_budget),
                "compatible_augmented_weight": float(compatible_augmented_weight),
                "compatible_dual_initial": 0.0,
                "compatible_dual_lr": 0.05,
                "compatible_dual_max": 100.0,
                "compatible_gate_weight": 0.0,
                "compatible_mse_budget": float(compatible_mse_budget),
                "compatible_mse_gate": float(compatible_mse_budget),
                "config_index": len(grid),
                "experiment_variant": V42_EXPERIMENT_VARIANT,
                "matched_edit_source": V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE,
                "projected_optimizer_event_prefix": "v42_compatible_dual_frontier_optimizer",
                "target_rank_score_tolerance": 0.15,
                "trajectory_frontier_enabled": True,
                "trajectory_frontier_event_prefix": "v42_compatible_dual_frontier",
                "v42_compatible_dual_enabled": True,
            })
    return grid
```

Update `select_v25_inner_validation_configs` to accept `v42-compatible-dual-frontier`.

- [ ] **Step 2: Add V42 routing and provenance tests**

Add these tests near the V41 grid and variant tests:

```python
def test_v42_compatible_dual_frontier_grid_is_bounded_and_binds_provenance() -> None:
    grid = v25.build_v42_compatible_dual_frontier_config_grid()

    assert len(grid) == 4
    assert v25.stable_hash_json(grid) == v25.V42_GRID_SHA256
    assert {item["matched_edit_source"] for item in grid} == {
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE,
    }
    assert {item["experiment_variant"] for item in grid} == {
        v25.V42_EXPERIMENT_VARIANT,
    }
    assert {item["projected_optimizer_event_prefix"] for item in grid} == {
        "v42_compatible_dual_frontier_optimizer",
    }
    assert {item["trajectory_frontier_event_prefix"] for item in grid} == {
        "v42_compatible_dual_frontier",
    }
    assert all(item["v42_compatible_dual_enabled"] is True for item in grid)
    assert all(item["trajectory_frontier_enabled"] is True for item in grid)
    assert all(item["projected_optimizer_epochs"] == 80 for item in grid)
    assert {item["compatible_mse_budget"] for item in grid} == {10.0, 20.0}
    assert {item["compatible_augmented_weight"] for item in grid} == {0.5, 2.0}


def test_v42_variant_routing_and_native_controls_are_explicit(monkeypatch) -> None:
    config = {
        **v25.build_v42_compatible_dual_frontier_config_grid()[0],
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }

    assert v25.experiment_variant_for_inner_validation_grid(
        "v42-compatible-dual-frontier"
    ) == v25.V42_EXPERIMENT_VARIANT
    assert v25.experiment_variant_for_config(config) == v25.V42_EXPERIMENT_VARIANT
    assert v25.v25_config_requires_spectral_basis(config) is False
    assert v25.v25_native_control_config(config) == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--phase",
            "development",
            "--run-inner-validation",
            "--inner-validation-config-grid",
            "v42-compatible-dual-frontier",
        ],
    )
    args = v25.parse_args()
    assert args.inner_validation_config_grid == "v42-compatible-dual-frontier"
```

After the grid implementation exists, compute the stable grid hash with:

```bash
python - <<'PY'
import importlib.util
path = "/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py"
spec = importlib.util.spec_from_file_location("v25", path)
v25 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v25)
print(v25.stable_hash_json(v25.build_v42_compatible_dual_frontier_config_grid()))
PY
```

Add `V42_GRID_SHA256` near the previous grid hash constants using the printed hash. Also update:

- `select_v25_inner_validation_configs` for `v42-compatible-dual-frontier`.
- `experiment_variant_for_inner_validation_grid`.
- `experiment_variant_for_config`.
- `v25_config_requires_spectral_basis`.
- `v25_native_control_config`.
- `parse_args` `--inner-validation-config-grid` choices.

- [ ] **Step 3: Add redaction test**

Add:

```python
def test_v42_frontier_redaction_omits_raw_fields() -> None:
    forbidden = {
        "basis": [1.0],
        "compatible_jacobian": [1.0],
        "final_subjects": ["sealed"],
        "final_subjects_path": "/sealed/final_subjects.json",
        "gradient": [1.0],
        "logits": [1.0],
        "projected_delta": [1.0],
        "raw_delta": [3.0],
        "selected_coordinates": [0],
        "sequence": [1, 0, 1],
        "subject_id": "raw-subject",
        "support_examples": [[1, 0, 1]],
        "weights": [1.0, 2.0],
    }
    redacted = v25.redact_v42_compatible_dual_frontier_progress_event({
        **forbidden,
        "candidate_index": 1,
        "compatible_constraint_feasible": True,
        "compatible_constraint_residual": 0.0,
        "compatible_dual_lambda": 2.5,
        "compatible_gap": 0.0,
        "compatible_mse_budget": 10.0,
        "frontier_candidate_count": 3,
        "localized_feasible_count": 1,
        "loss": 4.0,
        "record_id_hash": "a" * 64,
        "selected_config_hash": "b" * 64,
        "selection_mode": "frontier_target_and_compatible_feasible",
        "support_compatible_mse": 8.0,
        "support_target_margin": 0.1,
        "support_tournament_margin": 0.2,
        "target_feasible": True,
        "target_rank_score": 0.0,
        "trajectory_frontier_selected_epoch": 5,
    })

    assert redacted["compatible_constraint_feasible"] is True
    assert redacted["selection_mode"] == "frontier_target_and_compatible_feasible"
    for key in forbidden:
        assert key not in redacted
```

- [ ] **Step 4: Implement V42 redaction**

Add after `redact_v41_trajectory_frontier_progress_event`:

```python
def redact_v42_compatible_dual_frontier_progress_event(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    redacted = redact_v41_trajectory_frontier_progress_event(payload)
    allowed_float_keys = {
        "compatible_constraint_residual",
        "compatible_dual_lambda",
        "compatible_gap",
        "compatible_mse_budget",
    }
    allowed_int_keys = {
        "localized_feasible_count",
    }
    allowed_bool_keys = {
        "compatible_constraint_feasible",
    }
    allowed_selection_modes = {
        "frontier_target_and_compatible_feasible",
        "frontier_target_feasible_min_compatible_residual",
        "frontier_target_tolerance_min_compatible_residual",
    }
    for key in allowed_float_keys:
        if key in payload:
            redacted[key] = float(payload[key])
    for key in allowed_int_keys:
        if key in payload:
            redacted[key] = int(payload[key])
    for key in allowed_bool_keys:
        if key in payload:
            redacted[key] = bool(payload[key])
    if "selection_mode" in payload:
        mode = str(payload["selection_mode"])
        if mode not in allowed_selection_modes:
            raise ValueError(f"unknown V42 frontier selection mode: {mode}")
        redacted["selection_mode"] = mode
    finite_values = [
        value for key, value in redacted.items()
        if key in allowed_float_keys
        or key in {
            "best_target_rank_score",
            "compatible_gap",
            "loss",
            "preservation_energy_ratio",
            "projected_delta_norm",
            "support_compatible_mse",
            "support_runner_margin",
            "support_target_margin",
            "support_tournament_margin",
            "target_gap",
            "target_rank_score",
            "target_rank_score_tolerance",
            "tournament_gap",
        }
    ]
    redacted["finite"] = all(math.isfinite(float(value)) for value in finite_values)
    return redacted
```

- [ ] **Step 5: Modify optimizer loop**

Inside `solve_v37_projected_support_optimizer_edit`, read these config keys:

```python
v42_compatible_dual_enabled = bool(config.get("v42_compatible_dual_enabled", False))
compatible_mse_budget = float(config.get("compatible_mse_budget", compatible_mse_gate))
compatible_augmented_weight = float(config.get("compatible_augmented_weight", 0.0))
compatible_dual_lr = float(config.get("compatible_dual_lr", 0.0))
compatible_dual_max = float(config.get("compatible_dual_max", 0.0))
compatible_dual_lambda = float(config.get("compatible_dual_initial", 0.0))
```

Validate all are finite and nonnegative when enabled.

In both training loss and no-grad scalar recomputation, add:

```python
compatible_constraint = v42_compatible_constraint_terms(
    support_compatible_mse=compatible_mse,
    compatible_mse_budget=compatible_mse_budget,
    compatible_dual_lambda=compatible_dual_lambda,
    compatible_augmented_weight=compatible_augmented_weight,
)
```

Include `compatible_constraint["compatible_constraint_penalty"]` in `loss` only when `v42_compatible_dual_enabled` is true.

After `optimizer.step()` and norm clipping, recompute the projected delta, compatible logits, compatible MSE, and compatible constraint from the current clipped values. Use that post-step residual for both scalar logging and dual update:

```python
if v42_compatible_dual_enabled:
    current_compatible_constraint = v42_compatible_constraint_terms(
        support_compatible_mse=current_compatible_mse,
        compatible_mse_budget=compatible_mse_budget,
        compatible_dual_lambda=compatible_dual_lambda,
        compatible_augmented_weight=compatible_augmented_weight,
    )
    dual_residual = float(
        current_compatible_constraint["compatible_constraint_residual"].detach().item()
    )
    compatible_dual_lambda = min(
        compatible_dual_max,
        max(0.0, compatible_dual_lambda + compatible_dual_lr * dual_residual),
    )
```

Add these scalar fields to `scalar_losses` and `frontier_candidates` when enabled:

```python
compatible_constraint_residual = float(
    current_compatible_constraint["compatible_constraint_residual"].detach().item()
)
"compatible_constraint_feasible": bool(
    scalar_losses["support_compatible_mse"] <= compatible_mse_budget
),
"compatible_constraint_residual": compatible_constraint_residual,
"compatible_dual_lambda": float(compatible_dual_lambda),
"compatible_mse_budget": float(compatible_mse_budget),
```

Use `select_v42_compatible_dual_frontier_candidate` and `redact_v42_compatible_dual_frontier_progress_event` when `v42_compatible_dual_enabled` is true; otherwise keep V41 behavior.

- [ ] **Step 6: Add stale-dual regression**

Add this solver-level test near `test_v37_projected_optimizer_final_loss_matches_returned_delta`:

```python
def test_v42_dual_update_and_residual_use_post_step_projected_delta(
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
    compatible_gradient = torch.ones(v25.SOURCE_WEIGHT_DIM)
    compatible_jacobian = torch.zeros(1, v25.SOURCE_WEIGHT_DIM)

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
    monkeypatch.setattr(v25, "v32_support_behavior_margins", lambda **kwargs: {
        "has_majority": kwargs["weights"][1] - 0.30,
        "mountain_pattern": kwargs["weights"][1] - 0.20,
        "sorted_ascending": kwargs["weights"][1] - 0.10,
        "sorted_descending": kwargs["weights"][1],
    })

    config = {
        **v25.build_v42_compatible_dual_frontier_config_grid()[0],
        "compatible_augmented_weight": 0.0,
        "compatible_dual_initial": 0.0,
        "compatible_dual_lr": 0.5,
        "compatible_dual_max": 100.0,
        "compatible_mse_budget": 0.0,
        "projected_optimizer_epochs": 1,
        "projected_optimizer_lr": 0.2,
        "projection_strength": 0.0,
        "trust_norm_cap": 1.0,
    }
    result = v25.solve_v37_projected_support_optimizer_edit(
        compatible_gradient=compatible_gradient,
        compatible_jacobian=compatible_jacobian,
        coordinate_hash="a" * 64,
        selected_coordinates=[0, 1],
        config=config,
        source_weights=source,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
    )

    edited = source + result["delta"]
    compatible_logits = v25.v27_subject_logits_for_inputs(
        edited,
        support["compatible_inputs"],
    )
    compatible_mse = torch.nn.functional.mse_loss(
        compatible_logits,
        support["compatible_source_logits"].to(dtype=torch.float32),
    )
    expected_residual = float(compatible_mse.item())

    assert result["audit"]["compatible_constraint_residual"] == pytest.approx(
        expected_residual,
        rel=1e-5,
        abs=1e-6,
    )
    assert result["audit"]["compatible_dual_lambda"] == pytest.approx(
        0.5 * expected_residual,
        rel=1e-5,
        abs=1e-6,
    )
    assert result["audit"]["compatible_dual_update_source"] == "post_step_projected_delta"
```

Add `compatible_dual_update_source: "post_step_projected_delta"` to the V42 audit. If this test fails, the implementation is likely logging or updating from stale pre-step residuals.

- [ ] **Step 7: Add V42 matched-edit wrapper and dispatch route**

Add the wrapper near `evaluate_v25_trajectory_frontier_matched_edit`:

```python
def evaluate_v25_compatible_dual_frontier_matched_edit(
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate_v25_target_tolerance_locality_budget_matched_edit(**kwargs)
```

Add a development-job dispatch branch after the V41 branch:

```python
    elif matched_edit_source == V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE:
        if "train_pool_file_sha256" not in config:
            raise ValueError(
                "train_pool_file_sha256 is required for "
                "compatible_dual_frontier_projected_optimizer_sparse"
            )
        if "train_pool_summary_hash" not in config:
            raise ValueError(
                "train_pool_summary_hash is required for "
                "compatible_dual_frontier_projected_optimizer_sparse"
            )
        train_pool_file_sha256 = require_sha256_hex(
            config["train_pool_file_sha256"],
            field_name="train_pool_file_sha256",
        )
        train_pool_summary_hash = require_sha256_hex(
            config["train_pool_summary_hash"],
            field_name="train_pool_summary_hash",
        )
        matched = evaluate_v25_compatible_dual_frontier_matched_edit(
            subject=subject,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            config=config,
            norm_cap=norm_cap,
            selected_config_hash=selected_hash,
            train_pool_file_sha256=train_pool_file_sha256,
            train_pool_summary_hash=train_pool_summary_hash,
            script_sha256=str(script_sha256),
            progress_log_path=progress_log_path,
            started_at_monotonic=started_at_monotonic,
            now_monotonic=now_monotonic,
            record_id_hash=stable_hash_json(str(job["record_id"])),
        )
```

- [ ] **Step 8: Add dispatch test**

Add this test near the V41 dispatch test:

```python
def test_v42_dispatch_uses_compatible_dual_frontier_optimizer_and_v25_controls(
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
                "matched_edit_source": (
                    v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
                ),
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
        "evaluate_v25_compatible_dual_frontier_matched_edit",
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
            **v25.build_v42_compatible_dual_frontier_config_grid()[0],
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
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )
    assert called["matched_config"]["v42_compatible_dual_enabled"] is True
    assert called["control_config"] == {
        "compat_weight": 0.1,
        "projection": "rank1",
        "ridge_lambda": 1e-2,
    }
    assert proof["matched"]["editor"]["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )
```

- [ ] **Step 9: Add real-wrapper metadata regression**

Add this test near `test_v37_matched_edit_metadata_uses_projected_optimizer_boundary`:

```python
def test_v42_matched_edit_metadata_stamps_dual_frontier_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    subject = {
        "pattern": "sorted_ascending",
        "subject_id": "hidden",
        "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
    }

    monkeypatch.setattr(
        v25,
        "v28_anchor_gradients_and_compatible_jacobian",
        lambda **_kwargs: {
            "compatible_jacobian": torch.zeros(1, v25.SOURCE_WEIGHT_DIM),
            "g_compatible": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_conflict": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "g_target": torch.ones(v25.SOURCE_WEIGHT_DIM),
            "support_split_counts": {"compatible": 1, "conflict": 1, "target": 1},
        },
    )
    monkeypatch.setattr(
        v25,
        "solve_v37_projected_support_optimizer_edit",
        lambda **_kwargs: {
            "audit": {
                "compatible_constraint_feasible": True,
                "compatible_constraint_residual": 0.0,
                "compatible_dual_lambda": 0.0,
                "compatible_dual_update_source": "post_step_projected_delta",
                "compatible_mse_budget": 10.0,
                "final_loss": 1.0,
                "optimization_steps": 1,
                "optimizer_audit_hash": "f" * 64,
                "support_compatible_mse": 0.0,
                "support_runner_margin": 0.1,
                "support_target_margin": 0.2,
                "support_tournament_margin": 0.3,
                "trajectory_frontier_selection_mode": (
                    "frontier_target_and_compatible_feasible"
                ),
            },
            "delta": torch.zeros(v25.SOURCE_WEIGHT_DIM),
        },
    )
    monkeypatch.setattr(
        v25,
        "control_record_for_delta",
        lambda **kwargs: captured.setdefault(
            "metadata",
            dict(kwargs["metadata"]),
        ) or {
            "compatible_source_output_mse": 0.0,
            "delta_norm": 0.0,
            "editor": dict(kwargs["metadata"]),
            "target_margin": 0.0,
        },
    )

    result = v25.evaluate_v25_compatible_dual_frontier_matched_edit(
        subject=subject,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        config={
            **v25.build_v42_compatible_dual_frontier_config_grid()[0],
            "train_pool_file_sha256": "a" * 64,
            "train_pool_summary_hash": "b" * 64,
        },
        norm_cap=0.25,
        selected_config_hash="c" * 64,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        script_sha256="script",
    )

    metadata = captured["metadata"]
    assert metadata["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )
    assert metadata["experiment_variant"] == v25.V42_EXPERIMENT_VARIANT
    assert metadata["optimizer_audit"]["compatible_dual_update_source"] == (
        "post_step_projected_delta"
    )
    assert metadata["optimizer_audit"]["compatible_constraint_feasible"] is True
    assert "compatible_dual_frontier_provenance_hash" in metadata
    assert result["editor"]["matched_edit_source"] == (
        v25.V42_COMPATIBLE_DUAL_FRONTIER_MATCHED_EDIT_SOURCE
    )
```

- [ ] **Step 10: Verify focused tests**

Run V42-focused tests only:

```bash
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -k 'v42 or v41_frontier_selection' -q
```

Expected: all selected tests pass.

## Task 3: Run Bounded V42 Diagnostic and Write Results

**Files:**
- Create: `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v42-compatible-dual-frontier-results.md`

- [ ] **Step 1: Syntax and helper verification**

Run:

```bash
python -m py_compile /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: compile succeeds and helper suite passes. Do not run lint.

- [ ] **Step 2: Launch bounded run with terminal log**

Run:

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v42-compatible-dual-frontier \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions \
  > /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v42_compatible_dual_frontier_terminal.log 2>&1
```

Expected: run exits normally within the bounded development job count.

- [ ] **Step 3: Monitor long-run evidence**

During and after the run, check:

```bash
pgrep -fl 'train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py'
wc -l /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v42_compatible_dual_frontier_terminal.log
tail -n 8 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
shasum -a 256 /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v42_compatible_dual_frontier_terminal.log
```

Expected: monitor heartbeats advance, progress rows advance, terminal hash is recorded, final event is `monitor_stop`, and no script-specific orphan remains after completion.

- [ ] **Step 4: Parse result metrics**

Use a small Python parser that reads only `development_progress.jsonl` and `long_run_monitor.jsonl`. Extract:

- Event counts for `inner_validation_candidate_completed`, `development_evaluation_record_completed`, `v42_compatible_dual_frontier_optimizer_progress`, `v42_compatible_dual_frontier_optimizer_completed`, `v42_compatible_dual_frontier_selected`, `inner_validation_completed`.
- Candidate target prediction rate, proof failure count, compatible-MSE failure count.
- `localized_feasible_count` distribution.
- Selection modes.
- Compatible budget, residual, and dual lambda summaries.
- Strict raw-key scan over progress and monitor logs.

The strict raw-key scan must include:

```text
final_subjects
final_subjects_path
subject_id
weights
logits
gradient
selected_coordinates
support_examples
sequence
compatible_jacobian
raw_delta
projected_delta
basis
```

- [ ] **Step 5: Write V42 results**

Create `/Users/max/Desktop/muat/docs/superpowers/plans/2026-06-14-v42-compatible-dual-frontier-results.md` with:

- Status: pass, partial diagnostic, or negative result.
- Verification commands and outcomes.
- Run evidence: terminal/progress/monitor rows, elapsed time, CPU seconds, hashes, process checks.
- Leak checks, explicitly stating final holdout remained sealed.
- Candidate metrics and frontier diagnostics.
- Conservative interpretation relative to V41 and V31-V33.
- Literature-linked explanation of why the result does or does not support constrained steering.

- [ ] **Step 6: Reviewer gate**

Send the results doc to Kepler and require `5/5` before treating V42 as accepted. If confidence is below `5/5`, patch the writeup or analysis and re-review.

## Self-Review

- Spec coverage: The plan addresses V41's observed failure mode, keeps final holdout sealed, includes monitoring evidence, uses pytest, and requires a reviewer confidence gate.
- Placeholder scan: No placeholder or unspecified code-change steps remain.
- Type consistency: V42 field names are consistent across helper, selector, redaction, optimizer audit, and result parsing.
