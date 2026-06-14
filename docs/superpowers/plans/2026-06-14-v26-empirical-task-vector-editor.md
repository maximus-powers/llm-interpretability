# V26 Empirical Task-Vector Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a train-only empirical task-vector candidate editor so the next bounded run can determine whether direct weight arithmetic moves behavior better than the V25 Jacobian-synthesized edit.

**Architecture:** Keep the sealed final split untouched and continue using the V25 development/proof pipeline. Add a new `matched_edit_source` config axis with `jacobian` and `empirical_centroid_task_vector`; empirical candidates build source/target behavior centroids from train weights only, require train-pool provenance hashes, project/cap the direction with the existing projection code, and evaluate that delta as the matched edit under the same held-out development jobs and control gates. The existing `teacher_oracle_delta` remains diagnostic-only and must not be selectable as a matched editor.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing MUAT V25 script/test harness.

---

## Literature Support

- Task arithmetic shows that model behavior can sometimes be steered by vector arithmetic in parameter space, but also that task vectors need empirical validation rather than assumed transferability: [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089).
- Ties/merging work shows that interference and sign conflicts in weight deltas can make naive arithmetic misleading, motivating explicit controls and candidate ranking: [TIES-Merging](https://arxiv.org/abs/2306.01708).
- Model soups show that averaging weights can work when models share enough basin/initialization structure, but this is an empirical condition, not guaranteed: [Model Soups](https://arxiv.org/abs/2203.05482).
- Hypernetwork and weight-space learning literature supports learning from model weights, but also points to the need for direct weight-space baselines before claiming interpreted activation-to-weight editing: [HyperNetworks](https://arxiv.org/abs/1609.09106).
- Model-edit tracing literature supports explicit provenance for low-rank edits, which is why the new empirical vector should reuse existing projection/cap audit paths instead of bypassing them: [Tracing and Reversing Edits in LLMs](https://arxiv.org/abs/2505.20819).
- Representation engineering reliability work warns that activation steering results can be noisy or control-sensitive, so the plan keeps train/dev separation, shuffled-signature controls, and non-oracle proof gates: [On the Reliability of Representation Engineering](https://arxiv.org/html/2502.17601v1).

**Risk note:** Empirical centroid weight arithmetic is a diagnostic baseline, not an assumed steering method. It can be confounded by hidden-unit permutation, initialization basin differences, sign conflicts, and cancellation; a negative result is informative, and a positive result still requires control-margin review before any claim.

## Files

- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create or update result notes after run: `docs/superpowers/plans/2026-06-14-v26-empirical-task-vector-editor-results.md`

## Design Constraints

- Do not open or read `runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.
- Do not run lint unless explicitly requested.
- All progress logs must remain redacted: no raw `weights`, `subject_id`, raw `delta`, descriptor tensors, logits, or Jacobians.
- `teacher_oracle_delta` remains diagnostic-only. It cannot be included in `MATCHED_EDIT_SOURCE_GRID`.
- Every V26 result/progress payload must include `experiment_variant="v26_empirical_task_vector_editor"` so it is not confused with prior V25 runs.
- Long-running runs must use `--monitor-interval-seconds` and JSONL progress logs.

### Task 1: Add Config Axis and Hash Binding

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing tests for the new config axis**

Add tests near the existing config-grid tests:

```python
def test_v25_config_grid_interleaves_matched_edit_sources() -> None:
    grid = v25.build_v25_config_grid()
    first_sources = [config["matched_edit_source"] for config in grid[:2]]
    assert first_sources == ["jacobian", "empirical_centroid_task_vector"]
    assert {config["matched_edit_source"] for config in grid} == {
        "jacobian",
        "empirical_centroid_task_vector",
    }
    assert len(grid) == (
        len(v25.RIDGE_GRID)
        * len(v25.COMPAT_WEIGHT_GRID)
        * len(v25.PROJECTION_GRID)
        * len(v25.MATCHED_EDIT_SOURCE_GRID)
    )
    first_block = [
        (
            config["ridge_lambda"],
            config["compat_weight"],
            config["projection"],
            config["matched_edit_source"],
        )
        for config in grid[:8]
    ]
    assert first_block == [
        (1e-5, 0.0, "none", "jacobian"),
        (1e-5, 0.0, "none", "empirical_centroid_task_vector"),
        (1e-5, 0.0, "rank1", "jacobian"),
        (1e-5, 0.0, "rank1", "empirical_centroid_task_vector"),
        (1e-5, 0.0, "spectral_rank4", "jacobian"),
        (1e-5, 0.0, "spectral_rank4", "empirical_centroid_task_vector"),
        (1e-5, 0.0, "rank1_spectral_rank4", "jacobian"),
        (1e-5, 0.0, "rank1_spectral_rank4", "empirical_centroid_task_vector"),
    ]
    assert len(v25.stable_hash_json(grid)) == 64
    assert "teacher_oracle_delta" not in v25.MATCHED_EDIT_SOURCE_GRID
    assert "teacher_oracle_delta" in v25.DIAGNOSTIC_CONTROL_TYPES
    assert "teacher_oracle_delta" not in v25.PROOF_CRITICAL_CONTROL_TYPES


def test_v25_inner_validation_config_hash_binds_matched_edit_source() -> None:
    base = {
        "compat_weight": 0.0,
        "config_index": 0,
        "matched_edit_source": "jacobian",
        "projection": "none",
        "ridge_lambda": 1e-5,
    }
    changed = {**base, "matched_edit_source": "empirical_centroid_task_vector"}
    assert v25.v25_inner_validation_config_hash(base) != (
        v25.v25_inner_validation_config_hash(changed)
    )
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_config_grid_interleaves_matched_edit_sources model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_inner_validation_config_hash_binds_matched_edit_source -q
```

Expected: fail because `matched_edit_source` is not defined.

- [ ] **Step 3: Implement the config axis**

In `train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`, add:

```python
MATCHED_EDIT_SOURCE_GRID = ["jacobian", "empirical_centroid_task_vector"]
```

Update `build_v25_config_grid()` so the first bounded configs compare edit sources before sweeping more hyperparameters:

```python
def build_v25_config_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for ridge_lambda in RIDGE_GRID:
        for compat_weight in COMPAT_WEIGHT_GRID:
            for projection in PROJECTION_GRID:
                for matched_edit_source in MATCHED_EDIT_SOURCE_GRID:
                    grid.append({
                        "compat_weight": float(compat_weight),
                        "config_index": len(grid),
                        "matched_edit_source": str(matched_edit_source),
                        "projection": str(projection),
                        "ridge_lambda": float(ridge_lambda),
                    })
    return grid
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_config_grid_interleaves_matched_edit_sources model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_inner_validation_config_hash_binds_matched_edit_source -q
```

Expected: pass.

### Task 2: Build Train-Only Empirical Task-Vector Bank

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing tests for train-only centroid deltas**

Add:

```python
def test_v25_empirical_task_vector_bank_uses_train_weight_centroids_only() -> None:
    train_subjects = [
        {"pattern": "sorted_ascending", "subject_id": "a1", "weights": [1.0] * v25.SOURCE_WEIGHT_DIM},
        {"pattern": "sorted_ascending", "subject_id": "a2", "weights": [3.0] * v25.SOURCE_WEIGHT_DIM},
        {"pattern": "sorted_descending", "subject_id": "d1", "weights": [5.0] * v25.SOURCE_WEIGHT_DIM},
        {"pattern": "sorted_descending", "subject_id": "d2", "weights": [7.0] * v25.SOURCE_WEIGHT_DIM},
    ]
    bank = v25.build_v25_empirical_task_vector_bank(
        train_subjects=train_subjects,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        config={
            "compat_weight": 0.0,
            "matched_edit_source": "empirical_centroid_task_vector",
            "projection": "none",
            "ridge_lambda": 1e-5,
        },
        norm_cap=1000.0,
        script_sha256="script",
    )
    entry = next(
        item for item in bank["entries"]
        if item["source_behavior"] == "sorted_ascending"
        and item["target_behavior"] == "sorted_descending"
    )
    assert torch.allclose(entry["delta"], torch.full((v25.SOURCE_WEIGHT_DIM,), 4.0))
    assert entry["source_count"] == 2
    assert entry["target_count"] == 2
    assert len(entry["source_centroid_sha256"]) == 64
    assert len(entry["target_centroid_sha256"]) == 64
    assert len(entry["delta_sha256"]) == 64
    assert bank["train_pool_file_sha256"] == "a" * 64
    assert bank["train_pool_summary_hash"] == "b" * 64
    assert len(bank["bank_hash"]) == 64


def test_v25_empirical_task_vector_bank_requires_train_provenance() -> None:
    with pytest.raises(ValueError, match="train_pool_file_sha256"):
        v25.build_v25_empirical_task_vector_bank(
            train_subjects=[],
            train_pool_file_sha256="not-a-sha",
            train_pool_summary_hash="b" * 64,
            config={
                "compat_weight": 0.0,
                "matched_edit_source": "empirical_centroid_task_vector",
                "projection": "none",
                "ridge_lambda": 1e-5,
            },
            norm_cap=0.25,
            script_sha256="script",
        )
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_empirical_task_vector_bank_uses_train_weight_centroids_only -q
```

Expected: fail because the function does not exist.

- [ ] **Step 3: Implement centroid bank helpers**

Add helpers near the existing train delta bank functions:

```python
def behavior_weight_centroids(
    train_subjects: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for record in train_subjects:
        behavior = str(record["pattern"])
        grouped.setdefault(behavior, []).append(record_weights_tensor(record))
    centroids = {}
    for behavior, weights in sorted(grouped.items()):
        if not weights:
            raise ValueError(f"no train weights for behavior {behavior}")
        matrix = torch.stack([weight.reshape(-1).to(dtype=torch.float32) for weight in weights])
        centroid = matrix.mean(dim=0).to(dtype=torch.float32)
        if int(centroid.numel()) != SOURCE_WEIGHT_DIM:
            raise ValueError("behavior centroid has wrong dimension")
        if not torch.isfinite(centroid).all():
            raise ValueError("behavior centroid is nonfinite")
        centroids[behavior] = {"centroid": centroid, "count": len(weights)}
    return centroids


def build_v25_empirical_task_vector_bank(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    config: Mapping[str, Any],
    norm_cap: float,
    script_sha256: str,
    spectral_basis: torch.Tensor | None = None,
) -> dict[str, Any]:
    checked_train_pool_file_sha256 = require_sha256_hex(
        train_pool_file_sha256,
        field_name="train_pool_file_sha256",
    )
    checked_train_pool_summary_hash = require_sha256_hex(
        train_pool_summary_hash,
        field_name="train_pool_summary_hash",
    )
    centroids = behavior_weight_centroids(train_subjects)
    centroid_hash_by_behavior = {
        behavior: stable_hash_json(tensor_to_hashable(payload["centroid"]))
        for behavior, payload in centroids.items()
    }
    count_by_behavior = {
        behavior: int(payload["count"])
        for behavior, payload in centroids.items()
    }
    entries = []
    for source_behavior in sorted(centroids):
        for target_behavior in sorted(centroids):
            if source_behavior == target_behavior:
                continue
            raw_delta = (
                centroids[target_behavior]["centroid"]
                - centroids[source_behavior]["centroid"]
            )
            projected_delta = project_delta_for_config(
                raw_delta,
                projection=str(config["projection"]),
                spectral_basis=spectral_basis,
            )
            delta = apply_norm_cap(projected_delta, max_norm=float(norm_cap))
            entry = {
                "delta": delta,
                "delta_sha256": stable_hash_json(tensor_to_hashable(delta)),
                "direction": f"{source_behavior}->{target_behavior}",
                "editor_audit": {
                    "delta_sha256": stable_hash_json(tensor_to_hashable(delta)),
                    "edit_source": "empirical_centroid_task_vector",
                    "norm_cap": float(norm_cap),
                    "norm_cap_applied": bool(
                        float(torch.linalg.norm(projected_delta).item())
                        > float(norm_cap) + 1e-8
                    ),
                    "projected_delta_norm": float(torch.linalg.norm(projected_delta).item()),
                    "projection": str(config["projection"]),
                    "raw_delta_norm": float(torch.linalg.norm(raw_delta).item()),
                    "script_sha256": str(script_sha256),
                },
                "source_behavior": source_behavior,
                "source_centroid_sha256": centroid_hash_by_behavior[source_behavior],
                "source_count": int(centroids[source_behavior]["count"]),
                "target_behavior": target_behavior,
                "target_centroid_sha256": centroid_hash_by_behavior[target_behavior],
                "target_count": int(centroids[target_behavior]["count"]),
            }
            entries.append(entry)
    entry_hashes = [v25_train_delta_entry_hash(entry) for entry in entries]
    bank_hash = stable_hash_json({
        "config": dict(config),
        "count_by_behavior": count_by_behavior,
        "centroid_hash_by_behavior": centroid_hash_by_behavior,
        "entry_hashes": entry_hashes,
        "norm_cap": float(norm_cap),
        "scope": "v25_empirical_task_vector_bank",
        "script_sha256": str(script_sha256),
        "train_pool_file_sha256": checked_train_pool_file_sha256,
        "train_pool_summary_hash": checked_train_pool_summary_hash,
    })
    return {
        "bank_hash": bank_hash,
        "centroid_hash_by_behavior": centroid_hash_by_behavior,
        "count_by_behavior": count_by_behavior,
        "entries": entries,
        "entry_count": len(entries),
        "entry_hashes": entry_hashes,
        "train_pool_file_sha256": checked_train_pool_file_sha256,
        "train_pool_summary_hash": checked_train_pool_summary_hash,
    }
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_empirical_task_vector_bank_uses_train_weight_centroids_only -q
```

Expected: pass.

### Task 3: Evaluate Empirical Task Vector as Matched Edit

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing tests for matched-source dispatch**

Add:

```python
def test_v25_empirical_task_vector_matched_edit_uses_precomputed_direction_delta() -> None:
    weights = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    record = {
        "pattern": "sorted_ascending",
        "subject_id": "empirical-dev",
        "weights": weights.tolist(),
    }
    delta = torch.zeros(v25.SOURCE_WEIGHT_DIM)
    delta[0] = 0.125
    bank = {
        "bank_hash": "b" * 64,
        "entries": [{
            "delta": delta,
            "delta_sha256": v25.stable_hash_json(v25.tensor_to_hashable(delta)),
            "direction": "sorted_ascending->sorted_descending",
            "editor_audit": {"edit_source": "empirical_centroid_task_vector"},
            "source_behavior": "sorted_ascending",
            "target_behavior": "sorted_descending",
        }],
    }
    result = v25.evaluate_v25_empirical_task_vector_matched_edit(
        subject=record,
        source_behavior="sorted_ascending",
        target_behavior="sorted_descending",
        empirical_task_vector_bank=bank,
        selected_config_hash="c" * 64,
        spectral_basis=torch.eye(v25.SOURCE_WEIGHT_DIM, 4),
    )
    assert result["control_type"] == v25.EDITOR_METHOD
    assert result["editor"]["matched_edit_source"] == "empirical_centroid_task_vector"
    assert result["editor"]["empirical_task_vector_bank_hash"] == "b" * 64
    assert result["delta_norm"] == pytest.approx(0.125)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_empirical_task_vector_matched_edit_uses_precomputed_direction_delta -q
```

Expected: fail because the function does not exist.

- [ ] **Step 3: Implement matched-source evaluation**

Add:

```python
def empirical_task_vector_entry_for_direction(
    bank: Mapping[str, Any],
    *,
    source_behavior: str,
    target_behavior: str,
) -> Mapping[str, Any]:
    matches = [
        entry for entry in bank.get("entries", [])
        if str(entry["source_behavior"]) == str(source_behavior)
        and str(entry["target_behavior"]) == str(target_behavior)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one empirical task vector for "
            f"{source_behavior}->{target_behavior}, found {len(matches)}"
        )
    return matches[0]


def evaluate_v25_empirical_task_vector_matched_edit(
    *,
    subject: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    empirical_task_vector_bank: Mapping[str, Any],
    selected_config_hash: str,
    spectral_basis: torch.Tensor | None,
) -> dict[str, Any]:
    bank_hash = require_sha256_hex(
        empirical_task_vector_bank.get("bank_hash"),
        field_name="empirical_task_vector_bank.bank_hash",
    )
    selected_hash = require_sha256_hex(
        selected_config_hash,
        field_name="selected_config_hash",
    )
    entry = empirical_task_vector_entry_for_direction(
        empirical_task_vector_bank,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
    )
    delta = torch.as_tensor(entry["delta"], dtype=torch.float32).reshape(-1)
    spectral_norm = 0.0
    if spectral_basis is not None:
        spectral_norm = float(torch.linalg.norm(project_to_basis(delta, spectral_basis)).item())
    return control_record_for_delta(
        control_type=EDITOR_METHOD,
        delta=delta,
        source_behavior=source_behavior,
        source_weights=record_weights_tensor(subject),
        target_behavior=target_behavior,
        metadata={
            **dict(entry.get("editor_audit", {})),
            "empirical_task_vector_bank_hash": bank_hash,
            "matched_edit_source": "empirical_centroid_task_vector",
            "matched_spectral_projection_norm": spectral_norm,
            "selected_config_hash": selected_hash,
        },
    )
```

- [ ] **Step 4: Wire dispatch into development evaluation**

Update `evaluate_v25_development_job()` and downstream wrappers to accept `empirical_task_vector_bank: Mapping[str, Any] | None = None`. Dispatch:

```python
matched_edit_source = str(config.get("matched_edit_source", "jacobian"))
if matched_edit_source == "jacobian":
    matched = evaluate_v25_matched_edit(
        subject=subject,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        train_stats=train_stats,
        cache_entry=cache_entry,
        config=config,
        norm_cap=norm_cap,
        spectral_basis=spectral_basis,
    )
elif matched_edit_source == "empirical_centroid_task_vector":
    if empirical_task_vector_bank is None:
        raise ValueError("empirical_task_vector_bank is required")
    matched = evaluate_v25_empirical_task_vector_matched_edit(
        subject=subject,
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        empirical_task_vector_bank=empirical_task_vector_bank,
        selected_config_hash=selected_hash,
        spectral_basis=spectral_basis,
    )
else:
    raise ValueError(f"unknown matched_edit_source: {matched_edit_source}")
```

Pass the empirical bank from `run_v25_inner_validation_successive_halving_with_progress()` into `evaluate_v25_development_jobs_with_progress()`.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_empirical_task_vector_matched_edit_uses_precomputed_direction_delta -q
```

Expected: pass.

### Task 4: Add Progress Logging and Redaction Checks

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing redaction/progress test**

Add:

```python
def test_v25_empirical_task_vector_bank_progress_is_hash_only(tmp_path: Path) -> None:
    train_subjects = [
        {"pattern": "sorted_ascending", "subject_id": "secret-a", "weights": [1.0] * v25.SOURCE_WEIGHT_DIM},
        {"pattern": "sorted_descending", "subject_id": "secret-d", "weights": [2.0] * v25.SOURCE_WEIGHT_DIM},
    ]
    log_path = tmp_path / "progress.jsonl"
    bank = v25.build_v25_empirical_task_vector_bank_with_progress(
        train_subjects=train_subjects,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        config={
            "compat_weight": 0.0,
            "matched_edit_source": "empirical_centroid_task_vector",
            "projection": "none",
            "ridge_lambda": 1e-5,
        },
        norm_cap=0.25,
        script_sha256="script",
        progress_log_path=log_path,
        started_at_monotonic=1.0,
    )
    text = log_path.read_text()
    assert "empirical_task_vector_bank_start" in text
    assert "empirical_task_vector_bank_completed" in text
    assert "secret-a" not in text
    assert "secret-d" not in text
    assert "weights" not in text
    assert "delta" not in text
    assert len(bank["bank_hash"]) == 64
```

- [ ] **Step 2: Implement progress wrapper**

Add:

```python
def build_v25_empirical_task_vector_bank_with_progress(
    *,
    train_subjects: Sequence[Mapping[str, Any]],
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    config: Mapping[str, Any],
    norm_cap: float,
    script_sha256: str,
    spectral_basis: torch.Tensor | None = None,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    checked_train_pool_file_sha256 = require_sha256_hex(
        train_pool_file_sha256,
        field_name="train_pool_file_sha256",
    )
    checked_train_pool_summary_hash = require_sha256_hex(
        train_pool_summary_hash,
        field_name="train_pool_summary_hash",
    )
    record_progress_event(
        progress_log_path,
        event="empirical_task_vector_bank_start",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "config_hash": stable_hash_json(dict(config)),
            "experiment_variant": "v26_empirical_task_vector_editor",
            "norm_cap": float(norm_cap),
            "train_pool_file_sha256": checked_train_pool_file_sha256,
            "train_pool_summary_hash": checked_train_pool_summary_hash,
            "train_subject_count": len(train_subjects),
        },
    )
    bank = build_v25_empirical_task_vector_bank(
        train_subjects=train_subjects,
        train_pool_file_sha256=checked_train_pool_file_sha256,
        train_pool_summary_hash=checked_train_pool_summary_hash,
        config=config,
        norm_cap=norm_cap,
        script_sha256=script_sha256,
        spectral_basis=spectral_basis,
    )
    record_progress_event(
        progress_log_path,
        event="empirical_task_vector_bank_completed",
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
        extra={
            "bank_hash": bank["bank_hash"],
            "centroid_hashes_hash": stable_hash_json(bank["centroid_hash_by_behavior"]),
            "count_by_behavior": bank["count_by_behavior"],
            "entry_count": bank["entry_count"],
            "entry_hashes_hash": stable_hash_json(bank["entry_hashes"]),
            "experiment_variant": "v26_empirical_task_vector_editor",
            "train_pool_file_sha256": bank["train_pool_file_sha256"],
            "train_pool_summary_hash": bank["train_pool_summary_hash"],
        },
    )
    return bank
```

Do not include raw subjects, weights, deltas, descriptors, or subject IDs in either event.

- [ ] **Step 3: Run the focused redaction test**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_empirical_task_vector_bank_progress_is_hash_only -q
```

Expected: pass.

### Task 5: Wire Inner Validation and Run Verification

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing test for runner provenance plumbing**

Add a focused test near the existing inner-validation path tests:

```python
def test_v25_inner_validation_passes_train_pool_provenance_to_empirical_bank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}
    config = {
        "compat_weight": 0.0,
        "config_index": 0,
        "matched_edit_source": "empirical_centroid_task_vector",
        "projection": "none",
        "ridge_lambda": 1e-5,
    }
    train_subjects = [
        {
            "pattern": pattern,
            "subject_id": f"train-{pattern}",
            "weights": torch.zeros(v25.SOURCE_WEIGHT_DIM).tolist(),
        }
        for pattern in v25.PATTERNS
    ]
    train_stats = {
        "activation_descriptor_mean": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
        "activation_descriptor_std": torch.ones(v25.ACTIVATION_DESCRIPTOR_DIM),
        "descriptor_norm_hash": "norm",
        "probe_examples": [{"sequence": [0, 1, 2, 3, 4]}],
        "probe_examples_hash": "p" * 64,
        "target_activation_descriptor_by_behavior": {
            pattern: torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM)
            for pattern in v25.PATTERNS
        },
        "train_statistics_hash": "s" * 64,
    }

    def fake_train_bank(**kwargs: Any) -> dict[str, Any]:
        return {
            "bank_hash": "j" * 64,
            "entries": [{
                "delta": torch.zeros(v25.SOURCE_WEIGHT_DIM),
                "delta_sha256": "d" * 64,
                "direction": "sorted_ascending->sorted_descending",
                "source_behavior": "sorted_ascending",
                "source_descriptor": torch.zeros(v25.ACTIVATION_DESCRIPTOR_DIM),
                "target_behavior": "sorted_descending",
            }],
            "entry_count": 1,
            "entry_hashes": ["e" * 64],
        }

    def fake_empirical_bank(**kwargs: Any) -> dict[str, Any]:
        captured["train_pool_file_sha256"] = kwargs["train_pool_file_sha256"]
        captured["train_pool_summary_hash"] = kwargs["train_pool_summary_hash"]
        return {
            "bank_hash": "k" * 64,
            "entries": [{
                "delta": torch.zeros(v25.SOURCE_WEIGHT_DIM),
                "delta_sha256": "f" * 64,
                "direction": "sorted_ascending->sorted_descending",
                "editor_audit": {"edit_source": "empirical_centroid_task_vector"},
                "source_behavior": "sorted_ascending",
                "target_behavior": "sorted_descending",
            }],
            "entry_count": 1,
            "entry_hashes": ["g" * 64],
        }

    monkeypatch.setattr(v25, "build_v25_train_delta_bank_with_progress", fake_train_bank)
    monkeypatch.setattr(
        v25,
        "build_v25_empirical_task_vector_bank_with_progress",
        fake_empirical_bank,
    )
    monkeypatch.setattr(v25, "build_v25_train_only_control_contexts_with_progress", lambda **kwargs: {})
    monkeypatch.setattr(v25, "evaluate_v25_development_jobs_with_progress", lambda **kwargs: {
        "evaluated_count": 0,
        "proof_records": [],
    })

    v25.run_v25_inner_validation_successive_halving_with_progress(
        configs=[config],
        jobs=[],
        train_subjects=train_subjects,
        train_stats=train_stats,
        train_pool_file_sha256="a" * 64,
        train_pool_summary_hash="b" * 64,
        rung_job_counts=[0],
        keep_fractions=[1.0],
        norm_cap=0.25,
        job_plan_hash="c" * 64,
        script_sha256="script",
        progress_log_path=tmp_path / "progress.jsonl",
        started_at_monotonic=1.0,
    )

    assert captured == {
        "train_pool_file_sha256": "a" * 64,
        "train_pool_summary_hash": "b" * 64,
    }
```

- [ ] **Step 2: Add explicit runner parameters**

Change `run_v25_inner_validation_successive_halving_with_progress()` signature:

```python
def run_v25_inner_validation_successive_halving_with_progress(
    *,
    configs: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    train_subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    train_pool_file_sha256: str,
    train_pool_summary_hash: str,
    rung_job_counts: Sequence[int],
    keep_fractions: Sequence[float],
    norm_cap: float,
    job_plan_hash: str,
    script_sha256: str,
    progress_log_path: Path,
    started_at_monotonic: float,
    now_monotonic: Any | None = None,
) -> dict[str, Any]:
    checked_train_pool_file_sha256 = require_sha256_hex(
        train_pool_file_sha256,
        field_name="train_pool_file_sha256",
    )
    checked_train_pool_summary_hash = require_sha256_hex(
        train_pool_summary_hash,
        field_name="train_pool_summary_hash",
    )
```

Include these hashes in `inner_validation_start` progress:

```python
"experiment_variant": "v26_empirical_task_vector_editor",
"train_pool_file_sha256": checked_train_pool_file_sha256,
"train_pool_summary_hash": checked_train_pool_summary_hash,
```

- [ ] **Step 3: Pass safe redacted provenance from development setup**

In `run_v25_development_setup()`, compute the redacted summary once:

```python
train_pool_summary = redact_v25_loaded_pool_summary(prepared["train"])
development_pool_summary = redact_v25_loaded_pool_summary(prepared["development"])
train_pool_summary_hash = stable_hash_json(train_pool_summary)
```

Use these summaries in `result`:

```python
"development_pool": development_pool_summary,
"train_pool": train_pool_summary,
```

Pass provenance into the runner:

```python
inner_validation_result = run_v25_inner_validation_successive_halving_with_progress(
    configs=configs,
    jobs=ordered_development_jobs,
    train_subjects=prepared["train"]["subjects"],
    train_stats=train_stats,
    train_pool_file_sha256=str(train_pool_summary["pool_file_sha256"]),
    train_pool_summary_hash=train_pool_summary_hash,
    rung_job_counts=inner_rung_jobs,
    keep_fractions=inner_keep_fractions,
    norm_cap=0.25,
    job_plan_hash=development_job_summary["job_plan_hash"],
    script_sha256=sha256_file(Path(__file__)),
    progress_log_path=progress_log_path,
    started_at_monotonic=started_at_monotonic,
    now_monotonic=now_monotonic,
)
```

- [ ] **Step 4: Build both banks per candidate without leakage**

In `run_v25_inner_validation_successive_halving_with_progress()`:

```python
empirical_task_vector_bank = None
if str(config.get("matched_edit_source", "jacobian")) == "empirical_centroid_task_vector":
    empirical_task_vector_bank = build_v25_empirical_task_vector_bank_with_progress(
        train_subjects=train_subjects,
        train_pool_file_sha256=checked_train_pool_file_sha256,
        train_pool_summary_hash=checked_train_pool_summary_hash,
        config=config,
        norm_cap=norm_cap,
        script_sha256=script_sha256,
        spectral_basis=spectral_basis,
        progress_log_path=progress_log_path,
        started_at_monotonic=started_at_monotonic,
        now_monotonic=now_monotonic,
    )
```

Pass `empirical_task_vector_bank` into `evaluate_v25_development_jobs_with_progress()`.

- [ ] **Step 5: Include empirical bank hash in candidate summaries**

If the candidate used empirical task vectors, add:

```python
candidate["empirical_task_vector_bank_hash"] = str(empirical_task_vector_bank["bank_hash"])
candidate["experiment_variant"] = "v26_empirical_task_vector_editor"
```

Do not add raw records, weights, or deltas to the summary.

- [ ] **Step 6: Run targeted tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: all V25 helper tests pass.

- [ ] **Step 7: Run syntax verification**

Run:

```bash
python -m py_compile model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Expected: no output and exit code 0.

### Task 6: Bounded Empirical Inner-Validation Run

**Files:**
- No code edits.
- Output: `runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl`
- Output: `runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl`
- Result note: `docs/superpowers/plans/2026-06-14-v26-empirical-task-vector-editor-results.md`

- [ ] **Step 1: Confirm no compute process is already running**

Run:

```bash
ps -ax -o pid=,etime=,pcpu=,pmem=,command= | rg "train_four_behavior_functional_weight_editing_v25|hypernet.train|run_compute_packet" | rg -v rg
```

Expected: no output.

- [ ] **Step 2: Run bounded inner validation**

Run:

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

Expected: a bounded run where the first 8 configs exactly compare `jacobian` and `empirical_centroid_task_vector` across all four projection variants at `ridge_lambda=1e-5` and `compat_weight=0.0`.

- [ ] **Step 3: Monitor logs every 30-60 seconds**

Run in parallel while the process is active:

```bash
tail -n 8 runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
tail -n 30 runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl
ps -ax -o pid=,etime=,pcpu=,pmem=,command= | rg "train_four_behavior_functional_weight_editing_v25|hypernet.train|run_compute_packet" | rg -v rg
```

Expected: heartbeat/progress lines continue advancing; if progress stalls and CPU drops near zero for multiple checks, stop and debug before running more compute.

- [ ] **Step 4: Review result framing**

If `empirical_centroid_task_vector` beats `jacobian`, report it only as bounded development evidence and inspect control margins before claiming steering. If all target prediction rates remain 0, report it as a stronger negative diagnostic: direct train-only weight arithmetic also failed under the same proof gates.

- [ ] **Step 5: Send results to reviewer**

Send the final stdout summary, progress/monitor hashes, no-final-raw-access statement, process-stop check, and interpretation to the reviewer agent. Do not accept or build on the result until reviewer confidence is 5/5.

### Completed Result

The bounded V26 run completed on 2026-06-14 and was accepted by the reviewer at
`5/5` confidence as a negative diagnostic result only. See
`docs/superpowers/plans/2026-06-14-v26-empirical-task-vector-editor-results.md`.

## Self-Review

- Spec coverage: The plan adds a train-only empirical task-vector editor, keeps proof controls, logs long-running progress, and preserves the sealed final split.
- Placeholder scan: No TODO/TBD placeholders remain.
- Type consistency: New config key is `matched_edit_source`; new bank is `empirical_task_vector_bank`; the selectable source is exactly `empirical_centroid_task_vector`.
