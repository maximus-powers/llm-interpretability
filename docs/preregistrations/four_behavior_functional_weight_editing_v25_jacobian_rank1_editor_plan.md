# V25 Jacobian Rank-1 Functional Weight Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether fixed-probe activation signatures can support functional behavior edits when the editor is constrained to first-order activation/weight geometry and low-rank component updates instead of unconstrained full-delta prediction.

**Architecture:** V25 replaces V24's behavioral-distilled full-weight hypereditor with a Jacobian-constrained low-rank editor. It computes closed-form ridge edits that move fixed-probe activations toward target signatures while preserving source behavior, then projects edits into train-only spectral/rank-1 bases. Coefficient-hypernetwork distillation is explicitly out of scope for proof-critical V25 and may only be considered after this closed-form experiment is complete.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing V24/V23 subject-model utilities, JSONL progress logs, SHA-256 artifact binding.

---

## Literature Basis

- [HyperSteer: Activation Steering at Scale with Hypernetworks](https://arxiv.org/abs/2506.03292) motivates conditioning steering on model internals, but V25 deliberately avoids adding a new learned hypernetwork until the closed-form geometric editor is tested.
- [Steer2Edit: From Activation Steering to Component-Level Editing](https://arxiv.org/abs/2602.09870) argues that steering signals can be translated into component-level rank-1 weight edits; V25 adapts this to small subject MLPs by rank-1 projecting per-layer deltas.
- [Weight Updates as Activation Shifts](https://arxiv.org/abs/2603.00425) derives a first-order link between activation interventions and weight updates; V25 makes this the primary training target via explicit Jacobian ridge solves.
- [The Universal Weight Subspace Hypothesis](https://arxiv.org/abs/2512.05117) supports restricting edits to low-dimensional spectral subspaces learned from train-only subject/teacher deltas.
- [A Survey of Weight Space Learning](https://arxiv.org/abs/2603.10090) frames weight-space representation and generation as structured learning problems; V25 explicitly separates understanding, representation, and generation stages.
- [Steering Language Models with Weight Arithmetic](https://arxiv.org/abs/2511.05408) supports contrastive weight directions as behavior controls; V25 includes contrastive source-target and target-source direction baselines.
- [Improved Generalization of Weight Space Networks via Augmentations](https://arxiv.org/abs/2402.04081) warns that weight-space models overfit when model-zoo diversity is low; V25 adds train-only seed diversity and weight-space augmentations before any development selection.

## Non-Negotiable Validity Constraints

- Final raw data remains sealed. The final phase is forbidden unless development passes all preregistered gates and a reviewer returns 5/5 authorization.
- Use fresh V25 pools: train seed base `126400000`, development seed base `127400000`, final seed base `128400000`.
- Bind every candidate grid, selected config, source pool, result artifact, progress log, script, tests, and plan with hashes.
- Long-running phases must emit JSONL progress records that show record counts, worker counts or phase-level activity, finite scalar losses/residuals, and completion counts.
- Do not run lint. Verification is `py_compile` and targeted/full pytest only unless the user explicitly asks for lint.
- Development failure is a valid result. Do not inspect final raw after a negative or inconclusive development result.

## Files

- Create: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Create: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Create: `docs/preregistrations/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor_plan.md`
- Read/reference only: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Output: `runs/four_behavior_functional_weight_editing_v25_pools/`
- Output: `runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/`

## Candidate Method

For a subject model with flattened weights `w`, source pattern `s`, target pattern `t`, fixed probes `X`, source logits `f_w(X)`, target signature descriptor `a_t`, and current source descriptor `a_s`:

1. Build an activation target vector `b = a_t - a_s`.
2. Compute a Jacobian `J = d activation_descriptor(f_w, X) / d w`.
3. Solve a damped edit:

```python
delta = J.T @ torch.linalg.solve(
    J @ J.T + ridge_lambda * torch.eye(J.shape[0], device=J.device),
    b,
)
```

4. Add source-behavior preservation by augmenting `J` and `b` with source-logit rows:

```python
J_aug = torch.cat([J_activation, math.sqrt(compat_weight) * J_source_logits], dim=0)
b_aug = torch.cat([activation_delta, torch.zeros_like(source_logit_delta)], dim=0)
```

5. Project `delta` through one of:

```python
delta_projected = project_per_layer_rank1(delta, layer_shapes)
delta_projected = project_to_train_spectral_basis(delta, basis, rank=k)
delta_projected = project_rank1_then_spectral(delta, layer_shapes, basis, rank=k)
```

6. Evaluate edited weights directly. No learned coefficient hypernetwork participates in V25 proof gates.

## Exact Mathematical Specification

### Fixed Probe Set and Descriptor

- V25 reuses the V23/V24 probe builder exactly: `v23.build_probe_examples()`, which calls `v16.v15.build_digit_probe_examples(n_examples=256, seed=20260610, seq_len=5, base=10)`.
- Probe examples are built once per run from code, serialized in train statistics, and hash-bound as `probe_examples_hash = stable_hash_json(probe_examples)`.
- Probe input tensor is exactly `v23.v16.probe_inputs_tensor(probe_examples)`.
- Flat weight order is exactly the existing subject flat order used by `v23.record_weights_tensor(record)`: a float32 vector of length `SOURCE_WEIGHT_DIM = 345`.
- Subject forward pass is exactly `v23.v16.v15.v10.decoder_v1.subject_forward_flat_batch(weights.reshape(1, -1), inputs)`.
- The activation descriptor is exactly V24 `activation_descriptor_for_weights`: call `v23.hidden_rank1_descriptor_for_weights(weights=flat_weights, probe_examples=probe_examples)`, flatten `descriptor["hbar"]` in layer order followed by `descriptor["xbar"]` in layer order, cast each item to float32, concatenate, and fail closed unless the final length is `ACTIVATION_DESCRIPTOR_DIM = 77`.
- `hidden_rank1_descriptor_for_weights` computes hidden layer inputs/outputs through `v17.hidden_inputs_and_outputs_flat_batch`, then stores per-layer means over the 256 probes. GELU handling is inherited from the existing subject forward implementation; V25 does not reimplement activations.
- Descriptor normalization uses train-only mean/std from accepted train subjects. Std values below `1e-6` are replaced by `1.0`, matching V24 `safe_mean_std`; the zero-std count is hash-bound in train statistics.
- Descriptor row ordering for Jacobian rows is the normalized descriptor vector order: all flattened `hbar` entries, then all flattened `xbar` entries. Dtype is float32; device is CPU unless the script is explicitly extended with a device argument.
- Finite rule: any nonfinite descriptor, Jacobian row, solve output, projection output, candidate metric, or aggregate metric makes that candidate contract-invalid; final/dev phases fail closed.

### Jacobian Rows

- Activation Jacobian `J_activation` is computed by cloning source weights as `flat = weights.detach().clone().to(dtype=torch.float32).reshape(-1).requires_grad_(True)`, recomputing the normalized activation descriptor, and calling `torch.autograd.grad(descriptor_i, flat, retain_graph=True, allow_unused=False)[0]` for every descriptor row in order.
- Source-logit compatibility rows use the same probe inputs and `v23.logits_and_jacobian_for_inputs` convention: flatten logits over all 256 probes in probe order, compute one autograd row per logit, cast to float32.
- Compatibility target is zero change in source logits: `b_source_logits = torch.zeros_like(source_logits)`.
- Row weighting uses square-root weighting so the ridge solve corresponds to a weighted squared loss:

```python
J_aug = torch.cat(
    [J_activation, math.sqrt(compat_weight) * J_source_logits],
    dim=0,
)
b_aug = torch.cat(
    [activation_delta, torch.zeros_like(source_logits)],
    dim=0,
)
```

- `compat_weight = 0.0` omits source-logit rows entirely rather than appending zero-weight rows.
- Row order is always activation rows first, then source-logit rows.
- V25 uses logits rather than BCE residuals for compatibility because this preserves the local first-order interpretation and avoids target-label leakage into source preservation rows.

### Projection Rules

- Component specs and layer shapes are inherited from `v17.LAYER_COMPONENT_SPECS` through V23 `hidden_layer_specs(layer_index)`.
- Hidden layer indices are exactly `[0, 1, 2, 3, 4]`; output-layer indices are flat positions `336:345`.
- `projection="none"` keeps the full closed-form delta but applies a global norm cap.
- `projection="rank1"` projects each hidden layer weight delta matrix to rank 1 by SVD and keeps the paired bias delta unchanged for that layer. Output-layer deltas are zeroed for proof-critical rank1 projections so hidden component edits carry the claim.
- SVD sign canonicalization: for a rank-1 projection `s[0] * outer(u[:,0], vh[0])`, if the largest-absolute entry of `u[:,0]` is negative, multiply both `u[:,0]` and `vh[0]` by `-1` before reconstruction. This makes serialized audits stable.
- `projection="spectral_rank4"` projects the full 345-dimensional delta onto a train-only basis of rank 4 and applies the global norm cap.
- `projection="rank1_spectral_rank4"` applies per-hidden-layer rank1 projection first, then projects the resulting full vector onto the train-only spectral basis.
- Bias handling: hidden biases are preserved in `rank1`; output bias is zeroed when output-layer deltas are excluded; spectral projections include all 345 dimensions unless composed after rank1 output zeroing.
- Norm cap is `median_train_teacher_delta_norm * norm_cap_multiplier`, where `norm_cap_multiplier` is in `[0.25, 0.5, 1.0]` if added in an amended grid. The initial V25 grid fixes `norm_cap_multiplier = 0.5` to keep compute bounded.

### Train-Only Spectral Basis

- Basis source is train-only closed-form unprojected Jacobian deltas computed for train subjects and all non-identity source-target directions using fixed basis settings `ridge_lambda=1e-4`, `compat_weight=0.10`, `projection="none"`, and `norm_cap_multiplier=0.5`.
- Grid-specific unprojected deltas do not contribute to the basis. This avoids changing the basis across candidates and prevents a candidate from indirectly selecting a representation basis.
- Development and final subjects never contribute to the basis, normalization statistics, norm caps, augmentation, or candidate grid construction.
- Deltas are centered by train-only mean and decomposed with `torch.linalg.svd(centered, full_matrices=False)`.
- Rank is fixed to 4 for V25 initial grid.
- Basis audit records: source record count, base settings, rank, centered-delta SHA-256, augmented-delta SHA-256, basis SHA-256, explained variance per component, and train pool SHA-256.
- Augmentations are train-only and basis-only. They affect the spectral basis and spectral norm cap, but never create evaluation records.
- Augmentation order is deterministic:
  1. Original centered deltas sorted by `(source_behavior, target_behavior, subject_id)`.
  2. Sign-flipped centered deltas in the same order.
  3. Same-source-pattern MixUp rows. For each source behavior and target behavior, sort rows by `subject_id`; pair row `i` with row `(i + 1) mod n`; emit `0.2 * row_i + 0.8 * row_j`. If `n < 2`, emit no MixUp row and record `mixup_skipped_small_group=True`.
  4. Gaussian jitter rows. Seed each row with `stable_hash_json({"scope": "v25_basis_jitter", "source": source, "target": target, "subject_id": subject_id})` reduced to 32 bits; jitter std is `0.01 * median_abs_original_centered_delta`.
- Augmented deltas are tagged with `augmentation_type` and hash-bound in basis audit.

## Inner Validation Selection

- Train/development split is fixed by pool role only. Inner validation uses a deterministic subset of the development pool; train subjects are used only for train statistics, basis, norm caps, control precomputation, and optional source-pool audits.
- For each rung budget `B`, subjects per source behavior are `B // (len(PATTERNS) * (len(PATTERNS) - 1))`. With four patterns, budgets 24/72/156 correspond to 2/6/13 subjects per source behavior.
- Balanced subject selection per rung: group development subjects by source behavior, sort by `subject_id`, then take the first `subjects_per_behavior` from each behavior. Each selected source subject is evaluated against all three non-identity target behaviors.
- Candidate invalidity is separate from proof-gate failure. `invalid=True` only for contract failures: wrong record count, missing controls/control count mismatch, nonfinite values, exception, forbidden final raw access, or hash/schema mismatch. Low target rate, weak margins, or failed proof gates are diagnostics and ranking inputs, not invalidity.
- Invalid candidates are ranked after all valid candidates. If all candidates in a rung are invalid, the run stops and writes an invalid development result without final access.
- Exact ranking tuple for survivor selection, sorted ascending:

```python
ranking_tuple = (
    bool(candidate["invalid"]),
    -float(candidate["target_prediction_rate"]),
    -float(candidate["pareto_undominated_rate"]),
    -float(candidate["mean_target_margin"]),
    -float(candidate["mean_matched_minus_best_control_target_margin"]),
    -float(candidate["mean_matched_minus_shuffled_signature_target_margin"]),
    int(candidate["proof_gate_failure_count"]),
    str(candidate["config_hash"]),
)
```

- Rung 0 keeps the first 16 configs by ranking tuple, rung 1 keeps the first 4, and rung 2 selects the first 1.
- The selected config audit records full grid hash/count, evaluated subset hash, each rung survivor list, ranking tuples, invalidity reasons, proof-gate failures, and selected config hash.

## Proof Gates

Development passes only if all are true on the 288 development records:

- Aggregate target prediction rate >= `0.85`.
- Aggregate individual all-gate pass rate >= `0.85`.
- Aggregate Pareto-undominated rate >= `0.85`.
- Mean matched target margin >= `0.25`.
- Mean matched minus best proof-critical control target margin >= `0.02`.
- Mean matched minus no-signature target margin >= `0.02`.
- Mean matched minus no-signature-trained target margin >= `0.02`.
- Mean matched minus shuffled-signature target margin >= `0.05`.
- Mean matched minus V21/V22/V23 target margin >= `0.02`.
- Each direction target prediction rate >= `0.65`.
- Each direction Pareto-undominated rate >= `0.75`.
- Each direction target margin >= `0.15`.

## Controls

Proof-critical controls:

- `no_signature_ablation`
- `no_signature_trained`
- `source_behavior_target_ablation`
- `shuffled_signature`
- `v21_baseline`
- `v22_baseline`
- `v23_baseline`
- `closed_form_unprojected_jacobian`
- `rank1_random_direction`
- `spectral_basis_random_coefficients`
- `contrastive_weight_arithmetic`

Diagnostic-only controls:

- `nearest_train_delta`
- `teacher_oracle_delta`
- `target_only_no_source_compat`
- `activation_only_no_weight_projection`

### Exact Control Algorithms

- Each matched record must have exactly `EXPECTED_CONTROLS_PER_RECORD = 34` controls: 11 proof-critical named controls, 4 diagnostic-only named controls, and 19 random matched-norm controls.
- Random controls are named `random_matched_norm_00` through `random_matched_norm_18`. Seed is `stable_hash_json({"scope": "v25_random_control", "record_id": record_id, "index": i, "selected_config_hash": config_hash})` reduced to a 32-bit torch seed. Each random delta is sampled from `torch.randn(SOURCE_WEIGHT_DIM)`, projected with the same projection type as the matched candidate, then rescaled to the matched delta norm with epsilon `1e-8`.
- `no_signature_ablation`: replace target activation delta with all zeros while keeping source compatibility rows and projection identical.
- `no_signature_trained`: use the matched selected config, but replace every activation descriptor input with zeros after train normalization. Because V25 has no learned proof editor, this control does not run independent hyperparameter selection; it uses the selected `ridge_lambda`, `compat_weight`, and `projection` and solves with `activation_delta=zeros_like(activation_delta)`. Its train-only audit records the selected config hash and zero-descriptor norm hash.
- `source_behavior_target_ablation`: set activation target to the source descriptor rather than target descriptor.
- `shuffled_signature`: replace the target descriptor only; source weights, source descriptor, source compatibility rows, ridge, compat, projection, and norm cap remain matched. Derangement is within each source behavior group across all non-identity target requests for the current split/rung. Build rows sorted by `(source_behavior, target_behavior, subject_id)`, seed a torch generator with `stable_hash_json({"scope": "v25_shuffled_signature_order", "split": split_name, "rung_index": rung_index, "source": source_behavior, "config_hash": config_hash})` reduced to 32 bits, sample permutations until no row keeps its original target descriptor, with max 128 attempts. If group size is `<2` or no derangement is found, the candidate is contract-invalid for that rung rather than falling back to identity.
- `v21_baseline`, `v22_baseline`, `v23_baseline`: recompute prior baselines from source weights using their existing scripts/functions and the same train statistics/probe examples available to V25; prior final raw paths remain forbidden.
- `closed_form_unprojected_jacobian`: solve the same Jacobian ridge system with `projection="none"` and the same ridge/compat values.
- `rank1_random_direction`: replace `activation_delta` with a norm-matched random descriptor-space vector before rank1 projection.
- `spectral_basis_random_coefficients`: sample random coefficients in the rank-4 train spectral basis, norm-match to the matched spectral projection, and evaluate.
- `contrastive_weight_arithmetic`: compute train-only mean deltas from the same fixed basis-source closed-form deltas used for spectral basis. For each ordered pair `(source, target)`, `mean_delta[source,target]` is the mean fixed-setting unprojected train delta over train subjects with that source. The contrastive direction is `mean_delta[source,target] - mean_delta[target,source]`. Alpha grid is `[0.25, 0.5, 1.0, 2.0]`; select alpha once using train-only leave-one-source-subject-out target prediction rate, then mean target margin, then lower compatible-source-output MSE, then smaller alpha. The selected alpha table is hash-bound and reused unchanged on development/final.
- `nearest_train_delta`: diagnostic nearest train closed-form delta by cosine similarity of source descriptor to train source descriptors with identical target label.
- `teacher_oracle_delta`: diagnostic train-protocol teacher optimizer on the same source-target pair; never proof-critical.
- `target_only_no_source_compat`: diagnostic solve with compatibility rows omitted.
- `activation_only_no_weight_projection`: diagnostic activation ridge solve before projection/norm cap.

### Metrics and Pareto Rule

- Target prediction uses the existing four-pattern behavior classifier/evaluator from V24/V23. A record counts as target-predicted only when edited heldout behavior label equals requested target.
- Target margin is target logit/probability margin over the best non-target behavior under the existing evaluator.
- Compatible-source-output MSE is mean squared difference between edited and source logits on probes compatible with the source behavior, using V23 `compatible_probe_mask` semantics.
- Per-record individual pass requires target prediction, target margin >= `0.15`, Pareto-undominated against proof-critical controls, every named control target-margin advantage >= `0.02` except shuffled signature >= `0.05`, and compatible-source-output MSE no worse than best proof-critical control by more than `0.05`.
- Pareto dominance compares proof-critical controls only. A control dominates matched if it has target margin >= matched target margin, compatible-source-output MSE <= matched compatible MSE, and at least one strict improvement greater than `1e-8`.
- Best-control target margin is the maximum target margin over proof-critical controls, not diagnostic controls.

## Final Redaction and Authorization

- V25 defines new scopes:
  - `SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v25_source_pool"`
  - `SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v25_source_pool_construction"`
  - `FINAL_REDACTED_SCOPE = "redacted_final_functional_weight_editing_v25_source_pool_audit_surface_only"`
  - `DEVELOPMENT_SCOPE = "four_behavior_functional_weight_editing_v25_jacobian_rank1_editor_development"`
  - `FINAL_SCOPE = "four_behavior_functional_weight_editing_v25_jacobian_rank1_editor_final"`
- Prior final raw guards include every known prior final path from V15 through V24 plus V25 `final_subjects.json`.
- Recursive forbidden key terms are: `records`, `weights`, `signature`, `subject_id`, `seed`, `train_info`, `support_margin`, `heldout_margin`, `logits`, `descriptor`, `jacobian`, `delta`.
- Final redacted allowed top-level keys are exactly: `behavior_suite_hashes`, `candidate_pool_summary_hash`, `claim_scope`, `config_hash`, `pool`, `pool_file_sha256`, `pool_redacted_payload_sha256`, `summary`, `summary_payload_sha256`.
- Final redacted allowed summary keys are exactly: `accepted_counts_by_behavior`, `max_selected_train_vs_heldout_overlap_count`.
- Combined final-summary allowed keys are exactly: `accepted_counts_by_behavior`, `pool_file_sha256`, `pool_redacted_payload_sha256`.
- Final authorization payload binds: plan hash, script hash, helper-test hash, constants hash, train/development pool hashes, final redacted hash, combined audit hash, external development result file hash, selected config hash, selected model/result hash, reviewer authorization hash, reviewer confidence, full grid count/hash, evaluated subset hash, and inner-validation selection hash.
- If development result has `passed=False`, final authorization construction raises `ValueError` and final raw remains sealed.

## Compute, Cache, and Resume Rules

- Candidate-level parallelism uses `min(os.cpu_count() or 1, candidate_count, max_workers)` workers. Default `max_workers` is `None`, meaning use available CPU but never more workers than candidates.
- Jacobian cache key is `stable_hash_json({"subject_id": subject_id, "source": source, "probe_examples_hash": probe_examples_hash, "descriptor_norm_hash": descriptor_norm_hash, "script_sha256": script_sha256})`.
- Cache stores source descriptor, normalized descriptor, source logits, activation Jacobian, source-logit Jacobian, finite audit, and elapsed seconds. It never stores or reads final raw details in development.
- Rung cache reuse: compute a subject/source Jacobian once per rung record and reuse for all candidate configs in that rung.
- Selected development evaluation reuses train statistics and selected config only; it recomputes development Jacobians with cache events rather than reading inner-validation candidate outputs.
- Progress events required for long-running cache phases:
  - `jacobian_cache_start`
  - `jacobian_cache_record_completed`
  - `jacobian_cache_completed`
  - `inner_validation_candidate_start`
  - `inner_validation_candidate_completed`
  - `inner_validation_rung_completed`
  - `development_evaluation_record_completed`
- A partial run can resume only when plan hash, script hash, pool hashes, full grid hash, completed candidate hashes, and cache key schema hash match exactly. Any mismatch archives the partial output and starts a fresh run directory.
- A run is stopped as wasting compute only if one of these is true: traceback, nonfinite loss/residual, contract invalidity, no progress-log growth for 10 minutes while aggregate CPU < 25%, memory pressure causing swap exhaustion, or reviewer 5/5 stop recommendation.

## Task 1: Scaffold V25 Script and Guards

**Files:**
- Create: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Create: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing identity and final-raw guard tests**

```python
def test_v25_identity_scopes_and_fresh_seeds() -> None:
    assert v25.EDITOR_METHOD == "jacobian_rank1_editor_v25"
    assert v25.POOL_CONFIGS["train"]["base_seed"] == 126400000
    assert v25.POOL_CONFIGS["development"]["base_seed"] == 127400000
    assert v25.POOL_CONFIGS["final"]["base_seed"] == 128400000
    assert "v25" in str(v25.DEFAULT_POOL_DIR)
    assert "v25" in str(v25.DEFAULT_OUTPUT_DIR)


def test_v25_final_raw_guard_rejects_final_subjects_path() -> None:
    with pytest.raises(ValueError, match="final raw"):
        v25.assert_no_forbidden_final_raw_paths([v25.V25_FINAL_RAW])
```

- [ ] **Step 2: Run red test**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_identity_scopes_and_fresh_seeds model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py::test_v25_final_raw_guard_rejects_final_subjects_path -q
```

Expected: FAIL because the V25 module does not exist.

- [ ] **Step 3: Implement minimal constants and guard**

```python
EDITOR_METHOD = "jacobian_rank1_editor_v25"
DEFAULT_POOL_DIR = REPO_ROOT / "runs/four_behavior_functional_weight_editing_v25_pools"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor"
V25_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
POOL_CONFIGS = {
    "train": {"base_seed": 126400000},
    "development": {"base_seed": 127400000},
    "final": {"base_seed": 128400000},
}


def assert_no_forbidden_final_raw_paths(paths: Sequence[Path | str]) -> None:
    forbidden = V25_FINAL_RAW.resolve()
    for path in paths:
        if Path(path).resolve() == forbidden:
            raise ValueError("final raw path access is forbidden")
```

- [ ] **Step 4: Run green test**

Run the same two tests. Expected: PASS.

## Task 2: Jacobian Ridge Solver

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`

- [ ] **Step 1: Write failing synthetic solver test**

```python
def test_v25_jacobian_ridge_edit_reduces_linear_residual() -> None:
    jacobian = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    target_delta = torch.tensor([2.0, -4.0])
    delta = v25.solve_jacobian_ridge_edit(jacobian, target_delta, ridge_lambda=1e-6)
    before = torch.linalg.norm(target_delta)
    after = torch.linalg.norm(target_delta - jacobian @ delta)
    assert after < before * 1e-3
```

- [ ] **Step 2: Run red test**

Expected: FAIL because `solve_jacobian_ridge_edit` does not exist.

- [ ] **Step 3: Implement solver with finite checks**

```python
def solve_jacobian_ridge_edit(
    jacobian: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    ridge_lambda: float,
) -> torch.Tensor:
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be rank-2")
    if target_delta.ndim != 1:
        raise ValueError("target_delta must be rank-1")
    if jacobian.shape[0] != target_delta.shape[0]:
        raise ValueError("jacobian row count must match target_delta")
    eye = torch.eye(jacobian.shape[0], dtype=jacobian.dtype, device=jacobian.device)
    gram = jacobian @ jacobian.T + float(ridge_lambda) * eye
    delta = jacobian.T @ torch.linalg.solve(gram, target_delta)
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite jacobian edit")
    return delta
```

- [ ] **Step 4: Run green test**

Expected: PASS.

## Task 3: Low-Rank and Spectral Projection

**Files:**
- Modify script and tests.

- [ ] **Step 1: Write failing projection tests**

```python
def test_v25_project_matrix_rank1_returns_rank_at_most_one() -> None:
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    projected = v25.project_matrix_rank1(matrix)
    assert projected.shape == matrix.shape
    assert torch.linalg.matrix_rank(projected).item() <= 1


def test_v25_project_to_basis_uses_train_only_basis_columns() -> None:
    delta = torch.tensor([1.0, 2.0, 3.0])
    basis = torch.eye(3)[:, :2]
    projected = v25.project_to_basis(delta, basis)
    assert torch.allclose(projected, torch.tensor([1.0, 2.0, 0.0]))
```

- [ ] **Step 2: Run red tests**

Expected: FAIL because projection helpers do not exist.

- [ ] **Step 3: Implement projections**

```python
def project_matrix_rank1(matrix: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return s[0] * torch.outer(u[:, 0], vh[0])


def project_to_basis(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2 or basis.shape[0] != delta.numel():
        raise ValueError("basis shape must be [delta_dim, rank]")
    coeff = basis.T @ delta.flatten()
    return basis @ coeff
```

- [ ] **Step 4: Run green tests**

Expected: PASS.

## Task 4: Train-Only Spectral Basis and Augmentations

**Files:**
- Modify script and tests.

- [ ] **Step 1: Write failing basis test**

```python
def test_v25_train_spectral_basis_is_orthonormal_and_hash_bound() -> None:
    deltas = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
    basis, audit = v25.compute_train_spectral_basis(deltas, rank=2)
    assert basis.shape == (3, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-5)
    assert audit["rank"] == 2
    assert len(audit["basis_sha256"]) == 64
```

- [ ] **Step 2: Implement `compute_train_spectral_basis`**

```python
def compute_train_spectral_basis(deltas: torch.Tensor, *, rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
    centered = deltas - deltas.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    audit = {
        "basis_sha256": stable_hash_json(basis.detach().cpu().tolist()),
        "delta_count": int(deltas.shape[0]),
        "rank": int(rank),
    }
    return basis, audit
```

- [ ] **Step 3: Add train-only augmentation config**

Use only train split deltas for sign flips, same-function MixUp, and seed jitter. Never mix development or final records into basis construction.

- [ ] **Step 4: Verify**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: PASS.

## Task 5: Inner Validation Grid

**Files:**
- Modify script and tests.

- [ ] **Step 1: Candidate grid**

Use this exact initial grid:

```python
RIDGE_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
COMPAT_WEIGHT_GRID = [0.0, 0.05, 0.10, 0.20]
PROJECTION_GRID = ["none", "rank1", "spectral_rank4", "rank1_spectral_rank4"]
```

Total full grid count: `4 * 4 * 4 = 64`.

- [ ] **Step 2: Successive halving**

Evaluate 64 candidates on 24 records, keep 16; evaluate 16 on 72 records, keep 4; evaluate 4 on 156 records, keep 1.

- [ ] **Step 3: Required tests**

```python
def test_v25_full_grid_count_and_hash_are_stable() -> None:
    grid = v25.build_v25_config_grid()
    assert len(grid) == 64
    assert v25.stable_hash_json(grid) == v25.V25_FULL_GRID_SHA256
```

After first red run, fill `V25_FULL_GRID_SHA256` with the computed hash and never change it without a new plan.

## Task 6: Development Run and Logging

**Files:**
- Modify script and tests.

- [ ] **Step 1: Progress event requirements**

Every phase must log JSONL events:

```python
development_start
source_payloads_loaded
source_pool_contract_validated
inner_validation_start
inner_validation_candidate_completed
inner_validation_rung_completed
selected_model_training_start
selected_model_training_completed
development_evaluation_record_completed
development_evaluation_completed
development_results_written
```

- [ ] **Step 2: Long-running monitor command**

When launching development, also launch an external monitor that records:

```json
{
  "alive": true,
  "aggregate_cpu_percent": 0.0,
  "worker_count": 0,
  "inner_line_count": 0,
  "development_line_count": 0,
  "candidate_completed": 0,
  "evaluation_completed": 0
}
```

- [ ] **Step 3: Result artifact writer**

Reuse the V24 provenance pattern:

```python
result["development_results_payload_sha256"] = stable_hash_json(result)
write_json_atomic(output_path, result)
record_progress_event(
    progress_log_path,
    event="development_results_written",
    extra={
        "development_results_file_sha256": sha256_file(output_path),
        "development_results_payload_sha256": result["development_results_payload_sha256"],
        "passed": bool(result["passed"]),
    },
)
```

- [ ] **Step 4: Verification**

Run:

```bash
python -m py_compile model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Expected: compile passes and pytest passes.

## Task 7: Development Execution

**Files:**
- Output only under V25 run directories.

- [ ] **Step 1: Generate pools**

Run:

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py --phase generate-pools --summary-only-stdout
```

Expected: train/development/final pools exist; final raw is only hashed and never opened by development code.

- [ ] **Step 2: Reviewer checkpoint**

Send source pool hashes, redacted final audit hash, contract validation result, and plan hash to reviewer. Continue only on 5/5.

- [ ] **Step 3: Run development with monitor**

Run:

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py --phase development --summary-only-stdout
```

Start external monitor writing:

```text
runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/monitor_progress_v2.jsonl
```

- [ ] **Step 4: Reviewer checkpoints**

Send reviewer packages after:

- source pool generation
- each inner-validation rung completion
- selected training completion
- development result completion

Continue only when reviewer confidence is 5/5.

## Expected Interpretation

- If development passes, do not open final. First build final authorization from external result-file hash, test hash, script hash, plan hash, and reviewer 5/5 authorization.
- If development fails, record a negative result and keep final sealed.
- If V25 improves target rate but remains below gates, treat it as evidence that first-order low-rank geometry helps but is insufficient for proof.
- If V25 does not improve over V24, prioritize either richer subject diversity or direct contrastive weight arithmetic rather than larger hypernetworks.

## Self-Review

- Spec coverage: includes fresh seeds, sealed final, logs, reviewer checkpoints, literature support, proof gates, controls, provenance, tests, and long-running monitoring.
- Placeholder scan: no TBD/TODO/later placeholders.
- Type consistency: V25 helper names are consistent across tests and implementation tasks.
