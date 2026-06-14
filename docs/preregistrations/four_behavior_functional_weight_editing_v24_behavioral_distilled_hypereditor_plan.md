# V24 Behavioral-Distilled Hypereditor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a fixed probe set plus train-only behavioral distillation can produce signature-conditioned weight edits that outperform no-signature, shuffled-signature, and prior-editor controls on fresh development subjects.

**Architecture:** V24 replaces V23's hand-routed sparse subspace with a two-stage train-only pipeline: first optimize per-source/per-target teacher deltas against behavioral probe losses, then train a compact hypereditor to predict those deltas from source weights, source signature, fixed-probe activation descriptors, and target behavior. Development and final pools are fresh and sealed; development gates decide whether final is authorized.

**Tech Stack:** Python, PyTorch, existing four-behavior source-pool tooling, pytest, JSONL progress logs.

---

## Status From V23

V23 is a negative development result. The run completed 288/288 development records and failed 98 preregistered gates: aggregate target prediction rate was 0.4792, individual all-gate pass rate was 0.0, Pareto-undominated rate was 0.2743, and best-control margin advantage was -0.1610. The reviewer rated the result 5/5 as clean negative/inconclusive evidence and explicitly did not authorize final evaluation. Therefore V24 must not inspect V23 final raw data and must not present V23 as support for the hypothesis.

## Literature Basis

1. Herrmann, Faccio, and Schmidhuber, **Learning Useful Representations of Recurrent Neural Network Weight Matrices**, arXiv:2403.11998, argues for a "functionalist" route: interrogating networks through probing inputs can produce richer representations for predicting behavior than weight-only mechanistic encodings. V24 therefore uses fixed probe-set behavioral responses and activation descriptors as first-class inputs, not only static weights/signatures. Source: https://arxiv.org/abs/2403.11998

2. Meynent et al., **Structure Is Not Enough: Leveraging Behavior for Neural Network Weight Reconstruction**, arXiv:2503.17138, shows structural reconstruction losses can preserve weight distance while degrading functional performance, and that adding behavioral loss improves reconstruction/generation. V24 therefore trains teacher edits and the hypereditor with behavioral losses on probe inputs, not just delta MSE. Source: https://arxiv.org/abs/2503.17138

3. Horwitz et al., **Learning on Model Weights using Tree Experts**, arXiv:2410.13569, emphasizes nuisance variation in weight-space learning and the value of within-tree structure. V24 uses fresh train/development/final pools from the same subject architecture and explicitly includes no-signature and shuffled-signature controls to test whether the signature carries information beyond within-family nuisance structure. Source: https://arxiv.org/abs/2410.13569

4. **Steer2Edit: From Activation Steering to Component-Level Editing**, arXiv:2602.09870, reports that fixed global steering can create poor control/utility tradeoffs because behaviors are component-heterogeneous. V24 allows teacher/hypereditor deltas across the compact full source-weight vector, while recording output-layer and hidden-layer contribution diagnostics. Source: https://arxiv.org/html/2602.09870v2

5. **From Weights to Activations: Is Steering the Next Frontier of Adaptation?**, arXiv:2604.14090, frames steering as an adaptation method that should be judged by functional criteria, not only interpretability narratives. V24 keeps preregistered functional gates and control advantages as the decision rule. Source: https://arxiv.org/html/2604.14090v1

6. **The Universal Weight Subspace Hypothesis**, arXiv:2512.05117, motivates low-dimensional shared parametric structure across related models. V24 will log the effective rank and norm concentration of teacher deltas, but it will not assume a sparse global direction is sufficient because V23 failed that premise. Source: https://arxiv.org/abs/2512.05117

These papers motivate the V24 pivot; they do not prove that V24 should work. In particular, none of the cited papers validates this exact behavioral-distilled full-weight hypereditor on this four-behavior source-model setting. V24 remains an exploratory development experiment with strict controls and preregistered failure handling.

## Core Hypothesis

For a held-out source subject and a requested target behavior, a hypereditor trained only on train-pool teacher edits can use the fixed probe signature/activation descriptor to produce a functional weight delta that:

- increases target-behavior margin and target prediction on held-out development subjects,
- preserves compatibility on source behavior probe inputs better than naive target-only edits,
- beats no-signature, shuffled-signature, source-signature, and prior V21/V22/V23 controls on matched target-margin and Pareto metrics.

## Files

- Create: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Create: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py`
- Create: `docs/preregistrations/four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.md`
- Reuse read-only patterns from:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor.py`
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork.py`
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v21_behavioral_probe_residual_output_editor.py`

## Exact Method Preregistration

### Probe And Descriptor Construction

- `build_probe_examples()` is inherited from V23 and fixed before any V24 pool generation.
- Source signatures are the 560-dimensional accepted-record signatures already produced by source-pool generation.
- Activation descriptor is exactly 77 float values:
  - `hbar` means from V23 `hidden_rank1_descriptor_for_weights`: five hidden layers, each length 8, total 40.
  - `xbar` means from V23 `hidden_rank1_descriptor_for_weights`: layer input lengths `[5, 8, 8, 8, 8]`, total 37.
  - descriptor layout is `[hbar_layer_0..4, xbar_layer_0..4]`, flattened in layer order.
- Normalization statistics for weights, signatures, descriptors, and teacher deltas are computed from V24 inner-train subjects only. Development subjects never affect normalization or config selection.

### Inner Split

- V24 train pool has 64 accepted subjects per behavior.
- Inner split is deterministic by `stable_hash_json({"scope":"v24_inner_split","behavior":behavior,"subject_id":subject_id})`.
- First 51 subjects per behavior are inner-train; remaining 13 per behavior are inner-validation.
- The split hash is `stable_hash_json({"scope":"v24_inner_split_hash","inner_train_ids":[...],"inner_validation_ids":[...]})` and is written to every result artifact.

### Teacher Edit Optimization

- Teacher deltas are optimized from an all-zero 345-dimensional delta.
- Optimizer: `torch.optim.Adam`.
- Betas: `(0.9, 0.999)`.
- Epsilon: `1e-8`.
- Weight decay: `0.0`.
- Gradient clip: global norm `5.0`.
- Teacher configs are exactly:
  - `steps in [40, 80]`
  - `lr in [0.03, 0.01]`
  - `l2_weight = 0.001`
  - `source_compat_weight in [0.25, 0.5, 1.0]`
- Teacher loss reduction is mean over probe rows:
  - `target_bce`: BCE-with-logits against target-behavior labels on fixed probe rows.
  - `source_compat_mse`: MSE between edited and original logits on rows where the source behavior label is positive.
  - `l2`: mean squared delta.
- Teacher objective: `target_bce + source_compat_weight * source_compat_mse + l2_weight * delta_mse`.
- Teacher selection uses final optimizer step only; no best-of-steps or development feedback.
- Nonfinite loss, nonfinite delta, or optimizer exception marks the teacher invalid and logs a count-only invalid event.
- Teacher jobs are keyed by `(teacher_config_hash, source_subject_id, target_behavior)` but progress logs record only counts and config hashes, not subject IDs.

### Hypereditor Architecture And Training

- Input dimension is exactly `345 + 560 + 77 + 4 + 4 + 12 = 1002`.
- Feature layout is exactly:
  - normalized source weights: 345
  - normalized source signature: 560
  - normalized activation descriptor: 77
  - source behavior one-hot: 4
  - target behavior one-hot: 4
  - ordered source-target pair one-hot: 12
- Model: MLP with hidden dims `[768, 768, 384]`.
- Each hidden block is `Linear -> GELU -> LayerNorm`.
- Heads:
  - `delta_head: Linear(384, 345)`
  - `scale_head: Linear(384, 1)`, scale is `2.0 * sigmoid(raw_scale)`.
- Initialization seed: `20260624 + config_index`.
- Optimizer: `torch.optim.AdamW`.
- Betas: `(0.9, 0.999)`.
- Epsilon: `1e-8`.
- Weight decay: `1e-4`.
- Learning rate: `1e-3`.
- Batch size: 64 teacher records.
- Batch order is deterministic with replacement from `torch.Generator().manual_seed(20260724 + config_index)`.
- Hypereditor step configs are exactly `steps in [800, 1600]`.
- Hypereditor loss:
  - `delta_mse_weight in [0.25, 1.0]`
  - `behavior_weight in [0.5, 1.0, 2.0]`
  - `compat_weight in [0.25, 0.5, 1.0]`
  - `l2_weight = 0.0001`
  - objective: `delta_mse_weight * delta_mse + behavior_weight * target_bce + compat_weight * source_compat_mse + l2_weight * pred_delta_mse`
- Checkpoint selection uses final training step only; no best validation checkpoint.

### Inner-Validation Config Protocol

- Full grid count: `2 teacher_steps * 2 teacher_lr * 3 source_compat_weight * 2 hypereditor_steps * 2 delta_mse_weight * 3 behavior_weight * 3 compat_weight = 432`.
- Evaluated subset count: exactly 48 configs, selected before metrics by deterministic stratification over `(teacher_steps, teacher_lr, hypereditor_steps, delta_mse_weight)`.
- Each stratum gets `48 // 16 = 3` configs. Within stratum order is:
  - `stable_hash_json({"scope":"v24_evaluated_config_subset","plan_sha256":PLAN_SHA256,"config_hash":config_hash})`
- Successive-halving rungs are exact:
  - rung 0: 48 configs, 2 inner-validation subjects per behavior, 24 records, keep 12
  - rung 1: 12 configs, 6 inner-validation subjects per behavior, 72 records, keep 3
  - rung 2: 3 configs, 13 inner-validation subjects per behavior, 156 records, keep 1
- Ranking key is exactly:

```python
(
    bool(candidate["invalid"]),
    -candidate["target_prediction_rate"],
    -candidate["pareto_undominated_rate"],
    -candidate["mean_matched_minus_best_control_target_margin"],
    -candidate["mean_matched_minus_shuffled_signature_target_margin"],
    -candidate["mean_target_margin"],
    candidate["mean_compatible_source_mse"],
    candidate["config_hash"],
)
```

- Invalid rules:
  - wrong record count,
  - missing any proof-critical control,
  - any nonfinite aggregate metric,
  - any exception in teacher generation, hypereditor training, or evaluation.
- The output artifacts must include full grid hash, evaluated subset hash, rung candidate hashes, survivor hashes, and selected config hash.

## Exact Experimental Controls

V24 must include, per development record:

- `no_edit`
- `matched_signature_behavioral_hypereditor_v24`
- `no_signature_ablation_behavioral_hypereditor_v24`: same trained matched hypereditor, but signature and activation descriptor inputs are zeroed at inference.
- `no_signature_trained_behavioral_hypereditor_v24`: separately trained model where signature and activation descriptor inputs are zeroed during train and evaluation.
- `source_behavior_target_ablation_behavioral_hypereditor_v24`: same trained matched hypereditor, but target behavior and pair one-hot are replaced with the source behavior/self-pair zero vector.
- `shuffled_signature_behavioral_hypereditor_v24`: same trained matched hypereditor, but signature and descriptor are replaced by the deterministic next subject in the same `(source_behavior, target_behavior)` evaluation group ordered by stable hash.
- `nearest_train_target_signature_behavioral_hypereditor_v24`
- `teacher_oracle_support_optimizer_train_protocol_v24` only as a ceiling diagnostic, not as a proof baseline
- `v21_behavioral_probe_residual_output_editor_recomputed`
- `v22_component_activation_rank1_editor_recomputed`
- `v23_probe_routed_sparse_subspace_editor_recomputed`
- 20 random matched-norm controls

The proof-critical advantage controls are no-signature ablation, no-signature trained, shuffled-signature, source-behavior target ablation, V21, V22, and V23. Passing requires matched V24 to beat these controls, not merely to beat no edit.

`nearest_train_target_signature_behavioral_hypereditor_v24` uses the train subject of the requested target behavior whose activation descriptor has minimum Euclidean distance to the source descriptor. It is a diagnostic for retrieval-like explanations and does not count as a proof-critical signature ablation.

Random matched-norm controls are generated by sampling `torch.randn(345)` with seed:

```python
stable_hash_json({
    "scope": "v24_random_control",
    "subject_id": source_subject_id,
    "source": source_behavior,
    "target": target_behavior,
    "index": random_index,
})
```

The random delta is normalized to the matched V24 delta norm. If the matched norm is zero, the random control delta is exactly zero and marked `matched_norm_zero=True`.

Expected exact controls per record: 30 controls excluding the matched edit:

- 10 named controls listed above except `matched_signature_behavioral_hypereditor_v24`
- 20 random controls

Prior-editor controls are recomputed from V24 train subjects and the current V24 held-out request only. V24 may import helper functions from V21/V22/V23, but it must not read prior V21/V22/V23 pools, stats artifacts, result files, development artifacts, or final raw files.

## Gates

Use V23's strict aggregate/direction gates unless the implementation review identifies a metric schema incompatibility before development is run:

- expected records: 288
- expected controls per record: exactly 30
- random controls per record: 20
- aggregate target prediction rate >= 0.85
- aggregate individual all-gate pass rate >= 0.85
- aggregate Pareto-undominated rate >= 0.85
- mean target margin >= 0.25
- aggregate best-control target-margin advantage >= 0.02
- aggregate no-signature target-margin advantage >= 0.02
- aggregate shuffled-signature target-margin advantage >= 0.05
- aggregate source-signature target-margin advantage >= 0.02
- aggregate V21/V22/V23 target-margin advantage >= 0.02
- per-direction target prediction rate >= 0.65
- per-direction Pareto-undominated rate >= 0.75
- per-direction target margin >= 0.15

If development fails, next action is `log_negative_development_result_do_not_open_final_raw`. If development passes, final remains blocked until reviewer returns 5/5 on development artifacts and explicitly authorizes final.

## Data-Leakage Rules

- Generate fresh V24 pools with seeds:
  - train base seed: `123400000`
  - development base seed: `124400000`
  - final base seed: `125400000`
- Do not read any raw `runs/**/final_subjects.json` except through the final phase after reviewer authorization.
- Development code may read V24 train and development payloads plus V24 final redacted audit only.
- V24 may import V21/V22/V23 code as baselines but must not reuse V21/V22/V23 final raw payloads.
- V24 must not read prior V21/V22/V23 train, development, stats, or result artifacts for controls. All prior-editor controls are recomputed from V24 train subjects and current V24 requests.
- Inner validation must split only the V24 train pool.
- Hyperparameters selected after seeing V23 development are allowed only as V24 design rationale; no V24 hyperparameter may be selected from V24 development metrics.

## Exact Audit Scopes And Allowlists

- `SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v24_source_pool"`
- `SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v24_source_pool_construction"`
- `FINAL_REDACTED_SCOPE = "redacted_final_functional_weight_editing_v24_source_pool_audit_surface_only"`
- `DEVELOPMENT_SCOPE = "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_development"`
- `FINAL_SCOPE = "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor_final"`

Final redacted top-level allowlist:

```python
{
    "behavior_suite_hashes",
    "candidate_pool_summary_hash",
    "claim_scope",
    "config_hash",
    "pool",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
    "summary",
    "summary_payload_sha256",
}
```

Final redacted summary allowlist:

```python
{
    "accepted_counts_by_behavior",
    "max_selected_train_vs_heldout_overlap_count",
}
```

Combined-audit final summary allowlist:

```python
{
    "accepted_counts_by_behavior",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
}
```

Forbidden final-detail keys are inherited from V23/V10 `FORBIDDEN_FINAL_DETAIL_KEYS` and extended to reject top-level or nested keys containing:

```python
{
    "records",
    "weights",
    "signature",
    "subject_id",
    "seed",
    "train_info",
    "support_margin",
    "heldout_margin",
}
```

Hash-bound final authorization must include all fields below:

```python
{
    "plan_sha256": PLAN_SHA256,
    "formal_preregistration_sha256": FORMAL_PREREGISTRATION_SHA256,
    "script_sha256": sha256_file(SCRIPT_PATH),
    "helper_test_sha256": sha256_file(HELPER_TEST_PATH),
    "constants_sha256": stable_hash_json(constants_payload()),
    "development_results_sha256": sha256_file(development_results_path),
    "development_claim_scope": DEVELOPMENT_SCOPE,
    "development_phase": "development",
    "development_passed": True,
    "development_next_action": "run_hash_bound_final_after_reviewer_authorization",
    "editor_method": "behavioral_distilled_hypereditor_v24",
    "train_pool_sha256": sha256_file(train_subjects_path),
    "development_pool_sha256": sha256_file(development_subjects_path),
    "combined_audit_sha256": sha256_file(combined_audit_path),
    "final_redacted_audit_sha256": sha256_file(final_redacted_audit_path),
    "train_statistics_hash": train_statistics_hash,
    "selected_config_hash": selected_config_hash,
    "selected_model_hash": selected_model_hash,
    "inner_validation_selection_hash": inner_validation_selection_hash,
    "inner_validation_evaluated_config_subset_hash": evaluated_config_subset_hash,
    "reviewer_confidence": "5/5",
    "reviewer_authorization_sha256": reviewer_message_sha256,
}
```

At final runtime, the final runner must reconstruct this authorization payload from local artifacts before opening `final_subjects.json`. It must verify exact equality for every field above, including `development_results.json` fields `claim_scope`, `phase`, `passed=true`, `next_action`, `editor_method`, selected config/model hashes, train/development pool hashes, combined audit hash, final redacted audit hash, train statistics hash, helper-test SHA, and formal-preregistration SHA. If any field is absent or differs, the final phase must abort before opening final raw.

## Progress Logging

Long-running phases must append JSONL logs before and during work:

- `source_payloads_loaded`
- `source_pool_contract_validated`
- `teacher_edit_generation_start`
- `teacher_edit_completed` with count-only progress
- `hypereditor_training_start`
- `hypereditor_training_step` every 50 steps with loss scalars
- `inner_validation_start`
- `inner_validation_candidate_completed` with count-only progress
- `development_evaluation_jobs_queued`
- `development_evaluation_record_completed` with `completed_count` and `record_count`
- `development_results_written`

Logs must not include final subject IDs, raw final weights, raw final signatures, or final per-record metrics.

## Task 1: Copy V23 Guard Rails And Fresh V24 Identity

**Files:**
- Create: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Create: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py`

- [ ] **Step 1: Write identity and final-guard tests**

```python
from pathlib import Path

import pytest

import train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor as v24


def test_v24_identity_scopes_and_seeds_are_fresh() -> None:
    assert v24.EDITOR_METHOD == "behavioral_distilled_hypereditor_v24"
    assert "v24" in str(v24.DEFAULT_POOL_DIR)
    assert "v24" in str(v24.DEFAULT_OUTPUT_DIR)
    assert v24.POOL_CONFIGS["train"]["base_seed"] == 123400000
    assert v24.POOL_CONFIGS["development"]["base_seed"] == 124400000
    assert v24.POOL_CONFIGS["final"]["base_seed"] == 125400000
    assert v24.POOL_CONFIGS["train"]["base_seed"] != v24.v23.POOL_CONFIGS["train"]["base_seed"]


def test_v24_final_raw_guard_rejects_all_known_final_subjects_paths() -> None:
    with pytest.raises(ValueError):
        v24.assert_no_forbidden_final_raw_paths([v24.V24_FINAL_RAW])
    with pytest.raises(ValueError):
        v24.assert_no_forbidden_final_raw_paths([v24.v23.V23_FINAL_RAW])
    with pytest.raises(ValueError):
        v24.assert_no_forbidden_final_raw_paths([
            Path("/Users/max/Desktop/muat/runs/other/final_subjects.json")
        ])
```

- [ ] **Step 2: Implement constants and guard imports**

Create the V24 script by copying V23's CLI/pool/guard structure, then change:

```python
EDITOR_METHOD = "behavioral_distilled_hypereditor_v24"
POOL_CONFIGS = {
    "train": {"base_seed": 123400000, "target_accepted_per_behavior": 64, "max_attempts_per_behavior": 128},
    "development": {"base_seed": 124400000, "target_accepted_per_behavior": 24, "max_attempts_per_behavior": 64},
    "final": {"base_seed": 125400000, "target_accepted_per_behavior": 24, "max_attempts_per_behavior": 64},
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v24_pools"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor"
V24_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"
PRIOR_FINAL_RAW_PATHS = {
    v23.V23_FINAL_RAW,
    v23.v22.V22_FINAL_RAW,
    v23.v21.V21_FINAL_RAW,
    v23.v17.V17_FINAL_RAW,
    v23.v16.V16_FINAL_RAW,
    v23.v16.v15.V15_FINAL_RAW,
}
```

- [ ] **Step 3: Run identity tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py -q
```

Expected: the new tests pass.

## Task 2: Teacher Edit Optimizer

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py`

- [ ] **Step 1: Add teacher optimizer shape and determinism tests**

```python
def test_v24_teacher_edit_returns_weight_sized_delta(monkeypatch) -> None:
    source = {
        "behavior": v24.PATTERNS[0],
        "subject_id": "s0",
        "weights": [0.0] * v24.SOURCE_WEIGHT_DIM,
        "signature": [0.0] * v24.SIGNATURE_DIM,
    }
    result = v24.optimize_teacher_edit(
        source,
        target_behavior=v24.PATTERNS[1],
        config=v24.TeacherEditConfig(steps=2, lr=0.01, l2_weight=0.001, source_compat_weight=0.1),
        probe_examples=v24.build_probe_examples(),
    )
    assert result["delta"].shape == (v24.SOURCE_WEIGHT_DIM,)
    assert result["target_behavior"] == v24.PATTERNS[1]
    assert result["source_behavior"] == v24.PATTERNS[0]
    assert result["step_count"] == 2
```

- [ ] **Step 2: Implement `TeacherEditConfig` and `optimize_teacher_edit`**

Use differentiable source weights and V23/V21 helper functions for probe logits. The objective is:

```python
loss = (
    target_bce
    + config.source_compat_weight * source_compat_mse
    + config.l2_weight * delta.pow(2).mean()
)
```

The returned payload must include tensors only in memory; JSON artifacts store hashes/summaries.

- [ ] **Step 3: Add teacher progress logging**

Teacher generation must append `teacher_edit_completed` after each source-target teacher finishes:

```json
{"event":"teacher_edit_completed","completed_count":17,"record_count":612}
```

No subject IDs in progress events.

## Task 3: Hypereditor Model And Training

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py`

- [ ] **Step 1: Add model I/O tests**

```python
def test_v24_hypereditor_input_output_shapes() -> None:
    model = v24.BehavioralDistilledHypereditor(seed=20260624)
    features = torch.zeros(4, v24.HYPEREDITOR_INPUT_DIM)
    delta, scale = model(features)
    assert delta.shape == (4, v24.SOURCE_WEIGHT_DIM)
    assert scale.shape == (4,)
    assert torch.all(scale >= 0.0)
    assert torch.all(scale <= v24.MAX_HYPEREDITOR_SCALE)
```

- [ ] **Step 2: Implement features**

Feature layout:

```python
[
    normalized_source_weights,          # 345
    normalized_source_signature,        # 560
    normalized_activation_descriptor,   # fixed descriptor from probe examples
    source_behavior_onehot,             # 4
    target_behavior_onehot,             # 4
    source_target_pair_onehot,          # 12
]
```

Add tests that verify slice positions and one-hot sums.

- [ ] **Step 3: Implement training loop**

Train on teacher records from the train-only inner split. Loss:

```python
loss = (
    delta_mse_weight * F.mse_loss(pred_delta, teacher_delta)
    + behavior_weight * target_behavior_loss(edited_weights, target_behavior, probe_examples)
    + compat_weight * source_compatibility_loss(edited_weights, source_subject, probe_examples)
    + l2_weight * pred_delta.pow(2).mean()
)
```

Append `hypereditor_training_step` every 50 steps with scalar losses.

## Task 4: Train-Only Inner Validation

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py`

- [ ] **Step 1: Add split tests**

```python
def test_v24_inner_split_has_no_overlap_and_is_balanced() -> None:
    subjects = [
        {"behavior": behavior, "subject_id": f"{behavior}-{i}", "weights": [0.0] * 345, "signature": [0.0] * 560}
        for behavior in v24.PATTERNS
        for i in range(64)
    ]
    split = v24.inner_train_validation_split(subjects)
    train_ids = {x["subject_id"] for x in split["inner_train_subjects"]}
    valid_ids = {x["subject_id"] for x in split["inner_validation_subjects"]}
    assert not train_ids & valid_ids
    assert len(split["inner_train_subjects"]) == 204
    assert len(split["inner_validation_subjects"]) == 52
```

- [ ] **Step 2: Implement config grid**

Deterministic grid and subset:

```python
TEACHER_STEPS_GRID = [40, 80]
TEACHER_LR_GRID = [0.03, 0.01]
HYPEREDITOR_STEPS_GRID = [800, 1600]
BEHAVIOR_WEIGHT_GRID = [0.5, 1.0, 2.0]
COMPAT_WEIGHT_GRID = [0.25, 0.5, 1.0]
DELTA_MSE_WEIGHT_GRID = [0.25, 1.0]
TEACHER_SOURCE_COMPAT_WEIGHT_GRID = [0.25, 0.5, 1.0]
INNER_VALIDATION_EVALUATED_CONFIG_COUNT = 48
INNER_VALIDATION_RUNG_RECORD_BUDGETS = [24, 72, 156]
INNER_VALIDATION_RUNG_SURVIVORS = [12, 3, 1]
```

The full grid is 432 configs. Always evaluate exactly 48 configs selected by the stratified hash protocol above. All evaluated config hashes must be logged before metrics.

- [ ] **Step 3: Implement selection**

Sort candidates by:

```python
(
    invalid,
    -target_prediction_rate,
    -pareto_undominated_rate,
    -mean_matched_minus_best_control_target_margin,
    -mean_matched_minus_shuffled_signature_target_margin,
    -mean_target_margin,
    mean_compatible_source_mse,
    config_hash,
)
```

## Task 5: Development Evaluation And Artifacts

**Files:**
- Modify: `model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py`
- Test: `model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py`

- [ ] **Step 1: Add control contract test**

```python
def test_v24_required_controls_include_signature_ablations_and_prior_editors() -> None:
    required = set(v24.PROOF_CRITICAL_CONTROL_TYPES)
    assert "no_signature_ablation_behavioral_hypereditor_v24" in required
    assert "no_signature_trained_behavioral_hypereditor_v24" in required
    assert "shuffled_signature_behavioral_hypereditor_v24" in required
    assert "source_behavior_target_ablation_behavioral_hypereditor_v24" in required
    assert "v21_behavioral_probe_residual_output_editor_recomputed" in required
    assert "v22_component_activation_rank1_editor_recomputed" in required
    assert "v23_probe_routed_sparse_subspace_editor_recomputed" in required
    assert v24.EXPECTED_CONTROLS_PER_RECORD == 30
```

- [ ] **Step 2: Implement `evaluate_subjects` with count-only progress logs**

Use V23's `submit` + `as_completed` pattern and sort records before summarizing.

- [ ] **Step 3: Write development result artifact**

`development_results.json` must include:

- result hash inputs: script SHA, constants hash, plan hash, train pool hash, development pool hash
- selected config hash
- inner-validation progress log hash
- development progress log hash
- all aggregate/by-direction metrics
- full failure list
- next action

## Task 6: Verification And Reviewer Gate

**Files:**
- Modify: code/tests from Tasks 1-5
- Create: final preregistration document

- [ ] **Step 1: Run tests**

Run:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py
python -m py_compile model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py
```

Expected: pass. Do not run lint unless explicitly asked.

- [ ] **Step 2: Reviewer plan gate**

Send this plan, the literature basis, and the tests to the reviewer. Do not run development until reviewer returns confidence 5/5.

- [ ] **Step 3: Run generate-pools and development only after 5/5**

Commands:

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py --phase generate-pools --summary-only-stdout
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v24_behavioral_distilled_hypereditor.py --phase development --max-workers 8 --summary-only-stdout
```

Monitor JSONL progress and process CPU. If any long phase lacks count or step logs, stop and add progress instrumentation before continuing.

## Self-Review

- Spec coverage: V23 negative result, fresh data, behavioral loss, fixed probes, signature controls, prior-editor controls, progress logs, and final-raw sealing are all represented.
- Placeholder scan: no TBD/TODO/fill-later steps remain.
- Type consistency: V24 names use `behavioral_distilled_hypereditor_v24`; tests and script constants use the same names.
- Risk: teacher generation may be expensive. The plan requires JSONL progress and permits deterministic evaluated-subset selection before development metrics are viewed.
