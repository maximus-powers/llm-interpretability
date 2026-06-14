# Centroid Residual Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct condition decoding with an optional centroid-residual path so train-centroid controls cannot use the same prototype channel as matched signatures.

**Architecture:** Add a `use_condition_residual_decoder` config switch. When enabled, `WeightDecoder` decodes a neutral base from `z` and a behavior delta from `condition - condition_baseline`; callers pass same-label train centroids as `condition_baseline` where labels are available. Existing checkpoints remain compatible when the switch is false.

**Tech Stack:** Python 3.10+, PyTorch, existing direct Python test harness in `model_zoo/hypernet/tests/test_functional_hypernetwork.py`.

---

### Task 1: Decoder API And Unit Tests

**Files:**
- Modify: `model_zoo/hypernet/tests/test_functional_hypernetwork.py`
- Modify: `model_zoo/hypernet/models/functional_hypernetwork.py`

- [ ] **Step 1: Write failing tests**

Add tests proving:
- `HyperNetConfig` exposes `use_condition_residual_decoder` and `condition_residual_scale`.
- With residual decoding enabled, `decode_weights(z, condition, condition_baseline=condition)` equals the latent-only base decode.
- A non-centroid condition changes the output relative to the centroid condition.

- [ ] **Step 2: Verify RED**

Run:

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_functional_hypernetwork",
    "model_zoo/hypernet/tests/test_functional_hypernetwork.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in [
    "test_centroid_residual_decoder_config_defaults",
    "test_centroid_residual_decoder_zeroes_centroid_condition",
]:
    getattr(module, name)()
PY
```

Expected: FAIL because config fields or `condition_baseline` are missing.

- [ ] **Step 3: Implement minimal decoder changes**

Update `HyperNetConfig`, `WeightDecoder`, and `FunctionalHyperNetwork.decode_weights` so the old direct path remains default and the new residual path only activates when configured.

- [ ] **Step 4: Verify GREEN**

Run the same direct test command. Expected: PASS.

### Task 2: Train Centroid Baseline Wiring

**Files:**
- Modify: `model_zoo/hypernet/models/functional_hypernetwork.py`

- [ ] **Step 1: Write failing tests**

Add tests for helper behavior:
- `build_condition_baseline(condition, labels)` returns same-label encoded train centroids when available.
- missing labels or disabled residual decoding returns `None`.

- [ ] **Step 2: Verify RED**

Run the two new helper tests directly. Expected: FAIL because helper is absent.

- [ ] **Step 3: Implement baseline helper and training calls**

Add `build_condition_baseline()`. Pass baselines into:
- matched zero-latent condition decodes;
- control decodes;
- edit decodes;
- specificity loss decodes where the model owns the decode call.

- [ ] **Step 4: Verify GREEN**

Run all direct hypernet tests. Expected: PASS.

### Task 3: Evaluation And CLI Config Wiring

**Files:**
- Modify: `model_zoo/hypernet/evaluation/pipeline.py`
- Modify: `model_zoo/hypernet/train.py`
- Modify: `model_zoo/configs/hypernet/default.yaml`

- [ ] **Step 1: Write failing tests**

Extend existing train config tests to assert the new fields flow through `build_hypernet_config`.

- [ ] **Step 2: Implement wiring**

Thread `use_condition_residual_decoder` and `condition_residual_scale` through config loading, model config construction, and default YAML. In evaluation, compute same-label encoded train centroid baselines and pass them into all matched/control decode calls.

- [ ] **Step 3: Verify**

Run focused direct tests and py-compile touched files:

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_functional_hypernetwork",
    "model_zoo/hypernet/tests/test_functional_hypernetwork.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name, fn in sorted(module.__dict__.items()):
    if name.startswith("test_") and callable(fn):
        fn()
print("direct tests passed")
PY
python -m py_compile \
  model_zoo/hypernet/models/functional_hypernetwork.py \
  model_zoo/hypernet/train.py \
  model_zoo/hypernet/evaluation/pipeline.py \
  model_zoo/hypernet/tests/test_functional_hypernetwork.py
```

Expected: PASS. Do not run lint.

### Task 4: Result Checkpoint

**Files:**
- Modify: `research-log.md`

- [ ] **Step 1: Train and evaluate one architectural checkpoint**

Run the existing training entrypoint with residual decoding enabled and a new run directory.

- [ ] **Step 2: Summarize proof-gate metrics**

Extract matched decode, control matrices, subject specificity, validation gap, and clean proof gate status from `results.json`.

- [ ] **Step 3: Reviewer checkpoint**

Send the checkpoint evidence to Kepler and continue only after confidence is 5/5.

- [ ] **Step 4: Update research log**

Record the result and reviewer decision without overstating evidence.
