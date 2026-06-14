# Differentiable Functional Hypernet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fixed-probe activation-signature experiment train on functional behavior, then verify Interpret, Steer, and Decode with focused metrics.

**Architecture:** Keep the current `FunctionalHyperNetwork` path, but replace graph-breaking subject-model instantiation in functional training with a differentiable flat-weight forward pass. Use small automated checks to prove gradients flow into generated weights and hypernetwork parameters before running research-scale training.

**Tech Stack:** Python 3.10+, PyTorch, existing `model_zoo/hypernet` package, direct Python/pytest-style test scripts. Do not run linting unless explicitly requested.

---

### Task 1: Prove Functional Loss Currently Breaks Gradients

**Files:**
- Modify: `model_zoo/hypernet/tests/test_functional_hypernetwork.py`

- [ ] **Step 1: Add a failing gradient test**

Add a test function that creates a tiny `FunctionalHyperNetwork`, runs `compute_functional_loss()` on generated reconstructions, calls `backward()`, and asserts at least one decoder parameter receives a non-zero gradient.

- [ ] **Step 2: Run the targeted test**

Run: `python -m pytest model_zoo/hypernet/tests/test_functional_hypernetwork.py -k functional_loss_backpropagates -q`

Expected before implementation: FAIL because no hypernetwork parameter receives gradient from functional loss.

### Task 2: Implement Differentiable Subject Forward

**Files:**
- Modify: `model_zoo/hypernet/models/functional_hypernetwork.py`
- Test: `model_zoo/hypernet/tests/test_functional_hypernetwork.py`

- [ ] **Step 1: Add a flat-weight forward helper**

Implement a method that consumes flat generated weights and input probes, applies the same linear/GELU stack as `SubjectNetwork`, and returns logits without assigning `.data` or constructing detached modules.

- [ ] **Step 2: Replace functional loss internals**

Update `compute_functional_loss()` to compare original and reconstructed outputs through the differentiable helper, preserving the current output-MSE and margin-preservation intent.

- [ ] **Step 3: Verify gradients**

Run: `python -m pytest model_zoo/hypernet/tests/test_functional_hypernetwork.py -k functional_loss_backpropagates -q`

Expected after implementation: PASS with non-zero gradient on decoder parameters.

### Task 3: Add Focused Behavior Objective

**Files:**
- Modify: `model_zoo/hypernet/models/functional_hypernetwork.py`
- Modify: `model_zoo/hypernet/train.py`
- Test: `model_zoo/hypernet/tests/test_functional_hypernetwork.py`

- [ ] **Step 1: Add differentiable pattern-case utilities**

Represent positive and negative examples for the focused proof patterns as tensors and compute target margins directly from flat decoded weights.

- [ ] **Step 2: Add target-behavior loss**

For labeled batches, add a loss that increases target positive logits and decreases target negative logits for each reconstructed/generated network.

- [ ] **Step 3: Verify the target loss trains through decoded weights**

Run a targeted gradient test proving the target behavior loss produces non-zero decoder gradients.

### Task 4: Add Focused Experiment Runner and Metrics

**Files:**
- Modify: `model_zoo/hypernet/train.py`
- Modify: `model_zoo/hypernet/evaluation/pipeline.py`

- [ ] **Step 1: Restrict proof runs to behaviorally clear patterns**

Support a focused pattern subset containing `sorted_ascending`, `sorted_descending`, `increasing_pairs`, and `decreasing_pairs`.

- [ ] **Step 2: Emit three-operation metrics**

Write a results summary containing signature interpretation accuracy, latent steering margin improvement, and decoded behavior accuracy on held-out examples.

- [ ] **Step 3: Verify on a small sample**

Run a short CPU/MPS-compatible smoke experiment with limited samples and epochs. The expected result is not final research quality, but it must produce all three metric families and prove functional losses are active.

### Task 5: Run the Real Proof Experiment

**Files:**
- Output: `runs/`

- [ ] **Step 1: Run the focused experiment long enough to evaluate**

Use the fixed differentiable path with functional behavior loss enabled.

- [ ] **Step 2: Run comprehensive evaluation**

Evaluate Interpret, Steer, and Decode on held-out data and compare to random/signature-shuffle baselines where available.

- [ ] **Step 3: Summarize the result**

Report whether the fixed probe activation signatures support the three linked operations, and identify the remaining weakest link if the result is partial.
