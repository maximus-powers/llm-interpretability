# V25 Inner Validation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hash-bound, logged 64-config inner-validation runner for V25 so config choice is based on preregistered grid evidence rather than post-hoc tuning from the failed rank1 diagnostic.

**Architecture:** Reuse existing train/development splits, proof-record evaluator, balanced bounded job selection, train-only edit bank, and `inner_validation_ranking_tuple`. Add a thin runner that evaluates configs in successive rungs, redacts each candidate to aggregate metrics and hashes, ranks candidates deterministically, and logs each rung/config so long runs are auditable.

**Tech Stack:** Python 3.10+, PyTorch, pytest, JSONL progress logs.

---

## Literature Support

- Representation Engineering best-practice concerns: static activation differences can capture noise/spurious features, so config selection must be held to inner-validation gates and controls rather than intuition ([arXiv:2502.17601](https://arxiv.org/html/2502.17601v1)).
- Rank-one model editing work shows weight edits can be traceable and reversible, but one rank-one method is not enough evidence for general functional steering; the grid must test projection alternatives ([arXiv:2505.20819](https://arxiv.org/html/2505.20819v1)).
- Universal/shared weight subspace results motivate spectral-subspace alternatives to a single local rank-1 projection, especially after the 12-direction rank1 diagnostic failed ([arXiv:2512.05117](https://arxiv.org/abs/2512.05117)).

## Files

- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Modify: `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- No linting unless explicitly requested.

## Task 1: Candidate Summary Contract

- [ ] **Step 1: Write failing tests**

Add tests that build synthetic proof records and assert a candidate summary contains:
`config_hash`, `config_index`, `invalid`, `record_count`, `target_prediction_rate`, `pareto_undominated_rate`, `mean_target_margin`, `mean_matched_minus_best_control_target_margin`, `mean_matched_minus_shuffled_signature_target_margin`, `proof_gate_failure_count`, and `proof_record_hashes_hash`.

- [ ] **Step 2: Run focused test**

Run:
`python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q`

Expected: fail because the candidate summarizer does not exist.

- [ ] **Step 3: Implement minimal candidate summarizer**

Add `summarize_v25_inner_validation_candidate(...)` that calls `summarize_v25_records(...)`, copies aggregate fields, counts failures, includes config hash/index, and exposes only proof-record hash aggregate fields.

- [ ] **Step 4: Verify**

Run the focused test again. Expected: pass.

## Task 2: Successive-Halving Rung Planner

- [ ] **Step 1: Write failing tests**

Add tests for `build_v25_successive_halving_plan(...)`:
64 configs, rung job counts `[4, 12]`, keep fractions `[0.25, 0.25]` should produce rung 0 with 64 configs and rung 1 with 16 survivors. The summary must be hash-bound and contain no raw subject keys.

- [ ] **Step 2: Run focused test**

Expected: fail because the planner does not exist.

- [ ] **Step 3: Implement minimal planner**

Add a pure planner that validates rung counts, survivor counts, and returns redacted plan metadata. Runtime ranking happens after each rung.

- [ ] **Step 4: Verify**

Focused test passes.

## Task 3: Logged Runner and CLI

- [ ] **Step 1: Write failing tests**

Add tests using monkeypatched evaluator/build-bank hooks so the runner can be tested without expensive real Jacobians. Assert:
`inner_validation_start`, `inner_validation_rung_start`, `inner_validation_candidate_completed`, `inner_validation_rung_completed`, and `inner_validation_completed` are logged; selected config hash is the top-ranked candidate; no raw `weights` or `subject_id` appear in progress.

- [ ] **Step 2: Run focused test**

Expected: fail because runner/CLI mode does not exist.

- [ ] **Step 3: Implement minimal runner**

Add `run_v25_inner_validation_successive_halving_with_progress(...)` and CLI flags:
`--run-inner-validation`, `--inner-validation-rung-jobs`, and `--inner-validation-keep-fraction`.
The runner must reuse balanced job selection, per-config train edit bank progress, spectral basis summaries, train-only control contexts, and `evaluate_v25_development_jobs_with_progress(...)`.

- [ ] **Step 4: Verify**

Run:
`python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q`
`python -m py_compile /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
`python -m pytest /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v24_helpers.py /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q`

Expected: all pass.

## Task 4: Review and Smoke

- [ ] **Step 1: Send reviewer checkpoint**

Include test outputs, file hashes, plan citations, and the exact claim that this implements selection machinery, not final evidence.

- [ ] **Step 2: Run smoke after 5/5 review**

Start with a small logged smoke, for example 4 configs on 2 jobs, before attempting larger rungs.

- [ ] **Step 3: Review smoke result**

Send progress hashes, selected config summary, logs, and failure/success interpretation to reviewer.

