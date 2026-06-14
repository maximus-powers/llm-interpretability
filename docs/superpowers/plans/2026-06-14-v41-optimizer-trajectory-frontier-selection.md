# V41 Optimizer-Trajectory Frontier Selection Plan

## Goal

V41 should stop treating alpha selection as the only place to recover the target/locality tradeoff.

V38, V39, and V40 imply that the base projected optimizer often produces a final/best scalar delta whose tradeoff is already poor:

- V38 protected compatible-MSE locality but selected weak/no-op alphas.
- V39 restored target movement but compatible-MSE failures rose to `23`.
- V40 exposed target/locality tolerance pools but still selected best configs with compatible-MSE failures `23`; lower-compatible configs had worse proof failures.

V41 will select from the optimizer trajectory itself. Instead of returning only the best scalar support-score epoch, it will retain a redacted epoch-level frontier and select an epoch candidate using the V40 target-tolerance/locality-budget rule before alpha scaling.

## Literature Support

- Pareto/frontier methods emphasize retaining a set of tradeoff candidates rather than collapsing to one scalar early. Recent multi-objective work learns or extracts Pareto sets/frontiers to preserve downstream choice: [Parametric Pareto Set Learning for Expensive Multi-Objective Optimization](https://arxiv.org/html/2511.05815v1), [ParetoFlow](https://openreview.net/forum?id=mLyyB4le5u), [Hybrid Neural Pareto Front Extraction](https://arxiv.org/abs/2101.11684).
- Constrained multi-objective methods motivate solving a sequence of constrained single-objective problems or selecting from feasible/admissible sets: [C-MORL](https://arxiv.org/html/2410.02236v2), [IPRO](https://arxiv.org/pdf/2402.07182), [STAGE-BO](https://arxiv.org/html/2604.15959v2).
- Feasibility-seeking and constrained optimization work supports tracking feasible iterates and minimizing constraint violations rather than relying on one final iterate: [FSNet](https://arxiv.org/html/2506.00362v2), [Constrained Dual Unrolling](https://arxiv.org/html/2601.17274v1), [Learning Constrained Optimization with Deep Augmented Lagrangian Methods](https://arxiv.org/html/2403.03454v2).
- Editing reliability/locality papers continue to justify reporting both target behavior and locality-like compatible-MSE metrics directly rather than claiming success from one side: [BalancEdit](https://arxiv.org/html/2505.01343v2), [Edit Locality Evaluation](https://arxiv.org/pdf/2601.17343).

## Design

Add a new matched edit source:

- `trajectory_frontier_projected_optimizer_sparse`

Add a new experiment variant:

- `v41_trajectory_frontier_projected_optimizer_diagnostic`

V41 reuses V37's projected support optimizer, but adds a config-controlled trajectory frontier mode.

### Optimizer Frontier Capture

During each optimizer epoch, after the post-step metrics are recomputed, capture a sanitized candidate:

- `epoch`
- `support_compatible_mse`
- `support_target_margin`
- `support_tournament_margin`
- `support_runner_margin`
- `loss`
- `preservation_energy_ratio`
- `projected_delta_norm`
- `optimizer_score`
- `candidate_hash`

The in-memory candidate may keep the delta tensor for selection, but progress logs and metadata must contain only scalar/hash fields. No raw coordinates, gradients, logits, examples, weights, sequences, or final subject data are logged.

### Epoch Selection Rule

Add helper:

- `select_v41_trajectory_frontier_candidate(candidates, target_rank_score_tolerance, compatible_mse_soft_gate, target_margin_floor, tournament_margin_floor)`

Selection rule:

1. Compute target/tournament gaps and target rank score per epoch candidate.
2. If any epoch candidate is target-feasible, pool feasible candidates.
3. Otherwise pool candidates within `best_target_rank_score + target_rank_score_tolerance`.
4. Select the candidate with:
   `(support_compatible_mse, compatible_gap, -support_tournament_margin, -support_target_margin, projected_delta_norm, epoch)`.

This is V40's tolerance-budget rule applied before alpha scaling.

### Interaction With Alpha Selection

The selected epoch delta becomes the base delta passed into the existing V40 alpha selector. This gives two chances to preserve locality:

1. Epoch-level frontier selection chooses a better base delta.
2. Alpha-level V40 selection still prevents no-op outside target tolerance and optimizes locality inside target-tolerant pools.

### Bounded Grid

Use four configs derived from V37 base config index `2`:

- `alpha_compatible_mse_soft_gate`: `[10.0, 20.0]`
- `target_rank_score_tolerance`: `[0.05, 0.15]`
- `trajectory_frontier_enabled`: `true`
- `trajectory_frontier_event_prefix`: `v41_trajectory_frontier`
- `projected_optimizer_event_prefix`: `v41_trajectory_frontier_optimizer`
- stable grid hash constant: `V41_GRID_SHA256`

Keep the bounded run:

- `--inner-validation-rung-jobs 24`
- `--inner-validation-keep-fractions 1.0`
- `--development-job-selection balanced-directions`
- `--monitor-interval-seconds 5`

## TDD Plan

Write tests before implementation:

1. `test_v41_trajectory_frontier_grid_is_bounded_and_binds_provenance`
   - Asserts four configs, V41 source/variant routing, V41 optimizer/frontier event prefixes, and `V41_GRID_SHA256`.
2. `test_v41_frontier_selection_prefers_locality_within_target_tolerance`
   - Two epoch candidates are within target tolerance; lower-compatible candidate wins even with slightly worse target rank.
3. `test_v41_frontier_selection_excludes_noop_outside_target_tolerance`
   - A no-op/local candidate has much worse target rank; assert it is excluded.
4. `test_v41_projected_optimizer_returns_frontier_selected_epoch`
   - Monkeypatch support metrics so an early epoch has lower compatible MSE inside tolerance than the final/best scalar-score epoch.
   - Assert returned audit uses `trajectory_frontier_selected_epoch` and selected delta from the frontier candidate.
5. `test_v41_frontier_redaction_omits_raw_fields`
   - Keeps frontier count/hash, selected epoch, target-rank audit fields, compatible MSE, preservation ratio, delta norm.
   - Injects and omits the full forbidden raw-key set.
6. `test_v41_dispatch_uses_trajectory_frontier_optimizer_and_v25_controls`
   - Ensures dispatcher calls V41 wrapper and retains V25 native controls.

Regression coverage:

- Keep V37 returned-delta/loss regression.
- Keep V38 score default regression.
- Keep V39 no-op fallback regression.
- Keep V40 tolerance/no-op tests.
- Full helper suite and `py_compile` must pass.
- Do not run lint.

## Logging and Leak Controls

Long-running V41 execution must write:

- terminal log
- `development_progress.jsonl`
- `long_run_monitor.jsonl`

Post-run checks:

- no orphan training process
- `wc -l` for progress, monitor, terminal logs
- final monitor event is `monitor_stop`
- SHA-256 hashes for progress, monitor, terminal logs
- event counts include `v41_trajectory_frontier_selected` and `v40_target_tolerance_locality_budget_alpha_selected` or V41-specific alpha event if implemented
- strict JSON-key leak scan for:
  `final_subjects`, `subject_id`, `weights`, `logits`, `gradient`, `selected_coordinates`, `support_examples`, `sequence`, `compatible_jacobian`, `raw_delta`, `projected_delta`
- no read of `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`

## Bounded Diagnostic Improvement Criteria

V41 can be called a bounded diagnostic improvement only if:

- target prediction rate above V38's `0.2500`
- compatible-MSE failures below V37 best-plasticity candidate's `19`
- proof failures below V40's `19`
- no raw-field leaks
- reviewer confidence `5/5`

Behavioral editing success still requires the established proof gates.

