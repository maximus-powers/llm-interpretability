# V39 Target-Feasible Lexicographic Alpha Selector Plan

## Goal

V39 should recover target plasticity without discarding the compatible-MSE locality gains learned from V38.

The V38 run showed the failure mechanism clearly: when no alpha candidate satisfied all hard gates, fallback selection could choose alpha `0.0` because compatible MSE was perfect even while target and tournament margins were negative. V39 will make alpha fallback target-feasibility-first so compatible preservation cannot win through a no-op edit.

## Literature Support

- Multi-objective optimization work motivates treating target movement and locality as competing objectives rather than collapsing them into one brittle scalar. CAGrad explicitly balances objectives through conflict-aware optimization, while MGDA/gradient-surgery families preserve Pareto reasoning instead of allowing one objective to dominate: [CAGrad](https://arxiv.org/html/2110.14048v2), [PCGrad](https://arxiv.org/abs/2001.06782).
- Knowledge-editing literature emphasizes reliability/locality tradeoffs and warns against evaluating only one side of the edit objective. V38 was locality-biased and therefore failed reliability: [BalancEdit](https://arxiv.org/html/2505.01343v2), [Are We Evaluating the Edit Locality of LLM Model Editing Properly?](https://arxiv.org/pdf/2601.17343).
- Constraint-based methods support enforcing preservation, but V38 shows the constraint should not be the first fallback criterion when the target objective is infeasible. ENFORCE and AlphaEdit motivate projection/nullspace locality, but the present small-scale setting needs target feasibility as the primary guardrail: [ENFORCE](https://arxiv.org/html/2502.06774v4), [AlphaEdit](https://arxiv.org/abs/2410.02355).
- Recent representation/model-editing work frames editing-locality balance as the core design problem, which supports testing a selector-level balance before adding more compute-heavy optimizer machinery: [BaFT](https://arxiv.org/pdf/2503.00306), [Knowledge Updating? No More Model Editing!](https://arxiv.org/html/2503.05212v1).

## Design

Add a new matched edit source:

- `target_feasible_lexicographic_projected_optimizer_sparse`

Add a new experiment variant:

- `v39_target_feasible_lexicographic_projected_optimizer_diagnostic`

V39 reuses the V37/V38 projected support optimizer, but replaces V38 alpha selection with a target-feasible lexicographic selector.

### Alpha Selection Rule

For each alpha candidate, compute and log:

- `target_gap = max(0, alpha_target_margin_floor - support_target_margin)`
- `tournament_gap = max(0, alpha_tournament_margin_floor - support_tournament_margin)`
- `compatible_gap = max(0, support_compatible_mse - alpha_compatible_mse_soft_gate)`
- `target_feasible = target_gap == 0 and tournament_gap == 0`
- `target_rank_score = target_gap + tournament_gap`

Selection tiers:

1. If any candidate is target-feasible, choose the feasible candidate with best locality/tournament tuple:
   `(support_compatible_mse, -support_tournament_margin, -support_target_margin, -alpha)`.
2. If none are target-feasible, choose the candidate with best target movement before locality:
   `(target_rank_score, -support_tournament_margin, -support_target_margin, compatible_gap, support_compatible_mse, -alpha)`.

This preserves V38's locality evidence fields while preventing no-op alpha selection from winning solely through compatible MSE.

The method name intentionally avoids claiming Pareto-front selection. It uses the multi-objective literature as motivation for objective separation, but the implementation is a deterministic lexicographic selector.

### Bounded Grid

Use four configs, all derived from the V37 best-plasticity region and V38 compatible-MSE diagnostic:

- `alpha_compatible_mse_soft_gate`: `[10.0, 20.0]`
- `compatible_gate_weight`: `[0.25, 0.75]`
- base optimizer config: `build_v37_projected_support_optimizer_config_grid()[2]`
- event prefix: `v39_target_feasible_lexicographic_optimizer`
- stable grid hash constant: `V39_GRID_SHA256`, bound by a test using `stable_hash_json(build_v39_target_feasible_lexicographic_optimizer_config_grid())`

The run remains comparable to V37/V38:

- `--inner-validation-rung-jobs 24`
- `--inner-validation-keep-fractions 1.0`
- `--development-job-selection balanced-directions`
- `--monitor-interval-seconds 5`
- terminal log redirected to a V39-specific log file

## TDD Plan

Write failing tests before implementation:

1. `test_v39_target_feasible_grid_is_bounded_and_binds_provenance`
   - Asserts four configs.
   - Asserts V39 source/variant routing.
   - Asserts event prefix and soft gate fields.
   - Asserts `stable_hash_json(build_v39_target_feasible_lexicographic_optimizer_config_grid()) == V39_GRID_SHA256`.
2. `test_v39_alpha_selection_prefers_target_movement_over_noop_fallback`
   - Provides alpha `0.0` with compatible MSE `0.0` but negative margins.
   - Provides alpha `0.75` with higher compatible MSE but much better target/tournament margins.
   - Asserts V39 does not select alpha `0.0`.
3. `test_v39_alpha_selection_uses_locality_among_target_feasible_candidates`
   - Provides multiple target-feasible candidates.
   - Asserts the lower compatible-MSE candidate wins, preserving V38 locality intent.
4. `test_v39_alpha_redaction_keeps_target_feasibility_audit_and_omits_raw_fields`
   - Keeps `target_gap`, `tournament_gap`, `compatible_gap`, `target_feasible`, `target_rank_score`, `candidate_metrics_hash`.
   - Keeps selected mode.
   - Omits the full forbidden raw-key set: `final_subjects`, `subject_id`, `weights`, `logits`, `gradient`, `selected_coordinates`, `support_examples`, `sequence`, `compatible_jacobian`, `raw_delta`, `projected_delta`.
5. `test_v39_dispatch_uses_target_feasible_pareto_optimizer_and_v25_controls`
   - Ensures the inner-validation dispatcher calls the V39 wrapper and preserves existing V25 control/shuffled evaluation paths.

Regression coverage:

- Keep `test_v38_score_defaults_match_v37_formula`.
- Keep `test_v37_projected_optimizer_final_loss_matches_returned_delta`.
- Full helper suite must pass.
- `python -m py_compile` must pass for the edited script and helper tests.
- Do not run lint.

## Logging and Leak Controls

Long-running V39 execution must write:

- terminal log
- `development_progress.jsonl`
- `long_run_monitor.jsonl`

Post-run required checks:

- `pgrep` confirms no orphan training process.
- `wc -l` over progress, monitor, and terminal logs.
- final monitor event is `monitor_stop`.
- SHA-256 hashes for progress, monitor, and terminal logs.
- event counts include `v39_target_feasible_lexicographic_alpha_selected`.
- strict JSON-key scan for raw leak fields:
  `final_subjects`, `subject_id`, `weights`, `logits`, `gradient`, `selected_coordinates`, `support_examples`, `sequence`, `compatible_jacobian`, `raw_delta`, `projected_delta`.
- no read of `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`.

V39 alpha event contract:

- Event name: `v39_target_feasible_lexicographic_alpha_selected`
- Required fields:
  - `alpha`
  - `alpha_candidate_count`
  - `alpha_compatible_mse_soft_gate`
  - `candidate_index`
  - `candidate_metrics_hash`
  - `compatible_gap`
  - `eligible_count`
  - `selection_mode`
  - `support_compatible_mse`
  - `support_runner_margin`
  - `support_target_margin`
  - `support_tournament_margin`
  - `target_feasible`
  - `target_gap`
  - `target_rank_score`
  - `tournament_gap`

## Bounded Diagnostic Improvement Criteria

V39 can be called a bounded diagnostic improvement only if it improves over V38 target plasticity without regressing all the way to V37 locality:

- target prediction rate above V38's `0.2500`
- compatible-MSE failures below V37 best-plasticity candidate's `19`
- proof failures below V38's `23`
- no raw-field leaks
- reviewer confidence `5/5`

Behavioral editing success still requires passing the established proof gates. If V39 satisfies the bounded diagnostic improvement criteria but does not pass those proof gates, it must be reported as a partial diagnostic result, not as behavioral editing success.

If V39 recovers target prediction but compatible failures return to V37 levels, it is a partial result. If V39 still selects many no-op alphas, the selector failed and should not be extended.
