# V40 Target-Tolerance Locality-Budget Selector Plan

## Goal

V40 should combine the useful parts of V38 and V39:

- Keep V39's target-first fallback so no-op alpha selection cannot win through perfect compatible MSE.
- Reintroduce locality as a budgeted/tolerance-constrained selector so V39 does not blindly choose the strongest target-moving alpha when a slightly weaker target candidate has much better compatible preservation.

V39 recovered target prediction from V38's `0.2500` to `0.4583`, but compatible-MSE failures regressed to `23`. V40 targets the missing middle: target movement above V38 while compatible-MSE failures below V37's best-plasticity `19`.

## Literature Support

- Epsilon-constraint multi-objective optimization supports optimizing one objective while constraining or bounding others; recent adaptive epsilon work uses tolerances to cover the Pareto surface without collapsing all objectives into one scalar: [STAGE-BO](https://arxiv.org/html/2604.15959v2), [CMOBO](https://arxiv.org/html/2411.03641v1).
- Lexicographic multi-objective methods support priority-ordered objectives, but V39 shows pure lexicographic fallback is too locality-permissive. Thresholded/lexicographic work motivates preserving high-priority objectives while optimizing lower-priority ones inside an admissible set: [Thresholded Lexicographic MORL](https://arxiv.org/html/2408.13493v1), [Lexicographic MORL](https://arxiv.org/abs/2212.13769).
- Multi-task conflict methods such as CAGrad and PCGrad motivate objective balancing instead of one-objective domination: [CAGrad](https://arxiv.org/abs/2110.14048), [PCGrad](https://arxiv.org/abs/2001.06782).
- Model-editing reliability/locality work supports direct measurement of both reliability and locality; V40 keeps both target-rank and compatible-MSE audit fields: [BalancEdit](https://arxiv.org/html/2505.01343v2), [Edit Locality Evaluation](https://arxiv.org/pdf/2601.17343), [Knowledge Updating? No More Model Editing!](https://arxiv.org/html/2503.05212v1).

## Design

Add a new matched edit source:

- `target_tolerance_locality_budget_projected_optimizer_sparse`

Add a new experiment variant:

- `v40_target_tolerance_locality_budget_projected_optimizer_diagnostic`

V40 reuses the V37 projected support optimizer and V39 target-rank diagnostics, but changes alpha selection.

### Alpha Selection Rule

For each alpha candidate, compute:

- `target_gap = max(0, alpha_target_margin_floor - support_target_margin)`
- `tournament_gap = max(0, alpha_tournament_margin_floor - support_tournament_margin)`
- `compatible_gap = max(0, support_compatible_mse - alpha_compatible_mse_soft_gate)`
- `target_rank_score = target_gap + tournament_gap`
- `target_feasible = target_rank_score == 0`

Selection tiers:

1. If any candidate is target-feasible, set the pool to target-feasible candidates.
2. Otherwise, find `best_target_rank_score`, then set the pool to all candidates with:
   `target_rank_score <= best_target_rank_score + target_rank_score_tolerance`.
3. From the pool, choose the locality-preserving candidate:
   `(support_compatible_mse, compatible_gap, -support_tournament_margin, -support_target_margin, -alpha)`.

This is an epsilon/tolerance-constrained selector: first preserve target movement within tolerance, then optimize locality inside that admissible set.

### Bounded Grid

Use four configs derived from V37 base config index `2`:

- `alpha_compatible_mse_soft_gate`: `[10.0, 20.0]`
- `target_rank_score_tolerance`: `[0.05, 0.15]`
- `compatible_gate_weight`: inherited from V37/V38 optimizer score as `0.25`
- event prefix: `v40_target_tolerance_locality_budget_optimizer`
- stable grid hash constant: `V40_GRID_SHA256`

The run remains bounded:

- `--inner-validation-rung-jobs 24`
- `--inner-validation-keep-fractions 1.0`
- `--development-job-selection balanced-directions`
- `--monitor-interval-seconds 5`

## TDD Plan

Write tests before implementation:

1. `test_v40_target_tolerance_grid_is_bounded_and_binds_provenance`
   - Asserts four configs, source/variant routing, event prefix, and `V40_GRID_SHA256`.
2. `test_v40_alpha_selection_uses_locality_within_target_tolerance`
   - Candidate A has best target rank but very high compatible MSE.
   - Candidate B is within tolerance and has much lower compatible MSE.
   - Assert candidate B wins and `selection_mode == "target_tolerance_min_compatible_mse"`.
3. `test_v40_alpha_selection_excludes_noop_outside_target_tolerance`
   - No-op alpha has compatible MSE `0.0` but much worse target rank.
   - Assert no-op is excluded and nonzero alpha wins.
4. `test_v40_alpha_redaction_keeps_tolerance_audit_and_omits_raw_fields`
   - Keeps `target_rank_score_tolerance`, `within_target_tolerance_count`, `best_target_rank_score`, target gap fields, compatible gap, candidate hash, and selection mode.
   - Omits the full forbidden raw-key set.
5. `test_v40_dispatch_uses_target_tolerance_locality_budget_optimizer_and_v25_controls`
   - Ensures dispatcher calls the V40 wrapper and keeps existing V25 native controls.

Regression coverage:

- Keep V37 returned-delta/loss regression.
- Keep V38 score default regression.
- Keep V39 fallback no-op regression.
- Full helper suite and `py_compile` must pass.
- Do not run lint.

## Logging and Leak Controls

Long-running V40 execution must write:

- terminal log
- `development_progress.jsonl`
- `long_run_monitor.jsonl`

Post-run checks:

- no orphan training process
- `wc -l` for progress, monitor, terminal logs
- final monitor event is `monitor_stop`
- SHA-256 hashes for progress, monitor, terminal logs
- event counts include `v40_target_tolerance_locality_budget_alpha_selected`
- strict JSON-key leak scan for:
  `final_subjects`, `subject_id`, `weights`, `logits`, `gradient`, `selected_coordinates`, `support_examples`, `sequence`, `compatible_jacobian`, `raw_delta`, `projected_delta`
- no read of `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`

## Bounded Diagnostic Improvement Criteria

V40 can be called a bounded diagnostic improvement only if:

- target prediction rate above V38's `0.2500`
- compatible-MSE failures below V37 best-plasticity candidate's `19`
- proof failures below V39's `20`
- no raw-field leaks
- reviewer confidence `5/5`

Behavioral editing success still requires the established proof gates.

