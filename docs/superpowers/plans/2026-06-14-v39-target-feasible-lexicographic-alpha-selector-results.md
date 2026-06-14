# V39 Target-Feasible Lexicographic Alpha Selector Results

## Status

V39 completed as a bounded partial diagnostic result. It did not pass the proof gates.

The selector fixed the V38 no-op-alpha failure mode and recovered some target plasticity, but compatible-MSE locality regressed too far. This should not be reported as behavioral editing success.

## Implementation Under Test

- Experiment variant: `v39_target_feasible_lexicographic_projected_optimizer_diagnostic`
- Matched edit source: `target_feasible_lexicographic_projected_optimizer_sparse`
- Alpha event: `v39_target_feasible_lexicographic_alpha_selected`
- Main implementation:
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:254`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:588`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:836`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:966`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:998`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:1032`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:1068`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3060`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3292`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7762`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7885`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7916`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7926`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:9905`
- Tests:
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:2973`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3114`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3524`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3553`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3583`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3987`

## Verification

- Initial focused RED: 5 expected V39 missing-symbol failures, 2 existing regressions passed.
- Focused GREEN after implementation: `7 passed, 171 deselected`.
- Reviewer-requested fallback regression added.
- Focused GREEN after fallback regression: `8 passed, 171 deselected`.
- Full helper suite: `179 passed in 7.55s`.
- Syntax check: `python -m py_compile` passed for the train script and helper tests.
- No linting was run, per repository instruction.

## Run Command

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v39-target-feasible-lexicographic-projected-optimizer \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions \
  > /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v39_target_feasible_lexicographic_projected_optimizer_terminal.log 2>&1
```

## Long-Run Monitoring Evidence

- Training process after completion: no matching `train_four_behavior_functional_weight_editing_v25` or `python.*muat` process.
- Progress log rows: `974`
- Monitor log rows: `61`
- Terminal log rows: `176`
- Final monitor event: `monitor_stop`
- Final monitor elapsed seconds: `295.676521708`
- CPU user/system seconds: `499.254079` / `1810.204608`
- Final monitor latest progress event: `development_setup_completed`
- Final monitor progress line count: `974`
- Progress log SHA-256: `179a24de3a46108b9dce4aadbf6178a80a623cfddbe600137c340728ddba50cd`
- Monitor log SHA-256: `19aa3ff70e7829784e27ed74803d4856f69ea116261b5dd2e309d1d4a7621de3`
- Terminal log SHA-256: `9602ada1a9be4fc153145da4394ba8706e4f4b6c3d7b0952b4eb8b5ae7046b12`

The monitor showed advancing CPU time and progress line counts throughout the run, including V39 optimizer and alpha-selection events. This was not an idle or stuck process.

## Leak Checks

Strict raw-key scan over progress and monitor logs returned no matches for:

```text
final_subjects
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
```

The sealed final subject file was not opened or read.

V39 alpha-selection records preserved the required audit fields:

- `alpha_compatible_mse_soft_gate`
- `compatible_gap`
- `target_feasible`
- `target_gap`
- `target_rank_score`
- `tournament_gap`
- `candidate_metrics_hash`
- `selection_mode`

Missing required alpha audit fields: `0`

## Event Counts

- `inner_validation_candidate_completed`: `4`
- `development_evaluation_record_completed`: `96`
- `v39_target_feasible_lexicographic_optimizer_progress`: `480`
- `v39_target_feasible_lexicographic_optimizer_completed`: `96`
- `v39_target_feasible_lexicographic_alpha_selected`: `96`
- `inner_validation_completed`: `1`
- `development_setup_completed`: `1`

## Candidate Metrics

| Config | Hash Prefix | Target Pred Rate | Mean Target Margin | Matched - Best Control | Matched - Shuffled | Pareto Rate | Proof Fails | Compatible MSE Fails | Target Pred Fails | Target Margin Fails | Control Margin Fails |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `bbeab454` | 0.4167 | 0.2018 | 0.3583 | 0.3650 | 0.8333 | 21 | 23 | 14 | 14 | 79 |
| 1 | `961560e9` | 0.4167 | 0.2018 | 0.3559 | 0.3650 | 0.8333 | 21 | 23 | 14 | 14 | 79 |
| 2 | `4545487e` | 0.4583 | 0.2151 | 0.3727 | 0.3783 | 0.8333 | 20 | 23 | 13 | 13 | 68 |
| 3 | `ce2a2ceb` | 0.4583 | 0.2112 | 0.3602 | 0.3744 | 0.8333 | 21 | 23 | 13 | 13 | 68 |

Best selected config:

- Index: `2`
- Hash: `4545487ee64a190b355db1ae1057299604446d1696a5cd477f8e03d98aae0487`

## Alpha and Preservation Diagnostics

Alpha counts:

- `0.0`: `4`
- `0.5`: `4`
- `0.75`: `24`
- `1.0`: `64`

Selection modes:

- `target_feasible_min_compatible_mse`: `46`
- `fallback_target_feasible_lexicographic`: `50`

Target-feasible counts:

- `true`: `46`
- `false`: `50`

Soft gate values:

- `10.0`: `48`
- `20.0`: `48`

Preservation energy ratio:

- Count: `96`
- Mean: `0.5003175108383099`
- Min: `0.5000036358833313`
- Max: `0.5027034878730774`

## Interpretation

V39 fixed the immediate V38 no-op selector failure. V38 selected alpha `0.0` in `58/96` records; V39 selected alpha `0.0` in only `4/96` records and mostly selected alpha `0.75` or `1.0`.

This recovered target movement:

- V38 best selected candidate: target prediction rate `0.2500`, proof failures `23`
- V39 best selected candidate: target prediction rate `0.4583`, proof failures `20`

But it regressed locality:

- V37 best target-plastic candidate: compatible-MSE failures `19`
- V38 best selected candidate: compatible-MSE failures `10`
- V39 best selected candidate: compatible-MSE failures `23`

Therefore V39 is not a bounded diagnostic improvement under the preregistered criteria because compatible-MSE failures exceeded V37's `19` threshold. The result is still useful: it isolates the tradeoff. Target-feasible lexicographic selection solves the no-op problem but is too permissive about compatible preservation.

## Literature Context

The result is consistent with the reliability/locality tradeoff emphasized in knowledge-editing work and with multi-objective optimization literature:

- CAGrad and PCGrad motivate preserving objective separation when gradients or objectives conflict, but V39 shows that target-first lexicographic selection alone can overcorrect away from locality: [CAGrad](https://arxiv.org/html/2110.14048v2), [PCGrad](https://arxiv.org/abs/2001.06782)
- BalancEdit and edit-locality evaluation work emphasize that reliability and locality must both be measured; V39 improves reliability-like target movement while failing locality-like compatible-MSE gates: [BalancEdit](https://arxiv.org/html/2505.01343v2), [Edit Locality Evaluation](https://arxiv.org/pdf/2601.17343)
- AlphaEdit and ENFORCE support preservation constraints, but V39 suggests preservation needs to be reintroduced as an adaptive constraint after target feasibility, not removed from the primary decision: [AlphaEdit](https://arxiv.org/abs/2410.02355), [ENFORCE](https://arxiv.org/html/2502.06774v4)

## Recommended Next Step

V40 should combine V38 and V39 instead of extending either alone:

1. Keep V39's fallback rule that prevents no-op alpha selection.
2. Add a locality budget after target-feasibility ranking, such as selecting the lowest compatible-MSE candidate among candidates within a small target-rank-score tolerance of the best candidate.
3. Add an audit field for `target_rank_score_tolerance` and `within_target_tolerance_count`.
4. Keep the same bounded 4-config, 24-job grid until the method beats V38 on target prediction and beats V37 best-plasticity on compatible-MSE failures.

