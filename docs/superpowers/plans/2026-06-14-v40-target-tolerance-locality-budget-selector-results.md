# V40 Target-Tolerance Locality-Budget Selector Results

## Status

V40 completed as a bounded partial diagnostic failure. It did not pass proof gates and did not meet the bounded diagnostic improvement criteria.

The tolerance-budget selector reduced no-op behavior and exposed a clearer tradeoff, but it still could not jointly improve target plasticity and compatible-MSE locality.

## Verification

- Focused RED: 5 expected V40 missing-symbol failures, 3 existing regressions passed.
- Focused GREEN: `8 passed, 176 deselected`.
- Full helper suite: `184 passed in 7.48s`.
- Syntax check: `python -m py_compile` passed for the train script and helper tests.
- No linting was run.
- Implementation review by Kepler: `5/5`.

## Run Evidence

- Experiment variant: `v40_target_tolerance_locality_budget_projected_optimizer_diagnostic`
- Terminal log: `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v40_target_tolerance_locality_budget_terminal.log`
- Progress rows: `974`
- Monitor rows: `58`
- Terminal rows: `177`
- Final monitor event: `monitor_stop`
- Final monitor elapsed seconds: `285.125810667`
- CPU user/system seconds: `521.784302` / `1918.800097`
- No orphan training process remained after completion.
- Progress log SHA-256: `ac8e44d4584357d52e1a070646eb0835c02ea4985f7bba7b6db1aab1e0bdf0af`
- Monitor log SHA-256: `b82d6444dd79538d666ac3cc3e0016c2e3a0d200e63c7195087d5db92308c399`
- Terminal log SHA-256: `89190a0a24f706d6e64941bc21ac75912e7f450dc8fdb7c429872d0d640f3f07`

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

Missing required V40 alpha audit fields: `0`

## Event Counts

- `inner_validation_candidate_completed`: `4`
- `development_evaluation_record_completed`: `96`
- `v40_target_tolerance_locality_budget_optimizer_progress`: `480`
- `v40_target_tolerance_locality_budget_optimizer_completed`: `96`
- `v40_target_tolerance_locality_budget_alpha_selected`: `96`
- `inner_validation_completed`: `1`
- `development_setup_completed`: `1`

## Candidate Metrics

| Config | Hash Prefix | Target Pred Rate | Mean Target Margin | Matched - Best Control | Matched - Shuffled | Pareto Rate | Proof Fails | Compatible MSE Fails | Target Pred Fails | Target Margin Fails | Control Margin Fails |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `2916b104` | 0.4167 | 0.1990 | 0.3541 | 0.3623 | 0.8750 | 20 | 23 | 14 | 14 | 83 |
| 1 | `48060060` | 0.4167 | 0.1907 | 0.3465 | 0.3540 | 0.8333 | 22 | 20 | 14 | 14 | 123 |
| 2 | `b5b182f0` | 0.4583 | 0.2123 | 0.3681 | 0.3756 | 0.8750 | 19 | 23 | 13 | 13 | 72 |
| 3 | `2e3867b8` | 0.4583 | 0.2040 | 0.3612 | 0.3673 | 0.8333 | 21 | 20 | 13 | 13 | 112 |

Best selected config:

- Index: `2`
- Hash: `b5b182f0fec51862680da8450c0782017eafc7375bf9837ce7d7b176504fff43`

## Alpha Diagnostics

Alpha counts:

- `0.0`: `10`
- `0.25`: `2`
- `0.5`: `10`
- `0.75`: `35`
- `1.0`: `39`

Selection modes:

- `target_feasible_min_compatible_mse`: `46`
- `target_tolerance_min_compatible_mse`: `50`

Target-feasible counts:

- `true`: `46`
- `false`: `50`

Target tolerances:

- `0.05`: `48`
- `0.15`: `48`

Within-target-tolerance pool counts:

- `1`: `35`
- `2`: `35`
- `3`: `14`
- `4`: `4`
- `6`: `8`

Preservation energy ratio:

- Count: `96`
- Mean: `0.5003233924508095`
- Min: `0.5000036358833313`
- Max: `0.5027034878730774`

## Interpretation

V40 made the tradeoff more explicit but did not solve it.

Best target/proof candidate:

- V40 config 2: target prediction `0.4583`, proof failures `19`, compatible-MSE failures `23`

Best compatible-MSE candidates:

- V40 configs 1 and 3: compatible-MSE failures `20`, but proof failures `22` and `21`

Against prior versions:

- V38 best selected candidate: target prediction `0.2500`, compatible-MSE failures `10`, proof failures `23`
- V39 best selected candidate: target prediction `0.4583`, compatible-MSE failures `23`, proof failures `20`
- V40 best selected candidate: target prediction `0.4583`, compatible-MSE failures `23`, proof failures `19`

V40 improved proof failures by one relative to V39, but did not reduce compatible-MSE failures for the best selected config. The lower-compatible V40 configs still missed the compatible-MSE target of below V37's best-plasticity `19` and worsened proof failures. Therefore V40 is not a bounded diagnostic improvement under the preregistered criteria.

## Literature Context

The result is consistent with tolerance-constrained multi-objective framing: an epsilon/tolerance admissible set can reveal tradeoffs, but a single tolerance over target rank is not enough to recover locality here.

- Epsilon-constraint and constrained multi-objective work motivate the admissible-set design: [STAGE-BO](https://arxiv.org/html/2604.15959v2), [CMOBO](https://arxiv.org/html/2411.03641v1)
- Lexicographic and thresholded lexicographic work supports priority-preserving secondary optimization: [Thresholded Lexicographic MORL](https://arxiv.org/html/2408.13493v1), [Lexicographic MORL](https://arxiv.org/abs/2212.13769)
- Editing literature requires both reliability and locality to be reported; V40 improves reliability-like metrics slightly but fails locality-like compatible-MSE criteria: [BalancEdit](https://arxiv.org/html/2505.01343v2), [Edit Locality Evaluation](https://arxiv.org/pdf/2601.17343)

## Recommended Next Step

V41 should stop trying to fix the tradeoff only at alpha selection. The repeated pattern suggests the base projected optimizer is producing deltas whose locality/target tradeoff is already too poor before alpha selection.

Most promising next direction:

1. Add compatible-MSE budget directly into the differentiable optimizer trajectory, but with V39/V40 target-rank protection at selection time.
2. Track the optimizer's candidate frontier over epochs, not just the final/best scalar loss state.
3. Select from epoch-level candidates using the V40 tolerance-budget rule.
4. Keep bounded compute and the same leak/monitoring requirements.

