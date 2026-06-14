# V41 Optimizer-Trajectory Frontier Selection Results

## Status

V41 completed as a bounded partial diagnostic result. It did not pass proof gates and did not prove the hypothesis.

The result is the strongest target/proof recovery in the recent V37-V41 projected-optimizer line, but it remains weaker than the earlier V31-V33 target-rate diagnostics and catastrophically fails the compatible-MSE locality criterion. It is evidence that optimizer-trajectory frontier selection can recover functional target behavior, not evidence of controlled representation steering.

## Implementation Under Test

- Experiment variant: `v41_trajectory_frontier_projected_optimizer_diagnostic`
- Matched edit source: `trajectory_frontier_projected_optimizer_sparse`
- Frontier event: `v41_trajectory_frontier_selected`
- Main implementation:
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:268`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:643`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:926`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:1064`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:1100`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:1142`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:1186`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3400`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3771`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:4475`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:4551`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:8798`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:10806`
- Tests:
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3040`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3316`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3462`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3935`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3971`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:4549`

## Verification

- Focused RED: 6 expected V41 missing-symbol failures, 4 existing regressions passed.
- Focused GREEN: `10 passed, 180 deselected`.
- Full helper suite: `190 passed in 11.32s`.
- Syntax check: `python -m py_compile` passed for the train script and helper tests.
- No linting was run.
- Implementation review by Kepler: `5/5`.

## Run Evidence

- Terminal log: `/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v41_trajectory_frontier_terminal.log`
- Progress rows: `1070`
- Monitor rows: `108`
- Terminal rows: `179`
- Final monitor event: `monitor_stop`
- Final monitor elapsed seconds: `532.111802708`
- CPU user/system seconds: `739.363536` / `2558.623225`
- Script-specific orphan process check: clean for `train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
- Progress log SHA-256: `47136dfc467b456ecdf1959b3434ff6c698a403d55ca46d59f9633cda9b0fa37`
- Monitor log SHA-256: `600cd4f22160c660c692f84034cb018a74bb59835e79b4ab61ecb35ab4bad931`
- Terminal log SHA-256: `dc63d13fde579ea663ba4ac412add57b4a9ef76fb9bd0f8ac8804e574ad142a7`

Note: a broad `python.*muat` process check found an unrelated hypernet training process under `packages/experiments/...`. It was not the V41 script process and was not modified.

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

Missing required V41 frontier audit fields: `0`

## Event Counts

- `inner_validation_candidate_completed`: `4`
- `development_evaluation_record_completed`: `96`
- `v41_trajectory_frontier_optimizer_progress`: `480`
- `v41_trajectory_frontier_optimizer_completed`: `96`
- `v41_trajectory_frontier_selected`: `96`
- `v40_target_tolerance_locality_budget_alpha_selected`: `96`
- `inner_validation_completed`: `1`
- `development_setup_completed`: `1`

## Candidate Metrics

| Config | Hash Prefix | Target Pred Rate | Mean Target Margin | Matched - Best Control | Matched - Shuffled | Pareto Rate | Proof Fails | Compatible MSE Fails | Target Pred Fails | Target Margin Fails | Control Margin Fails |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `bfa07072` | 0.7500 | 0.3496 | 0.5047 | 0.5128 | 1.0000 | 8 | 24 | 6 | 6 | 1 |
| 1 | `5d12686b` | 0.7083 | 0.3473 | 0.5040 | 0.5105 | 1.0000 | 8 | 24 | 7 | 5 | 11 |
| 2 | `a8be199d` | 0.7500 | 0.3496 | 0.5031 | 0.5128 | 1.0000 | 8 | 24 | 6 | 6 | 1 |
| 3 | `c86f6e80` | 0.7083 | 0.3473 | 0.5035 | 0.5105 | 1.0000 | 8 | 24 | 7 | 5 | 11 |

Best selected config:

- Index: `0`
- Hash: `bfa07072deac62c2957ba417bd4435c236882334c96f3deb0e84f67faf6f657f`

## Frontier and Alpha Diagnostics

Frontier selection modes:

- `frontier_target_feasible_min_compatible_mse`: `68`
- `frontier_target_tolerance_min_compatible_mse`: `28`

Frontier target-feasible counts:

- `true`: `68`
- `false`: `28`

Frontier candidate count:

- `80`: `96`

Alpha counts:

- `0.5`: `6`
- `0.75`: `32`
- `1.0`: `58`

Alpha selection modes:

- `target_feasible_min_compatible_mse`: `76`
- `target_tolerance_min_compatible_mse`: `20`

Alpha target-feasible counts:

- `true`: `76`
- `false`: `20`

Preservation energy ratio:

- Count: `96`
- Mean: `0.500521820038557`
- Min: `0.500015139579773`
- Max: `0.5027034878730774`

## Interpretation

V41 is a major target/proof improvement within the recent projected-optimizer sequence but an unambiguous locality failure.

Compared with recent versions:

- V38 best selected: target prediction `0.2500`, compatible-MSE failures `10`, proof failures `23`
- V39 best selected: target prediction `0.4583`, compatible-MSE failures `23`, proof failures `20`
- V40 best selected: target prediction `0.4583`, compatible-MSE failures `23`, proof failures `19`
- V41 best selected: target prediction `0.7500`, compatible-MSE failures `24`, proof failures `8`

Within the V37-V41 projected-optimizer line, this is the clearest evidence that the fixed probe/support signatures can identify a functional edit direction that strongly changes behavior on development proof records. It should not be read as stronger target-rate evidence than the earlier V31-V33 diagnostics, and it is not evidence that the edit is localized or controlled. The compatible-MSE gate fails on every V41 proof record in every config, so the representation steering is still too destructive.

The evidence supports a narrowed hypothesis:

- Supported: fixed probe/signature-derived optimization can find functional behavior-changing weight edits in small subject models.
- Not supported yet: the same method can steer representations while preserving compatible behavior/locality.
- Not supported yet: this is sufficient to decode into robust altered functional models without collateral damage.

## Literature Context

The result is consistent with multi-objective frontier literature: retaining trajectory candidates can reveal and exploit high-performing target candidates, but if the frontier is dominated by target-feasible high-compatible-MSE points, locality must be enforced earlier or more directly.

- Pareto/frontier methods support retaining tradeoff candidates rather than one scalar final point: [Parametric Pareto Set Learning](https://arxiv.org/html/2511.05815v1), [ParetoFlow](https://openreview.net/forum?id=mLyyB4le5u), [Hybrid Neural Pareto Front Extraction](https://arxiv.org/abs/2101.11684)
- Feasibility/constrained optimization work supports tracking feasible iterates, but V41 shows target feasibility alone is insufficient: [FSNet](https://arxiv.org/html/2506.00362v2), [Constrained Dual Unrolling](https://arxiv.org/html/2601.17274v1)
- Editing-locality literature requires reporting locality failures directly; V41 must be treated as reliability gain with locality failure: [BalancEdit](https://arxiv.org/html/2505.01343v2), [Edit Locality Evaluation](https://arxiv.org/pdf/2601.17343)

## Recommended Next Step

V42 should make compatible-MSE preservation a hard part of the optimizer trajectory, not just selection:

1. Add a dual/Lagrangian-compatible penalty schedule over epochs.
2. Track both target feasibility and compatible-MSE feasibility in the frontier.
3. Select only from candidates satisfying a compatible-MSE budget if any exist; otherwise report no feasible localized candidate.
4. Keep V41 frontier logging because it produced the strongest target/proof recovery in the recent projected-optimizer line and clean auditability.
