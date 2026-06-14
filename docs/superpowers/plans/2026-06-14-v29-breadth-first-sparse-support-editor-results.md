# V29 Breadth-First Sparse Support Editor Results

Date: 2026-06-14

## Verdict

V29 is a weak positive diagnostic, not a successful development candidate.

The best config reached `target_prediction_rate=0.5` on the 24-job rung, with
positive proof-control lifts and zero contract failures. It still failed the
pre-registered success gate of `target_prediction_rate >= 0.85` and retained
`proof_gate_failure_count=18`. No sealed-final evaluation is justified.

## Run Command

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v29-breadth-first-sparse \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Verification Before Compute

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v29'
# 7 passed, 109 deselected

python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
# 116 passed

python -m py_compile \
  model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
# passed
```

No lint was run.

## Reviewer Status

Kepler reviewed the amended plan and implementation before compute and returned
`5/5` confidence after the misleading `locality_weight` axis was renamed to
`extra_compatible_weight`.

## Inputs And Provenance

- Train pool records: `264`
- Train pool SHA256:
  `888d539fe8efefcaad91bb6ce0ee48c55f3903d2ed75b6791c4c8b314c0bc35d`
- Development pool records: `98`
- Development pool SHA256:
  `d26f7506cd919de5eeabd9be9ebe205c707d04f169368698849484c3d819a659`
- Balanced selected jobs: `24`
- Selected jobs hash:
  `337ac71f480830a590d8f1bb5437bb8cf0cb2f66ef247faef8ec392cdeac1d59`
- Inner-validation plan hash:
  `08d2e1d7504b031a94de86b7a46103ea41af9af7b71fce9f6a1548fd3803fa26`
- Run-stage: `inner_validation_completed`
- Run `passed`: `false`

The raw V29 grid hash in the plan/test is
`ef40cccc68f4cf08e9e8373de9a8df7555170273f6aaa4ff4d524820859aa9d0`. The run
summary's selected-config grid hash includes train-pool provenance fields and
therefore differs from the raw grid hash.

## Candidate Results

| Rung | Config | Target Rate | Mean Target Margin | Best-Control Lift | Shuffled Lift | Pareto | Proof Failures | Contract Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.1667 | 0.0374 | 0.1774 | 0.1920 | 0.9167 | 24 | 0 |
| 0 | 1 | 0.0833 | -0.0693 | 0.0789 | 0.0854 | 0.7500 | 30 | 0 |
| 0 | 2 | 0.1667 | 0.0795 | 0.2268 | 0.2341 | 0.9167 | 24 | 0 |
| 0 | 3 | 0.0833 | 0.0180 | 0.1661 | 0.1727 | 0.8333 | 28 | 0 |
| 0 | 4 | 0.2500 | 0.0927 | 0.2407 | 0.2473 | 0.9167 | 22 | 0 |
| 0 | 5 | 0.2500 | 0.0882 | 0.2375 | 0.2428 | 0.9167 | 22 | 0 |
| 0 | 6 | 0.4167 | 0.1691 | 0.3167 | 0.3238 | 0.9167 | 18 | 0 |
| 0 | 7 | 0.2500 | 0.0733 | 0.2208 | 0.2279 | 0.8333 | 24 | 0 |
| 1 | 6 | 0.5000 | 0.1888 | 0.3428 | 0.3521 | 0.9167 | 18 | 0 |
| 1 | 4 | 0.3333 | 0.1149 | 0.2708 | 0.2781 | 0.9167 | 22 | 0 |
| 1 | 5 | 0.3333 | 0.0890 | 0.2459 | 0.2522 | 0.9167 | 23 | 0 |
| 1 | 7 | 0.2917 | 0.0934 | 0.2498 | 0.2567 | 0.8750 | 25 | 0 |

Best config selected by the pre-registered successive-halving ranking:

```text
config_index=6
sparse_top_k=32
trust_norm_cap=1.0
extra_compatible_weight=0.05
compatible_floor=0.05
target_prediction_rate=0.5
mean_target_margin=0.1888460420317036
mean_matched_minus_best_control_target_margin=0.34278685139217185
mean_matched_minus_shuffled_signature_target_margin=0.35209354601837123
pareto_undominated_rate=0.9166666666666666
proof_gate_failure_count=18
contract_failure_count=0
```

The selected best config was selected by the pre-registered ranking tuple, not
because it passed the development success gate.

## Monitoring Evidence

- Development progress log SHA256:
  `9212d8d43fa16da21a1113155fbed42e19412797f360e7c47810726f26f171e1`
- Long-run monitor log SHA256:
  `e13b690231107c9ed47356b6e40976ca19b6688cf742f37632c0e283798e8905`
- Progress rows: `2616`
- Monitor rows: `182`
- Monitor events: `monitor_start=1`, `monitor_heartbeat=180`, `monitor_stop=1`
- V29 optimizer progress rows: `1536`
- V29 coordinate-selection rows: `192`
- V29 optimizer-completed rows: `192`
- Candidate-completed rows: `12`

The process exited before result analysis. The last monitor event was
`monitor_stop`, with `progress_line_count=2616`.

## Leak And Logging Audit

The progress and monitor logs were searched for:

```text
final_subjects
subject_id
weights
logits
gradient
selected_coordinates
support_examples
sequence
```

No matches were found. Logs contain hashes, scalar metrics, counts, event names,
and redacted config/provenance fields. The sealed final raw file was not read.

## Interpretation

V29 provides meaningful evidence that the previous zero-flip failures were at
least partly caused by update under-capacity. The best V29 config moved from
V28's `target_prediction_rate=0.0` to `0.5` on the 24-job development rung, with
positive best-control and shuffled-signature lifts.

However, V29 also shows that stronger sparse support optimization is not yet
localized enough to pass proof gates. The proof failures remain high, and the
target rate is far below `0.85`. This supports another development iteration,
not sealed-final evaluation.

## Next Hypothesis

The most promising next direction is not a larger blind trust cap. V29 suggests
that sparse support optimization can cross target boundaries, but proof failures
remain. A better V30 should explicitly optimize per-direction reliability and
locality, likely by:

- adding direction-conditioned support batches or separate per-direction
  coordinate masks;
- incorporating a more exact support-only preservation set instead of only
  compatible-source rows;
- adding a fail-closed selected-config gate that rejects configs with high proof
  failure counts even if target flips improve.
