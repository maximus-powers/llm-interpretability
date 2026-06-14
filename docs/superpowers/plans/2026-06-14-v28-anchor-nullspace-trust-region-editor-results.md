# V28 Anchor Nullspace Trust-Region Editor Results

Date: 2026-06-14

## Verdict

V28 is a bounded negative diagnostic for functional behavior editing under the
tested trust-region grid. It should not be represented as a successful edit or
as support for escalating to final-run authorization.

The run produced diagnostic target-margin movement relative to matched and
shuffled controls, but every evaluated candidate had
`target_prediction_rate=0.0`, negative mean target margin, and proof-gate
failures.

## Run

Command:

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v28-anchor-nullspace \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

Final stage: `inner_validation_completed`

Final pass flag: `passed=false`

## Provenance

- Development progress log:
  `6c8d9cf1e42898ed0f462fe48f8f25c381011910d30dd64f325dd9118f6ac22e`
- Long-run monitor log:
  `e4eee626009ffad7e93734f801db11e93e5b3372047f5a7da16461c0be56d9ec`
- Train pool file:
  `888d539fe8efefcaad91bb6ce0ee48c55f3903d2ed75b6791c4c8b314c0bc35d`
- Development pool file:
  `d26f7506cd919de5eeabd9be9ebe205c707d04f169368698849484c3d819a659`
- Inner-validation plan:
  `022738c33771d16a486d409f0d6de19b49b2bc9aebb9c641cf85b0c61722b1c5`
- V28 grid:
  `e215accf0349931a8c357372e0caae0fdc1c134f4be8dee1ad120cae59012cba`

The final raw subject file remained sealed and unread.

## Inner Validation

The bounded scheduler evaluated 8 configs at 12 jobs and promoted 4 configs to
24 jobs. All candidate summaries had `target_prediction_rate=0.0`,
`invalid=false`, and `contract_failure_count=0`.

Final-rung summaries:

| Config | Target Rate | Mean Target Margin | Matched-Control Lift | Shuffled Lift | Pareto Rate | Proof Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.0 | -0.0756109410 | 0.0814946901 | 0.0876365630 | 0.5833333333 | 35 |
| 3 | 0.0 | -0.1023809163 | 0.0533864070 | 0.0608665877 | 0.6250000000 | 34 |
| 5 | 0.0 | -0.1056129403 | 0.0478934824 | 0.0576345637 | 0.5000000000 | 36 |
| 1 | 0.0 | -0.1215746955 | 0.0344576027 | 0.0416728085 | 0.5000000000 | 37 |

The run-selected best candidate was config 3:

- `anchor_count=8`
- `nullspace_rtol=0.01`
- `trust_norm_cap=0.5`
- `compatible_floor=0.05`
- `conflict_weight=0.5`

This config was selected by the preregistered ranking tuple, mainly
Pareto/aggregate ranking behavior, not because it approached the target
prediction gate.

## Monitoring

The run was observable throughout:

- `long_run_monitor.jsonl` ended with `monitor_stop`.
- The monitor showed progressing line counts and rising CPU time.
- The process exited cleanly; no matching experiment process remained.
- The progress log had no traceback or exception markers.
- Reviewer spot checks found no nonfinite numeric values.

## Reviewer

Reviewer confidence: `5/5`

Reviewer conclusion: no blockers. Accept as a bounded negative diagnostic only.
No data-leak issue was identified in the logs; logged values were hash/scalar
surfaces, not raw subject IDs, raw weights, final raw path/content, or raw
tensor payloads.

## Interpretation

V28 did not achieve functional behavior editing under the tested bounded grid.
The result is useful because it narrows the failure mode: the anchor/nullspace
trust-region editor can create consistent relative margin movement, but the
movement remains below the target decision boundary and does not satisfy proof
gates.

No final authorization follows from this result.
