# V34 Locality Pressure Grid Diagnostic Results

Date: 2026-06-14

## Verdict

V34 is useful negative diagnostic evidence, not success.

The locality-pressure grid did not reduce the compatible-MSE blocker discovered
in V33. Every V34 candidate had `compatible_mse_fail_count=24/24`. Lower norm
caps and higher compatible-preservation weight reduced target performance and
introduced more target/control failures, while the best V34 candidate still
failed compatible MSE on all records.

No sealed-final evaluation was run.

## Reviewed Inputs

- Plan:
  `docs/superpowers/plans/2026-06-14-v34-locality-pressure-grid-diagnostic.md`
- Implementation files:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
  - `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Reviewer gates:
  - V34 plan approved by Kepler with confidence `5/5`.
  - V34 implementation approved by Kepler with confidence `5/5`.

## Verification Before Compute

Focused V34 tests:

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q -k 'v34'
```

Result: `2 passed, 143 deselected in 1.39s`.

Full helper tests:

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q
```

Result: `145 passed in 10.84s`.

Compile check:

```bash
python -m py_compile \
  /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Result: passed.

No linting was run.

## Bounded Development Command

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v34-locality-pressure \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

The run exited cleanly with `passed=false`.

## Provenance

- Development pool file hash:
  `d26f7506cd919de5eeabd9be9ebe205c707d04f169368698849484c3d819a659`
- Train pool file hash:
  `888d539fe8efefcaad91bb6ce0ee48c55f3903d2ed75b6791c4c8b314c0bc35d`
- Train pool summary hash:
  `d24ab4c41f79ec97e8c936817eed723a24afed7d3259765e111bd970733d141e`
- Development selected jobs hash:
  `337ac71f480830a590d8f1bb5437bb8cf0cb2f66ef247faef8ec392cdeac1d59`
- Development selection hash:
  `ee27058c0cce00f21d28c49d915b68a312c322c9d09b9f98a73ae280a096c0d1`
- Raw V34 grid hash:
  `4ee212a749e6db4210ce7ac096e1d5884130d38a2693a7272a23ab354229f722`
- Runtime V34 config grid hash:
  `eabfdbaf5d63fadfedf4a781e919f29b9ea363fa29562abeb3a4077b18da0501`
- V34 inner-validation plan hash:
  `f288278fa504caac148abc64d06a0d4075ce38e211c78d811bae2789f89692fb`

## Candidate Table

| Config | Trust cap | Compat weight | Target rate | Mean target margin | Best-control lift | Shuffled lift | Pareto rate | Proof failures | Compatible MSE fails | Target pred fails | Target margin fails | Control margin fails |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.50 | 0.50 | 0.5000000000 | 0.2628832981 | 0.4163550639 | 0.4261308021 | 0.9583333333 | 15 | 24 | 12 | 10 | 45 |
| 1 | 0.50 | 2.00 | 0.2916666667 | 0.1220885765 | 0.2767436150 | 0.2853360805 | 0.9583333333 | 22 | 24 | 17 | 15 | 59 |
| 2 | 0.75 | 0.50 | 0.6250000000 | 0.3318368212 | 0.4876220500 | 0.4950843252 | 0.9583333333 | 12 | 24 | 9 | 7 | 34 |
| 3 | 0.75 | 2.00 | 0.4583333333 | 0.2536588616 | 0.4098016252 | 0.4169063656 | 0.9583333333 | 15 | 24 | 13 | 10 | 44 |
| 4 | 1.00 | 0.50 | 0.6666666667 | 0.4222446258 | 0.5789551543 | 0.5854921298 | 1.0000000000 | 10 | 24 | 8 | 7 | 34 |
| 5 | 1.00 | 2.00 | 0.4166666667 | 0.2658397919 | 0.4198211144 | 0.4290872959 | 0.9166666667 | 16 | 24 | 14 | 11 | 55 |

Best V34 candidate:

```text
config_index=4
config_hash=752dd62d7293bddcb9e775ecfcc1e7e2baa8c678b41f4e7b255cc801823c561e
trust_norm_cap=1.0
extra_compatible_weight=0.5
compatible_orthogonal_weight=0.15
target_prediction_rate=0.6666666666666666
mean_target_margin=0.42224462583544664
mean_matched_minus_best_control_target_margin=0.5789551543455218
mean_matched_minus_shuffled_signature_target_margin=0.5854921298221143
pareto_undominated_rate=1.0
proof_gate_failure_count=10
contract_failure_count=0
```

Best-candidate proof-gate breakdown:

```text
record_count=24
compatible_mse_fail_count=24
target_prediction_fail_count=8
target_margin_fail_count=7
pareto_fail_count=0
control_margin_fail_count=34
control_margin_failure_record_count=4
mean_control_margin_advantage=0.586792936195469
min_control_margin_advantage=0.00016933638835325837
```

## Monitoring Evidence

- Process `49838` was no longer alive after completion.
- Progress log rows: `1888`.
- Monitor log rows: `108`.
- Progress log SHA256:
  `68a6a69479986d8af15e8a51cef75f28ad1832b1e1a7f86feda7c7c7a371c347`
- Monitor log SHA256:
  `28d90087495edfe131a5a607861631cd96b7ed670221e64ea4e16e9111670fcd`
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `535.1616585830001`.
- Final monitor CPU seconds:
  `cpu_user_seconds=699.818548`, `cpu_system_seconds=3090.787628`.
- Final monitor progress line count: `1888`.

Progress event counts:

```text
v32_support_tournament_optimizer_progress  1152
development_evaluation_record_start         144
v32_support_tournament_margin_prepared      144
v32_support_tournament_optimizer_completed  144
development_evaluation_record_completed     144
train_edit_bank_progress                    102
inner_validation_candidate_start              6
train_edit_bank_start                         6
train_edit_bank_completed                     6
train_only_control_contexts_start             6
train_only_control_contexts_completed         6
development_evaluation_start                  6
development_evaluation_completed              6
inner_validation_candidate_completed          6
development_inputs_loaded                     1
train_statistics_start                        1
train_statistics_completed                    1
development_jobs_planned                      1
development_jobs_selected                     1
inner_validation_start                        1
inner_validation_rung_start                   1
inner_validation_rung_completed               1
inner_validation_completed                    1
development_setup_completed                   1
```

Monitor event counts:

```text
monitor_heartbeat  106
monitor_start        1
monitor_stop         1
```

Leak audit command over progress and monitor logs:

```bash
rg -n 'final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence' \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
```

Result: no matches.

## Interpretation

V34 rules out the simplest scalar locality-pressure fix. Smaller norm caps and
higher compatible-preservation weights did not reduce proof compatible-MSE
failures at all. Instead, they damaged target prediction and introduced more
control-margin failures.

Compared with V33's best replay candidate:

```text
V33 target_prediction_rate=0.8333333333333334
V33 compatible_mse_fail_count=24
V33 control_margin_fail_count=0

V34 best target_prediction_rate=0.6666666666666666
V34 best compatible_mse_fail_count=24
V34 best control_margin_fail_count=34
```

The next version should not keep tuning scalar compatible weights. It should add
an explicit source-preserving mechanism, most likely a support-side projection or
line-search that constrains source/compatible outputs directly while preserving a
minimum target-vs-runner support margin.

## Result Review Status

Accepted by Kepler with confidence `5/5`.

Reviewer summary:

- progress and monitor hashes, row counts, and terminal `monitor_stop` match the
  logs;
- candidate table matches `inner_validation_candidate_completed` events;
- every candidate has `compatible_mse_fail_count=24/24`, so the negative
  diagnostic conclusion is supported;
- verdict is conservative because V34 is not a success;
- leak audit over progress/monitor logs is clean;
- no sealed-final raw file was opened or read.
