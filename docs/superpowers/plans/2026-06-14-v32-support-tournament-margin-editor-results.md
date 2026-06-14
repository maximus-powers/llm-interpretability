# V32 Support Tournament Margin Editor Results

Date: 2026-06-14

## Verdict

V32 is incremental positive diagnostic evidence, not a success.

The best final-rung candidate improved final target prediction over V31 from
`0.7916666666666666` to `0.8333333333333334`, but it still failed the
preregistered development gate because `target_prediction_rate < 0.85` and
`proof_gate_failure_count=7`.

No sealed-final evaluation was run.

## Reviewed Inputs

- Plan:
  `docs/superpowers/plans/2026-06-14-v32-support-tournament-margin-editor.md`
- Implementation files:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
  - `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Reviewer gates before compute:
  - V32 plan approved by Kepler with confidence `5/5`.
  - V32 implementation initially blocked at `4/5` for counts-only support tensor
    hashing.
  - Corrected implementation approved by Kepler with confidence `5/5` after
    adding content-bound support tensor hashing and a regression test.

## Verification Before Compute

Focused V32 tests after the provenance fix:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v32'
```

Result: `9 passed, 131 deselected`.

Full helper tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Result: `140 passed in 10.63s`.

Compile check:

```bash
python -m py_compile \
  model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Result: passed.

No linting was run.

## Bounded Development Command

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v32-support-tournament-margin \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
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
- Runtime config grid hash:
  `28af92b88a574c25b125fe589653a7b44c1687603220a0610674ce5ea4448cf3`
- Inner-validation plan hash:
  `20a69e0a760bdf48d7b5021e0202f95816023527eab903d865d8866c83005f07`

The runtime config grid hash differs from the raw V32 grid constant because the
runtime configs are bound to train-pool provenance fields before evaluation.

## Best Candidate

Best final-rung candidate:

```text
config_index=7
config_hash=7d91fa2c71afe3849b0587c49e9116d94b479b86581a840f85b3fc48c0bb42d8
matched_edit_source=support_tournament_margin_sparse
sparse_top_k=64
trust_norm_cap=1.25
sign_conflict_penalty=1.0
compatible_orthogonal_weight=0.15
tournament_margin_floor=0.15
tournament_margin_weight=1.0
target_prediction_rate=0.8333333333333334
mean_target_margin=0.5318264380718271
mean_matched_minus_best_control_target_margin=0.6864466418347016
mean_matched_minus_shuffled_signature_target_margin=0.6950739420584947
pareto_undominated_rate=1.0
proof_gate_failure_count=7
contract_failure_count=0
```

Best-candidate hashes:

- train edit bank hash:
  `49ae771c03a517770828ff250b975dc071fc8d64f0684033fbbf10c11fbfe32b`
- proof record hashes hash:
  `f1a0a966c063df90ce34b04e2113a576a8611c295078c7ace5e22c177578b842`
- spectral basis hash:
  `de8269c1e34e98fc0a6299f2e124aff39206fd47f9cccbbc1feab7b3b71c161f`

## Candidate Table

| Rung | Jobs | Config | Target rate | Mean target margin | Best-control lift | Shuffled lift | Pareto rate | Proof failures | Contract failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 0 | 0.8333333333 | 0.5009988329 | 0.6475607978 | 0.6526648617 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 1 | 0.8333333333 | 0.5009988329 | 0.6474177628 | 0.6526648617 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 2 | 0.8333333333 | 0.4984952006 | 0.6478151883 | 0.6531563205 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 3 | 0.8333333333 | 0.4984952006 | 0.6387862734 | 0.6531563205 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 4 | 0.7500000000 | 0.5016651973 | 0.6510377861 | 0.6563263171 | 1.0000000000 | 7 | 0 |
| 0 | 12 | 5 | 0.7500000000 | 0.5016651973 | 0.6475837438 | 0.6563263171 | 1.0000000000 | 7 | 0 |
| 0 | 12 | 6 | 0.8333333333 | 0.5233046220 | 0.6671881926 | 0.6779657418 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 7 | 0.8333333333 | 0.5233046220 | 0.6680268973 | 0.6779657418 | 1.0000000000 | 6 | 0 |
| 1 | 24 | 7 | 0.8333333333 | 0.5318264381 | 0.6864466418 | 0.6950739421 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 6 | 0.8333333333 | 0.5318264381 | 0.6864294665 | 0.6950739421 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 0 | 0.7916666667 | 0.5436561185 | 0.6998613218 | 0.7087760955 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 1 | 0.7916666667 | 0.5436561185 | 0.6949172344 | 0.7087760955 | 1.0000000000 | 7 | 0 |

## Direction Diagnostics

Best final-rung direction aggregates from redacted progress events:

| Direction | Records | Target rate | All-gate rate | Mean target margin | Mean delta norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| has_majority->mountain_pattern | 2 | 1.000 | 0.000 | 0.379482 | 1.250000 |
| has_majority->sorted_ascending | 2 | 1.000 | 0.000 | 0.645943 | 1.250000 |
| has_majority->sorted_descending | 2 | 1.000 | 0.000 | 0.730699 | 1.250000 |
| mountain_pattern->has_majority | 2 | 0.500 | 0.000 | -0.010891 | 1.250000 |
| mountain_pattern->sorted_ascending | 2 | 1.000 | 0.000 | 0.834007 | 1.250000 |
| mountain_pattern->sorted_descending | 2 | 1.000 | 0.000 | 0.695763 | 1.250000 |
| sorted_ascending->has_majority | 2 | 0.000 | 0.000 | 0.193979 | 1.250000 |
| sorted_ascending->mountain_pattern | 2 | 1.000 | 0.000 | 0.544656 | 1.250000 |
| sorted_ascending->sorted_descending | 2 | 1.000 | 0.000 | 0.869459 | 1.250000 |
| sorted_descending->has_majority | 2 | 0.500 | 0.000 | 0.089176 | 1.250000 |
| sorted_descending->mountain_pattern | 2 | 1.000 | 0.000 | 0.626457 | 1.250000 |
| sorted_descending->sorted_ascending | 2 | 1.000 | 0.000 | 0.783190 | 1.250000 |

V32 improved `mountain_pattern->has_majority` from V31's `0.0` to `0.5`, kept
`sorted_descending->has_majority` at `0.5`, and still failed
`sorted_ascending->has_majority`.

## Monitoring Evidence

- Process `50769` was no longer alive after completion.
- Progress log rows: `2616`.
- Monitor log rows: `194`.
- Progress log SHA256:
  `a77441fba07af51d7d45fa2e88500f5254db2fb483b11120efeb0d76ca5627e5`
- Monitor log SHA256:
  `1c9e7700977a75b9e5c4d1050c032c383965899b904632a835a07d569d93ea6b`
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `968.597884292`.
- Final monitor CPU seconds:
  `cpu_user_seconds=1271.761581`, `cpu_system_seconds=5727.626306`.
- Final monitor progress line count: `2616`.

Progress event counts:

```text
v32_support_tournament_optimizer_progress  1536
train_edit_bank_progress                    204
development_evaluation_record_start         192
v32_support_tournament_margin_prepared      192
v32_support_tournament_optimizer_completed  192
development_evaluation_record_completed     192
inner_validation_candidate_start             12
train_edit_bank_start                        12
train_edit_bank_completed                    12
train_only_control_contexts_start            12
train_only_control_contexts_completed        12
development_evaluation_start                 12
development_evaluation_completed             12
inner_validation_candidate_completed         12
inner_validation_rung_start                   2
inner_validation_rung_completed               2
development_inputs_loaded                     1
train_statistics_start                        1
train_statistics_completed                    1
development_jobs_planned                      1
development_jobs_selected                     1
inner_validation_start                        1
inner_validation_completed                    1
development_setup_completed                   1
```

Monitor event counts:

```text
monitor_heartbeat  192
monitor_start        1
monitor_stop         1
```

Leak audit command over progress and monitor logs:

```bash
rg -n 'final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence' \
  runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl \
  runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
```

Result: no matches.

## Interpretation

V32 supports the narrower claim that aligning the support objective with the
proof's predicted-behavior tournament criterion helps target prediction on this
small development benchmark. It improved V31's final target rate from `0.7917`
to `0.8333` and partially improved target `has_majority`.

V32 does not support a success claim. It remains below the target prediction
gate and still has seven proof failures. All best-candidate direction-level
`all_gate_rate` values are zero, so the proof margin/control thresholds remain
the main blocker even when target prediction improves.

The next useful diagnostic should probably target proof-gate decomposition:
which proof sub-gates fail after target prediction succeeds, and whether those
failures are margin thresholds, control advantages, or source/locality effects.
That diagnostic should be redacted and development-only.

## Result Review Status

Accepted by Kepler with confidence `5/5`.

Reviewer summary:

- verdict is appropriately conservative;
- candidate table, best candidate, and direction diagnostics match redacted
  progress rows;
- monitor evidence is sufficient and no compute process remains;
- log hashes match the results doc;
- forbidden-field and guard scans over progress/monitor logs are clean;
- next work should decompose proof-gate failures rather than increase capacity.
