# V33 Proof Gate Decomposition Diagnostic Results

Date: 2026-06-14

## Verdict

V33 is a successful diagnostic run, not a model-editing success.

The bounded replay reproduced V32-level target prediction at
`0.8333333333333334`, below the existing `0.85` gate. The new proof-gate
breakdown shows the dominant all-gate blocker is compatible-source MSE/locality:
`compatible_mse_fail_count=24` out of `24` records for both replayed configs.

No sealed-final evaluation was run.

## Reviewed Inputs

- Plan:
  `docs/superpowers/plans/2026-06-14-v33-proof-gate-decomposition-diagnostic.md`
- Implementation files:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
  - `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Reviewer gates:
  - V33 plan approved by Kepler with confidence `5/5`.
  - Initial implementation blocked at confidence `4/5` because V33 candidates
    inherited the V32 variant label from the reused edit source.
  - Corrected implementation approved by Kepler with confidence `5/5` after the
    candidate variant was bound to the active run variant and covered by a V33
    regression test.

## Literature Basis

- Liu et al., "Are We Evaluating the Edit Locality of LLM Model Editing
  Properly?" (https://arxiv.org/pdf/2601.17343): motivates behavior-sensitive
  locality/specificity diagnostics.
- He et al., "Knowledge Updating? No More Model Editing! Just Selective
  Contextual Reasoning" (https://arxiv.org/html/2503.05212v1): motivates
  separating reliability, generalization, locality, and portability rather than
  treating edit success as one number.
- Balloccu et al., "Leak, Cheat, Repeat"
  (https://aclanthology.org/2024.eacl-long.5/): motivates explicit leakage,
  baseline, and reproducibility audits.
- Han et al., "A Survey of Weight Space Learning"
  (https://arxiv.org/html/2603.10090v1) and Kaushik et al., "The Universal
  Weight Subspace Hypothesis" (https://arxiv.org/abs/2512.05117): support the
  broader weight-space framing while reinforcing that functional proof gates
  remain necessary.

## Verification Before Compute

Focused V33/proof diagnostic tests:

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q -k 'v33 or proof_record_exposes_redacted_gate_decomposition or aggregate_records_compute_required_gate_metrics or development_job_evaluator_progress_logs_redacted_summary'
```

Result: `5 passed, 138 deselected in 6.54s`.

Full helper tests:

```bash
python -m pytest \
  /Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py \
  -q
```

Result: `143 passed in 10.94s`.

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
  --inner-validation-config-grid v33-proof-gate-diagnostic \
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
- V33 runtime config grid hash:
  `94322ccee0e42d8b51442c250ba8d928de1b3ad6c954f4f90dd7e77fcd80b832`
- V33 inner-validation plan hash:
  `d1a2385737b33feb47b76f46fe9ff464e26fbfa11804c5b1836f346e56cfa522`

## Candidate Table

| Config | Hash | Target rate | Mean target margin | Best-control lift | Shuffled lift | Pareto rate | Proof failures | Compatible MSE fails | Target pred fails | Target margin fails | Control margin fails |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `2830f89d58d5918cfc6aa0ab84cdb3e6e40ce4347bdd2b60ce06ea7c88b3b6f4` | 0.8333333333 | 0.5318264381 | 0.6873395247 | 0.6950739421 | 1.0000000000 | 7 | 24 | 4 | 4 | 0 |
| 0 | `8e1980446264fe683fa2cc73015b177af3c747098ef06309fafcaaa25b775d18` | 0.8333333333 | 0.5318264381 | 0.6869245880 | 0.6950739421 | 1.0000000000 | 7 | 24 | 4 | 4 | 0 |

Best candidate:

```text
config_index=1
matched_edit_source=support_tournament_margin_sparse
compatible_orthogonal_weight=0.05
sparse_top_k=64
trust_norm_cap=1.25
tournament_margin_floor=0.15
tournament_margin_weight=1.0
target_prediction_rate=0.8333333333333334
mean_target_margin=0.5318264380718271
mean_matched_minus_best_control_target_margin=0.6873395247382632
mean_matched_minus_shuffled_signature_target_margin=0.6950739420584947
pareto_undominated_rate=1.0
proof_gate_failure_count=7
contract_failure_count=0
```

Best-candidate proof-gate breakdown:

```text
record_count=24
compatible_mse_fail_count=24
target_prediction_fail_count=4
target_margin_fail_count=4
pareto_fail_count=0
control_margin_fail_count=0
control_margin_failure_record_count=0
mean_control_margin_advantage=0.6964630704336136
min_control_margin_advantage=0.049317045602947474
control_margin_failure_type_counts_hash=0ac3083f31ae2144d07166aa2f15a82146243c25f65e40129d567cc5efa05710
```

## Direction Breakdown

Best-candidate redacted progress rows show:

| Direction | Records | Target pred pass | Target margin pass | Compatible MSE pass | Pareto pass | Control margin fails | All gate pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| has_majority->mountain_pattern | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| has_majority->sorted_ascending | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| has_majority->sorted_descending | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| mountain_pattern->has_majority | 2 | 1 | 0 | 0 | 2 | 0 | 0 |
| mountain_pattern->sorted_ascending | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| mountain_pattern->sorted_descending | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| sorted_ascending->has_majority | 2 | 0 | 2 | 0 | 2 | 0 | 0 |
| sorted_ascending->mountain_pattern | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| sorted_ascending->sorted_descending | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| sorted_descending->has_majority | 2 | 1 | 0 | 0 | 2 | 0 | 0 |
| sorted_descending->mountain_pattern | 2 | 2 | 2 | 0 | 2 | 0 | 0 |
| sorted_descending->sorted_ascending | 2 | 2 | 2 | 0 | 2 | 0 | 0 |

## Monitoring Evidence

- Process `24737` was no longer alive after completion.
- Progress log rows: `636`.
- Monitor log rows: `38`.
- Progress log SHA256:
  `8badb931da86711d49bb50eceab030373e1afa14c5c18abf7de2900a8ac611ce`
- Monitor log SHA256:
  `e7c8f19385d784dff1ae01ef3c96b1231953d82fa6191259d3e44f073fa860f4`
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `181.641115375`.
- Final monitor CPU seconds:
  `cpu_user_seconds=237.94511`, `cpu_system_seconds=1011.221554`.
- Final monitor progress line count: `636`.

Progress event counts:

```text
v32_support_tournament_optimizer_progress  384
development_evaluation_record_start          48
v32_support_tournament_margin_prepared       48
v32_support_tournament_optimizer_completed   48
development_evaluation_record_completed      48
train_edit_bank_progress                     34
inner_validation_candidate_start              2
train_edit_bank_start                         2
train_edit_bank_completed                     2
train_only_control_contexts_start             2
train_only_control_contexts_completed         2
development_evaluation_start                  2
development_evaluation_completed              2
inner_validation_candidate_completed          2
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
monitor_heartbeat  36
monitor_start       1
monitor_stop        1
```

Leak audit command over progress and monitor logs:

```bash
rg -n 'final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence' \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/development_progress.jsonl \
  /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/long_run_monitor.jsonl
```

Result: no matches.

## Interpretation

V33 clarifies that V32's all-gate failures are not primarily caused by weak
control advantages or Pareto domination. Both replay configs beat proof-critical
controls by target margin on every record and have `pareto_fail_count=0`.

The main blocker is compatible-source MSE/locality: every record fails the
compatible-MSE gate. The secondary target-behavior blocker remains target
`has_majority`, with failures concentrated in:

```text
mountain_pattern->has_majority: target_prediction_pass=1/2, target_margin_pass=0/2
sorted_ascending->has_majority: target_prediction_pass=0/2, target_margin_pass=2/2
sorted_descending->has_majority: target_prediction_pass=1/2, target_margin_pass=0/2
```

The next editor version should therefore stop treating the remaining proof
failure as a generic margin/capacity issue. The most promising direction is a
locality-constrained edit: either add an explicit compatible-MSE projection or
line-search after the support-tournament edit, or optimize a bi-objective edit
that preserves compatible source outputs while keeping the target-vs-runner
margin.

## Result Review Status

Accepted by Kepler with confidence `5/5`.

Reviewer summary:

- progress and monitor hashes, row counts, and terminal `monitor_stop` match the
  logs;
- candidate metrics and proof-gate decomposition match progress events;
- verdict is conservative because V33 is diagnostic, not success;
- leak audit over progress/monitor logs is clean;
- no sealed-final raw file was opened or read.
