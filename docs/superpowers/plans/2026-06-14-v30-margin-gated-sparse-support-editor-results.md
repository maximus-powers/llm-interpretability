# V30 Margin-Gated Sparse Support Editor Results

Date: 2026-06-14

## Verdict

V30 is a moderate positive development diagnostic, not a success.

The best final-rung candidate improved over V29 from `target_prediction_rate=0.5`
to `0.6666666666666666`, while preserving `contract_failure_count=0` and
positive matched-vs-control margins. It still failed the preregistered
development gate because `target_prediction_rate < 0.85` and
`proof_gate_failure_count=10`.

No sealed-final evaluation was run.

## Reviewed Inputs

- Plan: `docs/superpowers/plans/2026-06-14-v30-margin-gated-sparse-support-editor.md`
- Implementation files:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
  - `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Reviewer gate before compute: Kepler returned `5/5` for the V30 plan and
  implementation package before the bounded run.

## Literature Context Used

- Yang et al., "Fine-tuning Done Right in Model Editing",
  https://arxiv.org/abs/2509.22072
- Iurada et al., "Efficient Model Editing with Task-Localized Sparse
  Fine-tuning", https://arxiv.org/abs/2504.02620
- Gangadhar and Stratos, "Model Editing by Pure Fine-Tuning",
  https://aclanthology.org/2024.findings-acl.352/
- Tan et al., "Massive Editing for Large Language Models via Meta Learning",
  https://arxiv.org/abs/2311.04661
- Yadav et al., "TIES-Merging", https://arxiv.org/abs/2306.01708
- Ortiz-Jimenez et al., "Task Arithmetic in the Tangent Space",
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html
- Mitchell et al., "Fast Model Editing at Scale", https://arxiv.org/abs/2110.11309

## Verification Before Compute

Focused V30 tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v30'
```

Result: `7 passed, 116 deselected`.

Compile check:

```bash
python -m py_compile \
  model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py
```

Result: passed.

Full helper tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Result: `123 passed in 10.71s`.

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
  --inner-validation-config-grid v30-margin-gated-sparse \
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
  `3db058a3ced82c4f6d2a3fcf020072300a687c257b6a6643d5203962edb686d5`
- Inner-validation plan hash:
  `ec4e7fb11ecf1916cf4b531d1469ec7f352156fc8ab733eb335ec36b73e8ba39`

The runtime config grid hash differs from the raw V30 grid constant because the
runtime configs are bound to train-pool provenance fields before evaluation.

## Best Candidate

Best final-rung candidate:

```text
config_index=7
config_hash=2d13597acb01bf41de5afa132f21f95f25eafe93e1fcdb4faf2e4111d54f1e68
sparse_top_k=64
trust_norm_cap=1.25
target_margin_floor=0.25
compatible_floor=0.05
extra_compatible_weight=0.05
target_prediction_rate=0.6666666666666666
mean_target_margin=0.32473520809435286
mean_matched_minus_best_control_target_margin=0.4790057574898583
mean_matched_minus_shuffled_signature_target_margin=0.4879827120810205
pareto_undominated_rate=1.0
proof_gate_failure_count=10
contract_failure_count=0
```

Best-candidate hashes:

- train edit bank hash:
  `49ae771c03a517770828ff250b975dc071fc8d64f0684033fbbf10c11fbfe32b`
- proof record hashes hash:
  `0e05f049efa11bb9867cb7379814a95026cfa6cf8ba05acaaeada4bb4eb31885`
- spectral basis hash:
  `de8269c1e34e98fc0a6299f2e124aff39206fd47f9cccbbc1feab7b3b71c161f`

## Candidate Table

| Rung | Jobs | Config | Target rate | Mean target margin | Best-control lift | Shuffled lift | Pareto rate | Proof failures | Contract failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 0 | 0.5000000000 | 0.1959909207 | 0.3381336569 | 0.3506520405 | 0.9166666667 | 17 | 0 |
| 0 | 12 | 1 | 0.5000000000 | 0.1768421499 | 0.3262160074 | 0.3315032697 | 0.9166666667 | 17 | 0 |
| 0 | 12 | 2 | 0.5000000000 | 0.1863971248 | 0.3335667349 | 0.3410582447 | 0.9166666667 | 16 | 0 |
| 0 | 12 | 3 | 0.5000000000 | 0.1765131514 | 0.3226437366 | 0.3311742712 | 0.9166666667 | 16 | 0 |
| 0 | 12 | 4 | 0.5833333333 | 0.2371589986 | 0.3781924780 | 0.3918201184 | 1.0000000000 | 13 | 0 |
| 0 | 12 | 5 | 0.5833333333 | 0.2421472841 | 0.3914233241 | 0.3968084039 | 1.0000000000 | 13 | 0 |
| 0 | 12 | 6 | 0.6666666667 | 0.3367022940 | 0.4801343632 | 0.4913634138 | 1.0000000000 | 9 | 0 |
| 0 | 12 | 7 | 0.6666666667 | 0.3090218796 | 0.4555562383 | 0.4636829994 | 1.0000000000 | 9 | 0 |
| 1 | 24 | 6 | 0.6666666667 | 0.3238539691 | 0.4780302359 | 0.4871014731 | 1.0000000000 | 10 | 0 |
| 1 | 24 | 7 | 0.6666666667 | 0.3247352081 | 0.4790057575 | 0.4879827121 | 1.0000000000 | 10 | 0 |
| 1 | 24 | 5 | 0.6666666667 | 0.2450086913 | 0.4017981525 | 0.4082561953 | 0.9583333333 | 14 | 0 |
| 1 | 24 | 4 | 0.6666666667 | 0.2549029908 | 0.4079555250 | 0.4181504948 | 0.9583333333 | 13 | 0 |

## Monitoring Evidence

- Process `7337` was no longer alive after completion.
- Progress log rows: `2616`.
- Monitor log rows: `181`.
- Progress log SHA256:
  `784b40d9dddd34ebad66b785f8eada757e42c01cc613a8500d6556f545d694c9`
- Monitor log SHA256:
  `fed88258f5a10a77f1ba98736580c17f0d5cf965ea5f627e8d6f8e2ec2bced01`
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `899.48288275`.
- Final monitor CPU seconds:
  `cpu_user_seconds=1230.66961`, `cpu_system_seconds=5743.750984`.
- Final monitor progress line count: `2616`.

Progress event counts:

```text
v30_margin_gated_optimizer_progress        1536
train_edit_bank_progress                    204
development_evaluation_record_start         192
v30_sparse_coordinate_selection_completed   192
v30_margin_gated_optimizer_completed        192
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
monitor_heartbeat  179
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

V30 supports the narrower claim that margin-gated sparse support editing is a
better development candidate than V29 under the current small-scale setup. The
increase to `sparse_top_k=64` and `trust_norm_cap=1.25` raised target flips and
margins without introducing contract failures.

V30 does not support a claim that the fixed probe set is sufficient for reliable
functional model editing. The remaining proof failures show that the learned
edit direction is still not robust across all balanced development directions.

The most likely next step is to diagnose the failure cases by redacted
direction-level aggregates and then test a sign-conflict-aware sparse merge or
per-direction support gate. This is consistent with the V30 literature basis:
TIES-style sign conflict handling and task-localized sparse fine-tuning both
predict that blindly expanding capacity can help but will not fully solve
interfering directions.

## Result Review Status

Accepted by Kepler with confidence `5/5`.

Reviewer summary:

- no blockers;
- verdict is appropriately conservative;
- log hashes and best-candidate metrics match files on disk;
- monitor evidence is sufficient to show progress and clean stop;
- leak check passed for the requested terms over progress/monitor logs;
- raw-grid versus runtime-grid hash discrepancy is adequately explained.
