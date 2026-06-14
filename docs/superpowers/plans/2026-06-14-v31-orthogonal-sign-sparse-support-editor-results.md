# V31 Orthogonal Sign-Sparse Support Editor Results

Date: 2026-06-14

## Verdict

V31 is a stronger positive development diagnostic than V30, but still not a
success.

The best final-rung candidate improved over V30 from
`target_prediction_rate=0.6666666666666666` to `0.7916666666666666`, reduced
proof failures from `10` to `7`, preserved `contract_failure_count=0`, and
increased matched-vs-control margins. It still failed the preregistered
development gate because `target_prediction_rate < 0.85` and
`proof_gate_failure_count > 0`.

No sealed-final evaluation was run.

## Reviewed Inputs

- Plan:
  `docs/superpowers/plans/2026-06-14-v31-orthogonal-sign-sparse-support-editor.md`
- Implementation files:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py`
  - `model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py`
- Reviewer gates before compute:
  - V31 plan approved by Kepler with confidence `5/5`.
  - V31 implementation approved by Kepler with confidence `5/5`.

## Literature Context Used

- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models",
  https://papers.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf
- Iurada et al., "Efficient Model Editing with Task-Localized Sparse
  Fine-tuning", https://arxiv.org/abs/2504.02620
- Porrello et al., "Dataless Weight Disentanglement in Task Arithmetic via
  Kronecker-Factored Approximate Curvature", https://arxiv.org/abs/2602.17385
- Sommariva et al., "Distilling Linearized Behavior into Non-Linear Fine-Tuning
  for Effective Task Arithmetic", https://arxiv.org/abs/2605.18993
- "Understanding and Enforcing Weight Disentanglement in Task Arithmetic",
  https://arxiv.org/html/2604.17078v1
- Gu and Yeung-Levy, "Foundation Models Secretly Understand Neural Network
  Weights", https://arxiv.org/abs/2503.00838
- Zhou, "Universal Hypernetworks for Arbitrary Models",
  https://arxiv.org/abs/2604.02215

## Verification Before Compute

Focused V31 tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q -k 'v31'
```

Result: `8 passed, 123 deselected`.

Full helper tests:

```bash
python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py -q
```

Result: `131 passed in 10.36s`.

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
  --inner-validation-config-grid v31-orthogonal-sign-sparse \
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
  `15877eee7219917fada8b6295ccdf3030b8a7f6c66ac5d01ebac7d8c30aaf21e`
- Inner-validation plan hash:
  `10b3f7493e917c75a785449839fa56f62406574aeed9aea6fa327f3f028421d1`

The runtime config grid hash differs from the raw V31 grid constant because the
runtime configs are bound to train-pool provenance fields before evaluation.

## Best Candidate

Best final-rung candidate:

```text
config_index=2
config_hash=7b8a7b84263d4165d7da8eb46f8fb0d2d710673828fdea6c356a29d8ae2b75f6
matched_edit_source=orthogonal_sign_sparse_support
sparse_top_k=64
trust_norm_cap=1.25
sign_conflict_penalty=1.0
compatible_orthogonal_weight=0.05
target_margin_floor=0.25
compatible_floor=0.05
extra_compatible_weight=0.05
hard_target_margin_weight=1.0
target_prediction_rate=0.7916666666666666
mean_target_margin=0.567085697936515
mean_matched_minus_best_control_target_margin=0.7222636344971534
mean_matched_minus_shuffled_signature_target_margin=0.7303332019231826
pareto_undominated_rate=1.0
proof_gate_failure_count=7
contract_failure_count=0
```

Best-candidate hashes:

- train edit bank hash:
  `49ae771c03a517770828ff250b975dc071fc8d64f0684033fbbf10c11fbfe32b`
- proof record hashes hash:
  `598b2b689a1bf02b6e49958c22e429c81dc00d173b915e6bc3c47a0a052961b8`
- spectral basis hash:
  `de8269c1e34e98fc0a6299f2e124aff39206fd47f9cccbbc1feab7b3b71c161f`

## Candidate Table

| Rung | Jobs | Config | Target rate | Mean target margin | Best-control lift | Shuffled lift | Pareto rate | Proof failures | Contract failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 0 | 0.7500000000 | 0.5438160046 | 0.6926360604 | 0.6984771244 | 1.0000000000 | 7 | 0 |
| 0 | 12 | 1 | 0.7500000000 | 0.5438160046 | 0.6835709986 | 0.6984771244 | 1.0000000000 | 7 | 0 |
| 0 | 12 | 2 | 0.8333333333 | 0.5417747827 | 0.6910646203 | 0.6964359025 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 3 | 0.8333333333 | 0.5417747827 | 0.6845673154 | 0.6964359025 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 4 | 0.7500000000 | 0.4848241073 | 0.6262342279 | 0.6394852272 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 5 | 0.7500000000 | 0.4848241073 | 0.6329993783 | 0.6394852272 | 1.0000000000 | 6 | 0 |
| 0 | 12 | 6 | 0.7500000000 | 0.5409253550 | 0.6892726951 | 0.6955864748 | 1.0000000000 | 7 | 0 |
| 0 | 12 | 7 | 0.7500000000 | 0.5409253550 | 0.6822904692 | 0.6955864748 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 2 | 0.7916666667 | 0.5670856979 | 0.7222636345 | 0.7303332019 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 3 | 0.7916666667 | 0.5670856979 | 0.7193310204 | 0.7303332019 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 0 | 0.7916666667 | 0.5504754297 | 0.7068024552 | 0.7137229337 | 1.0000000000 | 7 | 0 |
| 1 | 24 | 1 | 0.7916666667 | 0.5504754297 | 0.7009834773 | 0.7137229337 | 1.0000000000 | 7 | 0 |

## Direction Diagnostics

Best final-rung direction aggregates from redacted progress events:

| Direction | Records | Target rate | All-gate rate | Mean target margin | Mean delta norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| has_majority->mountain_pattern | 2 | 1.000 | 0.000 | 0.387056 | 1.250000 |
| has_majority->sorted_ascending | 2 | 1.000 | 0.000 | 0.658322 | 1.250000 |
| has_majority->sorted_descending | 2 | 1.000 | 0.000 | 0.739857 | 1.250000 |
| mountain_pattern->has_majority | 2 | 0.000 | 0.000 | -0.019759 | 1.250000 |
| mountain_pattern->sorted_ascending | 2 | 1.000 | 0.000 | 0.842746 | 1.250000 |
| mountain_pattern->sorted_descending | 2 | 1.000 | 0.000 | 0.746883 | 1.250000 |
| sorted_ascending->has_majority | 2 | 0.000 | 0.000 | 0.207635 | 1.250000 |
| sorted_ascending->mountain_pattern | 2 | 1.000 | 0.000 | 0.814501 | 1.250000 |
| sorted_ascending->sorted_descending | 2 | 1.000 | 0.000 | 0.879479 | 1.250000 |
| sorted_descending->has_majority | 2 | 0.500 | 0.000 | 0.083126 | 1.250000 |
| sorted_descending->mountain_pattern | 2 | 1.000 | 0.000 | 0.788029 | 1.250000 |
| sorted_descending->sorted_ascending | 2 | 1.000 | 0.000 | 0.677154 | 1.250000 |

V31 fixed `mountain_pattern->sorted_descending` relative to V30 and partially
improved `sorted_descending->has_majority`. The remaining target failures are
still concentrated in target `has_majority`, especially
`mountain_pattern->has_majority` and `sorted_ascending->has_majority`.

## Monitoring Evidence

- Process `83586` was no longer alive after completion.
- Progress log rows: `2616`.
- Monitor log rows: `178`.
- Progress log SHA256:
  `53109605ce464b0465046ee399d5c4683a099478014c2286bd24b5478c799dc6`
- Monitor log SHA256:
  `cad7774911914cbe8d6067a8243ecf42e80d9b6c243f21e2ccceb1a9ec3f428b`
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `883.9199528749999`.
- Final monitor CPU seconds:
  `cpu_user_seconds=1192.145365`, `cpu_system_seconds=5823.375438`.
- Final monitor progress line count: `2616`.

Progress event counts:

```text
v31_orthogonal_sign_optimizer_progress        1536
train_edit_bank_progress                       204
development_evaluation_record_start            192
v31_sign_coherent_coordinate_selection_completed 192
v31_orthogonal_sign_optimizer_completed        192
development_evaluation_record_completed        192
inner_validation_candidate_start                12
train_edit_bank_start                           12
train_edit_bank_completed                       12
train_only_control_contexts_start               12
train_only_control_contexts_completed           12
development_evaluation_start                    12
development_evaluation_completed                12
inner_validation_candidate_completed            12
inner_validation_rung_start                      2
inner_validation_rung_completed                  2
development_inputs_loaded                        1
train_statistics_start                           1
train_statistics_completed                       1
development_jobs_planned                         1
development_jobs_selected                        1
inner_validation_start                           1
inner_validation_completed                       1
development_setup_completed                      1
```

Monitor event counts:

```text
monitor_heartbeat  176
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

V31 supports the claim that sign-conflict-aware sparse coordinate selection and
compatible-gradient orthogonality are useful in this small-scale development
setup. The target rate, margins, and proof failures all improved over V30 while
contract failures stayed at zero.

V31 does not support a success claim. The final rung regressed from first-rung
`0.8333333333` to `0.7916666667`, so the smaller first-rung estimate was
optimistic. The remaining failures are not random across directions; target
`has_majority` remains the key weakness.

The bounded `trust_norm_cap=1.5` arms did not survive successive halving.
Capacity alone is therefore not the most supported next explanation. The best
candidate used `trust_norm_cap=1.25` with the stronger sign conflict penalty and
lower orthogonality weight.

## Result Review Status

Accepted by Kepler with confidence `5/5`.

Reviewer summary:

- verdict is appropriately conservative;
- final-rung best candidate and direction diagnostics match the redacted
  progress log;
- monitor evidence is sufficient and no compute process remains;
- progress and monitor log hashes match the results doc;
- leak scan over progress/monitor logs is clean;
- `trust_norm_cap=1.5` is not supported because those configs did not reach the
  final rung.
