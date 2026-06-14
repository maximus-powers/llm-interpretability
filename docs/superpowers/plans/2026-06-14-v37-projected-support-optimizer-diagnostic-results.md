# V37 Projected Support Optimizer Diagnostic Results

## Status

Review status: accepted after Kepler review, confidence 5/5.

V37 did not pass the development proof gates. It is a useful diagnostic result:
target plasticity recovered substantially compared with V36, but compatible-MSE
locality regressed and remains the dominant blocker.

## Command

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v37-projected-support-optimizer \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

Terminal stdout/stderr was redirected to:

```text
/Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v37_projected_support_optimizer_terminal.log
```

## Monitoring Evidence

- Process exited with status `0`; no matching training PID remained after completion.
- Terminal summary reported `passed=false`.
- Progress rows: `974`.
- Monitor rows: `58`.
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `282.92810625000004`.
- Final CPU user seconds: `513.848107`.
- Final CPU system seconds: `1996.39847`.
- Progress log SHA256:
  `d0b7e581cddf1e17472a818c5fa590ab236bbaf8e74ad49c684eeb1827068ee7`.
- Monitor log SHA256:
  `98f0092d2f6e64092490ad2b339c22cffa354f7a3d28603ec63bcae2bc208d64`.
- Terminal log SHA256:
  `0bd180134beb9aa39cca025c98de7f5a2056c171eb748ecd1b308a731c2c750e`.
- Progress log location SHA256:
  `36e44d8a323a29eb59872ffba3470525575cbdb60d136d346abac5ac46c36076`.
- Monitor log location SHA256:
  `3cd918b4b1bc052dc22d3dbbec4046994ada11d1355b0026965a63b0c9eba77e`.
- Final inner-validation event: `inner_validation_completed`.
- Final progress row: `development_setup_completed`.

## Event Counts

| event | count |
| --- | ---: |
| `inner_validation_candidate_completed` | 4 |
| `development_evaluation_record_completed` | 96 |
| `v37_projected_optimizer_progress` | 480 |
| `v37_projected_optimizer_completed` | 96 |
| `v35_support_source_alpha_selected` | 96 |
| `inner_validation_rung_completed` | 1 |
| `inner_validation_completed` | 1 |
| `development_setup_completed` | 1 |

## Leak Scan

The broad raw-field scan overmatched allowed scalar field names such as
`projected_delta_norm`. The strict JSON-key scan found no matches for raw keys:

```text
"(final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence|compatible_jacobian|raw_delta|projected_delta)"
```

`spectral_basis_sha256` appears in candidate summaries as a provenance hash. It
is not a raw basis-vector leak.

## Candidate Table

| config | target pred | mean target margin | best-control lift | shuffled lift | pareto | proof failures | compatible MSE fails | target pred fails | target margin fails | control margin fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5000 | 0.2193 | 0.3749 | 0.3826 | 0.9167 | 18 | 17 | 12 | 10 | 110 |
| 1 | 0.2917 | 0.0531 | 0.2046 | 0.2163 | 0.9167 | 22 | 10 | 17 | 17 | 187 |
| 2 | 0.7083 | 0.2846 | 0.4410 | 0.4479 | 0.9583 | 11 | 19 | 7 | 8 | 66 |
| 3 | 0.2917 | 0.1595 | 0.3145 | 0.3227 | 0.9167 | 23 | 13 | 17 | 15 | 154 |

Best selected config by inner-validation ranking:

- `config_index=2`
- `config_hash=e280e859cfe4be5a3398115d2b34f8104f8f363cee10afe556eca2124413fa88`
- `target_prediction_rate=0.7083333333333334`
- `mean_target_margin=0.28460461435573353`
- `proof_gate_failure_count=11`
- `compatible_mse_fail_count=19`
- `control_margin_fail_count=66`
- `pareto_undominated_rate=0.9583333333333334`

Best locality candidate by compatible-MSE failure count:

- `config_index=1`
- `compatible_mse_fail_count=10`
- `target_prediction_rate=0.2916666666666667`
- `proof_gate_failure_count=22`

## Projection Summary

V37 optimizer-completion event count: `96`.

Preservation-energy ratio across optimizer-completion events:

```text
count=96
mean=0.3752587445390721
min=0.24999982118606567
max=0.5027034878730774
```

This confirms V37 used a looser preservation tradeoff than V36. That recovered
target plasticity, but also increased compatible-MSE proof failures.

## Alpha Selection Summary

Selected alpha counts across 96 proof records:

| alpha | count |
| ---: | ---: |
| 0.0 | 37 |
| 0.125 | 2 |
| 0.25 | 6 |
| 0.5 | 7 |
| 0.75 | 19 |
| 1.0 | 25 |

Selection modes:

| mode | count |
| --- | ---: |
| eligible_min_compatible_mse | 49 |
| fallback_penalized | 47 |

Compared with V36's `eligible_min_compatible_mse=3`, V37 often found support
edits that satisfied target/tournament constraints before heldout proof.

## Interpretation

V37 supports the diagnostic hypothesis for this iteration: optimizing through
the compatible-nullspace projection is much more target-aware than post-hoc
projection. V36's best target prediction was `0.125`; V37's best target
prediction rose to `0.7083`, with positive mean target margin.

However, V37 is not a proof success. The best target-plastic config fails
compatible-MSE locality on `19/24` records. The locality-friendliest config
still fails `10/24` compatible-MSE checks and has weak target prediction
(`0.2917`). The next direction should keep the target-aware projected optimizer
but add an explicit compatible-MSE gate/penalty schedule or a two-stage
selection that rejects high-compatible-MSE projected deltas before alpha
selection.

## Non-Claims

- This is not a development proof success.
- This does not use or evaluate sealed-final data.
- This does not justify final evaluation.
- This does not prove robust steering.
- This does not show compatible locality is solved.
- This supports a narrower claim: target-aware optimization inside the projected
  subspace restores plasticity relative to V36, at a locality cost.
