# V36 Compatible-Nullspace Projection Diagnostic Results

## Status

Review status: accepted after Kepler review, confidence 5/5.

V36 did not pass the development proof gates. It is a strong negative/diagnostic
result: raw compatible-Jacobian projection sharply reduced compatible-MSE
failures, but it removed too much target plasticity.

## Command

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v36-compatible-nullspace-projection \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

## Monitoring Evidence

- Process exited cleanly; no matching training PID remained after completion.
- `passed=false`.
- Progress rows: `1358`.
- Monitor rows: `46`.
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `224.93872950000002`.
- Final CPU user seconds: `411.417041`.
- Final CPU system seconds: `1494.932359`.
- Progress log SHA256:
  `87b063db1afad61063bf216ead8e4ca9679f1327dfda943d4d23d427f335d5af`.
- Monitor log SHA256:
  `a44ec70aa4a76504f68b54f402f9e20c3108b0bd8e39b07237e8959bc1756d85`.
- Progress log location SHA256:
  `36e44d8a323a29eb59872ffba3470525575cbdb60d136d346abac5ac46c36076`.
- Monitor log location SHA256:
  `3cd918b4b1bc052dc22d3dbbec4046994ada11d1355b0026965a63b0c9eba77e`.
- Candidate completion events: `4`.
- Projection completion events: `96`.
- Alpha selection events: `96`.
- Final inner-validation event: `inner_validation_completed`.
- Final progress row: `development_setup_completed`.

## Leak Scan

The first broad regex from the plan included the bare substring `basis`, which
matched only existing redacted `spectral_basis_sha256` fields in candidate
summaries. That was a false positive for raw leakage, not a raw basis-vector
leak. The plan was corrected to avoid this hash-field false positive.

Refined raw-field scan over progress and monitor logs found no matches for:

```text
final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence|compatible_jacobian|raw_delta
```

## Candidate Table

| config | target pred | mean target margin | best-control lift | shuffled lift | pareto | proof failures | compatible MSE fails | target pred fails | target margin fails | control margin fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1250 | -0.0661 | 0.0896 | 0.0972 | 1.0000 | 25 | 11 | 21 | 21 | 199 |
| 1 | 0.0000 | -0.1493 | 0.0084 | 0.0139 | 0.6250 | 41 | 4 | 24 | 24 | 225 |
| 2 | 0.0833 | -0.0635 | 0.0904 | 0.0998 | 0.9583 | 27 | 11 | 22 | 21 | 199 |
| 3 | 0.0000 | -0.1471 | 0.0009 | 0.0161 | 0.7500 | 40 | 8 | 24 | 24 | 223 |

Best selected config by inner-validation ranking:

- `config_index=0`
- `config_hash=ea0f8fd80d332fdbaac0d6f927cd77d047f43399adb2cfb0922636c547500e02`
- `compatible_nullspace_rtol=0.0001`
- `projection_strength=0.75`
- `target_prediction_rate=0.125`
- `mean_target_margin=-0.06607986835691311`
- `proof_gate_failure_count=25`
- `compatible_mse_fail_count=11`
- `control_margin_fail_count=199`

Best compatible-MSE locality candidate:

- `config_index=1`
- `compatible_mse_fail_count=4`
- `target_prediction_rate=0.0`
- `target_prediction_fail_count=24`
- `target_margin_fail_count=24`
- `proof_gate_failure_count=41`

## Projection Summary

Projection event count: `96`.

Preservation-energy ratio across projection events:

```text
count=96
mean=0.12544
min=2.7261969045317465e-7
max=0.25006959197543244
```

This confirms the projection substantially reduced first-order compatible
logit-change energy as designed.

## Alpha Selection Summary

Selected alpha counts across 96 proof records:

| alpha | count |
| ---: | ---: |
| 0.0 | 33 |
| 0.125 | 7 |
| 0.25 | 4 |
| 0.5 | 7 |
| 0.75 | 10 |
| 1.0 | 35 |

Selection modes:

| mode | count |
| --- | ---: |
| eligible_min_compatible_mse | 3 |
| fallback_penalized | 93 |

The high fallback rate means the projected deltas usually failed the support
target/tournament constraints before proof evaluation.

## Interpretation

V36 validates that compatible-Jacobian null-space projection can preserve source
behavior much more aggressively than scalar line search. It reduced
compatible-MSE failures from V35's best `18/24` to as low as `4/24`.

However, it also collapsed target plasticity: target prediction fell to
`0.0-0.125`, and mean target margins became negative for every config. The
projection is therefore too strong or too blunt when applied post hoc to the
V32 sparse delta.

The next step should keep the source-preserving projection idea but make the
edit direction target-aware inside the projected subspace, rather than
projecting after an unconstrained sparse optimizer. A promising V37 direction is
a constrained projected optimizer that optimizes support target/tournament loss
within the compatible null-space basis, with an explicit minimum plasticity
gate, instead of post-hoc projection.

## Non-Claims

- This is not a development proof success.
- This does not use or evaluate sealed-final data.
- This does not show robust target steering.
- This does not justify final evaluation.
- This does not prove the projection approach is wrong; it shows post-hoc
  projection of the V32 sparse delta over-preserves source behavior.
