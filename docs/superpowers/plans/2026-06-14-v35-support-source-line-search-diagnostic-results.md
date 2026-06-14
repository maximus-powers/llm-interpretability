# V35 Support Source Line Search Diagnostic Results

## Status

Review status: accepted by Kepler at confidence `5/5`.

V35 did not pass the development proof gates. It is a useful diagnostic:
support-only alpha line search reduced compatible-MSE failures from V33/V34's
`24/24` to `18/24` in the best locality candidates, but the edits still failed
the proof contract and introduced substantial control-margin failures.

## Command

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v35-support-source-line-search \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions
```

## Monitoring Evidence

- Process exited cleanly; no matching training PID remained after completion.
- `passed=false`.
- Progress rows: `1358`.
- Monitor rows: `74`.
- Monitor terminal event: `monitor_stop`.
- Final monitor elapsed seconds: `365.88223`.
- Final CPU user seconds: `476.351051`.
- Final CPU system seconds: `2007.306785`.
- Progress log SHA256:
  `deba4eadd65974d66209518af784fa5294785b22fe70107583b931c8c37285d8`.
- Monitor log SHA256:
  `31829b34cfdf8f164b09fe31a8fd14512ddac7f7c6ee66f6fa11cb09e3328ee1`.
- Progress log location SHA256:
  `36e44d8a323a29eb59872ffba3470525575cbdb60d136d346abac5ac46c36076`.
- Monitor log location SHA256:
  `3cd918b4b1bc052dc22d3dbbec4046994ada11d1355b0026965a63b0c9eba77e`.
- Candidate completion events: `4`.
- Alpha selection events: `96`.
- Final event: `inner_validation_completed`.
- Forbidden-field scan over progress and monitor logs found no matches for:
  `final_subjects|subject_id|weights|logits|gradient|selected_coordinates|support_examples|sequence`.

## Candidate Table

| config | target pred | mean target margin | best-control lift | shuffled lift | pareto | proof failures | compatible MSE fails | target pred fails | target margin fails | control margin fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7500 | 0.3483 | 0.5061 | 0.5115 | 1.0000 | 9 | 19 | 6 | 6 | 56 |
| 1 | 0.7083 | 0.3471 | 0.4994 | 0.5103 | 1.0000 | 9 | 18 | 7 | 6 | 66 |
| 2 | 0.7500 | 0.3810 | 0.5380 | 0.5442 | 1.0000 | 9 | 19 | 6 | 7 | 55 |
| 3 | 0.7083 | 0.3789 | 0.5322 | 0.5421 | 1.0000 | 9 | 18 | 7 | 7 | 66 |

Best selected config by inner-validation ranking:

- `config_index=2`
- `config_hash=077876adf78e8cb54ca103bd98104ad6a3059cc66e03a0d97235439bce55850e`
- `trust_norm_cap=1.25`
- `alpha_target_margin_floor=0.05`
- `target_prediction_rate=0.75`
- `mean_target_margin=0.3809892361362775`
- `proof_gate_failure_count=9`
- `compatible_mse_fail_count=19`
- `control_margin_fail_count=55`

## Alpha Selection Summary

Selected alpha counts across 96 proof records:

| alpha | count |
| ---: | ---: |
| 0.0 | 22 |
| 0.25 | 2 |
| 0.5 | 6 |
| 0.75 | 28 |
| 1.0 | 38 |

Selection modes:

| mode | count |
| --- | ---: |
| eligible_min_compatible_mse | 74 |
| fallback_penalized | 22 |

Every alpha-selection event carried `candidate_metrics_hash` and
`eligible_count`, binding the selected alpha to the normalized support scalar
candidate table without logging raw candidate details.

## Interpretation

V35 supports the narrower hypothesis that support-only source-preserving line
search can reduce compatible-MSE failures relative to the previous all-fail
state. The improvement is incomplete: best compatible-MSE failures are still
`18/24`, and the run has high control-margin failures. This suggests scalar
post-optimization shrinking is not enough.

The next promising direction should not be another scalar pressure grid. The
data points toward a true source-preserving projection or constrained edit
operator, such as an explicit compatible-function null-space or dual-projection
method, because V35 improved one locality metric only partially while preserving
substantial target lift.

## Non-Claims

- This is not a development proof success.
- This does not use or evaluate sealed-final data.
- This does not show robust locality.
- This does not justify final evaluation.
- This does not show the fixed probe set can decode into altered functional
  models at the desired standard yet.
