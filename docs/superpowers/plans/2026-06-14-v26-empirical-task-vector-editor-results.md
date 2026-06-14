# V26 Empirical Task-Vector Editor Results

Date: 2026-06-14

## Verdict

Accepted as a bounded negative diagnostic result only.

This run does not support successful functional weight editing or behavioral
steering. Every evaluated candidate had `target_prediction_rate=0.0`, the final
summary had `passed=false`, and the reviewer approved only the negative framing.

## Command

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Long-Run Observability

- Process PID `9959` exited cleanly.
- Post-run process scan for
  `train_four_behavior_functional_weight_editing_v25|hypernet.train|run_compute_packet`
  returned no matches.
- `long_run_monitor.jsonl` ended with `monitor_stop`.
- Final monitor elapsed seconds: `1305.230559334`.
- Final monitor latest progress event: `development_setup_completed`.
- Final monitor progress line count: `841`.
- `development_progress.jsonl` SHA-256:
  `8f59e7c3f51e07809e74c33dde17ddf06293650b3cb8b20746253d6a20d45975`.
- `long_run_monitor.jsonl` SHA-256:
  `454ae02e2e12961666cab6740fd4b69ed5a1776bac1bce1c7f4e51c69a07d763`.

## Data Boundaries

- Final raw file remained sealed:
  `runs/four_behavior_functional_weight_editing_v25_pools/final_subjects.json`
  was not opened or read.
- Train pool: `264` records; behavior counts `72/64/64/64`.
- Train pool file SHA-256:
  `888d539fe8efefcaad91bb6ce0ee48c55f3903d2ed75b6791c4c8b314c0bc35d`.
- Train summary hash:
  `d24ab4c41f79ec97e8c936817eed723a24afed7d3259765e111bd970733d141e`.
- Development pool: `98` records; behavior counts `26/24/24/24`.
- Development pool file SHA-256:
  `d26f7506cd919de5eeabd9be9ebe205c707d04f169368698849484c3d819a659`.
- Development selection: `24` jobs, exactly `2` per non-identity direction.
- Selected jobs hash:
  `337ac71f480830a590d8f1bb5437bb8cf0cb2f66ef247faef8ec392cdeac1d59`.
- Selection hash:
  `ee27058c0cce00f21d28c49d915b68a312c322c9d09b9f98a73ae280a096c0d1`.

## Inner Validation

- Variant: `v26_empirical_task_vector_editor`.
- Config count: `8`.
- Config grid hash:
  `a2003d81993e757ce8bbf6409851a5f2ceece70e7a3d403cb969a8331deb36db`.
- Plan hash:
  `a298a9dc52370fcf4dec7f9fd2eaeba1fc160593b7a8d8c82de49e6ae6947a87`.
- Rung 0: `8` candidates, `12` jobs each, keep `4`.
- Rung 1: `4` candidates, `24` jobs each, keep `2`.
- Candidate completions: `12`.
- Invalid candidates: `0`.
- Contract failures: `0`.

## Final-Rung Results

| Candidate | Source | Projection | Target prediction rate | Mean target margin | Matched minus best control | Shuffled-signature lift | Pareto rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6 | jacobian | rank1_spectral_rank4 | 0.0 | -0.11784304970016517 | -0.0490827791243286 | -0.001303030740378593 | 0.5833333333333334 |
| 4 | jacobian | spectral_rank4 | 0.0 | -0.14404215443846624 | -0.03863552696975603 | 0.0027896384241709407 | 0.375 |
| 1 | empirical_centroid_task_vector | none | 0.0 | -0.14848260811919545 | 0.005627399133572908 | 0.01909245927805614 | 0.5 |
| 5 | empirical_centroid_task_vector | spectral_rank4 | 0.0 | -0.11411868715731543 | -0.005776969579083395 | 0.032713105705321745 | 0.5416666666666666 |

The registered ranking selected candidate `6` as best:

- Config hash:
  `90c052d271876babc1d0763e0681017c8174baa037138b346fa495b56650188b`.
- `target_prediction_rate=0.0`.
- `mean_target_margin=-0.11784304970016517`.
- `mean_matched_minus_best_control_target_margin=-0.0490827791243286`.
- `mean_matched_minus_shuffled_signature_target_margin=-0.001303030740378593`.
- `proof_gate_failure_count=37`.
- `contract_failure_count=0`.

Candidate `5` was the stronger empirical final-rung candidate on
shuffled-signature lift, but still had no target prediction flips and a negative
matched-minus-best-control margin.

## Interpretation

The empirical train-only centroid task-vector source produced weak relative
margin signals in some settings, especially candidate `1` and candidate `5`.
Those signals did not translate into functional target behavior. This rules out
the first eight low-ridge, zero-compat configurations under the bounded
balanced-development selection. It does not justify claims about decoding into
functional models with altered behavior.

Useful next hypotheses should focus on why target margins remain mostly
negative despite weak signature-relative lift. Plausible directions include
stronger source-preserving objectives, learned target-specific edit magnitudes,
or moving from centroid task vectors to localized low-rank adapters trained
against behavioral loss with sealed split controls.

## Reviewer Outcome

Reviewer: Kepler, agent `019eaf1b-397e-71b0-977f-15b0ad738095`.

Confidence: `5/5`.

Reviewer accepted this as a clean negative diagnostic result and confirmed:

- reproduced both log hashes;
- confirmed process exit and no matching compute processes;
- confirmed monitor stop, progress line counts, and balanced job selection;
- confirmed `12` candidate completions, `invalid_count=0`, and
  `contract_failure_count_sum=0`;
- confirmed all candidate target rates were exactly `0.0`;
- confirmed the final-rung best config follows the registered ranking;
- confirmed empirical bank provenance used train hashes/counts and variant
  labeling;
- found no raw weights, subject IDs, final raw references, tracebacks,
  exceptions, or nonfinite markers in the logs;
- ran the V25 helper suite: `88 passed`;
- ran `py_compile`: passed;
- did not run lint;
- did not open or read final raw.

