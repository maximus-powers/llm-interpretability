# V27 Localized Behavior-Loss Subspace Editor Results

Date: 2026-06-14

## Verdict

Accepted as a bounded negative diagnostic result only.

This run does not support successful functional weight editing or behavioral
steering. Every evaluated candidate had `target_prediction_rate=0.0`, the final
summary had `passed=false`, and the reviewer approved only the negative
diagnostic framing.

The useful signal is narrower: the best V27 localized editor moved target
margins in the intended direction relative to proof controls and
shuffled-signature controls, but the edits did not cross the target prediction
gate.

## Command

```bash
python model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v27-localized \
  --inner-validation-max-configs 8 \
  --inner-validation-rung-jobs 12,24 \
  --inner-validation-keep-fractions 0.5,0.5 \
  --development-job-selection balanced-directions
```

## Long-Run Observability

- Process PID `65919` exited cleanly.
- The run emitted monitor heartbeats about every `5` seconds with CPU time,
  latest progress event, latest progress elapsed time, and progress line count.
- Post-run log hashes matched the final stdout summary.
- `development_progress.jsonl` SHA-256:
  `837acc1f8e060f34b301908ad515ee62f431c7297a32f3ac0704c55570305cb0`.
- `long_run_monitor.jsonl` SHA-256:
  `545d13814e708f228b2280216a50fb64ac94e0a408293c80f55c52bb5d409f90`.
- Reviewer independently found `1,732` progress lines, `231` monitor lines,
  and `960` `v27_localized_optimizer_progress` events.

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

- Variant: `v27_localized_behavior_loss_subspace_editor`.
- Config count: `8`.
- Config grid hash:
  `c6fac5eca342cbb001dc03a3b6b38fdb07466c13397d376e06b1ab14986bca19`.
- Plan hash:
  `5ab8df9fe68255c74715de59240f5176cfd3e06b97aa1a5ed6bb447dfa522c2e`.
- Rung 0: `8` candidates, `12` jobs each, keep `4`.
- Rung 1: `4` candidates, `24` jobs each, keep `2`.
- Candidate completions: `12`.
- Invalid candidates: `0`.
- Contract failures: `0`.

Proof controls used the fixed V25-native baseline control config. The matched
edit used the V27 localized behavior-loss editor.

## Final-Rung Results

| Candidate | Basis | Delta L2 | Target prediction rate | Mean target margin | Matched minus best control | Shuffled-signature lift | Pareto rate | Proof gate failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | target_source_logit_gradient_rank4 | 0.0 | 0.0 | -0.06566180438839335 | 0.08483018685421939 | 0.0975856995982743 | 0.7083333333333334 | 33 |
| 5 | target_source_logit_gradient_rank4 | 0.01 | 0.0 | -0.06567064146652986 | 0.08943266368665566 | 0.0975768625201378 | 0.7083333333333334 | 33 |
| 3 | output_layer_topk | 0.0 | 0.0 | -0.15140267723241627 | 0.0016449689049219767 | 0.011844826754251395 | 0.5833333333333334 | 42 |
| 7 | output_layer_topk | 0.01 | 0.0 | -0.15140267874774813 | 0.004676455146939891 | 0.011844825238919535 | 0.5833333333333334 | 41 |

The registered ranking selected candidate `1` as best:

- Config hash:
  `b2d7a4be0f671628aac65fc80d5ec5630d6b4036326bc0e0c40bb61cf5078068`.
- `localized_basis=target_source_logit_gradient_rank4`.
- `localized_steps=25`.
- `localized_lr=0.05`.
- `localized_source_mse_weight=0.5`.
- `localized_delta_l2_weight=0.0`.
- `localized_norm_cap=0.25`.
- `target_prediction_rate=0.0`.
- `mean_target_margin=-0.06566180438839335`.
- `mean_matched_minus_best_control_target_margin=0.08483018685421939`.
- `mean_matched_minus_shuffled_signature_target_margin=0.0975856995982743`.
- `proof_gate_failure_count=33`.
- `contract_failure_count=0`.

Candidate `5` had a slightly higher matched-minus-best-control margin, but the
registered ranking still selected candidate `1` because its mean target margin
was slightly less negative. Both had zero target prediction flips.

## Interpretation

V27 should be treated as a bounded negative diagnostic. The localized
behavior-loss subspace objective produced directional margin movement for the
gradient-basis configs, but it did not produce functional target flips under the
registered `0.25` norm cap, `25` optimization steps, and balanced development
selection.

This result does not justify scaling V27 unchanged. A useful next attempt
should make a concrete methodological change, such as relaxing the norm
schedule with an explicit safety sweep, adding a stronger source-compatible
constraint that allows larger target motion, or learning edit magnitudes rather
than using a fixed cap. Any such follow-up should remain development-only until
it passes proof-control gates.

## Reviewer Outcome

Reviewer: Kepler, agent `019eaf1b-397e-71b0-977f-15b0ad738095`.

Confidence: `5/5`.

Reviewer accepted this as a valid bounded negative diagnostic and confirmed:

- reproduced both log hashes;
- counted `1,732` progress lines and `231` monitor lines;
- counted `960` V27 optimizer progress events;
- confirmed `12` candidate completions, with `8` in rung 0 and `4` in rung 1;
- confirmed `invalid_count=0` and total `contract_failure_count=0`;
- confirmed every candidate had `target_prediction_rate=0.0`;
- confirmed no exceptions, tracebacks, nonfinite numeric values, raw weights,
  subject IDs, final raw references, raw support examples, raw logits, or raw
  Jacobian markers in the logs;
- confirmed balanced selection was exactly `24` jobs with `2` per non-identity
  direction;
- required the wording "bounded negative diagnostic";
- required noting that proof controls used the fixed V25-native baseline config
  while the matched edit used the V27 localized editor.

