# Four-Behavior Stored-Probe Decoder V1 Preregistration

Date: 2026-06-10

## Purpose

This experiment tests whether a stored-probe signature-to-weight decoder can
decode fresh functional subject models for all four clean behaviors:

- `sorted_ascending`
- `sorted_descending`
- `has_majority`
- `mountain_pattern`

This is a new proof attempt. It must not reuse final subjects for method
selection. If the final gates fail, the result is logged as a negative or
limited result and the next method revision must use a new preregistration.

## Current Baseline

The accepted evidence package currently supports only:

- stored-probe interpretability over four behaviors;
- restricted functional decoding for `sorted_ascending <-> sorted_descending`;
- restricted robust steering for `sorted_ascending <-> sorted_descending`.

The locked two-behavior decoder failed on fresh `has_majority` and
`mountain_pattern` subjects in
`runs/stored_probe_additional_behavior_decode_feasibility_v1/results.json`.

## Fixed Architecture Scope

All subject models use the existing `SubjectNetwork` architecture:

- sequence length: `5`
- input base: `10`
- hidden layers: `5`
- neurons per hidden layer: `8`
- flat weight dimension: `345`

This experiment does not claim evidence for larger models.

## Probe Set

Use the existing deterministic stored digit probe set:

- probe count: `256`
- probe seed: `20260610`
- probe id: `stored_digit_probe_v1_seed_20260610_n256`
- signature extractor:
  `paired_contrast.extract_signature_with_stored_probes.v1`

The probe examples and extraction code must be hashed in the resulting
artifacts. Final-subject signatures must be regenerated from stored weights and
stored probes, not loaded from a mutable feature column.

## Subject Pools

Generate three disjoint subject pools from non-overlapping seed ranges:

- train pool: decoder fitting only;
- development pool: method selection and early stopping only;
- final pool: one-shot proof evaluation only.

No subject weight hash or signature hash may overlap across pools. The final
pool must not be read by training or method-selection code before the final
evaluation command.

Recommended minimum accepted subjects:

- train: at least `64` accepted subjects per behavior;
- development: at least `24` accepted subjects per behavior;
- final: at least `24` accepted subjects per behavior.

Source subject acceptance gates:

- heldout target margin must be at least `0.40`;
- support and heldout behavior cases must have zero sequence overlap;
- accepted subject metadata must include seed, behavior, weight hash, signature
  hash, support margin, heldout margin, and train loss.

If any behavior cannot produce enough accepted subjects under these gates, the
experiment fails before decoder training.

## Decoder Training

Train a decoder that maps normalized stored-probe signatures to flat subject
weights.

Allowed inputs:

- stored-probe signature for the source subject;
- train-only normalization statistics.

Disallowed inputs:

- behavior label at decode time;
- final-pool weights, signatures, labels, or metrics;
- final-pool normalization statistics;
- hand-authored final examples for training.

Training objective must include:

- weight reconstruction loss against train subject weights;
- generated heldout behavior margin loss for the source behavior;
- control penalty against behavior priors where applicable.

Method selection may use train and development pools only. Hyperparameters,
checkpoint choice, thresholds, and controls must be frozen before final
evaluation.

## Final Evaluation Controls

For each final subject, compare the matched decode against all of these controls:

- worst-of-32 normalized Gaussian noise signatures;
- null signature;
- train global centroid signature;
- same-label train centroid signature;
- every other-label train centroid signature;
- same-label other-subject signature;
- every different-label other-subject signature, with one control for each
  non-target behavior;
- different-label same-direction signature where direction is defined;
- opposite-direction signature for the sorted behaviors;
- condition-ablation decode if the architecture has any non-signature path.

Controls that are not applicable to an architecture must be reported as
`not_applicable` with a reason. They must not be silently omitted.

The final artifact must report metrics and pass/fail status for every
control type. Best-control summaries are required, but they are not sufficient
on their own.

## Final Proof Gates

The final result passes only if every gate below passes.

Aggregate gates:

- mean matched target margin is at least `0.20`;
- mean matched-minus-best-control target margin is at least `0.20`;
- mean best-control-minus-matched subject-output MSE is at least `0.05`;
- individual pass rate is at least `0.95`.

Per-behavior gates:

- realized final subject count is at least the preregistered final accepted
  count;
- mean matched target margin is at least `0.20`;
- mean matched-minus-best-control target margin is at least `0.15`;
- mean best-control-minus-matched subject-output MSE is at least `0.02`;
- individual pass rate is at least `0.90`.

Per-subject individual gates:

- matched target margin is at least `0.20`;
- matched-minus-best-control target margin is at least `0.10`;
- best-control-minus-matched subject-output MSE is greater than `0.00`.

If any aggregate, per-behavior, or individual-rate gate fails, the final result
is not proof. Passing aggregate metrics cannot override a failed per-behavior
gate.

## Leakage Audit

The final artifact must report:

- train/development/final subject counts by behavior;
- final realized accepted-subject count by behavior;
- overlap counts for subject ids, seeds, weight hashes, and signature hashes;
- train-only normalization hashes;
- probe examples hash;
- behavior-suite hash;
- support/heldout behavior-case overlap count;
- decoder checkpoint hash;
- per-control-type target-margin metrics, subject-output MSE metrics, and
  pass/fail status;
- code version or explicit dirty-worktree caveat.

Any final overlap with train or development subjects makes the final proof fail.

## Result Interpretation

Allowed positive claim if all gates pass:

> Under this fixed small-network setup, stored-probe signatures contain enough
> information for a trained decoder to generate functional subject-like weights
> for four clean behaviors on a fresh final holdout, beating strong signature
> controls and subject-output controls.

Required limitation even if all gates pass:

- no larger-model claim;
- no broad MUAT generality claim;
- no claim beyond these four clean behaviors;
- no claim that the decoder recovers mechanistic equivalence beyond the measured
  subject-output controls.

If the final gates fail, the result must be logged as a negative or limited
result. Any follow-up method change requires a new development cycle and a new
fresh final holdout.
