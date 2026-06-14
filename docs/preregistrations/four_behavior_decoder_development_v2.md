# Four-Behavior Stored-Probe Decoder Development V2 Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a second train/development attempt for a
four-behavior stored-probe signature-to-weight decoder.

This is not a final proof. It may train models and select a checkpoint using the
train and development source pools only. It must not read, parse, summarize, or
evaluate the sealed final source pool.

## Prior Development Result

V1 development failed:

- development aggregate `n`: `96`;
- individual all-gate pass count: `0/96`;
- mean matched target margin: `0.0045465377`;
- mean matched-minus-best-control target margin: `-0.4180624041`;
- mean best-control-minus-matched subject-output MSE: `-54.8715616030`.

The failure shape suggests the direct MLP trained mostly as normalized weight
regression did not learn functional subject behavior. V2 is an adaptive
development revision after observing V1. It does not preserve eligibility for
the V1 method; it starts a new development cycle.

## Inputs

Allowed raw source-pool inputs:

- `runs/four_behavior_decoder_source_pools_v2/train_subjects.json`;
- `runs/four_behavior_decoder_source_pools_v2/development_subjects.json`.

Allowed final-pool inputs before final evaluation:

- `runs/four_behavior_decoder_source_pools_v2/combined_audit.json`;
- `runs/four_behavior_decoder_source_pools_v2/final_redacted_audit.json`.

The train/development script must reject any `--final-pool` argument or final raw
path. The script must not open or name
`runs/four_behavior_decoder_source_pools_v2/final_subjects.json` in any result
artifact.

Use only accepted source subjects from the train and development raw pools.
Rejected subjects remain provenance but are not training/evaluation examples.

## Fixed Subject And Probe Setup

Subject architecture:

- `SubjectNetwork`;
- sequence length: `5`;
- input base: `10`;
- hidden layers: `5`;
- neurons per hidden layer: `8`;
- flat weight dimension: `345`.

Stored-probe setup:

- probe count: `256`;
- probe seed: `20260610`;
- probe id: `stored_digit_probe_v1_seed_20260610_n256`;
- signature dimension: `560`;
- signature extractor:
  `paired_contrast.extract_signature_with_stored_probes.v1`.

Behavior suite:

- use `build_clean_behavior_suite` with all four clean behaviors;
- support cases per class: `160`;
- heldout cases per class: `64`;
- seed: `20260609`.

## Decoder Architecture

Train a direct MLP decoder from normalized stored-probe signatures to normalized
flat weights.

Architecture:

- input dimension: `560`;
- hidden layers: `[1024, 1024, 1024, 512]`;
- activation: `GELU`;
- normalization: `LayerNorm` after each hidden linear;
- dropout: `0.0`;
- output dimension: `345`.

The decoder must not receive behavior labels, subject IDs, seeds, accepted
indices, source margins, train losses, pool names, or any final-pool field as
model inputs.

## Normalization

Compute normalization statistics on accepted train subjects only:

- signature mean and standard deviation;
- weight mean and standard deviation.

Clamp standard deviations below `1e-6` to `1.0`.

Development and future final inputs must use the frozen train-only statistics.
The artifact must report hashes of the normalization statistics.

## Train-Only Functional Distillation Cases

Build a deterministic train-only distillation input set before training:

- enumerate the full digit-sequence universe for length `5`, base `10`;
- exclude every sequence that appears in the behavior-suite heldout split for any
  behavior;
- shuffle remaining candidates with `random.Random(20260613)`;
- select the first `4096` cases.

For every accepted train subject, compute source logits on these `4096`
distillation cases. These logits are training targets only.

Development heldout behavior cases remain evaluation-only. Final raw data remains
sealed.

## Training Objective

Train for at most `1200` epochs with:

- optimizer: `AdamW`;
- learning rate: `0.0005`;
- weight decay: `0.0001`;
- batch size: `32`;
- distillation cases per batch: `256`;
- random seed: `20260614`.

Loss terms:

- train-subject logit distillation MSE on deterministic sampled distillation
  cases;
- support behavior BCE for the subject's target behavior;
- support margin hinge loss on support cases, margin target `0.40`;
- normalized weight reconstruction MSE.

Loss weights:

- distillation MSE: `1.0`;
- support BCE: `0.5`;
- support margin hinge: `1.0`;
- reconstruction MSE: `0.05`.

Distillation case subsets during training must be sampled by a deterministic
`torch.Generator().manual_seed(20260614)`. Development metrics must not affect
batch sampling.

No differentiable stored-probe signature reconstruction loss is allowed in this
V2 development run because the current registered extractor detaches weights and
runs under `no_grad`.

## Model Selection

Evaluate on the development pool every `50` epochs.

Select the checkpoint with the highest development score:

`mean_matched_minus_best_control_target_margin + 0.1 * mean_best_control_minus_matched_subject_output_mse`

Ties are broken by lower development reconstruction MSE, then earlier epoch.

All model selection uses development metrics only. No final-pool metrics may be
used.

## Development Controls

For each accepted development subject, compare the matched decode against:

- worst-of-32 normalized Gaussian noise signatures;
- null signature;
- train global centroid signature;
- same-label train centroid signature;
- every other-label train centroid signature;
- same-label other-subject signature, deterministically chosen from train;
- one different-label other-subject signature per non-target behavior,
  deterministically chosen from train.

Controls are evaluated using the same train-only normalization statistics and the
same decoder checkpoint as the matched decode.

### Deterministic Train-Subject Control Selection

For train-subject controls, build candidate sets from accepted train subjects
only.

Selection rule:

1. Exclude no candidates by output value, margin, MSE, loss, or decoder result.
2. Sort candidates by `(weight_hash, signature_hash, subject_id)`.
3. For a development subject and control family, compute:
   `stable_hash_json([development_subject_id, control_family, control_behavior])`.
4. Interpret the first 16 hex characters of that hash as an integer and take
   modulo the candidate count.
5. Select that candidate.

For same-label other-subject controls, `control_behavior` is the development
subject behavior. For different-label controls, one candidate is selected
separately for each non-target behavior.

This rule must not depend on decoder outputs, target margins, source margins,
subject-output MSE, or post-hoc inspection.

### Gaussian Noise Controls

Noise controls are sampled in normalized signature space.

For each development subject, draw `32` standard Gaussian normalized signatures
using `torch.Generator().manual_seed(noise_seed)`, where:

`noise_seed = int(first_16_hex(stable_hash_json([development_subject_id, "noise", 20260615])), 16) % 2**31`

Decode all 32 noise signatures. For reporting:

- target-margin adversarial noise control is the sampled noise decode with the
  maximum target behavior margin;
- subject-output adversarial noise control is the sampled noise decode with the
  minimum subject-output MSE;
- all 32 sampled control summaries must be retained in the development artifact.

The best-control gates use the adversarial values, not the average noise value.

### Best-Control Definitions

For target-margin separation, `best_control` means the control decode with the
maximum target behavior margin among all applicable controls for that subject.

For subject-output specificity, `best_control` means the control decode with the
minimum subject-output MSE among all applicable controls for that subject.

Therefore per-subject gates use:

- `matched_minus_best_control_target_margin =
  matched_target_margin - max(control_target_margin)`;
- `best_control_minus_matched_subject_output_mse =
  min(control_subject_output_mse) - matched_subject_output_mse`.

The control selected for the target-margin gate may differ from the control
selected for the subject-output MSE gate. Both identities must be reported.

## Development Metrics

For every matched/control decode, report:

- target heldout behavior margin;
- source subject output MSE on the target behavior heldout positive and negative
  cases;
- matched-minus-control target margin;
- control-minus-matched subject-output MSE.

For the matched decode, also report:

- normalized reconstruction MSE against the source weights;
- train-only distillation-case output MSE against the source subject.

Summaries must include aggregate, per-behavior, per-control-type, and individual
subject metrics.

## Development Pass Gates

The development result passes only if every gate below passes.

Aggregate gates:

- mean matched target margin is at least `0.20`;
- mean matched-minus-best-control target margin is at least `0.20`;
- mean best-control-minus-matched subject-output MSE is at least `0.05`;
- individual all-gate pass rate is at least `0.90`.

Per-behavior gates:

- development accepted subject count is at least `24`;
- mean matched target margin is at least `0.20`;
- mean matched-minus-best-control target margin is at least `0.15`;
- mean best-control-minus-matched subject-output MSE is at least `0.02`;
- individual all-gate pass rate is at least `0.80`.

Per-subject gates:

- matched target margin is at least `0.20`;
- matched-minus-best-control target margin is at least `0.10`;
- best-control-minus-matched subject-output MSE is greater than `0.00`.

If development gates fail, do not run final evaluation. Log the result as a
negative or limited development result.

## Leakage Audit

The development artifact must report:

- train/development accepted counts by behavior;
- train/development raw pool file hashes;
- final redacted audit hash only;
- exact command-line arguments;
- exact input paths opened by the script;
- assertion that no opened path ends with `final_subjects.json`;
- assertion that no opened path equals the sealed final raw pool path;
- assertion that the result artifact text does not contain `final_subjects.json`;
- overlap counts between train and development for subject id, seed, weight hash,
  and signature hash;
- train-only normalization hashes;
- train-only distillation-case hash;
- probe examples hash;
- behavior-suite support and heldout hashes;
- decoder checkpoint hash;
- explicit statement that final raw pool was not an input.

The development artifact must not contain final raw subject IDs, labels, weights,
signatures, source margins, or per-subject records.

## Result Interpretation

Allowed positive claim if development passes:

> A four-behavior stored-probe signature-to-weight decoder development run clears
> train/development gates and is eligible for a separately locked one-shot final
> evaluation.

Required limitations even if development passes:

- no final decoder proof claim;
- no steering claim;
- no larger-model claim;
- no broad MUAT generality claim;
- final source pool remains sealed.

If development fails, the result must be logged as a negative or limited
development result. Any method change after observing development metrics
requires a new preregistration before another development run.
