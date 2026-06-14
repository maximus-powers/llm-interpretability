# Four-Behavior Stored-Probe Decoder Development V3 Signature Inversion Preregistration

Date: 2026-06-11

## Purpose

This preregistration freezes a third train/development attempt for a
four-behavior stored-probe decoder.

V3 tests a per-subject signature-inversion decoder: given only a stored-probe
signature, initialize from a train-only nearest neighbor in signature space, then
optimize weights to match the target stored-probe signature with a differentiable
version of the registered extractor.

This is not a final proof. It may use train and development pools only. It must
not read, parse, summarize, or evaluate the sealed final source pool.

## Prior Development Results

V1 direct MLP development failed with near-zero matched target behavior:

- individual all-gate pass count: `0/96`;
- mean matched target margin: `0.0045465377`;
- mean matched-minus-best-control target margin: `-0.4180624041`.

V2 train-only functional distillation improved matched target behavior but still
failed all specificity/control gates:

- individual all-gate pass count: `0/96`;
- mean matched target margin: `0.4117191500`;
- mean matched-minus-best-control target margin: `-0.3154511039`;
- mean best-control-minus-matched subject-output MSE: `-27.4087092181`.

V3 is an adaptive development revision after observing V1 and V2. It starts a
new development cycle and does not preserve eligibility for either earlier
method.

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
- registered extractor:
  `paired_contrast.extract_signature_with_stored_probes.v1`.

Behavior suite:

- use `build_clean_behavior_suite` with all four clean behaviors;
- support cases per class: `160`;
- heldout cases per class: `64`;
- seed: `20260609`.

## Differentiable Signature Extractor

Implement a differentiable extractor that matches the registered stored-probe
feature layout:

- per post-activation hidden neuron over the stored probe inputs:
  - mean;
  - standard deviation with `unbiased=False`;
  - first five FFT magnitude features;
  - safe correlation with each of the five input positions;
  - mean again as the pre-activation proxy;
  - standard deviation again as the pre-activation proxy.

Before any development run, a helper test must compare the differentiable
extractor against `extract_signature_with_stored_probes` on a fixed random flat
weight vector and require max absolute difference at most `1e-5`.

## Train-Only Statistics And Classifier

Compute train-only normalization statistics:

- signature mean and standard deviation;
- weight mean and standard deviation.

Clamp standard deviations below `1e-6` to `1.0`.

For behavior inference, use train-only nearest centroid in normalized signature
space:

- compute one normalized-signature centroid per behavior from accepted train
  subjects;
- assign a query to the behavior whose centroid has minimum squared Euclidean
  distance.

The true development behavior label must not be used by the decoder or optimizer.
It may be used only for evaluation and reporting.

## Decoder Method

For each accepted development subject:

1. Normalize its stored-probe signature using train-only signature statistics.
2. Infer behavior using train-only nearest centroid in normalized signature
   space.
3. Select the nearest accepted train subject by squared Euclidean distance in
   normalized signature space across all train subjects; labels are not used.
4. Initialize the decoded flat weights from that nearest train subject.
5. Optimize the flat weights for exactly `300` Adam steps.

Optimization config:

- optimizer: `Adam`;
- learning rate: `0.03`;
- gradient clipping norm: `10.0`;
- seed: `20260616`;
- no dropout or stochastic loss sampling.

Loss terms:

- normalized differentiable signature MSE to the query signature;
- inferred-behavior support BCE;
- inferred-behavior support margin hinge with target `0.40`;
- L2 weight distance to the nearest-train initialization.

Loss weights:

- signature MSE: `5.0`;
- support BCE: `0.5`;
- support margin hinge: `1.0`;
- initialization L2: `0.01`.

The optimizer must not receive the true development behavior label, final data,
subject id as a feature, source margin, or development heldout metrics.

## Development Controls

For each accepted development subject, compare the matched inversion decode
against:

- nearest accepted train subject by normalized signature distance;
- worst-of-32 normalized Gaussian noise signatures decoded by the same V3
  inversion pipeline;
- null signature decoded by the same V3 inversion pipeline;
- train global centroid signature decoded by the same V3 inversion pipeline;
- same-label train centroid signature decoded by the same V3 inversion pipeline;
- every other-label train centroid signature decoded by the same V3 inversion
  pipeline;
- same-label other-subject train weights, deterministically chosen from train;
- one different-label other-subject train weight per non-target behavior,
  deterministically chosen from train.

The nearest accepted train subject control is mandatory. It tests whether
signature inversion improves beyond train-subject retrieval.

Every signature-valued control listed above must use the same V3 inversion
algorithm as the matched query:

- infer behavior from train-only centroids for the control signature;
- select nearest train initialization by normalized signature distance;
- optimize exactly `300` steps with the same objective and hyperparameters.

The true development behavior is used only to evaluate the resulting control
weights, never to decode or optimize them.

V2 MLP-based controls may be reported only as auxiliary diagnostics. They are not
part of the pass/fail gates and must not substitute for same-decoder V3 controls.

### Gaussian Noise Controls

Noise controls are sampled in normalized signature space.

For each development subject, draw `32` standard Gaussian normalized signatures
using `torch.Generator().manual_seed(noise_seed)`, where:

`noise_seed = int(first_16_hex(stable_hash_json([development_subject_id, "v3_noise", 20260617])), 16) % 2**31`

Each sampled noise signature is decoded through the full V3 inversion pipeline.
For reporting:

- target-margin adversarial noise control is the sampled noise decode with the
  maximum target behavior margin;
- subject-output adversarial noise control is the sampled noise decode with the
  minimum subject-output MSE;
- all 32 sampled control summaries must be retained in the development artifact.

The best-control gates use the adversarial values, not the average noise value.

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

This rule must not depend on decoder outputs, target margins, source margins,
subject-output MSE, or post-hoc inspection.

### Best-Control Definitions

For target-margin separation, `best_control` means the applicable control with
the maximum target behavior margin for that subject.

For subject-output specificity, `best_control` means the applicable control with
the minimum subject-output MSE for that subject.

Therefore per-subject gates use:

- `matched_minus_best_control_target_margin =
  matched_target_margin - max(control_target_margin)`;
- `best_control_minus_matched_subject_output_mse =
  min(control_subject_output_mse) - matched_subject_output_mse`.

The control selected for the target-margin gate may differ from the control
selected for the subject-output MSE gate. Both identities must be reported.

## Development Metrics

For every matched/control decode, report:

- target heldout behavior margin using the true development behavior label;
- source subject output MSE on the target behavior heldout positive and negative
  cases;
- matched-minus-control target margin;
- control-minus-matched subject-output MSE.
- inferred behavior used by the decoder for every V3-inverted matched or control
  signature.

For the matched decode, also report:

- normalized signature MSE to target signature after optimization;
- decoded behavior inferred from train-only centroids;
- whether inferred behavior equals true development behavior;
- nearest-train subject id hash, behavior, and distance.

The artifact must include an inferred-behavior confusion matrix by true behavior
for matched development subjects, plus inferred-behavior counts for V3-inverted
control signatures by control type.

Summaries must include aggregate, per-behavior, per-control-type, and individual
subject metrics.

## Development Pass Gates

The development result passes only if every gate below passes.

Aggregate gates:

- inferred behavior accuracy is at least `0.90`;
- mean matched target margin is at least `0.20`;
- mean matched-minus-best-control target margin is at least `0.20`;
- mean best-control-minus-matched subject-output MSE is at least `0.05`;
- individual all-gate pass rate is at least `0.90`.

Per-behavior gates:

- development accepted subject count is at least `24`;
- inferred behavior accuracy is at least `0.80`;
- mean matched target margin is at least `0.20`;
- mean matched-minus-best-control target margin is at least `0.15`;
- mean best-control-minus-matched subject-output MSE is at least `0.02`;
- individual all-gate pass rate is at least `0.80`.

Per-subject gates:

- inferred behavior equals true development behavior;
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
- train-only behavior centroid hash;
- probe examples hash;
- behavior-suite support and heldout hashes;
- explicit statement that final raw pool was not an input.

The development artifact must not contain final raw subject IDs, labels, weights,
signatures, source margins, or per-subject records.

## Result Interpretation

Allowed positive claim if development passes:

> A per-subject stored-probe signature-inversion decoder with train-only centroid
> behavior inference clears train/development gates and is eligible for a
> separately locked one-shot final evaluation.

Required limitations even if development passes:

- no final decoder proof claim;
- no steering claim;
- no larger-model claim;
- no broad MUAT generality claim;
- final source pool remains sealed.

If development fails, the result must be logged as a negative or limited
development result. Any method change after observing development metrics
requires a new preregistration before another development run.
