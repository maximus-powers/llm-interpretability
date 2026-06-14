# Four-Behavior Source Generation V4 Accept-Reject Preregistration

Date: 2026-06-10

## Purpose

This pilot tests whether deterministic accept-reject sampling with the V3
heldout-excluded full-pool source-training method can collect accepted source
subjects for all four clean behaviors.

This is source-generation feasibility only. It does not train a decoder and does
not create stored-probe decoding evidence.

## Prior Results

V1 support-only source generation failed for `has_majority` at `0/8`.

V2 expanded-support source generation improved `has_majority` to `4/8`, but
failed the `8/8` source-gate requirement.

V3 heldout-excluded full-pool source generation improved `has_majority` to `7/8`,
with one source at heldout margin `0.3839 < 0.40`. The other three behaviors
passed `8/8`.

## V4 Source-Generation Protocol

Use the same source-training method as V3:

- enumerate the full finite `10^5` length-`5` digit sequence universe;
- remove every sequence used in any heldout positive or heldout negative case
  across the four behaviors;
- sample up to `2048` positives, `1024` hard negatives, and `1024` generic
  negatives per subject;
- record candidate-pool hashes, selected-case hashes, and per-subject
  selected-training-vs-heldout overlap counts.

Heldout suite:

- patterns:
  - `sorted_ascending`
  - `sorted_descending`
  - `has_majority`
  - `mountain_pattern`
- suite seed: `20260609`
- support cases per class used only for metadata compatibility: `160`
- heldout cases per class: `64`
- sequence length: `5`
- digit base: `10`

Source training:

- `350` epochs;
- learning rate `0.003`;
- training mode: `heldout_excluded_full_pool`;
- source heldout margin gate: `>= 0.40`.

Accept-reject schedule:

- target accepted subjects per behavior: `8`;
- max attempts per behavior: `32`;
- deterministic seed schedule:
  `20261310 + behavior_index * 10000 + attempt_index`;
- evaluate attempts sequentially;
- accept an attempt if heldout target margin is at least `0.40` and selected
  training cases have zero overlap with heldout acceptance cases;
- stop a behavior when either `8` subjects are accepted or `32` attempts are
  exhausted.

## Required Artifact Fields

The result artifact must include:

- behavior-suite metadata;
- candidate-pool counts and hashes after heldout exclusion;
- every attempted subject, not only accepted subjects;
- per-attempt seed, attempt index, accepted/rejected status, support margin,
  heldout margin, weight hash, signature hash, selected-case hashes, and
  selected-training-vs-heldout overlap count;
- accepted subject ids per behavior;
- rejected subject ids per behavior;
- attempts used per behavior;
- acceptance rate per behavior;
- source-margin gate;
- max attempts per behavior;
- target accepted subjects per behavior;
- top-level max selected-training-vs-heldout overlap count;
- explicit `claim_scope` stating this is source-generation feasibility only;
- caveats stating this is not decoder evidence and not broad MUAT evidence;
- a payload hash embedded in the JSON result.

## Pass/Fail Rule

The V4 pilot passes only if every behavior has:

- accepted subject count `8/8`;
- attempts used no more than `32`;
- every accepted subject heldout margin at least `0.40`;
- every attempted subject has zero selected-training-vs-heldout overlap.

If any behavior fails, the result is logged as a negative or limited
source-generation feasibility result.

## Result Interpretation

Allowed positive claim if the V4 pilot passes:

> Under the heldout-excluded full-pool source-training method with deterministic
> accept-reject sampling, all four clean behaviors can produce accepted pilot
> source subjects that clear the `0.40` heldout source-margin gate.

Required limitations even if the V4 pilot passes:

- no stored-probe decoder claim;
- no steering claim;
- no larger-model claim;
- no broad MUAT generality claim;
- accept-reject sampling shows source-subject availability, not decoder
  performance;
- a future decoder proof still requires disjoint train, development, and final
  source pools under the four-behavior decoder preregistration.

If the V4 pilot fails, the result must not be tuned around in-place. A method
change after failure requires a new preregistration.
