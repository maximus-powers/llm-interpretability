# Four-Behavior Source Generation V3 Full-Pool Preregistration

Date: 2026-06-10

## Purpose

This pilot tests whether source subjects can clear the preregistered source gate
when trained from larger predicate-derived training pools that exclude all
heldout acceptance cases.

This is source-generation feasibility only. It does not train a decoder and does
not create stored-probe decoding evidence.

## Prior Results

V1 support-only source generation used `32` positive and `32` negative support
cases per behavior. It failed for `has_majority` at `0/8`.

V2 expanded-support source generation used `160` positive and `160` negative
support cases per behavior. It improved `has_majority` to `4/8`, but still
failed the preregistered `8/8` source-gate requirement.

## V3 Source-Generation Protocol

Use the same clean behavior predicates and `SubjectNetwork` architecture as the
four-behavior decoder preregistration.

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

Training pools:

- enumerate the full finite sequence universe of `10^5` length-`5` digit
  sequences;
- remove every sequence used in any heldout positive or heldout negative case
  across the four behaviors;
- for each target behavior, build:
  - positive candidates: sequences satisfying the target predicate;
  - hard negative candidates: sequences satisfying a different clean behavior
    but not the target predicate;
  - generic negative candidates: sequences satisfying none of the clean behavior
    predicates.

Sampling:

- positive cap: `2048`;
- hard negative cap: `1024`;
- generic negative cap: `1024`;
- if fewer candidates exist than a cap, use all available candidates;
- sample deterministically with seed:
  `20261210 + behavior_index * 10000 + subject_index`;
- combine selected positives, hard negatives, and generic negatives;
- shuffle the combined training set deterministically with the same seed.

Source training:

- train only on the heldout-excluded predicate-derived training pool for the
  target behavior;
- `350` epochs;
- learning rate `0.003`;
- deterministic seed schedule:
  `20261210 + behavior_index * 10000 + subject_index`;
- `n=8` pilot seeds per behavior.

Source acceptance:

- evaluate on heldout cases only;
- heldout target margin must be at least `0.40`;
- all four behaviors must pass `8/8` pilot subjects.

## Required Artifact Fields

The result artifact must include:

- behavior-suite metadata;
- support and heldout case counts;
- support/heldout overlap count;
- global heldout-exclusion count;
- available positive, hard-negative, and generic-negative candidate counts per
  behavior after heldout exclusion;
- positive, hard-negative, and generic-negative candidate-pool hashes per
  behavior after heldout exclusion;
- selected positive, hard-negative, and generic-negative counts per subject;
- selected positive, hard-negative, and generic-negative case hashes per subject;
- selected combined training-case hash per subject;
- selected-training-vs-heldout overlap count per subject;
- source-margin gate;
- train epochs and learning rate;
- seed schedule;
- per-subject support margin and heldout margin;
- per-subject weight hash and signature hash;
- per-behavior pass counts and margin summaries;
- aggregate pass count and margin summary;
- explicit `claim_scope` stating this is source-generation feasibility only;
- caveats stating this is not decoder evidence and not broad MUAT evidence;
- a payload hash embedded in the JSON result.

## Pass/Fail Rule

The V3 pilot passes only if every behavior has:

- `n = 8`;
- source-gate pass count `8/8`;
- heldout margin minimum at least `0.40`;
- every subject has zero overlap between selected training cases and heldout
  acceptance cases.

If any behavior fails, the result is logged as a negative or limited
source-generation feasibility result.

## Result Interpretation

Allowed positive claim if the V3 pilot passes:

> Under the heldout-excluded full-pool source-generation protocol, all four clean
> behaviors can produce pilot source subjects that clear the `0.40` heldout
> source-margin gate.

Required limitations even if the V3 pilot passes:

- no stored-probe decoder claim;
- no steering claim;
- no larger-model claim;
- no broad MUAT generality claim;
- `n=8` per behavior is only a pilot;
- a future decoder proof still requires disjoint train, development, and final
  source pools under the four-behavior decoder preregistration.

If the V3 pilot fails, the result must not be tuned around in-place. A method
change after failure requires a new preregistration.
