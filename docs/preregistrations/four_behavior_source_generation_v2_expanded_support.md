# Four-Behavior Source Generation V2 Expanded-Support Preregistration

Date: 2026-06-10

## Purpose

This pilot tests whether expanded support-case coverage fixes the source-subject
generation blocker found in
`runs/four_behavior_source_generation_feasibility_v1/results.json`.

This is source-generation feasibility only. It does not train a decoder and does
not create stored-probe decoding evidence.

## Prior Result

The V1 support-only source-generation pilot used `32` positive and `32` negative
support cases per behavior, then evaluated source acceptance on disjoint heldout
cases. Under the `0.40` heldout source-margin gate:

- `sorted_ascending` passed `8/8`;
- `sorted_descending` passed `8/8`;
- `mountain_pattern` passed `8/8`;
- `has_majority` failed `0/8`.

Reviewer-accepted interpretation: `has_majority` is a blocker for that
support-only source-generation protocol, not an impossibility result.

## V2 Source-Generation Protocol

Use the same clean behavior predicates and `SubjectNetwork` architecture as the
four-behavior decoder preregistration.

Behavior suite:

- patterns:
  - `sorted_ascending`
  - `sorted_descending`
  - `has_majority`
  - `mountain_pattern`
- support cases per class: `160`
- heldout cases per class: `64`
- suite seed: `20260609`
- sequence length: `5`
- digit base: `10`

The suite must report zero support/heldout sequence overlap. If the expanded
suite cannot be generated with zero support/heldout overlap, the pilot fails
before subject training.

Source training:

- train only on the expanded support cases for the target behavior;
- `160` positive and `160` negative support cases per behavior;
- `350` epochs;
- learning rate `0.003`;
- deterministic seed schedule:
  `20261110 + behavior_index * 10000 + subject_index`;
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

The V2 pilot passes only if every behavior has:

- `n = 8`;
- source-gate pass count `8/8`;
- heldout margin minimum at least `0.40`.

If any behavior fails, the result is logged as a negative or limited
source-generation feasibility result.

## Result Interpretation

Allowed positive claim if the V2 pilot passes:

> Under the expanded-support source-generation protocol, all four clean
> behaviors can produce pilot source subjects that clear the `0.40` heldout
> source-margin gate.

Required limitations even if the V2 pilot passes:

- no stored-probe decoder claim;
- no steering claim;
- no larger-model claim;
- no broad MUAT generality claim;
- `n=8` per behavior is only a pilot;
- a future decoder proof still requires disjoint train, development, and final
  source pools under the four-behavior decoder preregistration.

If the V2 pilot fails, the result must not be tuned around in-place. A method
change after failure requires a new preregistration.
