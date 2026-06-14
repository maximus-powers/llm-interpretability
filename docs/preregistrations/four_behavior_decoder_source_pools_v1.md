# Four-Behavior Decoder Source Pools V1 Preregistration

Date: 2026-06-10

## Purpose

This preregistration freezes how to generate source-subject pools for the
four-behavior stored-probe decoder proof attempt.

This is source-pool construction only. It does not train a decoder and does not
create stored-probe decoding evidence.

## Background

The four-behavior stored-probe decoder preregistration requires disjoint source
pools:

- train pool: at least `64` accepted subjects per behavior;
- development pool: at least `24` accepted subjects per behavior;
- final pool: at least `24` accepted subjects per behavior.

V4 source-generation feasibility showed that the heldout-excluded full-pool
source-training method can produce accepted pilot source subjects for all four
behaviors under the `0.40` heldout source-margin gate.

## Source-Generation Method

Use the V4 source-generation method:

- training mode: `heldout_excluded_full_pool`;
- collection mode: `accept_reject`;
- source heldout margin gate: `>= 0.40`;
- support cases per class used for metadata compatibility: `160`;
- heldout cases per class: `64`;
- positive cap: `2048`;
- hard-negative cap: `1024`;
- generic-negative cap: `1024`;
- train epochs: `350`;
- learning rate: `0.003`;
- stored probe count: `256`;
- stored probe seed: `20260610`.

Every generated source subject must include:

- behavior label;
- pool name;
- seed;
- attempt index;
- accepted/rejected status;
- support margin;
- heldout margin;
- flat weights;
- stored-probe signature;
- weight hash;
- signature hash;
- selected-case hashes;
- selected-training-vs-heldout overlap count.

Every attempted subject must be retained in the pool artifact, not only accepted
subjects.

## Pool Definitions

Generate three pools with non-overlapping seed ranges:

- train:
  - base seed: `20261410`;
  - target accepted subjects per behavior: `64`;
  - max attempts per behavior: `128`;
- development:
  - base seed: `20271410`;
  - target accepted subjects per behavior: `24`;
  - max attempts per behavior: `64`;
- final:
  - base seed: `20281410`;
  - target accepted subjects per behavior: `24`;
  - max attempts per behavior: `64`.

Seed schedule for each pool:

`base_seed + behavior_index * 10000 + attempt_index`

The final pool may be generated and sealed as a source artifact, but decoder
training and method selection must not read final-pool weights, signatures,
labels, source margins, or metrics before the final decoder evaluation command.

## Final-Pool Access Policy

After generation, the final pool raw subject artifact is sealed for proof use.

Before final decoder evaluation, humans and scripts may inspect only this limited
audit surface for the final pool:

- pass/fail status;
- accepted counts by behavior;
- overlap counts;
- file and payload hashes;
- stored-probe hash;
- behavior-suite hashes.

Before final decoder evaluation, decoder training and method-selection scripts
must not load or parse final-pool weights, signatures, labels, per-subject
margins, per-subject records, or acceptance-rate details beyond the minimal
pass/fail/count audit.

The final pool path must not be an input to any train or development script.
Only the final decoder evaluation command may load final accepted weights,
signatures, and labels.

Any accidental human inspection or script access to final raw contents before the
final decoder evaluation invalidates the final pool for proof use.

## Required Pool Artifact Fields

Each pool artifact must include:

- pool name;
- source-generation config;
- behavior-suite metadata;
- candidate-pool counts and hashes after heldout exclusion;
- all attempted subjects;
- accepted subjects;
- rejected subjects;
- accepted counts by behavior;
- attempt counts by behavior;
- rejection counts by behavior;
- acceptance rates by behavior;
- max selected-training-vs-heldout overlap count;
- pool-level payload hash.

## Required Combined Audit

After generating all three pools, create a combined audit artifact reporting:

- accepted subject counts by pool and behavior;
- attempted subject counts by pool and behavior;
- accepted seed overlap counts between every pair of pools;
- accepted subject id overlap counts between every pair of pools;
- accepted weight hash overlap counts between every pair of pools;
- accepted signature hash overlap counts between every pair of pools;
- pool payload hashes;
- behavior-suite hashes;
- stored-probe hash;
- pass/fail status.

The combined audit passes only if:

- train has at least `64` accepted subjects per behavior;
- development has at least `24` accepted subjects per behavior;
- final has at least `24` accepted subjects per behavior;
- every accepted subject has heldout margin at least `0.40`;
- every attempted subject has selected-training-vs-heldout overlap count `0`;
- accepted seeds, subject ids, weight hashes, and signature hashes have zero
  overlap across train/development/final pools.

## Result Interpretation

Allowed positive claim if the combined audit passes:

> The four-behavior decoder proof now has disjoint accepted source-subject
> train, development, and final pools satisfying the preregistered source gates.

Required limitations even if the combined audit passes:

- no stored-probe decoder claim;
- no steering claim;
- no larger-model claim;
- no broad MUAT generality claim;
- the final source pool is sealed for future one-shot decoder evaluation and
  must not be used for method selection.

If any gate fails, the result must be logged as a negative or limited
source-pool construction result. A method change after failure requires a new
preregistration.
