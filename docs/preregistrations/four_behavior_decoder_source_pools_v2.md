# Four-Behavior Decoder Source Pools V2 Preregistration

Date: 2026-06-10

## Purpose

This preregistration freezes a repaired source-pool construction attempt for the
four-behavior stored-probe decoder proof.

This is source-pool construction only. It does not train a decoder and does not
create stored-probe decoding evidence.

## Prior Failed Attempt

V1 source-pool construction produced enough accepted source subjects by behavior,
but failed the preregistered cross-pool seed-disjointness gate:

- train/development accepted seed overlap: `71`;
- train/final accepted seed overlap: `47`;
- development/final accepted seed overlap: `67`.

The cause was a defective seed schedule: pool base seeds differed by `10000`,
which was also the behavior stride. Those V1 pools are invalid for decoder-proof
use.

V2 changes only the source-pool seed schedule and the public final-pool audit
surface. The source-training method, source-margin gate, behavior suite, and
stored probe set remain unchanged from the accepted V4 source-generation pilot.

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

Every generated source subject must include in its raw pool artifact:

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

Every attempted subject must be retained in the raw pool artifact, not only
accepted subjects.

## Pool Definitions

Generate three pools with non-overlapping seed ranges:

- train:
  - base seed: `20300000`;
  - target accepted subjects per behavior: `64`;
  - max attempts per behavior: `128`;
- development:
  - base seed: `21300000`;
  - target accepted subjects per behavior: `24`;
  - max attempts per behavior: `64`;
- final:
  - base seed: `22300000`;
  - target accepted subjects per behavior: `24`;
  - max attempts per behavior: `64`.

Seed schedule for each pool:

`base_seed + behavior_index * 100000 + attempt_index`

Before training any source subjects, the generation script must run a seed-range
preflight check over every configured pool/behavior attempt range. The preflight
must pass only if no configured seed ranges overlap, regardless of which attempts
eventually produce accepted subjects.

The final pool may be generated and sealed as a source artifact, but decoder
training and method selection must not read final-pool weights, signatures,
labels, source margins, per-subject records, attempt counts, rejected counts,
accepted subject IDs, rejected subject IDs, acceptance rates, or per-subject
metrics before the final decoder evaluation command.

## Final-Pool Access Policy

After generation, the final pool raw subject artifact is sealed for proof use.

Before final decoder evaluation, humans and scripts may inspect only this limited
audit surface for the final pool:

- pass/fail status;
- accepted counts by behavior;
- selected-training-vs-heldout overlap pass/fail or max overlap count;
- cross-pool overlap counts;
- file and redacted-payload hashes;
- stored-probe hash;
- behavior-suite hashes;
- config hash.

Before final decoder evaluation, public audit artifacts must not expose final
per-subject records, final subject IDs, final attempt counts, final rejection
counts, final acceptance rates, final source margins, final weights, or final
signatures.

The final pool path must not be an input to any train or development script.
Only the final decoder evaluation command may load final accepted weights,
signatures, and labels.

Any accidental human inspection or script access to final raw contents before the
final decoder evaluation invalidates the final pool for proof use.

## Required Raw Pool Artifact Fields

Each raw pool artifact must include:

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
- raw file hash recorded in a separate audit artifact;
- redacted-payload hash recorded in the raw artifact and audit artifact.

## Required Combined Audit

After generating all three pools, create a combined audit artifact reporting:

- accepted subject counts by pool and behavior;
- attempted subject counts by behavior for train and development only;
- accepted seed overlap counts between every pair of pools;
- accepted subject id overlap counts between every pair of pools;
- accepted weight hash overlap counts between every pair of pools;
- accepted signature hash overlap counts between every pair of pools;
- pool file hashes;
- redacted payload hashes;
- behavior-suite hashes;
- stored-probe hash;
- seed-range preflight pass/fail status and configured seed ranges;
- pass/fail status.

The combined audit passes only if:

- seed-range preflight passes before generation;
- train has at least `64` accepted subjects per behavior;
- development has at least `24` accepted subjects per behavior;
- final has at least `24` accepted subjects per behavior;
- every accepted subject has heldout margin at least `0.40`;
- every attempted subject has selected-training-vs-heldout overlap count `0`;
- accepted seeds, subject ids, weight hashes, and signature hashes have zero
  overlap across train/development/final pools.

The combined audit must use a redacted final summary. It must not expose final
attempt counts, rejected counts, acceptance rates, subject IDs, margins, records,
weights, or signatures.

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

If any gate fails, the result must be logged as a negative or limited source-pool
construction result. A method change after failure requires a new
preregistration.
