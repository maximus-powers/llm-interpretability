# V14 Literature-Supported Plan: Subspace Task-Vector Signature-Gated Editing

Date: 2026-06-12

Status: planning document, not an executable preregistration. This plan must be
reviewed before a V14 preregistration is written.

## Prior Result Context

V13 is negative development evidence for the claim that adding a fixed-probe
signature target as a small regularizer inside a source-initialized support-set
optimizer provides signature-specific functional editing value. V13 achieved
strong mean target/conflict metrics, but failed aggregate pass, Pareto,
target-prediction, and signature-control advantage gates. Final evaluation is
blocked and V13 final raw remains sealed.

The most important V13 failure for theory is that no-signature and
shuffled-signature optimizer controls were nearly as good as the matched
signature optimizer. Therefore V14 should not simply tune the same optimizer.

## Literature Basis

V14 is motivated by the following online literature review:

1. Han et al. (2026), *A Survey of Weight Space Learning*,
   https://arxiv.org/abs/2603.10090.
   - Weight-space learning should distinguish understanding, representation, and
     generation. V14 targets a narrow weight-to-weight editing operator, not a
     broad proof of MUAT.

2. Kaushik et al. (2025), *The Universal Weight Subspace Hypothesis*,
   https://arxiv.org/abs/2512.05117.
   - This is motivational, not direct evidence for the tiny four-behavior subject
     setting. If useful behavior edits live in low-dimensional shared subspaces,
     V14 should constrain edits to train-only aligned edit subspaces instead of
     optimizing all raw weights.

3. Ilharco et al. (2023), *Editing Models with Task Arithmetic*,
   https://arxiv.org/abs/2212.04089.
   - V14 should include task-vector baselines and ask whether signatures improve
     task-vector selection/scaling, not whether target behavior can be produced
     by any supervised support loss.

4. Ainsworth et al. (2022), *Git Re-Basin*,
   https://arxiv.org/abs/2209.04836.
   - Hidden-neuron permutation symmetry can make naive weight interpolation
     misleading. V14 must treat alignment as a primary variable with
     no-alignment and random-permutation controls.

5. Navon et al. (2023), *Equivariant Architectures for Learning in Deep Weight
   Spaces*, https://arxiv.org/abs/2301.12780, and Zhou et al. (2023),
   *Permutation Equivariant Neural Functionals*,
   https://arxiv.org/abs/2302.14040.
   - Future learned editors should be symmetry-aware. V14 remains deterministic
     rather than learned, but it should preserve this lesson by canonicalizing or
     aligning all train-derived directions before using them.

6. Wortsman et al. (2022), *Model soups*,
   https://arxiv.org/abs/2203.05482.
   - Averaging/merging can work inside compatible basins. V14 should measure
     interpolation barriers and include basin-compatibility diagnostics.

7. Yadav et al. (2023), *TIES-Merging: Resolving Interference When Merging
   Models*, https://arxiv.org/abs/2306.01708.
   - TIES shows that task-vector/model merging can fail from redundant small
     updates and sign disagreement. V14 should include sign-conflict and
     magnitude-trimming diagnostics so a signature gate is not credited for
     solving an omitted task-vector interference problem.

8. Jordan et al. (2022), *REPAIR: REnormalizing Permuted Activations for
   Interpolation Repair*, https://arxiv.org/abs/2211.08403.
   - REPAIR shows that alignment alone can still leave interpolation barriers
     through activation variance collapse. V14 should measure hidden activation
     variance/scale preservation and include a variance-collapse diagnostic.

9. Turner et al. (2024), *Steering Language Models With Activation
   Engineering*, https://arxiv.org/abs/2308.10248.
   - This is motivational, not direct evidence that stored-probe signatures
     should select weight-space task vectors. V14 should explicitly test whether
     signature-space target directions predict useful weight-space edit choices.

10. Dayan et al. (2026), *On the Expressive Power of
   Permutation-Equivariant Weight-Space Networks*,
   https://arxiv.org/abs/2602.01083.
   - V14 must specify that it tests a narrow aligned weight-to-weight operator,
     not general function-space universality.

## V14 Hypothesis

For the same four clean behaviors and subject architecture, fixed stored-probe
signatures provide useful information for selecting and scaling train-only,
alignment-canonicalized source-to-target task-vector edits inside a low-rank
weight subspace.

This is narrower than V13:

- not full raw-weight optimization;
- not pure signature-only editing;
- not a learned hypernetwork;
- not source-label inference;
- not larger-model evidence;
- not arbitrary behavior preservation.

## Proposed Method

V14 should use fresh V14 train/development/final pools before any proof claim.

Before V14 implementation, the executable preregistration must freeze all of the
following. None may be chosen from development metrics:

- V14 seed schedule, pool sizes, max attempts, claim scopes, and final redaction
  allowlist;
- alignment algorithm and exact tie-break rule;
- edit-subspace construction, rank cap, variance threshold, and PCA/SVD
  centering convention;
- signature weighting kernel, distance metric, temperature, top-k policy, and
  behavior when weights are tied;
- support-only scale grid and scale-selection objective;
- support loss weights and reductions;
- TIES-style trimming/sign-conflict diagnostic thresholds, if used;
- REPAIR/activation-variance diagnostic definition, if used;
- exact random-control count and seeds;
- exact development/final gates and final authorization hashes.

Train-only construction:

1. Generate fresh train/development/final source pools with V14-specific scopes
   and seeds.
2. Fit V9-style train-only signature statistics and selected target-attractor
   machinery.
3. For each ordered source-target behavior pair, build train-only aligned edit
   directions:
   - choose train source subjects and train target subjects only;
   - align each target subject to each source subject using the reviewed
     layerwise Hungarian logic;
   - compute target-minus-source deltas;
   - build a low-rank PCA/SVD edit basis from these deltas;
   - choose the rank by train-only variance threshold or a preregistered fixed
     cap, never by development metrics.
4. Compute a train-only centroid task vector for each ordered behavior pair.

Evaluation-time matched edit:

1. Given a development source subject and requested target behavior, compute the
   V9-style selected target-attractor signature using train-only statistics.
2. Score train target subjects by distance between their normalized signatures
   and the selected target-attractor signature.
3. Build a signature-weighted average of aligned train target-minus-source deltas
   in the ordered-pair edit subspace.
4. Select an edit scale from a fixed preregistered grid using support-only losses:
   target support BCE, source compatible support logit MSE, and conflict support
   relabeling. Heldout cases must not be used for scale selection.
5. Apply the scaled low-rank edit to the source weights and evaluate heldout
   functional metrics.

## Required Controls

V14 must include controls tied to the cited literature:

- no edit;
- V13 no-signature support optimizer;
- V12-style aligned full nearest-target retrieval;
- V12-style aligned interpolation;
- train-only centroid task vector without signature weighting;
- nearest-signature target task vector;
- uniform average target task vector;
- shuffled-signature weighted task vector;
- source-signature weighted task vector;
- TIES-style trimmed/sign-resolved task-vector control;
- activation-variance or REPAIR-style normalized interpolation control;
- random same-rank edit subspace;
- random neuron-permutation alignment control;
- no-alignment task-vector control;
- deterministic random norm-matched deltas.

A fixed-probe-signature-specific claim requires the matched edit to beat
no-signature, shuffled-signature, source-signature, and target-label-only
task-vector controls.

This is a direct V13 lesson: if no-signature, source-signature,
shuffled-signature, or target-label-only controls match or dominate V14, then V14
is negative for the fixed-probe-signature-specific claim even if target behavior
improves.

## Metrics And Gates To Preregister

V14 should keep V13's conflict-aware functional metrics:

- target heldout prediction and target margin;
- compatible source heldout output MSE;
- conflict target-label accuracy and improvement;
- Pareto-undominated rate against all controls;
- per-direction pass rates.

New V14 diagnostics should include:

- edit-subspace rank and explained variance;
- matched edit norm vs task-vector/edit-subspace controls;
- interpolation barrier along source-to-edited path using support and heldout
  metrics separately;
- alignment sensitivity: matched vs no-alignment and random-permutation controls;
- signature-weight concentration over train target subjects.

No gate should be weakened after seeing development results. If development
fails, final evaluation remains blocked.

## Reviewer Questions

The reviewer should explicitly assess:

- whether the cited literature supports the mechanism under test;
- whether V14 is narrower than the literature and does not overclaim;
- whether the plan properly accounts for permutation symmetry and basin
  alignment;
- whether support-only scale selection leaks heldout information;
- whether fixed-probe signatures are isolated from target-label-only and
  no-signature baselines;
- whether any control can trivially dominate and make the claim impossible;
- whether fresh V14 pools are required for proof use.
