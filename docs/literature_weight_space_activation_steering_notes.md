# Literature Notes For Next MUAT Experiments

Date: 2026-06-12

This note is for future preregistrations after V13. It is not a retroactive
change to V13, which was already preregistered and reviewed before this note.

## Core Literature

1. **Weight Space Learning survey.** Han et al. (2026), *A Survey of Weight
   Space Learning: Understanding, Representation, and Generation*,
   https://arxiv.org/abs/2603.10090.
   - Relevance: frames neural weights as a structured data modality with three
     main directions: understanding, representation, and generation.
   - Implication for MUAT: future plans should distinguish whether an experiment
     is testing weight-space understanding, representation learning over
     weights/signatures, or generation/editing of weights. Mixing these claims
     weakens interpretability.

2. **Universal weight subspaces.** Kaushik et al. (2025), *The Universal Weight
   Subspace Hypothesis*, https://arxiv.org/abs/2512.05117.
   - Relevance: reports shared low-dimensional spectral subspaces across many
     trained models, tasks, and domains.
   - Implication for MUAT: next experiments should test whether the small-subject
     behavior directions occupy a stable train-only subspace before optimizing
     arbitrary raw weights. A subspace-constrained editor is a better motivated
     next step than unconstrained full-weight optimization.

3. **Hypernetworks.** Ha, Dai, and Le (2016), *HyperNetworks*,
   https://arxiv.org/abs/1609.09106.
   - Relevance: establishes generating one network's weights from another
     network as a trainable architecture.
   - Implication for MUAT: if V13 suggests signature-conditioned weight editing
     has signal, a later experiment should compare optimization-based editing
     against a small hypernetwork that maps fixed-probe signatures plus requested
     target labels into constrained deltas.

4. **Task arithmetic.** Ilharco et al. (2023), *Editing Models with Task
   Arithmetic*, https://arxiv.org/abs/2212.04089.
   - Relevance: shows behavior can be steered by parameter-space task vectors,
     including addition/negation/composition.
   - Implication for MUAT: future controls should include task-vector baselines
     computed from train-only behavior centroids and source-centered deltas.
     A signature-specific claim only holds if it beats these simpler
     parameter-space directions.

5. **Model soups.** Wortsman et al. (2022), *Model soups: averaging weights of
   multiple fine-tuned models improves accuracy without increasing inference
   time*, https://arxiv.org/abs/2203.05482.
   - Relevance: supports the idea that some trained models lie in compatible
     low-error regions where weight averaging can preserve behavior.
   - Implication for MUAT: preservation failures in V11/V12 may reflect bad
     basin alignment rather than absence of useful target information. Future
     plans should include interpolation barriers and source/target basin
     compatibility diagnostics.

6. **Git Re-Basin.** Ainsworth, Hayase, and Srinivasa (2022), *Git Re-Basin:
   Merging Models modulo Permutation Symmetries*,
   https://arxiv.org/abs/2209.04836.
   - Relevance: shows that permutation symmetries can obstruct naive weight-space
     interpolation and merging, and proposes algorithms to align independently
     trained networks.
   - Implication for MUAT: future weight-editing plans should treat alignment as
     a first-class hypothesis. Every raw-weight editing result needs alignment,
     random-permutation, and no-alignment controls.

7. **Equivariant deep weight-space architectures.** Navon et al. (2023),
   *Equivariant Architectures for Learning in Deep Weight Spaces*,
   https://arxiv.org/abs/2301.12780.
   - Relevance: weight-space networks should respect hidden-neuron permutation
     symmetries; equivariant designs improve generalization on weight-space
     tasks.
   - Implication for MUAT: an unconstrained MLP over flattened weights is a weak
     baseline for serious weight-space learning. Future learned editors should be
     permutation-aware or explicitly justify why fixed canonicalization is enough.

8. **Permutation-equivariant neural functionals.** Zhou et al. (2023),
   *Permutation Equivariant Neural Functionals*,
   https://arxiv.org/abs/2302.14040.
   - Relevance: neural functional networks process weights/gradients of other
     networks while preserving permutation equivariance, and are applied to tasks
     including network editing.
   - Implication for MUAT: fixed-probe signatures plus weights should be tested
     with symmetry-respecting operators, not only raw vector regressors or
     unconstrained optimizers.

9. **Expressivity of weight-space networks.** Dayan, Eitan, and Maron (2026),
   *On the Expressive Power of Permutation-Equivariant Weight-Space Networks*,
   https://arxiv.org/abs/2602.01083.
   - Relevance: analyzes when permutation-equivariant weight-space networks can
     approximate relevant functionals and operators.
   - Implication for MUAT: future claims about decoding signatures into edited
     weights should specify whether the desired map is a function-space
     functional, weight-space functional, or weight-to-weight operator.

10. **Model merging failure modes.** Qu and Horvath (2025), *Vanishing Feature:
   Diagnosing Model Merging and Beyond*, https://arxiv.org/abs/2402.05966.
   - Relevance: identifies feature degradation mechanisms in model merging and
     motivates preservation-first strategies.
   - Implication for MUAT: compatible-source preservation should not only use
     output MSE. Future plans should measure layerwise activation preservation,
     especially early hidden layers, to distinguish target transfer from feature
     collapse.

11. **TIES-Merging.** Yadav et al. (2023), *TIES-Merging: Resolving
   Interference When Merging Models*, https://arxiv.org/abs/2306.01708.
   - Relevance: task-vector/model merging can fail from redundant small updates
     and sign disagreement across parameters.
   - Implication for MUAT: task-vector baselines should include sign-conflict and
     magnitude-trimming diagnostics. A signature-gated task vector may only look
     good if simpler interference handling was omitted.

12. **REPAIR.** Jordan et al. (2022), *REPAIR: REnormalizing Permuted
   Activations for Interpolation Repair*, https://arxiv.org/abs/2211.08403.
   - Relevance: alignment alone can still leave interpolation barriers because
     interpolated networks suffer activation variance collapse.
   - Implication for MUAT: future merge/interpolation plans should measure hidden
     activation statistics and include activation-renormalization or
     variance-collapse controls.

13. **Activation engineering.** Turner et al. (2024), *Steering Language Models
   With Activation Engineering*, https://arxiv.org/abs/2308.10248.
   - Relevance: activation additions show behavior steering can be obtained from
     activation-space contrasts without direct weight editing.
   - Implication for MUAT: this is motivational, not direct evidence that
     stored-probe signatures select weight-space task vectors. Future plans
     should test that bridge explicitly.

14. **Weight generation as a first-class modality.** Wang, Wang, and Wang (2026),
   *Position: Weight Space Should Be a First-Class Generative AI Modality*,
   https://arxiv.org/html/2605.18632v1.
   - Relevance: argues that checkpoints/weights are becoming a generative
     modeling target and emphasizes low-dimensional structure, symmetry, and
     modularity as current constraints.
   - Implication for MUAT: future work should explicitly state which structural
     assumptions are being tested and avoid broad claims about unrestricted
     checkpoint synthesis.

## Planning Requirements Going Forward

Every future experiment plan or preregistration should include:

- a literature section citing at least the papers above that directly motivate
  the method;
- an explicit hypothesis tied to one literature-backed mechanism, such as
  subspace structure, task-vector arithmetic, hypernetwork generation,
  basin-aligned averaging, or activation steering;
- controls derived from the same literature, not only ad hoc controls;
- explicit handling of hidden-neuron permutation symmetry and basin alignment
  when operating in raw weight space;
- reviewer questions asking whether the literature actually supports the
  claimed mechanism and whether the experiment's evidence is narrower than the
  cited theory.

## Candidate Literature-Grounded Next Experiments

1. **Subspace-constrained editor.**
   - Motivation: universal weight subspaces and WSL.
   - Method: learn a train-only behavior/edit subspace from accepted subjects;
     optimize or regress deltas only inside that subspace.
   - Key controls: full-weight optimizer, random same-rank subspace, PCA subspace
     with shuffled target labels, task-vector centroid baseline.

2. **Task-vector plus signature gating.**
   - Motivation: task arithmetic.
   - Method: start from train-only source-to-target centroid task vector, then use
     fixed-probe signature to select scale, low-rank mask, or mixture.
   - Key controls: ungated task vector, random gate, no-signature gate, shuffled
     signature gate.

3. **Preservation-first aligned merge.**
   - Motivation: model soups, re-basin/model merging, and vanishing-feature
     diagnostics.
   - Method: align source/target models, then optimize a merge coefficient or
     layerwise coefficient under early-activation preservation constraints.
   - Key controls: V12 aligned interpolation, no preservation loss, late-layer-only
     preservation, shuffled target.

4. **Signature-to-hypernetwork delta.**
   - Motivation: hypernetworks and weight-space generation.
   - Method: train a small hypernetwork on V13-style train subjects to generate
     constrained source-to-target deltas from source signature plus target label.
   - Key controls: target-label-only hypernetwork, signature-only with shuffled
     labels, nearest-neighbor retrieval, task-vector centroid.

5. **Permutation-equivariant learned functional.**
   - Motivation: equivariant deep weight-space architectures, neural
     functionals, and expressivity results.
   - Method: train a small permutation-aware functional that receives aligned
     source weights plus fixed-probe signatures and predicts a constrained edit
     or edit score.
   - Key controls: flattened MLP, random neuron permutations, canonicalized-only
     input, target-label-only input, shuffled signature.

6. **Activation-first validation.**
   - Motivation: activation engineering.
   - Method: before editing weights, test whether fixed-probe signature
     directions predict activation-space interventions that change heldout
     behavior while preserving compatible cases.
   - Key controls: random activation direction, mean-centered direction,
     shuffled-target direction, no intervention.

## V15 Literature Update After V13/V14

V13 showed that direct support optimization could often produce functional
target behavior, but did not establish a robust fixed-signature advantage over
no-signature or shuffled-signature controls. V14 showed that linear low-rank
task-vector transfer, even with train-only alignment and signature gating, was
too weak on development: target prediction was only 25/288 and no record passed
all gates. The next experiment should therefore avoid assuming that the
signature-to-edit map is linear.

Additional literature support gathered for V15:

- **Hypernetworks as conditional weight generators.** Ha, Dai, and Le (2016),
  *HyperNetworks*, https://arxiv.org/abs/1609.09106, introduced networks that
  generate weights for another network. Recent reviews emphasize
  data-conditioned and task-conditioned hypernetworks as a natural way to map a
  task/context embedding into target-network parameters:
  https://arxiv.org/html/2306.06955v3.
- **Few-shot/task-conditioned hypernetworks.** Przewiezlikowski et al. (2022),
  *Few-Shot Adaptation of Deep Models with Hypernetworks*,
  https://arxiv.org/html/2205.15745v3, motivate replacing per-task gradient
  optimization with a learned generator conditioned on task information.
  Continual-learning hypernetwork work similarly uses task-conditioned
  hypernetworks to generate target-model weights:
  https://www.research-collection.ethz.ch/bitstreams/7f8d1f6a-0950-4d5d-9d77-eb02458305c2/download.
- **Weight-space learning and neural functionals.** Zhou et al. (2023),
  *Permutation Equivariant Neural Functionals*,
  https://arxiv.org/abs/2302.14040, and Dayan et al. (2026),
  *On the Expressive Power of Permutation-Equivariant Weight-Space Networks*,
  https://arxiv.org/abs/2602.01083, support learned nonlinear maps over weights
  and functions. V15 should use alignment/canonicalization controls because a
  flattened MLP editor does not itself encode full permutation symmetry.
- **Foundation models and hypernetworks over weights.** Wang et al. (2025),
  *Foundation Models Secretly Understand Neural Network Weights*,
  https://arxiv.org/abs/2503.00838, motivates treating neural weights as a
  learnable/modellable modality and using transformer-like hypernetworks, but
  V15 should keep the implementation small and explicit for auditability.
- **Activation steering beyond one vector.** Postmus and Abreu (2024),
  *Steering Large Language Models using Conceptors*,
  https://arxiv.org/abs/2410.16314, show that richer activation-set geometry can
  outperform single-vector activation steering. This supports testing a
  nonlinear signature-conditioned editor rather than another single averaged
  vector.

V15 implication: train a conditional hypernetwork/editor on train-only
source-target pairs, where the target fixed-probe activation signature is an
explicit conditioning variable. The decisive controls should be target-label-only
conditioning, source-signature conditioning, shuffled-target-signature
conditioning, nearest retrieval, V13 no-signature support optimization, and
random/architecture ablations. Literature support is motivational; the claim
must remain narrow: whether train-only fixed-probe signatures improve
development behavior editing in this small subject-model setting.
