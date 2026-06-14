# V23 Inner-Validation Compute Amendment

Date: 2026-06-13

This amendment is made after V23 source-pool generation and before any V23 development
evaluation metrics are produced. It does not authorize final raw access.

## Reason

The approved V23 plan requires evaluating 9,216 sparse-subspace hyperparameter
configurations on all 156 train-only inner-validation direction records with all proof-critical
non-random controls. Implementation review accepted this as methodologically clean but flagged
the full sweep as likely impractical on the available 12-logical-CPU workstation.

The compute issue is a resource-allocation problem, not a data-leakage issue. The amendment
keeps all selection train-only and keeps the final selected configuration evaluated on the full
156-record inner-validation set.

## Literature Basis

- Successive Halving and Hyperband allocate small budgets to many configurations and reserve
  higher budgets for promising configurations; recent work continues to frame this as a standard
  approach for expensive hyperparameter optimization:
  [Provably Reduced Sample Cost in Prior-Guided Hyperparameter Optimization](https://arxiv.org/html/2606.04866v1).
- Random search is a standard alternative to exhaustive grid search when only a subset of
  hyperparameters materially affects performance; deterministic hash sampling is used here as a
  reproducible random-search analogue:
  [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).
- Hyperband explicitly frames HPO as adaptive resource allocation over randomly sampled
  configurations and reports large speedups over full-budget alternatives:
  [Hyperband](https://arxiv.org/abs/1603.06560).
- Recent studies of Successive Halving emphasize that it reduces HPO cost by discarding weak
  configurations at early rungs while increasing resources for survivors:
  [Successive Halving with Learning Curve Prediction via Latent Kronecker Gaussian Processes](https://arxiv.org/html/2508.14818v1).
- Energy-aware and HPC HPO work similarly motivates early resource allocation to avoid wasting
  compute on low-performing configurations:
  [Spend More to Save More](https://arxiv.org/abs/2412.08526),
  [Resource-Adaptive Successive Doubling](https://arxiv.org/abs/2412.02729).
- The broader V23 motivation remains behavior-conditioned weight-space learning, consistent with
  recent surveys and behavior-plus-structure findings:
  [A Survey of Weight Space Learning](https://arxiv.org/html/2603.10090v1),
  [Structure Is Not Enough](https://arxiv.org/abs/2503.17138).

## Amended Inner-Validation Selection

All records come only from the original V23 train pool inner split. Development and final pools
remain unavailable to selection.

Inner-validation jobs keep the approved source/subject/target order, but rung record budgets are
allocated as balanced source-behavior prefixes rather than global record prefixes. This is required
because the behavior suite has four source behaviors and each selected source subject produces
three target-direction records. A global prefix can omit source behaviors and make direction-level
validity checks undefined; balanced prefixes preserve every source behavior at every rung.

Within each rung, selected subjects are ordered by:

1. source behavior in `PATTERNS` order
2. subject ID ascending

Each selected subject is then expanded into direction records ordered by:

1. source behavior in `PATTERNS` order
2. subject ID ascending
3. target behavior in `PATTERNS` order, excluding source

The full 9,216-config grid remains the declared search space for audit, but the available
workstation cannot evaluate all 9,216 configs without multi-day runtime. Before any development
metrics are produced, a deterministic train-only 128-config subset is selected from the full grid.
The subset is stratified by `(k, cap_multiplier)` so every sparse width/norm-cap stratum is
represented, then configurations inside each stratum are ordered by:

`stable_hash_json({"scope":"four_behavior_functional_weight_editing_v23_config_subset","amendment_sha256":amendment_sha256,"config_hash":config_hash})`

The first quota-matched configs from each stratum are selected, with remainder quota assigned by
stratum key order. The subset is then evaluated by deterministic successive halving:

| Rung | Candidate set | Balanced subject budget | Record budget | Survivor count |
| --- | ---: | ---: | ---: | ---: |
| 0 | 128 deterministic stratified configs | first 1 subject per behavior | 12 records | 32 |
| 1 | rung-0 survivors | first 4 subjects per behavior | 48 records | 8 |
| 2 | rung-1 survivors | all 13 inner-validation subjects per behavior | 156 records | 1 selected config |

Each rung uses the same validity checks as the original plan on the records evaluated in that
rung:

- every evaluated record must produce finite public metrics;
- every evaluated record must include exactly one of each proof-critical non-random control;
- any exception marks the config invalid for that rung;
- invalid configs sort after valid configs.

Rung ranking uses the original lexicographic selection key over the evaluated budget:

1. higher target prediction rate
2. higher Pareto-undominated rate
3. higher mean matched-minus-best-control target margin
4. higher mean matched-minus-shuffled-signature target margin
5. higher mean target margin
6. lower mean compatible source-output MSE
7. lower scale/effective-zero coefficient rate
8. lower total hidden edit norm
9. lower deterministic config index

The final selected config must have a valid rung-2 full-budget result over all 156 records.
If no config survives validly through rung 2, development fails closed before development-set
evaluation.

## Audit Requirements

Development output must include:

- amendment path and SHA-256;
- rung budgets `[12, 48, 156]`;
- balanced per-behavior subject budgets `[1, 4, 13]`;
- rung survivor counts `[32, 8, 1]`;
- full-grid config count;
- evaluated subset config count;
- evaluated subset config hash;
- invalid config counts per rung;
- selected config hash;
- final full-budget inner-validation metrics for the selected config;
- selected train-statistics hash.

No final raw file may be read during amendment, inner selection, or development evaluation.
