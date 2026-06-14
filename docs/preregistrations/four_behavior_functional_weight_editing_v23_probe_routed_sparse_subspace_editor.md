# V23 Formal Preregistration: Probe-Routed Sparse Functional Subspace Editor

This preregistration binds V23 to the companion plan:

`docs/preregistrations/four_behavior_functional_weight_editing_v23_probe_routed_sparse_subspace_editor_plan.md`

The plan is the operational source of truth for implementation, testing, pool generation,
development evaluation, controls, thresholds, leakage checks, and final authorization.

## Scope

V23 tests whether fixed-probe activation signatures can route sparse, low-interference hidden
component subspace edits that outperform strong no-probe, source-probe, shuffled-probe,
target-label, nearest-target, random, and historical controls.

## Bound Requirements

The implementation must follow the plan sections covering:

- fresh V23 pool seeds and disjoint source-pool construction;
- final raw sealing and redacted-final allowlists;
- sparse hidden component-set selection;
- linearized sparse coefficient solve with exact post-edit evaluation;
- train-only inner-validation hyperparameter selection;
- proof-critical controls and random norm-matched controls;
- V22 historical-control comparison;
- aggregate, direction, per-record, and diagnostic gates;
- development `next_action` values;
- final fail-closed behavior until a passing development artifact receives reviewer authorization.

## Negative-Result Rule

If development fails any preregistered gate, the result must be treated as negative or
inconclusive. The final raw pool must remain sealed and the next action must be:

`log_negative_development_result_do_not_open_final_raw`

## Passing-Result Rule

If development passes all gates, this alone does not authorize final evaluation. The next action
must be:

`run_hash_bound_final_after_reviewer_authorization`

The final runner must remain unavailable until reviewer authorization is explicitly bound to the
exact development artifact hash, implementation hash, test hash, plan hash, formal prereg hash,
train/development/final pool hashes, combined audit hash, and final redacted audit hash.

## Literature Support

The plan cites current work on component-level editing, sparse activation steering, sparse
task-localized editing, model merging/interference, behavior-aware weight reconstruction,
weight-space learning, and mechanism-guided activation steering. These sources motivate V23's
move from single-component rank-1 editing to sparse train-routed functional subspace editing.

## Reviewer Gate

No V23 implementation or source-pool generation may proceed until a reviewer returns confidence
`5/5` on this formal preregistration and plan.
