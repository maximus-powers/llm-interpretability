# MUAT Research Log

This log tracks experiment changes, reviewer concerns, results, and next actions for the
fixed-probe activation-signature research thread.

## Review Protocol

After each meaningful research step, spawn a reviewer agent to audit the result. Do not
treat a step as accepted unless the reviewer returns confidence `5/5`. If confidence is
below `5/5`, record the critique here and address it before relying on that result.

Required reviewer fields:

- Confidence: `X/5`
- Blocking issues:
- Decision: `accepted`, `revise`, or `reject`
- Next action:

## Current Research Question

Can a fixed probe set produce activation signatures that contain enough behavioral
information to support three linked operations?

- Interpret: predict which behavior a subject model implements from its signature.
- Steer: move a learned representation toward another behavior in a measurable way.
- Decode: produce weights whose actual functional behavior changes accordingly.

## Baseline Result Under Review

Run: `runs/hypernet_focused_120e_lf001`

Focused patterns:

- `sorted_descending`
- `sorted_ascending`
- `decreasing_pairs`
- `increasing_pairs`

Latest saved evaluation:

- Interpret: raw fixed-probe signatures reached `61.4%` random-forest accuracy on
  `617` heldout focused samples, versus `32.6%` majority baseline.
- Steer: target success was `100%` across `300` edits, with mean target margin delta
  `+0.689`.
- Decode: condition-only decoded weights reached `77%` behavior accuracy on `100`
  focused validation samples.
- Reconstruction behavior accuracy was `79%`.

## Reviewer Concerns

1. The 4-label focused behavior set is partly duplicated: `sorted_ascending` and
   `increasing_pairs` are both strict adjacent increase, while `sorted_descending` and
   `decreasing_pairs` are both strict adjacent decrease.
2. The target behavior loss and evaluator use the same small hand-authored behavior
   cases, so decode and steering results may reflect case-template fitting rather than
   robust behavior changes.
3. Steering success only checks target margin greater than zero. It does not yet require
   source suppression, non-target preservation, or improvement over no-edit/random-edit
   controls.
4. Signature classifiers and weight classifiers have similar accuracy. Current evidence
   supports that signatures contain behavior information, but not that signatures are
   uniquely superior or causal.
5. Key decode/steer evaluations use small validation counts, with high per-pattern
   variance.
6. Functional reconstruction remains weak despite behavior-case success.

## Iteration 1 Plan

Address the highest-risk measurement flaw first: add generated heldout behavior-case
evaluation that samples fresh positive and negative examples from pattern predicates,
rather than only reusing the fixed cases used in training.

Acceptance criteria:

- Generated-heldout evaluation uses deterministic seed `42`.
- Generated-heldout evaluation samples at least `100` positive and `100` negative cases
  per focused pattern.
- Proof metrics include aggregate and per-pattern generated-heldout decode accuracy and
  margin.
- Proof metrics include aggregate and per-target generated-heldout steering target
  success and margin delta.
- Existing fixed-case metrics remain available for comparison.
- Direct tests cover the new metric keys.
- The focused checkpoint is reevaluated and results are saved in
  `runs/hypernet_focused_120e_lf001/evaluation/results.json`.
- The final iteration entry records command, artifact path, aggregate metrics,
  per-pattern or per-target metric summaries, reviewer confidence, decision, and next
  action.

## Iteration Entry Template

### YYYY-MM-DD - <Iteration Title>

- Objective:
- Change summary:
- Commands:
- Run IDs / artifact paths:
- Metrics:
  - Interpret:
  - Steer:
  - Decode:
  - Controls:
- Per-pattern / per-target notes:
- Reviewer:
  - Confidence:
  - Blocking issues:
  - Decision:
  - Next action:
- Follow-ups:

## Iteration Entries

### 2026-06-09 - Research Log Initialized

- Created this log.
- Reviewer:
  - Confidence: `4/5`
  - Blocking issues: missing reusable iteration template; Iteration 1 acceptance
    criteria were not concrete enough.
  - Decision: `revise`
  - Next action: add template, concrete sample counts, deterministic seed, artifact
    path, per-pattern reporting, and reviewer outcome fields.
- Revision: added required reviewer fields, a reusable iteration template, and concrete
  Iteration 1 acceptance criteria.
- Next: rerun reviewer audit of log structure, then implement generated-heldout behavior
  evaluation only if reviewer confidence is `5/5`.

### 2026-06-09 - Iteration 1: Generated-Heldout Behavior Evaluation

- Objective: reduce train/eval behavior-case overlap by evaluating Decode and Steer on
  deterministic generated cases rather than only the fixed hand-authored cases used by
  the target behavior loss.
- Change summary:
  - Added generated heldout case construction for the focused monotonic patterns.
  - Added generated-heldout decode metrics: aggregate accuracy, aggregate margin, and
    per-pattern accuracy/margin.
  - Added generated-heldout steering metrics: aggregate target success, aggregate margin
    delta, and per-target success/delta.
  - Kept existing fixed-case metrics for comparison.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_proof_metrics_report_interpret_steer_and_decode_sections; test_proof_metrics_report_interpret_steer_and_decode_sections(); print('generated heldout proof metric check passed')"`
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_functional_loss_backpropagates_to_hypernetwork_parameters, test_flat_subject_forward_matches_subject_network, test_target_behavior_loss_backpropagates_to_hypernetwork_parameters, test_fit_records_target_behavior_loss_when_labels_are_available, test_proof_metrics_report_interpret_steer_and_decode_sections, test_model_save_load_preserves_dataset_patterns; test_functional_loss_backpropagates_to_hypernetwork_parameters(); test_flat_subject_forward_matches_subject_network(); test_target_behavior_loss_backpropagates_to_hypernetwork_parameters(); test_fit_records_target_behavior_loss_when_labels_are_available(); test_proof_metrics_report_interpret_steer_and_decode_sections(); test_model_save_load_preserves_dataset_patterns(); print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_focused_120e_lf001/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_focused_120e_lf001/evaluation/results.json`
- Metrics:
  - Interpret:
    - All-focused raw signature RF accuracy remains `61.4%` against `32.6%`
      majority baseline.
  - Steer:
    - Fixed-case target success: `100.0%`; mean margin delta `+0.689`.
    - Generated-heldout target success: `100.0%`; mean margin delta `+0.377`.
    - Generated-heldout seed: `42`; cases per class per focused pattern: `100`.
  - Decode:
    - Fixed-case condition-only behavior accuracy: `77.0%`; mean margin `+0.270`.
    - Generated-heldout behavior accuracy: `71.0%`; mean margin `+0.211`.
    - Generated-heldout seed: `42`; cases per class per focused pattern: `100`.
  - Controls:
    - No new no-edit/random-edit controls yet.
- Per-pattern / per-target notes:
  - Generated-heldout decode:
    - `sorted_descending`: `51.6%`, margin `+0.097`, `31` samples.
    - `sorted_ascending`: `92.3%`, margin `+0.354`, `26` samples.
    - `decreasing_pairs`: `61.5%`, margin `+0.153`, `13` samples.
    - `increasing_pairs`: `76.7%`, margin `+0.229`, `30` samples.
  - Generated-heldout steer by target:
    - `sorted_descending`: `100.0%`, margin delta `+0.317`, `69` edits.
    - `sorted_ascending`: `100.0%`, margin delta `+0.402`, `74` edits.
    - `decreasing_pairs`: `100.0%`, margin delta `+0.174`, `87` edits.
    - `increasing_pairs`: `100.0%`, margin delta `+0.663`, `70` edits.
- Reviewer:
  - Confidence: `4/5`
  - Blocking issues:
    - Generated-heldout metrics are implemented, but the focused 4-label evaluator still
      collapses to two monotonic predicates.
    - Decode evidence is uneven: `sorted_descending` generated-heldout accuracy is
      `51.6%`; `decreasing_pairs` has only `13` validation networks.
    - Tests only check metric key presence, not deterministic case generation or case
      correctness.
    - No controls yet for no-edit, random-target, or shuffled-signature behavior.
  - Decision: `revise`
  - Next action: add deterministic generated-case tests, duplicate-label/collapsed
    direction diagnostics, and basic controls before relying on the result.
- Follow-ups:
  - Add no-edit, random-target, and shuffled-signature controls.
  - Replace or collapse duplicate focused behaviors.
  - Increase validation sample count and repeat across seeds.

### 2026-06-09 - Iteration 2 Plan: Controls and Label Validity

- Objective: address reviewer blockers from Iteration 1.
- Planned changes:
  - Add direct tests proving generated heldout cases are deterministic and match their
    predicates.
  - Add collapsed monotonic-direction diagnostics so the current focused set is not
    misrepresented as four independent behaviors.
  - Add generated-heldout no-edit steering control.
  - Add generated-heldout shuffled-signature decode control.
- Acceptance criteria:
  - Direct tests verify generated case determinism and predicate correctness.
  - `results.json` reports generated-heldout decode accuracy alongside shuffled-signature
    decode control.
  - `results.json` reports generated-heldout steering success alongside no-edit target
    success control.
  - `results.json` reports collapsed-direction generated-heldout decode and steering
    metrics.
  - Research log records commands, metrics, reviewer outcome, and next action.

### 2026-06-09 - Iteration 2: Controls and Collapsed-Direction Diagnostics

- Objective: address Iteration 1 reviewer blockers by adding generated-case tests,
  shuffled/no-edit controls, and explicit collapsed-direction diagnostics.
- Change summary:
  - Added direct tests for deterministic generated heldout cases and predicate
    correctness.
  - Added generated-heldout shuffled-signature decode control.
  - Added generated-heldout no-edit steering target-success control.
  - Added collapsed monotonic-direction decode and steering diagnostics.
  - Updated evaluator console output to print heldout controls.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_generated_heldout_cases_are_deterministic_and_predicate_correct, test_proof_metrics_report_interpret_steer_and_decode_sections; test_generated_heldout_cases_are_deterministic_and_predicate_correct(); test_proof_metrics_report_interpret_steer_and_decode_sections(); print('iteration 2 targeted checks passed')"`
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_functional_loss_backpropagates_to_hypernetwork_parameters, test_flat_subject_forward_matches_subject_network, test_target_behavior_loss_backpropagates_to_hypernetwork_parameters, test_fit_records_target_behavior_loss_when_labels_are_available, test_proof_metrics_report_interpret_steer_and_decode_sections, test_generated_heldout_cases_are_deterministic_and_predicate_correct, test_model_save_load_preserves_dataset_patterns; test_functional_loss_backpropagates_to_hypernetwork_parameters(); test_flat_subject_forward_matches_subject_network(); test_target_behavior_loss_backpropagates_to_hypernetwork_parameters(); test_fit_records_target_behavior_loss_when_labels_are_available(); test_proof_metrics_report_interpret_steer_and_decode_sections(); test_generated_heldout_cases_are_deterministic_and_predicate_correct(); test_model_save_load_preserves_dataset_patterns(); print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_focused_120e_lf001/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_focused_120e_lf001/evaluation/results.json`
- Metrics:
  - Interpret:
    - All-focused raw signature RF accuracy remains `61.4%` against `32.6%`
      majority baseline.
  - Steer:
    - Generated-heldout target success: `100.0%`.
    - Generated-heldout no-edit target success control: `33.0%`.
    - Generated-heldout cross-direction success: `100.0%`.
    - Generated-heldout cross-direction margin delta: `+0.571`.
  - Decode:
    - Generated-heldout behavior accuracy: `71.0%`.
    - Generated-heldout shuffled-signature control accuracy: `33.0%`.
    - Collapsed-direction generated-heldout accuracy: `71.0%`.
  - Controls:
    - Shuffled-signature decode control is substantially below matched-signature decode
      (`33.0%` vs `71.0%`).
    - No-edit steering target success is substantially below edited target success
      (`33.0%` vs `100.0%`).
- Per-pattern / per-target notes:
  - Decode by collapsed direction:
    - `increasing`: `83.9%`, margin `+0.287`, `56` samples.
    - `decreasing`: `54.5%`, margin `+0.114`, `44` samples.
  - Steering by collapsed target direction:
    - `increasing`: `100.0%`, margin delta `+0.529`, `144` edits.
    - `decreasing`: `100.0%`, margin delta `+0.237`, `156` edits.
  - Weakest remaining decode region is decreasing-direction generation, especially
    `sorted_descending` at `51.6%`.
- Reviewer:
  - Confidence: `4/5`
  - Blocking issues:
    - Reviewer fields were pending at review time.
    - Shuffled-signature control used a different label but not necessarily a different
      collapsed monotonic direction, so duplicate labels can leak into the negative
      control.
    - No-edit steering control aggregates same-direction duplicate targets with
      opposite-direction targets.
    - Top-level collapsed decode accuracy is a relabeling summary; per-direction metrics
      carry the useful diagnostic signal.
  - Decision: `revise`
  - Next action: add direction-stratified controls: opposite-direction shuffled decode,
    within-direction shuffled decode, and cross-direction no-edit steering success.
- Follow-ups:
  - Replace duplicate behaviors with genuinely distinct focused predicates.
  - Add random-target edit controls and confidence intervals across seeds.
  - Increase validation set size before making strong claims about per-pattern decode.

### 2026-06-09 - Iteration 3 Plan: Direction-Stratified Controls

- Objective: make controls meaningful despite duplicate monotonic labels.
- Planned changes:
  - Add opposite-direction shuffled-signature decode control.
  - Add within-direction shuffled-signature decode sanity check.
  - Add cross-direction no-edit steering target-success control.
  - Keep current aggregate controls for continuity, but rely on direction-stratified
    controls in reviewer interpretation.
- Acceptance criteria:
  - Direct tests cover the new direction-stratified metric keys.
  - `results.json` reports opposite-direction and within-direction shuffled decode
    controls.
  - `results.json` reports cross-direction edited steering success and cross-direction
    no-edit target success.
  - Research log records commands, metrics, reviewer outcome, and next action.

### 2026-06-09 - Iteration 3: Direction-Stratified Controls

- Objective: revise controls so duplicate monotonic labels cannot contaminate the
  control interpretation.
- Change summary:
  - Added opposite-direction shuffled-signature decode control.
  - Added within-direction shuffled-signature decode sanity check.
  - Added cross-direction no-edit steering target-success control.
  - Kept aggregate shuffled/no-edit controls for continuity.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_proof_metrics_report_interpret_steer_and_decode_sections; test_proof_metrics_report_interpret_steer_and_decode_sections()"`
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_generated_heldout_cases_are_deterministic_and_predicate_correct, test_proof_metrics_report_interpret_steer_and_decode_sections; test_generated_heldout_cases_are_deterministic_and_predicate_correct(); test_proof_metrics_report_interpret_steer_and_decode_sections(); print('iteration 3 targeted checks passed')"`
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_functional_loss_backpropagates_to_hypernetwork_parameters, test_flat_subject_forward_matches_subject_network, test_target_behavior_loss_backpropagates_to_hypernetwork_parameters, test_fit_records_target_behavior_loss_when_labels_are_available, test_proof_metrics_report_interpret_steer_and_decode_sections, test_generated_heldout_cases_are_deterministic_and_predicate_correct, test_model_save_load_preserves_dataset_patterns; test_functional_loss_backpropagates_to_hypernetwork_parameters(); test_flat_subject_forward_matches_subject_network(); test_target_behavior_loss_backpropagates_to_hypernetwork_parameters(); test_fit_records_target_behavior_loss_when_labels_are_available(); test_proof_metrics_report_interpret_steer_and_decode_sections(); test_generated_heldout_cases_are_deterministic_and_predicate_correct(); test_model_save_load_preserves_dataset_patterns(); print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_focused_120e_lf001/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_focused_120e_lf001/evaluation/results.json`
- Metrics:
  - Interpret:
    - Unchanged in this iteration.
  - Steer:
    - Generated-heldout all-target success: `100.0%`.
    - Generated-heldout cross-direction edited success: `100.0%`.
    - Generated-heldout cross-direction no-edit target success: `0.0%`.
    - Generated-heldout cross-direction margin delta: `+0.571`.
  - Decode:
    - Generated-heldout matched-signature behavior accuracy: `71.0%`.
    - Opposite-direction shuffled-signature control accuracy: `22.0%`.
    - Within-direction shuffled-signature sanity-check accuracy: `85.0%`.
    - Aggregate different-label shuffled-signature control accuracy: `33.0%`.
  - Controls:
    - Opposite-direction control confirms matched signatures carry directional behavior
      signal beyond a wrong-direction signature.
    - Within-direction control being higher than matched-signature accuracy confirms the
      duplicate-label concern: same-direction labels are not independent behaviors.
- Per-pattern / per-target notes:
  - Decode by collapsed direction remains uneven:
    - `increasing`: `83.9%`, `56` samples.
    - `decreasing`: `54.5%`, `44` samples.
  - Steering cross-direction control is strong in this checkpoint, but still needs
    random-target controls and more seeds before being used as final evidence.
- Reviewer:
  - Confidence: `4/5`
  - Blocking issues:
    - Reviewer outcome fields were pending at review time.
    - Tests covered control key presence, but did not yet directly assert direction
      mapping or non-empty direction-stratified control sample counts.
  - Decision: `revise`
  - Next action: record this outcome, add a semantic direction-mapping test, assert
    direction-stratified controls have non-zero sample counts, then rerun checks and
    reviewer gate.
- Follow-ups:
  - Build a new focused set with genuinely distinct predicates.
  - Add random-target edit controls.
  - Repeat across multiple train/eval seeds with larger validation counts.

### 2026-06-09 - Iteration 3 Revision: Direction-Control Test Semantics

- Objective: address reviewer concerns that the log was pending and tests only checked
  metric-key presence.
- Change summary:
  - Recorded the Iteration 3 reviewer outcome.
  - Added direct test coverage for collapsed monotonic direction mapping.
  - Added assertions that opposite-direction shuffled, within-direction shuffled, and
    cross-direction no-edit controls have non-zero sample counts.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_functional_loss_backpropagates_to_hypernetwork_parameters, test_flat_subject_forward_matches_subject_network, test_target_behavior_loss_backpropagates_to_hypernetwork_parameters, test_fit_records_target_behavior_loss_when_labels_are_available, test_proof_metrics_report_interpret_steer_and_decode_sections, test_generated_heldout_cases_are_deterministic_and_predicate_correct, test_focused_patterns_have_expected_collapsed_directions, test_model_save_load_preserves_dataset_patterns; test_functional_loss_backpropagates_to_hypernetwork_parameters(); test_flat_subject_forward_matches_subject_network(); test_target_behavior_loss_backpropagates_to_hypernetwork_parameters(); test_fit_records_target_behavior_loss_when_labels_are_available(); test_proof_metrics_report_interpret_steer_and_decode_sections(); test_generated_heldout_cases_are_deterministic_and_predicate_correct(); test_focused_patterns_have_expected_collapsed_directions(); test_model_save_load_preserves_dataset_patterns(); print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
- Run IDs / artifact paths:
  - `runs/hypernet_focused_120e_lf001/evaluation/results.json`
- Metrics:
  - Interpret: unchanged.
  - Steer: unchanged from Iteration 3; evaluator code was not changed after the last
    artifact refresh.
  - Decode: unchanged from Iteration 3; evaluator code was not changed after the last
    artifact refresh.
  - Controls: unchanged from Iteration 3; test coverage was strengthened.
- Per-pattern / per-target notes:
  - The proof-metric fixture now includes all four focused labels so within-direction
    duplicate controls are exercised.
- Reviewer:
  - Confidence: `5/5`
  - Blocking issues: none.
  - Decision: `accepted`
  - Next action: replace duplicate focused behaviors with genuinely distinct predicates.
- Follow-ups:
  - Continue to replacement of duplicate focused behaviors if this revision is accepted.

### 2026-06-09 - Adversarial Audit: Current Evidence Demoted

- Objective: answer the concern that hidden leakage or misleading metrics may still be
  present.
- Change summary:
  - Spawned independent auditor agents for data/split leakage, metric/control leakage,
    and behavior-label validity.
  - Reclassified the current focused run as an exploratory contaminated artifact, not a
    publishable proof.
- Commands:
  - `PYTHONPATH=model_zoo python - <<'PY' ... torch.load('runs/hypernet_focused_120e_lf001/model.pt') ... PY`
  - `PYTHONPATH=model_zoo python - <<'PY' ... load_dataset('maximuspowers/hypernet_validated') ... PY`
  - Multiple `rg`, `sed`, and `nl -ba` inspections of `model_zoo/hypernet`.
- Run IDs / artifact paths:
  - `runs/hypernet_focused_120e_lf001/model.pt`
  - `runs/hypernet_focused_120e_lf001/evaluation/results.json`
- Findings:
  - Critical: fixed behavior cases were used both as training targets and as evaluation
    cases. Existing fixed-case reconstruction/editing numbers are training-probe
    diagnostics, not heldout evidence.
  - High: normalizer buffers were fit on all data before the train/validation split.
    Future training must split first and fit normalization on train only.
  - High: comprehensive editing metrics build target centroids from validation/test
    target signatures. Future editing metrics must use train/calibration centroids.
  - High: `success_rate_05` and `success_rate_optimal` in editing metrics are actually
    margin-sign success, not thresholded positive/negative success.
  - High: the focused 4-label setup collapses to two monotonic predicates. Same-direction
    shuffled decode is `85%`, higher than matched-signature decode at `71%`.
  - High: generated-heldout cases still use the same finite monotonic predicate family.
    They are better than fixed cases but not a fully independent adversarial set.
  - Medium: saved split indices are disjoint for this checkpoint (`900` train, `100`
    validation, overlap `0`), but checkpoint provenance is fragile because dataset
    revision/fingerprint/source row ids are not saved.
  - Medium: full-dataset signature baseline (`2470/617`) is a dataset diagnostic, not
    directly comparable to the model's saved `900/100` split.
  - Medium: latent linear separability is in-sample and conflicts with negative
    silhouette/low ARI.
  - Medium: `generate_validated_dataset.py` is inconsistent with the hypernet
    architecture and label map.
  - Medium: the actual `hypernet_validated` signatures lack probe-set provenance
    metadata. The fixed-probe premise is therefore not auditable from the artifact alone.
- Metrics:
  - Interpret:
    - Treat raw-signature classifier metrics as evidence of dataset-level behavior signal,
      not as proof of fixed-probe causal interpretability.
  - Steer:
    - Treat fixed-case 100% edit success as contaminated.
    - Treat cross-direction generated-heldout improvement as promising but still
      requiring retraining with clean split/normalization and distinct labels.
  - Decode:
    - Treat fixed-case decode as contaminated.
    - Treat generated-heldout decode as exploratory; decreasing direction remains weak.
  - Controls:
    - Direction-stratified controls exposed duplicate-label leakage rather than resolving
      it.
- Decision:
  - Current result is useful for debugging the pipeline, but not strong enough to support
    the original research claim.
- Next action:
  - Patch safeguards that prevent future artifacts from repeating the confirmed issues:
    train-only normalization, threshold-correct editing metrics, and explicit validity
    audit warnings in evaluation output.

### 2026-06-09 - Rigor Iteration 4: Leakage Safeguards and Metric Demotion

- Objective: convert adversarial audit findings into code safeguards and artifact-level
  warnings so future runs are harder to overclaim.
- Change summary:
  - Changed `FunctionalHyperNetwork.fit()` to split before fitting normalization stats,
    so `weight_mean`, `weight_std`, `sig_mean`, and `sig_std` are train-only for future
    checkpoints.
  - Corrected editing threshold success semantics: `success_rate_05` and
    `success_rate_optimal` now require both target positive outputs above threshold and
    target negative outputs below threshold.
  - Preserved margin-sign success as a separately named metric.
  - Expanded serialized editing pair details with pos/neg accuracies and margin
    min/max/std.
  - Added `validity_audit` to saved evaluation results, explicitly warning that this
    checkpoint may have normalization leakage, duplicate labels, reused fixed cases,
    incomplete dataset provenance, and non-auditable signature probe provenance.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_fit_normalization_stats_use_training_split_only, test_threshold_success_requires_positive_and_negative_threshold_accuracy; test_threshold_success_requires_positive_and_negative_threshold_accuracy(); test_fit_normalization_stats_use_training_split_only(); print('leak/threshold checks passed')"`
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_functional_loss_backpropagates_to_hypernetwork_parameters, test_flat_subject_forward_matches_subject_network, test_target_behavior_loss_backpropagates_to_hypernetwork_parameters, test_fit_records_target_behavior_loss_when_labels_are_available, test_fit_normalization_stats_use_training_split_only, test_threshold_success_requires_positive_and_negative_threshold_accuracy, test_proof_metrics_report_interpret_steer_and_decode_sections, test_generated_heldout_cases_are_deterministic_and_predicate_correct, test_focused_patterns_have_expected_collapsed_directions, test_model_save_load_preserves_dataset_patterns; test_functional_loss_backpropagates_to_hypernetwork_parameters(); test_flat_subject_forward_matches_subject_network(); test_target_behavior_loss_backpropagates_to_hypernetwork_parameters(); test_fit_records_target_behavior_loss_when_labels_are_available(); test_fit_normalization_stats_use_training_split_only(); test_threshold_success_requires_positive_and_negative_threshold_accuracy(); test_proof_metrics_report_interpret_steer_and_decode_sections(); test_generated_heldout_cases_are_deterministic_and_predicate_correct(); test_focused_patterns_have_expected_collapsed_directions(); test_model_save_load_preserves_dataset_patterns(); print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/models/functional_hypernetwork.py model_zoo/hypernet/evaluation/editing_metrics.py model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_focused_120e_lf001/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_focused_120e_lf001/evaluation/results.json`
- Metrics:
  - Interpret:
    - Still demoted to dataset-level signal, not fixed-probe causal proof.
  - Steer:
    - Corrected legacy editing threshold success remains `100%` for this checkpoint.
    - Margin-sign edit success is also `100%`.
    - Validity audit warns legacy editing matrices use eval target centroids and fixed
      cases are contaminated.
  - Decode:
    - Generated-heldout decode remains `71%`; fixed-case decode remains demoted.
  - Controls:
    - Validity audit now travels with the artifact so readers see known risks without
      needing this log.
- Reviewer:
  - Confidence: `5/5` for safeguard/demotion pass, not for the underlying research
    result.
  - Blocking issues: none for this iteration.
  - Decision: `accepted`
  - Next action: retrain a new checkpoint after these safeguards, with duplicate focused
    labels replaced and dataset/probe provenance saved. Do not use the current focused
    run as proof evidence.
- Follow-ups:
  - Retrain after train-only normalization; current checkpoint cannot be retrospectively
    de-leaked.
  - Replace duplicate labels before next proof run.
  - Stop using fixed-case metrics as proof claims.
  - Save dataset fingerprint/source row ids and signature probe-set provenance.

### 2026-06-09 - Clean Proof Design Gate

- Objective: define a non-contaminated proof run before retraining.
- Claim standard:
  - Interpret: fixed-probe signatures must predict heldout model behavior labels above
    majority/shuffled controls on a saved train/validation split.
  - Steer: editing a heldout source model toward a target behavior must improve target
    heldout-case margin and beat the no-edit target-success control.
  - Decode: condition-only/generated weights from heldout signatures must satisfy the
    target behavior on heldout cases and beat wrong-signature controls.
- Clean proof behavior set:
  - `sorted_ascending`
  - `sorted_descending`
  - `has_majority`
  - `mountain_pattern`
- Predicate definitions:
  - `sorted_ascending`: all adjacent digits strictly increase.
  - `sorted_descending`: all adjacent digits strictly decrease.
  - `has_majority`: at least one digit appears three or more times in the length-5
    sequence.
  - `mountain_pattern`: first three digits strictly increase to the center and last
    three strictly decrease from the center.
- Exhaustive predicate audit over all `100000` length-5 digit sequences:
  - Pairwise overlap among the four selected predicates: `0` for every off-diagonal
    pair.
  - Positive counts: `sorted_ascending=252`, `sorted_descending=252`,
    `has_majority=8560`, `mountain_pattern=2892`.
- Planned safeguards:
  - Generate deterministic support and heldout behavior cases from the full sequence
    universe with no sequence reused across support and heldout splits.
  - Use support cases only for `target_behavior_loss`.
  - Use heldout cases only for proof decode/steer metrics.
  - Generate behavior-suite cases and split hashes before training, then save the
    metadata in the checkpoint and evaluation artifact.
  - Deduplicate exact duplicate rows before splitting. Report duplicate counts for row
    payload hashes, flattened weight hashes, signature hashes, and `(weights,
    signature, label)` hashes.
  - Save behavior-suite metadata in checkpoints/evaluation artifacts, including seed,
    split hashes, case counts, predicate counts, and overlap matrix.
  - Save HuggingFace dataset id, split, fingerprint, row hashes, row indices, and
    pattern filter in checkpoints/evaluation artifacts.
  - Compute steering target centroids/signatures from training rows only.
  - Keep existing fixed-case/legacy metrics demoted; proof claims must refer to the
    clean behavior-suite metrics.
- Probe provenance limitation:
  - The existing `maximuspowers/hypernet_validated` rows include fixed signature columns
    but do not embed the full probe-set provenance in each row.
  - Unless the run regenerates signatures from a saved canonical probe dataset and saves
    its file hash, the claim must be worded as fixed-signature-column evidence, not a
    fully audited fixed-probe-set proof.
- Numeric acceptance threshold for first small-scale proof:
  - Clean behavior suite metadata reports global support/heldout case overlap `0`.
  - Each selected behavior has at least `50` heldout model samples after deduplication
    and splitting.
  - Heldout raw-signature Random Forest accuracy over the clean four behaviors is at
    least `0.45` and at least `0.15` above heldout majority baseline.
  - Heldout encoded-condition classifier accuracy over the clean four behaviors is at
    least `0.40` and at least `0.10` above heldout majority baseline.
  - Per-behavior interpret recall is at least `0.30` for every selected behavior.
  - Heldout generated decode accuracy is at least `0.60` overall, at least `0.50` for
    every selected behavior, and at least `0.20` above wrong-signature/shuffled control.
  - Heldout generated decode mean margin is positive overall and non-negative for every
    selected behavior.
  - Heldout steering target success is at least `0.60` overall, at least `0.50` for
    every target behavior, and at least `0.25` above no-edit target success.
  - Heldout steering mean target margin delta is at least `+0.05` overall and positive
    for every target behavior.
  - Any missed threshold demotes the run to exploratory evidence; no aggregate score can
    override a failed per-behavior threshold.
  - Reviewer must return `5/5` confidence for the design before implementation and
    `5/5` confidence for each result-bearing checkpoint before it is called evidence.
- Reviewer:
  - Confidence: `4/5`.
  - Blocking issues:
    - Acceptance thresholds were not pre-registered numerically.
    - Aggregate metrics could hide per-behavior failures.
    - Deduplication was not required before train/validation split.
    - Steering target signatures/centroids had to be explicitly train-only.
    - Probe provenance was still not fully auditable from the existing dataset artifact.
  - Decision: `revise`.
  - Revision:
    - Added numeric pass/fail thresholds for interpret, decode, and steer.
    - Added per-behavior minimums and sample-count requirements.
    - Added pre-split deduplication and duplicate reporting requirements.
    - Made train-only steering centroids explicit.
    - Required behavior-suite metadata before training.
    - Added explicit fixed-signature-column limitation unless canonical probe-set
      provenance is regenerated and hashed.
  - Revision reviewer:
    - Confidence: `5/5`.
    - Blocking issues: none for implementation-gate purposes.
    - Decision: `accepted for implementation`.
    - Required implementation posture: enforce the gate in code/artifacts as hard
      assertions/demotion logic, not only as prose in this log.

### 2026-06-09 - Clean Proof Implementation Gate: Suite, Provenance, and Artifact Gate

- Objective: implement the accepted clean-proof foundation before retraining.
- Change summary:
  - Added canonical clean proof behavior suite in `model_zoo/hypernet/behavior_suite.py`.
  - Added deterministic support/heldout behavior case generation with disjoint sequence
    hashes and exhaustive overlap metadata.
  - Added pre-registered proof thresholds in code, including per-behavior and margin
    gates.
  - Added dataset provenance/dedup helpers in `model_zoo/hypernet/dataset_provenance.py`.
  - Changed loader-level deduplication to remove duplicate weight/signature examples
    before train/validation splitting, independent of label.
  - Saved dataset provenance and behavior-suite metadata in checkpoints and evaluation
    artifacts.
  - Passed clean support cases into `target_behavior_loss`; heldout cases are reserved
    for proof metrics.
  - Added `clean_proof_gate` to proof metrics. It uses heldout majority baseline,
    per-behavior thresholds, positive margin gates, no-edit/shuffled controls, and
    behavior-suite checkpoint hash matching. Failed gates demote runs to `exploratory`.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import ...; ...; print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/behavior_suite.py model_zoo/hypernet/dataset_provenance.py model_zoo/hypernet/models/functional_hypernetwork.py model_zoo/hypernet/train.py model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
- Reviewer:
  - Confidence: `5/5`.
  - Blocking issues: none for foundation/plumbing/evaluation-gate step.
  - Decision: `accepted`.
  - Notes:
    - Reviewer verified heldout-majority gating and behavior-suite checkpoint metadata
      matching.
    - Validation behavior loss still uses support cases as a training diagnostic only;
      proof evidence must come from heldout proof metrics.
- Next action:
  - Run a small clean smoke train/eval artifact to verify provenance, behavior-suite
    hashes, and demotion logic are present before any longer proof run.

### 2026-06-09 - Clean Smoke Run: Artifact Structure Check

- Objective: produce a small clean checkpoint/evaluation artifact to verify the proof
  gate and provenance machinery, not to test the hypothesis.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 1 --batch-size 16 --latent-dim 16 --condition-dim 32 --hidden-dim 64 --lr 0.001 --lambda-kl 0.01 --lambda-functional 1.0 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --max-samples 120 --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_clean_smoke_1e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_clean_smoke_1e/model.pt`
  - `PYTHONPATH=model_zoo python - <<'PY' ... inspect results.json ... PY`
- Run IDs / artifact paths:
  - `runs/hypernet_clean_smoke_1e/model.pt`
  - `runs/hypernet_clean_smoke_1e/evaluation/results.json`
- Structural checks:
  - `clean_proof_gate.status`: `exploratory`.
  - `clean_proof_gate.passed`: `False`.
  - `behavior_suite.name`: `clean_proof_v1`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `dataset_provenance.dataset_id`: `maximuspowers/hypernet_validated`.
  - `dataset_provenance.deduplication.deduplication_key`: `weight_signature_hash`.
  - `dataset_provenance.probe_provenance.claim_scope`: `fixed_signature_column`.
- Metrics:
  - Heldout sample counts were intentionally too small for proof:
    - `sorted_ascending=1`, `sorted_descending=3`, `has_majority=5`,
      `mountain_pattern=3`.
  - Heldout majority baseline: `41.7%`.
  - Generated-heldout decode accuracy: `41.7%`.
  - Shuffled-signature decode control: `66.7%`.
  - Generated-heldout steering target success: `63.9%`.
  - No-edit target success: `19.4%`.
- Gate failures:
  - All four behaviors failed the minimum heldout sample-count threshold.
  - `sorted_descending` failed interpret recall.
  - Decode failed overall accuracy and delta-vs-control thresholds.
  - `sorted_descending` and `has_majority` failed decode thresholds.
  - `has_majority` failed steering success and positive margin-delta thresholds.
- Decision:
  - Smoke artifact is structurally useful and correctly demoted.
  - It is not evidence for the research hypothesis.
- Reviewer:
  - Confidence: `4/5`.
  - Blocking issues: none for the smoke artifact; accepted as safe to proceed to a
    larger clean proof run.
  - Required revisions for `5/5`:
    - Compare evaluation-reloaded dataset provenance against checkpoint provenance.
    - Make `validity_audit` context-aware so clean checkpoints do not carry stale
      normalization/provenance warnings.
    - Clarify that sample-count failures refer to validation model samples, not
      generated behavior-suite heldout cases.
  - Revision:
    - Added `compare_dataset_provenance()` and stored reload match/mismatch fields in
      `dataset_provenance`.
    - Made `clean_proof_gate` fail if evaluation-reloaded provenance does not match the
      checkpoint.
    - Regenerated the smoke checkpoint/evaluation so checkpoint metadata includes
      `normalization_fit_scope=train_split`.
    - Made `validity_audit` report complete dataset provenance and no normalization
      leakage warning for the regenerated clean checkpoint.
    - Changed gate failure wording to `validation model sample count`.
  - Revision metrics:
    - `clean_proof_gate.status`: `exploratory`.
    - `dataset_provenance.reload_matches_checkpoint`: `True`.
    - `dataset_provenance.reload_comparison.mismatched_fields`: `[]`.
    - `validity_audit.dataset_provenance_complete`: `True`.
    - `validity_audit.dataset_reload_matches_checkpoint`: `True`.
    - `validity_audit.normalization_fit_scope`: `train_split`.
    - `validity_audit.normalization_leakage_in_this_checkpoint_possible`: `False`.
  - Revision commands:
    - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 1 --batch-size 16 --latent-dim 16 --condition-dim 32 --hidden-dim 64 --lr 0.001 --lambda-kl 0.01 --lambda-functional 1.0 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --max-samples 120 --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_clean_smoke_1e --no-tensorboard`
    - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_clean_smoke_1e/model.pt`
    - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import ...; ...; print('direct hypernet checks passed')"`
    - `python -m py_compile model_zoo/hypernet/behavior_suite.py model_zoo/hypernet/dataset_provenance.py model_zoo/hypernet/models/functional_hypernetwork.py model_zoo/hypernet/train.py model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
  - Revision reviewer:
    - Confidence: `5/5`.
    - Blocking issues: none.
    - Decision: `accepted for the smoke-results step`.
    - Scope note: accepted only as structurally rigorous and honestly demoted; the
      1-epoch smoke result is not evidence for the hypothesis.

### 2026-06-09 - Clean Proof Run Attempt: 40 Epochs

- Objective: run the first larger clean four-behavior checkpoint under the accepted
  proof gate.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_clean_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_clean_40e/model.pt`
  - `PYTHONPATH=model_zoo python - <<'PY' ... inspect results.json ... PY`
- Run IDs / artifact paths:
  - `runs/hypernet_clean_40e/model.pt`
  - `runs/hypernet_clean_40e/evaluation/results.json`
- Structural checks:
  - Deduplication removed `205` exact weight/signature duplicates before splitting.
  - Train/validation split: `3592/399`.
  - Validation samples per behavior:
    - `sorted_descending=76`, `sorted_ascending=109`, `mountain_pattern=91`,
      `has_majority=123`.
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`, not fully audited fixed-probe
    provenance.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
    - Failure: `decode delta vs shuffled control below threshold`.
  - Interpret:
    - Heldout majority baseline: `30.8%`.
    - Raw signature Random Forest: `76.7%`.
    - Encoded condition classifier: `72.9%`.
    - Per-behavior RF recall:
      - `sorted_descending=67.1%`
      - `sorted_ascending=79.8%`
      - `mountain_pattern=51.6%`
      - `has_majority=98.4%`
  - Decode:
    - Generated-heldout matched-signature behavior accuracy: `99.5%`.
    - Shuffled-signature control accuracy: `85.5%`.
    - Delta vs shuffled control: `14.0pp`, below the pre-registered `20pp` threshold.
    - Per-behavior generated-heldout decode:
      - `sorted_ascending=98.2%`, mean margin `+0.884`
      - `sorted_descending=100.0%`, mean margin `+0.869`
      - `has_majority=100.0%`, mean margin `+0.216`
      - `mountain_pattern=100.0%`, mean margin `+0.825`
  - Steer:
    - Generated-heldout target success: `100.0%`.
    - No-edit target success: `24.4%`.
    - Mean target margin delta: `+0.831`.
    - Per-target steering success: `100.0%` for all four behaviors.
- Decision:
  - This is strong exploratory evidence for interpretability and steering, and strong
    generated behavior satisfaction, but it is not proof under the pre-registered gate
    because the decoder control is too high.
  - The next investigation should focus on why shuffled signatures still decode to valid
    target behavior so often: possible behavior-prior collapse, target loss overpowering
    signature specificity, or an insufficiently adversarial/wrong-target control.
- Reviewer:
  - Confidence: `4/5`.
  - Decision: accepted as a correctly demoted exploratory result; not accepted as proof
    evidence.
  - Blocking issues:
    - Decode is not specific enough to matched signatures: matched generated-heldout
      decode is `99.5%`, but shuffled-signature control is `85.5%`.
    - The result should not be described as decoded functional models from fixed
      signatures yet; it currently supports behavior satisfaction more than
      signature-specific decoding.
    - Weight reconstruction remains weak (`overall weight cosine=-0.004`) despite high
      behavior satisfaction.
    - Probe provenance remains unaudited; claim scope is still fixed stored signature
      columns.
    - Legacy editing matrices remain demoted; proof claims must use
      `proof.clean_proof_gate`.
  - Required revisions before another proof attempt:
    - Add stronger decode controls: wrong-target signatures stratified by target,
      random/noise/null signatures, train-centroid-only signatures, and
      condition-ablation decoder.
    - Report per-target shuffled accuracy and margin, not only aggregate.
    - Keep behavior satisfaction distinct from signature-conditioned decoding in labels.
    - Preserve fixed-signature-column limitation unless probe regeneration/hash
      provenance is added.

### 2026-06-09 - Decode Specificity Revision: Hard Negatives and Ablation Controls

- Objective: diagnose and fix the high shuffled-signature decode control from the clean
  40-epoch run.
- Finding:
  - Added null/noise/train-centroid/condition-ablation decode controls and reran
    evaluation on `runs/hypernet_clean_40e`.
  - Control results showed the failure was more severe than aggregate shuffled accuracy:
    - Null signature decode accuracy: `100.0%`.
    - Train-centroid signature decode accuracy: `100.0%`.
    - Noise signature decode accuracy: `92.7%`.
    - Condition-ablation decode accuracy: `72.7%`.
  - Root cause: behavior-suite negative cases excluded other clean-behavior positives.
    A generated model could learn "any selected clean behavior vs generic negative" and
    satisfy all four target evaluations without target-specific signature decoding.
- Change summary:
  - Added per-target decode controls:
    - wrong/shuffled target signatures
    - null train-mean signature
    - noise signature
    - train-centroid signature
    - zero-condition decoder ablation
  - Changed clean behavior-suite generation so each target's negative set includes
    other selected behavior positives as hard negatives plus generic negatives.
  - Added tests requiring hard negatives in support and heldout cases.
  - Added pre-registered gate thresholds for null/noise/train-centroid/condition
    ablation controls, plus per-behavior control deltas.
  - Added `generated_heldout_shuffled_source_target` matrix to expose which wrong-source
    labels drive shuffled-control success.
  - Made `clean_proof_gate` fail when any new decode specificity control is too high.
- Commands:
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import test_decode_specificity_controls_are_reported_per_target, test_proof_metrics_report_interpret_steer_and_decode_sections; ..."`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_clean_40e/model.pt`
  - `PYTHONPATH=model_zoo python -c "from hypernet.tests.test_functional_hypernetwork import ...; ...; print('direct hypernet checks passed')"`
  - `python -m py_compile model_zoo/hypernet/behavior_suite.py model_zoo/hypernet/dataset_provenance.py model_zoo/hypernet/models/functional_hypernetwork.py model_zoo/hypernet/train.py model_zoo/hypernet/evaluation/pipeline.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`
- Decision:
  - The previous `runs/hypernet_clean_40e` checkpoint remains exploratory and is now
    known to have been evaluated against insufficiently target-specific negatives.
  - Future checkpoints must be trained and evaluated against the hard-negative behavior
    suite; old suite hashes should not match current evaluation hashes.
- Reviewer:
  - Confidence: `4/5` on first review.
  - Blocking issue:
    - Hard negatives fixed the root cause, but the proof gate reported rather than
      enforced the new null/noise/train-centroid/condition-ablation controls.
  - Revision:
    - Added gate enforcement for all new decode controls.
    - Added regression tests that high null/ablation controls demote proof status.
    - Added source-target shuffled matrix reporting.
  - Second review:
    - Confidence: `4/5`.
    - Blocking issues:
      - Missing decode-control fields could still pass the proof gate.
      - Threshold registry test did not require the new control thresholds.
      - Source-target shuffled matrix needed either coverage enforcement or explicit
        diagnostic-only scope.
  - Second revision:
    - Missing null/noise/train-centroid/condition-ablation controls now fail
      `clean_proof_gate`.
    - Each aggregate and per-target decode specificity control must have `n_samples > 0`.
    - Threshold registry test now includes all new decode-control thresholds.
    - Added regression test for missing specificity controls.
    - Marked the source-target shuffled matrix as diagnostic-only while aggregate and
      per-target shuffled controls remain gated.
  - Revision reviewer:
    - Confidence: `5/5`.
    - Blocking issues: none for proof-grade retraining readiness.
    - Decision: accepted for a fresh clean retrain.
    - Scope note: even a passing retrain remains fixed stored signature-column evidence
      unless probe-generation provenance is regenerated and hashed.

### 2026-06-09 - Clean Proof Run Attempt: Hard-Negative 40 Epochs

- Objective: retrain under the hard-negative behavior suite and strict decode-control
  proof gate.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_clean_hardneg_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_clean_hardneg_40e/model.pt`
  - `PYTHONPATH=model_zoo python - <<'PY' ... inspect results.json ... PY`
- Run IDs / artifact paths:
  - `runs/hypernet_clean_hardneg_40e/model.pt`
  - `runs/hypernet_clean_hardneg_40e/evaluation/results.json`
- Structural checks:
  - Deduplication removed `205` exact weight/signature duplicates before splitting.
  - Train/validation split: `3592/399`.
  - Validation samples per behavior:
    - `sorted_descending=81`, `sorted_ascending=92`, `mountain_pattern=98`,
      `has_majority=128`.
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `behavior_suite.hard_negative_fraction`: `0.5`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
    - Main failures: decode deltas vs train-centroid, null, noise, condition-ablation,
      and per-target shuffled controls.
  - Interpret:
    - Heldout majority baseline: `32.1%`.
    - Raw signature Random Forest: `74.9%`.
    - Encoded condition classifier: `81.5%`.
    - Per-behavior RF recall:
      - `sorted_descending=69.1%`
      - `sorted_ascending=77.2%`
      - `mountain_pattern=48.0%`
      - `has_majority=97.7%`
  - Decode:
    - Matched generated-heldout accuracy: `95.5%`.
    - Shuffled-signature control: `40.4%`.
    - Null-signature control: `67.9%`.
    - Noise-signature control: `54.1%`.
    - Train-centroid signature control: `100.0%`.
    - Condition-ablation control: `75.4%`.
    - Per-target matched decode:
      - `sorted_ascending=100.0%`
      - `sorted_descending=97.5%`
      - `has_majority=95.3%`
      - `mountain_pattern=89.8%`
  - Steer:
    - Generated-heldout target success: `100.0%`.
    - No-edit target success: `22.0%`.
    - Mean target margin delta: `+0.819`.
    - Per-target steering success: `100.0%` for all four behaviors.
- Decision:
  - Hard negatives fixed the universal clean-behavior shortcut enough to reduce shuffled
    control from `85.5%` to `40.4%`.
  - The run is still not proof because train-centroid and zero/noisy condition controls
    remain too high. This indicates the decoder can generate class-prototype or default
    behavior models without relying sufficiently on the specific heldout signature.
  - Strong evidence remains for signature interpretability and steering under stored
    signature columns; decode remains the limiting hypothesis component.
- Reviewer:
  - Confidence: `4/5`.
  - Decision: accepted as a correctly demoted exploratory result; not accepted as proof
    evidence.
  - Blocking issues:
    - Decode specificity is still not established. The decoder can use behavior priors
      or class prototypes instead of instance-specific signatures.
    - `sorted_ascending` controls are especially weak: null/noise/centroid/ablation
      controls nearly solve the target.
    - Steering and interpretation remain promising but do not rescue the decode claim.
  - Required revisions:
    - Separate "behavior-class prototype decoding" from "instance-specific
      signature-conditioned decoding."
    - Add a condition-dependence objective and gate.
    - Add subject-specific functional distillation/ranking: generated model from
      signature `i` should match subject `i`'s heldout functional outputs better than
      same-class centroid, null/noise, zero-condition, or wrong signatures.
    - Report source-target shuffled failures as first-class gate metrics.
  - Next intervention:
    - Make decode specificity a contrastive/distillation problem. More epochs under the
      current behavior-only target are unlikely to prove the stronger claim.

### 2026-06-09 - Decode Specificity Gate Revision for Next Retrain

- Objective: close the class-prototype loophole exposed by
  `runs/hypernet_clean_hardneg_40e`.
- Reviewer pre-patch confidence: `2/5`.
- Blocking issues found:
  - Subject-functional specificity omitted `noise_mse`, so the reported best control
    was incomplete.
  - Training used naive rolled wrong conditions, which could compare mostly against
    other classes instead of same-class different subjects.
  - Subject-functional specificity was aggregate-only, so a failed behavior or weak
    paired control could be hidden by global averages.
- Revisions:
  - Added `noise_mse` to subject-functional specificity controls.
  - Replaced unpaired aggregate lists with per-subject paired records.
  - Gate now requires aggregate and per-behavior:
    - minimum sample counts,
    - mean improvement over each subject's best available control,
    - paired win rate,
    - median improvement.
  - Controls include wrong signature, train-mean/null signature, noise signature,
    train-class centroid signature, and zero-condition ablation.
  - Added `FunctionalHyperNetwork.build_wrong_condition()` to prefer same-label,
    different-subject contrast conditions; it falls back to a rolled condition only
    when a batch has no same-label alternative.
  - Renamed a test helper that was accidentally pytest-collectable as `test_behavior`.
- Verification:
  - Targeted regression checks failed before the production patch and passed after it.
  - Direct hypernet checks passed.
  - `python -m py_compile` passed on touched hypernet files.
  - No linting was run, per project instruction.
- Reviewer post-patch confidence: `4/5`.
- Reviewer decision:
  - No blocking fixes remain for retraining readiness.
  - Confidence remains below `5/5` until a fresh trained/evaluated artifact clears the
    stricter gate.
- Next action:
  - Fresh retrain through `hypernet.train` so behavior-suite metadata and dataset
    provenance are recorded in the checkpoint.

### 2026-06-09 - Clean Specificity 40 Epoch Run

- Objective: test whether the same-label contrast objective and paired subject-specific
  proof gate establish instance-specific signature-conditioned decoding.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_clean_specificity_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_clean_specificity_40e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_clean_specificity_40e/model.pt`
  - `runs/hypernet_clean_specificity_40e/evaluation/results.json`
- Structural checks:
  - Deduplication removed `205` exact weight/signature duplicates before splitting.
  - Train/validation split: `3592/399`.
  - Validation samples per behavior:
    - `sorted_descending=82`, `sorted_ascending=93`, `mountain_pattern=92`,
      `has_majority=132`.
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
    - Main failures: train-centroid and condition-ablation decode controls, per-target
      decode-control deltas, and aggregate/per-behavior subject-functional specificity.
  - Interpret:
    - Heldout majority baseline: `33.1%`.
    - Raw signature Random Forest: `75.4%`.
    - Encoded condition classifier: `79.4%`.
    - Per-behavior RF recall:
      - `sorted_descending=69.5%`
      - `sorted_ascending=81.7%`
      - `mountain_pattern=40.2%`
      - `has_majority=99.2%`
  - Decode:
    - Matched generated-heldout accuracy: `94.5%`.
    - Shuffled-signature control: `52.6%`.
    - Null-signature control: `66.9%`.
    - Noise-signature control: `62.9%`.
    - Train-centroid signature control: `100.0%`.
    - Condition-ablation control: `76.9%`.
    - Per-target matched decode:
      - `sorted_ascending=100.0%`
      - `sorted_descending=91.5%`
      - `has_majority=95.5%`
      - `mountain_pattern=90.2%`
  - Subject-functional specificity:
    - Matched MSE: `228.93`.
    - Best paired control MSE: `124.13`.
    - Mean improvement vs best control: `-104.79`.
    - Paired win rate: `50.6%`.
    - Median improvement: `+0.24`.
    - Per-behavior improvement / win rate:
      - `sorted_ascending=-12.46`, `53.8%`
      - `sorted_descending=-491.00`, `12.2%`
      - `has_majority=-0.13`, `76.5%`
      - `mountain_pattern=-4.07`, `44.6%`
  - Steer:
    - Generated-heldout target success: `100.0%`.
    - No-edit target success: `21.9%`.
    - Mean target margin delta: `+0.870`.
- Decision:
  - Correctly demoted. The run still learns behavior/prototype priors rather than
    reliable subject-specific condition decoding.
  - Interpretation and steering remain strong under fixed signature columns.
  - Decode remains the limiting component.
- Reviewer:
  - Confidence: `5/5` on the interpretation.
  - Blocking validity concerns: none that invalidate the demotion.
  - Required next intervention:
    - Put the proof controls into the training objective: matched condition-only decode
      must beat same-label wrong subject, null/mean, noise, train-centroid, and
      zero-condition ablation controls on subject-output MSE with margin.
    - Penalize zero-condition and centroid decodes when they satisfy clean heldout
      behavior too well.

### 2026-06-09 - Control-Objective Training Revision

- Objective: train directly against the controls that demoted
  `runs/hypernet_clean_specificity_40e`.
- Revisions:
  - Functional distillation probes now use deterministic digit-domain inputs instead of
    normal random inputs.
  - Condition-specificity loss now ranks matched condition-only decode against the hard
    minimum of available controls:
    - same-label/different-subject wrong condition,
    - train-mean/null signature,
    - train-mean plus noise signature,
    - train-class centroid signature,
    - zero-condition ablation.
  - Train-split signature mean/std/centroids are computed after the train/validation
    split and persisted in checkpoints.
  - Added a behavior-prior penalty for train-centroid and zero-condition control
    decodes when they solve support behavior cases too well.
  - Exposed control-objective knobs through `train()`, YAML, CLI, and checkpoint config:
    - `lambda_condition_specificity`
    - `lambda_control_behavior_penalty`
    - `functional_loss_samples`
- Verification:
  - New regression tests first failed on the old implementation, then passed after the
    patch.
  - Full direct hypernet checks passed.
  - `python -m py_compile` passed on touched hypernet files.
  - No linting was run, per project instruction.
- Reviewer:
  - First readiness review: `3/5`.
  - Blocking issue: `lambda_control_behavior_penalty` and related knobs were not wired
    through the training entrypoint.
  - After plumbing fix: `4/5`.
  - Blocking issues: none.
  - Non-blocking note: use explicit CLI flags or YAML because some legacy CLI defaults
    are weaker than the intended proof-run defaults.
- Next action:
  - Fresh retrain with explicit control-objective flags.

### 2026-06-09 - Control-Objective 40 Epoch Run

- Objective: test whether training directly against hard subject-specific controls fixes
  instance-specific decode without relying on post-hoc metrics.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 2.0 --lambda-control-behavior-penalty 1.0 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_control_objective_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_control_objective_40e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_control_objective_40e/model.pt`
  - `runs/hypernet_control_objective_40e/evaluation/results.json`
- Structural checks:
  - Deduplication removed `205` exact weight/signature duplicates before splitting.
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
    - Main failures: steering target success/delta and remaining per-target decode
      deltas against null/noise/centroid/ablation controls.
  - Interpret:
    - Heldout majority baseline: `33.1%`.
    - Raw signature Random Forest: `75.7%`.
    - Encoded condition classifier: `80.2%`.
  - Decode:
    - Matched generated-heldout accuracy: `88.2%`.
    - Shuffled-signature control: `31.1%`.
    - Null-signature control: `47.6%`.
    - Noise-signature control: `52.9%`.
    - Train-centroid signature control: `33.1%`.
    - Condition-ablation control: `66.9%`.
  - Subject-functional specificity:
    - Matched MSE: `50.66`.
    - Best paired control MSE: `103.47`.
    - Mean improvement vs best control: `+52.81`.
    - Paired win rate: `81.5%`.
    - Median improvement: `+13.95`.
    - Per-behavior improvements are positive for all four behaviors.
  - Steer:
    - Generated-heldout target success: `17.8%`.
    - No-edit target success: `21.6%`.
    - Mean target margin delta: `+0.091`.
    - Per-target success:
      - `sorted_ascending=0%`
      - `sorted_descending=0%`
      - `has_majority=79.8%`
      - `mountain_pattern=0%`
- Decision:
  - Correctly demoted. The control objective fixed the subject-functional specificity
    failure but exposed a missing edit-path objective.
  - This is useful progress, not proof.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns: none that invalidate the demotion.
  - Required next intervention:
    - Train the actual steering path `decode(source_z, target_condition)` using sampled
      cross-label targets.
    - Apply target behavior loss and margin-delta loss to edited weights.
    - Keep hard subject-specificity controls.
    - Strengthen null/ablation control penalties so those controls are driven below
      zero behavior margin.

### 2026-06-09 - Edit-Path Objective Revision

- Objective: train the same edit path used by proof steering:
  `decode(source_z, target_condition)`.
- Revisions:
  - Added edit-path loss weights:
    - `lambda_edit_behavior`
    - `lambda_edit_margin_delta`
  - Added train/YAML/CLI/checkpoint plumbing for both edit weights.
  - Added `build_edit_targets()` and `compute_edit_margin_delta_loss()`.
  - Training now decodes edited weights from source latent means and target conditions,
    then applies target behavior loss and target margin-delta loss.
  - Control behavior-prior penalty now includes null, train-centroid, and
    zero-condition controls and uses a negative allowed margin.
  - Initial reviewer readiness was `4/5`; residual concern was mismatch between
    in-batch edit targets and proof steering's train-centroid targets.
  - Added `build_all_edit_targets()` so each source trains against every different-label
    train-centroid target when centroids are available.
  - Training and validation edit losses now use that expanded target set.
- Verification:
  - New edit-target coverage regression failed before implementation and passed after.
  - Full direct hypernet checks passed.
  - `python -m py_compile` passed on touched hypernet files.
  - No linting was run, per project instruction.
- Reviewer:
  - Final readiness confidence: `5/5`.
  - Blocking issues: none.
  - Non-blocking notes:
    - Expanded edit targets multiply edit-loss compute by `n_classes - 1`.
    - Target centroid conditions are encoded with training-mode dropout during training.
    - Use explicit CLI/YAML values for all objective weights.
- Next action:
  - Fresh retrain with explicit specificity, control, and edit objective weights.

### 2026-06-09 - Edit-Objective 40 Epoch Run

- Objective: test the full edit-path objective with all source-to-centroid target
  coverage while preserving hard subject-specificity controls.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 2.0 --lambda-control-behavior-penalty 1.0 --lambda-edit-behavior 1.0 --lambda-edit-margin-delta 1.0 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_edit_objective_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_edit_objective_40e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_edit_objective_40e/model.pt`
  - `runs/hypernet_edit_objective_40e/evaluation/results.json`
- Structural checks:
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
    - Main failures: null/ablation control deltas and `has_majority`
      subject-functional specificity.
  - Interpret:
    - Heldout majority baseline: `34.1%`.
    - Raw signature Random Forest: `75.9%`.
    - Encoded condition classifier: `85.2%`.
  - Decode:
    - Matched generated-heldout accuracy: `93.0%`.
    - Shuffled-signature control: `29.6%`.
    - Null-signature control: `79.2%`.
    - Noise-signature control: `46.4%`.
    - Train-centroid signature control: `34.1%`.
    - Condition-ablation control: `45.1%`.
    - Null control solves `sorted_ascending`, `has_majority`, and
      `mountain_pattern` at `100%`.
    - Condition-ablation solves `sorted_ascending` and `mountain_pattern` at `100%`.
    - Train-centroid control solves `has_majority` at `100%`.
  - Subject-functional specificity:
    - Aggregate improvement vs best control: `+15.28`.
    - Aggregate win rate: `61.7%`.
    - `has_majority` fails:
      - improvement `-45.25`,
      - win rate `30.9%`,
      - median improvement `-25.21`.
  - Steer:
    - Generated-heldout target success: `100%`.
    - Per-target success: `100%` for all four behaviors.
    - Mean target margin delta: `+0.858`.
- Decision:
  - Correctly demoted. The edit-path objective fixed steering, while decode remains
    blocked by behavior-prior controls and `has_majority` subject specificity.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns: none that invalidate the demotion.
  - Required next intervention:
    - Make control penalties behavior-aware/per-target.
    - Penalize null, ablation, and train-centroid controls on each clean behavior target
      separately using support/generated digit cases.
    - Use threshold/accuracy-style losses that force controls below the decision
      boundary, not just low average margin.
    - Add class-balanced or `has_majority`-upweighted subject-specificity probes.
    - Keep the edit-path objective unchanged.

### 2026-06-09 - Behavior-Aware Control Revision

- Objective: address the remaining null/ablation/train-centroid behavior-prior controls
  and `has_majority` subject-specificity failure from
  `runs/hypernet_edit_objective_40e`.
- Revisions:
  - Control behavior-prior penalty now uses sigmoid positive-vs-negative margin,
    matching the generated-heldout proof criterion more closely than raw-logit margin.
  - Null, train-centroid, and zero-condition controls remain penalized per target
    behavior.
  - Added label-specific subject-specificity probes built from clean support
    positive+negative cases.
  - Train/validation subject-specificity loss now uses those label-specific probes when
    behavior cases are available.
  - Edit-path objective remains unchanged.
- Verification:
  - New label-specific probe regression failed before implementation and passed after.
  - Full direct hypernet checks passed.
  - `python -m py_compile` passed on touched hypernet files.
  - No linting was run, per project instruction.
- Reviewer:
  - Readiness confidence: `4/5`.
  - Blocking issues: none.
  - Non-blocking notes:
    - Control penalty is still differentiable margin-threshold style, not discrete
      accuracy.
    - Training uses support probes while proof uses heldout/random query outputs.
    - `has_majority` remains the behavior to watch.
- Next action:
  - Fresh retrain with behavior-aware specificity/control penalties.

### 2026-06-09 - Behavior-Aware 40 Epoch Run

- Objective: test behavior-aware control penalties and label-specific
  subject-specificity probes while keeping the edit-path objective.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 2.0 --lambda-control-behavior-penalty 1.0 --lambda-edit-behavior 1.0 --lambda-edit-margin-delta 1.0 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_behavior_aware_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_behavior_aware_40e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_behavior_aware_40e/model.pt`
  - `runs/hypernet_behavior_aware_40e/evaluation/results.json`
- Structural checks:
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
  - Interpret:
    - Heldout majority baseline: `30.1%`.
    - Raw signature Random Forest: `75.2%`.
    - Encoded condition classifier: `82.2%`.
  - Decode:
    - Matched generated-heldout accuracy: `87.7%`.
    - Shuffled-signature control: `27.8%`.
    - Null-signature control: `55.1%`.
    - Noise-signature control: `49.4%`.
    - Train-centroid signature control: `77.9%`.
    - Condition-ablation control: `55.1%`.
    - Train-centroid control solves `sorted_ascending`, `sorted_descending`, and
      `has_majority` at `100%`.
    - Null/ablation controls solve `sorted_descending` and `has_majority` at `100%`.
  - Subject-functional specificity:
    - Aggregate improvement vs best control: `+29.71`.
    - Aggregate win rate: `75.7%`.
    - `has_majority` is fixed relative to the previous run:
      - improvement `+8.30`,
      - win rate `83.3%`.
    - `sorted_descending` regressed:
      - improvement `-7.85`,
      - win rate `47.0%`.
  - Steer:
    - Generated-heldout target success: `100%`.
    - Per-target success: `100%` for all four behaviors.
    - Mean target margin delta: `+0.811`.
- Decision:
  - Correctly demoted. The run fixed steering and `has_majority` specificity, but
    zero-latent controls became behavior prototypes.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns: none that invalidate the demotion.
  - Required next intervention:
    - Target the zero-latent control path directly and separately from edit-path
      centroid targets.
    - Evaluate each zero-latent control decode against every clean behavior target.
    - Penalize positive-vs-negative sigmoid margin above a negative threshold for every
      target.
    - Weight train-centroid controls highest.
    - Upweight or add probes for `sorted_descending` specificity.

### 2026-06-10 - All-Target Control Evaluation Revision

- Objective: align training and proof evaluation for zero-latent controls.
- Revisions:
  - Added all-target zero-latent control penalty:
    - every null/train-centroid/zero-condition control decode is penalized against
      every clean behavior target, not just its source label.
    - train-centroid controls receive additional configurable weight.
    - `sorted_descending` subject-specificity samples receive additional configurable
      weight.
  - Added all-target control matrices to proof metrics:
    - `generated_heldout_null_signature_all_target`
    - `generated_heldout_noise_signature_all_target`
    - `generated_heldout_train_centroid_signature_all_target`
    - `generated_heldout_condition_ablation_all_target`
  - Each matrix reports source-control pattern -> target pattern -> accuracy,
    mean margin, and sample count.
  - Clean proof gate now requires each all-target cell to have samples and to remain
    below the matched target decode by the pre-registered control delta threshold.
  - Added new objective knobs to the default YAML for auditability.
- Verification:
  - New evaluation regression tests failed before implementation and passed after.
  - Full direct hypernet checks passed.
  - `python -m py_compile` passed on touched hypernet files.
  - No linting was run, per project instruction.
- Reviewer:
  - Readiness confidence: `5/5`.
  - Blocking issues: none.
  - Non-blocking notes:
    - all-target gate thresholds accuracy deltas; margins are reported for analysis.
    - default YAML remains conservative unless explicit proof-run flags are used.
- Next action:
  - Fresh retrain with explicit all-target control objective flags and upgraded proof
    evaluation.

### 2026-06-10 - All-Target Control 40 Epoch Run

- Objective: test the all-target zero-latent control objective and upgraded proof gate.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 2.0 --lambda-control-behavior-penalty 1.0 --train-centroid-control-weight 3.0 --sorted-descending-specificity-weight 2.0 --lambda-edit-behavior 1.0 --lambda-edit-margin-delta 1.0 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_all_target_control_40e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_all_target_control_40e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_all_target_control_40e/model.pt`
  - `runs/hypernet_all_target_control_40e/evaluation/results.json`
- Structural checks:
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Duplicate weight/signature pairs removed before splitting/evaluation: `205`.
  - Probe scope remains `fixed_signature_column`; probe provenance is not yet
    regenerated and hashed.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
  - Interpret:
    - Heldout majority baseline: `36.6%`.
    - Raw signature Random Forest: `74.4%`.
    - Encoded condition classifier: `82.0%`.
    - Per-behavior raw-signature RF recall:
      - `sorted_descending`: `62.2%`.
      - `sorted_ascending`: `72.4%`.
      - `mountain_pattern`: `49.4%`.
      - `has_majority`: `95.9%`.
  - Decode:
    - Matched generated-heldout accuracy: `87.2%`.
    - Per-target matched generated-heldout accuracy:
      - `sorted_ascending`: `93.9%`.
      - `sorted_descending`: `81.1%`.
      - `has_majority`: `97.9%`.
      - `mountain_pattern`: `65.4%`.
    - Shuffled-signature control: `37.6%`.
    - Noise-signature control: `54.6%`.
    - Condition-ablation control: `0.0%`.
    - Null-signature control: `20.3%` aggregate, but all-target null controls solve
      `mountain_pattern` at `100%` for every source behavior.
    - Train-centroid signature control: `100%` aggregate and `100%` for each
      matched target.
    - All-target train-centroid controls still form prototypes:
      - source `sorted_ascending` solves `sorted_ascending` and `mountain_pattern`
        at `100%`;
      - source `sorted_descending` solves `sorted_descending` and `has_majority`
        at `100%`;
      - source `has_majority` solves `has_majority` at `100%`;
      - source `mountain_pattern` solves `mountain_pattern` at `100%`.
  - Subject-functional specificity:
    - Aggregate improvement vs best control: `+43.08`.
    - Aggregate win rate: `80.7%`.
    - Aggregate median improvement: `+8.16`.
    - `sorted_descending` fails:
      - matched MSE `36.27`;
      - train-centroid MSE `33.10`;
      - best-control MSE `26.60`;
      - improvement `-9.67`.
  - Steer:
    - Generated-heldout target success: `100%`.
    - No-edit target success: `20.3%`.
    - Per-target success: `100%` for all four behaviors.
    - Mean target margin delta: `+0.821`.
- Decision:
  - Correctly demoted. The run has real signal in interpretation, steering, and
    matched decode metrics, but it is not clean evidence for matched fixed-probe
    signature decoding into subject-specific functional behavior. Zero-latent
    controls still decode behavior prototypes, and `sorted_descending` does not beat
    the best subject-functional control.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns:
    - train-centroid zero-latent controls are decisive confounds;
    - null control is not inert for `mountain_pattern`;
    - `sorted_descending` subject specificity fails;
    - probe provenance is not auditable beyond fixed-signature-column evidence.
  - Non-blocking notes:
    - no evidence of ordinary split or normalization leakage in this checkpoint;
    - support/heldout behavior cases have zero overlap;
    - positive results are not meaningless, but controls prevent proof-level claims.
- Next action:
  - Make the next run prioritize subject-specificity over behavior success:
    matched signatures must beat same-label train-centroid and other zero-latent
    controls per behavior before decode/steering headlines count as evidence.

### 2026-06-10 - Specificity-Balanced 30 Epoch Run

- Objective: test stronger subject-specificity and zero-latent control pressure without
  the validation explosion seen in the interrupted specificity-first run.
- Commands:
  - Interrupted, no checkpoint saved:
    - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 40 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 8.0 --lambda-control-behavior-penalty 8.0 --train-centroid-control-weight 12.0 --sorted-descending-specificity-weight 6.0 --lambda-edit-behavior 0.5 --lambda-edit-margin-delta 0.5 --functional-loss-samples 48 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_specificity_first_40e --no-tensorboard`
  - Completed:
    - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 30 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 4.0 --lambda-control-behavior-penalty 4.0 --train-centroid-control-weight 8.0 --sorted-descending-specificity-weight 4.0 --lambda-edit-behavior 0.75 --lambda-edit-margin-delta 0.75 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_specificity_balanced_30e --no-tensorboard`
    - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_specificity_balanced_30e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_specificity_balanced_30e/model.pt`
  - `runs/hypernet_specificity_balanced_30e/evaluation/results.json`
- Structural checks:
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Duplicate weight/signature pairs removed: `205`.
  - Probe scope remains `fixed_signature_column`; probe provenance is still not
    regenerated and hashed.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
  - Interpret:
    - Heldout majority baseline: `34.1%`.
    - Raw signature Random Forest: `78.2%`.
    - Encoded condition classifier: `84.2%`.
    - Per-behavior raw-signature RF recall:
      - `sorted_descending`: `74.7%`.
      - `sorted_ascending`: `76.6%`.
      - `mountain_pattern`: `54.4%`.
      - `has_majority`: `97.1%`.
  - Decode:
    - Matched generated-heldout accuracy: `89.5%`.
    - Per-target matched generated-heldout accuracy:
      - `sorted_ascending`: `97.9%`.
      - `sorted_descending`: `84.8%`.
      - `has_majority`: `98.5%`.
      - `mountain_pattern`: `71.1%`.
    - Shuffled-signature control: `30.6%`.
    - Noise-signature control: `49.1%`.
    - Null-signature control: `0.0%` aggregate and `0.0%` for every all-target
      control cell.
    - Train-centroid signature control: `53.9%` aggregate, still `100%` for
      `sorted_descending` and `has_majority`.
    - Condition-ablation control: `34.1%` aggregate, with every source behavior
      solving `has_majority` at `100%` despite a small positive margin.
    - All-target train-centroid failures still include:
      - `sorted_ascending -> mountain_pattern`: `100%`;
      - `sorted_descending -> sorted_descending`: `100%`;
      - `sorted_descending -> has_majority`: `100%`;
      - `has_majority -> has_majority`: `100%`.
  - Subject-functional specificity:
    - Aggregate improvement vs best control: `+36.14`.
    - Aggregate win rate: `83.2%`.
    - Aggregate median improvement: `+15.70`.
    - `sorted_descending` still fails:
      - matched MSE `40.16`;
      - train-centroid MSE `36.97`;
      - best-control MSE `35.62`;
      - improvement `-4.54`.
  - Steer:
    - Generated-heldout target success: `100%`.
    - No-edit target success: `21.3%`.
    - Per-target success: `100%` for all four behaviors.
    - Mean target margin delta: `+0.717`.
- Decision:
  - Correctly demoted, with meaningful progress. The stricter run fixed the null
    prototype and improved matched decode/interpret metrics, but same-label
    train-centroid controls and condition-ablation controls still solve key behaviors.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns:
    - train-centroid controls still solve `sorted_descending` and `has_majority`;
    - condition ablation solves `has_majority` for every source behavior;
    - `sorted_descending` subject specificity still fails against controls;
    - probe provenance remains fixed-signature-column only.
  - Non-blocking notes:
    - no evidence of ordinary split or normalization leakage;
    - progress is real but not proof.
- Next action:
  - Add a hard-negative control objective or architectural constraint. The loss must
    punish the worst behavior a zero-latent control solves, not only the average
    all-target control margin.

### 2026-06-10 - Hard-Negative Control Objective Revision

- Objective: address reviewer feedback that scalar average reweighting lets a
  zero-latent control solve one target while averaging the failure away.
- Revisions:
  - Added target-weighted all-target control loss.
  - Added explicit hard-negative control loss:
    - evaluates every clean behavior target for a zero-latent control;
    - penalizes the worst target loss rather than the average target loss.
  - Added configurable objective knobs:
    - `lambda_control_hard_negative_penalty`;
    - `condition_ablation_control_weight`;
    - `control_sorted_descending_target_weight`;
    - `control_has_majority_target_weight`.
  - Added training and validation history fields:
    - `control_hard_negative_penalty_loss`;
    - `val_control_hard_negative_penalty_loss`.
  - Saved the new knobs in checkpoint config metadata.
- Verification:
  - New hard-negative regression failed before implementation and passed after.
  - Focused target-weight/config tests passed.
  - Full direct hypernet checks: `46` direct tests passed.
  - `python -m py_compile` passed on touched hypernet/training/evaluation files.
  - No linting was run, per project instruction.

### 2026-06-10 - Hard-Control 20 Epoch Run

- Objective: test whether explicit worst-target control pressure removes the remaining
  zero-latent prototypes.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 20 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 4.0 --lambda-control-behavior-penalty 3.0 --lambda-control-hard-negative-penalty 4.0 --train-centroid-control-weight 10.0 --condition-ablation-control-weight 6.0 --control-sorted-descending-target-weight 4.0 --control-has-majority-target-weight 4.0 --sorted-descending-specificity-weight 5.0 --lambda-edit-behavior 0.5 --lambda-edit-margin-delta 0.5 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_hard_control_20e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_hard_control_20e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_hard_control_20e/model.pt`
  - `runs/hypernet_hard_control_20e/evaluation/results.json`
- Structural checks:
  - `dataset_provenance.reload_matches_checkpoint`: `True`.
  - `behavior_suite.matches_checkpoint_metadata`: `True`.
  - `behavior_suite.support_heldout_overlap_count`: `0`.
  - `validity_audit.normalization_fit_scope`: `train_split`.
  - Probe scope remains `fixed_signature_column`; probe provenance is still not
    regenerated and hashed.
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
  - Interpret:
    - Heldout majority baseline: `30.1%`.
    - Raw signature Random Forest: `74.7%`.
    - Encoded condition classifier: `77.9%`.
    - Per-behavior raw-signature RF recall:
      - `sorted_descending`: `76.5%`.
      - `sorted_ascending`: `74.5%`.
      - `mountain_pattern`: `43.5%`.
      - `has_majority`: `97.5%`.
  - Decode:
    - Matched generated-heldout accuracy: `85.0%`.
    - Per-target matched generated-heldout accuracy:
      - `sorted_ascending`: `95.1%`.
      - `sorted_descending`: `88.2%`.
      - `has_majority`: `97.5%`.
      - `mountain_pattern`: `54.3%`.
    - Shuffled-signature control: `35.3%`.
    - Null-signature control: `30.1%`, with every source behavior solving
      `has_majority` at `100%`.
    - Noise-signature control: `57.9%`, with broad partial all-target success.
    - Train-centroid signature control: `78.7%`; prototypes shifted rather than
      disappearing.
    - Condition-ablation control: `30.1%`, with every source behavior solving
      `has_majority` at `100%` despite near-zero positive margin.
  - Subject-functional specificity:
    - Aggregate improvement vs best control: `+41.72`.
    - Aggregate win rate: `83.5%`.
    - Aggregate median improvement: `+13.09`.
    - `sorted_descending` is fixed relative to prior run:
      - matched MSE `38.61`;
      - best-control MSE `72.82`;
      - improvement `+34.21`;
      - win rate `87.1%`.
  - Steer:
    - Generated-heldout target success: `100%`.
    - No-edit target success: `22.6%`.
    - Per-target success: `100%` for all four behaviors.
    - Mean target margin delta: `+0.653`.
  - Training history:
    - Final hard-control loss: `0.560`.
    - Final validation hard-control loss: `0.572`.
    - Final condition-specificity loss: `8.366`.
    - Final validation condition-specificity loss: `30.253`.
- Decision:
  - Correctly demoted. The hard-negative objective fixed the `sorted_descending`
    subject-specificity blocker but did not produce proof: zero-latent behavior
    prototypes remain and have shifted toward `has_majority` and train-centroid target
    prototypes.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns:
    - null control regressed into a `has_majority` prior;
    - condition ablation solves `has_majority` at `100%` due near-zero positive
      margins;
    - train-centroid prototypes shifted rather than disappeared;
    - noise controls retain broad partial behavior success.
  - Non-blocking notes:
    - no evidence of ordinary data leakage;
    - probe provenance still caps the claim at fixed-signature-column evidence;
    - `sorted_descending` subject specificity is real progress.
- Next action:
  - Add a calibrated margin deadband for behavior correctness so near-zero positive
    margins do not count as clean behavior success. Apply it to proof metrics and
    controls symmetrically, then rerun evaluation on existing checkpoints before
    retraining.

### 2026-06-10 - Calibrated Behavior-Correctness Deadband

- Objective: prevent near-zero positive margins from counting as clean behavior
  success in proof metrics.
- Revisions:
  - Added `BEHAVIOR_CORRECTNESS_MARGIN_THRESHOLD = 0.02` to the proof evaluator.
  - `_evaluate_network_on_cases` now counts behavior as correct only when
    `positive_mean - negative_mean > 0.02`.
  - Added raw correctness reporting for audit:
    - `raw_correct`;
    - `raw_accuracy`;
    - `generated_heldout_raw_behavior_accuracy`.
  - Applied the calibrated threshold symmetrically to matched generated-heldout decode
    and all decode controls.
- Verification:
  - New deadband regression failed before implementation and passed after.
  - Full direct hypernet checks: `47` direct tests passed.
  - `python -m py_compile` passed on touched hypernet/training/evaluation files.
  - No linting was run, per project instruction.
- Re-evaluation:
  - Command:
    - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_hard_control_20e/model.pt`
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
  - Decode:
    - Matched generated-heldout calibrated accuracy: `80.2%`.
    - Matched generated-heldout raw accuracy: `85.0%`.
    - `mountain_pattern` matched calibrated accuracy falls to `42.4%`
      (`54.3%` raw), exposing weak margin.
    - Shuffled-signature calibrated control: `20.8%` (`35.3%` raw).
    - Condition-ablation calibrated control: `0.0%`; raw `has_majority` remains
      `100%`, confirming this was a near-zero threshold artifact.
    - Null `has_majority` remains a real blocker:
      - calibrated `100%`;
      - raw `100%`;
      - mean margin `+0.0322`.
    - Train-centroid prototypes remain real blockers:
      - aggregate calibrated `78.7%`;
      - `sorted_ascending -> sorted_ascending`: `100%`;
      - `sorted_descending -> has_majority`: `100%`;
      - `has_majority -> has_majority`: `100%`;
      - `has_majority -> mountain_pattern`: `100%`;
      - `mountain_pattern -> mountain_pattern`: `100%`.
- Decision:
  - Correctly demoted. The deadband improves measurement rigor and separates near-zero
    threshold artifacts from real behavior priors, but the run is still not proof.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns:
    - `mountain_pattern` matched calibrated decode is only `42.4%`;
    - null control still solves `has_majority` at `100%` calibrated/raw;
    - train-centroid prototypes remain calibrated successes;
    - noise controls still partially solve multiple targets;
    - probe provenance remains fixed-signature-column only.
  - Non-blocking notes:
    - no evidence of ordinary leakage;
    - subject specificity still passes overall and per behavior;
    - raw metrics are auditable in per-target/all-target summaries.
- Next action:
  - Align training with the calibrated proof target:
    - matched behavior margins must exceed the calibrated positive threshold;
    - null/centroid/ablation controls must fall below a negative buffer;
    - `has_majority` controls and matched `mountain_pattern` margin need explicit
      pressure.

### 2026-06-10 - Calibrated Training Objective Revision

- Objective: align training with the calibrated proof evaluator.
- Revisions:
  - Added `compute_calibrated_behavior_margin_loss`:
    - uses sigmoid positive-vs-negative margin;
    - trains matched condition-only decodes to exceed the proof margin threshold;
    - supports target-specific weighting for weak targets.
  - Added training knobs:
    - `lambda_calibrated_behavior_margin`;
    - `matched_behavior_min_margin`;
    - `matched_mountain_target_weight`;
    - `control_max_allowed_margin`.
  - Control penalties now use configurable `control_max_allowed_margin` instead of a
    hard-coded negative buffer.
  - Added training/validation history fields:
    - `calibrated_behavior_margin_loss`;
    - `val_calibrated_behavior_margin_loss`.
- Verification:
  - New calibrated matched-margin regression failed before implementation and passed
    after.
  - Focused calibrated-margin tests passed.
  - Full direct hypernet checks: `48` direct tests passed.
  - `python -m py_compile` passed on touched hypernet/training/evaluation files.
  - No linting was run, per project instruction.

### 2026-06-10 - Calibrated-Margin 20 Epoch Run

- Objective: test whether explicitly training matched condition-only decodes to clear a
  positive calibrated margin improves proof decode while stricter control buffers
  suppress remaining control priors.
- Commands:
  - `PYTHONPATH=model_zoo python -m hypernet.train --epochs 20 --batch-size 64 --latent-dim 128 --condition-dim 128 --hidden-dim 512 --lr 0.001 --lambda-kl 0.01 --lambda-functional 10.0 --lambda-condition-specificity 4.0 --lambda-calibrated-behavior-margin 4.0 --matched-behavior-min-margin 0.05 --matched-mountain-target-weight 6.0 --lambda-control-behavior-penalty 2.0 --lambda-control-hard-negative-penalty 3.0 --control-max-allowed-margin -0.08 --train-centroid-control-weight 10.0 --condition-ablation-control-weight 6.0 --control-sorted-descending-target-weight 4.0 --control-has-majority-target-weight 6.0 --sorted-descending-specificity-weight 5.0 --lambda-edit-behavior 0.5 --lambda-edit-margin-delta 0.5 --functional-loss-samples 32 --use-functional-loss --functional-loss-start-epoch 0 --device cpu --patterns sorted_ascending,sorted_descending,has_majority,mountain_pattern --run-dir runs/hypernet_calibrated_margin_20e --no-tensorboard`
  - `PYTHONPATH=model_zoo python -m hypernet.evaluation.pipeline --model runs/hypernet_calibrated_margin_20e/model.pt`
- Run IDs / artifact paths:
  - `runs/hypernet_calibrated_margin_20e/model.pt`
  - `runs/hypernet_calibrated_margin_20e/evaluation/results.json`
- Metrics:
  - Clean proof gate:
    - `status`: `exploratory`.
    - `passed`: `False`.
  - Decode:
    - Matched generated-heldout calibrated accuracy: `80.7%`.
    - Matched generated-heldout raw accuracy: `84.0%`.
    - Per-target matched calibrated accuracy:
      - `sorted_ascending`: `89.5%`;
      - `sorted_descending`: `77.3%`;
      - `has_majority`: `96.5%`;
      - `mountain_pattern`: `47.1%`.
    - Null control still solves `has_majority`:
      - calibrated `100%`;
      - mean margin `+0.0232`.
    - Condition-ablation control still solves `has_majority`:
      - calibrated `100%`;
      - mean margin `+0.0336`.
    - Train-centroid aggregate improved to `21.3%`, but remaining calibrated
      all-target failures include:
      - `sorted_descending -> has_majority`: `100%`;
      - `mountain_pattern -> mountain_pattern`: `100%`.
    - Noise controls retain broad partial success.
  - Subject-functional specificity:
    - Aggregate improvement vs best control: `+35.70`.
    - Aggregate win rate: `83.2%`.
    - Per-behavior specificity passes, including `sorted_descending`.
  - Training history:
    - Final calibrated margin loss: `0.024`.
    - Final validation calibrated margin loss: `0.061`.
    - Final validation specificity loss worsened to `39.28`.
- Decision:
  - Correctly demoted. The calibrated-margin objective improved train-centroid
    aggregate controls but did not remove the `has_majority` prior or recover reliable
    `mountain_pattern` matched decode.
- Reviewer:
  - Confidence: `5/5` on result interpretation.
  - Blocking validity concerns:
    - `mountain_pattern` matched decode remains below threshold;
    - null control still solves `has_majority`;
    - condition ablation now has a calibrated `has_majority` prior;
    - train-centroid still has calibrated prototypes;
    - noise controls still partially solve targets.
  - Non-blocking notes:
    - no evidence of ordinary data leakage;
    - probe provenance remains fixed-signature-column only;
    - subject specificity is a genuine positive result across all behaviors.
- Next action:
  - Stop adding scalar penalties to the same decoder path. The next methodological
    change should be architectural control separation: null/centroid/ablation paths
    must be structurally unable to express behavior prototypes, while matched
    signatures add behavior through a gated or residual path.

### 2026-06-10 - Centroid-Residual Decoder 20 Epoch Run

- Objective: test whether architectural separation removes train-centroid prototype
  leakage while preserving matched fixed-signature decode.
- Implementation:
  - Added optional centroid-residual decoder:
    - `base_decoder(z)` supplies the neutral latent path;
    - residual path uses `condition - condition_baseline`;
    - residual output subtracts `residual_decoder(0)` so centroid residuals contribute
      exactly zero behavior delta.
  - Training/evaluation now pass same-label train-centroid encoded baselines where
    labels are available.
  - Added config knobs:
    - `use_condition_residual_decoder`;
    - `condition_residual_scale`.
- Commands:
  - `python model_zoo/hypernet/train.py --config model_zoo/configs/hypernet/centroid_residual_20e.yaml --run-dir runs/hypernet_centroid_residual_20e --no-tensorboard`
  - `python model_zoo/hypernet/evaluation/pipeline.py --model runs/hypernet_centroid_residual_20e/model.pt --output runs/hypernet_centroid_residual_20e/evaluation`
- Verification:
  - Full direct hypernet checks: `52` direct tests passed.
  - `python -m py_compile` passed on touched hypernet model/training/evaluation/test
    files.
  - No linting was run, per project instruction.
- Run IDs / artifact paths:
  - `model_zoo/configs/hypernet/centroid_residual_20e.yaml`
  - `runs/hypernet_centroid_residual_20e/model.pt`
  - `runs/hypernet_centroid_residual_20e/evaluation/results.json`
- Metrics:
  - Clean proof gate:
    - `passed`: `False`.
  - Interpret:
    - Focused heldout raw signature random-forest accuracy: `76.2%`.
    - Focused heldout majority baseline: `33.8%`.
  - Decode:
    - Matched generated-heldout calibrated accuracy: `94.5%`.
    - Matched generated-heldout raw accuracy: `100.0%`.
    - Train-centroid control calibrated accuracy: `0.0%`.
    - Null control calibrated accuracy: `0.0%`.
    - Condition-ablation control calibrated accuracy: `0.0%`.
    - Noise-signature control calibrated accuracy: `80.5%`.
    - Shuffled-signature control calibrated accuracy: `70.9%`.
    - Opposite-direction shuffled-signature control calibrated accuracy: `98.3%`.
  - Subject-functional specificity:
    - Matched MSE: `22.32`.
    - Best-control MSE: `52.79`.
    - Mean improvement vs best control: `+30.47`.
    - Win rate vs best control: `67.9%`.
    - Weak spots:
      - `has_majority` improvement only `+0.826`;
      - `sorted_descending` win rate only `56.9%`.
  - Steering:
    - Generated-heldout target success: `49.6%`.
    - Generated-heldout no-edit target success: `13.7%`.
    - Generated-heldout cross-direction target success: `22.0%`.
    - Generated-heldout mean target margin delta: `+0.134`.
  - Training summary warnings:
    - final reconstruction cosine: `-0.0335`;
    - lightweight train-script editing success: `0/2` pairs.
- Decision:
  - Correctly demoted. The architecture removed train-centroid prototype leakage, but
    matched decode is still invalidated by high noise and shuffled/opposite-direction
    controls. Steering remains below threshold.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `5/5` on the demotion.
  - Main concern:
    - shuffled/noise/opposite controls likely show broad behavior or direction
      information in residual condition space, not ordinary split leakage, but still
      invalidate the matched-signature proof claim.
- Next action:
  - Make shuffled/noise residual controls first-class training negatives. The next
    experiment should require matched residuals to beat shuffled, opposite-direction,
    and noise residuals on behavior accuracy and subject-output MSE before accepting
    decode or steering headlines.

### 2026-06-10 - Centroid-Residual + Shuffled-Negative 20 Epoch Run

- Objective: test whether first-class different-label shuffled residual negatives
  suppress the high shuffled/opposite-direction controls from the previous residual
  checkpoint.
- Implementation:
  - Added config knobs:
    - `lambda_shuffled_residual_contrastive`;
    - `shuffled_residual_min_delta`.
  - Added deterministic different-label shuffled controls via
    `build_different_label_condition`.
  - Added matched-vs-control behavior-margin loss requiring matched decodes to beat
    shuffled controls on the source-label behavior margin.
  - Added different-label shuffled controls to the subject-output specificity control
    set during training and validation.
- Commands:
  - `python model_zoo/hypernet/train.py --config model_zoo/configs/hypernet/centroid_residual_shuffled_20e.yaml --run-dir runs/hypernet_centroid_residual_shuffled_20e --no-tensorboard`
  - `python model_zoo/hypernet/evaluation/pipeline.py --model runs/hypernet_centroid_residual_shuffled_20e/model.pt --output runs/hypernet_centroid_residual_shuffled_20e/evaluation`
- Verification:
  - Full direct hypernet checks: `55` direct tests passed.
  - `python -m py_compile` passed on touched hypernet model/training/evaluation/test
    files.
  - No linting was run, per project instruction.
- Run IDs / artifact paths:
  - `model_zoo/configs/hypernet/centroid_residual_shuffled_20e.yaml`
  - `runs/hypernet_centroid_residual_shuffled_20e/model.pt`
  - `runs/hypernet_centroid_residual_shuffled_20e/evaluation/results.json`
- Training metrics:
  - Shuffled contrastive loss:
    - epoch 10: `0.0905`;
    - epoch 20: `0.0656`.
  - Final reconstruction cosine: `-0.0227`.
  - Lightweight train-script editing success: `0/2` pairs.
- Formal metrics:
  - Clean proof gate:
    - `passed`: `False`.
  - Interpret:
    - Focused heldout raw signature random-forest accuracy: `74.4%`.
    - Focused heldout majority baseline: `32.1%`.
  - Decode:
    - Matched generated-heldout calibrated accuracy: `94.7%`.
    - Matched raw behavior accuracy: `98.7%`.
    - Train-centroid control calibrated accuracy: `0.0%`.
    - Null control calibrated accuracy: `32.1%`.
    - Condition-ablation control calibrated accuracy: `32.1%`.
    - Noise-signature control calibrated accuracy: `91.2%`.
    - Shuffled-signature control calibrated accuracy: `65.9%`.
    - Opposite-direction shuffled-signature control calibrated accuracy: `100.0%`.
    - Noise all-target failures include:
      - `has_majority -> has_majority`: `94.5%`;
      - `sorted_ascending -> sorted_ascending`: `100.0%`;
      - `sorted_descending -> sorted_descending`: `100.0%`;
      - `sorted_descending -> has_majority`: `94.8%`;
      - `mountain_pattern -> sorted_ascending`: `100.0%`;
      - `mountain_pattern -> mountain_pattern`: `73.1%`.
  - Subject-functional specificity:
    - Matched MSE: `25.03`.
    - Best-control MSE: `89.47`.
    - Mean improvement vs best control: `+64.44`.
    - Win rate vs best control: `80.95%`.
    - Per-behavior improvements all positive.
  - Steering:
    - Generated-heldout target success: `66.8%`.
    - Generated-heldout no-edit target success: `15.0%`.
    - Cross-direction no-edit success: `0.0%`.
    - Collapsed/cross-direction success: `97.6%`.
    - Remaining blocking failures:
      - `has_majority` target success only `27.3%` with negative margin delta;
      - `mountain_pattern` target success `44.7%`.
- Decision:
  - Correctly demoted. The shuffled-negative objective improved subject-output
    specificity and some aggregate steering metrics, but did not establish
    fixed-signature-specific behavior decoding.
  - Failure should be described as a behavior-decode specificity failure, not a total
    subject-specificity failure.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `5/5` on demotion.
  - Key methodological concern:
    - the new shuffled contrastive loss targets source-label margin for one
      deterministic different-label control; it does not directly suppress
      noise/opposite-direction/all-target class priors, which remain fatal.
- Next action options:
  - Write this up as negative evidence for the broad behavior-decode proof claim; or
  - Redesign around stricter paired/contrastive data where matched signatures must
    beat label-matched, opposite-direction, noise, and ablation controls on both
    behavior margins and subject-output MSE as first-class train and gate targets.

### 2026-06-10 - Centroid-Residual Expanded-Control 20 Epoch Run

- Objective: close the train/evaluation mismatch where noise and different-label
  shuffled controls were evaluator failures but were not first-class all-target
  anti-behavior controls during training.
- Implementation:
  - Added config weights:
    - `noise_control_weight`;
    - `shuffled_control_weight`.
  - Added `build_behavior_control_conditions` to unify specificity and anti-behavior
    controls.
  - Added `weight_control_penalty` so `null`, `train_centroid`, `condition_ablation`,
    `noise`, and `different_label` controls can all receive all-target and
    hard-negative penalties.
  - Train/validation all-target and hard-negative control loops now iterate over all
    behavior controls, including `noise` and `different_label`.
- Commands:
  - `python model_zoo/hypernet/train.py --config model_zoo/configs/hypernet/centroid_residual_expanded_controls_20e.yaml --run-dir runs/hypernet_centroid_residual_expanded_controls_20e --no-tensorboard`
  - `python model_zoo/hypernet/evaluation/pipeline.py --model runs/hypernet_centroid_residual_expanded_controls_20e/model.pt --output runs/hypernet_centroid_residual_expanded_controls_20e/evaluation`
- Verification:
  - Full direct hypernet checks: `56` direct tests passed.
  - `python -m py_compile` passed on touched hypernet model/training/evaluation/test
    files.
  - No linting was run, per project instruction.
- Run IDs / artifact paths:
  - `model_zoo/configs/hypernet/centroid_residual_expanded_controls_20e.yaml`
  - `runs/hypernet_centroid_residual_expanded_controls_20e/model.pt`
  - `runs/hypernet_centroid_residual_expanded_controls_20e/evaluation/results.json`
- Training metrics:
  - Final reconstruction cosine: `-0.0449`.
  - Lightweight train-script editing success: `1/2` pairs.
  - Epoch 20 metrics:
    - shuffled contrastive loss: `0.1018`;
    - specificity loss: `14.9264`;
    - control penalty: `0.7438`;
    - hard control penalty: `1.2692`.
- Formal metrics:
  - Clean proof gate:
    - `passed`: `False`.
  - Interpret:
    - Focused heldout raw signature random-forest accuracy: `75.7%`.
    - Focused heldout majority baseline: `31.1%`.
  - Decode:
    - Matched generated-heldout calibrated accuracy: `82.2%`.
    - Matched raw behavior accuracy: `99.7%`.
    - Per-behavior matched calibrated accuracy:
      - `has_majority`: `98.4%`;
      - `mountain_pattern`: `78.9%`;
      - `sorted_ascending`: `50.5%`;
      - `sorted_descending`: `95.5%`.
    - Shuffled-signature control calibrated accuracy: `57.6%`.
    - Opposite-direction shuffled-signature control calibrated accuracy: `55.0%`.
    - Noise-signature control calibrated accuracy: `59.1%`.
    - Null control calibrated accuracy: `22.8%`.
    - Condition-ablation aggregate calibrated accuracy: `0.0%`.
    - Train-centroid control calibrated accuracy: `0.0%`.
    - All-target control failures remain, including:
      - null `sorted_ascending -> sorted_ascending`: `100.0%`;
      - null `has_majority -> sorted_ascending`: `100.0%`;
      - noise `has_majority -> sorted_ascending`: `100.0%`;
      - noise `mountain_pattern -> mountain_pattern`: `92.6%`;
      - condition-ablation `has_majority -> mountain_pattern`: `100.0%`;
      - condition-ablation `has_majority -> sorted_ascending`: `100.0%`.
  - Subject-functional specificity:
    - Matched MSE: `33.82`.
    - Best-control MSE: `70.23`.
    - Mean improvement vs best control: `+36.41`.
    - Win rate vs best control: `81.7%`.
    - Per-behavior improvements all positive.
  - Steering:
    - Generated-heldout target success: `60.9%`.
    - Generated-heldout no-edit target success: `15.7%`.
    - Collapsed/cross-direction success: `83.9%`.
    - `has_majority` target success: `8.0%`.
    - `has_majority` mean margin delta: `-0.0308`.
- Decision:
  - Correctly demoted. The expanded penalties materially reduced some controls:
    - noise: `91.2% -> 59.1%`;
    - opposite-direction shuffled: `100.0% -> 55.0%`;
    - shuffled: `65.9% -> 57.6%`.
  - However, matched decode also degraded:
    - `94.7% -> 82.2%`;
    - `sorted_ascending` calibrated decode only `50.5%`.
  - This supports a tradeoff interpretation: stronger control suppression in this
    architecture degrades matched behavior decode rather than producing clean
    fixed-signature-specific behavior decoding.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `5/5` on demotion.
  - Methodological note:
    - aggregate control accuracy is misleading; all-target matrices are the correct
      evidence surface, because condition-ablation aggregate is `0.0%` while several
      all-target off-target cells remain `100.0%`.
- Next action:
  - Stop scaling penalties in this architecture.
  - Write up negative evidence for the broad behavior-decode proof claim, unless
    restarting with a paired-contrast dataset/task design where controls are baked in
    from data generation onward.

### 2026-06-10 - Paired-Contrast Proof Infrastructure

- Objective: begin the proof-grade dataset redesign recommended after repeated
  behavior-decode specificity failures.
- Design artifact:
  - `docs/superpowers/specs/2026-06-10-paired-contrast-proof-dataset-design.md`.
  - Kepler reviewed the design direction with confidence `4/5`.
  - Required additions were patched into the design:
    - registered decode policy;
    - transitive split rules;
    - control member provenance;
    - stored probe set, not only hashes;
    - behavior-by-control sample-count thresholds;
    - separate validation/test split for final proof.
- Implementation:
  - Added `model_zoo/hypernet/paired_contrast.py`.
  - New helpers:
    - `validate_registered_decode_policy`;
    - `build_probe_provenance`;
    - `group_subject_ids`;
    - `validate_transitive_group_splits`;
    - `summarize_behavior_control_counts`;
    - `require_behavior_control_counts`.
  - Added tests for:
    - content-addressed probe provenance;
    - transitive split overlap across matched subjects and control members;
    - behavior-by-control count reporting;
    - behavior-by-control minimum count gate;
    - registered decode policy validation.
- Follow-up implementation after Kepler review:
  - Added `validate_paired_group_schema`.
  - Schema validation now checks:
    - `group_id` presence and uniqueness;
    - `subject.subject_id` presence;
    - required controls per row;
    - same-label control target-pattern consistency;
    - same-direction/different-label control semantics;
    - opposite-direction control semantics;
    - centroid member IDs;
    - noise control seed.
  - Added tests for malformed rows, duplicate group IDs, and invalid control
    semantics.
- Verification:
  - Full direct hypernet checks: `64` direct tests passed after schema additions.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `4/5` on paired-contrast infrastructure after schema additions.
  - Blocking issues:
    - same-label controls could omit `target_pattern`;
    - unknown control-type keys could pass validation and later become proof cells;
    - centroid provenance needed explicit split semantics so train centroids do not
      become validation/test subject leaks;
    - invariant helpers needed a single artifact-level validator to prevent accidental
      bypass.
- Follow-up implementation after Kepler review:
  - Same-label controls now require explicit `target_pattern == group.target_pattern`.
  - Schema validation rejects unknown control types.
  - Centroid controls can use a row-local `member_subject_ids` list or a split-explicit
    artifact reference using `centroid_id`, `member_split`, and
    `member_subject_ids_hash`.
  - Transitive split validation ignores centroid artifact references and still tracks
    raw row-local centroid member IDs.
  - Added `validate_paired_contrast_artifact` as a fail-closed entry point for:
    - registered decode policy;
    - stored probe provenance fields;
    - paired group schema;
    - transitive split validation;
    - behavior-by-control minimum count gates over selected proof splits.
  - Added tests for explicit same-label target patterns, unknown controls,
    train-centroid artifact references, and artifact-level pass/fail behavior.
- Follow-up verification:
  - New paired-contrast validator tests passed.
  - Full direct hypernet checks: `69` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Second reviewer pass:
  - Reviewer: Kepler.
  - Confidence: `4/5` on revised validator layer.
  - Blocking issues:
    - artifact-level provenance required only a subset of the leak-audit fields;
    - validation/test behavior-control counts were pooled, so validation rows could
      mask an empty or underpowered final test split.
- Second follow-up implementation:
  - `validate_paired_contrast_artifact` now requires the full probe provenance audit
    field set:
    - `probe_set_id`;
    - `probe_examples`;
    - `probe_examples_hash`;
    - `behavior_suite_hash`;
    - `probe_generation_config_hash`;
    - `extractor_config_hash`;
    - `extractor_code_hash`;
    - `normalization_stats_hash`;
    - `dataset_source_hash`;
    - `git_commit`.
  - Behavior-by-control count gates now run independently for each configured proof
    split, with per-split pass/fail details.
  - Added tests proving incomplete provenance fails and validation rows cannot rescue
    an underpowered test split.
- Second follow-up verification:
  - New artifact rigor tests passed.
  - Full paired-contrast validator test group passed.
  - Full direct hypernet checks: `71` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Third reviewer pass:
  - Reviewer: Kepler.
  - Confidence: `4/5` on second validator revision.
  - Blocking issue:
    - provenance validation checked field presence but still allowed non-proof values:
      `git_commit: null` and hashes derived from empty extractor code,
      normalization stats, or dataset source defaults.
- Third follow-up implementation:
  - Artifact provenance validation now rejects:
    - empty or non-string `git_commit`;
    - `extractor_code_hash` equal to the empty extractor-code hash;
    - `normalization_stats_hash` equal to the empty mapping hash;
    - `dataset_source_hash` equal to the empty mapping hash.
  - Added a regression test proving default empty provenance fails proof-mode artifact
    validation.
- Third follow-up verification:
  - Empty provenance-value test passed.
  - Full paired-contrast validator test group passed.
  - Full direct hypernet checks: `72` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Fourth reviewer pass:
  - Reviewer: Kepler.
  - Confidence: `4/5` on third validator revision.
  - Blocking issues:
    - `probe_examples_hash` was required but not recomputed against stored
      `probe_examples`;
    - an empty stored probe set could pass if its hash matched.
- Fourth follow-up implementation:
  - Artifact provenance validation now requires `probe_examples` to be a non-empty
    sequence.
  - Artifact provenance validation recomputes
    `stable_hash_json(list(probe_examples))` and rejects mismatched
    `probe_examples_hash`.
  - Added tests for stale/fabricated probe-example hashes and empty stored probe sets.
- Fourth follow-up verification:
  - Probe provenance hash/non-empty tests passed.
  - Full paired-contrast validator test group passed.
  - Full direct hypernet checks: `74` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Fifth reviewer pass:
  - Reviewer: Kepler.
  - Confidence: `4/5` on fourth validator revision.
  - Blocking issue:
    - `different_label_same_direction` accepted non-directional source/control pairs
      because both directions resolved to `None`.
- Fifth follow-up implementation:
  - `different_label_same_direction` now requires both source and control patterns to
    have registered non-null directions before checking equality.
  - Added a regression test proving `has_majority -> mountain_pattern` is rejected as
    an invalid same-direction control.
- Fifth follow-up verification:
  - Non-directional same-direction control test passed.
  - Full paired-contrast validator test group passed.
  - Full direct hypernet checks: `75` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Fifth reviewer re-check:
  - Reviewer: Kepler.
  - Confidence: `5/5` for this infrastructure checkpoint.
  - Blocking issues: none for the invariant layer.
  - Decision: `accepted`.
  - Scope caveat:
    - this does not support the hypothesis yet; it only accepts the invariant layer as
      ready for paired dataset/evaluator scaffolding.
  - Next action:
    - implement paired evaluator scaffolding wired to
      `validate_paired_contrast_artifact`, with proof outputs defined as
      matched-minus-control behavior margins and subject-output MSE deltas per split,
      behavior, and control type.
- Status:
  - This is not a result checkpoint and does not support the hypothesis yet.
  - It creates the invariant layer needed before implementing paired dataset
    generation and paired proof evaluation.

### 2026-06-10 - Paired-Contrast Evaluator Scaffold

- Objective: add the first evaluator scaffold recommended by the accepted invariant
  review, without claiming any experimental evidence yet.
- Implementation:
  - Added `evaluate_paired_contrast_predictions`.
  - The evaluator first calls `validate_paired_contrast_artifact` and fails before
    metrics if the artifact does not pass the proof gates.
  - Inputs are precomputed per-group predictions:
    - matched `behavior_margin`;
    - matched `subject_output_mse`;
    - per-control `behavior_margin`;
    - per-control `subject_output_mse`.
  - Outputs are grouped by proof split, behavior, and control type:
    - mean matched behavior margin;
    - mean control behavior margin;
    - mean matched-minus-control behavior margin;
    - mean matched subject-output MSE;
    - mean control subject-output MSE;
    - mean control-minus-matched subject-output MSE.
  - Missing group, matched, control, or numeric metric entries fail the scaffold.
- Tests:
  - evaluator fails invalid artifacts before computing metrics;
  - evaluator reports synthetic matched-minus-control behavior margins and
    control-minus-matched subject-output MSE deltas by split/behavior/control;
  - evaluator fails missing group/control predictions.
- Verification:
  - Paired evaluator scaffold tests passed.
  - Full direct hypernet checks: `78` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `4/5` on evaluator scaffold.
  - Blocking issues:
    - failed prediction evaluations still emitted normal-looking partial metrics;
    - non-finite numeric values such as `NaN` and `inf` could enter proof summaries.
- Follow-up implementation:
  - `evaluate_paired_contrast_predictions` now returns `metrics: {}` whenever any
    prediction-level failure is present.
  - `_metric_value` now rejects non-finite numeric values.
  - Metric validation continues through a group after a matched-side failure so
    control-side metric failures are also reported, while no incomplete rows enter
    summaries.
  - Added tests for partial-metric suppression and non-finite values.
- Follow-up verification:
  - Evaluator failure-mode tests passed.
  - Full paired evaluator tests passed.
  - Full direct hypernet checks: `79` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer re-check:
  - Reviewer: Kepler.
  - Confidence: `5/5` for this evaluator scaffold checkpoint.
  - Blocking issues: none for this scaffold.
  - Decision: `accepted`.
  - Next action:
    - proceed to paired dataset generator scaffolding wired to
      `validate_paired_contrast_artifact`;
    - keep the next evaluator increment focused on thresholded proof gates over the
      emitted matched-minus-control deltas.
- Status:
  - This is scaffolding only; it does not support the hypothesis until real paired
    artifacts and model predictions are generated and evaluated.

### 2026-06-10 - Paired-Contrast Generator Scaffold

- Objective: add a conservative paired dataset generator scaffold that emits artifacts
  already wired to the accepted proof validator.
- Implementation:
  - Added `build_paired_contrast_artifact_from_subjects`.
  - Inputs:
    - subject metadata by split;
    - registered decode policy;
    - proof probe provenance;
    - required behaviors and control types;
    - proof split count thresholds.
  - The generator builds disjoint same-split groups using available subject metadata:
    - `same_label_other_subject`;
    - `different_label_same_direction`;
    - `opposite_direction`;
    - `noise_signature`;
    - optional `same_label_centroid`, `null_signature`, and `condition_ablation`
      controls.
  - The generated artifact is immediately passed to
    `validate_paired_contrast_artifact`.
  - If required controls cannot be formed, the returned validation result fails rather
    than producing a proof-ready artifact.
- Tests:
  - generator builds a validator-ready artifact from synthetic validation/test
    subjects with same-label, same-direction, opposite-direction, and noise controls;
  - generator fails when required same-direction and opposite-direction controls are
    unavailable.
- Verification:
  - Paired generator scaffold tests passed.
  - Full direct hypernet checks: `81` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `4/5` on generator scaffold.
  - Blocking issues:
    - source-pool split leakage could pass if a duplicated input subject was not
      selected into emitted groups;
    - emitted matched subjects did not preserve immutable `weights` and `signature`
      references.
- Follow-up implementation:
  - Added source-pool preflight over all input subject IDs before generation:
    - duplicate subject IDs within an input split fail;
    - subject IDs crossing input splits fail even if unused;
    - subject rows with target labels must include a weights reference
      (`weights_hash` or `weights_uri`) and a signature reference
      (`signature_hash` or `signature_uri`).
  - Emitted matched and subject-control payloads now preserve subject ID, target
    pattern, and available weight/signature reference fields.
  - Generator return values now include combined `failures`, `preflight`, `artifact`,
    and artifact `validation`.
  - Added tests for unused cross-split source-pool leakage and missing matched
    weight/signature references.
- Follow-up verification:
  - Paired generator scaffold tests passed.
  - Full direct hypernet checks: `83` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Second reviewer pass:
  - Reviewer: Kepler.
  - Confidence: `4/5` on revised generator scaffold.
  - Blocking issues:
    - source-pool preflight failures were not embedded in the artifact, so a
      builder-failed artifact could pass standalone validation downstream;
    - `validate_paired_group_schema` did not require matched/control subject payloads
      to include weight and signature references.
- Second follow-up implementation:
  - Generated artifacts now include `source_pool_preflight`.
  - `validate_paired_contrast_artifact` now requires `source_pool_preflight` to be
    present and passed.
  - Failed preflight details are propagated into artifact validation failures.
  - `validate_paired_group_schema` now requires subject-bearing matched and control
    payloads to include a weights reference and a signature reference.
  - Added tests proving:
    - a builder-failed artifact cannot pass standalone artifact validation;
    - hand-authored matched/control subject payloads without refs fail schema
      validation.
- Second follow-up verification:
  - Artifact contract tests passed.
  - Paired generator/evaluator tests passed.
  - Full direct hypernet checks: `85` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer re-check:
  - Reviewer: Kepler.
  - Confidence: `5/5` for this generator scaffold checkpoint.
  - Blocking issues: none for this scaffold.
  - Decision: `accepted`.
  - Next action:
    - add thresholded paired proof gates over matched-minus-control behavior margins and
      control-minus-matched subject-output MSE deltas, reported per proof split,
      behavior, and control type.
- Status:
  - This is artifact-generation scaffolding only; it has not generated a real dataset
    or evaluated model predictions.

### 2026-06-10 - Paired-Contrast Thresholded Proof Gates

- Objective: add thresholded proof gates over paired evaluator outputs so future
  results cannot be accepted on aggregate metrics alone.
- Implementation:
  - `evaluate_paired_contrast_predictions` now accepts optional `proof_thresholds`.
  - Registered thresholds:
    - `min_mean_matched_minus_control_behavior_margin`;
    - `min_mean_control_minus_matched_subject_output_mse`.
  - Gates are evaluated per proof split, behavior, and control type.
  - Gate failures set top-level `passed: False` while preserving complete metrics for
    diagnosis.
  - Prediction-level failures still suppress metrics to `{}`.
  - Threshold values must be numeric and finite.
  - Comparisons use a small epsilon to avoid false failures at exact floating-point
    boundaries.
- Tests:
  - thresholded proof gates pass when every synthetic proof cell clears both
    thresholds;
  - thresholded proof gates fail when valid paired metrics do not clear registered
    thresholds, while retaining complete diagnostic metrics.
- Verification:
  - Threshold proof-gate tests passed.
  - Paired generator/evaluator tests passed.
  - Full direct hypernet checks: `87` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `4/5` on threshold proof gates.
  - Blocking issues:
    - unknown `proof_thresholds` keys were silently ignored;
    - missing, nonnumeric, non-finite, and unknown threshold-key behavior needed
      explicit regression coverage.
- Follow-up implementation:
  - Added `REGISTERED_PROOF_THRESHOLDS`.
  - `_evaluate_paired_proof_gates` now rejects any threshold key outside the
    registered set.
  - Added tests for:
    - unknown threshold keys;
    - missing required thresholds;
    - nonnumeric thresholds;
    - non-finite thresholds.
- Follow-up verification:
  - Proof threshold validation tests passed.
  - Paired generator/evaluator tests passed.
  - Full direct hypernet checks: `89` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Reviewer re-check:
  - Reviewer: Kepler.
  - Confidence: `5/5` for this threshold proof-gate checkpoint.
  - Blocking issues: none for this scaffold.
  - Decision: `accepted`.
  - Next action:
    - generate the first real paired-contrast dataset artifact, run
      `validate_paired_contrast_artifact`, and review the dataset audit before any
      model training or hypothesis claims.
- Status:
  - This is evaluator-gate scaffolding only; no real paired artifact has been generated
    or evaluated yet.

### 2026-06-10 - First Real Paired-Contrast Artifact Audit

- Objective: generate the first paired-contrast artifact from real deduplicated HF
  subject rows and run the proof artifact validator before any model training or
  hypothesis claims.
- Source:
  - Checkpoint: `runs/hypernet_centroid_residual_expanded_controls_20e/model.pt`.
  - Dataset loader: `load_data(include_patterns=checkpoint["dataset_patterns"])`.
  - Dataset ID: `maximuspowers/hypernet_validated`.
  - Artifact output:
    - `runs/paired_contrast_first_artifact/paired_contrast_artifact.json`;
    - `runs/paired_contrast_first_artifact/validation.json`;
    - `runs/paired_contrast_first_artifact/summary.json`.
- Artifact construction:
  - Focused first artifact on directional behaviors:
    - `sorted_ascending`;
    - `sorted_descending`.
  - Required controls:
    - `same_label_other_subject`;
    - `opposite_direction`;
    - `noise_signature`.
  - Heldout checkpoint validation indices were split deterministically into separate
    validation/test proof splits.
  - Built balanced source pools with exactly `10` paired groups per behavior/control
    cell in each proof split.
  - Matched/control subject payloads include dedup index, source HF row index,
    `weights_hash`, and `signature_hash`.
- Validation result:
  - Overall `passed`: `False`.
  - Failure count: `1`.
  - Failure:
    - `probe_provenance probe_examples is empty`.
  - Passing subchecks:
    - schema: `True`;
    - split/source preflight: `True`;
    - per-split behavior/control counts: `True`.
  - Counts:
    - validation:
      - `sorted_ascending`: `10` same-label, `10` opposite, `10` noise;
      - `sorted_descending`: `10` same-label, `10` opposite, `10` noise.
    - test:
      - `sorted_ascending`: `10` same-label, `10` opposite, `10` noise;
      - `sorted_descending`: `10` same-label, `10` opposite, `10` noise.
- Decision:
  - Correctly blocked as a proof artifact.
  - The available HF rows contain fixed `improved_signature` columns, but the actual
    fixed probe examples used to extract those signatures are not embedded in the
    rows/checkpoint. Treating the signature column as proof-grade fixed-probe evidence
    without stored probes would be a provenance leak/misleading-result risk.
- Status:
  - This artifact can be used as a subject/control pairing audit, but not as a proof
    dataset until probe examples are recovered or signatures are regenerated with
    stored probe provenance.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `5/5` that the artifact is correctly blocked/demoted.
  - Blocking issues:
    - proof artifact is correctly blocked by
      `probe_provenance probe_examples is empty`;
    - stored signatures cannot be tied to embedded or recoverable probe examples.
  - Decision: `reject` as proof evidence.
  - Non-blocking audit result:
    - no additional pairing leak found;
    - schema, source-pool preflight, split isolation, immutable subject refs, and
      per-split behavior/control counts pass.
  - Next action:
    - use this only as a subject/control pairing audit;
    - regenerate signatures with stored probe examples and full provenance, or recover
      the exact original fixed probe set and embed it so `probe_examples_hash`
      validates against non-empty `probe_examples`.

### 2026-06-10 - Stored-Probe Regenerated Paired Artifact

- Objective: remove the fixed-probe provenance blocker by regenerating signatures from
  real subject weights using an explicitly stored deterministic probe set.
- Implementation:
  - Added stored-probe helpers in `model_zoo/hypernet/paired_contrast.py`:
    - `build_digit_probe_examples`;
    - `build_stored_probe_provenance`;
    - `extract_signature_with_stored_probes`.
  - Regenerated signatures from flat subject weights via `SubjectNetwork.from_weights`
    and deterministic stored probes.
  - Stored all probe examples directly in artifact provenance.
  - Saved regenerated signature values and hashes in a sidecar artifact.
- Tests:
  - deterministic stored digit probe generation;
  - deterministic signature extraction from flat weights;
  - stored-probe provenance passes the artifact validator contract.
- Verification:
  - Stored-probe regeneration tests passed.
  - Paired generator/evaluator tests passed.
  - Full direct hypernet checks: `92` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Artifact output:
  - `runs/paired_contrast_stored_probe_artifact/paired_contrast_artifact.json`;
  - `runs/paired_contrast_stored_probe_artifact/validation.json`;
  - `runs/paired_contrast_stored_probe_artifact/summary.json`;
  - `runs/paired_contrast_stored_probe_artifact/regenerated_signatures.json`.
- Source:
  - Checkpoint: `runs/hypernet_centroid_residual_expanded_controls_20e/model.pt`.
  - Dataset loader: `load_data(include_patterns=checkpoint["dataset_patterns"])`.
  - Dataset ID: `maximuspowers/hypernet_validated`.
- Stored probe set:
  - seed: `20260610`;
  - examples: `256`;
  - sequence length: `5`;
  - base: `10`;
  - all probe examples stored in provenance and hashed.
- Artifact construction:
  - Behaviors:
    - `sorted_ascending`;
    - `sorted_descending`.
  - Required controls:
    - `same_label_other_subject`;
    - `opposite_direction`;
    - `noise_signature`.
  - Balanced proof splits:
    - validation: `10` groups per behavior/control cell;
    - test: `10` groups per behavior/control cell.
- Validation result:
  - Overall `passed`: `True`.
  - Failure count: `0`.
  - Passing subchecks:
    - probe provenance: `True`;
    - schema: `True`;
    - split/source preflight: `True`;
    - per-split behavior/control counts: `True`.
  - Regenerated signature dimension: `560`.
- Caveat:
  - This artifact fixes the probe provenance problem, but it is not yet model evidence.
  - The regenerated stored-probe signature dimension is `560`, while the prior
    hypernet checkpoints used `510`-dimensional HF signatures, so existing checkpoints
    are not directly compatible with this artifact.
  - Next proof step must train/evaluate a compatible model or add an adapter with
    registered policy before making hypothesis claims.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `4/5` on first stored-probe regenerated artifact.
  - Blocking issues:
    - artifact validation passed, but regenerated sidecar values were not yet
      cryptographically bound to artifact `signature_hash` refs with a recorded and
      validated signature-hash algorithm;
    - stale or mismatched `regenerated_signatures.json` could sit beside a passing
      artifact.
- Follow-up implementation:
  - Added `REGISTERED_SIGNATURE_HASH_ALGORITHMS` with
    `stable_hash_json_float_list_v1`.
  - Added `signature_hash_stable_float_list`.
  - Added `audit_regenerated_signature_sidecar`.
  - Sidecar audit validates every subject-bearing artifact payload against the stored
    regenerated signature vector for that subject.
  - Added tests proving sidecar refs bind correctly and mismatches fail.
- Follow-up verification:
  - Sidecar audit tests passed.
  - Stored-probe and paired tests passed.
  - Full direct hypernet checks: `94` direct tests passed.
  - `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py` passed.
  - No linting was run, per project instruction.
- Regenerated artifact after sidecar-audit patch:
  - `runs/paired_contrast_stored_probe_artifact/paired_contrast_artifact.json`;
  - `runs/paired_contrast_stored_probe_artifact/validation.json`;
  - `runs/paired_contrast_stored_probe_artifact/summary.json`;
  - `runs/paired_contrast_stored_probe_artifact/regenerated_signatures.json`;
  - `runs/paired_contrast_stored_probe_artifact/sidecar_audit.json`.
  - Artifact validation passed: `True`.
  - Sidecar audit passed: `True`.
  - Sidecar checked subject-bearing signature refs: `120`.
  - Failure count: `0`.
- Reviewer re-check:
  - Reviewer: Kepler.
  - Confidence: `5/5` for dataset-infrastructure acceptance.
  - Blocking issues: none for dataset infrastructure.
  - Decision: `accepted`.
  - Accepted scope:
    - proof-grade paired dataset infrastructure only;
    - no model-evidence or hypothesis claim yet.
  - Next action:
    - train/evaluate a compatible model on the `560`-dimensional stored-probe
      signatures, or define and review a registered adapter policy before any
      model-evidence or hypothesis claims.

### 2026-06-10 - Stored-Probe Interpreter Evidence V1

- Objective: test the first component of the hypothesis under auditable stored-probe
  provenance: whether fixed-probe activation signatures contain behavior information.
- Scope:
  - Interpretability evidence only.
  - Does not test steering.
  - Does not test decoding or functional generated models.
- Source:
  - Checkpoint split: `runs/hypernet_centroid_residual_expanded_controls_20e/model.pt`.
  - Dataset loader: `load_data(include_patterns=checkpoint["dataset_patterns"])`.
  - Stored probes:
    - seed: `20260610`;
    - examples: `256`;
    - sequence length: `5`;
    - base: `10`.
  - Regenerated signatures:
    - dimension: `560`;
    - extracted from flat subject weights via stored probes.
  - Output:
    - `runs/stored_probe_interpret_v1/results.json`;
    - `runs/stored_probe_interpret_v1/stored_probe_signatures.pt`.
- Data audit:
  - Dataset provenance comparison against checkpoint: `matches=True`;
  - mismatched fields: `[]`.
  - Train samples: `3592`;
  - heldout samples: `399`.
  - Heldout class counts:
    - `sorted_descending`: `89`;
    - `sorted_ascending`: `91`;
    - `mountain_pattern`: `95`;
    - `has_majority`: `124`.
- Metrics:
  - Majority baseline:
    - accuracy: `31.08%`;
    - balanced accuracy: `25.00%`.
  - Logistic regression:
    - accuracy: `92.23%`;
    - balanced accuracy: `92.27%`;
    - delta accuracy vs majority: `+61.15` points;
    - shuffled-train-label control accuracy: `22.56%`;
    - shuffled-train-label control balanced accuracy: `21.97%`.
    - per-behavior recall:
      - `sorted_descending`: `91.01%`;
      - `sorted_ascending`: `95.60%`;
      - `mountain_pattern`: `90.53%`;
      - `has_majority`: `91.94%`.
  - Random forest:
    - accuracy: `97.49%`;
    - balanced accuracy: `97.28%`;
    - delta accuracy vs majority: `+66.42` points;
    - shuffled-train-label control accuracy: `28.32%`;
    - shuffled-train-label control balanced accuracy: `23.47%`.
    - per-behavior recall:
      - `sorted_descending`: `100.00%`;
      - `sorted_ascending`: `91.21%`;
      - `mountain_pattern`: `97.89%`;
      - `has_majority`: `100.00%`.
- Interpretation:
  - Positive evidence for the interpretability subclaim under regenerated
    stored-probe signatures.
  - Not evidence for representation steering or functional decoding.
  - Must not be used to claim the full hypothesis is proven.
- Reviewer:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the stated interpretability-only claim.
  - Blocking issues: none within stated scope.
  - Decision: `accepted`.
  - Accepted claim:
    - regenerated `560`-dimensional stored-probe activation signatures contain strong
      behavior-label information on the checkpoint heldout split.
  - Explicitly not accepted:
    - steering evidence;
    - decoding evidence;
    - full hypothesis proof.
  - Next action:
    - run a compatible `560`-dimensional stored-probe paired model/evaluator before
      making decode, steer, or full-hypothesis claims.

### 2026-06-10 - Stored-Probe 1-NN Decode Diagnostic V1

- Objective: run the first compatible paired evaluator pass on `560`-dimensional
  stored-probe signatures without reusing incompatible `510`-dimensional checkpoints.
- Scope:
  - Diagnostic decode-style baseline only.
  - Nonparametric `1`-nearest-neighbor retrieval from stored-probe signatures to
    train-set weights.
  - Not a learned generator.
  - No steering operation tested.
  - Does not prove the full hypothesis.
- Output:
  - `runs/stored_probe_knn_decode_v1/results.json`;
  - `runs/stored_probe_knn_decode_v1/predictions.json`.
- Dataset/artifact:
  - Paired artifact:
    `runs/paired_contrast_stored_probe_artifact/paired_contrast_artifact.json`.
  - Train retrieval pool: checkpoint train split, `3592` subjects.
  - Proof splits:
    - validation;
    - test.
  - Evaluated paired comparisons: `120`.
- Decoder/evaluator:
  - Fit `StandardScaler` and `NearestNeighbors(n_neighbors=1)` on train stored-probe
    signatures.
  - For each matched/control signature, retrieved nearest train weights.
  - Behavior margin evaluated on clean heldout behavior cases for the source target.
  - Subject-output MSE compared decoded network vs source network on the stored probe
    examples.
- Leakage audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Proof subjects from the paired artifact: validation `60`, test `60`.
  - Proof-overlap-with-train count: `0`.
  - Proof-overlap-with-checkpoint-validation count: `120`.
  - Unique retrieved train dedup indices: `139`.
  - Retrieved indices not in checkpoint train: `0`.
  - Retrieved indices overlapping proof subjects: `0`.
  - Note: `decoded_train_index` in `predictions.json` is a checkpoint train dedup
    index, not a zero-based neighbor-array position.
- Aggregate metrics:
  - mean matched behavior margin: `+0.1949`;
  - mean control behavior margin: `-0.0966`;
  - mean matched-minus-control behavior margin: `+0.2915`;
  - mean matched subject-output MSE: `0.1011`;
  - mean control subject-output MSE: `0.2101`;
  - mean control-minus-matched subject-output MSE: `+0.1090`.
- Key per-cell notes:
  - Validation, `sorted_ascending`:
    - same-label behavior delta: `+0.1770`;
    - opposite behavior delta: `+0.4370`;
    - noise behavior delta: `+0.3820`;
    - same-label MSE delta: `+0.0589`.
  - Validation, `sorted_descending`:
    - same-label behavior delta: `+0.0169`;
    - same-label MSE delta: `-0.0120` (matched worse than same-label control);
    - opposite behavior delta: `+0.3401`;
    - noise behavior delta: `+0.6369`.
  - Test, `sorted_ascending`:
    - same-label behavior delta: `+0.1206`;
    - same-label MSE delta: `+0.0401`;
    - opposite behavior delta: `+0.4396`;
    - noise behavior delta: `+0.4438`.
  - Test, `sorted_descending`:
    - same-label behavior delta: `-0.0923` (same-label control has higher target
      behavior margin);
    - same-label MSE delta: `+0.0177`;
    - opposite behavior delta: `+0.1881`;
    - noise behavior delta: `+0.4085`.
- Interpretation:
  - Positive diagnostic evidence that stored-probe signatures support behavior-aware
    retrieval against opposite/noise controls.
  - Same-label subject-specificity is weak/mixed, especially for `sorted_descending`.
  - This should not be treated as proof of functional decoding, because the decoder is
    nearest-neighbor retrieval rather than generated weights and no thresholded proof
    gate was applied.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the stated diagnostic/mixed characterization after the
    leakage audit.
  - Blocking issues: none within stated scope.
  - Required next step before stronger decode claims: add thresholded paired proof
    gates, including control-specific treatment for same-label controls.

### 2026-06-10 - Control-Specific Paired Decode Proof Gates

- Objective: make proof thresholds match the semantics of each control type instead
  of applying one global behavior-margin gate to all controls.
- Code changes:
  - Added explicit `by_control_type` proof-threshold support in
    `model_zoo/hypernet/paired_contrast.py`.
  - The legacy flat threshold schema remains supported.
  - Control-specific thresholds fail closed when:
    - a required control type lacks thresholds;
    - an unused control type is configured;
    - a threshold key is unregistered;
    - a threshold value is missing, nonnumeric, or non-finite.
  - Same-label controls can now require subject-output specificity without requiring
    behavior-margin separation, because same-label behavior margins are not the right
    contrast surface.
- Verification:
  - Red check: the new control-specific gate test failed before implementation because
    `by_control_type` was unsupported.
  - Targeted direct tests passed:
    - `test_paired_contrast_evaluator_allows_control_specific_proof_gates`;
    - `test_paired_contrast_evaluator_requires_thresholds_for_each_required_control_type`;
    - `test_paired_contrast_evaluator_rejects_unknown_control_specific_threshold_keys`.
  - Broader direct paired-contrast regression sweep passed: `23` zero-argument
    `test_paired_contrast_*` functions.
  - Compile check passed:
    `python -m py_compile model_zoo/hypernet/paired_contrast.py model_zoo/hypernet/tests/test_functional_hypernetwork.py`.
- Applied gate to the stored-probe `1`-NN diagnostic:
  - Output:
    `runs/stored_probe_knn_decode_v1/control_specific_gated_results.json`.
  - Threshold policy: `control_specific_decode_proof_gate_v1`.
  - Thresholds:
    - `same_label_other_subject`:
      - minimum mean control-minus-matched subject-output MSE: `0.02`;
    - `opposite_direction`:
      - minimum mean matched-minus-control behavior margin: `0.20`;
      - minimum mean control-minus-matched subject-output MSE: `0.05`;
    - `noise_signature`:
      - minimum mean matched-minus-control behavior margin: `0.20`;
      - minimum mean control-minus-matched subject-output MSE: `0.05`.
  - Result: failed proof gates.
  - Failures:
    - validation `sorted_descending` / `same_label_other_subject` subject-output MSE
      delta: `-0.0120 < 0.02`;
    - test `sorted_descending` / `same_label_other_subject` subject-output MSE delta:
      `0.0177 < 0.02`;
    - test `sorted_descending` / `opposite_direction` subject-output MSE delta:
      `0.0369 < 0.05`;
    - test `sorted_descending` / `opposite_direction` behavior-margin delta:
      `0.1881 < 0.20`.
- Interpretation:
  - The `1`-NN stored-probe diagnostic remains useful as a baseline, but it fails the
    explicit decode proof gate.
  - Stronger decode claims require a learned/generative method that clears these
    preregistered control-specific gates, especially for `sorted_descending`.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the control-specific gate design and non-misleading
    demotion of the `1`-NN diagnostic as failed decode proof.
  - Blocking issues: none before the next result step.
  - Next experiment/control recommendation:
    - use these gates as preregistered proof criteria for a learned/generative decoder;
    - keep `1`-NN retrieval as the baseline to beat;
    - prioritize `sorted_descending` same-label subject specificity.

### 2026-06-10 - Stored-Probe MLP Weight Decoder V1

- Objective: test a learned/generative decoder baseline from `560`-dimensional
  stored-probe signatures to flat subject weights, evaluated with the preregistered
  control-specific paired proof gates.
- Output:
  - `runs/stored_probe_mlp_decoder_v1/model.pt`;
  - `runs/stored_probe_mlp_decoder_v1/predictions.json`;
  - `runs/stored_probe_mlp_decoder_v1/results.json`.
- Method:
  - Reloaded the deduplicated `3991 x 345` weight tensor with the existing
    `hypernet.train.load_data` path.
  - Compared reloaded dataset provenance against
    `runs/hypernet_centroid_residual_expanded_controls_20e/model.pt`; no compared
    fields mismatched.
  - Used stored-probe signatures from
    `runs/stored_probe_interpret_v1/stored_probe_signatures.pt`.
  - Trained an MLP decoder on checkpoint train indices only.
  - Internal early-stop split was carved only from checkpoint train indices:
    - train: `3233`;
    - internal validation: `359`.
  - Best internal normalized validation MSE: `0.9148`.
  - Early stopped at epoch `40`.
  - Proof evaluation used only the paired validation/test artifact.
  - Noise-signature control policy:
    - `raw_signature = train_signature_mean + train_signature_std * standard_normal`;
    - seed: `20260610`;
    - one draw per proof group in artifact iteration order.
- Leakage audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Proof subject/control references: `120`.
  - Proof-overlap-with-train count: `0`.
  - Proof-overlap-with-checkpoint-validation count: `120`.
- Aggregate paired metrics:
  - mean matched behavior margin: `+0.0496`;
  - mean control behavior margin: `+0.0008`;
  - mean matched-minus-control behavior margin: `+0.0488`;
  - mean matched subject-output MSE: `268.5368`;
  - mean control subject-output MSE: `283.7776`;
  - mean control-minus-matched subject-output MSE: `+15.2407`.
- Split aggregate metrics:
  - Validation behavior delta: `+0.0431`; subject-output MSE delta: `+13.0874`.
  - Test behavior delta: `+0.0545`; subject-output MSE delta: `+17.3941`.
- Proof-gate result: failed.
- Gate failures:
  - validation `sorted_ascending` / `opposite_direction` behavior-margin delta:
    `0.0900 < 0.20`;
  - validation `sorted_ascending` / `noise_signature` behavior-margin delta:
    `0.0696 < 0.20`;
  - validation `sorted_descending` / `same_label_other_subject` subject-output MSE
    delta: `-1.9984 < 0.02`;
  - validation `sorted_descending` / `opposite_direction` subject-output MSE delta:
    `-7.5325 < 0.05`;
  - validation `sorted_descending` / `opposite_direction` behavior-margin delta:
    `0.0696 < 0.20`;
  - validation `sorted_descending` / `noise_signature` behavior-margin delta:
    `0.0215 < 0.20`;
  - test `sorted_ascending` / `opposite_direction` behavior-margin delta:
    `0.1206 < 0.20`;
  - test `sorted_ascending` / `noise_signature` behavior-margin delta:
    `0.1106 < 0.20`;
  - test `sorted_descending` / `opposite_direction` behavior-margin delta:
    `0.0604 < 0.20`;
  - test `sorted_descending` / `noise_signature` behavior-margin delta:
    `-0.0005 < 0.20`.
- Interpretation:
  - This learned MLP weight decoder does not support a decode claim.
  - It is worse than the `1`-NN retrieval baseline on behavior-margin separation and
    has very high subject-output MSE, indicating poor functional reconstruction.
  - Current evidence still supports only the interpretability part of the hypothesis;
    decode remains unproven.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the failed-decoder characterization and split/leakage
    bookkeeping.
  - Blocking issues: none before the next result step.
  - Minor wording fix applied: the `120` count is proof subject/control references,
    while the paired artifact has `40` proof groups.
  - Next experiment/control recommendation:
    - evaluate train-mean or label-centroid weight decoder baselines under the same
      paired gates;
    - require future learned decoders to beat `1`-NN and mean/centroid baselines per
      split, behavior, and control type;
    - optimize future learned decoders with proof-aligned functional/paired losses
      rather than plain weight MSE.

### 2026-06-10 - Mean and Label-Centroid Decode Baselines V1

- Objective: quantify central-tendency baselines under the same paired proof gates so
  future decoders must beat trivial train-set averages.
- Output:
  - `runs/stored_probe_centroid_decoder_baselines_v1/results.json`;
  - `runs/stored_probe_centroid_decoder_baselines_v1/train_mean_weight_decoder_predictions.json`;
  - `runs/stored_probe_centroid_decoder_baselines_v1/train_label_centroid_weight_decoder_predictions.json`.
- Baselines:
  - `train_mean_weight_decoder`: always emits the checkpoint-train mean weight vector.
  - `train_label_centroid_weight_decoder`: emits the checkpoint-train mean weight
    vector for the matched/control subject label; `noise_signature` emits the global
    train mean.
- Dataset/provenance:
  - Reloaded the same `3991 x 345` deduplicated weight tensor through
    `hypernet.train.load_data`.
  - Compared reloaded dataset provenance with
    `runs/hypernet_centroid_residual_expanded_controls_20e/model.pt`; no compared
    fields mismatched.
- Leakage audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Proof subject/control references: `120`.
  - Proof-overlap-with-train count: `0`.
  - Proof-overlap-with-checkpoint-validation count: `120`.
- `train_mean_weight_decoder` result:
  - Proof gates: failed.
  - Failure count: `20`.
  - mean matched-minus-control behavior margin: `0.0000`;
  - mean control-minus-matched subject-output MSE: `0.0000`;
  - mean matched subject-output MSE: `293.5306`.
  - Interpretation: pure average decoder provides no paired separation and no
    specificity.
- `train_label_centroid_weight_decoder` result:
  - Proof gates: failed.
  - Failure count: `16`.
  - mean matched-minus-control behavior margin: `0.0000` (`8.8e-06`);
  - mean control-minus-matched subject-output MSE: `+0.0830`;
  - mean matched subject-output MSE: `293.6079`.
  - Same-label subject-output MSE delta is `0.0` by construction and fails in both
    splits/behaviors.
  - Interpretation: label centroids do not produce behavior-functional decoded models
    under the clean heldout behavior cases.
- Comparison:
  - The MLP decoder is only modestly above centroid/mean baselines on behavior-margin
    separation and remains far below the `1`-NN retrieval diagnostic.
  - This strengthens the conclusion that current decode evidence is absent; the
    interpretable signatures alone are not enough for naive weight decoding.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the mean/centroid baseline characterization.
  - Blocking issues: none before the next result step.
  - Next experiment/control recommendation:
    - move away from plain weight-MSE decoding;
    - train with proof-aligned functional losses over behavior-margin separation and
      subject-output reconstruction;
    - require future learned decoders to beat `1`-NN per split/behavior/control.

### 2026-06-10 - Stored-Probe Functional Decoder V1

- Objective: train a learned decoder with proof-aligned losses instead of plain
  normalized weight MSE.
- Output:
  - `runs/stored_probe_functional_decoder_v1/model.pt`;
  - `runs/stored_probe_functional_decoder_v1/predictions.json`;
  - `runs/stored_probe_functional_decoder_v1/results.json`.
- Method:
  - MLP maps `560`-dimensional stored-probe signatures to `345` flat subject weights.
  - Trained only on checkpoint train indices.
  - Internal early-stop split was carved only from checkpoint train:
    - train: `3233`;
    - internal validation: `359`.
  - Loss terms:
    - normalized weight MSE weight: `0.05`;
    - raw stored-probe output MSE weight: `0.01`;
    - behavior-margin hinge weight: `2.0`;
    - behavior-margin hinge target: `0.20`;
    - probe subset per batch: `64` of `256`.
  - Best internal validation loss: `0.2151`.
  - Best internal validation checkpoint epoch: `72`.
  - Trained through epoch `80`.
  - The saved/evaluated model uses the best internal-validation checkpoint, not the
    final epoch checkpoint.
- Leakage/provenance audit:
  - Reloaded dataset provenance matched the checkpoint on compared fields.
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Proof subject/control references: `120`.
  - Proof-overlap-with-train count: `0`.
  - Proof-overlap-with-checkpoint-validation count: `120`.
- Aggregate paired metrics:
  - mean matched behavior margin: `+0.4048`;
  - mean control behavior margin: `+0.0515`;
  - mean matched-minus-control behavior margin: `+0.3533`;
  - mean matched subject-output MSE: `18.6459`;
  - mean control subject-output MSE: `230.8808`;
  - mean control-minus-matched subject-output MSE: `+212.2348`.
- Split aggregate metrics:
  - Validation behavior delta: `+0.3722`; subject-output MSE delta: `+198.4024`;
    matched subject-output MSE: `12.2244`.
  - Test behavior delta: `+0.3344`; subject-output MSE delta: `+226.0673`;
    matched subject-output MSE: `25.0675`.
- Key per-cell metrics:
  - Validation `sorted_ascending` / `same_label_other_subject`:
    - subject-output MSE delta: `+183.6531`;
    - behavior-margin delta: `-0.0034` (not gated for same-label).
  - Validation `sorted_ascending` / `opposite_direction`:
    - behavior-margin delta: `+0.7629`;
    - subject-output MSE delta: `+302.9308`.
  - Validation `sorted_ascending` / `noise_signature`:
    - behavior-margin delta: `+0.1680`;
    - subject-output MSE delta: `+109.7502`.
  - Validation `sorted_descending` / `same_label_other_subject`:
    - subject-output MSE delta: `+51.0202`;
    - behavior-margin delta: `+0.0028` (not gated for same-label).
  - Test `sorted_ascending` / `noise_signature`:
    - behavior-margin delta: `+0.2031`;
    - subject-output MSE delta: `+244.2322`.
  - Test `sorted_descending` / `same_label_other_subject`:
    - subject-output MSE delta: `+24.7806`;
    - behavior-margin delta: `-0.0361` (not gated for same-label).
- Proof-gate result: failed.
- Gate failure:
  - validation `sorted_ascending` / `noise_signature` behavior-margin delta:
    `0.1680 < 0.20`.
- Interpretation:
  - This is the first strong learned decode result, but it is still not proof because
    one preregistered cell fails.
  - It materially improves over the MLP, mean, and centroid baselines and beats the
    `1`-NN diagnostic on aggregate behavior separation and subject-output specificity.
  - The remaining failure is narrow and localized to the validation
    `sorted_ascending` / `noise_signature` behavior-margin gate.
  - Stronger decode claims should wait for a rerun that clears every preregistered
    split/behavior/control gate without changing thresholds after observing results.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the near-miss characterization and rejection as proof.
  - Blocking issues for exploratory rerun: none.
  - Required clarification applied:
    - evaluation used the best internal-validation checkpoint from epoch `72`, while
      training continued through epoch `80`.
  - Methodology caveat:
    - further tuning against this same paired artifact is adaptive development because
      validation and test outcomes have now been inspected repeatedly;
    - a final proof claim requires either a fresh untouched final proof artifact or a
      strictly reserved final test artifact after the objective is locked.
  - Rerun acceptability:
    - acceptable as adaptive development with unchanged thresholds and artifact schema;
    - not acceptable as final heldout proof unless later confirmed on fresh/reserved
      proof data.

### 2026-06-10 - Stored-Probe Functional Decoder V2 Adaptive

- Objective: rerun the proof-aligned decoder with the same artifact and thresholds, but
  add a noise-signature behavior penalty to address the V1 failure.
- Status:
  - Adaptive development result.
  - Not final heldout proof, because V1 validation/test outcomes on this artifact had
    already been inspected before this objective change.
- Output:
  - `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`;
  - `runs/stored_probe_functional_decoder_v2_adaptive/predictions.json`;
  - `runs/stored_probe_functional_decoder_v2_adaptive/results.json`.
- Method:
  - Same stored-probe signature-to-weight MLP family as V1.
  - Same paired artifact and same `control_specific_decode_proof_gate_v1`
    thresholds as V1.
  - Added loss term:
    - noise-control behavior penalty weight: `3.0`;
    - noise margin ceiling: `0.0`.
  - Other loss terms:
    - normalized weight MSE weight: `0.05`;
    - raw stored-probe output MSE weight: `0.01`;
    - behavior-margin hinge weight: `2.0`;
    - behavior-margin hinge target: `0.20`.
  - Best internal validation checkpoint epoch: `37`.
  - Early stopped at epoch `62`.
  - The saved/evaluated model uses the best internal-validation checkpoint.
- Leakage/provenance audit:
  - Reloaded dataset provenance matched the checkpoint on compared fields.
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Proof subject/control references: `120`.
  - Proof-overlap-with-train count: `0`.
  - Proof-overlap-with-checkpoint-validation count: `120`.
- Aggregate paired metrics:
  - mean matched behavior margin: `+0.3038`;
  - mean control behavior margin: `+0.0249`;
  - mean matched-minus-control behavior margin: `+0.2788`;
  - mean matched subject-output MSE: `20.8812`;
  - mean control subject-output MSE: `233.5323`;
  - mean control-minus-matched subject-output MSE: `+212.6511`.
- Split aggregate metrics:
  - Validation behavior delta: `+0.2907`; subject-output MSE delta: `+193.9743`;
    matched subject-output MSE: `13.9844`.
  - Test behavior delta: `+0.2670`; subject-output MSE delta: `+231.3279`;
    matched subject-output MSE: `27.7779`.
- Proof-gate result on current artifact: passed.
- Weakest gated cells:
  - Validation `sorted_ascending` / `noise_signature`:
    - behavior-margin delta: `+0.2350`;
    - subject-output MSE delta: `+158.1373`.
  - Test `sorted_ascending` / `noise_signature`:
    - behavior-margin delta: `+0.2312`;
    - subject-output MSE delta: `+273.3456`.
  - Test `sorted_descending` / `same_label_other_subject`:
    - subject-output MSE delta: `+20.4481`;
    - behavior-margin delta: `-0.0703` (not gated for same-label).
- Interpretation:
  - Adaptive V2 clears the current paired proof gates and is the strongest decode
    evidence so far.
  - It supports the feasibility of a proof-aligned learned decoder under the current
    development artifact.
  - It is not final proof because the same validation/test artifact informed the V2
    objective change.
  - Next rigorous step: lock the V2 objective and evaluate once on a fresh or reserved
    final proof artifact generated from checkpoint-validation subjects not used in the
    current paired artifact.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the adaptive-pass characterization.
  - Blocking issues before fresh final proof artifact: none, provided the V2 objective,
    thresholds, artifact generator, control set, and evaluator are now locked.
  - Fresh final artifact constraints:
    - use checkpoint-validation subjects not present in the current paired artifact;
    - generate the artifact once with accepted stored-probe provenance, sidecar hash
      audit, source-pool preflight, split validation, and per-split behavior/control
      count gates;
    - keep the same thresholds and control-specific gate policy;
    - keep at minimum `same_label_other_subject`, `opposite_direction`, and
      `noise_signature` controls;
    - preserve balanced per-split/per-behavior/per-control counts, preferably `n >= 10`
      per cell;
    - record all subject/control dedup IDs and assert zero overlap with train and zero
      overlap with development artifact subjects;
    - evaluate once with the locked V2 checkpoint/objective;
    - if final artifact evaluation fails, report failure and do not iterate on the
      final artifact.

### 2026-06-10 - Fresh Final Paired Artifact V1

- Objective: create a fresh final proof artifact from checkpoint-validation subjects
  that were not present in the development paired artifact.
- Status:
  - Artifact generated once after locking the V2 adaptive objective and existing
    control-specific thresholds.
  - This artifact has not been used to tune the decoder.
- Output:
  - `runs/paired_contrast_final_artifact_v1/paired_contrast_artifact.json`;
  - `runs/paired_contrast_final_artifact_v1/validation.json`;
  - `runs/paired_contrast_final_artifact_v1/regenerated_signatures.json`;
  - `runs/paired_contrast_final_artifact_v1/sidecar_audit.json`;
  - `runs/paired_contrast_final_artifact_v1/summary.json`.
- Design:
  - Single proof split named `validation` with empty `train` and `test` splits to
    preserve the existing validator contract.
  - `18` final proof groups:
    - `9` `sorted_ascending` source groups;
    - `9` `sorted_descending` source groups.
  - Required controls:
    - `same_label_other_subject`;
    - `opposite_direction`;
    - `noise_signature`.
  - Per-behavior/control count: `9` for every required cell.
  - `n=10` was impossible without reusing subjects because the unused checkpoint
    validation pool had only:
    - `31` `sorted_ascending`;
    - `29` `sorted_descending`.
  - To preserve no-reuse invariants, the final artifact uses `n=9` per cell.
- Validation:
  - Artifact builder passed.
  - `validate_paired_contrast_artifact` passed with `min_count=9` and
    `count_splits=["validation"]`.
  - Regenerated-signature sidecar audit passed.
  - Signature dimension: `560`.
- Leakage audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Development artifact subject/control refs: `120`.
  - Unused checkpoint-validation refs before final selection: `279`.
  - Final subject/control refs: `54`.
  - Final-overlap-with-train count: `0`.
  - Final-overlap-with-development-artifact count: `0`.
  - Final-overlap-with-checkpoint-validation count: `54`.
- Interpretation:
  - This is a clean final proof artifact for a one-shot locked-objective evaluation.
  - If the locked V2 decoder fails here, the failure should be reported without
    further tuning on this artifact.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for artifact rigor and non-misleading characterization.
  - Blocking issues before locked V2 evaluation: none.
  - Evaluation constraints:
    - use the artifact exactly as generated;
    - do not regenerate, rebalance, tune thresholds, or adjust the V2 objective after
      seeing final results.
  - Final evaluation caveats to report:
    - proof split is named `validation` for validator compatibility, but should be
      described as a fresh final holdout artifact;
    - sample size is smaller than development: `9` groups per behavior/control cell,
      `18` groups total;
    - scope is only `sorted_ascending` and `sorted_descending` with
      `same_label_other_subject`, `opposite_direction`, and `noise_signature` controls;
    - this evaluates a locked adaptive V2 decoder after prior development on a separate
      artifact;
    - it is not evidence for other behaviors or steering;
    - any failure here must be reported without further tuning on this artifact.

### 2026-06-10 - Locked V2 Final Holdout Decode Evaluation

- Objective: evaluate the locked adaptive V2 decoder exactly once on the fresh final
  holdout artifact.
- Status:
  - Final holdout evaluation for the two-behavior decode claim only.
  - No training, threshold changes, objective changes, artifact regeneration, or
    rebalancing occurred after seeing this result.
- Output:
  - `runs/stored_probe_functional_decoder_v2_final_eval/results.json`;
  - `runs/stored_probe_functional_decoder_v2_final_eval/predictions.json`.
- Inputs:
  - Locked model:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`.
  - Fresh final artifact:
    `runs/paired_contrast_final_artifact_v1/paired_contrast_artifact.json`.
  - Fresh final sidecar:
    `runs/paired_contrast_final_artifact_v1/regenerated_signatures.json`.
- Scope:
  - Behaviors:
    - `sorted_ascending`;
    - `sorted_descending`.
  - Controls:
    - `same_label_other_subject`;
    - `opposite_direction`;
    - `noise_signature`.
  - Proof split is named `validation` for validator compatibility, but represents a
    fresh final holdout split.
  - Per-cell count: `9`.
  - Paired comparisons: `54`.
- Leakage/provenance audit:
  - Reloaded dataset provenance matched the checkpoint on compared fields.
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Development artifact subject/control refs: `120`.
  - Final subject/control refs: `54`.
  - Final-overlap-with-train count: `0`.
  - Final-overlap-with-development-artifact count: `0`.
  - Final-overlap-with-checkpoint-validation count: `54`.
- Proof-gate result: passed.
  - Failure count: `0`.
- Aggregate paired metrics:
  - mean matched behavior margin: `+0.3337`;
  - mean control behavior margin: `+0.0141`;
  - mean matched-minus-control behavior margin: `+0.3196`;
  - mean matched subject-output MSE: `19.5971`;
  - mean control subject-output MSE: `206.8643`;
  - mean control-minus-matched subject-output MSE: `+187.2671`.
- Final per-cell metrics:
  - `sorted_ascending` / `same_label_other_subject`:
    - subject-output MSE delta: `+152.0249`;
    - behavior-margin delta: `-0.0381` (not gated for same-label);
    - matched behavior margin: `+0.1925`;
    - matched subject-output MSE: `10.9871`.
  - `sorted_ascending` / `opposite_direction`:
    - behavior-margin delta: `+0.6211`;
    - subject-output MSE delta: `+286.6485`.
  - `sorted_ascending` / `noise_signature`:
    - behavior-margin delta: `+0.2143`;
    - subject-output MSE delta: `+81.2484`.
  - `sorted_descending` / `same_label_other_subject`:
    - subject-output MSE delta: `+101.2381`;
    - behavior-margin delta: `+0.0467` (not gated for same-label);
    - matched behavior margin: `+0.4749`;
    - matched subject-output MSE: `28.2072`.
  - `sorted_descending` / `opposite_direction`:
    - behavior-margin delta: `+0.5878`;
    - subject-output MSE delta: `+316.6555`.
  - `sorted_descending` / `noise_signature`:
    - behavior-margin delta: `+0.4860`;
    - subject-output MSE delta: `+185.7874`.
- Interpretation:
  - This is the first proof-grade decode evidence in this project for the restricted
    two-behavior setting.
  - The claim supported is narrow:
    - fixed stored-probe activation signatures can condition a learned decoder that
      emits functional small-network weights for `sorted_ascending` and
      `sorted_descending`;
    - decoded weights beat same-label, opposite-direction, and noise-signature
      controls on the preregistered paired gates;
    - this held on a fresh final holdout artifact disjoint from train and disjoint from
      the development artifact.
  - This does not prove steering, other behaviors, larger models, or the full MUAT
    hypothesis.
  - The weakest final gated cell is `sorted_ascending` / `noise_signature` behavior
    delta `+0.2143`, only moderately above the `0.20` threshold, so replication with a
    larger untouched final artifact remains important.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for rigor and non-misleading characterization.
  - Required corrections before summary: none.
  - Accepted claim:
    - restricted two-behavior stored-probe decode evidence for small subject networks,
      with all preregistered control-specific gates passing on the fresh final holdout
      artifact.
  - Explicitly not accepted:
    - steering;
    - other behaviors;
    - larger models;
    - full MUAT proof.
  - Residual risks / next steps:
    - replicate with a larger untouched final artifact when more subjects are available;
    - preregister controls/gates before extending to more behaviors;
    - test steering as a separate claim with separate controls;
    - preserve this as a one-shot final result and do not tune on this artifact.

### 2026-06-10 - Signature-Space Steering Diagnostic V1

- Objective: test whether train-only behavior-centroid directions in stored-probe
  signature space can steer the locked V2 decoder's generated weights from one clean
  behavior toward the opposite clean behavior.
- Status:
  - Steering diagnostic only.
  - Not final steering proof, because it reuses the already evaluated final decode
    artifact source subjects.
  - No model training or threshold tuning was performed for this steering result.
- Output:
  - `runs/stored_probe_signature_steering_v1_diagnostic/results.json`.
- Method:
  - Source subjects: the `18` fresh final artifact source subjects.
  - Source/target pairs:
    - `sorted_ascending -> sorted_descending`;
    - `sorted_descending -> sorted_ascending`.
  - Train-only centroid direction:
    - `steered_signature = source_signature + 1.0 * (target_train_centroid - source_train_centroid)`.
  - Decoder:
    - locked V2 decoder from
      `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`.
  - Controls:
    - no edit: original source signature;
    - reverse direction: source signature minus the centroid direction;
    - noise signature: train-normalized Gaussian signature.
- Pre-set steering diagnostic thresholds:
  - mean steered-minus-no-edit target margin: `>= 0.20`;
  - mean steered-minus-reverse-direction target margin: `>= 0.20`;
  - mean steered-minus-noise target margin: `>= 0.20`;
  - mean steered target margin: `>= 0.20`.
- Leakage/provenance audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Development artifact subject/control refs: `120`.
  - Steering source subjects: `18`.
  - Steering-source-overlap-with-train count: `0`.
  - Steering-source-overlap-with-development-artifact count: `0`.
  - Steering-source-overlap-with-checkpoint-validation count: `18`.
- Aggregate steering metrics:
  - mean steered target margin: `+0.1174`;
  - mean no-edit target margin: `-0.2468`;
  - mean reverse-direction target margin: `-0.2597`;
  - mean noise target margin: `-0.0106`;
  - mean steered-minus-no-edit target margin: `+0.3642`;
  - mean steered-minus-reverse-direction target margin: `+0.3771`;
  - mean steered-minus-noise target margin: `+0.1280`;
  - mean steered source-margin change: `-0.4418`.
- Per-target metrics:
  - Target `sorted_ascending`:
    - n: `9`;
    - mean steered target margin: `+0.0535`;
    - mean steered-minus-no-edit target margin: `+0.4625`;
    - mean steered-minus-reverse-direction target margin: `+0.5226`;
    - mean steered-minus-noise target margin: `+0.0645`;
    - mean steered source-margin change: `-0.4992`.
  - Target `sorted_descending`:
    - n: `9`;
    - mean steered target margin: `+0.1813`;
    - mean steered-minus-no-edit target margin: `+0.2660`;
    - mean steered-minus-reverse-direction target margin: `+0.2315`;
    - mean steered-minus-noise target margin: `+0.1916`;
    - mean steered source-margin change: `-0.3845`.
- Gate result: failed.
- Failures:
  - aggregate steered-minus-noise target margin: `0.1280 < 0.20`;
  - aggregate steered target margin: `0.1174 < 0.20`;
  - target `sorted_ascending` steered-minus-noise target margin: `0.0645 < 0.20`;
  - target `sorted_ascending` steered target margin: `0.0535 < 0.20`;
  - target `sorted_descending` steered-minus-noise target margin: `0.1916 < 0.20`;
  - target `sorted_descending` steered target margin: `0.1813 < 0.20`.
- Interpretation:
  - The centroid direction changes behavior in the intended direction relative to
    no-edit and reverse-direction controls.
  - It does not clear the stricter noise-control and absolute target-margin gates.
  - This is not proof-grade steering evidence.
  - Current project evidence therefore remains:
    - strong interpretability evidence;
    - proof-grade restricted two-behavior decode evidence;
    - steering remains unproven under the new rigorous standard.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the failed-steering characterization.
  - Required corrections: none.
  - Accepted characterization:
    - centroid steering moves target behavior relative to no-edit and reverse-direction
      controls, but fails the stricter noise-control and absolute target-margin gates.
  - Required caveat:
    - keep `diagnostic` and `steering unproven` prominent;
    - do not summarize the no-edit/reverse improvements as proof.
  - Suggested next steering experiment/control:
    - preregister a steering-only protocol before more tuning;
    - sweep `alpha` on development subjects only;
    - choose a fixed alpha or selection rule;
    - evaluate once on a fresh steering holdout;
    - keep no-edit, reverse-direction, and noise controls;
    - add an off-target/source-retention gate so steering must increase target margin
      without collapsing into a generic weak target prior.

### 2026-06-10 - Signature-Space Steering Alpha Sweep V1

- Objective: sweep the centroid-steering strength `alpha` on development subjects only
  to see whether a fixed alpha can clear steering diagnostic gates.
- Status:
  - Development sweep only.
  - Not proof.
  - Uses the already inspected development paired artifact.
- Output:
  - `runs/stored_probe_signature_steering_alpha_sweep_v1/results.json`.
- Method:
  - Source subjects: development paired artifact source subjects.
  - Source/target pairs:
    - `sorted_ascending -> sorted_descending`;
    - `sorted_descending -> sorted_ascending`.
  - Steering rule:
    - `steered_signature = source_signature + alpha * (target_train_centroid - source_train_centroid)`.
  - Alphas:
    - `0.25`, `0.5`, `0.75`, `1.0`, `1.25`, `1.5`, `2.0`, `2.5`, `3.0`.
  - Controls:
    - no edit;
    - reverse direction;
    - noise signature.
  - Added source-margin-change gate:
    - mean steered source-margin change must be `<= -0.05`.
- Development thresholds:
  - mean steered-minus-no-edit target margin: `>= 0.20`;
  - mean steered-minus-reverse-direction target margin: `>= 0.20`;
  - mean steered-minus-noise target margin: `>= 0.20`;
  - mean steered target margin: `>= 0.20`;
  - mean steered source-margin change: `<= -0.05`.
- Leakage/provenance audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Development source subjects: `40`.
  - Development-source-overlap-with-train count: `0`.
  - Development-source-overlap-with-checkpoint-validation count: `40`.
- Result:
  - No alpha passed all development steering gates.
  - Selection rule recorded for future protocol design only:
    - no alpha passed;
    - selected lowest-failure/highest-noise-margin candidate.
  - Selected candidate: `alpha=3.0`.
- Best/high-alpha development metrics:
  - `alpha=1.0`:
    - aggregate steered target margin: `+0.1677`;
    - aggregate steered-minus-noise target margin: `+0.1839`;
    - target `sorted_ascending` steered target margin: `+0.1321`;
    - target `sorted_ascending` steered-minus-noise target margin: `+0.1539`;
    - target `sorted_descending` steered target margin: `+0.2034`;
    - target `sorted_descending` steered-minus-noise target margin: `+0.2138`;
    - failure count: `4`.
  - `alpha=1.25`:
    - aggregate steered target margin: `+0.2596`;
    - aggregate steered-minus-noise target margin: `+0.2757`;
    - target `sorted_ascending` steered target margin: `+0.1577`;
    - target `sorted_ascending` steered-minus-noise target margin: `+0.1795`;
    - target `sorted_descending` steered target margin: `+0.3615`;
    - target `sorted_descending` steered-minus-noise target margin: `+0.3720`;
    - failure count: `2`.
  - `alpha=3.0`:
    - aggregate steered target margin: `+0.3624`;
    - aggregate steered-minus-noise target margin: `+0.3785`;
    - target `sorted_ascending` steered target margin: `+0.1660`;
    - target `sorted_ascending` steered-minus-noise target margin: `+0.1878`;
    - target `sorted_descending` steered target margin: `+0.5587`;
    - target `sorted_descending` steered-minus-noise target margin: `+0.5692`;
    - mean source-margin change: `-0.5718`;
    - failure count: `2`.
- Interpretation:
  - Alpha scaling produces a strong monotonic steering effect toward
    `sorted_descending`.
  - Steering toward `sorted_ascending` saturates below the absolute target-margin and
    noise-control gates.
  - No fixed alpha in this sweep supports a proof-grade steering claim.
  - Steering remains unproven.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for the alpha-sweep characterization.
  - Required corrections: none.
  - Reporting caveat:
    - preserve alpha labels from JSON object keys; alpha is not stored inside each
      alpha-result object.
  - Accepted conclusion:
    - simple centroid-vector steering is not proof-grade;
    - `sorted_descending` steering works directionally;
    - `sorted_ascending` steering saturates below gates.
  - Recommended next direction:
    - stop alpha-only tuning;
    - train a steering-specific signed edit module or signature transformation
      objective with paired source-target supervision;
    - include explicit noise, reverse-direction, and no-edit controls;
    - tune only on development subjects;
    - evaluate once on a fresh steering holdout.

### 2026-06-10 - Learned Signature Edit Vectors V1 Development

- Objective: test whether a steering-specific learned edit in normalized stored-probe
  signature space can fix the centroid-steering failure.
- Status:
  - Development result only.
  - Not final steering proof.
  - Evaluated on the previously inspected development paired artifact.
- Output:
  - `runs/stored_probe_signature_edit_vectors_v1_development/results.json`;
  - `runs/stored_probe_signature_edit_vectors_v1_development/edit_vectors.pt`.
- Method:
  - Frozen decoder:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`.
  - Trainable parameters:
    - one normalized signature-space edit vector for
      `sorted_ascending -> sorted_descending`;
    - one normalized signature-space edit vector for
      `sorted_descending -> sorted_ascending`.
  - Initialization:
    - train-centroid difference vectors in normalized signature space.
  - Training data:
    - checkpoint-train subjects only;
    - internal train/validation split carved from checkpoint-train sources with
      `sorted_ascending` or `sorted_descending` labels.
  - Training objective:
    - target margin hinge target: `0.30`;
    - target-margin improvement hinge: `0.25`;
    - source margin ceiling: `0.05`;
    - source suppression weight: `0.5`;
    - edit-vector L2 weight: `0.0005`.
  - Best internal validation checkpoint:
    - epoch `189`;
    - internal validation loss `0.000573`.
  - Early stopped at epoch `249`.
- Development evaluation:
  - Source subjects: development paired artifact source subjects, `n=40`.
  - Controls:
    - no edit;
    - reverse direction;
    - noise signature.
  - Same steering gates as alpha sweep:
    - mean steered-minus-no-edit target margin: `>= 0.20`;
    - mean steered-minus-reverse-direction target margin: `>= 0.20`;
    - mean steered-minus-noise target margin: `>= 0.20`;
    - mean steered target margin: `>= 0.20`;
    - mean steered source-margin change: `<= -0.05`.
- Leakage/provenance audit:
  - Checkpoint train indices: `3592`.
  - Checkpoint validation indices: `399`.
  - Development source subjects: `40`.
  - Development-source-overlap-with-train count: `0`.
  - Development-source-overlap-with-checkpoint-validation count: `40`.
- Gate result on development artifact: passed.
  - Failure count: `0`.
- Aggregate metrics:
  - mean steered target margin: `+0.4274`;
  - mean no-edit target margin: `-0.2309`;
  - mean reverse-direction target margin: `-0.2984`;
  - mean noise target margin: `-0.0184`;
  - mean steered-minus-no-edit target margin: `+0.6583`;
  - mean steered-minus-reverse-direction target margin: `+0.7258`;
  - mean steered-minus-noise target margin: `+0.4458`;
  - mean steered source-margin change: `-0.6227`.
- Per-target metrics:
  - Target `sorted_ascending`:
    - n: `20`;
    - mean steered target margin: `+0.2970`;
    - mean steered-minus-no-edit target margin: `+0.6612`;
    - mean steered-minus-reverse-direction target margin: `+0.8442`;
    - mean steered-minus-noise target margin: `+0.3154`;
    - mean steered source-margin change: `-0.5438`.
  - Target `sorted_descending`:
    - n: `20`;
    - mean steered target margin: `+0.5578`;
    - mean steered-minus-no-edit target margin: `+0.6553`;
    - mean steered-minus-reverse-direction target margin: `+0.6073`;
    - mean steered-minus-noise target margin: `+0.5763`;
    - mean steered source-margin change: `-0.7017`.
- Interpretation:
  - A steering-specific learned signature edit clears the development steering gates.
  - This fixes the centroid-alpha failure on the development artifact, including the
    weak `sorted_ascending` target direction.
  - This is not final steering proof because no fresh balanced steering holdout remains
    in the current checkpoint-validation pool after the final decode artifact was used.
  - Current steering status:
    - development evidence is now positive for learned signature edits;
    - proof-grade steering remains pending fresh heldout data.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for rigor and non-misleading characterization.
  - Required corrections: none.
  - Accepted characterization:
    - learned edit vectors pass development steering gates and fix the centroid-alpha
      `sorted_ascending` weakness;
    - result is correctly scoped as development only.
  - Caveat to keep visible:
    - learned vectors are large in normalized signature space, so future proof must
      include controls for out-of-distribution signature edits.
  - Next honest step:
    - obtain or reserve fresh balanced steering holdout data with the same stored-probe
      provenance;
    - lock the edit-vector training recipe and gates;
    - evaluate once;
    - include existing no-edit, reverse-direction, and noise controls;
    - add OOD/edit-magnitude controls or gates, such as max normalized edit norm,
      interpolation/path sanity checks, or random vectors matched to learned edit norm.

### 2026-06-10 - Fresh External Steering Holdout V1

- Objective: create fresh steering holdout subjects outside the checkpoint-validation
  pool, so learned steering can be evaluated without reusing the exhausted decode
  artifacts.
- Status:
  - Fresh externally generated holdout artifact.
  - Not used for training the locked decoder or learned edit vectors.
- Output:
  - `runs/fresh_external_steering_holdout_v1/subjects.json`;
  - `runs/fresh_external_steering_holdout_v1/summary.json`.
- Method:
  - Trained new `SubjectNetwork` instances from scratch with the same `5 x 8`
    architecture and flat weight dimension `345`.
  - Behaviors:
    - `sorted_ascending`;
    - `sorted_descending`.
  - Count:
    - `12` subjects per behavior;
    - `24` total subjects.
  - Training data:
    - exhaustive positives for the target predicate;
    - sampled negatives from the finite digit-sequence universe;
    - `350` epochs;
    - AdamW learning rate `0.003`.
  - Acceptance gate for source subjects:
    - heldout behavior margin must be `>= 0.40`.
  - Stored-probe signatures:
    - same accepted probe set: `stored_digit_probe_v1_seed_20260610_n256`;
    - probe examples hash:
      `b156dabece5a9eb58a966271388c8e5479fd308712dcca7b373e0f253e670279`;
    - signature dimension: `560`;
    - signature hash algorithm: `stable_hash_json_float_list_v1`.
- Source behavior quality:
  - `sorted_ascending`:
    - min heldout margin: `0.9857`;
    - mean heldout margin: `0.9973`;
    - max heldout margin: `1.0000`.
  - `sorted_descending`:
    - min heldout margin: `0.9790`;
    - mean heldout margin: `0.9963`;
    - max heldout margin: `0.9999`.
- Validation:
  - Generated subject count passed.
  - Behavior balance passed.
  - All source subjects cleared the heldout behavior-margin gate.
- Interpretation:
  - This provides fresh source signatures for a steering evaluation that is independent
    of the prior checkpoint-validation artifacts.
  - It is an external-distribution holdout relative to the HF/checkpoint rows, so
    passing here would be stronger; failure would indicate the learned edit/decoder is
    distribution-sensitive.
- Review:
  - Reviewer: Kepler.
  - Confidence: `5/5` for artifact rigor and non-misleading characterization.
  - Required evaluator guard:
    - source margin field is `heldout_margin`, not `heldout_behavior_margin`.
  - Required result binding:
    - record a hash of `subjects.json` in the steering evaluation log.
  - Evaluation constraints:
    - use as one-shot holdout;
    - locked decoder;
    - locked learned edit vectors;
    - locked thresholds;
    - no alpha/objective tuning after seeing results;
    - include no-edit, reverse-direction, noise, and norm-matched random edit controls;
    - keep source-margin suppression gate.
  - Caveat:
    - this tests external-distribution generalization to newly trained same-architecture
      subjects, not broader task/model proof.

### 2026-06-10 - Locked Learned Edit Vectors External Steering Evaluation

- Objective: evaluate the locked learned signature edit vectors exactly once on the
  fresh external steering holdout.
- Status:
  - Fresh external holdout evaluation for restricted two-behavior steering.
  - No decoder training, edit-vector training, threshold changes, or holdout changes
    occurred after seeing this result.
- Output:
  - `runs/stored_probe_signature_edit_vectors_v1_external_eval/results.json`.
- Inputs:
  - Fresh holdout subjects:
    `runs/fresh_external_steering_holdout_v1/subjects.json`.
  - Holdout SHA-256:
    `6116ad4af8e10fbd515e41e09ed7ed738c28ff2611917d7561fad0cf74825754`.
  - Locked decoder:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`.
  - Locked edit vectors:
    `runs/stored_probe_signature_edit_vectors_v1_development/edit_vectors.pt`.
- Scope:
  - Source/target steering:
    - `sorted_ascending -> sorted_descending`;
    - `sorted_descending -> sorted_ascending`.
  - Fresh source subjects:
    - `12` per source behavior;
    - `24` total.
  - Same subject-network architecture as the training distribution.
  - External distribution relative to HF/checkpoint rows because source models were
    trained from scratch for this holdout.
- Locked gates:
  - mean steered-minus-no-edit target margin: `>= 0.20`;
  - mean steered-minus-reverse-direction target margin: `>= 0.20`;
  - mean steered-minus-noise target margin: `>= 0.20`;
  - mean steered-minus-random-norm-matched target margin: `>= 0.20`;
  - mean steered target margin: `>= 0.20`;
  - mean steered source-margin change: `<= -0.05`.
- Controls:
  - no edit;
  - reverse direction;
  - noise signature;
  - random normalized edit vector matched to the learned edit-vector norm.
- Gate result: passed.
  - Failure count: `0`.
  - Gate level: mean aggregate and per-target means.
  - These are not per-source reliability gates.
- Aggregate metrics:
  - n: `24`;
  - mean steered target margin: `+0.3583`;
  - mean no-edit target margin: `-0.0358`;
  - mean reverse-direction target margin: `-0.2171`;
  - mean noise target margin: `-0.0048`;
  - mean random-norm-matched target margin: `-0.0278`;
  - mean steered-minus-no-edit target margin: `+0.3941`;
  - mean steered-minus-reverse-direction target margin: `+0.5755`;
  - mean steered-minus-noise target margin: `+0.3631`;
  - mean steered-minus-random-norm-matched target margin: `+0.3861`;
  - mean steered source-margin change: `-0.3041`.
- Per-target metrics:
  - Target `sorted_ascending`:
    - n: `12`;
    - mean steered target margin: `+0.2486`;
    - mean steered-minus-no-edit target margin: `+0.2568`;
    - mean steered-minus-reverse-direction target margin: `+0.6281`;
    - mean steered-minus-noise target margin: `+0.2516`;
    - mean steered-minus-random-norm-matched target margin: `+0.2568`;
    - mean steered source-margin change: `-0.1182`.
  - Target `sorted_descending`:
    - n: `12`;
    - mean steered target margin: `+0.4681`;
    - mean steered-minus-no-edit target margin: `+0.5315`;
    - mean steered-minus-reverse-direction target margin: `+0.5228`;
    - mean steered-minus-noise target margin: `+0.4747`;
    - mean steered-minus-random-norm-matched target margin: `+0.5155`;
    - mean steered source-margin change: `-0.4899`.
- Individual-subject audit:
  - Individual all-gate pass rate using the same numeric thresholds as the mean gates:
    `23/24` (`95.8%`).
  - Per-check individual pass rates:
    - steered target margin: `23/24` (`95.8%`);
    - steered-minus-no-edit target margin: `23/24` (`95.8%`);
    - steered-minus-reverse-direction target margin: `23/24` (`95.8%`);
    - steered-minus-noise target margin: `23/24` (`95.8%`);
    - steered-minus-random-norm-matched target margin: `23/24` (`95.8%`);
    - steered source-margin change: `24/24` (`100.0%`).
  - Individual failure:
    - `fresh:sorted_ascending:6:seed:20260623`
    - source: `sorted_ascending`;
    - target: `sorted_descending`;
    - steered target margin: approximately `0.0000`, below `0.20`;
    - steered-minus-no-edit target margin: `+0.1106`, below `0.20`;
    - steered-minus-reverse-direction target margin: `+0.0559`, below `0.20`;
    - steered-minus-noise target margin: `+0.0018`, below `0.20`;
    - steered-minus-random-norm-matched target margin: `+0.1112`, below `0.20`.
- Interpretation:
  - This is proof-grade mean steering evidence for the restricted two-behavior,
    same-architecture small-network setting under the locked protocol.
  - The learned signature edit vectors steer decoded models toward the target behavior
    and away from the source behavior on fresh externally generated source models on
    aggregate and per-target means.
  - The result beats no-edit, reverse-direction, noise-signature, and norm-matched
    random-vector controls on aggregate and per-target means.
  - It does not prove per-subject reliable steering because one of the 24 fresh
    external subjects fails individual target/noise/random/no-edit steering margins.
  - The claim remains narrow:
    - two sorted behaviors only;
    - same small subject architecture only;
    - locked stored-probe signature/decoder setup only.
  - It does not prove larger models, additional behaviors, or broad MUAT generality.
- Reviewer outcome:
  - Initial reviewer confidence: `4/5`.
  - Blocker:
    - the first interpretation did not clearly distinguish mean proof gates from
      per-source reliability;
    - one individual source subject failed target/noise/random/no-edit steering
      margins.
  - Correction:
    - added the individual-subject audit above;
    - narrowed the accepted wording to proof-grade mean steering evidence.
  - Final reviewer confidence after correction: `5/5`.
  - Accepted scope:
    - proof-grade mean steering evidence for locked learned signature edit vectors in
      the restricted two-behavior, same-architecture small-network setting;
    - evaluated once on fresh external source subjects;
    - not evidence for per-subject reliable steering, additional behaviors, larger
      models, or broad MUAT generality.

### 2026-06-10 - Strict External Steering Robustness Evaluation

- Objective: test whether the previously accepted learned signature edit vectors also
  satisfy stronger per-subject reliability gates under harder random-vector controls.
- Status:
  - Fresh robustness audit for restricted two-behavior steering.
  - This is a limiting result, not a positive proof result.
  - No threshold changes or method tuning were performed after seeing this result.
- Reproducible runner:
  - `model_zoo/scripts/run_stored_probe_steering_robustness.py`.
  - Verification:
    - `python -m py_compile model_zoo/scripts/run_stored_probe_steering_robustness.py`
      passed.
  - Smoke check:
    - output:
      `runs/stored_probe_signature_edit_vectors_v1_robust_existing_holdout_smoke/results.json`;
    - ran the stricter evaluator on the previously accepted 24-subject holdout with
      `8` random controls;
    - reproduced the accepted aggregate behavior and known individual limitation.
- Fresh robustness output:
  - `runs/stored_probe_signature_edit_vectors_v1_robust_external_eval/results.json`.
- Fresh holdout:
  - subjects:
    `runs/fresh_external_steering_holdout_v2_robust/subjects.json`;
  - SHA-256:
    `a0f1727294b7bb188a461b0222592d890245408394e7a87cb806000d9ad53e9f`;
  - `48` fresh subjects total;
  - `24` per source behavior;
  - same small subject architecture;
  - source models trained from scratch;
  - source acceptance gate: heldout source-behavior margin `>= 0.40`.
- Locked inputs:
  - decoder:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`;
  - edit vectors:
    `runs/stored_probe_signature_edit_vectors_v1_development/edit_vectors.pt`.
- Coordinate audit:
  - no-edit and steered margins from the previous accepted external evaluation were
    exactly reproduced before the robustness run;
  - decoder input is normalized stored-probe signature space;
  - learned edit vectors are applied directly in normalized-signature coordinates.
- Strict gates:
  - mean steered-minus-no-edit target margin: `>= 0.20`;
  - mean steered-minus-reverse-direction target margin: `>= 0.20`;
  - mean steered-minus-noise target margin: `>= 0.20`;
  - mean steered-minus-worst-random-norm-matched target margin: `>= 0.20`;
  - mean steered target margin: `>= 0.20`;
  - mean steered source-margin change: `<= -0.05`;
  - individual all-gate pass rate: `>= 0.95`;
  - per-target individual all-gate pass rate: `>= 0.90`.
- Controls:
  - no edit;
  - reverse direction;
  - noise signature;
  - worst-of-32 norm-matched random edit vectors per subject.
- Gate result: failed.
  - Mean gates all passed.
  - Individual pass-rate gates failed.
- Aggregate metrics:
  - n: `48`;
  - mean steered target margin: `+0.3851`;
  - mean no-edit target margin: `-0.0464`;
  - mean reverse-direction target margin: `-0.2353`;
  - mean noise target margin: `-0.0089`;
  - mean worst-random-norm-matched target margin: `+0.0058`;
  - mean steered-minus-no-edit target margin: `+0.4315`;
  - mean steered-minus-reverse-direction target margin: `+0.6204`;
  - mean steered-minus-noise target margin: `+0.3940`;
  - mean steered-minus-worst-random-norm-matched target margin: `+0.3793`;
  - mean steered source-margin change: `-0.3750`.
- Per-target metrics:
  - Target `sorted_ascending`:
    - n: `24`;
    - mean steered target margin: `+0.2358`;
    - mean steered-minus-no-edit target margin: `+0.2469`;
    - mean steered-minus-reverse-direction target margin: `+0.6586`;
    - mean steered-minus-noise target margin: `+0.2459`;
    - mean steered-minus-worst-random-norm-matched target margin: `+0.2122`;
    - mean steered source-margin change: `-0.1323`;
    - individual all-gate pass rate: `19/24` (`79.2%`), below `90.0%`.
  - Target `sorted_descending`:
    - n: `24`;
    - mean steered target margin: `+0.5345`;
    - mean steered-minus-no-edit target margin: `+0.6161`;
    - mean steered-minus-reverse-direction target margin: `+0.5823`;
    - mean steered-minus-noise target margin: `+0.5421`;
    - mean steered-minus-worst-random-norm-matched target margin: `+0.5465`;
    - mean steered source-margin change: `-0.6176`;
    - individual all-gate pass rate: `24/24` (`100.0%`).
- Individual-subject audit:
  - overall individual all-gate pass rate: `43/48` (`89.6%`), below `95.0%`;
  - per-check pass rates:
    - steered target margin: `46/48` (`95.8%`);
    - steered-minus-no-edit target margin: `45/48` (`93.8%`);
    - steered-minus-reverse-direction target margin: `48/48` (`100.0%`);
    - steered-minus-noise target margin: `47/48` (`97.9%`);
    - steered-minus-worst-random-norm-matched target margin: `44/48` (`91.7%`);
    - steered source-margin change: `46/48` (`95.8%`).
  - failed records:
    - `5` total;
    - all are source `sorted_descending` to target `sorted_ascending`.
- Interpretation:
  - This result strengthens the mean-steering evidence because mean gates still pass
    under a harder worst-of-32 norm-matched random-vector control.
  - It also falsifies a stronger robustness claim for the current method:
    the locked learned edit vectors do not prove per-subject reliable steering under
    the preregistered individual pass-rate gates.
  - The weakness is directional:
    steering toward `sorted_ascending` is less reliable than steering toward
    `sorted_descending`.
  - This holdout should remain a robustness audit and must not be tuned against.
- Reviewer outcome:
  - Initial reviewer confidence: `4/5`.
  - Blocker:
    - result metrics were sound, but failure-string wording in `results.json` read
      backwards.
  - Correction:
    - fixed failure strings to report observed value below required threshold;
    - added per-record `individual_all_gates_passed`;
    - reran evaluation on the same frozen holdout without regeneration.
  - Final reviewer confidence after correction: `5/5`.
  - Accepted scope:
    - rigorous negative/limiting robustness result;
    - locked learned edit vectors support mean steering under worst-of-32
      norm-matched random controls;
    - they fail preregistered robust per-subject reliability gates:
      `43/48` overall vs `95.0%` required, and `19/24` for target
      `sorted_ascending` vs `90.0%` required.

### 2026-06-10 - Robust Signature Edit Vectors V2 Development

- Objective: train a new robust steering method without tuning on the failed
  robustness holdout.
- Status:
  - Development result only.
  - Eligible for a new fresh final holdout evaluation.
  - Not proof of robust steering by itself.
- Reproducible runner:
  - `model_zoo/scripts/train_robust_signature_edit_vectors.py`.
  - Verification:
    - `python -m py_compile model_zoo/scripts/train_robust_signature_edit_vectors.py model_zoo/scripts/run_stored_probe_steering_robustness.py`
      passed.
- Result artifact:
  - `runs/stored_probe_signature_edit_vectors_v2_robust_development/results.json`.
- Stored edit vectors:
  - `runs/stored_probe_signature_edit_vectors_v2_robust_development/edit_vectors.pt`.
- Top-level result keys used:
  - `training_pool_subjects_sha256`;
  - `train_pool_passed`;
  - `validation_pool_passed`;
  - `edit_vectors_path`;
  - `init_edit_vectors_path`.
- Inputs:
  - frozen decoder:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`;
  - initialization:
    `runs/stored_probe_signature_edit_vectors_v1_development/edit_vectors.pt`;
  - fresh training/development pool:
    `runs/fresh_robust_edit_v2_train_pool/subjects.json`;
  - `training_pool_subjects_sha256`:
    `ce49fc086eaab211c48e45c59f92d22c7288d6cb7da980c8cba8e65b0004e8dd`.
- Pool:
  - `80` fresh subjects total;
  - `40` per source behavior;
  - split into `64` train subjects and `16` validation subjects;
  - no subject-ID overlap with either prior external holdout.
- Method:
  - trainable parameters:
    - one normalized-signature edit vector for
      `sorted_ascending -> sorted_descending`;
    - one normalized-signature edit vector for
      `sorted_descending -> sorted_ascending`.
  - decoder frozen.
  - initial edit vectors from v1.
  - model selection by training loss.
  - validation pool used as development evaluation, not as final proof.
- Training objective:
  - target margin hinge: `0.35`;
  - target improvement hinge: `0.30`;
  - random delta hinge: `0.25`;
  - source margin ceiling: `0.0`;
  - source margin reduction target: approximately `0.10`;
  - random train controls per step: `2`;
  - vector L2 mean penalty: `0.0001`.
- Training result:
  - best epoch: `300`;
  - best train loss: `0.000426`;
  - vector norm caveat:
    - `sorted_ascending -> sorted_descending`: about `20.50`;
    - `sorted_descending -> sorted_ascending`: about `44.34`.
- Development evaluation protocol:
  - same strict evaluator as the failed robustness result;
  - worst-of-32 norm-matched random-vector controls;
  - same mean and individual pass-rate gates.
- Train-pool evaluation:
  - artifact:
    `runs/stored_probe_signature_edit_vectors_v2_robust_development/train_pool_results.json`;
  - `train_pool_passed`: `true`;
  - n: `64`;
  - mean steered target margin: `+0.5021`;
  - mean steered-minus-no-edit target margin: `+0.5353`;
  - mean steered-minus-reverse-direction target margin: `+0.5308`;
  - mean steered-minus-noise target margin: `+0.5138`;
  - mean steered-minus-worst-random-norm-matched target margin: `+0.4974`;
  - mean steered source-margin change: `-0.4415`;
  - individual all-gate pass rate: `64/64` (`100.0%`).
- Validation-pool evaluation:
  - artifact:
    `runs/stored_probe_signature_edit_vectors_v2_robust_development/validation_pool_results.json`;
  - `validation_pool_passed`: `true`;
  - n: `16`;
  - mean steered target margin: `+0.5142`;
  - mean steered-minus-no-edit target margin: `+0.5639`;
  - mean steered-minus-reverse-direction target margin: `+0.5417`;
  - mean steered-minus-noise target margin: `+0.5227`;
  - mean steered-minus-worst-random-norm-matched target margin: `+0.5155`;
  - mean steered source-margin change: `-0.4608`;
  - individual all-gate pass rate: `16/16` (`100.0%`).
- Interpretation:
  - V2 robust edit vectors cleared strict development gates on a separate fresh
    train/development pool.
  - This result does not prove robust steering because the method was developed
    after the previous robustness failure.
  - The next admissible result is a one-shot fresh final holdout with frozen v2
    vectors, frozen decoder, and frozen strict gates.
- Reviewer outcome:
  - Reviewer confidence: `5/5` for the narrow development characterization.
  - Accepted scope:
    - development evidence only;
    - v2 robust edit vectors are eligible for one fresh final holdout evaluation;
    - failed robustness holdout must remain diagnostic/development history and not be
      reused as final proof.

### 2026-06-10 - Robust Signature Edit Vectors V2 Final Holdout

- Objective: evaluate the frozen v2 robust edit vectors exactly once on a new fresh
  final holdout with the strict robustness gates.
- Status:
  - Final robust steering evaluation for the restricted two-behavior setting.
  - Proof-grade within the narrow scope below.
  - No method, threshold, decoder, or edit-vector changes were made after seeing this
    result.
- Output:
  - `runs/stored_probe_signature_edit_vectors_v2_robust_final_eval/results.json`.
- Fresh final holdout:
  - subjects:
    `runs/fresh_external_steering_holdout_v3_robust_final/subjects.json`;
  - SHA-256:
    `8c9f2cc2ddf1f407c52155f6b483dbed96c00c02a9ad846b1f64ac9f5c1e1124`;
  - `48` fresh subjects total;
  - `24` per source behavior;
  - same small subject architecture;
  - source models trained from scratch;
  - source acceptance gate: heldout source-behavior margin `>= 0.40`.
- No-overlap audit:
  - no subject-ID overlap with the failed strict robustness holdout;
  - no subject-ID overlap with the v2 train/development pool;
  - no subject-ID overlap with the earlier accepted external holdout.
- Frozen inputs:
  - decoder:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`;
  - v2 edit vectors:
    `runs/stored_probe_signature_edit_vectors_v2_robust_development/edit_vectors.pt`.
- Vector norms:
  - `sorted_ascending -> sorted_descending`: `20.4962`;
  - `sorted_descending -> sorted_ascending`: `44.3385`.
- Strict gates:
  - mean steered-minus-no-edit target margin: `>= 0.20`;
  - mean steered-minus-reverse-direction target margin: `>= 0.20`;
  - mean steered-minus-noise target margin: `>= 0.20`;
  - mean steered-minus-worst-random-norm-matched target margin: `>= 0.20`;
  - mean steered target margin: `>= 0.20`;
  - mean steered source-margin change: `<= -0.05`;
  - individual all-gate pass rate: `>= 0.95`;
  - per-target individual all-gate pass rate: `>= 0.90`.
- Controls:
  - no edit;
  - reverse direction;
  - noise signature;
  - worst-of-32 norm-matched random edit vectors per subject.
- Gate result: passed.
  - Failure count: `0`.
  - Individual all-gate pass rate: `48/48` (`100.0%`).
  - Per-target individual all-gate pass rates:
    - target `sorted_ascending`: `24/24` (`100.0%`);
    - target `sorted_descending`: `24/24` (`100.0%`).
  - Each individual check passed `48/48`.
- Aggregate metrics:
  - n: `48`;
  - mean steered target margin: `+0.5063`;
  - mean no-edit target margin: `-0.0386`;
  - mean reverse-direction target margin: `-0.0267`;
  - mean noise target margin: `-0.0119`;
  - mean worst-random-norm-matched target margin: `+0.0073`;
  - mean steered-minus-no-edit target margin: `+0.5449`;
  - mean steered-minus-reverse-direction target margin: `+0.5329`;
  - mean steered-minus-noise target margin: `+0.5181`;
  - mean steered-minus-worst-random-norm-matched target margin: `+0.4990`;
  - mean steered source-margin change: `-0.4580`.
- Per-target metrics:
  - Target `sorted_ascending`:
    - n: `24`;
    - mean steered target margin: `+0.4601`;
    - mean steered-minus-no-edit target margin: `+0.4505`;
    - mean steered-minus-reverse-direction target margin: `+0.4660`;
    - mean steered-minus-noise target margin: `+0.4764`;
    - mean steered-minus-worst-random-norm-matched target margin: `+0.4338`;
    - mean steered source-margin change: `-0.2989`.
  - Target `sorted_descending`:
    - n: `24`;
    - mean steered target margin: `+0.5524`;
    - mean steered-minus-no-edit target margin: `+0.6393`;
    - mean steered-minus-reverse-direction target margin: `+0.5998`;
    - mean steered-minus-noise target margin: `+0.5599`;
    - mean steered-minus-worst-random-norm-matched target margin: `+0.5642`;
    - mean steered source-margin change: `-0.6172`.
- Record-level reviewer audit:
  - weakest individual steered target margin: `+0.3068`;
  - weakest steered-minus-worst-random delta: `+0.3026`;
  - weakest source suppression still cleared the source-change gate:
    max source change `-0.0879 <= -0.05`.
- Interpretation:
  - This is proof-grade robust steering evidence for the restricted two-behavior,
    same-architecture small-network setting.
  - Frozen v2 normalized-signature edit vectors steer fixed stored-probe signatures
    through the locked decoder and pass strict mean plus per-subject gates on a fresh
    final holdout.
  - Controls beaten:
    - no edit;
    - reverse direction;
    - noise signature;
    - worst-of-32 norm-matched random edit vectors.
  - The claim remains narrow:
    - only `sorted_ascending <-> sorted_descending`;
    - same 5x8 subject architecture;
    - same stored-probe/decoder setup;
    - not additional behaviors;
    - not larger models;
    - not broad MUAT generality.
  - Caveat:
    - the `sorted_descending -> sorted_ascending` vector is large (`44.3385` norm),
      so the result may rely on an aggressive normalized-signature edit.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none methodological.
  - Accepted scope:
    - proof-grade robust steering evidence for the restricted two-behavior,
      same-architecture small-network setting;
    - frozen v2 normalized-signature edit vectors pass strict mean and per-subject
      gates on a fresh final holdout;
    - controls beaten are no-edit, reverse-direction, noise signature, and
      worst-of-32 norm-matched random edit vectors.

### 2026-06-10 - Evidence Package Audit

- Objective: create a machine-checkable audit of the current narrow evidence bundle.
- Status:
  - Artifact-level verifier passed.
  - Reviewer accepted the corrected verifier at `5/5`.
  - This does not rerun training or regenerate datasets.
- Verifier:
  - `model_zoo/scripts/audit_muat_evidence_package.py`.
- Output:
  - `runs/muat_evidence_package_audit/results.json`.
- Verification:
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py`
    passed.
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
- Evidence-supported claims:
  - stored-probe signatures contain heldout-decodable behavior information for four
    clean behaviors under logistic/RF classifiers with shuffled-label controls;
  - restricted two-behavior functional decoding on fresh final holdout;
  - restricted two-behavior robust steering on fresh final holdout.
- Explicitly not proven:
  - larger models;
  - additional behaviors beyond `sorted_ascending`/`sorted_descending` for
    decode/steering;
  - broad MUAT generality;
  - non-aggressive steering-vector norm requirement.
- Audit checks:
  - interpretability metrics and shuffled-label controls;
  - final decode pass, leakage counts, and exact registered control-specific
    thresholds;
  - V1 strict robustness negative result;
  - V2 robust development result;
  - V2 final robust steering pass with exact strict thresholds;
  - subject-pool SHA and subject-ID separation;
  - research-log review status.
- Reviewer outcome:
  - Initial reviewer confidence: `4/5`.
  - Required corrections:
    - replace `proven` wording with `evidence_supported`;
    - narrow the interpretability claim;
    - hardcode expected decode and steering threshold values;
    - independently check all V2 final aggregate and per-target mean gates;
    - include steering vector norms.
  - Corrections applied.
  - Final reviewer confidence: `5/5`.
  - Accepted scope:
    - valid artifact-level verifier for the current narrow evidence bundle;
    - does not support larger-model, additional-behavior decode/steering, broad MUAT
      generality, or non-aggressive edit-vector claims.

### 2026-06-10 - Additional-Behavior Decode Feasibility

- Objective: test the next audit gap by checking whether the locked stored-probe
  decoder produces useful matched decode margins for additional clean behaviors.
- Status:
  - Negative feasibility result.
  - Not a preregistered final proof.
  - Accepted by reviewer at `5/5` as a bounded limitation.
- Script:
  - `model_zoo/scripts/evaluate_additional_behavior_decode_feasibility.py`.
- Verification:
  - `python -m py_compile model_zoo/scripts/evaluate_additional_behavior_decode_feasibility.py`
    passed.
- Output:
  - `runs/stored_probe_additional_behavior_decode_feasibility_v1/results.json`.
- Fresh holdout:
  - subjects:
    `runs/fresh_additional_behavior_decode_holdout_v1/subjects.json`;
  - SHA-256:
    `03b72098c773690011fa330487e51d08f69c2f3b4558e7ab1ae31ae82f5aeb6b`;
  - n: `16`;
  - `8` `has_majority`;
  - `8` `mountain_pattern`.
- Method:
  - locked decoder:
    `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`;
  - no decoder or method training on these fresh additional-behavior subjects;
  - decode matched weights from each stored-probe signature;
  - compare target behavior margin against worst-of-8 normalized-signature noise
    controls.
- Source model caveat:
  - source acceptance gate lowered to `0.20` for this bounded feasibility run;
  - `has_majority` source margins were weak:
    - min `0.2017`;
    - mean `0.2279`;
    - max `0.2725`;
  - `mountain_pattern` source margins were strong:
    - min `0.8179`;
    - mean `0.8921`;
    - max `0.9386`.
- Feasibility gates:
  - mean matched target margin: `>= 0.20`;
  - mean matched-minus-worst-noise target margin: `>= 0.20`;
  - individual pass rate: `>= 0.90`;
  - per-behavior individual pass rate: `>= 0.80`.
- Gate result: failed.
  - Overall individual pass rate: `0/16`.
  - `has_majority` individual pass rate: `0/8`.
  - `mountain_pattern` individual pass rate: `0/8`.
- Aggregate metrics:
  - mean matched target margin: `-0.0174`;
  - mean worst-noise target margin: `+0.0219`;
  - mean matched-minus-worst-noise target margin: `-0.0393`.
- Per-behavior metrics:
  - `has_majority`:
    - mean matched target margin: `+0.0061`;
    - mean matched-minus-worst-noise target margin: `-0.0291`.
  - `mountain_pattern`:
    - mean matched target margin: `-0.0409`;
    - mean matched-minus-worst-noise target margin: `-0.0495`.
- Interpretation:
  - The current locked stored-probe decoder does not produce usable matched decode
    margins for fresh `has_majority` or `mountain_pattern` subjects under this
    bounded feasibility protocol.
  - This reinforces the evidence-package limitation:
    decode/steering proof remains restricted to
    `sorted_ascending <-> sorted_descending`.
  - This does not imply additional-behavior decoding is impossible.
  - A stronger test would require a preregistered additional-behavior method and a
    fresh final holdout.
- Reviewer outcome:
  - Initial reviewer confidence: `4/5`.
  - Required corrections:
    - avoid or define `zero-shot`;
    - add weak-source caveat;
    - state that this is not an impossibility result.
  - Corrections applied.
  - Final reviewer confidence: `5/5`.
  - Accepted scope:
    - rigorous negative feasibility result for this locked decoder/protocol;
    - not evidence against all possible additional-behavior decoders.

### 2026-06-10 - Evidence Package Audit Update: Additional-Behavior Limitation

- Objective: update the machine-checkable evidence package to include the accepted
  additional-behavior decode feasibility failure.
- Status:
  - Artifact-level verifier passed.
  - Reviewer accepted the updated verifier at `5/5`.
- Verifier:
  - `model_zoo/scripts/audit_muat_evidence_package.py`.
- Output:
  - `runs/muat_evidence_package_audit/results.json`.
- Verification:
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py model_zoo/scripts/evaluate_additional_behavior_decode_feasibility.py`
    passed.
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
- New audit check:
  - `additional_behavior_decode_feasibility_negative`.
- Check details:
  - exact `claim_scope`:
    `fresh_subject_additional_behavior_decode_feasibility_not_proof`;
  - exact `development_status`:
    `feasibility_additional_behavior_no_final_claim`;
  - result must be failed, not passed;
  - exact feasibility thresholds;
  - noise-control count `8`;
  - aggregate n `16`;
  - holdout SHA:
    `03b72098c773690011fa330487e51d08f69c2f3b4558e7ab1ae31ae82f5aeb6b`;
  - individual pass counts:
    - overall `0/16`;
    - `has_majority` `0/8`;
    - `mountain_pattern` `0/8`;
  - both behavior-level mean gates fail;
  - weak `has_majority` source-margin caveat is preserved.
- Evidence-supported claims remain unchanged:
  - stored-probe signatures contain heldout-decodable behavior information for four
    clean behaviors under logistic/RF classifiers with shuffled-label controls;
  - restricted two-behavior functional decoding on fresh final holdout;
  - restricted two-behavior robust steering on fresh final holdout.
- Explicitly not proven remains unchanged:
  - larger models;
  - additional behaviors beyond `sorted_ascending`/`sorted_descending` for
    decode/steering;
  - broad MUAT generality;
  - non-aggressive steering-vector norm requirement.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - updated evidence package correctly incorporates the additional-behavior result
      as a negative feasibility limitation;
    - it does not add a new proof claim.

### 2026-06-10 - Human-Readable Evidence Report

- Objective: create a portable human-readable summary of the current artifact-audited
  evidence package.
- Status:
  - Report completed.
  - Reviewer accepted the report at `5/5`.
- Report:
  - `docs/muat_small_scale_evidence_report.md`.
- Verification:
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py model_zoo/scripts/evaluate_additional_behavior_decode_feasibility.py model_zoo/scripts/run_stored_probe_steering_robustness.py model_zoo/scripts/train_robust_signature_edit_vectors.py`
    passed.
- Report scope:
  - human-readable summary of:
    - four-behavior stored-probe interpretability signal;
    - restricted two-behavior functional decode;
    - restricted two-behavior robust steering;
    - V1 robust steering limitation;
    - additional-behavior decode feasibility failure.
- Report limitations:
  - same small 5x8 architecture;
  - same stored-probe/decoder setup;
  - decode/steering only for `sorted_ascending <-> sorted_descending`;
  - aggressive steering norm caveat;
  - artifact-level audit rather than full rerun.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - faithful human-readable summary of the current artifact-audited MUAT
      small-scale evidence package.

### 2026-06-10 - Evidence Package Checksum Manifest

- Objective: bind the current evidence package files to SHA-256 hashes so artifact
  drift is detectable.
- Status:
  - Manifest generated.
  - Reviewer accepted the manifest at `5/5`.
  - Because `research-log.md` is included, the manifest was regenerated after this
    log entry and reverified.
- Script:
  - `model_zoo/scripts/build_evidence_manifest.py`.
- Manifest:
  - `runs/muat_evidence_package_audit/manifest.json`.
- Verification:
  - `python -m py_compile model_zoo/scripts/build_evidence_manifest.py` passed.
  - `python model_zoo/scripts/build_evidence_manifest.py --verify` passed with zero
    failures after manifest regeneration.
- Manifest scope:
  - `35` curated evidence-package files;
  - report;
  - research log;
  - verifier scripts;
  - final decode/steering artifacts;
  - negative feasibility artifacts;
  - holdout subjects/summaries;
  - decoder checkpoint;
  - v2 edit vectors.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - useful SHA-256 manifest for detecting drift in the listed evidence-package
      files;
    - complements but does not replace `audit_muat_evidence_package.py`.
  - Residual risks:
    - curated file list only;
    - not a signed attestation;
    - does not pin a git commit;
    - checks byte identity, not metric correctness or methodology.

### 2026-06-10 - One-Command Evidence Package Verification

- Objective: provide a single command that runs the current evidence package
  verification sequence without retraining, rerunning experiments, invoking linting,
  or adding new scientific claims.
- Status:
  - Wrapper completed.
  - Reviewer accepted the wrapper at `5/5`.
  - Because `research-log.md` is included in the checksum manifest, the manifest
    must be regenerated and reverified after this log entry.
- Script:
  - `model_zoo/scripts/verify_muat_evidence_package.py`.
- Verification sequence:
  - `python -m py_compile` over the evidence scripts.
  - `python model_zoo/scripts/audit_muat_evidence_package.py`.
  - `python model_zoo/scripts/build_evidence_manifest.py --verify`.
- Fresh results before this log entry:
  - `python model_zoo/scripts/verify_muat_evidence_package.py` passed with zero
    failures.
  - `python model_zoo/scripts/build_evidence_manifest.py --verify` passed with
    `36` checked files and zero failures.
- Accepted scope:
  - orchestration wrapper for the accepted evidence package checks;
  - improves reproducibility of the current verification sequence;
  - does not retrain, rerun experiments, run lint, or create new scientific
    evidence.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Residual risks:
    - validates only curated artifacts and checks already covered by the audit
      and manifest;
    - does not remove narrow-scope limitations around small architecture,
      two-behavior decode/steering, no larger-model evidence, and no broad MUAT
      generality;
    - any future change to `research-log.md` or another tracked file requires
      manifest regeneration before the wrapper result is current.

### 2026-06-10 - Four-Behavior Decoder Preregistration

- Objective: freeze the methodology for the next four-behavior stored-probe decoder
  proof attempt before implementation or final evaluation.
- Status:
  - Preregistration drafted.
  - Reviewer initially returned `4/5`.
  - Required corrections were applied.
  - Reviewer accepted the corrected preregistration at `5/5`.
- Artifact:
  - `docs/preregistrations/four_behavior_stored_probe_decoder_v1.md`.
- Accepted scope:
  - future four-behavior stored-probe decoder proof attempt;
  - fixed `5x8` `SubjectNetwork` setup;
  - same deterministic stored-probe set;
  - methodology only, not new evidence.
- Key frozen requirements:
  - disjoint train, development, and final subject pools;
  - final pool is one-shot and cannot be used for method selection;
  - final subject weights/signatures must be regenerated from stored probes;
  - train-only normalization;
  - minimum final accepted subject count per behavior;
  - every other-label train-centroid control;
  - every different-label other-subject control;
  - per-control-type metrics and pass/fail status;
  - aggregate individual pass rate at least `0.95`;
  - per-behavior individual pass rate at least `0.90`;
  - any final overlap with train/development subjects fails the proof.
- Reviewer outcome:
  - Final reviewer confidence: `5/5`.
  - Required corrections:
    - none remaining.
  - Residual risks:
    - source subject generation may remain uneven across behaviors;
    - thresholds are chosen proof standards, not universal criteria;
    - even a passing result would cover only these four clean behaviors, this
      architecture, this stored-probe setup, and measured behavior/subject-output
      controls.

### 2026-06-10 - Evidence Manifest Update: Preregistration Included

- Objective: include the accepted four-behavior decoder preregistration in the
  checksum manifest.
- Status:
  - Manifest file list updated.
  - Manifest regenerated with `37` files.
  - Reviewer accepted the packaging update at `5/5`.
  - Because `research-log.md` is included in the checksum manifest, the manifest
    was regenerated and reverified after this log entry.
- Updated manifest scope:
  - previous evidence package files;
  - `docs/preregistrations/four_behavior_stored_probe_decoder_v1.md`.
- Verification before this log entry:
  - `python -m py_compile model_zoo/scripts/build_evidence_manifest.py model_zoo/scripts/verify_muat_evidence_package.py`
    passed.
  - `python model_zoo/scripts/build_evidence_manifest.py` passed with `37` files
    and no missing files.
  - `python model_zoo/scripts/verify_muat_evidence_package.py` passed with
    `37` checked files and zero failures.
  - Final direct `python model_zoo/scripts/build_evidence_manifest.py --verify`
    passed with `37` checked files and zero failures.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - integrity coverage now includes the accepted four-behavior decoder
      preregistration artifact;
    - this is not a new scientific result and does not broaden the current
      evidence-supported claims.
  - Residual risk:
    - manifest verifies byte identity for the curated file list only.

### 2026-06-10 - Four-Behavior Source-Generation Feasibility

- Objective: test whether the accepted four-behavior decoder preregistration can
  produce source subjects under support-only training before any decoder training.
- Status:
  - Feasibility script completed.
  - Initial reviewer confidence: `4/5`.
  - Required corrections were applied.
  - Final reviewer confidence: `5/5`.
- Script:
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`.
- Result:
  - `runs/four_behavior_source_generation_feasibility_v1/results.json`.
  - File SHA-256 reported by the script:
    `b988c173d5c31b5944de595e496f82b79e8f55bacad009f9b7bf7a1a79cefe65`.
  - Embedded payload hash:
    `b4d985f0e34cb131529e6ee93b0ef126db0cb7b148af563bb82eee99421f54fd`.
- Protocol:
  - no decoder was trained;
  - support-only source training;
  - `32` positive and `32` negative support cases per behavior;
  - disjoint heldout acceptance cases with `64` positive and `64` negative cases per
    behavior;
  - `n=8` pilot seeds per behavior;
  - `350` epochs;
  - learning rate `0.003`;
  - source heldout margin gate `>= 0.40`.
- Result: failed.
  - Aggregate source-gate pass count: `24/32`.
  - `sorted_ascending`: `8/8`, mean heldout margin `0.9673`, min `0.9085`.
  - `sorted_descending`: `8/8`, mean heldout margin `0.9460`, min `0.8652`.
  - `mountain_pattern`: `8/8`, mean heldout margin `0.8585`, min `0.7657`.
  - `has_majority`: `0/8`, mean heldout margin `0.2239`, min `0.0916`,
    max `0.3221`.
- Interpretation:
  - Under this support-only source-generation protocol, `has_majority` fails the
    preregistered heldout source-margin gate in this pilot.
  - This is source-generation feasibility only, not stored-probe decoder evidence.
  - This is not an impossibility result for `has_majority`; a revised source-generation
    method may still work but should be preregistered before proof use.
- Reviewer outcome:
  - Final reviewer confidence: `5/5`.
  - Required corrections:
    - none remaining.
  - Accepted scope:
    - negative source-generation feasibility result for this support-only protocol;
    - `has_majority` fails `0/8` under the preregistered `0.40` heldout source-margin
      gate while the other three behaviors pass `8/8`.
  - Residual risks:
    - pilot sample only;
    - not decoder evidence;
    - revised `has_majority` source-generation remains possible with a new
      preregistered method.

### 2026-06-10 - Evidence Package Audit Update: Source-Generation Limitation

- Objective: add the accepted four-behavior source-generation feasibility result to
  the machine audit and checksum manifest.
- Status:
  - Audit updated.
  - Manifest updated.
  - Reviewer accepted the package update at `5/5`.
  - Because `research-log.md` is included in the checksum manifest, the manifest
    was regenerated and reverified after this log entry.
- Updated files:
  - `model_zoo/scripts/audit_muat_evidence_package.py`;
  - `model_zoo/scripts/build_evidence_manifest.py`;
  - `model_zoo/scripts/verify_muat_evidence_package.py`;
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`;
  - `runs/four_behavior_source_generation_feasibility_v1/results.json`.
- New audit check:
  - `four_behavior_source_generation_feasibility_negative`.
- Verification before this log entry:
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py model_zoo/scripts/build_evidence_manifest.py model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py model_zoo/scripts/verify_muat_evidence_package.py`
    passed.
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
  - `python model_zoo/scripts/build_evidence_manifest.py` regenerated a `39`-file
    manifest with no missing files.
  - `python model_zoo/scripts/verify_muat_evidence_package.py` passed.
  - Final direct `python model_zoo/scripts/build_evidence_manifest.py --verify`
    passed with `39` checked files and zero failures.
- Evidence-supported claims:
  - unchanged.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - package integrity and audit coverage now include the accepted
      four-behavior source-generation pilot;
    - current evidence-supported claims are not broadened;
    - the new check records a limiting result for this support-only source-generation
      protocol.
  - Residual risks:
    - artifact-level verification only;
    - source-generation pilot is not decoder evidence;
    - source-generation pilot is not an impossibility result for `has_majority`.

### 2026-06-10 - Four-Behavior Source-Generation V2 Expanded Support

- Objective: test whether expanded support-case coverage fixes the `has_majority`
  source-generation blocker from V1 before any decoder training.
- Status:
  - Preregistration accepted by reviewer at `5/5`.
  - V2 pilot completed.
  - Reviewer accepted the result interpretation at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_source_generation_v2_expanded_support.md`.
- Script:
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`.
- Result:
  - `runs/four_behavior_source_generation_v2_expanded_support/results.json`.
  - File SHA-256 reported by the script:
    `39800a4828dc7b83c8c9be5d018ec93bf1d17ef519c351e5f4980aefc80b29bc`.
  - Embedded payload hash:
    `083a75ce4ea763066fb3f0ba349dcd81e4ef51298761bf8e779ccfeba3fc544b`.
- Protocol:
  - no decoder was trained;
  - source training support cases per class increased from `32` to `160`;
  - heldout cases per class remained `64`;
  - support/heldout overlap count: `0`;
  - `n=8` pilot seeds per behavior;
  - `350` epochs;
  - learning rate `0.003`;
  - source heldout margin gate `>= 0.40`.
- Result: failed.
  - Aggregate source-gate pass count: `28/32`.
  - `sorted_ascending`: `8/8`, mean heldout margin `0.9988`, min `0.9937`.
  - `sorted_descending`: `8/8`, mean heldout margin `0.9797`, min `0.9532`.
  - `mountain_pattern`: `8/8`, mean heldout margin `0.9149`, min `0.8164`.
  - `has_majority`: `4/8`, mean heldout margin `0.3492`, min `0.2790`,
    max `0.4140`.
- Comparison to V1:
  - `has_majority` pass count improved from `0/8` to `4/8`.
  - `has_majority` mean heldout margin improved by `+0.1253`.
- Interpretation:
  - Expanded support helps `has_majority` source generation but does not solve the
    source-generation blocker under the preregistered V2 pilot gates.
  - This is source-generation feasibility only, not stored-probe decoder evidence.
  - This is not an impossibility result for `has_majority`.
  - Any next source-generation revision requires a new preregistration.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - negative/limited V2 source-generation feasibility result;
    - expanded support helps but does not clear the preregistered gate.
  - Residual risks:
    - pilot sample only;
    - not decoder evidence;
    - no in-place tuning on this result.

### 2026-06-10 - Evidence Package Audit Update: Source-Generation V2 Limitation

- Objective: add the accepted V2 expanded-support source-generation result to the
  machine audit and checksum manifest.
- Status:
  - Audit updated.
  - Manifest updated.
  - Reviewer accepted the package update at `5/5`.
  - Because `research-log.md` is included in the checksum manifest, the manifest
    was regenerated and reverified after this log entry.
- Updated files:
  - `docs/preregistrations/four_behavior_source_generation_v2_expanded_support.md`;
  - `model_zoo/scripts/audit_muat_evidence_package.py`;
  - `model_zoo/scripts/build_evidence_manifest.py`;
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`;
  - `runs/four_behavior_source_generation_v2_expanded_support/results.json`.
- New audit check:
  - `four_behavior_source_generation_v2_expanded_support_negative`.
- Verification before this log entry:
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py model_zoo/scripts/build_evidence_manifest.py model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py model_zoo/scripts/verify_muat_evidence_package.py`
    passed.
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
  - `python model_zoo/scripts/build_evidence_manifest.py` regenerated a `41`-file
    manifest with no missing files.
  - `python model_zoo/scripts/verify_muat_evidence_package.py` passed.
  - Final direct `python model_zoo/scripts/build_evidence_manifest.py --verify`
    passed with `41` checked files and zero failures.
- Evidence-supported claims:
  - unchanged.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - package audit and integrity coverage now include the accepted V2
      expanded-support source-generation pilot;
    - V2 is treated as a negative/limited source-generation feasibility result,
      not decoder evidence;
    - current evidence-supported claims remain limited to four-behavior
      interpretability, two-behavior decode, and two-behavior robust steering.
  - Residual risks:
    - artifact-level verification only;
    - pilot `n=8`;
    - any next source-generation method revision needs a new preregistration.

### 2026-06-10 - Four-Behavior Source-Generation V3 Full Pool

- Objective: test whether heldout-excluded full-pool predicate-derived source
  training fixes the remaining `has_majority` source-generation blocker before any
  decoder training.
- Status:
  - Preregistration accepted by reviewer at `5/5`.
  - V3 pilot completed.
  - Initial result review returned `4/5`.
  - Required artifact corrections were applied.
  - Reviewer accepted the corrected result at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_source_generation_v3_full_pool.md`.
- Script:
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`.
- Result:
  - `runs/four_behavior_source_generation_v3_full_pool/results.json`.
  - File SHA-256 reported by the corrected script:
    `12da2ca6763b7900d359756dc68286154a4addb42112eeafa12c75ee8d04e4d5`.
  - Embedded payload hash:
    `668895d541a6a46c0ddab15c405b85d66ca76e102917be5e5c7ecff7ac8ebcb1`.
- Protocol:
  - no decoder was trained;
  - enumerated the full finite `10^5` length-`5` digit sequence universe;
  - excluded every sequence used in heldout acceptance cases;
  - selected up to `2048` positives, `1024` hard negatives, and `1024` generic
    negatives per subject;
  - recorded candidate-pool hashes and per-subject selected-case hashes;
  - max selected-training-vs-heldout overlap count: `0`;
  - `n=8` pilot seeds per behavior;
  - `350` epochs;
  - learning rate `0.003`;
  - source heldout margin gate `>= 0.40`.
- Result: failed.
  - Aggregate source-gate pass count: `31/32`.
  - `sorted_ascending`: `8/8`, mean heldout margin `0.9534`, min `0.9012`.
  - `sorted_descending`: `8/8`, mean heldout margin `0.9492`, min `0.8957`.
  - `mountain_pattern`: `8/8`, mean heldout margin `0.9816`, min `0.9529`.
  - `has_majority`: `7/8`, mean heldout margin `0.4574`, min `0.3839`,
    max `0.5199`.
- Comparison:
  - V1 `has_majority`: `0/8`, mean heldout margin `0.2239`.
  - V2 `has_majority`: `4/8`, mean heldout margin `0.3492`.
  - V3 `has_majority`: `7/8`, mean heldout margin `0.4574`.
- Interpretation:
  - Heldout-excluded full-pool source training improves `has_majority` substantially
    but still fails the preregistered strict gate because one `has_majority` subject
    is below `0.40`.
  - This is source-generation feasibility only, not stored-probe decoder evidence.
  - This is not an impossibility result for `has_majority`.
  - Any next source-generation revision requires a new preregistration.
- Reviewer outcome:
  - Final reviewer confidence: `5/5`.
  - Required corrections:
    - none remaining.
  - Accepted scope:
    - rigorous negative/limited V3 source-generation feasibility result;
    - full-pool training improves `has_majority` but does not clear the
      preregistered strict source gate.
  - Residual risks:
    - source-generation only;
    - not decoder evidence;
    - any next source-generation method revision needs a fresh preregistration.

### 2026-06-10 - Evidence Package Audit Update: Source-Generation V3 Limitation

- Objective: add the accepted V3 full-pool source-generation result to the machine
  audit and checksum manifest.
- Status:
  - Audit updated.
  - Manifest updated.
  - Reviewer accepted the package update at `5/5`.
  - Because `research-log.md` is included in the checksum manifest, the manifest
    was regenerated and reverified after this log entry.
- Updated files:
  - `docs/preregistrations/four_behavior_source_generation_v3_full_pool.md`;
  - `model_zoo/scripts/audit_muat_evidence_package.py`;
  - `model_zoo/scripts/build_evidence_manifest.py`;
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`;
  - `runs/four_behavior_source_generation_v3_full_pool/results.json`.
- New audit check:
  - `four_behavior_source_generation_v3_full_pool_negative`.
- Verification before this log entry:
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py model_zoo/scripts/build_evidence_manifest.py model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py model_zoo/scripts/verify_muat_evidence_package.py`
    passed.
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
  - `python model_zoo/scripts/build_evidence_manifest.py` regenerated a `43`-file
    manifest with no missing files.
  - `python model_zoo/scripts/verify_muat_evidence_package.py` passed.
  - Final direct `python model_zoo/scripts/build_evidence_manifest.py --verify`
    passed with `43` checked files and zero failures.
- Evidence-supported claims:
  - unchanged.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - package integrity and audit coverage now include the accepted V3 full-pool
      source-generation pilot;
    - V3 is treated as a negative/limited source-generation feasibility result,
      not decoder or steering evidence;
    - current evidence-supported claims remain unchanged.
  - Residual risks:
    - artifact-level verification only;
    - source-generation only;
    - no additional-behavior decode/steering claim.

### 2026-06-10 - Four-Behavior Source-Generation V4 Accept-Reject

- Objective: test whether deterministic accept-reject collection with the V3
  heldout-excluded full-pool source-training method can collect accepted source
  subjects for all four clean behaviors.
- Status:
  - Preregistration accepted by reviewer at `5/5`.
  - V4 pilot completed.
  - Initial result review returned `4/5`.
  - Required artifact correction was applied.
  - Reviewer accepted the corrected result at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_source_generation_v4_accept_reject.md`.
- Script:
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`.
- Result:
  - `runs/four_behavior_source_generation_v4_accept_reject/results.json`.
  - File SHA-256 reported by the corrected script:
    `6c53a32141285de0299617e6ea34973bbd0927d676ffc997999d0c420fa113f0`.
  - Embedded payload hash:
    `f5944a735151141a0876df8d6929c001f74845347920a92af581ff6827a66a57`.
- Protocol:
  - no decoder was trained;
  - used V3 heldout-excluded full-pool source training;
  - collection mode: `accept_reject`;
  - target accepted subjects per behavior: `8`;
  - max attempts per behavior: `32`;
  - source heldout margin gate `>= 0.40`;
  - recorded all attempted subjects;
  - max selected-training-vs-heldout overlap count: `0`.
- Result: passed.
  - Aggregate source-gate pass count: `32/32`.
  - `sorted_ascending`: `8` accepted in `8` attempts, min heldout margin `0.7557`.
  - `sorted_descending`: `8` accepted in `8` attempts, min heldout margin `0.7936`.
  - `has_majority`: `8` accepted in `8` attempts, min heldout margin `0.4063`.
  - `mountain_pattern`: `8` accepted in `8` attempts, min heldout margin `0.9308`.
  - No rejections occurred under this preregistered seed schedule.
- Interpretation:
  - Under this deterministic V4 seed schedule, the heldout-excluded full-pool method
    collected `8` accepted pilot source subjects for each of the four behaviors.
  - This is source-generation feasibility only, not stored-probe decoder evidence.
  - This does not show that accept-reject was necessary, because no rejections occurred.
  - This unblocks construction of source pools for a separately preregistered decoder
    proof, but does not itself prove decoding.
- Reviewer outcome:
  - Final reviewer confidence: `5/5`.
  - Required corrections:
    - none remaining.
  - Accepted scope:
    - positive source-generation feasibility result only;
    - no decoder, steering, larger-model, or broad MUAT claim.
  - Residual risks:
    - pilot source-generation only;
    - downstream decoder proof still needs disjoint train/development/final pools.

### 2026-06-10 - Evidence Package Audit Update: Source-Generation V4 Feasibility

- Objective: add the accepted V4 accept-reject source-generation result to the
  machine audit and checksum manifest.
- Status:
  - Audit updated.
  - Manifest updated.
  - Reviewer accepted the package update at `5/5`.
  - Because `research-log.md` is included in the checksum manifest, the manifest
    was regenerated and reverified after this log entry.
- Updated files:
  - `docs/preregistrations/four_behavior_source_generation_v4_accept_reject.md`;
  - `model_zoo/scripts/audit_muat_evidence_package.py`;
  - `model_zoo/scripts/build_evidence_manifest.py`;
  - `model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py`;
  - `runs/four_behavior_source_generation_v4_accept_reject/results.json`.
- New audit check:
  - `four_behavior_source_generation_v4_accept_reject_positive`.
- Verification before this log entry:
  - `python -m py_compile model_zoo/scripts/audit_muat_evidence_package.py model_zoo/scripts/build_evidence_manifest.py model_zoo/scripts/evaluate_four_behavior_source_generation_feasibility.py model_zoo/scripts/verify_muat_evidence_package.py`
    passed.
  - `python model_zoo/scripts/audit_muat_evidence_package.py` passed with zero
    failures.
  - `python model_zoo/scripts/build_evidence_manifest.py` regenerated a `45`-file
    manifest with no missing files.
  - `python model_zoo/scripts/verify_muat_evidence_package.py` passed.
  - Final direct `python model_zoo/scripts/build_evidence_manifest.py --verify`
    passed with `45` checked files and zero failures.
- Evidence-supported claims:
  - unchanged.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Required corrections:
    - none.
  - Accepted scope:
    - package integrity and audit coverage now include the accepted V4
      source-generation pilot;
    - V4 records source subjects suitable for future separately preregistered
      four-behavior decoder work;
    - V4 does not prove four-behavior decoding.
  - Residual risks:
  - artifact-level verification only;
  - source-generation only;
  - no decoder or steering evidence.

### 2026-06-10 - Four-Behavior Decoder Source Pools V1 Failed Attempt

- Objective: construct disjoint train/development/final source-subject pools for
  the preregistered four-behavior stored-probe decoder proof.
- Status:
  - V1 source-pool generation completed.
  - Combined audit failed.
  - Reviewer accepted the failed-result interpretation at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_decoder_source_pools_v1.md`.
- Script:
  - `model_zoo/scripts/generate_four_behavior_decoder_source_pools.py`.
- Result artifacts:
  - `runs/four_behavior_decoder_source_pools_v1/combined_audit.json`;
  - `runs/four_behavior_decoder_source_pools_v1/final_redacted_audit.json`.
- Result: failed.
  - Accepted counts met the target:
    - train: `64` accepted subjects per behavior;
    - development: `24` accepted subjects per behavior;
    - final: `24` accepted subjects per behavior.
  - Selected-training-vs-heldout overlap count was `0` in each pool.
  - Accepted seed disjointness failed:
    - train/development accepted seed overlap: `71`;
    - train/final accepted seed overlap: `47`;
    - development/final accepted seed overlap: `67`.
- Root cause:
  - V1 pool base seeds differed by `10000`, which was also the behavior-index
    stride in the seed schedule.
  - This made cross-pool seed ranges overlap across adjacent behavior indices.
- Interpretation:
  - The V1 pools are not usable for decoder-proof training/evaluation.
  - The only positive observation is that accepted counts and source-margin gates
    were sufficient under the generated attempts.
  - The preregistered cross-pool seed-disjointness gate failed, so no
    four-behavior decoder source-pool proof artifact was established.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - rigorous negative source-pool construction checkpoint.
  - Additional reviewer finding:
    - the V1 `final_redacted_audit.json` exposed more final-pool detail than the
      minimal final-pool audit policy should allow for a future valid proof
      artifact.
    - Because V1 already failed and is invalid for proof use, this does not
      change the V1 conclusion.
    - The next preregistered attempt must restrict final public audit fields to
      pass/fail, accepted counts, overlap counts, file/payload hashes,
      stored-probe hash, behavior-suite hashes, and config hash.

### 2026-06-10 - Four-Behavior Decoder Source Pools V2

- Objective: construct disjoint train/development/final source-subject pools for
  the preregistered four-behavior stored-probe decoder proof after the V1 seed
  schedule failure.
- Status:
  - V2 preregistration accepted by reviewer at `5/5` after correction.
  - V2 source-pool generation completed.
  - Reviewer accepted the result at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_decoder_source_pools_v2.md`.
- Script:
  - `model_zoo/scripts/generate_four_behavior_decoder_source_pools.py`.
- Result artifacts:
  - `runs/four_behavior_decoder_source_pools_v2/combined_audit.json`;
  - `runs/four_behavior_decoder_source_pools_v2/final_redacted_audit.json`;
  - `runs/four_behavior_decoder_source_pools_v2/train_subjects.json`;
  - `runs/four_behavior_decoder_source_pools_v2/development_subjects.json`;
  - `runs/four_behavior_decoder_source_pools_v2/final_subjects.json`.
- Protocol:
  - source-training method unchanged from the accepted V4 source-generation pilot;
  - seed schedule: `base_seed + behavior_index * 100000 + attempt_index`;
  - train base seed: `20300000`;
  - development base seed: `21300000`;
  - final base seed: `22300000`;
  - source heldout margin gate: `>= 0.40`;
  - final raw pool is sealed for future one-shot decoder evaluation.
- Result: passed.
  - Combined audit `passed: true`.
  - Combined audit failures: `[]`.
  - Seed preflight passed with no overlapping configured seed ranges.
  - Accepted counts:
    - train: `64` accepted subjects per behavior;
    - development: `24` accepted subjects per behavior;
    - final: `24` accepted subjects per behavior.
  - Train attempts used:
    - `sorted_ascending`: `64`;
    - `sorted_descending`: `64`;
    - `has_majority`: `77`;
    - `mountain_pattern`: `64`.
  - Development attempts used:
    - `sorted_ascending`: `24`;
    - `sorted_descending`: `24`;
    - `has_majority`: `27`;
    - `mountain_pattern`: `24`.
  - Final attempt counts are sealed and are not exposed in public audit artifacts.
  - All cross-pool accepted overlaps are `0` for:
    - seed;
    - subject id;
    - weight hash;
    - signature hash.
  - Max selected-training-vs-heldout overlap count is `0`.
- File hashes from the combined audit:
  - train:
    `3794b1ab0b0013cd91323945fb6f967fde78860d4cac0040b106d669fa91a36d`;
  - development:
    `958b77dab1f18339bd4f96b229f7632402f00e02aaa35f36784d1b49aeef0813`;
  - final:
    `a6e7467c3946f0499422a4003176de7ab8c790b4a3642466e975b312f9fe8111`.
- Interpretation:
  - This is a positive source-pool construction checkpoint only.
  - It establishes disjoint source pools satisfying the source gates for a future
    four-behavior decoder proof attempt.
  - It is not stored-probe decoder evidence, steering evidence, larger-model
    evidence, or broad MUAT evidence.
  - The final pool must not be passed to train/development scripts or inspected
    manually before the final decoder evaluation.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - positive source-pool construction checkpoint only.
  - Residual risk:
    - procedural final-pool sealing must be preserved until one-shot final decoder
      evaluation.

### 2026-06-10 - Four-Behavior Decoder Development V1

- Objective: train and evaluate a direct stored-probe signature-to-weight decoder
  on the four-behavior train/development pools without reading the sealed final
  raw pool.
- Status:
  - Development preregistration accepted by reviewer at `5/5`.
  - Helper tests were written before implementation and initially failed because
    `train_four_behavior_decoder_development.py` did not exist.
  - Full development run completed.
  - Reviewer accepted the result interpretation at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_decoder_development_v1.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_decoder_development.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_decoder_development_helpers.py`.
- Result artifacts:
  - `runs/four_behavior_decoder_development_v1/results.json`;
  - `runs/four_behavior_decoder_development_v1/model.pt`.
- Protocol:
  - direct MLP decoder from normalized stored-probe signatures to normalized flat
    weights;
  - train-only signature/weight normalization;
  - no behavior label input;
  - full best-control development scoring for checkpoint selection;
  - final raw pool remained sealed.
- Result: failed.
  - `passed: false`.
  - Best epoch: `25`.
  - Development aggregate `n`: `96`.
  - Individual all-gate pass count: `0/96`.
  - Mean matched target margin: `0.0045465377`.
  - Mean matched-minus-best-control target margin: `-0.4180624041`.
  - Mean best-control-minus-matched subject-output MSE: `-54.8715616030`.
  - Mean matched reconstruction MSE: `1.3148743622`.
  - Every behavior had `0/24` individual pass count and failed all development
    gates.
- Leakage/path audit:
  - opened raw pools:
    - train source pool;
    - development source pool;
  - opened final-related artifacts:
    - combined audit;
    - final redacted audit;
  - `final_subjects.json` was not opened and is not named in the result JSON.
  - train/development overlap counts were all `0` for seed, subject id,
    weight hash, and signature hash.
- Interpretation:
  - This is a negative train/development decoder result for this direct MLP
    configuration.
  - It blocks final evaluation under the preregistration.
  - It is not final decoder proof evidence and does not show four-behavior
    decoding is impossible.
  - Any architecture, objective, or control change after observing this result
    requires a new preregistration before another development run.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - negative train/development decoder checkpoint only.

### 2026-06-11 - Four-Behavior Decoder Development V2

- Objective: test an adaptive second train/development decoder method using
  train-only functional distillation after the V1 direct MLP decoder failed.
- Status:
  - V2 development preregistration accepted by reviewer at `5/5`.
  - Helper tests were written before implementation and initially failed because
    `train_four_behavior_decoder_development_v2.py` did not exist.
  - Full V2 development run completed.
  - Reviewer accepted the result interpretation at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_decoder_development_v2.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_decoder_development_v2.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_decoder_development_v2_helpers.py`.
- Result artifacts:
  - `runs/four_behavior_decoder_development_v2/results.json`;
  - `runs/four_behavior_decoder_development_v2/model.pt`.
- Protocol:
  - adaptive V2 method after V1 failure;
  - direct MLP decoder from normalized stored-probe signatures to normalized flat
    weights;
  - train-only signature/weight normalization;
  - train-only functional distillation on `4096` deterministic universe cases
    excluding all behavior-suite heldout sequences;
  - no behavior label input;
  - full best-control development scoring for checkpoint selection;
  - final raw pool remained sealed.
- Result: failed.
  - `passed: false`.
  - Best epoch: `50`.
  - Development aggregate `n`: `96`.
  - Individual all-gate pass count: `0/96`.
  - Mean matched target margin: `0.4117191500`.
  - Mean matched-minus-best-control target margin: `-0.3154511039`.
  - Mean best-control-minus-matched subject-output MSE: `-27.4087092181`.
  - Mean matched reconstruction MSE: `2.1119750018`.
  - Per-behavior matched target margins:
    - `sorted_ascending`: `0.6703819491`;
    - `sorted_descending`: `0.3970812022`;
    - `has_majority`: `0.0833433951`;
    - `mountain_pattern`: `0.4960700536`.
  - Every behavior had `0/24` individual pass count.
  - Controls beating matched were mostly same-label train subjects, same-label
    centroids, and noise controls.
- Leakage/path audit:
  - opened raw pools:
    - train source pool;
    - development source pool;
  - opened final-related artifacts:
    - combined audit;
    - final redacted audit;
  - `final_subjects.json` was not opened and is not named in the result JSON.
  - train/development overlap counts were all `0` for seed, subject id,
    weight hash, and signature hash.
  - helper tests confirmed distillation cases exclude all behavior-suite heldout
    sequences.
- Interpretation:
  - This is a negative adaptive train/development decoder result.
  - It shows useful limited progress over V1 on matched target behavior, but it
    fails the control-specificity and subject-specificity gates.
  - It blocks final evaluation under the preregistration.
  - It is not final decoder proof evidence and does not show four-behavior
    decoding is impossible.
  - Any further architecture, objective, or control change after observing this
    result requires a new preregistration before another development run.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - negative adaptive V2 train/development decoder checkpoint only.

### 2026-06-11 - Four-Behavior Decoder Development V3 Signature Inversion

- Objective: test an adaptive per-subject signature-inversion decoder after the
  V1 and V2 decoder-development failures.
- Status:
  - V3 development preregistration accepted by reviewer at `5/5` after control
    corrections.
  - Helper tests were written before implementation and initially failed because
    `train_four_behavior_decoder_development_v3_signature_inversion.py` did not
    exist.
  - Full V3 development run completed.
  - Initial result review returned `4/5` because per-control deltas were
    recoverable but not explicitly reported.
  - The artifact reporting issue was corrected and the full V3 run was
    regenerated.
  - Reviewer accepted the corrected result interpretation at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_decoder_development_v3_signature_inversion.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_decoder_development_v3_signature_inversion.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_decoder_development_v3_helpers.py`.
- Result artifacts:
  - `runs/four_behavior_decoder_development_v3_signature_inversion/results.json`;
  - `runs/four_behavior_decoder_development_v3_signature_inversion/decoded_weights.pt`.
- Protocol:
  - adaptive V3 method after V1/V2 failures;
  - differentiable stored-probe signature extractor matched to the registered
    extractor by helper test;
  - train-only centroid behavior inference;
  - train-only nearest-neighbor initialization in normalized signature space;
  - per-query weight optimization for matched signatures and proof-critical
    signature controls;
  - same V3 inversion pipeline for null, noise, and centroid controls;
  - final raw pool remained sealed.
- Result: failed.
  - `passed: false`.
  - Development aggregate `n`: `96`.
  - Individual all-gate pass count: `0/96`.
  - Inferred behavior accuracy: `0.78125`.
  - Mean matched target margin: `0.2400836338`.
  - Mean matched-minus-best-control target margin: `-0.6036293377`.
  - Mean best-control-minus-matched subject-output MSE: `-141.8004977169`.
  - Mean matched signature MSE: `1.0847266445`.
  - Every behavior had `0/24` individual pass count.
  - Same-label train controls and nearest-train signature-neighbor controls
    dominated the matched inversion outputs.
- Leakage/path/control audit:
  - opened raw pools:
    - train source pool;
    - development source pool;
  - opened final-related artifacts:
    - combined audit;
    - final redacted audit;
  - `final_subjects.json` was not opened and is not named in the result JSON.
  - train/development overlap counts were all `0` for seed, subject id,
    weight hash, and signature hash.
  - V3 inverted `3744` matched/control signatures.
  - Each development subject had `43` controls, including `32` V3-inverted noise
    controls.
  - Every control explicitly reports matched-minus-control target margin and
    control-minus-matched subject-output MSE.
- Interpretation:
  - This is a negative adaptive train/development decoder result.
  - It shows the differentiable inversion machinery ran and produced nonzero
    target behavior, but behavior inference and specificity/control gates failed.
  - It blocks final evaluation under the preregistration.
  - It is not final decoder proof evidence and does not show four-behavior
    decoding is impossible.
  - Any further architecture, objective, or control change after observing this
    result requires a new preregistration before another development run.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - negative adaptive V3 signature-inversion decoder checkpoint only.

### 2026-06-11 - Four-Behavior Representation Steering V1

- Objective: test whether fixed stored-probe signatures can be steered across
  all four clean behaviors in representation space under train-only edit
  vectors, train-only evaluators, fresh steering-specific source pools, and
  strict controls.
- Status:
  - Preregistration accepted by reviewer at `5/5` after corrections.
  - Helper tests were written before implementation and initially failed because
    `train_four_behavior_representation_steering.py` did not exist.
  - Steering-specific source-pool construction completed and was accepted by
    reviewer at `5/5` as source-pool construction only.
  - Development run completed under the frozen V1 method.
  - Initial development review returned `4/5` because failure messages used
    misleading inequality wording.
  - Failure-message formatting was corrected, a regression test was added, and
    the deterministic development run was regenerated.
  - Reviewer accepted that development result at `5/5`.
  - A later implementation audit found that the training objective's
    centroid-improvement term was not actually no-edit-relative as
    preregistered, even though evaluation gates used the correct concept.
  - Added a failing regression for no-edit-relative centroid improvement,
    corrected the objective, and regenerated the deterministic development
    result.
  - Reviewer accepted the bug discovery and corrected rerun at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v1.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_representation_steering.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_representation_steering_helpers.py`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v1_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v1_pools/final_redacted_audit.json`.
- Development result artifacts:
  - `runs/four_behavior_representation_steering_v1/development_results.json`;
  - `runs/four_behavior_representation_steering_v1/edit_vectors.pt`.
- Protocol:
  - representation-space steering only;
  - fresh steering-specific train/development/final source pools;
  - decoder final raw pool remained sealed and unused;
  - steering final raw pool was generated and then remained sealed;
  - train-only signature normalization, centroids, and affine primary evaluator;
  - twelve ordered source-target edit vectors initialized as train-centroid
    deltas;
  - development go/no-go gates matched the final proof gates except for pool
    identity.
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0` for seed, subject id, weight hash,
    and signature hash;
  - final redacted audit exposed only accepted counts and max selected-training
    overlap, plus hashes/config metadata.
- Superseded flawed development artifact:
  - The earlier V1 development artifact had `55/288` individual passes, but was
    trained with a flawed centroid-improvement objective.
  - It is superseded by the corrected no-edit-relative rerun and must not be
    treated as the accepted V1 development result.
- Corrected development result: failed.
  - `passed: false`.
  - Development aggregate `n`: `288`.
  - Individual all-gate pass count: `16/288`.
  - Individual all-gate pass rate: `0.0555555556`.
  - Mean matched primary target margin: `101.2225835986`.
  - Mean matched-minus-best-control primary target margin: `60.9470050931`.
  - Mean matched centroid improvement: `0.2559432056`.
  - Mean matched-minus-best-control centroid improvement:
    `-1.0535927349`.
  - Mean source primary margin change: `-177.3604026188`.
  - Per-target individual pass rates:
    - `sorted_ascending`: `3/72`;
    - `sorted_descending`: `0/72`;
    - `has_majority`: `9/72`;
    - `mountain_pattern`: `4/72`.
- Leakage/path audit:
  - development evaluated only
    `runs/four_behavior_representation_steering_v1_pools/development_subjects.json`;
  - development used the final redacted audit only;
  - result text does not name the decoder final raw path or steering final raw
    path;
  - every development record retained `32` random norm-matched controls.
- Interpretation:
  - This is a negative development checkpoint for the frozen V1
    representation-steering method.
  - It shows strong primary-evaluator movement and source suppression, but it
    fails centroid-control specificity and individual/pass-rate gates.
  - It blocks final steering evaluation under the preregistration.
  - It is not final steering evidence, not four-behavior functional decoding
    evidence, and not evidence against all possible representation-steering
    methods.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - negative four-behavior representation-steering V1 development checkpoint
      only.

### 2026-06-11 - Four-Behavior Representation Steering V2 Centroid Delta

- Objective: test whether exact train-only behavior centroid deltas in fixed
  stored-probe signature space can steer heldout source representations across
  all four clean behaviors under fresh V2 pools and preregistered controls.
- Status:
  - V2 preregistration accepted by reviewer at `5/5`.
  - V2 implementation accepted by reviewer at `5/5`.
  - Fresh V2 source-pool construction accepted by reviewer at `5/5`.
  - V2 development result accepted by reviewer at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v2_centroid_delta.md`.
- Failure diagnosis:
  - `docs/representation_steering_v2_failure_diagnosis.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v2_centroid_delta.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_representation_steering_v2_helpers.py`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v2_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v2_pools/final_redacted_audit.json`.
- Development result artifacts:
  - `runs/four_behavior_representation_steering_v2_centroid_delta/development_results.json`;
  - `runs/four_behavior_representation_steering_v2_centroid_delta/centroid_delta_vectors.pt`.
- Protocol:
  - exact centroid-delta vectors:
    `centroid[target] - centroid[source]`;
  - no learned edit-vector optimizer;
  - fresh V2 train/development/final source pools;
  - V2 final raw pool remained sealed;
  - train-only signature normalization, centroids, and affine primary
    evaluator;
  - controls included no edit, null vector, reverse centroid delta,
    same-source other-target centroid deltas, same-target other-source centroid
    deltas, deterministic shuffled direction, and `32` random norm-matched
    vectors.
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0` for seed, subject id, weight
    hash, and signature hash;
  - final redacted audit exposed only allowed aggregate/hash fields.
- Development result: failed.
  - `passed: false`.
  - Development aggregate `n`: `288`.
  - Individual all-gate pass count: `142/288`.
  - Individual all-gate pass rate: `0.4930555556`.
  - Mean matched primary target margin: `47.6160739958`.
  - Mean matched-minus-best-control primary target margin: `24.2469312698`.
  - Mean matched centroid improvement: `1.5253242254`.
  - Mean matched-minus-best-control centroid improvement: `0.4016633564`.
  - Mean source primary margin change: `-116.9670098159`.
  - Every target missed the `0.80` pass-rate gate.
  - Every ordered direction missed the `0.90` pass-rate gate.
- Leakage/path audit:
  - development evaluated only
    `runs/four_behavior_representation_steering_v2_pools/development_subjects.json`;
  - development used the V2 final redacted audit only;
  - result text does not name any `final_subjects.json` path;
  - every development record retained `32` random norm-matched controls.
- Interpretation:
  - This is a negative development checkpoint for the frozen V2
    centroid-delta representation-steering method.
  - It shows aggregate representation movement, but not proof-grade reliable
    four-behavior steering under the registered per-record/target/direction
    gates.
  - The accepted failure diagnosis identifies source-specificity against
    same-target other-source centroid controls as the main next-method
    bottleneck.
  - It blocks final steering evaluation under the V2 preregistration.
  - It is not final steering evidence, not four-behavior functional decoding
    evidence, and not evidence against all possible representation-steering
    methods.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - negative four-behavior representation-steering V2 development checkpoint
      only.

### 2026-06-11 - Four-Behavior Representation Steering V3 Diagonal Transport

- Objective: test whether train-only diagonal covariance transport can improve
  four-behavior representation steering beyond V2 centroid deltas while
  preserving source-specificity under the full control suite.
- Status:
  - V3 preregistration accepted by reviewer at `5/5` after tightening the
    random-control, shuffled-control, and source-pool contracts.
  - V3 implementation accepted by reviewer at `5/5`.
  - Fresh V3 source-pool construction accepted by reviewer at `5/5`.
  - V3 development result accepted by reviewer at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v3_diagonal_transport.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v3_diagonal_transport.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_representation_steering_v3_helpers.py`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v3_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v3_pools/final_redacted_audit.json`.
- Development result artifacts:
  - `runs/four_behavior_representation_steering_v3_diagonal_transport/development_results.json`;
  - `runs/four_behavior_representation_steering_v3_diagonal_transport/diagonal_transport_stats.pt`.
- Protocol:
  - closed-form train-only diagonal covariance transport:
    `centroid[target] + ratio[source,target] * (z - centroid[source])`;
  - no learned edit-vector optimizer;
  - fresh V3 train/development/final source pools;
  - V3 final raw pool remained sealed;
  - train-only signature normalization, centroids, diagonal standard deviations,
    and affine primary evaluator;
  - controls included no edit, null vector, V2 centroid delta, reverse diagonal
    transport, same-source other-target diagonal transports, same-target
    other-source diagonal transports, deterministic shuffled diagonal
    transport, and `32` random norm-matched vectors.
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0` for seed, subject id, weight hash,
    and signature hash;
  - final redacted audit exposed only allowed aggregate/hash fields.
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - Development aggregate `n`: `288`.
  - Individual all-gate pass count: `30/288`.
  - Individual all-gate pass rate: `0.1041666667`.
  - Mean matched primary target margin: `45.6755528839`.
  - Mean matched centroid improvement: `1.8963179257`.
  - Mean matched-minus-V2-centroid-delta primary target margin:
    `1.9002874485`.
  - Mean matched-minus-V2-centroid-delta centroid improvement:
    `0.4344081978`.
  - Mean matched-minus-best-control primary target margin: `-3.9445673236`.
  - Mean matched-minus-best-control centroid improvement: `-0.9124838478`.
  - Mean source primary margin change: `-109.6813265847`.
  - Every target missed the `0.80` pass-rate gate.
  - Every ordered direction missed the `0.90` pass-rate gate.
- Leakage/path audit:
  - development evaluated only
    `runs/four_behavior_representation_steering_v3_pools/development_subjects.json`;
  - development used the V3 final redacted audit only;
  - result text does not name any `final_subjects.json` path;
  - every development record retained `41` controls, including `32` random
    norm-matched controls.
- Interpretation:
  - This is a negative development checkpoint for the frozen V3 diagonal
    transport representation-steering method.
  - It shows strong aggregate target movement and beats the V2 centroid-delta
    baseline on mean primary and centroid metrics, but it fails proof-grade
    source-specificity against the full best-control set.
  - The best primary control was usually `v2_centroid_delta`, and the best
    centroid controls were split across V2 centroid delta and other diagonal
    transports.
  - It blocks final steering evaluation under the V3 preregistration.
  - It is not final steering evidence, not four-behavior functional decoding
    evidence, and not evidence against all possible representation-steering
    methods.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - valid negative V3 diagonal-transport development checkpoint only.

### 2026-06-11 - Four-Behavior Representation Steering V4 Low-Rank Residual Transport

- Objective: test whether train-only low-rank residual covariance transport can
  improve four-behavior representation steering beyond V2 centroid deltas and
  V3 diagonal transport while preserving source-specificity under the full
  control suite.
- Status:
  - V4 preregistration accepted by reviewer at `5/5` after tightening the
    final raw validation language.
  - V4 implementation accepted by reviewer at `5/5` after adding raw-pool hash
    binding and stricter redacted-final leak checks.
  - Fresh V4 source-pool construction accepted by reviewer at `5/5`.
  - V4 development result accepted by reviewer at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v4_low_rank_residual_transport.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v4_low_rank_residual_transport.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_representation_steering_v4_helpers.py`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v4_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v4_pools/final_redacted_audit.json`.
- Development result artifacts:
  - `runs/four_behavior_representation_steering_v4_low_rank_residual_transport/development_results.json`;
  - `runs/four_behavior_representation_steering_v4_low_rank_residual_transport/low_rank_residual_transport_stats.pt`.
- Protocol:
  - closed-form train-only low-rank residual covariance transport:
    `centroid[target] + U @ sqrt_cov[target] @ inv_sqrt_cov[source] @ U.T @ (z - centroid[source])`;
  - no learned edit-vector optimizer;
  - fresh V4 train/development/final source pools;
  - V4 final raw pool remained sealed;
  - train-only signature normalization, centroids, PCA basis, residual
    covariances, and affine primary evaluator;
  - controls included no edit, null vector, V2 centroid delta, V3 diagonal
    transport, reverse low-rank residual transport, same-source other-target
    low-rank transports, same-target other-source low-rank transports,
    deterministic shuffled low-rank transport, and `32` random norm-matched
    vectors.
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0` for seed, subject id, weight hash,
    and signature hash;
  - final redacted audit exposed only allowed aggregate/hash fields.
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - Development aggregate `n`: `288`.
  - Individual all-gate pass count: `42/288`.
  - Individual all-gate pass rate: `0.1458333333`.
  - Mean matched primary target margin: `46.5516936464`.
  - Mean matched centroid improvement: `4.4940608740`.
  - Mean matched-minus-V2-centroid-delta primary target margin:
    `-0.0098890116`.
  - Mean matched-minus-V2-centroid-delta centroid improvement:
    `3.2327019506`.
  - Mean matched-minus-V3-diagonal-transport primary target margin:
    `-1.2330002536`.
  - Mean matched-minus-V3-diagonal-transport centroid improvement:
    `3.0668179062`.
  - Mean matched-minus-best-control primary target margin: `-14.2722247789`.
  - Mean matched-minus-best-control centroid improvement: `-1.7965679599`.
  - Mean source primary margin change: `-108.6704917749`.
  - Every target missed the `0.80` pass-rate gate.
  - Every ordered direction missed the `0.90` pass-rate gate.
- Leakage/path audit:
  - development evaluated only
    `runs/four_behavior_representation_steering_v4_pools/development_subjects.json`;
  - development used the V4 final redacted audit only;
  - result text does not name any `final_subjects.json` path;
  - every development record retained `42` controls, including `32` random
    norm-matched controls.
- Interpretation:
  - This is a negative development checkpoint for the frozen V4 low-rank
    residual transport representation-steering method.
  - It improves over V2 and V3 on centroid metrics but not on primary
    target-margin specificity.
  - The best primary control was split across V2 centroid delta, V3 diagonal
    transport, and same-target other-source low-rank residual transports.
  - The best centroid controls were dominated by same-source other-target and
    same-target other-source low-rank residual transports.
  - It blocks final steering evaluation under the V4 preregistration.
  - It is not final steering evidence, not four-behavior functional decoding
    evidence, and not evidence against all possible representation-steering
    methods.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - valid negative V4 low-rank residual-transport development checkpoint only.

### 2026-06-11 - Four-Behavior Representation Steering V5 Contrastive Residual Calibration

- Objective: test whether train-only contrastive residual calibration on top of
  V4 low-rank transport can fix the source-specificity failures that beat V4
  while preserving target movement.
- Status:
  - V5 preregistration accepted by reviewer at `5/5` after freezing gradient
    policy, loss aggregation, deterministic controls, null control semantics,
    and V2/V3/V4 baseline formulas.
  - V5 implementation accepted by reviewer at `5/5` after adding fail-closed
    source-pool contract handling and current-artifact final authorization.
  - Fresh V5 source-pool construction accepted by reviewer at `5/5`.
  - V5 development result accepted by reviewer at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v5_contrastive_residual_calibration.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v5_contrastive_residual_calibration.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_representation_steering_v5_helpers.py`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v5_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v5_pools/final_redacted_audit.json`.
- Development result artifacts:
  - `runs/four_behavior_representation_steering_v5_contrastive_residual_calibration/development_results.json`;
  - `runs/four_behavior_representation_steering_v5_contrastive_residual_calibration/contrastive_residual_calibration_stats.pt`.
- Protocol:
  - train-only V4 low-rank transport recomputed from V5 train subjects;
  - learned `12 x 48` PCA-subspace calibration coefficients;
  - train-time ranking controls were detached, so ranking gradients updated only
    the matched source-target coefficient;
  - full-batch final-epoch training, no development-selected checkpoint;
  - fresh V5 train/development/final source pools;
  - V5 final raw pool remained sealed;
  - controls included no edit, null vector, V2 centroid delta, V3 diagonal
    transport, uncalibrated V4 low-rank transport, reverse V5, same-source
    other-target V5, same-target other-source V5, deterministic shuffled V5,
    and `32` random norm-matched vectors.
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0` for seed, subject id, weight hash,
    and signature hash;
  - final redacted audit exposed only allowed aggregate/hash fields.
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - Development aggregate `n`: `288`.
  - Individual all-gate pass count: `20/288`.
  - Individual all-gate pass rate: `0.0694444444`.
  - Mean matched primary target margin: `121.8755627208`.
  - Mean source primary margin change: `-170.5525876068`.
  - Mean matched centroid improvement: `4.1053234041`.
  - Mean matched-minus-best-control primary target margin: `9.9655429009`.
  - Mean matched-minus-best-control centroid improvement: `-2.5652369327`.
  - Mean matched-minus-V2-centroid-delta primary target margin:
    `75.2867757827`.
  - Mean matched-minus-V3-diagonal-transport primary target margin:
    `74.9781137043`.
  - Mean matched-minus-V4-low-rank primary target margin: `79.1476116118`.
  - Mean matched-minus-V4-low-rank centroid improvement: `-1.0327701767`.
  - Every target missed the `0.80` pass-rate gate.
  - Every ordered direction missed the `0.90` pass-rate gate.
- Leakage/path audit:
  - development evaluated only
    `runs/four_behavior_representation_steering_v5_pools/development_subjects.json`;
  - development used the V5 final redacted audit only;
  - result text does not name any `final_subjects.json` path;
  - every development record retained `43` controls, including `32` random
    norm-matched controls.
- Interpretation:
  - This is a negative development checkpoint for the frozen V5 contrastive
    residual calibration representation-steering method.
  - It greatly improves primary target-margin movement and source suppression,
    and beats V2/V3/V4 on mean primary target-margin metrics.
  - It fails proof-grade reliability and centroid best-control specificity.
  - V4 remains stronger on mean centroid improvement.
  - The best primary control was usually same-target other-source V5 calibrated
    transport, while the best centroid control was usually uncalibrated V4
    low-rank transport.
  - It blocks final steering evaluation under the V5 preregistration.
  - It is not final steering evidence, not four-behavior functional decoding
    evidence, and not evidence against all possible representation-steering
    methods.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - valid negative V5 contrastive residual-calibration development checkpoint
      only.

### 2026-06-11 - Four-Behavior Representation Steering V6 Centroid-Constrained Primary Correction

- Objective: test whether per-example train-only centroid-constrained primary
  correction can retain V4's target-centroid geometry while adding enough
  primary classifier margin and source suppression to beat V2/V3/V4/V5,
  non-matched V6, shuffled, and random controls.
- Status:
  - V6 preregistration accepted by reviewer at `5/5` after clarifying plain SGD
    with no momentum or weight decay and projected `q` overwrite after every
    step.
  - V6 implementation accepted by reviewer at `5/5` after correcting the
    aggregate best-control centroid threshold to match the preregistered `0.05`.
  - Fresh V6 source-pool construction accepted by reviewer at `5/5`.
  - V6 development result accepted by reviewer at `5/5`.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v6_centroid_constrained_primary_correction.md`.
- Script:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v6_centroid_constrained_primary_correction.py`.
- Helper tests:
  - `model_zoo/scripts/test_four_behavior_representation_steering_v6_helpers.py`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v6_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v6_pools/final_redacted_audit.json`.
- Development result artifacts:
  - `runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/development_results.json`;
  - `runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/centroid_constrained_primary_correction_stats.pt`.
- Commands:
  - `python model_zoo/scripts/test_four_behavior_representation_steering_v6_helpers.py`
  - `python -m py_compile model_zoo/scripts/train_four_behavior_representation_steering_v6_centroid_constrained_primary_correction.py model_zoo/scripts/test_four_behavior_representation_steering_v6_helpers.py`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v6_centroid_constrained_primary_correction.py --phase generate-pools`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v6_centroid_constrained_primary_correction.py --phase development`
- Protocol:
  - train-only V4 low-rank transport recomputed from V6 train subjects;
  - per-evaluation-record `48`-dimensional PCA-subspace correction optimized
    for `80` plain-SGD steps;
  - every step projected the capped candidate into the target-centroid ball and
    overwrote `q` with `U.T @ (projected_candidate - v4_uncapped)`;
  - V5 was retrained from V6 train-only statistics as a baseline control, not
    as the matched method;
  - fresh V6 train/development/final source pools;
  - V6 final raw pool remained sealed after development failed;
  - controls included no edit, null vector, V2 centroid delta, V3 diagonal
    transport, uncorrected V4 low-rank transport, V5 contrastive residual
    calibration, reverse V6, same-source other-target V6, same-target
    other-source V6, deterministic shuffled V6, and `32` random norm-matched
    vectors.
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0` for seed, subject id, weight hash,
    and signature hash;
  - final redacted audit exposed only allowed aggregate/hash fields.
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - Development aggregate `n`: `288`.
  - Individual all-gate pass count: `2/288`.
  - Individual all-gate pass rate: `0.0069444444`.
  - Mean matched primary target margin: `45.0174989502`.
  - Mean matched centroid improvement: `5.2879611254`.
  - Mean source primary margin change: `-103.4365548342`.
  - Mean matched-minus-best-control primary target margin: `-63.6636278828`.
  - Mean matched-minus-best-control centroid improvement: `-1.9438294139`.
  - Mean matched-minus-V5-calibrated primary target margin: `-62.6472493344`.
  - Mean matched-minus-V5-calibrated centroid improvement: `0.8543339902`.
  - Mean matched-minus-V4-low-rank primary target margin: `5.9002181060`.
  - Mean matched-minus-V4-low-rank centroid improvement: `0.0930981868`.
  - Every target missed the `0.80` pass-rate gate.
  - Every ordered direction missed the `0.90` pass-rate gate.
- Leakage/path audit:
  - development evaluated only
    `runs/four_behavior_representation_steering_v6_pools/development_subjects.json`;
  - development used the V6 final redacted audit only;
  - result text does not name any `final_subjects.json` path;
  - every development record retained `44` controls, including `32` random
    norm-matched controls.
- Interpretation:
  - This is a negative development checkpoint for the frozen V6
    centroid-constrained primary-correction representation-steering method.
  - It preserves strong primary/centroid movement and beats V2/V3/V4 on
    aggregate mean metrics.
  - It fails proof-grade control specificity and reliability.
  - V5 remains much stronger on primary target-margin controls.
  - It blocks final steering evaluation under the V6 preregistration.
  - It is not final steering evidence, not four-behavior functional decoding
    evidence, and not evidence against all possible representation-steering
    methods.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - valid negative V6 centroid-constrained primary-correction development
      checkpoint only.

### 2026-06-11 - V6 Posthoc Pareto Diagnosis

- Objective: diagnose whether the failed V6 method was usually dominated by a
  single control on both primary target margin and centroid improvement, or
  whether the scalar best-control gates rejected a primary/centroid tradeoff.
- Status:
  - Posthoc diagnosis accepted by reviewer at `5/5` after adding explicit
    development-only status, no-final-authorization status, next-action
    constraints, limitations, and source-artifact SHA-256 binding.
- Artifact:
  - `runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/posthoc_pareto_diagnosis.json`.
- Source artifact:
  - `runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/development_results.json`.
  - SHA-256:
    `4dcb37d961d00dcdb4535208e1caed32961c4719e49f79580719c2960f3c35a3`.
- Scope and limitations:
  - claim scope: `v6_development_posthoc_diagnosis_not_proof`;
  - development status:
    `posthoc_development_only_diagnosis_after_preregistered_v6_failure`;
  - final access status:
    `does_not_authorize_opening_or_evaluating_v6_final_raw`;
  - next action:
    `use_only_to_motivate_fresh_preregistered_v7_design_do_not_open_v6_final_raw`.
- Metrics:
  - Pareto-undominated records: `226/288`.
  - Pareto-undominated rate: `0.7847222222`.
  - Dominated records: `62/288`.
  - Worst direction undominated count: `16/24`.
  - Main dominator counts:
    - same-target other-source V6: `33`;
    - V5 contrastive residual calibration: `27`;
    - V2 centroid delta: `12`;
    - V3 diagonal transport: `12`;
    - shuffled V6: `5`.
- Interpretation:
  - This is diagnosis only, not proof and not final evidence.
  - V6 is often not Pareto-dominated, so a future V7 could test a
    multiobjective or Pareto-aware representation-space steering claim.
  - Direction reliability is too weak for a gate-only reinterpretation, so any
    V7 must use a new preregistration and fresh pools.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - development-only posthoc diagnosis that motivates future design but does
      not alter the V6 failure or authorize final evaluation.

### 2026-06-11 - V7 Pareto-Frontier Correction

- Objective: test a fresh-pool, preregistered multiobjective
  representation-space steering claim after the V6 posthoc Pareto diagnosis.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v7_pareto_frontier_correction.md`.
  - Reviewer accepted preregistration at `5/5` after tightening final-redaction,
    final-authorization, Pareto-rate, and control definitions.
- Implementation:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v7_pareto_frontier_correction.py`.
  - `model_zoo/scripts/test_four_behavior_representation_steering_v7_helpers.py`.
  - Reviewer accepted implementation at `5/5`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v7_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v7_pools/final_redacted_audit.json`.
  - Reviewer accepted source-pool construction at `5/5`.
- Development artifact:
  - `runs/four_behavior_representation_steering_v7_pareto_frontier_correction/development_results.json`.
- Commands:
  - `python -m py_compile model_zoo/scripts/train_four_behavior_representation_steering_v7_pareto_frontier_correction.py model_zoo/scripts/test_four_behavior_representation_steering_v7_helpers.py`
  - `python model_zoo/scripts/test_four_behavior_representation_steering_v7_helpers.py`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v7_pareto_frontier_correction.py --phase generate-pools`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v7_pareto_frontier_correction.py --phase development`
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0`;
  - final raw pool remained sealed.
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - `n`: `288`.
  - Individual all-gate pass count/rate: `245/288`, `0.8506944444`.
  - Pareto-undominated count/rate: `257/288`, `0.8923611111`.
  - Target-prediction pass count/rate: `278/288`, `0.9652777778`.
  - Mean selected primary target margin: `121.9106095102`.
  - Mean selected centroid improvement: `5.0538051493`.
  - Mean selected-minus-V6 centroid improvement: `0.0154168175`
    against required `> 0.05`.
  - Mean source primary margin change: `-170.7438490523`.
  - Failed gates:
    - aggregate individual pass rate;
    - aggregate Pareto-undominated rate;
    - aggregate selected-minus-V6 centroid improvement;
    - target `has_majority` individual and Pareto rates;
    - weak ordered directions including
      `mountain_pattern_to_has_majority` and
      `mountain_pattern_to_sorted_descending`.
- Diagnosis:
  - Most failed records were Pareto non-domination failures.
  - Selected dominators were mostly
    `same_target_other_source_v7_pareto_frontier_correction`.
- Interpretation:
  - V7 made substantial progress over V6 but missed proof gates.
  - V7 is not proof-grade four-behavior steering evidence.
  - V7 final raw evaluation is blocked.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - valid negative V7 development checkpoint only.

### 2026-06-11 - V8 Source-Conditional Tournament Correction

- Objective: target the V7 failure mode where same-target other-source controls
  dominated selected candidates, using fixed detached V7 same-target tournament
  competitors during V8 candidate optimization.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v8_source_conditional_tournament_correction.md`.
  - Initial reviewer score: `3/5`.
  - Blockers addressed:
    - explicit final redacted allowlist and forbidden final-detail fields;
    - exact hash-bound final authorization fields;
    - explicit Pareto-rate definition and denominators;
    - deterministic shuffled/random control rules;
    - exact tournament competitor count and detach semantics;
    - fixed constants and reporting requirements.
  - Reviewer accepted revised preregistration at `5/5`.
- Implementation:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v8_source_conditional_tournament_correction.py`.
  - `model_zoo/scripts/test_four_behavior_representation_steering_v8_helpers.py`.
  - Reviewer accepted implementation at `5/5`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v8_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v8_pools/final_redacted_audit.json`.
  - Reviewer accepted source-pool construction at `5/5`.
- Development artifact:
  - `runs/four_behavior_representation_steering_v8_source_conditional_tournament_correction/development_results.json`.
- Commands:
  - `python -m py_compile model_zoo/scripts/train_four_behavior_representation_steering_v8_source_conditional_tournament_correction.py model_zoo/scripts/test_four_behavior_representation_steering_v8_helpers.py`
  - `python model_zoo/scripts/test_four_behavior_representation_steering_v8_helpers.py`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v8_source_conditional_tournament_correction.py --phase generate-pools`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v8_source_conditional_tournament_correction.py --phase development`
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0`;
  - final raw pool remained sealed.
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - `n`: `288`.
  - Individual all-gate pass count/rate: `237/288`, `0.8229166667`.
  - Pareto-undominated count/rate: `243/288`, `0.84375`.
  - Target-prediction pass count/rate: `281/288`, `0.9756944444`.
  - Mean selected primary target margin: `129.3434276382`.
  - Mean selected centroid improvement: `6.0069983933`.
  - Mean selected-minus-V5 centroid improvement: `1.6989225083`.
  - Mean selected-minus-V6 primary target margin: `79.5348633743`.
  - Mean selected-minus-V6 centroid improvement: `0.6251051459`.
  - Mean source primary margin change: `-161.3274385167`.
  - Failed gates:
    - aggregate individual pass rate;
    - aggregate Pareto-undominated rate;
    - target Pareto rates for `sorted_ascending`, `sorted_descending`, and
      `mountain_pattern`;
    - six ordered-direction reliability/Pareto gates.
- Diagnosis:
  - V8 improved absolute primary/centroid/source-suppression metrics and cleared
    the V6 centroid-margin issue.
  - It failed proof-grade reliability because V8 same-target controls also
    strengthened.
  - Failed check counts were dominated by `pareto_undominated` (`45` records),
    with a smaller centroid-prediction component (`7` records).
  - Selected dominators were primarily
    `same_target_other_source_v8_source_conditional_tournament_correction`.
- Interpretation:
  - V8 is a negative development checkpoint.
  - The result suggests same-target other-source procedures are alternate target
    steering methods under the registered target-only representation metrics,
    not reliable bad controls.
  - V8 is not proof-grade four-behavior steering evidence.
  - V8 final raw evaluation is blocked.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - valid negative V8 source-conditional tournament development checkpoint
      only.

### 2026-06-11 - V9 Source-Invariant Target-Attractor

- Objective: test whether the V7/V8 same-target other-source failure mode is
  better interpreted as target-attractor transfer, using fresh V9 pools and a
  preregistered split between negative controls and same-target transfer probes.
- Preregistration:
  - `docs/preregistrations/four_behavior_representation_steering_v9_source_invariant_target_attractor.md`.
  - Reviewer accepted preregistration at `5/5`.
- Implementation:
  - `model_zoo/scripts/train_four_behavior_representation_steering_v9_source_invariant_target_attractor.py`.
  - `model_zoo/scripts/test_four_behavior_representation_steering_v9_helpers.py`.
  - Reviewer accepted implementation at `5/5`.
- Source-pool artifacts:
  - `runs/four_behavior_representation_steering_v9_pools/combined_audit.json`;
  - `runs/four_behavior_representation_steering_v9_pools/final_redacted_audit.json`.
  - Reviewer accepted source-pool construction at `5/5`.
- Development artifact:
  - `runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/development_results.json`.
- Final artifact:
  - `runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/final_results.json`.
- Commands:
  - `python -m py_compile model_zoo/scripts/train_four_behavior_representation_steering_v9_source_invariant_target_attractor.py model_zoo/scripts/test_four_behavior_representation_steering_v9_helpers.py`
  - `python model_zoo/scripts/test_four_behavior_representation_steering_v9_helpers.py`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v9_source_invariant_target_attractor.py --phase generate-pools`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v9_source_invariant_target_attractor.py --phase development`
  - `python model_zoo/scripts/train_four_behavior_representation_steering_v9_source_invariant_target_attractor.py --phase final`
- Source-pool result: passed.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0`.
- Development correction:
  - The first V9 development run was invalid because shuffled V9 controls with
    same-target other-source sampled directions were incorrectly classified as
    transfer probes.
  - The inflated transfer-probe count was `3125` instead of the preregistered
    `2880`.
  - A regression test was added:
    `test_v9_shuffled_same_target_direction_remains_negative_control`.
  - The splitter now requires explicit
    `same_target_other_source_v9_source_invariant_target_attractor` control
    type before moving a candidate into `transfer_probes`.
  - Reviewer accepted the corrected implementation at `5/5`; the invalid run
    is excluded from evidence.
- Corrected development result: passed.
  - `passed: true`.
  - `next_action`: `eligible_for_one_shot_final_eval_without_method_changes`.
  - `n`: `288`.
  - Individual all-gate pass count/rate: `277/288`, `0.9618055556`.
  - Pareto-undominated count/rate: `279/288`, `0.96875`.
  - Target-prediction pass count/rate: `286/288`, `0.9930555556`.
  - Same-target transfer probe count: `2880`.
  - Same-target transfer target-prediction count/rate:
    `2735/2880`, `0.9496527778`.
  - Same-target transfer gate-pass count/rate:
    `2735/2880`, `0.9496527778`.
  - Reviewer accepted the corrected development result at `5/5`.
- Final result: passed.
  - `passed: true`.
  - `n`: `288`.
  - Individual all-gate pass count/rate: `278/288`, `0.9652777778`.
  - Pareto-undominated count/rate: `285/288`, `0.9895833333`.
  - Target-prediction pass count/rate: `281/288`, `0.9756944444`.
  - Same-target transfer probe count: `2880`.
  - Same-target transfer target-prediction count/rate: `2664/2880`, `0.925`.
  - Same-target transfer gate-pass count/rate: `2664/2880`, `0.925`.
  - Mean selected primary target margin: `121.8615035945`.
  - Mean selected centroid improvement: `6.0319956938`.
  - Mean selected-minus-best-control primary target margin: `6.5221479436`.
  - Mean selected-minus-best-control centroid improvement: `-0.9368553758`.
  - Mean selected-minus-V6 primary target margin: `80.5274579376`.
  - Mean selected-minus-V6 centroid improvement: `0.6951629122`.
  - Mean source primary margin change: `-149.7924740546`.
- Interpretation:
  - V9 supports a narrow four-behavior representation-space
    source-invariant target-attractor claim on the fresh final pool.
  - It does not show scalar centroid dominance over all controls.
  - It does not prove functional decoding, behavioral model editing, a
    single-vector steering method, larger-model generality, or broad MUAT
    generality.
  - The aggregate same-target transfer gate passed, but weak per-direction
    transfer remains; `has_majority_to_mountain_pattern` had transfer rate
    `0.6625`.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - positive V9 source-invariant target-attractor final result with the
      limitations above.

### 2026-06-12 - V10 Functional Weight Editing V9-Conditioned Delta

- Objective: test the next functional bridge after V9 by using a V9-style
  target-attractor representation as conditioning for a deterministic,
  train-only ridge editor that edits heldout source weights into requested
  target behaviors.
- Claim scope:
  - source-label-known, target-label-requested functional weight editing on the
    same small four-behavior synthetic subjects;
  - not source-label inference;
  - not broad MUAT proof;
  - not larger-model evidence;
  - not non-target capability preservation evidence.
- Preregistration:
  - `docs/preregistrations/four_behavior_functional_weight_editing_v10_v9_conditioned_delta.md`.
  - Initial reviewer confidence: `3/5`.
  - Required fixes:
    - make source-label-known evaluation explicit;
    - strengthen reliability gates;
    - fully specify nearest-train retrieval;
    - define all controls in Pareto and best-control aggregates.
  - Revised preregistration reviewer confidence: `5/5`.
- Implementation:
  - `model_zoo/scripts/train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta.py`.
  - `model_zoo/scripts/test_four_behavior_functional_weight_editing_v10_helpers.py`.
  - Helper verification:
    - `python -m pytest model_zoo/scripts/test_four_behavior_functional_weight_editing_v10_helpers.py -q`
      passed with `8` tests.
    - `python -m py_compile model_zoo/scripts/train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta.py model_zoo/scripts/test_four_behavior_functional_weight_editing_v10_helpers.py`
      passed.
  - No linting was run.
  - Implementation reviewer confidence after adding fail-closed final-detail
    redaction scans: `5/5`.
- Source-pool artifacts:
  - `runs/four_behavior_functional_weight_editing_v10_pools/combined_audit.json`;
  - `runs/four_behavior_functional_weight_editing_v10_pools/final_redacted_audit.json`.
  - train accepted counts: `64` per behavior;
  - development accepted counts: `24` per behavior;
  - final redacted accepted counts: `24` per behavior;
  - all cross-pool accepted overlaps were `0`;
  - final public surfaces exposed only accepted counts and hashes plus the
    allowed selected-train overlap summary.
  - Source-pool reviewer confidence: `5/5`.
- Development artifact:
  - `runs/four_behavior_functional_weight_editing_v10_v9_conditioned_delta/development_results.json`.
- Commands:
  - `python model_zoo/scripts/train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta.py --phase generate-pools`
  - `python model_zoo/scripts/train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta.py --phase development`
- Development result: failed.
  - `passed: false`.
  - `next_action`: `log_negative_development_result_do_not_open_final_raw`.
  - `n`: `288`.
  - Individual all-gate pass count/rate: `4/288`, `0.0138888889`.
  - Target-prediction count/rate: `73/288`, `0.2534722222`.
  - Pareto-undominated count/rate: `32/288`, `0.1111111111`.
  - Mean matched target margin: `0.0149848285`.
  - Mean matched target-vs-source margin: `0.0049861868`.
  - Mean matched-minus-no-edit target margin: `0.1921917884`.
  - Mean matched-minus-nearest-train target margin: `-0.7972723529`.
  - Mean matched-minus-best-control target margin: `-0.8004080978`.
  - Mean nearest-train-minus-matched source-output MSE:
    `-5320.2650570969`.
- Interpretation:
  - V10 is a clean negative development checkpoint.
  - The ridge weight-delta bridge suppressed source behavior but did not
    produce reliable target behavior and was dominated by nearest-train target
    retrieval and other controls.
  - It does not support a four-behavior functional weight-editing claim.
  - It strengthens the boundary that V9 is currently representation-space
    evidence only.
  - The V10 final raw pool remains sealed and final evaluation is blocked.
- Reviewer outcome:
  - Reviewer confidence: `5/5`.
  - Accepted scope:
    - clean negative V10 development checkpoint only.
