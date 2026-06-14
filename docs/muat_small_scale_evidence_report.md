# MUAT Small-Scale Evidence Report

Date: 2026-06-11

This report summarizes the current evidence package for the MUAT small-scale
experiment. It is intentionally narrow. The machine-checkable audit is
`runs/muat_evidence_package_audit/results.json`.

## Evidence-Supported Claims

The current artifacts support four claims:

1. Stored-probe activation signatures contain heldout-decodable behavior
   information for four clean behaviors under logistic/RF classifiers with
   shuffled-label controls.
2. A locked decoder performs restricted two-behavior functional decoding on a
   fresh final holdout for `sorted_ascending` and `sorted_descending`.
3. Frozen v2 normalized-signature edit vectors perform restricted two-behavior
   robust steering on a fresh final holdout for
   `sorted_ascending <-> sorted_descending`.
4. A preregistered V9 representation-space source-invariant target-attractor
   method passes on a fresh four-behavior final pool under Pareto/non-domination
   gates and aggregate same-target transfer-probe gates.

The current artifacts do not prove:

- larger-model generality;
- functional decode or behavioral steering for behaviors beyond
  `sorted_ascending <-> sorted_descending`;
- broad MUAT generality;
- non-aggressive steering-vector norms.

## Audit Status

Artifact:
`runs/muat_evidence_package_audit/results.json`

Status:

- `passed`: `true`
- failure count: `0`
- reviewer confidence: `5/5`

The audit verifies interpretability metrics, final decode metrics, robust
steering metrics, representation-space V9 target-attractor metrics, accepted
negative results, expected thresholds, holdout hashes,
subject-pool separation, and research-log review status.

## 1. Stored-Probe Signature Interpretability

Artifact:
`runs/stored_probe_interpret_v1/results.json`

Scope:

- four clean behaviors:
  - `sorted_descending`;
  - `sorted_ascending`;
  - `mountain_pattern`;
  - `has_majority`;
- fixed stored probe set;
- heldout classifier evaluation.

Key metrics:

- logistic regression balanced accuracy: `0.9227`;
- random forest balanced accuracy: `0.9728`;
- logistic shuffled-label balanced accuracy: `0.2197`;
- random forest shuffled-label balanced accuracy: `0.2347`;
- majority baseline balanced accuracy: `0.2500`.

Interpretation:

Stored-probe signatures carry behavior-decodable signal for the four clean
behaviors. This is not a decode or steering result by itself.

## 2. Restricted Two-Behavior Functional Decode

Artifact:
`runs/stored_probe_functional_decoder_v2_final_eval/results.json`

Scope:

- target behaviors:
  - `sorted_ascending`;
  - `sorted_descending`;
- fresh final paired artifact;
- locked V2 decoder;
- no method or threshold changes after the final result.

Leakage audit:

- final overlap with train: `0`;
- final overlap with development artifact: `0`;
- final subject/control references: `54`;
- checkpoint-validation overlap: `54`.

Aggregate final metrics:

- n: `54`;
- mean matched behavior margin: `+0.3337`;
- mean control behavior margin: `+0.0141`;
- mean matched-minus-control behavior margin: `+0.3196`;
- mean matched subject-output MSE: `19.5971`;
- mean control subject-output MSE: `206.8643`;
- mean control-minus-matched subject-output MSE: `187.2671`.

Controls:

- `noise_signature`;
- `opposite_direction`;
- `same_label_other_subject`.

The audit checks the registered control-specific gates exactly:

- noise/opposite controls require behavior-margin delta `>= 0.20` and subject-MSE
  delta `>= 0.05`;
- same-label controls require subject-MSE delta `>= 0.02`.

Interpretation:

The locked decoder can decode fixed stored-probe signatures into functional
small-network weights for the two sorted behaviors on a fresh final holdout.

## 3. Robust Two-Behavior Steering

Artifact:
`runs/stored_probe_signature_edit_vectors_v2_robust_final_eval/results.json`

Fresh final holdout:

- subjects:
  `runs/fresh_external_steering_holdout_v3_robust_final/subjects.json`;
- SHA-256:
  `8c9f2cc2ddf1f407c52155f6b483dbed96c00c02a9ad846b1f64ac9f5c1e1124`;
- n: `48`;
- `24` per source behavior.

Frozen inputs:

- decoder:
  `runs/stored_probe_functional_decoder_v2_adaptive/model.pt`;
- v2 edit vectors:
  `runs/stored_probe_signature_edit_vectors_v2_robust_development/edit_vectors.pt`.

Vector norms:

- `sorted_ascending -> sorted_descending`: `20.4962`;
- `sorted_descending -> sorted_ascending`: `44.3385`.

Strict gates:

- mean steered-minus-no-edit target margin `>= 0.20`;
- mean steered-minus-reverse target margin `>= 0.20`;
- mean steered-minus-noise target margin `>= 0.20`;
- mean steered-minus-worst-random-norm-matched target margin `>= 0.20`;
- mean steered target margin `>= 0.20`;
- mean steered source-margin change `<= -0.05`;
- individual all-gate pass rate `>= 0.95`;
- per-target individual all-gate pass rate `>= 0.90`.

Controls:

- no edit;
- reverse direction;
- noise signature;
- worst-of-32 norm-matched random edit vectors per subject.

Aggregate final metrics:

- n: `48`;
- mean steered target margin: `+0.5063`;
- mean no-edit target margin: `-0.0386`;
- mean reverse target margin: `-0.0267`;
- mean noise target margin: `-0.0119`;
- mean worst-random target margin: `+0.0073`;
- mean steered-minus-no-edit target margin: `+0.5449`;
- mean steered-minus-reverse target margin: `+0.5329`;
- mean steered-minus-noise target margin: `+0.5181`;
- mean steered-minus-worst-random target margin: `+0.4990`;
- mean steered source-margin change: `-0.4580`.

Individual gate audit:

- overall: `48/48`;
- target `sorted_ascending`: `24/24`;
- target `sorted_descending`: `24/24`;
- failed records: `0`.

Interpretation:

Frozen v2 normalized-signature edit vectors robustly steer decoded models between
the two sorted behaviors under strict mean and per-subject gates. The large
`sorted_descending -> sorted_ascending` vector norm is a caveat.

## 4. Negative and Limiting Results

### V1 Robust Steering Limitation

Artifact:
`runs/stored_probe_signature_edit_vectors_v1_robust_external_eval/results.json`

Holdout SHA-256:
`a0f1727294b7bb188a461b0222592d890245408394e7a87cb806000d9ad53e9f`

Result:

- overall individual all-gate pass: `43/48`;
- target `sorted_ascending`: `19/24`;
- target `sorted_descending`: `24/24`.

Interpretation:

The v1 learned edit vectors supported mean steering but failed stricter
per-subject reliability gates, especially when steering toward
`sorted_ascending`.

### Additional-Behavior Decode Feasibility Failure

Artifact:
`runs/stored_probe_additional_behavior_decode_feasibility_v1/results.json`

Fresh holdout SHA-256:
`03b72098c773690011fa330487e51d08f69c2f3b4558e7ab1ae31ae82f5aeb6b`

Scope:

- `has_majority`;
- `mountain_pattern`;
- fresh subjects;
- no decoder or method training on these fresh additional-behavior subjects;
- worst-of-8 normalized-signature noise controls.

Source-model caveat:

- source gate was lowered to `0.20`;
- `has_majority` sources were weak:
  - mean margin `0.2279`;
- `mountain_pattern` sources were strong:
  - mean margin `0.8921`.

Result:

- aggregate mean matched target margin: `-0.0174`;
- aggregate matched-minus-worst-noise target margin: `-0.0393`;
- individual pass rate: `0/16`;
- `has_majority`: `0/8`;
- `mountain_pattern`: `0/8`.

Interpretation:

The current locked decoder/protocol does not produce usable matched decode
margins for fresh `has_majority` or `mountain_pattern` subjects under this
bounded feasibility setup. This reinforces that additional-behavior
decode/steering remains unproven. It does not show additional-behavior decoding
is impossible.

### Four-Behavior Decoder Development Failures

Source pools:

- train: `64` accepted subjects per behavior;
- development: `24` accepted subjects per behavior;
- final: `24` accepted subjects per behavior, sealed for a future one-shot
  evaluation only.

Pool-separation audit:

- all train/development/final overlaps are `0` for seed, subject id, weight hash,
  and signature hash;
- `final_subjects.json` remained sealed during V1, V2, and V3 development;
- final redacted audit confirms `24` accepted final subjects per behavior and
  max selected-training-vs-heldout overlap count `0`.

Three adaptive train/development decoder attempts failed the preregistered
control gates:

- V1 direct MLP decoder:
  - artifact: `runs/four_behavior_decoder_development_v1/results.json`;
  - individual pass count: `0/96`;
  - mean matched target margin: `0.0045465377`;
  - mean matched-minus-best-control target margin: `-0.4180624041`;
  - mean best-control-minus-matched subject-output MSE: `-54.8715616030`.
- V2 functional-distillation decoder:
  - artifact: `runs/four_behavior_decoder_development_v2/results.json`;
  - individual pass count: `0/96`;
  - mean matched target margin: `0.4117191500`;
  - mean matched-minus-best-control target margin: `-0.3154511039`;
  - mean best-control-minus-matched subject-output MSE: `-27.4087092181`.
- V3 signature-inversion decoder:
  - artifact:
    `runs/four_behavior_decoder_development_v3_signature_inversion/results.json`;
  - individual pass count: `0/96`;
  - V3 inferred behavior accuracy: `0.78125`;
  - mean matched target margin: `0.2400836338`;
  - mean matched-minus-best-control target margin: `-0.6036293377`;
  - mean best-control-minus-matched subject-output MSE: `-141.8004977169`.

Interpretation:

V1, V2, and V3 are accepted negative train/development checkpoints. V3 is a
negative adaptive train/development signature-inversion checkpoint, not proof of
four-behavior functional decoding. The failures are dominated by
control-specificity and subject-specificity problems: same-label train controls,
nearest-train signature neighbors, and V3-inverted controls beat matched outputs
under the registered gates.

No four-behavior decoder should be run on the sealed final raw pool from these
methods. Any further architecture, objective, or control change requires a new
preregistration and reviewer acceptance before development continues.

### Four-Behavior Representation Steering V1 Development Failure

Artifacts:

- preregistration:
  `docs/preregistrations/four_behavior_representation_steering_v1.md`;
- source-pool audit:
  `runs/four_behavior_representation_steering_v1_pools/combined_audit.json`;
- final redacted audit:
  `runs/four_behavior_representation_steering_v1_pools/final_redacted_audit.json`;
- development result:
  `runs/four_behavior_representation_steering_v1/development_results.json`.

Scope:

- four clean behaviors;
- representation-space steering only;
- train-only signature normalization, centroids, and affine primary evaluator;
- twelve ordered source-target edit vectors;
- fresh steering-specific train/development/final pools;
- development go/no-go gates matched the final proof gates except for pool
  identity.

Source-pool checkpoint:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final accepted counts: `24` per behavior;
- accepted train/development/final overlaps: `0` for seed, subject id, weight
  hash, and signature hash;
- steering final raw pool remained sealed after generation.

Development result:

- A first development artifact reported `55/288` individual passes but was later
  found to have trained edit vectors with a flawed centroid-improvement
  objective. It is superseded by the corrected no-edit-relative rerun below.
- `passed: false`;
- aggregate n: `288`;
- individual pass count: `16/288`;
- individual pass rate: `0.0555555556`;
- mean matched primary target margin: `101.2225835986`;
- mean matched-minus-best-control primary target margin: `60.9470050931`;
- mean matched centroid improvement: `0.2559432056`;
- mean matched-minus-best-control centroid improvement: `-1.0535927349`;
- mean source primary margin change: `-177.3604026188`.

Interpretation:

The frozen V1 representation-steering method strongly drives the train-only
primary linear evaluator and suppresses source margins, but fails the
centroid-control specificity and individual/pass-rate gates on development.
Therefore it blocks final steering evaluation. It is not final steering evidence
and does not justify opening the steering final raw pool for this method.

### Four-Behavior Representation Steering V2 Centroid-Delta Development Failure

Artifacts:

- preregistration:
  `docs/preregistrations/four_behavior_representation_steering_v2_centroid_delta.md`;
- failure diagnosis:
  `docs/representation_steering_v2_failure_diagnosis.md`;
- source-pool audit:
  `runs/four_behavior_representation_steering_v2_pools/combined_audit.json`;
- final redacted audit:
  `runs/four_behavior_representation_steering_v2_pools/final_redacted_audit.json`;
- development result:
  `runs/four_behavior_representation_steering_v2_centroid_delta/development_results.json`.

V2 tested exact train-only centroid deltas as the steering vectors, with no
learned edit-vector optimizer. It used fresh V2 train/development/final source
pools because the V1 development pool had been inspected during diagnosis.

Source-pool checkpoint:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final redacted accepted counts: `24` per behavior;
- accepted train/development/final overlaps: `0` for seed, subject id, weight
  hash, and signature hash;
- V2 final raw pool remained sealed after generation.

Development result:

- `passed: false`;
- aggregate n: `288`;
- individual pass count: `142/288`;
- individual pass rate: `0.4930555556`;
- mean matched primary target margin: `47.6160739958`;
- mean matched-minus-best-control primary target margin: `24.2469312698`;
- mean matched centroid improvement: `1.5253242254`;
- mean matched-minus-best-control centroid improvement: `0.4016633564`;
- mean source primary margin change: `-116.9670098159`.

Interpretation:

V2 shows useful aggregate movement in the fixed-probe representation space, but
it fails the preregistered reliability gates: aggregate pass rate is below
`0.90`, every target is below the `0.80` pass-rate gate, and every ordered
direction is below the `0.90` pass-rate gate. This blocks final evaluation. It
is not proof-grade four-behavior representation steering and does not justify
opening the V2 final raw pool for this method. The accepted failure diagnosis
identifies source-specificity against same-target other-source centroid controls
as the main next-method bottleneck.

### Four-Behavior Representation Steering V3 Diagonal-Transport Development Failure

Artifacts:

- preregistration:
  `docs/preregistrations/four_behavior_representation_steering_v3_diagonal_transport.md`;
- source-pool audit:
  `runs/four_behavior_representation_steering_v3_pools/combined_audit.json`;
- final redacted audit:
  `runs/four_behavior_representation_steering_v3_pools/final_redacted_audit.json`;
- development result:
  `runs/four_behavior_representation_steering_v3_diagonal_transport/development_results.json`.

V3 tested closed-form train-only diagonal covariance transport, with no learned
edit-vector optimizer. It used fresh V3 train/development/final source pools
because the V1 and V2 development pools had been inspected during failure
analysis.

Source-pool checkpoint:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final redacted accepted counts: `24` per behavior;
- accepted train/development/final overlaps: `0` for seed, subject id, weight
  hash, and signature hash;
- V3 final raw pool remained sealed after generation.

Development result:

- `passed: false`;
- `next_action`: `log_negative_development_result_do_not_open_final_raw`;
- aggregate n: `288`;
- individual pass count: `30/288`;
- individual pass rate: `0.1041666667`;
- mean matched primary target margin: `45.6755528839`;
- mean matched centroid improvement: `1.8963179257`;
- mean matched-minus-V2-centroid-delta primary target margin: `1.9002874485`;
- mean matched-minus-V2-centroid-delta centroid improvement: `0.4344081978`;
- mean matched-minus-best-control primary target margin: `-3.9445673236`;
- mean matched-minus-best-control centroid improvement: `-0.9124838478`;
- mean source primary margin change: `-109.6813265847`.

Interpretation:

V3 shows strong aggregate target movement and improves over the V2 centroid
delta baseline on mean primary and centroid metrics, but it fails proof-grade
source-specificity against the full best-control set. Aggregate pass rate is
below `0.90`, every target is below the `0.80` pass-rate gate, and every
ordered direction is below the `0.90` pass-rate gate. This blocks final
evaluation. It is not proof-grade four-behavior representation steering and
does not justify opening the V3 final raw pool for this method.

### Four-Behavior Representation Steering V4 Low-Rank Residual-Transport Development Failure

Artifacts:

- preregistration:
  `docs/preregistrations/four_behavior_representation_steering_v4_low_rank_residual_transport.md`;
- source-pool audit:
  `runs/four_behavior_representation_steering_v4_pools/combined_audit.json`;
- final redacted audit:
  `runs/four_behavior_representation_steering_v4_pools/final_redacted_audit.json`;
- development result:
  `runs/four_behavior_representation_steering_v4_low_rank_residual_transport/development_results.json`.

V4 tested closed-form train-only low-rank residual covariance transport, with
no learned edit-vector optimizer. It used fresh V4 train/development/final
source pools because the prior steering development pools had been inspected
during failure analysis.

Source-pool checkpoint:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final redacted accepted counts: `24` per behavior;
- accepted train/development/final overlaps: `0` for seed, subject id, weight
  hash, and signature hash;
- V4 final raw pool remained sealed after generation.

Development result:

- `passed: false`;
- `next_action`: `log_negative_development_result_do_not_open_final_raw`;
- aggregate n: `288`;
- individual pass count: `42/288`;
- individual pass rate: `0.1458333333`;
- mean matched primary target margin: `46.5516936464`;
- mean matched centroid improvement: `4.4940608740`;
- mean matched-minus-V2-centroid-delta primary target margin: `-0.0098890116`;
- mean matched-minus-V2-centroid-delta centroid improvement: `3.2327019506`;
- mean matched-minus-V3-diagonal-transport primary target margin: `-1.2330002536`;
- mean matched-minus-V3-diagonal-transport centroid improvement: `3.0668179062`;
- mean matched-minus-best-control primary target margin: `-14.2722247789`;
- mean matched-minus-best-control centroid improvement: `-1.7965679599`;
- mean source primary margin change: `-108.6704917749`.

Interpretation:

V4 improves over V2 and V3 on centroid metrics but not on primary target-margin
specificity, and it still fails proof-grade specificity against the full
best-control set. Aggregate pass rate is below `0.90`, every target is below
the `0.80` pass-rate gate, and every ordered direction is below the `0.90`
pass-rate gate. This blocks final evaluation. It is not proof-grade
four-behavior representation steering and does not justify opening the V4 final
raw pool for this method.

### Four-Behavior Representation Steering V5 Contrastive Residual-Calibration Development Failure

Artifacts:

- preregistration:
  `docs/preregistrations/four_behavior_representation_steering_v5_contrastive_residual_calibration.md`;
- source-pool audit:
  `runs/four_behavior_representation_steering_v5_pools/combined_audit.json`;
- final redacted audit:
  `runs/four_behavior_representation_steering_v5_pools/final_redacted_audit.json`;
- development result:
  `runs/four_behavior_representation_steering_v5_contrastive_residual_calibration/development_results.json`.

V5 tested train-only contrastive residual calibration on top of V4 low-rank
transport. It used fresh V5 train/development/final source pools because all
prior steering development pools had been inspected during failure analysis.

Source-pool checkpoint:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final redacted accepted counts: `24` per behavior;
- accepted train/development/final overlaps: `0` for seed, subject id, weight
  hash, and signature hash;
- V5 final raw pool remained sealed after generation.

Development result:

- `passed: false`;
- `next_action`: `log_negative_development_result_do_not_open_final_raw`;
- aggregate n: `288`;
- individual pass count: `20/288`;
- individual pass rate: `0.0694444444`;
- mean matched primary target margin: `121.8755627208`;
- mean source primary margin change: `-170.5525876068`;
- mean matched centroid improvement: `4.1053234041`;
- mean matched-minus-best-control primary target margin: `9.9655429009`;
- mean matched-minus-best-control centroid improvement: `-2.5652369327`;
- mean matched-minus-V2-centroid-delta primary target margin: `75.2867757827`;
- mean matched-minus-V3-diagonal-transport primary target margin: `74.9781137043`;
- mean matched-minus-V4-low-rank primary target margin: `79.1476116118`;
- mean matched-minus-V4-low-rank centroid improvement: `-1.0327701767`.

Interpretation:

V5 produces much stronger primary target movement and source suppression than
the prior four-behavior steering attempts, and it beats V2, V3, and V4 on mean
primary target-margin metrics. It still fails proof-grade reliability and
centroid best-control specificity. Aggregate pass rate is below `0.90`, every
target is below the `0.80` pass-rate gate, and every ordered direction is below
the `0.90` pass-rate gate. V4 remains stronger on mean centroid improvement.
This blocks final evaluation. It is not proof-grade four-behavior
representation steering and does not justify opening the V5 final raw pool for
this method.

### Four-Behavior Representation Steering V6 Centroid-Constrained Primary-Correction Development Failure

Artifacts:

- preregistration:
  `docs/preregistrations/four_behavior_representation_steering_v6_centroid_constrained_primary_correction.md`;
- source-pool audit:
  `runs/four_behavior_representation_steering_v6_pools/combined_audit.json`;
- final redacted audit:
  `runs/four_behavior_representation_steering_v6_pools/final_redacted_audit.json`;
- development result:
  `runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/development_results.json`.

V6 tested per-example train-only centroid-constrained primary correction on top
of V4 low-rank residual transport. It used fresh V6 train/development/final
source pools because all prior steering development pools had been inspected
during failure analysis.

Source-pool checkpoint:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final redacted accepted counts: `24` per behavior;
- accepted train/development/final overlaps: `0` for seed, subject id, weight
  hash, and signature hash;
- V6 final raw pool remained sealed after development failed.

Development result:

- `passed: false`;
- `next_action`: `log_negative_development_result_do_not_open_final_raw`;
- aggregate n: `288`;
- individual pass count: `2/288`;
- individual pass rate: `0.0069444444`;
- mean matched primary target margin: `45.0174989502`;
- mean matched centroid improvement: `5.2879611254`;
- mean source primary margin change: `-103.4365548342`;
- mean matched-minus-best-control primary target margin: `-63.6636278828`;
- mean matched-minus-best-control centroid improvement: `-1.9438294139`;
- mean matched-minus-V5-calibrated primary target margin: `-62.6472493344`;
- mean matched-minus-V5-calibrated centroid improvement: `0.8543339902`;
- mean matched-minus-V4-low-rank primary target margin: `5.9002181060`;
- mean matched-minus-V4-low-rank centroid improvement: `0.0930981868`.

Interpretation:

V6 preserves strong primary and centroid movement and improves over V2, V3, and
V4 on aggregate mean metrics. It still fails proof-grade control specificity and
reliability: best controls dominate both primary and centroid gates, V5 remains
much stronger on primary margin, aggregate pass rate is below `0.90`, every
target is below the `0.80` pass-rate gate, and every ordered direction is below
the `0.90` pass-rate gate. This blocks final evaluation. It is not proof-grade
four-behavior representation steering and does not justify opening the V6 final
raw pool for this method.

Posthoc development-only diagnosis:

- artifact:
  `runs/four_behavior_representation_steering_v6_centroid_constrained_primary_correction/posthoc_pareto_diagnosis.json`;
- claim scope: `v6_development_posthoc_diagnosis_not_proof`;
- Pareto-undominated records: `226/288`;
- aggregate Pareto-undominated rate: `0.7847222222`;
- worst direction: `16/24`;
- next action:
  `use_only_to_motivate_fresh_preregistered_v7_design_do_not_open_v6_final_raw`.

This posthoc diagnosis does not change the V6 development failure, does not
authorize opening or evaluating the V6 final raw pool, and is not final evidence.
It only motivates future method design under a new preregistration and fresh
pools.

### Four-Behavior Representation Steering V7/V8 Fresh-Pool Development Failures

V7 and V8 were fresh-pool follow-ups motivated by the V6 posthoc Pareto
diagnosis. Both were preregistered, implementation-reviewed, source-pool
reviewed, and development-result reviewed at `5/5`. Both failed development
gates, so both final raw pools remain sealed.

V7 artifact:

`runs/four_behavior_representation_steering_v7_pareto_frontier_correction/development_results.json`

V7 result:

- `passed: false`;
- individual pass count/rate: `245/288`, `0.8506944444`;
- Pareto-undominated count/rate: `257/288`, `0.8923611111`;
- target-prediction pass count/rate: `278/288`, `0.9652777778`;
- mean selected primary target margin: `121.9106095102`;
- mean selected centroid improvement: `5.0538051493`;
- mean selected-minus-V6 centroid improvement: `0.0154168175`;
- mean source primary margin change: `-170.7438490523`.

V8 artifact:

`runs/four_behavior_representation_steering_v8_source_conditional_tournament_correction/development_results.json`

V8 result:

- `passed: false`;
- individual pass count/rate: `237/288`, `0.8229166667`;
- Pareto-undominated count/rate: `243/288`, `0.84375`;
- target-prediction pass count/rate: `281/288`, `0.9756944444`;
- mean selected primary target margin: `129.3434276382`;
- mean selected centroid improvement: `6.0069983933`;
- mean selected-minus-V5 centroid improvement: `1.6989225083`;
- mean selected-minus-V6 centroid improvement: `0.6251051459`;
- mean source primary margin change: `-161.3274385167`.

Interpretation:

V7 almost met the Pareto-rate gate but still failed aggregate reliability and
selected-minus-V6 centroid gates. V8 improved absolute primary/centroid/source
suppression metrics and cleared the V6 centroid-margin issue, but failed
Pareto reliability because V8 same-target controls strengthened too.

These results are useful negative evidence about the source-conditional
four-behavior steering formulation. They are not proof-grade four-behavior
steering evidence, not final evidence, and do not authorize opening the V7 or
V8 final raw pools.

### Four-Behavior Representation-Space V9 Source-Invariant Target-Attractor

V9 was a fresh-pool follow-up motivated by the V7/V8 result that same-target
other-source candidates behaved like target-attracting edits rather than
reliable bad controls. V9 preregistered those candidates as transfer probes,
not negative dominance controls.

Artifacts:

- `docs/preregistrations/four_behavior_representation_steering_v9_source_invariant_target_attractor.md`;
- `runs/four_behavior_representation_steering_v9_pools/combined_audit.json`;
- `runs/four_behavior_representation_steering_v9_pools/final_redacted_audit.json`;
- `runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/development_results.json`;
- `runs/four_behavior_representation_steering_v9_source_invariant_target_attractor/final_results.json`.

Source-pool result:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final accepted counts: `24` per behavior;
- all cross-pool accepted overlaps: `0`.

Development correction:

The first V9 development run exposed a splitter bug: shuffled V9 controls whose
sampled direction shared the matched target were incorrectly classified as
transfer probes. The invalid run had `3125` transfer probes instead of the
preregistered `2880`. The splitter was corrected to require the explicit
`same_target_other_source_v9_source_invariant_target_attractor` control type,
the regression test
`test_v9_shuffled_same_target_direction_remains_negative_control` was added,
and the development run was repeated. The invalid run is excluded from evidence.

Corrected development result:

- `passed: true`;
- individual pass count/rate: `277/288`, `0.9618055556`;
- Pareto-undominated count/rate: `279/288`, `0.96875`;
- target-prediction pass count/rate: `286/288`, `0.9930555556`;
- same-target transfer probe count: `2880`;
- same-target transfer target-prediction count/rate:
  `2735/2880`, `0.9496527778`;
- same-target transfer gate-pass count/rate:
  `2735/2880`, `0.9496527778`.

Final result:

- `passed: true`;
- individual pass count/rate: `278/288`, `0.9652777778`;
- Pareto-undominated count/rate: `285/288`, `0.9895833333`;
- target-prediction pass count/rate: `281/288`, `0.9756944444`;
- same-target transfer probe count: `2880`;
- same-target transfer target-prediction count/rate: `2664/2880`, `0.925`;
- same-target transfer gate-pass count/rate: `2664/2880`, `0.925`;
- mean selected primary target margin: `121.8615035945`;
- mean selected centroid improvement: `6.0319956938`;
- mean selected-minus-best-control primary target margin: `6.5221479436`;
- mean selected-minus-best-control centroid improvement: `-0.9368553758`;
- mean selected-minus-V6 primary target margin: `80.5274579376`;
- mean selected-minus-V6 centroid improvement: `0.6951629122`;
- mean source primary margin change: `-149.7924740546`.

Interpretation:

V9 supports a narrow four-behavior representation-space source-invariant
target-attractor claim on the fresh final pool. It does not show scalar
centroid dominance over all controls: the mean selected-minus-best-control
centroid improvement is negative. It is not functional decoding evidence, not
behavioral model-editing evidence, not a single-vector steering proof, and not
larger-model or broad MUAT evidence. Transfer is also not uniformly strong by
direction: `has_majority_to_mountain_pattern` had final same-target transfer
rate `0.6625`, even though the aggregate transfer gate passed.

### Four-Behavior Functional Weight Editing V10 Development Failure

V10 tested the smallest functional bridge after V9: use a V9-style
target-attractor representation as conditioning for a deterministic train-only
ridge editor that directly edits heldout source weights. The preregistered
claim was explicitly source-label-known and target-label-requested.

Artifacts:

- `docs/preregistrations/four_behavior_functional_weight_editing_v10_v9_conditioned_delta.md`;
- `runs/four_behavior_functional_weight_editing_v10_pools/combined_audit.json`;
- `runs/four_behavior_functional_weight_editing_v10_pools/final_redacted_audit.json`;
- `runs/four_behavior_functional_weight_editing_v10_v9_conditioned_delta/development_results.json`.

Source-pool result:

- train accepted counts: `64` per behavior;
- development accepted counts: `24` per behavior;
- final redacted accepted counts: `24` per behavior;
- all cross-pool accepted overlaps: `0`;
- final public surfaces exposed no per-subject final details.

Development result:

- `passed: false`;
- `next_action`: `log_negative_development_result_do_not_open_final_raw`;
- individual pass count/rate: `4/288`, `0.0138888889`;
- target-prediction count/rate: `73/288`, `0.2534722222`;
- Pareto-undominated count/rate: `32/288`, `0.1111111111`;
- mean matched target margin: `0.0149848285`;
- mean matched target-vs-source margin: `0.0049861868`;
- mean matched-minus-no-edit target margin: `0.1921917884`;
- mean matched-minus-nearest-train target margin: `-0.7972723529`;
- mean matched-minus-best-control target margin: `-0.8004080978`;
- mean nearest-train-minus-matched source-output MSE: `-5320.2650570969`.

Interpretation:

V10 is a clean negative development checkpoint. The ridge weight-delta bridge
suppressed source behavior but did not produce reliable target behavior, and it
was strongly beaten by the nearest-train target retrieval control on target
margin and source-output similarity. This does not support a four-behavior
functional weight-editing claim. It reinforces that the current positive
four-behavior evidence is representation-space V9 target-attractor evidence,
not behavioral model editing. The V10 final raw pool remains sealed.

## Reproducibility Commands

Run the artifact-level audit:

```bash
python model_zoo/scripts/audit_muat_evidence_package.py
```

Run the package-level verifier:

```bash
python model_zoo/scripts/verify_muat_evidence_package.py
```

Build and verify the checksum manifest:

```bash
python model_zoo/scripts/build_evidence_manifest.py
python model_zoo/scripts/build_evidence_manifest.py --verify
```

Do not treat this report as a substitute for the JSON artifacts. The verifier is
the authoritative package-level consistency check.
