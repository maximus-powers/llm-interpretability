# V38 Compatible-MSE Gated Projected Optimizer Results

## Status

V38 completed as a bounded negative result. It did not pass the proof gates.

The run supports the diagnostic hypothesis that hard compatible-MSE gating can improve locality-like compatible-MSE counts relative to the best V37 projected optimizer candidate, but it suppresses the target behavior edit too strongly to be useful as the next successful method.

## Implementation Under Test

- Experiment variant: `v38_compatible_mse_gated_projected_optimizer_diagnostic`
- Matched edit source: `compatible_mse_gated_projected_optimizer_sparse`
- Main implementation:
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:247`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:2890`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3073`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3352`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:3666`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7312`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7438`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:7473`
  - `/Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py:9448`
- Tests:
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:2947`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3025`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3167`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3286`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3348`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3371`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3396`
  - `/Users/max/Desktop/muat/model_zoo/scripts/test_four_behavior_functional_weight_editing_v25_helpers.py:3722`

## Verification

- Focused V38/parser/V37 regression tests: `8 passed, 165 deselected`
- Full helper suite: `173 passed in 7.53s`
- Syntax check: `python -m py_compile` passed for the train script and helper tests
- No linting was run, per repository instruction.

## Run Command

```bash
python /Users/max/Desktop/muat/model_zoo/scripts/train_four_behavior_functional_weight_editing_v25_jacobian_rank1_editor.py \
  --phase development \
  --pool-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_pools \
  --output-dir /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor \
  --monitor-interval-seconds 5 \
  --summary-only-stdout \
  --run-inner-validation \
  --inner-validation-config-grid v38-compatible-mse-gated-projected-optimizer \
  --inner-validation-rung-jobs 24 \
  --inner-validation-keep-fractions 1.0 \
  --development-job-selection balanced-directions \
  > /Users/max/Desktop/muat/runs/four_behavior_functional_weight_editing_v25_jacobian_rank1_editor/v38_compatible_mse_gated_projected_optimizer_terminal.log 2>&1
```

## Long-Run Monitoring Evidence

- Training process after completion: no matching `train_four_behavior_functional_weight_editing_v25` or `python.*muat` process.
- Progress log rows: `974`
- Monitor log rows: `60`
- Terminal log rows: `176`
- Final monitor event: `monitor_stop`
- Final monitor elapsed seconds: `291.60785383300004`
- CPU user/system seconds: `491.262394` / `1882.616583`
- Final monitor latest progress event: `development_setup_completed`
- Final monitor progress line count: `974`
- Progress log SHA-256: `b6244535ac9933fb07a804e91d26426cd903369b5d98a6313c86f60122f261e8`
- Monitor log SHA-256: `a175c5cf1cb8d13774255516f399799d08896e193e93a33648da2d6ddaec557e`
- Terminal log SHA-256: `7ef7122b85114e9e029b70b87a881f9d754b4141ff565c792ee7eba2b532b16c`

The monitor shows steady progress through optimizer events near the end of the run, then a clean stop. This is not consistent with an idle or stuck process.

## Leak Checks

Strict raw-key scan over progress and monitor logs returned no matches for:

```text
final_subjects
subject_id
weights
logits
gradient
selected_coordinates
support_examples
sequence
compatible_jacobian
raw_delta
projected_delta
```

The sealed final subject file was not opened or read.

V38 alpha selection records also preserved the required audit fields:

- `compatible_gate_pass`
- `alpha_compatible_mse_gate`
- `fallback_compatible_penalty`
- `eligible_count`
- `candidate_metrics_hash`

Missing required alpha audit fields: `0`

## Event Counts

- `inner_validation_candidate_completed`: `4`
- `development_evaluation_record_completed`: `96`
- `v38_compatible_gated_optimizer_progress`: `480`
- `v38_compatible_gated_optimizer_completed`: `96`
- `v38_compatible_gated_alpha_selected`: `96`
- `inner_validation_completed`: `1`
- `development_setup_completed`: `1`

## Candidate Metrics

| Config | Hash Prefix | Target Pred Rate | Mean Target Margin | Matched - Best Control | Matched - Shuffled | Pareto Rate | Proof Fails | Compatible MSE Fails | Target Pred Fails | Target Margin Fails | Control Margin Fails |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1ab58df4` | 0.2500 | 0.0255 | 0.1759 | 0.1888 | 0.9167 | 23 | 9 | 18 | 19 | 198 |
| 1 | `9e217f12` | 0.2500 | 0.0255 | 0.1783 | 0.1888 | 0.9167 | 23 | 9 | 18 | 19 | 198 |
| 2 | `d3fc70d6` | 0.2500 | 0.0450 | 0.1963 | 0.2083 | 0.9167 | 23 | 10 | 18 | 18 | 187 |
| 3 | `60975db9` | 0.2500 | 0.0450 | 0.1991 | 0.2083 | 0.9167 | 23 | 10 | 18 | 18 | 187 |

Best selected config:

- Index: `3`
- Hash: `60975db901e5fec9297679cd66d8ecd4c2dc13b69e95385b44da68ebe872ac51`

## Alpha and Preservation Diagnostics

Alpha counts:

- `0.0`: `58`
- `0.25`: `8`
- `0.5`: `8`
- `0.75`: `18`
- `1.0`: `4`

Selection modes:

- `eligible_min_compatible_mse`: `26`
- `fallback_penalized`: `70`

Gate values:

- `5.0`: `48`
- `15.0`: `48`

Compatible gate pass counts:

- `true`: `96`

Preservation energy ratio:

- Count: `96`
- Mean: `0.5002674783269564`
- Min: `0.5000036358833313`
- Max: `0.5014910101890564`

## Interpretation

V38 lowered compatible-MSE failures versus V37's best target-plastic candidate, but it selected mostly weak or zero alphas and never recovered target prediction success:

- V37 best plasticity candidate: target prediction rate `0.7083`, compatible-MSE failures `19`, proof failures `11`
- V38 best selected candidate: target prediction rate `0.2500`, compatible-MSE failures `10`, proof failures `23`

This is a clear locality/plasticity tradeoff. The compatible-MSE gate improved one locality proxy but made the edit too conservative. The high fallback count (`70/96`) and alpha distribution dominated by `0.0` indicate the configured gate and score were too restrictive for functional target movement.

The result should not be reported as behavioral editing success. It should be reported as evidence that nullspace/projection-compatible locality constraints can be enforced, but the next method needs a softer or staged constraint that preserves minimum target feasibility.

## Literature Context

The V38 outcome is consistent with recent editing and constrained-learning work:

- AlphaEdit frames locality preservation through null-space constraints, but V38 shows that locality constraints alone can overconstrain a small functional editing setup: [AlphaEdit](https://arxiv.org/abs/2410.02355)
- BalancEdit emphasizes balancing reliability and locality; V38 is a concrete failure mode of over-weighting locality: [BalancEdit](https://arxiv.org/abs/2505.01343)
- Edit locality evaluation work argues that locality must be directly measured rather than assumed from low-level constraints; V38 logs direct compatible-MSE and control-margin failures: [Locality in Knowledge Editing](https://arxiv.org/pdf/2601.17343)
- ENFORCE-style constrained learning supports constraint-aware objectives, but V38 suggests the constraint schedule needs target-feasibility safeguards: [ENFORCE](https://arxiv.org/html/2502.06774v4)
- Output-space projection and task-arithmetic work motivate preserving function while editing directions, but V38 shows the need to tune the preservation/editing tradeoff at this scale: [Model Merging by Output-Space Projection](https://arxiv.org/abs/2605.29101), [Task Arithmetic in Tangent Space](https://arxiv.org/abs/2305.12827)

## Recommended Next Step

V39 should avoid hard alpha eligibility as the primary selection rule. The next most promising direction is a Pareto-front or staged constrained optimizer:

1. Require a minimum target feasibility threshold before locality ranking.
2. Use compatible-MSE as a soft penalty or adaptive schedule rather than a hard alpha gate.
3. Preserve the V38 audit fields and add a target-feasibility audit field to expose whether locality improvements are coming from no-op edits.
4. Keep the bounded 4-candidate, 24-job grid until a candidate improves both target prediction and compatible-MSE failures relative to V37/V38.

