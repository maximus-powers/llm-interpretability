# Paired Contrast Proof Dataset Design

## Goal

Build a proof-grade dataset and evaluator that can test the broad MUAT hypothesis without relying on post-hoc controls. Each generated subject row must carry its matched activation signature and its control signatures as a paired contrast set, with probe provenance hashed and saved in the artifact.

## Why The Current Runs Failed

The latest reviewed runs show a stable pattern:

- matched signatures contain subject-output specificity signal;
- train-centroid prototype leakage can be structurally removed;
- behavior decoding still collapses into broad class/direction priors under noise, shuffled, null, and ablation controls;
- stronger control penalties reduce some controls but degrade matched behavior decode.

This means the current dataset/training setup cannot prove fixed-signature-specific behavior decoding. Controls are reconstructed after the fact rather than being generated as part of the training target.

## Required Dataset Unit

Each proof row should be a grouped tuple, not an independent sample:

```json
{
  "group_id": "stable id",
  "target_pattern": "sorted_ascending",
  "subject": {
    "weights": [...],
    "signature": [...],
    "behavior_metrics": {...}
  },
  "controls": {
    "same_label_centroid": {"signature": [...], "member_subject_ids": [...]},
    "same_label_other_subject": {"signature": [...], "subject_id": "..."},
    "different_label_same_direction": {"signature": [...], "subject_id": "...", "target_pattern": "..."},
    "opposite_direction": {"signature": [...], "subject_id": "...", "target_pattern": "..."},
    "null_signature": {"signature": [...]},
    "noise_signature": {"signature": [...], "seed": 123}
  },
  "probe_provenance": {
    "probe_set_id": "...",
    "probe_examples": [...],
    "probe_examples_hash": "...",
    "behavior_suite_hash": "...",
    "probe_generation_config_hash": "...",
    "extractor_config_hash": "...",
    "extractor_code_hash": "...",
    "normalization_stats_hash": "...",
    "dataset_source_hash": "...",
    "git_commit": "..."
  }
}
```

The tuple is the atomic split unit. No subject or control member from a
validation/test group may appear anywhere in train: not as a matched subject, not as a
same-label-other control, not as an opposite-direction control, not as a centroid
member, and not as a source for null/noise statistics. Centroids are train-only
artifacts and must record member IDs and member hashes.

## Proof Requirements

The proof gate should require matched signatures to beat every paired control for the same row:

- matched behavior margin must exceed each control margin by a registered delta;
- matched subject-output MSE must beat each control on fixed query probes;
- each behavior must pass per-target thresholds, not only aggregate thresholds;
- all-target control matrices must pass because aggregate control accuracy can hide off-target prototypes;
- train/validation/test split must be by `group_id` with transitive exclusion of
  every subject/control member;
- model-selection uses validation; final proof uses a separate heldout test split;
- probe set must be regenerated or loaded from a versioned file, stored or referenced
  in the artifact, and hashed into the dataset;
- sample counts must clear minimum thresholds for every `behavior x control_type`
  cell, not only per behavior.

## Registered Decode Policies

The primary proof policy must be selected before training the first proof candidate:

- `condition_only`: decode every matched/control condition with `z=0`;
- `subject_latent`: decode every matched/control condition with the same source
  subject latent `z`;
- `both`: report both policies and require both to pass separate gates.

No matched/control comparison may mix policies. The first implementation should use
`both` as diagnostics, then pre-register one primary policy before any proof claim.
For the current hypothesis, `condition_only` is the cleanest test of fixed activation
signature decode; `subject_latent` is the cleanest test of steering a fixed subject.

## Generation Changes

Add a new dataset-generation mode instead of modifying the existing LLM-format pipeline in place:

- `model_zoo/dataset_generation/paired_contrast_pipeline.py`
  - trains subject models using the existing `SubjectModelTrainer`;
  - extracts signatures with `ActivationSignatureExtractor`;
  - groups matched and control signatures into paired rows;
  - saves local JSONL/Parquet before any Hub upload.

- `model_zoo/dataset_generation/probe_provenance.py`
  - computes stable hashes for the signature probe examples;
  - computes stable hashes for extractor config and generation config;
  - stores provenance beside every dataset shard.

- `model_zoo/configs/dataset_gen/paired_contrast_proof.yaml`
  - enables only clean proof behaviors initially:
    - `sorted_ascending`;
    - `sorted_descending`;
    - `has_majority`;
    - `mountain_pattern`;
  - uses no inline Hub token;
  - writes local artifacts under `runs/paired_contrast_proof_*`.

## Training Changes

The hypernet training loader should consume grouped rows directly:

- matched decode uses the subject signature;
- same batch includes paired controls;
- losses compute matched-minus-control deltas using row-aligned controls;
- no random reconstruction of shuffled/noise controls is needed for the primary proof loss.

Existing post-hoc controls may remain as secondary diagnostics, but proof claims must come from paired controls.

## Evaluation Changes

Add a paired-contrast proof evaluator:

- reports matched and control behavior margins per row;
- reports matched-minus-control deltas by control type and behavior;
- reports subject-output MSE deltas by control type and behavior;
- fails if any behavior/control pair lacks enough validation/test groups;
- reports every metric separately for the registered decode policy;
- preserves the current clean proof gate as a legacy diagnostic, but does not use it as the primary gate for paired datasets.

## Minimum Viable Experiment

Start small:

- 4 behaviors;
- 50 validation groups per behavior minimum;
- one fixed architecture: 5 layers, 8 neurons, GELU, sequence length 5;
- no modification task;
- local artifact only;
- one training run with paired controls;
- Kepler review after dataset audit and after first formal result.

## Stop Criteria

If matched signatures still cannot beat paired controls on behavior margins while subject-output specificity remains positive, the broad behavior-decode claim should be written up as negative evidence. The narrower subject-output specificity claim can remain as a separate positive result.
