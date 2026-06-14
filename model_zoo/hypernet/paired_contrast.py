"""Helpers for proof-grade paired contrast datasets."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from .dataset_provenance import stable_hash_json
from .models import SubjectNetwork


REGISTERED_DECODE_POLICIES = {"condition_only", "subject_latent", "both"}
REGISTERED_CONTROL_TYPES = {
    "same_label_centroid",
    "same_label_other_subject",
    "different_label_same_direction",
    "opposite_direction",
    "null_signature",
    "noise_signature",
    "condition_ablation",
}
REGISTERED_PROOF_THRESHOLDS = {
    "min_mean_matched_minus_control_behavior_margin",
    "min_mean_control_minus_matched_subject_output_mse",
}
PROOF_THRESHOLD_METRIC_MAP = {
    "min_mean_matched_minus_control_behavior_margin": (
        "mean_matched_minus_control_behavior_margin",
        "behavior_margin_delta_passed",
    ),
    "min_mean_control_minus_matched_subject_output_mse": (
        "mean_control_minus_matched_subject_output_mse",
        "subject_output_mse_delta_passed",
    ),
}
REGISTERED_SIGNATURE_HASH_ALGORITHMS = {"stable_hash_json_float_list_v1"}
REQUIRED_PROBE_PROVENANCE_FIELDS = {
    "probe_set_id",
    "probe_examples",
    "probe_examples_hash",
    "behavior_suite_hash",
    "probe_generation_config_hash",
    "extractor_config_hash",
    "extractor_code_hash",
    "normalization_stats_hash",
    "dataset_source_hash",
    "git_commit",
}
EMPTY_EXTRACTOR_CODE_HASH = stable_hash_json("")
EMPTY_MAPPING_HASH = stable_hash_json({})
PROOF_GATE_EPSILON = 1e-12
WEIGHT_REFERENCE_FIELDS = ("weights_hash", "weights_uri")
SIGNATURE_REFERENCE_FIELDS = ("signature_hash", "signature_uri")
PATTERN_DIRECTIONS = {
    "sorted_ascending": "increasing",
    "increasing_pairs": "increasing",
    "sorted_descending": "decreasing",
    "decreasing_pairs": "decreasing",
}


def validate_registered_decode_policy(policy: str) -> str:
    """Return a registered decode policy or raise a clear error."""
    if policy not in REGISTERED_DECODE_POLICIES:
        supported = ", ".join(sorted(REGISTERED_DECODE_POLICIES))
        raise ValueError(f"Unsupported decode policy: {policy}. Supported: {supported}")
    return policy


def build_probe_provenance(
    probe_set_id: str,
    probe_examples: Sequence[Mapping],
    behavior_suite: Mapping,
    probe_generation_config: Mapping,
    extractor_config: Mapping,
    extractor_code: str | None = None,
    normalization_stats: Mapping | None = None,
    dataset_source: Mapping | None = None,
    git_commit: str | None = None,
) -> Dict:
    """Build content-addressed provenance for signature probes."""
    normalization_stats = normalization_stats or {}
    dataset_source = dataset_source or {}
    extractor_code = extractor_code or ""
    return {
        "probe_set_id": probe_set_id,
        "probe_examples": list(probe_examples),
        "probe_examples_hash": stable_hash_json(list(probe_examples)),
        "behavior_suite_hash": stable_hash_json(behavior_suite),
        "probe_generation_config_hash": stable_hash_json(probe_generation_config),
        "extractor_config_hash": stable_hash_json(extractor_config),
        "extractor_code_hash": stable_hash_json(extractor_code),
        "normalization_stats_hash": stable_hash_json(normalization_stats),
        "dataset_source_hash": stable_hash_json(dataset_source),
        "git_commit": git_commit,
    }


STORED_PROBE_EXTRACTOR_CODE_ID = "paired_contrast.extract_signature_with_stored_probes.v1"


def build_digit_probe_examples(
    n_examples: int,
    seed: int,
    seq_len: int = 5,
    base: int = 10,
) -> List[Dict]:
    """Build deterministic digit probes and store every probe sequence."""
    rng = random.Random(int(seed))
    probes = []
    for probe_index in range(int(n_examples)):
        probes.append({
            "probe_index": int(probe_index),
            "sequence": [rng.randrange(int(base)) for _ in range(int(seq_len))],
        })
    return probes


def build_stored_probe_provenance(
    probe_set_id: str,
    probe_examples: Sequence[Mapping],
    behavior_suite: Mapping,
    probe_generation_config: Mapping,
    extractor_config: Mapping,
    normalization_stats: Mapping,
    dataset_source: Mapping,
    git_commit: str,
) -> Dict:
    """Build proof-grade provenance for regenerated stored-probe signatures."""
    return build_probe_provenance(
        probe_set_id=probe_set_id,
        probe_examples=probe_examples,
        behavior_suite=behavior_suite,
        probe_generation_config=probe_generation_config,
        extractor_config=extractor_config,
        extractor_code=STORED_PROBE_EXTRACTOR_CODE_ID,
        normalization_stats=normalization_stats,
        dataset_source=dataset_source,
        git_commit=git_commit,
    )


def extract_signature_with_stored_probes(
    flat_weights: torch.Tensor,
    probe_examples: Sequence[Mapping],
    num_layers: int = 5,
    neurons_per_layer: int = 8,
    input_dim: int = 5,
) -> torch.Tensor:
    """Extract a deterministic activation signature using explicitly stored probes."""
    if not probe_examples:
        raise ValueError("probe_examples must be non-empty")
    probe_inputs = torch.tensor(
        [example["sequence"] for example in probe_examples],
        dtype=torch.float32,
    )
    if probe_inputs.ndim != 2 or probe_inputs.shape[1] != input_dim:
        raise ValueError(
            f"probe sequences must have shape [n, {input_dim}], got {tuple(probe_inputs.shape)}"
        )

    model = SubjectNetwork.from_weights(
        flat_weights.detach().cpu().float(),
        num_layers=num_layers,
        neurons_per_layer=neurons_per_layer,
        input_dim=input_dim,
    )
    model.eval()
    with torch.no_grad():
        activations = model.get_activations(probe_inputs)

    features: List[float] = []
    n_samples = int(probe_inputs.shape[0])
    for layer_idx in sorted(activations.keys()):
        layer_activations = activations[layer_idx]
        for neuron_idx in range(layer_activations.shape[1]):
            neuron_acts = layer_activations[:, neuron_idx]
            mean = neuron_acts.mean()
            std = neuron_acts.std(unbiased=False)
            features.append(float(mean.item()))
            features.append(float(std.item()))

            fft = torch.fft.fft(neuron_acts)
            fft_mag = torch.abs(fft)[: max(1, n_samples // 2)]
            fourier_features = fft_mag[:5].tolist()
            fourier_features += [0.0] * (5 - len(fourier_features))
            features.extend(float(value) for value in fourier_features)

            for input_idx in range(input_dim):
                corr = _safe_corrcoef(neuron_acts, probe_inputs[:, input_idx])
                features.append(float(corr))

            # This mirrors the existing generated-dataset signature path, which uses
            # post-activation statistics as the available pre-activation proxy.
            features.append(float(mean.item()))
            features.append(float(std.item()))

    return torch.tensor(features, dtype=torch.float32)


def _safe_corrcoef(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    value = torch.dot(left_centered, right_centered) / denom
    if not torch.isfinite(value):
        return 0.0
    return float(value.item())


def signature_hash_stable_float_list(signature_values: Sequence[float]) -> str:
    """Hash regenerated signature values with the registered sidecar algorithm."""
    return stable_hash_json([float(value) for value in signature_values])


def audit_regenerated_signature_sidecar(
    artifact: Mapping,
    sidecar: Mapping,
) -> Dict:
    """Validate artifact signature refs against stored regenerated signature vectors."""
    failures: List[str] = []
    algorithm = sidecar.get("signature_hash_algorithm")
    if algorithm not in REGISTERED_SIGNATURE_HASH_ALGORITHMS:
        failures.append(f"unsupported signature hash algorithm {algorithm}")

    signatures = sidecar.get("regenerated_signatures")
    if not isinstance(signatures, Mapping):
        failures.append("missing regenerated_signatures")
        signatures = {}

    n_checked = 0
    for payload in _iter_subject_payloads(artifact):
        subject_id = payload.get("subject_id")
        expected_hash = payload.get("signature_hash")
        if not subject_id:
            continue
        if not expected_hash:
            failures.append(f"missing signature_hash for {subject_id}")
            continue
        if subject_id not in signatures:
            failures.append(f"missing regenerated signature for {subject_id}")
            continue
        signature_values = signatures[subject_id]
        if not isinstance(signature_values, Sequence) or isinstance(
            signature_values, (str, bytes)
        ):
            failures.append(f"regenerated signature for {subject_id} is not a list")
            continue
        actual_hash = signature_hash_stable_float_list(signature_values)
        n_checked += 1
        if actual_hash != expected_hash:
            failures.append(f"signature hash mismatch for {subject_id}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_checked": int(n_checked),
        "signature_hash_algorithm": algorithm,
    }


def _iter_subject_payloads(artifact: Mapping):
    splits = artifact.get("splits", {}) or {}
    for groups in splits.values():
        if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            subject = group.get("subject")
            if isinstance(subject, Mapping):
                yield subject
            controls = group.get("controls", {}) or {}
            if not isinstance(controls, Mapping):
                continue
            for control in controls.values():
                if isinstance(control, Mapping) and control.get("subject_id"):
                    yield control


def group_subject_ids(group: Mapping) -> List[str]:
    """Collect every subject ID referenced by a paired contrast group."""
    ids: List[str] = []
    subject_id = group.get("subject", {}).get("subject_id")
    if subject_id:
        ids.append(str(subject_id))

    controls = group.get("controls", {}) or {}
    for control in controls.values():
        if not isinstance(control, Mapping):
            continue
        control_subject_id = control.get("subject_id")
        if control_subject_id:
            ids.append(str(control_subject_id))
        if not control.get("member_split"):
            for member_id in control.get("member_subject_ids", []) or []:
                ids.append(str(member_id))

    return ids


def pattern_direction(pattern: str | None) -> str | None:
    """Return a coarse behavior direction for contrast-control validation."""
    if pattern is None:
        return None
    return PATTERN_DIRECTIONS.get(pattern)


def validate_paired_group_schema(
    groups: Sequence[Mapping],
    required_control_types: Sequence[str] | None = None,
) -> Dict:
    """Validate paired contrast group structure and control semantics."""
    required_control_types = list(required_control_types or [])
    failures: List[str] = []
    seen_group_ids: set[str] = set()

    for row_idx, group in enumerate(groups):
        group_label = str(group.get("group_id", f"row[{row_idx}]"))
        group_id = group.get("group_id")
        if not group_id:
            failures.append(f"{group_label}: missing group_id")
        elif str(group_id) in seen_group_ids:
            failures.append(f"{group_label}: duplicate group_id {group_id}")
        else:
            seen_group_ids.add(str(group_id))

        target_pattern = group.get("target_pattern")
        if not target_pattern:
            failures.append(f"{group_label}: missing target_pattern")

        subject = group.get("subject")
        if not isinstance(subject, Mapping):
            failures.append(f"{group_label}: missing subject")
        elif not subject.get("subject_id"):
            failures.append(f"{group_label}: missing subject.subject_id")
        else:
            _validate_subject_references(failures, group_label, "subject", subject)

        controls = group.get("controls")
        if not isinstance(controls, Mapping):
            failures.append(f"{group_label}: missing controls")
            controls = {}

        for control_type in required_control_types:
            if control_type not in controls:
                failures.append(
                    f"{group_label}: missing required control {control_type}"
                )

        for control_type, control in controls.items():
            if not isinstance(control, Mapping):
                failures.append(f"{group_label}: control {control_type} is not an object")
                continue
            if control_type not in REGISTERED_CONTROL_TYPES:
                failures.append(f"{group_label}: unknown control type {control_type}")
                continue
            if control.get("subject_id"):
                _validate_subject_references(
                    failures,
                    group_label,
                    str(control_type),
                    control,
                )
            _validate_control_schema(
                failures,
                group_label,
                str(control_type),
                control,
                str(target_pattern) if target_pattern else None,
            )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_groups": int(len(groups)),
    }


def _validate_control_schema(
    failures: List[str],
    group_label: str,
    control_type: str,
    control: Mapping,
    target_pattern: str | None,
) -> None:
    if control_type in {
        "same_label_other_subject",
        "different_label_same_direction",
        "opposite_direction",
    } and not control.get("subject_id"):
        failures.append(f"{group_label}: {control_type} missing subject_id")

    if control_type == "same_label_centroid":
        member_ids = control.get("member_subject_ids")
        has_artifact_reference = all(
            control.get(key)
            for key in ("centroid_id", "member_split", "member_subject_ids_hash")
        )
        if not member_ids and not has_artifact_reference:
            failures.append(
                (
                    f"{group_label}: same_label_centroid missing member_subject_ids "
                    "or centroid_id/member_split/member_subject_ids_hash"
                )
            )
        if control.get("member_split") and not has_artifact_reference:
            failures.append(
                (
                    f"{group_label}: same_label_centroid member_split requires "
                    "centroid_id and member_subject_ids_hash"
                )
            )

    if control_type == "noise_signature" and "seed" not in control:
        failures.append(f"{group_label}: noise_signature missing seed")

    control_pattern = control.get("target_pattern")
    if control_type in {"same_label_other_subject", "same_label_centroid"}:
        if not control_pattern:
            failures.append(f"{group_label}: {control_type} missing target_pattern")
        elif control_pattern != target_pattern:
            failures.append(
                (
                    f"{group_label}: {control_type} target_pattern {control_pattern} "
                    f"does not match {target_pattern}"
                )
            )
    elif control_type == "different_label_same_direction":
        if not control_pattern:
            failures.append(
                f"{group_label}: different_label_same_direction missing target_pattern"
            )
        elif control_pattern == target_pattern:
            failures.append(
                f"{group_label}: different_label_same_direction must use a different label"
            )
        else:
            source_direction = pattern_direction(target_pattern)
            control_direction = pattern_direction(control_pattern)
            if source_direction is None or control_direction is None:
                failures.append(
                    (
                        f"{group_label}: different_label_same_direction requires "
                        "directional source and control patterns"
                    )
                )
            elif control_direction != source_direction:
                failures.append(
                    (
                        f"{group_label}: different_label_same_direction must share direction "
                        f"with {target_pattern}"
                    )
                )
    elif control_type == "opposite_direction":
        if not control_pattern:
            failures.append(
                f"{group_label}: opposite_direction missing target_pattern"
            )
        else:
            source_direction = pattern_direction(target_pattern)
            control_direction = pattern_direction(control_pattern)
            if source_direction is None or control_direction is None:
                failures.append(
                    (
                        f"{group_label}: opposite_direction requires directional "
                        "source and control patterns"
                    )
                )
            elif source_direction == control_direction:
                failures.append(
                    (
                        f"{group_label}: opposite_direction must have opposite direction "
                        f"from {target_pattern}"
                    )
                )


def _validate_subject_references(
    failures: List[str],
    group_label: str,
    payload_label: str,
    payload: Mapping,
) -> None:
    if not _has_any_reference(payload, WEIGHT_REFERENCE_FIELDS):
        failures.append(f"{group_label}: {payload_label} missing weights reference")
    if not _has_any_reference(payload, SIGNATURE_REFERENCE_FIELDS):
        failures.append(f"{group_label}: {payload_label} missing signature reference")


def validate_transitive_group_splits(
    groups_by_split: Mapping[str, Sequence[Mapping]],
) -> Dict:
    """Validate that no subject/control member crosses split boundaries."""
    split_subject_ids: Dict[str, set[str]] = {}
    failures: List[str] = []

    for split_name, groups in groups_by_split.items():
        split_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for group in groups:
            for subject_id in group_subject_ids(group):
                if subject_id in split_ids:
                    duplicate_ids.add(subject_id)
                split_ids.add(subject_id)
        split_subject_ids[split_name] = split_ids
        for subject_id in sorted(duplicate_ids):
            failures.append(
                f"subject/control member {subject_id} appears multiple times in {split_name}"
            )

    split_names = list(groups_by_split.keys())
    for idx, left_name in enumerate(split_names):
        for right_name in split_names[idx + 1:]:
            overlap = split_subject_ids[left_name] & split_subject_ids[right_name]
            for subject_id in sorted(overlap):
                failures.append(
                    f"subject/control member {subject_id} crosses {left_name}/{right_name}"
                )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "split_subject_ids": {
            split_name: sorted(subject_ids)
            for split_name, subject_ids in split_subject_ids.items()
        },
    }


def summarize_behavior_control_counts(
    groups: Iterable[Mapping],
) -> Dict[str, Dict[str, int]]:
    """Count paired contrast groups by behavior and control type."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for group in groups:
        pattern = group.get("target_pattern")
        if pattern is None:
            continue
        controls = group.get("controls", {}) or {}
        for control_type in controls:
            counts[str(pattern)][str(control_type)] += 1

    return {
        pattern: dict(control_counts)
        for pattern, control_counts in counts.items()
    }


def require_behavior_control_counts(
    counts: Mapping[str, Mapping[str, int]],
    required_behaviors: Sequence[str],
    required_control_types: Sequence[str],
    min_count: int,
) -> Dict:
    """Check every behavior/control cell has enough paired groups."""
    failures: List[str] = []
    for behavior in required_behaviors:
        behavior_counts = counts.get(behavior, {})
        for control_type in required_control_types:
            count = int(behavior_counts.get(control_type, 0))
            if count < min_count:
                failures.append(
                    (
                        f"{behavior}/{control_type} has {count} paired groups; "
                        f"requires at least {min_count}"
                    )
                )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "counts": {
            behavior: dict(control_counts)
            for behavior, control_counts in counts.items()
        },
        "min_count": int(min_count),
    }


def validate_paired_contrast_artifact(
    artifact: Mapping,
    required_behaviors: Sequence[str],
    required_control_types: Sequence[str],
    min_count: int,
    count_splits: Sequence[str] = ("validation", "test"),
) -> Dict:
    """Run the proof artifact gates from one fail-closed entry point."""
    failures: List[str] = []

    decode_policy = _validate_artifact_decode_policy(artifact, failures)
    probe_provenance = _validate_artifact_probe_provenance(artifact, failures)
    source_pool_preflight = _validate_source_pool_preflight(artifact, failures)
    splits = artifact.get("splits")
    if not isinstance(splits, Mapping):
        failures.append("missing splits")
        splits = {}

    for split_name in ("train", "validation", "test"):
        if split_name not in splits:
            failures.append(f"missing split {split_name}")

    groups_by_split: Dict[str, Sequence[Mapping]] = {}
    for split_name, groups in splits.items():
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
            groups_by_split[str(split_name)] = groups
        else:
            failures.append(f"split {split_name} is not a list")
            groups_by_split[str(split_name)] = []

    all_groups: List[Mapping] = []
    for split_name in ("train", "validation", "test"):
        all_groups.extend(groups_by_split.get(split_name, []))

    schema = validate_paired_group_schema(
        all_groups,
        required_control_types=required_control_types,
    )
    failures.extend(schema["failures"])

    split_validation = validate_transitive_group_splits(groups_by_split)
    failures.extend(split_validation["failures"])

    counts = _validate_per_split_behavior_control_counts(
        groups_by_split,
        required_behaviors=required_behaviors,
        required_control_types=required_control_types,
        min_count=min_count,
        count_splits=count_splits,
    )
    failures.extend(counts["failures"])

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "decode_policy": decode_policy,
        "probe_provenance": probe_provenance,
        "source_pool_preflight": source_pool_preflight,
        "schema": schema,
        "splits": split_validation,
        "counts": counts,
    }


def build_paired_contrast_artifact_from_subjects(
    subjects_by_split: Mapping[str, Sequence[Mapping]],
    decode_policy: str,
    probe_provenance: Mapping,
    required_behaviors: Sequence[str],
    required_control_types: Sequence[str],
    min_count: int,
    proof_splits: Sequence[str] = ("validation", "test"),
    centroid_references: Mapping[str, Mapping] | None = None,
    noise_seed: int = 0,
) -> Dict:
    """Build disjoint paired contrast groups from subject metadata and validate them."""
    centroid_references = centroid_references or {}
    preflight = _validate_subject_source_pool(subjects_by_split)
    splits: Dict[str, List[Dict]] = {}
    for split_name in ("train", "validation", "test"):
        splits[split_name] = _build_paired_groups_for_split(
            split_name=split_name,
            subjects=subjects_by_split.get(split_name, []),
            required_control_types=required_control_types,
            centroid_references=centroid_references,
            noise_seed=noise_seed,
        )

    artifact = {
        "decode_policy": decode_policy,
        "probe_provenance": dict(probe_provenance),
        "source_pool_preflight": preflight,
        "splits": splits,
    }
    validation = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=required_behaviors,
        required_control_types=required_control_types,
        min_count=min_count,
        count_splits=proof_splits,
    )
    failures = list(preflight["failures"]) + list(validation["failures"])
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "artifact": artifact,
        "preflight": preflight,
        "validation": validation,
    }


def _validate_subject_source_pool(
    subjects_by_split: Mapping[str, Sequence[Mapping]],
) -> Dict:
    failures: List[str] = []
    subject_splits: Dict[str, str] = {}
    for split_name, subjects in subjects_by_split.items():
        split_seen: set[str] = set()
        for subject in subjects:
            subject_id = subject.get("subject_id")
            if not subject_id:
                continue
            subject_id = str(subject_id)
            if subject_id in split_seen:
                failures.append(
                    f"input subject {subject_id} appears multiple times in {split_name}"
                )
            split_seen.add(subject_id)

            previous_split = subject_splits.get(subject_id)
            if previous_split is not None and previous_split != split_name:
                failures.append(
                    f"input subject {subject_id} crosses {previous_split}/{split_name}"
                )
            subject_splits.setdefault(subject_id, str(split_name))

            if subject.get("target_pattern"):
                if not _has_any_reference(subject, WEIGHT_REFERENCE_FIELDS):
                    failures.append(
                        f"input subject {subject_id} missing weights reference"
                    )
                if not _has_any_reference(subject, SIGNATURE_REFERENCE_FIELDS):
                    failures.append(
                        f"input subject {subject_id} missing signature reference"
                    )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
    }


def _build_paired_groups_for_split(
    split_name: str,
    subjects: Sequence[Mapping],
    required_control_types: Sequence[str],
    centroid_references: Mapping[str, Mapping],
    noise_seed: int,
) -> List[Dict]:
    groups: List[Dict] = []
    used_subject_ids: set[str] = set()
    subject_rows = [
        subject for subject in subjects
        if subject.get("subject_id") and subject.get("target_pattern")
    ]

    for subject in subject_rows:
        subject_id = str(subject["subject_id"])
        target_pattern = str(subject["target_pattern"])
        if subject_id in used_subject_ids:
            continue

        controls: Dict[str, Dict] = {}
        control_subject_ids: List[str] = []
        for control_type in required_control_types:
            control = _build_required_control(
                control_type=control_type,
                source=subject,
                subjects=subject_rows,
                unavailable_subject_ids=used_subject_ids | {subject_id} | set(control_subject_ids),
                centroid_references=centroid_references,
                noise_seed=noise_seed + len(groups),
            )
            if control is None:
                continue
            controls[control_type] = control
            if control.get("subject_id"):
                control_subject_ids.append(str(control["subject_id"]))

        groups.append({
            "group_id": f"{split_name}:{subject_id}",
            "target_pattern": target_pattern,
            "subject": _subject_reference_payload(subject),
            "controls": controls,
        })
        used_subject_ids.add(subject_id)
        used_subject_ids.update(control_subject_ids)

    return groups


def _build_required_control(
    control_type: str,
    source: Mapping,
    subjects: Sequence[Mapping],
    unavailable_subject_ids: set[str],
    centroid_references: Mapping[str, Mapping],
    noise_seed: int,
) -> Dict | None:
    target_pattern = str(source["target_pattern"])
    if control_type == "same_label_centroid":
        centroid = centroid_references.get(target_pattern)
        if not isinstance(centroid, Mapping):
            return None
        return {"target_pattern": target_pattern, **dict(centroid)}

    if control_type == "noise_signature":
        return {"seed": int(noise_seed)}

    if control_type == "null_signature":
        return {}

    if control_type == "condition_ablation":
        return {"target_pattern": target_pattern}

    if control_type == "same_label_other_subject":
        candidate = _find_control_subject(
            source=source,
            subjects=subjects,
            unavailable_subject_ids=unavailable_subject_ids,
            predicate=lambda pattern: pattern == target_pattern,
        )
    elif control_type == "different_label_same_direction":
        source_direction = pattern_direction(target_pattern)
        candidate = _find_control_subject(
            source=source,
            subjects=subjects,
            unavailable_subject_ids=unavailable_subject_ids,
            predicate=lambda pattern: (
                source_direction is not None
                and pattern != target_pattern
                and pattern_direction(pattern) == source_direction
            ),
        )
    elif control_type == "opposite_direction":
        source_direction = pattern_direction(target_pattern)
        candidate = _find_control_subject(
            source=source,
            subjects=subjects,
            unavailable_subject_ids=unavailable_subject_ids,
            predicate=lambda pattern: (
                source_direction is not None
                and pattern_direction(pattern) is not None
                and pattern_direction(pattern) != source_direction
            ),
        )
    else:
        return None

    if candidate is None:
        return None
    return _subject_reference_payload(candidate)


def _subject_reference_payload(subject: Mapping) -> Dict:
    payload = {
        "subject_id": str(subject["subject_id"]),
    }
    if subject.get("target_pattern"):
        payload["target_pattern"] = str(subject["target_pattern"])
    for field in (*WEIGHT_REFERENCE_FIELDS, *SIGNATURE_REFERENCE_FIELDS):
        if subject.get(field):
            payload[field] = subject[field]
    return payload


def _has_any_reference(subject: Mapping, fields: Sequence[str]) -> bool:
    return any(bool(subject.get(field)) for field in fields)


def _find_control_subject(
    source: Mapping,
    subjects: Sequence[Mapping],
    unavailable_subject_ids: set[str],
    predicate,
) -> Mapping | None:
    source_id = str(source["subject_id"])
    for candidate in subjects:
        candidate_id = str(candidate.get("subject_id"))
        candidate_pattern = candidate.get("target_pattern")
        if (
            not candidate_id
            or candidate_id == source_id
            or candidate_id in unavailable_subject_ids
            or candidate_pattern is None
        ):
            continue
        if predicate(str(candidate_pattern)):
            return candidate
    return None


def evaluate_paired_contrast_predictions(
    artifact: Mapping,
    predictions: Mapping[str, Mapping],
    required_behaviors: Sequence[str],
    required_control_types: Sequence[str],
    min_count: int,
    proof_splits: Sequence[str] = ("validation", "test"),
    proof_thresholds: Mapping[str, float] | None = None,
) -> Dict:
    """Evaluate precomputed paired predictions with proof-gated contrast metrics."""
    validator = validate_paired_contrast_artifact(
        artifact,
        required_behaviors=required_behaviors,
        required_control_types=required_control_types,
        min_count=min_count,
        count_splits=proof_splits,
    )
    failures: List[str] = list(validator["failures"])
    if not validator["passed"]:
        return {
            "passed": False,
            "failures": failures,
            "validator": validator,
            "metrics": {},
        }

    rows: List[Dict[str, Any]] = []
    splits = artifact.get("splits", {}) or {}
    for split_name in proof_splits:
        for group in splits.get(split_name, []) or []:
            group_id = str(group.get("group_id"))
            prediction = predictions.get(group_id)
            if not isinstance(prediction, Mapping):
                failures.append(f"missing prediction for group {group_id}")
                continue

            matched = prediction.get("matched")
            if not isinstance(matched, Mapping):
                failures.append(f"{group_id} missing matched prediction")
                continue

            matched_behavior_margin = _metric_value(
                matched,
                "behavior_margin",
                f"{group_id} matched",
                failures,
            )
            matched_subject_output_mse = _metric_value(
                matched,
                "subject_output_mse",
                f"{group_id} matched",
                failures,
            )

            control_predictions = prediction.get("controls", {}) or {}
            if not isinstance(control_predictions, Mapping):
                failures.append(f"{group_id} controls prediction is not an object")
                continue

            for control_type in required_control_types:
                control_prediction = control_predictions.get(control_type)
                if not isinstance(control_prediction, Mapping):
                    failures.append(
                        f"{group_id} missing control prediction {control_type}"
                    )
                    continue
                control_behavior_margin = _metric_value(
                    control_prediction,
                    "behavior_margin",
                    f"{group_id} {control_type}",
                    failures,
                )
                control_subject_output_mse = _metric_value(
                    control_prediction,
                    "subject_output_mse",
                    f"{group_id} {control_type}",
                    failures,
                )
                if (
                    matched_behavior_margin is None
                    or matched_subject_output_mse is None
                    or control_behavior_margin is None
                    or control_subject_output_mse is None
                ):
                    continue
                rows.append({
                    "split": str(split_name),
                    "behavior": str(group.get("target_pattern")),
                    "control_type": str(control_type),
                    "matched_behavior_margin": matched_behavior_margin,
                    "control_behavior_margin": control_behavior_margin,
                    "matched_minus_control_behavior_margin": (
                        matched_behavior_margin - control_behavior_margin
                    ),
                    "matched_subject_output_mse": matched_subject_output_mse,
                    "control_subject_output_mse": control_subject_output_mse,
                    "control_minus_matched_subject_output_mse": (
                        control_subject_output_mse - matched_subject_output_mse
                    ),
                })

    metrics = (
        {}
        if failures
        else _summarize_paired_prediction_rows(rows)
    )
    proof_gates: Dict = {}
    if not failures and proof_thresholds is not None:
        proof_gates = _evaluate_paired_proof_gates(metrics, proof_thresholds)
        failures.extend(proof_gates["failures"])

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "validator": validator,
        "metrics": metrics,
        "proof_gates": proof_gates,
    }


def _validate_artifact_decode_policy(artifact: Mapping, failures: List[str]) -> Dict:
    policy = artifact.get("decode_policy")
    if not policy:
        failures.append("missing decode_policy")
        return {"passed": False, "policy": None}

    try:
        validate_registered_decode_policy(str(policy))
    except ValueError as exc:
        failures.append(str(exc))
        return {"passed": False, "policy": policy}

    return {"passed": True, "policy": str(policy)}


def _metric_value(
    metrics: Mapping,
    metric_name: str,
    label: str,
    failures: List[str],
) -> float | None:
    if metric_name not in metrics:
        failures.append(f"{label} missing {metric_name}")
        return None
    try:
        value = float(metrics[metric_name])
    except (TypeError, ValueError):
        failures.append(f"{label} {metric_name} is not numeric")
        return None
    if not math.isfinite(value):
        failures.append(f"{label} {metric_name} is not finite")
        return None
    return value


def _summarize_paired_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> Dict:
    by_split_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split_rows[str(row["split"])].append(row)

    by_split: Dict[str, Dict] = {}
    for split_name, split_rows in by_split_rows.items():
        behavior_rows: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in split_rows:
            behavior_rows[str(row["behavior"])][str(row["control_type"])].append(row)

        by_behavior: Dict[str, Dict[str, Dict]] = {}
        for behavior, control_rows in behavior_rows.items():
            by_behavior[behavior] = {
                control_type: _summarize_metric_rows(control_specific_rows)
                for control_type, control_specific_rows in control_rows.items()
            }

        by_split[split_name] = {
            "aggregate": _summarize_metric_rows(split_rows),
            "by_behavior": by_behavior,
        }

    return {
        "n_pairs": int(len(rows)),
        "aggregate": _summarize_metric_rows(rows),
        "by_split": by_split,
    }


def _summarize_metric_rows(rows: Sequence[Mapping[str, Any]]) -> Dict:
    metric_names = [
        "matched_behavior_margin",
        "control_behavior_margin",
        "matched_minus_control_behavior_margin",
        "matched_subject_output_mse",
        "control_subject_output_mse",
        "control_minus_matched_subject_output_mse",
    ]
    summary: Dict[str, float | int | None] = {"n": int(len(rows))}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary[f"mean_{metric_name}"] = (
            sum(values) / len(values)
            if values
            else None
        )
    return summary


def _evaluate_paired_proof_gates(
    metrics: Mapping,
    proof_thresholds: Mapping[str, float],
) -> Dict:
    if "by_control_type" in proof_thresholds:
        return _evaluate_control_specific_paired_proof_gates(
            metrics,
            proof_thresholds,
        )

    failures: List[str] = []
    for key in sorted(set(proof_thresholds) - REGISTERED_PROOF_THRESHOLDS):
        failures.append(f"unsupported proof threshold {key}")
    min_behavior_delta = _threshold_value(
        proof_thresholds,
        "min_mean_matched_minus_control_behavior_margin",
        failures,
    )
    min_mse_delta = _threshold_value(
        proof_thresholds,
        "min_mean_control_minus_matched_subject_output_mse",
        failures,
    )
    by_split: Dict[str, Dict] = {}
    if min_behavior_delta is None or min_mse_delta is None:
        return {
            "passed": False,
            "failures": failures,
            "thresholds": dict(proof_thresholds),
            "by_split": by_split,
        }

    for split_name, split_metrics in metrics.get("by_split", {}).items():
        split_results: Dict[str, Dict] = {}
        for behavior, control_metrics in split_metrics.get("by_behavior", {}).items():
            behavior_results: Dict[str, Dict] = {}
            for control_type, cell in control_metrics.items():
                behavior_delta = cell["mean_matched_minus_control_behavior_margin"]
                mse_delta = cell["mean_control_minus_matched_subject_output_mse"]
                behavior_passed = (
                    behavior_delta + PROOF_GATE_EPSILON >= min_behavior_delta
                )
                mse_passed = mse_delta + PROOF_GATE_EPSILON >= min_mse_delta
                if not behavior_passed:
                    failures.append(
                        (
                            f"{split_name}/{behavior}/{control_type} behavior margin "
                            f"delta {behavior_delta} below {min_behavior_delta}"
                        )
                    )
                if not mse_passed:
                    failures.append(
                        (
                            f"{split_name}/{behavior}/{control_type} subject-output "
                            f"MSE delta {mse_delta} below {min_mse_delta}"
                        )
                    )
                behavior_results[str(control_type)] = {
                    "n": int(cell["n"]),
                    "mean_matched_minus_control_behavior_margin": behavior_delta,
                    "mean_control_minus_matched_subject_output_mse": mse_delta,
                    "min_mean_matched_minus_control_behavior_margin": min_behavior_delta,
                    "min_mean_control_minus_matched_subject_output_mse": min_mse_delta,
                    "behavior_margin_delta_passed": bool(behavior_passed),
                    "subject_output_mse_delta_passed": bool(mse_passed),
                    "passed": bool(behavior_passed and mse_passed),
                }
            split_results[str(behavior)] = behavior_results
        by_split[str(split_name)] = split_results

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "thresholds": dict(proof_thresholds),
        "by_split": by_split,
    }


def _evaluate_control_specific_paired_proof_gates(
    metrics: Mapping,
    proof_thresholds: Mapping,
) -> Dict:
    failures: List[str] = []
    top_level_keys = set(proof_thresholds)
    for key in sorted(top_level_keys - {"by_control_type"}):
        failures.append(f"unsupported proof threshold {key} with by_control_type")

    raw_by_control_type = proof_thresholds.get("by_control_type")
    if not isinstance(raw_by_control_type, Mapping):
        failures.append("proof threshold by_control_type is not an object")
        raw_by_control_type = {}

    encountered_control_types = _control_types_from_paired_metrics(metrics)
    configured_control_types = {str(key) for key in raw_by_control_type}
    for control_type in sorted(encountered_control_types - configured_control_types):
        failures.append(f"missing proof thresholds for control_type {control_type}")
    for control_type in sorted(configured_control_types - encountered_control_types):
        failures.append(f"unsupported proof thresholds for control_type {control_type}")

    thresholds_by_control_type: Dict[str, Dict[str, float]] = {}
    for control_type, raw_thresholds in raw_by_control_type.items():
        control_type = str(control_type)
        if not isinstance(raw_thresholds, Mapping):
            failures.append(f"proof thresholds for control_type {control_type} are not an object")
            continue
        if not raw_thresholds:
            failures.append(f"missing registered proof thresholds for control_type {control_type}")
            continue

        parsed_thresholds: Dict[str, float] = {}
        for key in sorted(raw_thresholds):
            if key not in REGISTERED_PROOF_THRESHOLDS:
                failures.append(f"unsupported proof threshold {control_type}.{key}")
                continue
            value = _threshold_value(
                raw_thresholds,
                str(key),
                failures,
                label=f"{control_type}.{key}",
            )
            if value is not None:
                parsed_thresholds[str(key)] = value
        if parsed_thresholds:
            thresholds_by_control_type[control_type] = parsed_thresholds

    by_split: Dict[str, Dict] = {}
    for split_name, split_metrics in metrics.get("by_split", {}).items():
        split_results: Dict[str, Dict] = {}
        for behavior, control_metrics in split_metrics.get("by_behavior", {}).items():
            behavior_results: Dict[str, Dict] = {}
            for control_type, cell in control_metrics.items():
                control_type = str(control_type)
                thresholds = thresholds_by_control_type.get(control_type, {})
                cell_result = _evaluate_proof_gate_cell(
                    split_name=str(split_name),
                    behavior=str(behavior),
                    control_type=control_type,
                    cell=cell,
                    thresholds=thresholds,
                    failures=failures,
                )
                behavior_results[control_type] = cell_result
            split_results[str(behavior)] = behavior_results
        by_split[str(split_name)] = split_results

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "thresholds": {
            "by_control_type": {
                control_type: dict(thresholds)
                for control_type, thresholds in thresholds_by_control_type.items()
            },
        },
        "by_split": by_split,
    }


def _control_types_from_paired_metrics(metrics: Mapping) -> set[str]:
    control_types: set[str] = set()
    for split_metrics in metrics.get("by_split", {}).values():
        for control_metrics in split_metrics.get("by_behavior", {}).values():
            control_types.update(str(control_type) for control_type in control_metrics)
    return control_types


def _evaluate_proof_gate_cell(
    split_name: str,
    behavior: str,
    control_type: str,
    cell: Mapping,
    thresholds: Mapping[str, float],
    failures: List[str],
) -> Dict:
    cell_result: Dict[str, Any] = {
        "n": int(cell["n"]),
        "mean_matched_minus_control_behavior_margin": cell[
            "mean_matched_minus_control_behavior_margin"
        ],
        "mean_control_minus_matched_subject_output_mse": cell[
            "mean_control_minus_matched_subject_output_mse"
        ],
    }
    passed = True
    for threshold_key, threshold_value in thresholds.items():
        metric_name, passed_key = PROOF_THRESHOLD_METRIC_MAP[str(threshold_key)]
        metric_value = float(cell[metric_name])
        metric_passed = metric_value + PROOF_GATE_EPSILON >= float(threshold_value)
        cell_result[str(threshold_key)] = float(threshold_value)
        cell_result[passed_key] = bool(metric_passed)
        if not metric_passed:
            passed = False
            failures.append(
                (
                    f"{split_name}/{behavior}/{control_type} {metric_name} "
                    f"{metric_value} below {threshold_value}"
                )
            )
    cell_result["passed"] = bool(passed)
    return cell_result


def _threshold_value(
    thresholds: Mapping[str, float],
    key: str,
    failures: List[str],
    label: str | None = None,
) -> float | None:
    label = label or key
    if key not in thresholds:
        failures.append(f"missing proof threshold {label}")
        return None
    try:
        value = float(thresholds[key])
    except (TypeError, ValueError):
        failures.append(f"proof threshold {label} is not numeric")
        return None
    if not math.isfinite(value):
        failures.append(f"proof threshold {label} is not finite")
        return None
    return value


def _validate_artifact_probe_provenance(artifact: Mapping, failures: List[str]) -> Dict:
    probe_provenance = artifact.get("probe_provenance")
    if not isinstance(probe_provenance, Mapping):
        failures.append("missing probe_provenance")
        return {
            "passed": False,
            "missing_fields": sorted(REQUIRED_PROBE_PROVENANCE_FIELDS),
            "invalid_fields": [],
        }

    missing_fields = sorted(
        field
        for field in REQUIRED_PROBE_PROVENANCE_FIELDS
        if field not in probe_provenance
    )
    for field in missing_fields:
        failures.append(f"probe_provenance missing {field}")

    invalid_fields: List[str] = []
    git_commit = probe_provenance.get("git_commit")
    if "git_commit" not in missing_fields and (
        not isinstance(git_commit, str) or not git_commit.strip()
    ):
        invalid_fields.append("git_commit")
        failures.append("probe_provenance git_commit is empty")

    if "probe_examples" not in missing_fields:
        probe_examples = probe_provenance.get("probe_examples")
        if (
            not isinstance(probe_examples, Sequence)
            or isinstance(probe_examples, (str, bytes))
        ):
            invalid_fields.append("probe_examples")
            failures.append("probe_provenance probe_examples is not a sequence")
        elif len(probe_examples) == 0:
            invalid_fields.append("probe_examples")
            failures.append("probe_provenance probe_examples is empty")
        elif (
            "probe_examples_hash" not in missing_fields
            and stable_hash_json(list(probe_examples))
            != probe_provenance.get("probe_examples_hash")
        ):
            invalid_fields.append("probe_examples_hash")
            failures.append(
                "probe_provenance probe_examples_hash does not match probe_examples"
            )

    if (
        "extractor_code_hash" not in missing_fields
        and probe_provenance.get("extractor_code_hash") == EMPTY_EXTRACTOR_CODE_HASH
    ):
        invalid_fields.append("extractor_code_hash")
        failures.append(
            "probe_provenance extractor_code_hash matches empty extractor_code"
        )

    for field in ("normalization_stats_hash", "dataset_source_hash"):
        if field not in missing_fields and probe_provenance.get(field) == EMPTY_MAPPING_HASH:
            invalid_fields.append(field)
            failures.append(f"probe_provenance {field} matches empty mapping")

    return {
        "passed": len(missing_fields) == 0 and len(invalid_fields) == 0,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
    }


def _validate_source_pool_preflight(artifact: Mapping, failures: List[str]) -> Dict:
    preflight = artifact.get("source_pool_preflight")
    if not isinstance(preflight, Mapping):
        failures.append("missing source_pool_preflight")
        return {
            "passed": False,
            "failures": ["missing source_pool_preflight"],
        }

    preflight_failures = [
        str(failure)
        for failure in preflight.get("failures", []) or []
    ]
    if preflight.get("passed") is not True:
        failures.append("source_pool_preflight failed")
        for failure in preflight_failures:
            failures.append(f"source_pool_preflight: {failure}")
        return {
            "passed": False,
            "failures": preflight_failures,
        }

    return {
        "passed": True,
        "failures": preflight_failures,
    }


def _validate_per_split_behavior_control_counts(
    groups_by_split: Mapping[str, Sequence[Mapping]],
    required_behaviors: Sequence[str],
    required_control_types: Sequence[str],
    min_count: int,
    count_splits: Sequence[str],
) -> Dict:
    failures: List[str] = []
    per_split: Dict[str, Dict] = {}

    for split_name in count_splits:
        split_counts = require_behavior_control_counts(
            summarize_behavior_control_counts(groups_by_split.get(split_name, [])),
            required_behaviors=required_behaviors,
            required_control_types=required_control_types,
            min_count=min_count,
        )
        split_failures = [
            f"{split_name}: {failure}"
            for failure in split_counts["failures"]
        ]
        failures.extend(split_failures)
        per_split[str(split_name)] = {
            **split_counts,
            "failures": split_failures,
        }

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "per_split": per_split,
        "min_count": int(min_count),
    }
