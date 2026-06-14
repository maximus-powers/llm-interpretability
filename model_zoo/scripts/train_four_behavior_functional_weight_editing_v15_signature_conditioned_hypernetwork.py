"""V15 functional editing via signature-conditioned hypernetworks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import generate_four_behavior_decoder_source_pools as poolgen  # noqa: E402
import train_four_behavior_functional_weight_editing_v14_signature_gated_subspace_task_vectors as v14  # noqa: E402
import train_four_behavior_functional_weight_editing_v10_v9_conditioned_delta as v10  # noqa: E402
import train_four_behavior_representation_steering as v1  # noqa: E402
import train_four_behavior_representation_steering_v9_source_invariant_target_attractor as v9  # noqa: E402
from evaluate_four_behavior_source_generation_feasibility import (  # noqa: E402
    PATTERNS,
    build_candidate_pools,
    build_heldout_sequences,
    build_suite,
    summarize_candidate_pools,
)
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402


SEED_BEHAVIOR_STRIDE = 100000
POOL_CONFIGS = {
    "train": {
        "base_seed": 75300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 76300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 77300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
DEFAULT_POOL_DIR = REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v15_pools"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork"
)
PREREG_PATH = (
    REPO_ROOT
    / "docs"
    / "preregistrations"
    / "four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork.md"
)
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_POOL_SCOPE = "four_behavior_functional_weight_editing_v15_source_pool"
SOURCE_AUDIT_SCOPE = "four_behavior_functional_weight_editing_v15_source_pool_construction"
FINAL_REDACTED_SCOPE = (
    "redacted_final_functional_weight_editing_v15_source_pool_audit_surface_only"
)
DEVELOPMENT_SCOPE = (
    "four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork_development"
)
FINAL_SCOPE = "four_behavior_functional_weight_editing_v15_signature_conditioned_hypernetwork_final"
EDITOR_METHOD = "signature_conditioned_delta_hypernetwork_v15"
V15_FINAL_RAW = DEFAULT_POOL_DIR / "final_subjects.json"

SOURCE_WEIGHT_DIM = 345
SIGNATURE_DIM = 560
BEHAVIOR_DIM = len(PATTERNS)
PAIR_DIM = len(PATTERNS) * (len(PATTERNS) - 1)
EDITOR_INPUT_DIM = SOURCE_WEIGHT_DIM + SIGNATURE_DIM + SIGNATURE_DIM + BEHAVIOR_DIM + BEHAVIOR_DIM + PAIR_DIM
EDITOR_HIDDEN_DIMS = [768, 768, 384]
EDITOR_SEED = 20260615
CONTROL_MODEL_SEEDS = {
    "target_label_only_hypernetwork": 20260621,
    "source_signature_hypernetwork": 20260622,
    "shuffled_signature_hypernetwork": 20260623,
    "nearest_train_target_signature_hypernetwork": 20260624,
    "random_signature_hypernetwork": 20260625,
    "random_neuron_permutation_hypernetwork": 20260626,
    "target_weight_mse_only_hypernetwork": 20260627,
    "functional_only_hypernetwork": 20260628,
    "signature_only_hypernetwork": 20260629,
}
TRAINING_STEPS = 3000
TRAINING_BATCH_SIZE = 64
TRAINING_LR = 1e-3
TRAINING_BETAS = (0.9, 0.999)
TRAINING_WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 10.0
SCALE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
MATCHED_ALPHA = 0.975
PRIMARY_LOSS_WEIGHTS = {
    "compatible": 0.01,
    "conflict": 2.0,
    "delta_norm": 0.0001,
    "signature": 0.05,
    "source_l2": 0.0005,
    "target_bce": 4.0,
    "weight_mse": 1.0,
}

REQUIRED_NON_RANDOM_CONTROL_TYPES = [
    "no_edit",
    "v13_no_signature_support_optimizer",
    "v14_signature_gated_task_vector",
    "aligned_full_nearest_target_retrieval",
    "aligned_interpolation_alpha_0.975",
    "target_label_only_hypernetwork",
    "source_signature_hypernetwork",
    "shuffled_signature_hypernetwork",
    "nearest_train_target_signature_hypernetwork",
    "random_signature_hypernetwork",
    "random_neuron_permutation_hypernetwork",
    "target_weight_mse_only_hypernetwork",
    "functional_only_hypernetwork",
    "signature_only_hypernetwork",
]
THRESHOLDS = {
    **v14.THRESHOLDS,
    "expected_controls_per_record": 30,
    "random_controls_per_record": 16,
}
PREREGISTERED_ARG_VALUES = {
    **v14.PREREGISTERED_ARG_VALUES,
    "random_controls": 16,
}
COMBINED_FINAL_SUMMARY_ALLOWLIST = {
    "accepted_counts_by_behavior",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
}
FINAL_REDACTED_ALLOWLIST = {
    "behavior_suite_hashes",
    "candidate_pool_summary_hash",
    "claim_scope",
    "config_hash",
    "pool",
    "pool_file_sha256",
    "pool_redacted_payload_sha256",
    "summary",
    "summary_payload_sha256",
}
FORBIDDEN_FINAL_DETAIL_KEYS = v10.FORBIDDEN_FINAL_DETAIL_KEYS
PRIOR_FINAL_RAW_PATHS = {
    *v14.PRIOR_FINAL_RAW_PATHS,
    REPO_ROOT / "runs" / "four_behavior_functional_weight_editing_v14_pools" / "final_subjects.json",
}


class SignatureConditionedDeltaHypernetwork(nn.Module):
    """Small MLP editor that emits a raw delta and diagnostic scale."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            layers = []
            input_dim = EDITOR_INPUT_DIM
            for hidden_dim in EDITOR_HIDDEN_DIMS:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
                input_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.delta_head = nn.Linear(input_dim, SOURCE_WEIGHT_DIM)
            self.scale_head = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        delta = self.delta_head(hidden)
        diagnostic_scale = 1.5 * torch.sigmoid(self.scale_head(hidden).squeeze(-1))
        return delta, diagnostic_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["generate-pools", "development", "final"],
        required=True,
    )
    parser.add_argument(
        "--pool-dir",
        default=str(DEFAULT_POOL_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=160)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    parser.add_argument("--random-controls", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_preregistered_args(args)
    pool_dir = REPO_ROOT / args.pool_dir
    output_dir = REPO_ROOT / args.output_dir
    assert_preregistered_pool_dir(pool_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate-pools":
        result = generate_pools(args, pool_dir)
    elif args.phase == "development":
        result = run_development(args, pool_dir, output_dir)
    else:
        result = run_final(args, pool_dir, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("passed", False):
        raise SystemExit(1)


def validate_preregistered_args(args: argparse.Namespace) -> None:
    failures = []
    for name, expected in PREREGISTERED_ARG_VALUES.items():
        actual = getattr(args, name)
        if actual != expected:
            failures.append(f"{name}={actual!r} does not match preregistered {expected!r}")
    if failures:
        raise ValueError("Non-preregistered V15 parameter override: " + "; ".join(failures))


def development_input_paths(pool_dir: Path) -> Dict[str, Path]:
    return {
        "combined_audit": pool_dir / "combined_audit.json",
        "development": pool_dir / "development_subjects.json",
        "final_redacted_audit": pool_dir / "final_redacted_audit.json",
        "train": pool_dir / "train_subjects.json",
    }


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    paths = development_input_paths(pool_dir)
    assert_no_forbidden_final_raw_paths(paths.values(), allow_v15_final=False)
    payload = train_and_evaluate(
        train_path=paths["train"],
        eval_path=paths["development"],
        combined_audit_path=paths["combined_audit"],
        final_redacted_path=paths["final_redacted_audit"],
        output_dir=output_dir,
        phase="development",
        random_controls=args.random_controls,
        allow_eval_final_raw=False,
    )
    if payload["passed"]:
        payload["next_action"] = "eligible_for_one_shot_final_eval_without_method_changes"
    else:
        payload["next_action"] = "log_negative_development_result_do_not_open_final_raw"
    result_path = output_dir / "development_results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_final(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    development_path = output_dir / "development_results.json"
    if not development_path.exists():
        raise FileNotFoundError("development_results.json is required before final eval")
    development = json.loads(development_path.read_text())
    authorization_failures = validate_development_authorizes_final(
        development=development,
        pool_dir=pool_dir,
    )
    if authorization_failures:
        raise RuntimeError(
            "development artifact does not authorize final raw evaluation: "
            + "; ".join(authorization_failures)
        )
    eval_path = pool_dir / "final_subjects.json"
    assert_no_forbidden_final_raw_paths([eval_path], allow_v15_final=True)
    payload = train_and_evaluate(
        train_path=pool_dir / "train_subjects.json",
        eval_path=eval_path,
        combined_audit_path=pool_dir / "combined_audit.json",
        final_redacted_path=pool_dir / "final_redacted_audit.json",
        output_dir=output_dir,
        phase="final",
        random_controls=args.random_controls,
        allow_eval_final_raw=True,
    )
    result_path = output_dir / "final_results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def validate_development_authorizes_final(
    *,
    development: Mapping[str, Any],
    pool_dir: Path,
) -> list[str]:
    failures = []
    expected = {
        "claim_scope": DEVELOPMENT_SCOPE,
        "combined_audit_sha256": v1.sha256_file(pool_dir / "combined_audit.json"),
        "editor_method": EDITOR_METHOD,
        "eval_pool_sha256": v1.sha256_file(pool_dir / "development_subjects.json"),
        "final_redacted_audit_sha256": v1.sha256_file(pool_dir / "final_redacted_audit.json"),
        "implementation_sha256": v1.sha256_file(SCRIPT_PATH),
        "next_action": "eligible_for_one_shot_final_eval_without_method_changes",
        "phase": "development",
        "preregistration_sha256": v1.sha256_file(PREREG_PATH),
        "train_pool_sha256": v1.sha256_file(pool_dir / "train_subjects.json"),
    }
    if development.get("passed") is not True:
        failures.append("development did not pass")
    for key, expected_value in expected.items():
        if development.get(key) != expected_value:
            failures.append(
                f"development {key}={development.get(key)!r} does not match {expected_value!r}"
            )
    return failures


def assert_preregistered_pool_dir(pool_dir: Path) -> None:
    resolved = pool_dir.resolve()
    if resolved != DEFAULT_POOL_DIR.resolve():
        raise ValueError(f"V15 requires preregistered pool directory: {DEFAULT_POOL_DIR}")


def assert_no_forbidden_final_raw_paths(
    paths: Iterable[Path | str],
    *,
    allow_v15_final: bool,
) -> None:
    forbidden = {path.resolve() for path in PRIOR_FINAL_RAW_PATHS}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in forbidden:
            raise ValueError(f"prior sealed final raw path is forbidden: {path}")
        if path.name != "final_subjects.json":
            continue
        if allow_v15_final and path == V15_FINAL_RAW.resolve():
            continue
        raise ValueError(f"V15 final raw path is forbidden before final eval: {path}")


def forbidden_combined_final_summary_keys(summary: Mapping[str, Any]) -> list[str]:
    return sorted(set(summary) - COMBINED_FINAL_SUMMARY_ALLOWLIST)


def forbidden_final_redacted_keys(summary: Mapping[str, Any]) -> list[str]:
    return sorted(set(summary) - FINAL_REDACTED_ALLOWLIST)


def behavior_one_hot(behavior: str) -> torch.Tensor:
    values = torch.zeros(BEHAVIOR_DIM, dtype=torch.float32)
    values[PATTERNS.index(behavior)] = 1.0
    return values


def pair_one_hot(source: str, target: str) -> torch.Tensor:
    if source == target:
        raise ValueError("source and target must differ for pair one-hot")
    values = torch.zeros(PAIR_DIM, dtype=torch.float32)
    index = 0
    for source_behavior in PATTERNS:
        for target_behavior in PATTERNS:
            if target_behavior == source_behavior:
                continue
            if source_behavior == source and target_behavior == target:
                values[index] = 1.0
                return values
            index += 1
    raise ValueError(f"unknown source-target pair: {source}->{target}")


def build_editor_features(
    *,
    source_weights_norm: torch.Tensor,
    source_signature_norm: torch.Tensor,
    target_signature_norm: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
) -> torch.Tensor:
    features = torch.cat([
        source_weights_norm.to(dtype=torch.float32),
        source_signature_norm.to(dtype=torch.float32),
        target_signature_norm.to(dtype=torch.float32),
        behavior_one_hot(source_behavior),
        behavior_one_hot(target_behavior),
        pair_one_hot(source_behavior, target_behavior),
    ])
    if features.numel() != EDITOR_INPUT_DIM:
        raise ValueError(f"expected editor input dim {EDITOR_INPUT_DIM}, got {features.numel()}")
    return features


def hypernetwork_control_configs() -> list[Dict[str, Any]]:
    weight_mse_only = {
        "compatible": 0.0,
        "conflict": 0.0,
        "delta_norm": 0.0001,
        "signature": 0.0,
        "source_l2": 0.0005,
        "target_bce": 0.0,
        "weight_mse": 1.0,
    }
    functional_only = {**PRIMARY_LOSS_WEIGHTS, "signature": 0.0}
    signature_only = {
        **PRIMARY_LOSS_WEIGHTS,
        "compatible": 0.0,
        "conflict": 0.0,
        "target_bce": 0.0,
    }
    return [
        {
            "control_type": "target_label_only_hypernetwork",
            "loss_weights": dict(PRIMARY_LOSS_WEIGHTS),
            "seed": CONTROL_MODEL_SEEDS["target_label_only_hypernetwork"],
            "target_signature_mode": "target_centroid",
        },
        {
            "control_type": "source_signature_hypernetwork",
            "loss_weights": dict(PRIMARY_LOSS_WEIGHTS),
            "seed": CONTROL_MODEL_SEEDS["source_signature_hypernetwork"],
            "target_signature_mode": "source_signature",
        },
        {
            "control_type": "shuffled_signature_hypernetwork",
            "loss_weights": dict(PRIMARY_LOSS_WEIGHTS),
            "seed": CONTROL_MODEL_SEEDS["shuffled_signature_hypernetwork"],
            "target_signature_mode": "shuffled_behavior_centroid",
        },
        {
            "control_type": "nearest_train_target_signature_hypernetwork",
            "loss_weights": dict(PRIMARY_LOSS_WEIGHTS),
            "seed": CONTROL_MODEL_SEEDS["nearest_train_target_signature_hypernetwork"],
            "target_signature_mode": "paired_target_signature",
        },
        {
            "control_type": "random_signature_hypernetwork",
            "loss_weights": dict(PRIMARY_LOSS_WEIGHTS),
            "seed": CONTROL_MODEL_SEEDS["random_signature_hypernetwork"],
            "target_signature_mode": "random_signature",
        },
        {
            "alignment_mode": "random_neuron_permutation",
            "control_type": "random_neuron_permutation_hypernetwork",
            "loss_weights": dict(PRIMARY_LOSS_WEIGHTS),
            "seed": CONTROL_MODEL_SEEDS["random_neuron_permutation_hypernetwork"],
            "target_signature_mode": "paired_target_signature",
        },
        {
            "control_type": "target_weight_mse_only_hypernetwork",
            "loss_weights": weight_mse_only,
            "seed": CONTROL_MODEL_SEEDS["target_weight_mse_only_hypernetwork"],
            "target_signature_mode": "paired_target_signature",
        },
        {
            "control_type": "functional_only_hypernetwork",
            "loss_weights": functional_only,
            "seed": CONTROL_MODEL_SEEDS["functional_only_hypernetwork"],
            "target_signature_mode": "paired_target_signature",
        },
        {
            "control_type": "signature_only_hypernetwork",
            "loss_weights": signature_only,
            "seed": CONTROL_MODEL_SEEDS["signature_only_hypernetwork"],
            "target_signature_mode": "paired_target_signature",
        },
    ]


def sample_training_indices(
    *,
    pair_count: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(
        low=0,
        high=int(pair_count),
        size=(int(steps), int(batch_size)),
        generator=generator,
        dtype=torch.long,
    )


def differentiable_signature_batch(
    flat_weights: torch.Tensor,
    probe_examples: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    probe_inputs = torch.tensor(
        [example["sequence"] for example in probe_examples],
        dtype=torch.float32,
        device=flat_weights.device,
    )
    activations = hidden_activations_flat_batch(flat_weights, probe_inputs)
    features = []
    n_samples = int(probe_inputs.shape[0])
    for layer_activations in activations:
        for neuron_idx in range(layer_activations.shape[2]):
            neuron_acts = layer_activations[:, :, neuron_idx]
            mean = neuron_acts.mean(dim=1)
            std = neuron_acts.std(dim=1, unbiased=False)
            features.append(mean)
            features.append(std)
            fft_mag = torch.abs(torch.fft.fft(neuron_acts, dim=1))[
                :,
                : max(1, n_samples // 2),
            ]
            for value_idx in range(5):
                features.append(fft_mag[:, value_idx])
            for input_idx in range(5):
                features.append(safe_corrcoef_batch(neuron_acts, probe_inputs[:, input_idx]))
            features.append(mean)
            features.append(std)
    return torch.stack(features, dim=1)


def hidden_activations_flat_batch(
    flat_weights: torch.Tensor,
    inputs: torch.Tensor,
) -> list[torch.Tensor]:
    batch_size = int(flat_weights.shape[0])
    x = inputs.unsqueeze(0).expand(batch_size, -1, -1)
    offset = 0
    activations = []
    for out_dim, in_dim in [(8, 5), (8, 8), (8, 8), (8, 8), (8, 8)]:
        size = out_dim * in_dim
        weight = flat_weights[:, offset:offset + size].view(batch_size, out_dim, in_dim)
        offset += size
        bias = flat_weights[:, offset:offset + out_dim]
        offset += out_dim
        x = torch.nn.functional.gelu(torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1))
        activations.append(x)
    return activations


def safe_corrcoef_batch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    right = right.to(device=left.device, dtype=left.dtype).unsqueeze(0)
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean(dim=1, keepdim=True)
    denom = left_centered.norm(dim=1) * right_centered.norm(dim=1)
    numer = (left_centered * right_centered).sum(dim=1)
    return torch.where(
        denom > 0,
        numer / denom.clamp_min(1e-12),
        torch.zeros_like(numer),
    )


def fit_v15_train_statistics(
    subjects: Sequence[Mapping[str, Any]],
    *,
    build_v14_stats: bool = True,
) -> Dict[str, Any]:
    if build_v14_stats:
        stats = v14.fit_v14_train_statistics(subjects)
    else:
        signatures = torch.tensor([record["signature"] for record in subjects], dtype=torch.float32)
        stats = {
            "sig_mean": signatures.mean(dim=0),
            "sig_std": signatures.std(dim=0, unbiased=False).clamp_min(1e-6),
        }
    train_weights = torch.tensor([record["weights"] for record in subjects], dtype=torch.float32)
    signatures = torch.tensor([record["signature"] for record in subjects], dtype=torch.float32)
    sig_std = stats["sig_std"].clamp_min(1e-6)
    stats["weight_mean"] = train_weights.mean(dim=0)
    stats["weight_std"] = train_weights.std(dim=0, unbiased=False).clamp_min(1e-6)
    stats["train_subjects"] = list(subjects)
    stats["train_by_behavior"] = records_by_behavior(subjects)
    stats["train_by_id"] = {
        str(record["subject_id"]): record
        for record in subjects
    }
    stats["training_pair_table"] = build_training_pair_table(subjects)
    stats["training_pair_table_hash"] = stable_hash_json(stats["training_pair_table"])
    stats["signature_centroids"] = {}
    for pattern in PATTERNS:
        indices = [
            index for index, record in enumerate(subjects)
            if behavior_of_record(record) == pattern
        ]
        pattern_signatures = (signatures[indices] - stats["sig_mean"]) / sig_std
        stats["signature_centroids"][pattern] = pattern_signatures.mean(dim=0)
    return stats


def normalized_signature(
    record: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> torch.Tensor:
    signature = torch.tensor(record["signature"], dtype=torch.float32)
    return (signature - stats["sig_mean"]) / stats["sig_std"].clamp_min(1e-6)


def normalized_weights(
    weights: torch.Tensor,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    return (weights - stats["weight_mean"]) / stats["weight_std"].clamp_min(1e-6)


def aligned_training_target_weights(
    *,
    source_weights: torch.Tensor,
    source_behavior: str,
    target_behavior: str,
    source_subject_id: str,
    target_record: Mapping[str, Any],
    alignment_mode: str,
) -> torch.Tensor:
    target_weights = torch.tensor(target_record["weights"], dtype=torch.float32)
    if alignment_mode == "hungarian":
        return v14.align_target_to_source(
            source_weights=source_weights,
            target_weights=target_weights,
        )
    if alignment_mode == "random_neuron_permutation":
        return v14.random_permute_target_weights(
            target_weights=target_weights,
            subject_id=source_subject_id,
            source=source_behavior,
            target=target_behavior,
            target_subject_id=str(target_record["subject_id"]),
        )
    if alignment_mode == "none":
        return target_weights
    raise ValueError(f"unknown alignment mode: {alignment_mode}")


def training_target_signature(
    *,
    source_record: Mapping[str, Any],
    target_record: Mapping[str, Any],
    source_behavior: str,
    target_behavior: str,
    stats: Mapping[str, Any],
    target_signature_mode: str,
) -> torch.Tensor:
    if target_signature_mode in {
        "paired_target_signature",
        "nearest_train_target_signature",
    }:
        return normalized_signature(target_record, stats)
    if target_signature_mode == "target_centroid":
        return stats["signature_centroids"][target_behavior]
    if target_signature_mode == "source_signature":
        return normalized_signature(source_record, stats)
    if target_signature_mode == "shuffled_behavior_centroid":
        shuffled = v14.select_shuffled_target(
            subject_id=str(source_record["subject_id"]),
            source=source_behavior,
            target=target_behavior,
        )
        return stats["signature_centroids"][shuffled]
    if target_signature_mode == "random_signature":
        return deterministic_random_signature(
            key_parts=[
                str(source_record["subject_id"]),
                str(target_record["subject_id"]),
                "v15_train_random_signature",
            ],
            dim=SIGNATURE_DIM,
        )
    raise ValueError(f"unknown target signature mode: {target_signature_mode}")


def build_training_batch_tensors(
    *,
    stats: Mapping[str, Any],
    pair_indices: torch.Tensor,
    target_signature_mode: str,
    alignment_mode: str = "hungarian",
) -> Dict[str, Any]:
    features = []
    source_weights_batch = []
    target_weights_batch = []
    source_signature_batch = []
    target_signature_batch = []
    source_behaviors = []
    target_behaviors = []
    pair_table = stats["training_pair_table"]
    records_by_id = stats["train_by_id"]
    for raw_index in pair_indices.tolist():
        pair = pair_table[int(raw_index)]
        source_record = records_by_id[pair["source_subject_id"]]
        target_record = records_by_id[pair["target_subject_id"]]
        source_behavior = pair["source_behavior"]
        target_behavior = pair["target_behavior"]
        source_weights = torch.tensor(source_record["weights"], dtype=torch.float32)
        target_weights = aligned_training_target_weights(
            source_weights=source_weights,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            source_subject_id=str(source_record["subject_id"]),
            target_record=target_record,
            alignment_mode=alignment_mode,
        )
        source_signature_norm = normalized_signature(source_record, stats)
        target_signature_norm = training_target_signature(
            source_record=source_record,
            target_record=target_record,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
            stats=stats,
            target_signature_mode=target_signature_mode,
        )
        features.append(build_editor_features(
            source_weights_norm=normalized_weights(source_weights, stats),
            source_signature_norm=source_signature_norm,
            target_signature_norm=target_signature_norm,
            source_behavior=source_behavior,
            target_behavior=target_behavior,
        ))
        source_weights_batch.append(source_weights)
        target_weights_batch.append(target_weights)
        source_signature_batch.append(source_signature_norm)
        target_signature_batch.append(target_signature_norm)
        source_behaviors.append(source_behavior)
        target_behaviors.append(target_behavior)
    return {
        "features": torch.stack(features),
        "source_behaviors": source_behaviors,
        "source_signature_norm": torch.stack(source_signature_batch),
        "source_weights": torch.stack(source_weights_batch),
        "target_behaviors": target_behaviors,
        "target_signature_norm": torch.stack(target_signature_batch),
        "target_weights": torch.stack(target_weights_batch),
    }


def precompute_training_tensors(
    *,
    stats: Mapping[str, Any],
    target_signature_mode: str,
    alignment_mode: str,
) -> Dict[str, Any]:
    all_indices = torch.arange(len(stats["training_pair_table"]), dtype=torch.long)
    return build_training_batch_tensors(
        stats=stats,
        pair_indices=all_indices,
        target_signature_mode=target_signature_mode,
        alignment_mode=alignment_mode,
    )


def batch_from_precomputed_training_tensors(
    precomputed: Mapping[str, Any],
    pair_indices: torch.Tensor,
) -> Dict[str, Any]:
    indices = pair_indices.to(dtype=torch.long)
    index_list = [int(index) for index in indices.tolist()]
    return {
        "features": precomputed["features"][indices],
        "source_behaviors": [
            precomputed["source_behaviors"][index]
            for index in index_list
        ],
        "source_signature_norm": precomputed["source_signature_norm"][indices],
        "source_weights": precomputed["source_weights"][indices],
        "target_behaviors": [
            precomputed["target_behaviors"][index]
            for index in index_list
        ],
        "target_signature_norm": precomputed["target_signature_norm"][indices],
        "target_weights": precomputed["target_weights"][indices],
    }


def train_hypernetwork_editor(
    *,
    stats: Mapping[str, Any],
    seed: int,
    target_signature_mode: str,
    alignment_mode: str,
    loss_weights: Mapping[str, float],
    steps: int = TRAINING_STEPS,
    batch_size: int = TRAINING_BATCH_SIZE,
) -> Dict[str, Any]:
    model = SignatureConditionedDeltaHypernetwork(seed=seed).to(dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_LR,
        betas=TRAINING_BETAS,
        weight_decay=TRAINING_WEIGHT_DECAY,
    )
    pair_count = len(stats["training_pair_table"])
    precomputed = precompute_training_tensors(
        stats=stats,
        target_signature_mode=target_signature_mode,
        alignment_mode=alignment_mode,
    )
    sampled_indices = sample_training_indices(
        pair_count=pair_count,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )
    history = []
    probe_examples = stats.get("probe_examples")
    if probe_examples is None:
        probe_examples = build_digit_probe_examples(
            n_examples=256,
            seed=20260610,
            seq_len=5,
            base=10,
        )
    for step_index in range(int(steps)):
        batch = batch_from_precomputed_training_tensors(
            precomputed,
            sampled_indices[step_index],
        )
        optimizer.zero_grad(set_to_none=True)
        delta, diagnostic_scale = model(batch["features"])
        edited_weights = batch["source_weights"] + delta
        loss, loss_parts = hypernetwork_training_loss(
            edited_weights=edited_weights,
            delta=delta,
            diagnostic_scale=diagnostic_scale,
            batch=batch,
            loss_weights=loss_weights,
            probe_examples=probe_examples,
            sig_mean=stats["sig_mean"],
            sig_std=stats["sig_std"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        history.append({
            "diagnostic_scale_mean": float(diagnostic_scale.detach().mean().item()),
            "loss": float(loss.detach().item()),
            "step": int(step_index + 1),
            **loss_parts,
        })
    return {
        "final_step": int(steps),
        "history": history,
        "model": model.cpu(),
        "sampled_indices_hash": stable_hash_json(sampled_indices.tolist()),
        "stats": {
            "weight_mean": stats["weight_mean"],
            "weight_std": stats["weight_std"],
        },
    }


def hypernetwork_training_loss(
    *,
    edited_weights: torch.Tensor,
    delta: torch.Tensor,
    diagnostic_scale: torch.Tensor,
    batch: Mapping[str, Any],
    loss_weights: Mapping[str, float],
    probe_examples: Sequence[Mapping[str, Any]],
    sig_mean: torch.Tensor,
    sig_std: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    del diagnostic_scale
    loss = torch.zeros((), dtype=edited_weights.dtype)
    parts: Dict[str, float] = {}
    if loss_weights.get("weight_mse", 0.0):
        value = F.mse_loss(edited_weights, batch["target_weights"])
        loss = loss + float(loss_weights["weight_mse"]) * value
        parts["weight_mse"] = float(value.detach().item())
    if loss_weights.get("source_l2", 0.0):
        value = F.mse_loss(edited_weights, batch["source_weights"])
        loss = loss + float(loss_weights["source_l2"]) * value
        parts["source_l2"] = float(value.detach().item())
    if loss_weights.get("delta_norm", 0.0):
        value = delta.pow(2).mean()
        loss = loss + float(loss_weights["delta_norm"]) * value
        parts["delta_norm"] = float(value.detach().item())
    if loss_weights.get("signature", 0.0):
        signature = differentiable_signature_batch(edited_weights, probe_examples)
        signature_norm = (signature - sig_mean.to(signature.device)) / sig_std.to(
            signature.device
        ).clamp_min(1e-6)
        value = F.mse_loss(signature_norm, batch["target_signature_norm"])
        loss = loss + float(loss_weights["signature"]) * value
        parts["signature_mse"] = float(value.detach().item())
    if (
        loss_weights.get("target_bce", 0.0)
        or loss_weights.get("conflict", 0.0)
        or loss_weights.get("compatible", 0.0)
    ):
        functional_loss, functional_parts = hypernetwork_functional_training_loss(
            edited_weights=edited_weights,
            batch=batch,
            loss_weights=loss_weights,
        )
        loss = loss + functional_loss
        parts.update(functional_parts)
    return loss, parts


def hypernetwork_functional_training_loss(
    *,
    edited_weights: torch.Tensor,
    batch: Mapping[str, Any],
    loss_weights: Mapping[str, float],
) -> tuple[torch.Tensor, Dict[str, float]]:
    loss = torch.zeros((), dtype=edited_weights.dtype)
    parts: Dict[str, float] = {}
    accum: Dict[str, list[torch.Tensor]] = {
        "compatible": [],
        "conflict": [],
        "target_bce": [],
    }
    for source in PATTERNS:
        for target in PATTERNS:
            if target == source:
                continue
            indices = [
                index for index, (source_behavior, target_behavior)
                in enumerate(zip(batch["source_behaviors"], batch["target_behaviors"]))
                if source_behavior == source and target_behavior == target
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long)
            weights_subset = edited_weights[index_tensor]
            source_weights_subset = batch["source_weights"][index_tensor]
            support = v14.support_loss_tensors(source=source, target=target)
            split = v14.source_target_support_split(source=source, target=target)
            if loss_weights.get("target_bce", 0.0):
                logits = v10.decoder_v1.subject_forward_flat_batch(
                    weights_subset,
                    support["target_inputs"],
                )
                target_labels = support["target_labels"].unsqueeze(0).expand_as(logits)
                accum["target_bce"].append(
                    F.binary_cross_entropy_with_logits(logits, target_labels)
                )
            if loss_weights.get("conflict", 0.0):
                logits = v10.decoder_v1.subject_forward_flat_batch(
                    weights_subset,
                    split["conflict_inputs"],
                )
                labels = split["conflict_target_labels"].unsqueeze(0).expand_as(logits)
                accum["conflict"].append(
                    F.binary_cross_entropy_with_logits(logits, labels)
                )
            if loss_weights.get("compatible", 0.0):
                edited = v10.decoder_v1.subject_forward_flat_batch(
                    weights_subset,
                    split["compatible_inputs"],
                )
                with torch.no_grad():
                    source_logits = v10.decoder_v1.subject_forward_flat_batch(
                        source_weights_subset,
                        split["compatible_inputs"],
                    )
                accum["compatible"].append(F.mse_loss(edited, source_logits))
    if accum["target_bce"]:
        value = torch.stack(accum["target_bce"]).mean()
        loss = loss + float(loss_weights["target_bce"]) * value
        parts["target_bce"] = float(value.detach().item())
    if accum["conflict"]:
        value = torch.stack(accum["conflict"]).mean()
        loss = loss + float(loss_weights["conflict"]) * value
        parts["conflict_bce"] = float(value.detach().item())
    if accum["compatible"]:
        value = torch.stack(accum["compatible"]).mean()
        loss = loss + float(loss_weights["compatible"]) * value
        parts["compatible_mse"] = float(value.detach().item())
    return loss, parts


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> Dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_v15_seed_preflight()
    if seed_preflight["failures"]:
        result = {"failures": seed_preflight["failures"], "passed": False}
        (pool_dir / "combined_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        return result

    suite = build_suite(args.support_per_class, args.heldout_per_class)
    heldout_sequences = build_heldout_sequences(suite)
    candidate_pools = build_candidate_pools(heldout_sequences)
    candidate_pool_summary = summarize_candidate_pools(candidate_pools)
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    source_args = SimpleNamespace(
        generic_negative_cap=args.generic_negative_cap,
        hard_negative_cap=args.hard_negative_cap,
        heldout_per_class=args.heldout_per_class,
        lr=args.lr,
        positive_cap=args.positive_cap,
        source_margin_gate=args.source_margin_gate,
        support_per_class=args.support_per_class,
        train_epochs=args.train_epochs,
    )
    pool_payloads = {}
    pool_summaries = {}
    for pool_name, pool_config in POOL_CONFIGS.items():
        payload = poolgen.generate_pool(
            args=source_args,
            pool_name=pool_name,
            pool_config=pool_config,
            suite=suite,
            heldout_sequences=heldout_sequences,
            candidate_pools=candidate_pools,
            candidate_pool_summary=candidate_pool_summary,
            probe_examples=probe_examples,
        )
        payload["claim_scope"] = SOURCE_POOL_SCOPE
        payload["config"]["base_seed"] = int(pool_config["base_seed"])
        payload["config"]["seed_behavior_stride"] = int(SEED_BEHAVIOR_STRIDE)
        payload["pool_redacted_payload_sha256"] = stable_hash_json(
            poolgen.redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_path = pool_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = poolgen.summarize_pool(payload)
        summary["pool_file_sha256"] = v1.sha256_file(pool_path)
        summary["pool_redacted_payload_sha256"] = payload["pool_redacted_payload_sha256"]
        pool_summaries[pool_name] = summary

    final_redacted = poolgen.build_final_redacted_summary(pool_payloads["final"])
    final_redacted["claim_scope"] = FINAL_REDACTED_SCOPE
    final_redacted["pool_file_sha256"] = pool_summaries["final"]["pool_file_sha256"]
    final_redacted["summary_payload_sha256"] = stable_hash_json(final_redacted)
    forbidden_redacted = forbidden_final_redacted_keys(final_redacted)
    if forbidden_redacted:
        raise ValueError(
            "final_redacted_audit exposes forbidden keys: "
            + ", ".join(forbidden_redacted)
        )
    (pool_dir / "final_redacted_audit.json").write_text(
        json.dumps(final_redacted, indent=2, sort_keys=True)
    )

    audit = poolgen.build_combined_audit(
        pool_payloads=pool_payloads,
        pool_summaries=pool_summaries,
        seed_preflight=seed_preflight,
        suite=suite,
        probe_examples=probe_examples,
    )
    audit["claim_scope"] = SOURCE_AUDIT_SCOPE
    audit = v10.redact_combined_audit(audit)
    final_summary = audit.get("pool_summaries", {}).get("final", {})
    audit["pool_summaries"]["final"] = {
        key: final_summary[key]
        for key in sorted(COMBINED_FINAL_SUMMARY_ALLOWLIST)
        if key in final_summary
    }
    forbidden_final_summary = forbidden_combined_final_summary_keys(
        audit["pool_summaries"]["final"]
    )
    if forbidden_final_summary:
        raise ValueError(
            "combined_audit.pool_summaries.final exposes forbidden keys: "
            + ", ".join(forbidden_final_summary)
        )
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": v1.rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": v1.rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def build_v15_seed_preflight() -> Dict[str, Any]:
    return v9.build_seed_preflight(
        POOL_CONFIGS,
        behavior_stride=SEED_BEHAVIOR_STRIDE,
    )


def train_and_evaluate(
    *,
    train_path: Path,
    eval_path: Path,
    combined_audit_path: Path,
    final_redacted_path: Path,
    output_dir: Path,
    phase: str,
    random_controls: int,
    allow_eval_final_raw: bool,
) -> Dict[str, Any]:
    assert_no_forbidden_final_raw_paths(
        [train_path, eval_path, combined_audit_path, final_redacted_path],
        allow_v15_final=allow_eval_final_raw,
    )
    train_payload = v1.load_json(train_path)
    eval_payload = v1.load_json(eval_path)
    combined_audit = v1.load_json(combined_audit_path)
    final_redacted = v1.load_json(final_redacted_path)
    contract_failures = validate_source_pool_contract(
        train_path=train_path,
        eval_path=eval_path,
        combined_audit_path=combined_audit_path,
        final_redacted_path=final_redacted_path,
        train_payload=train_payload,
        eval_payload=eval_payload,
        combined_audit=combined_audit,
        final_redacted=final_redacted,
        phase=phase,
    )
    if contract_failures:
        raise ValueError("V15 source-pool contract validation failed: " + "; ".join(contract_failures))

    train_subjects = v1.accepted_records(train_payload)
    eval_subjects = v1.accepted_records(eval_payload)
    train_stats = fit_v15_train_statistics(train_subjects)
    train_stats["probe_examples"] = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    train_stats["probe_examples_hash"] = stable_hash_json(train_stats["probe_examples"])
    classifier, classifier_summary = v9.fit_primary_classifier(
        train_stats["z_train"],
        train_stats["y_train"],
    )
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    calibration_coefficients, calibration_summary = v9.fit_contrastive_calibration(
        subjects=train_subjects,
        train_stats=train_stats,
        classifier=classifier,
    )
    train_stats["calibration_coefficients"] = calibration_coefficients

    trained_editors = train_all_hypernetwork_editors(train_stats)
    stats_path = output_dir / "v15_signature_conditioned_hypernetwork_stats.pt"
    torch.save(
        {
            "control_summaries": {
                name: summary_without_model(payload)
                for name, payload in trained_editors.items()
            },
            "method": EDITOR_METHOD,
            "probe_examples_hash": train_stats["probe_examples_hash"],
            "train_statistics_hash": train_only_statistics_hash(train_stats),
        },
        stats_path,
    )
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        classifier=classifier,
        trained_editors=trained_editors,
        random_controls=random_controls,
    )
    failures = [*contract_failures, *eval_result["failures"]]
    result = {
        **eval_result,
        "calibration_summary": calibration_summary,
        "claim_scope": DEVELOPMENT_SCOPE if phase == "development" else FINAL_SCOPE,
        "classifier_summary": classifier_summary,
        "combined_audit_path": v1.rel(combined_audit_path),
        "combined_audit_sha256": v1.sha256_file(combined_audit_path),
        "dirty_worktree_caveat": True,
        "editor_method": EDITOR_METHOD,
        "eval_pool_path": v1.rel(eval_path),
        "eval_pool_sha256": v1.sha256_file(eval_path),
        "final_redacted_audit_path": v1.rel(final_redacted_path),
        "final_redacted_audit_sha256": v1.sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "forbidden_prior_final_raw_opened": False,
        "forbidden_v15_final_raw_opened_before_authorization": False,
        "implementation_sha256": v1.sha256_file(SCRIPT_PATH),
        "limitations": (
            "Small-subject source-label-known, target-label-requested "
            "signature-conditioned hypernetwork evidence only; not pure "
            "signature-only editing, source-label inference, source-free decoding, "
            "larger-model evidence, broad MUAT proof, or arbitrary capability preservation."
        ),
        "phase": phase,
        "pool_overlap_counts": combined_audit.get("overlap_counts", {}),
        "preregistration_sha256": v1.sha256_file(PREREG_PATH),
        "probe_examples_hash": train_stats["probe_examples_hash"],
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "stats_path": v1.rel(stats_path),
        "stats_sha256": v1.sha256_file(stats_path),
        "thresholds": THRESHOLDS,
        "train_only_statistics_hash": train_only_statistics_hash(train_stats),
        "train_pool_path": v1.rel(train_path),
        "train_pool_sha256": v1.sha256_file(train_path),
    }
    result["failures"] = failures
    result["passed"] = not failures
    return result


def train_all_hypernetwork_editors(
    train_stats: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    editors = {
        "matched": train_hypernetwork_editor(
            stats=train_stats,
            seed=EDITOR_SEED,
            target_signature_mode="paired_target_signature",
            alignment_mode="hungarian",
            loss_weights=PRIMARY_LOSS_WEIGHTS,
        )
    }
    for config in hypernetwork_control_configs():
        editors[config["control_type"]] = train_hypernetwork_editor(
            stats=train_stats,
            seed=int(config["seed"]),
            target_signature_mode=str(config["target_signature_mode"]),
            alignment_mode=str(config.get("alignment_mode", "hungarian")),
            loss_weights=config["loss_weights"],
        )
    return editors


def summary_without_model(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model", "stats"}
    }


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    trained_editors: Mapping[str, Mapping[str, Any]],
    random_controls: int,
) -> Dict[str, Any]:
    records = []
    for subject in subjects:
        source = v1.behavior_of(subject)
        source_signature = torch.tensor(subject["signature"], dtype=torch.float32)
        z = (source_signature - train_stats["sig_mean"]) / train_stats["sig_std"].clamp_min(1e-6)
        source_weights = torch.tensor(subject["weights"], dtype=torch.float32)
        for target in PATTERNS:
            if target == source:
                continue
            records.append(evaluate_record(
                subject=subject,
                z=z,
                source=source,
                target=target,
                source_weights=source_weights,
                train_stats=train_stats,
                classifier=classifier,
                trained_editors=trained_editors,
                random_controls=random_controls,
            ))
    aggregate = summarize_records(records)
    by_direction = {
        v1.vector_key(source, target): summarize_records([
            record for record in records
            if record["source_behavior"] == source and record["target_behavior"] == target
        ])
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    failures = gate_failures(aggregate=aggregate, by_direction=by_direction, records=records)
    return {
        "aggregate": aggregate,
        "by_direction": by_direction,
        "failures": failures,
        "passed": not failures,
        "records": records,
    }


def evaluate_record(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    trained_editors: Mapping[str, Mapping[str, Any]],
    random_controls: int,
) -> Dict[str, Any]:
    selected = v10.selected_v9_conditioning(
        z=z,
        source=source,
        target=target,
        train_stats=train_stats,
        classifier=classifier,
    )
    matched_weights, matched_meta = hypernetwork_scaled_weights(
        editor_payload=trained_editors["matched"],
        source_weights=source_weights,
        source_signature_norm=z,
        target_signature_norm=selected["candidate_z"],
        source=source,
        target=target,
    )
    matched_delta_norm = (matched_weights - source_weights).norm()
    matched = {
        **v14.functional_metrics(matched_weights, source, target, source_weights),
        "delta_norm": float(matched_delta_norm.item()),
        "editor": matched_meta,
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_centroid_improvement": selected["selected_centroid_improvement"],
        "selected_displacement_norm": selected["selected_displacement_norm"],
        "selected_primary_target_margin": selected["selected_primary_target_margin"],
    }
    controls = build_controls(
        subject=subject,
        z=z,
        selected_signature_norm=selected["candidate_z"],
        source=source,
        target=target,
        source_weights=source_weights,
        matched_delta_norm=matched_delta_norm,
        train_stats=train_stats,
        classifier=classifier,
        trained_editors=trained_editors,
        random_controls=random_controls,
    )
    for control in controls:
        control["matched_minus_control_target_margin"] = (
            matched["target_margin"] - control["target_margin"]
        )
        control["control_minus_matched_compatible_source_output_mse"] = (
            control["compatible_source_output_mse"]
            - matched["compatible_source_output_mse"]
        )
    full_retrieval = single_control(controls, "aligned_full_nearest_target_retrieval")
    best_target = max(controls, key=lambda item: item["target_margin"])
    pareto_dominators = [
        control for control in controls if v14.pareto_dominates_functional(control, matched)
    ]
    matched["pareto_dominator_count"] = len(pareto_dominators)
    matched["pareto_dominator_types"] = sorted({
        control["control_type"] for control in pareto_dominators
    })
    matched["pareto_undominated"] = not pareto_dominators
    matched["target_prediction_pass"] = matched["predicted_behavior"] == target
    for metric_name, control_name in SIGNATURE_ADVANTAGE_CONTROLS.items():
        control = single_control(controls, control_name)
        matched[f"matched_minus_{metric_name}_target_margin"] = (
            matched["target_margin"] - control["target_margin"]
        )
        matched[f"{metric_name}_minus_matched_compatible_source_output_mse"] = (
            control["compatible_source_output_mse"]
            - matched["compatible_source_output_mse"]
        )
    matched["individual_all_gates_passed"] = individual_passed(
        matched=matched,
        full_retrieval=full_retrieval,
    )
    return {
        "controls": v10.strip_control_weights(controls),
        "individual_all_gates_passed": matched["individual_all_gates_passed"],
        "matched": v10.strip_weight(matched),
        "random_control_count": sum(
            1
            for control in controls
            if control["control_type"].startswith("random_norm_matched_weight_delta")
        ),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": {
            "best_control_target_margin": best_target["target_margin"],
            "best_control_type": best_target["control_type"],
            "full_retrieval_minus_matched_compatible_source_output_mse": (
                full_retrieval["compatible_source_output_mse"]
                - matched["compatible_source_output_mse"]
            ),
            "matched_minus_best_control_target_margin": (
                matched["target_margin"] - best_target["target_margin"]
            ),
            "matched_minus_full_retrieval_target_margin": (
                matched["target_margin"] - full_retrieval["target_margin"]
            ),
            "pareto_undominated": matched["pareto_undominated"],
            "target_prediction_pass": matched["target_prediction_pass"],
            **signature_summary_fields(matched),
        },
        "target_behavior": target,
    }


SIGNATURE_ADVANTAGE_CONTROLS = {
    "shuffled_signature": "shuffled_signature_hypernetwork",
    "source_signature": "source_signature_hypernetwork",
    "target_label": "target_label_only_hypernetwork",
    "v13_no_signature": "v13_no_signature_support_optimizer",
}


def signature_summary_fields(matched: Mapping[str, Any]) -> Dict[str, float]:
    fields = {}
    for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
        fields[f"matched_minus_{metric_name}_target_margin"] = matched[
            f"matched_minus_{metric_name}_target_margin"
        ]
        fields[f"{metric_name}_minus_matched_compatible_source_output_mse"] = matched[
            f"{metric_name}_minus_matched_compatible_source_output_mse"
        ]
    return fields


def single_control(controls: Sequence[Mapping[str, Any]], control_type: str) -> Mapping[str, Any]:
    matches = [control for control in controls if control["control_type"] == control_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {control_type} control, found {len(matches)}")
    return matches[0]


def build_controls(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    selected_signature_norm: torch.Tensor,
    source: str,
    target: str,
    source_weights: torch.Tensor,
    matched_delta_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    trained_editors: Mapping[str, Mapping[str, Any]],
    random_controls: int,
) -> list[Dict[str, Any]]:
    retrieval = v10.nearest_train_target_retrieval(
        selected_signature_norm=selected_signature_norm,
        target=target,
        train_stats=train_stats,
    )
    retrieved_weights = v14.align_target_to_source(
        source_weights=source_weights,
        target_weights=retrieval["weights"],
    )
    controls = [
        v14.control_record("no_edit", source_weights, source, target, source_weights),
        v14.optimizer_control_record(
            control_type="v13_no_signature_support_optimizer",
            source_weights=source_weights,
            source=source,
            target=target,
            train_stats=train_stats,
            signature_target_norm=selected_signature_norm,
            loss_weights={**v14.matched_loss_weights(), "signature": 0.0},
        ),
    ]
    v14_direction, _v14_unprojected, v14_meta = v14.signature_weighted_task_direction(
        subject_id=str(subject["subject_id"]),
        source_weights=source_weights,
        source=source,
        target=target,
        train_stats=train_stats,
        signature_target_norm=selected_signature_norm,
    )
    controls.append(v14.task_vector_control_record(
        control_type="v14_signature_gated_task_vector",
        direction=v14_direction,
        source_weights=source_weights,
        source=source,
        target=target,
        metadata=v14_meta,
    ))
    controls.append(v14.control_record(
        "aligned_full_nearest_target_retrieval",
        retrieved_weights,
        source,
        target,
        source_weights,
        {"retrieved_subject_id": retrieval["subject_id"]},
    ))
    controls.append(v14.control_record(
        "aligned_interpolation_alpha_0.975",
        v14.interpolate_weights(
            source_weights=source_weights,
            target_weights=retrieved_weights,
            alpha=MATCHED_ALPHA,
        ),
        source,
        target,
        source_weights,
        {"alpha": MATCHED_ALPHA, "retrieved_subject_id": retrieval["subject_id"]},
    ))
    for config in hypernetwork_control_configs():
        control_type = config["control_type"]
        target_signature = evaluation_target_signature(
            subject=subject,
            z=z,
            selected_signature_norm=selected_signature_norm,
            source=source,
            target=target,
            train_stats=train_stats,
            classifier=classifier,
            mode=str(config["target_signature_mode"]),
        )
        weights, metadata = hypernetwork_scaled_weights(
            editor_payload=trained_editors[control_type],
            source_weights=source_weights,
            source_signature_norm=z,
            target_signature_norm=target_signature,
            source=source,
            target=target,
        )
        controls.append(v14.control_record(
            control_type,
            weights,
            source,
            target,
            source_weights,
            metadata,
        ))
    controls.extend(v14.random_weight_delta_controls(
        subject_id=str(subject["subject_id"]),
        source=source,
        target=target,
        source_weights=source_weights,
        matched_delta_norm=matched_delta_norm,
        random_controls=random_controls,
    ))
    return controls


def hypernetwork_scaled_weights(
    *,
    editor_payload: Mapping[str, Any],
    source_weights: torch.Tensor,
    source_signature_norm: torch.Tensor,
    target_signature_norm: torch.Tensor,
    source: str,
    target: str,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    model = editor_payload["model"]
    model.eval()
    with torch.no_grad():
        features = build_editor_features(
            source_weights_norm=normalized_weights(source_weights, editor_payload["stats"]),
            source_signature_norm=source_signature_norm,
            target_signature_norm=target_signature_norm,
            source_behavior=source,
            target_behavior=target,
        ).unsqueeze(0)
        direction, diagnostic_scale = model(features)
    weights, scale_meta = v14.select_scaled_task_vector_weights(
        source_weights=source_weights,
        source=source,
        target=target,
        direction=direction[0],
    )
    return weights, {
        "diagnostic_scale": float(diagnostic_scale[0].item()),
        "direction_norm": float(direction[0].norm().item()),
        "scale_selection": scale_meta,
        "selected_scale": scale_meta["selected_scale"],
    }


def evaluation_target_signature(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    selected_signature_norm: torch.Tensor,
    source: str,
    target: str,
    train_stats: Mapping[str, Any],
    classifier: torch.nn.Module,
    mode: str,
) -> torch.Tensor:
    if mode in {"paired_target_signature", "nearest_train_target_signature"}:
        if mode == "nearest_train_target_signature":
            return nearest_train_target_signature(
                target=target,
                signature_target_norm=selected_signature_norm,
                train_stats=train_stats,
            )
        return selected_signature_norm
    if mode == "target_centroid":
        return train_stats["signature_centroids"][target]
    if mode == "source_signature":
        return z
    if mode == "shuffled_behavior_centroid":
        shuffled = v14.select_shuffled_target(
            subject_id=str(subject["subject_id"]),
            source=source,
            target=target,
        )
        selected = v10.selected_v9_conditioning(
            z=z,
            source=source,
            target=shuffled,
            train_stats=train_stats,
            classifier=classifier,
        )
        return selected["candidate_z"]
    if mode == "random_signature":
        return deterministic_random_signature(
            key_parts=[
                str(subject["subject_id"]),
                source,
                target,
                "v15_random_signature",
            ],
            dim=SIGNATURE_DIM,
        )
    raise ValueError(f"unknown evaluation target signature mode: {mode}")


def nearest_train_target_signature(
    *,
    target: str,
    signature_target_norm: torch.Tensor,
    train_stats: Mapping[str, Any],
) -> torch.Tensor:
    scored = []
    for record in train_stats["train_by_behavior"][target]:
        z_record = normalized_signature(record, train_stats)
        distance = float(F.mse_loss(z_record, signature_target_norm).item())
        scored.append((distance, str(record["subject_id"]), z_record))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def individual_passed(
    *,
    matched: Mapping[str, Any],
    full_retrieval: Mapping[str, Any],
) -> bool:
    return (
        matched["target_prediction_pass"]
        and matched["target_margin"] > THRESHOLDS["min_per_record_matched_target_margin"]
        and matched["compatible_source_output_mse"]
        < full_retrieval["compatible_source_output_mse"]
        and matched["conflict_target_accuracy"]
        >= THRESHOLDS["min_per_record_conflict_target_accuracy"]
        and matched["conflict_target_accuracy_improvement"]
        >= THRESHOLDS["min_per_record_conflict_target_accuracy_improvement"]
        and matched["pareto_undominated"]
        and (
            matched["matched_minus_target_label_target_margin"]
            >= THRESHOLDS["min_per_record_target_label_target_margin_advantage"]
            or matched["target_label_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_target_label_compatible_mse_advantage"]
        )
        and (
            matched["matched_minus_source_signature_target_margin"]
            >= THRESHOLDS["min_per_record_source_signature_target_margin_advantage"]
            or matched["source_signature_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_source_signature_compatible_mse_advantage"]
        )
        and (
            matched["matched_minus_shuffled_signature_target_margin"]
            >= THRESHOLDS["min_per_record_shuffled_target_margin_advantage"]
            or matched["shuffled_signature_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_shuffled_compatible_mse_advantage"]
        )
        and (
            matched["matched_minus_v13_no_signature_target_margin"]
            >= THRESHOLDS["min_per_record_no_signature_target_margin_advantage"]
            or matched["v13_no_signature_minus_matched_compatible_source_output_mse"]
            >= THRESHOLDS["min_per_record_no_signature_compatible_mse_advantage"]
        )
    )


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "individual_all_gate_pass_count": 0,
            "individual_all_gate_pass_rate": 0.0,
            "n": 0,
            "pareto_undominated_count": 0,
            "pareto_undominated_rate": 0.0,
            "target_prediction_count": 0,
            "target_prediction_rate": 0.0,
        }
    passed = sum(1 for record in records if record["individual_all_gates_passed"])
    target_pred = sum(1 for record in records if record["summary"]["target_prediction_pass"])
    pareto = sum(1 for record in records if record["summary"]["pareto_undominated"])
    summary = {
        "individual_all_gate_pass_count": int(passed),
        "individual_all_gate_pass_rate": float(passed / n),
        "mean_conflict_target_accuracy": v10.mean(
            record["matched"]["conflict_target_accuracy"] for record in records
        ),
        "mean_conflict_target_accuracy_improvement": v10.mean(
            record["matched"]["conflict_target_accuracy_improvement"] for record in records
        ),
        "mean_full_retrieval_minus_matched_compatible_source_output_mse": v10.mean(
            record["summary"]["full_retrieval_minus_matched_compatible_source_output_mse"]
            for record in records
        ),
        "mean_matched_minus_best_control_target_margin": v10.mean(
            record["summary"]["matched_minus_best_control_target_margin"] for record in records
        ),
        "mean_matched_minus_full_retrieval_target_margin": v10.mean(
            record["summary"]["matched_minus_full_retrieval_target_margin"] for record in records
        ),
        "mean_matched_target_margin": v10.mean(
            record["matched"]["target_margin"] for record in records
        ),
        "n": int(n),
        "pareto_undominated_count": int(pareto),
        "pareto_undominated_rate": float(pareto / n),
        "target_prediction_count": int(target_pred),
        "target_prediction_rate": float(target_pred / n),
    }
    for metric_name in SIGNATURE_ADVANTAGE_CONTROLS:
        summary[f"mean_matched_minus_{metric_name}_target_margin"] = v10.mean(
            record["summary"][f"matched_minus_{metric_name}_target_margin"]
            for record in records
        )
        summary[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"] = v10.mean(
            record["summary"][f"{metric_name}_minus_matched_compatible_source_output_mse"]
            for record in records
        )
    return summary


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_direction: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures = []
    v10.require_equal(failures, aggregate["n"], THRESHOLDS["expected_record_count"], "aggregate n")
    v10.require_at_least(
        failures,
        aggregate["individual_all_gate_pass_rate"],
        THRESHOLDS["min_aggregate_individual_pass_rate"],
        "aggregate individual pass rate",
    )
    v10.require_at_least(
        failures,
        aggregate["target_prediction_rate"],
        THRESHOLDS["min_aggregate_target_prediction_rate"],
        "aggregate target prediction rate",
    )
    v10.require_at_least(
        failures,
        aggregate["pareto_undominated_rate"],
        THRESHOLDS["min_aggregate_pareto_undominated_rate"],
        "aggregate Pareto-undominated rate",
    )
    v10.require_greater(
        failures,
        aggregate["mean_matched_target_margin"],
        THRESHOLDS["min_mean_matched_target_margin"],
        "mean matched target margin",
    )
    v10.require_greater(
        failures,
        aggregate["mean_full_retrieval_minus_matched_compatible_source_output_mse"],
        THRESHOLDS["min_mean_full_retrieval_minus_matched_compatible_source_output_mse"],
        "mean full-retrieval-minus-matched compatible source-output MSE",
    )
    v10.require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy"],
        "aggregate conflict target accuracy",
    )
    v10.require_at_least(
        failures,
        aggregate["mean_conflict_target_accuracy_improvement"],
        THRESHOLDS["min_aggregate_conflict_target_accuracy_improvement"],
        "aggregate conflict target accuracy improvement",
    )
    aggregate_advantage_specs = {
        "shuffled_signature": (
            "min_aggregate_shuffled_target_margin_advantage",
            "min_aggregate_shuffled_compatible_mse_advantage",
        ),
        "source_signature": (
            "min_aggregate_source_signature_target_margin_advantage",
            "min_aggregate_source_signature_compatible_mse_advantage",
        ),
        "target_label": (
            "min_aggregate_target_label_target_margin_advantage",
            "min_aggregate_target_label_compatible_mse_advantage",
        ),
        "v13_no_signature": (
            "min_aggregate_v13_no_signature_target_margin_advantage",
            "min_aggregate_v13_no_signature_compatible_mse_advantage",
        ),
    }
    for metric_name, (margin_key, mse_key) in aggregate_advantage_specs.items():
        if not (
            aggregate[f"mean_matched_minus_{metric_name}_target_margin"]
            > THRESHOLDS[margin_key]
            or aggregate[f"mean_{metric_name}_minus_matched_compatible_source_output_mse"]
            > THRESHOLDS[mse_key]
        ):
            failures.append(f"aggregate {metric_name} advantage gate failed")
    for direction, summary in by_direction.items():
        v10.require_equal(
            failures,
            summary["n"],
            THRESHOLDS["expected_per_direction_count"],
            f"{direction} n",
        )
        v10.require_at_least(
            failures,
            summary["individual_all_gate_pass_rate"],
            THRESHOLDS["min_direction_individual_pass_rate"],
            f"{direction} individual pass rate",
        )
        v10.require_at_least(
            failures,
            summary["pareto_undominated_rate"],
            THRESHOLDS["min_direction_pareto_undominated_rate"],
            f"{direction} Pareto-undominated rate",
        )
        v10.require_greater(
            failures,
            summary["mean_full_retrieval_minus_matched_compatible_source_output_mse"],
            THRESHOLDS["min_direction_mean_full_retrieval_minus_matched_compatible_source_output_mse"],
            f"{direction} full-retrieval-minus-matched compatible source-output MSE",
        )
        v10.require_at_least(
            failures,
            summary["target_prediction_rate"],
            THRESHOLDS["min_direction_target_prediction_rate"],
            f"{direction} target prediction rate",
        )
        v10.require_greater(
            failures,
            summary["mean_matched_target_margin"],
            THRESHOLDS["min_direction_mean_target_margin"],
            f"{direction} target margin",
        )
        v10.require_at_least(
            failures,
            summary["mean_conflict_target_accuracy"],
            THRESHOLDS["min_direction_conflict_target_accuracy"],
            f"{direction} conflict target accuracy",
        )
        v10.require_at_least(
            failures,
            summary["mean_conflict_target_accuracy_improvement"],
            THRESHOLDS["min_direction_conflict_target_accuracy_improvement"],
            f"{direction} conflict target accuracy improvement",
        )
    required_control_types = set(REQUIRED_NON_RANDOM_CONTROL_TYPES)
    for record in records:
        direction = v1.vector_key(record["source_behavior"], record["target_behavior"])
        expected_split = v14.EXPECTED_SPLIT_COUNTS[direction]
        if record["matched"]["compatible_count"] != expected_split["compatible"]:
            failures.append(f"{record['subject_id']} compatible count mismatch")
        if record["matched"]["conflict_count"] != expected_split["conflict"]:
            failures.append(f"{record['subject_id']} conflict count mismatch")
        if len(record["controls"]) != THRESHOLDS["expected_controls_per_record"]:
            failures.append(f"{record['subject_id']} control count mismatch")
        if record["random_control_count"] != THRESHOLDS["random_controls_per_record"]:
            failures.append(f"{record['subject_id']} random control count mismatch")
        control_types = {control["control_type"] for control in record["controls"]}
        for control_type in required_control_types:
            if control_type not in control_types:
                failures.append(f"{record['subject_id']} missing {control_type} control")
    return failures


def train_only_statistics_hash(stats: Mapping[str, Any]) -> str:
    return stable_hash_json({
        "control_model_seeds": CONTROL_MODEL_SEEDS,
        "editor_hidden_dims": EDITOR_HIDDEN_DIMS,
        "editor_input_dim": EDITOR_INPUT_DIM,
        "primary_loss_weights": PRIMARY_LOSS_WEIGHTS,
        "probe_examples_hash": stats.get("probe_examples_hash"),
        "sig_mean": v10.tensor_to_hashable(stats["sig_mean"]),
        "sig_std": v10.tensor_to_hashable(stats["sig_std"]),
        "training_pair_table_hash": stats["training_pair_table_hash"],
        "weight_mean": v10.tensor_to_hashable(stats["weight_mean"]),
        "weight_std": v10.tensor_to_hashable(stats["weight_std"]),
    })


def validate_source_pool_contract(
    *,
    train_path: Path,
    eval_path: Path,
    combined_audit_path: Path,
    final_redacted_path: Path,
    train_payload: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
    combined_audit: Mapping[str, Any],
    final_redacted: Mapping[str, Any],
    phase: str,
) -> list[str]:
    failures = []
    if train_payload.get("claim_scope") != SOURCE_POOL_SCOPE:
        failures.append("train pool claim_scope mismatch")
    if eval_payload.get("claim_scope") != SOURCE_POOL_SCOPE:
        failures.append("eval pool claim_scope mismatch")
    if combined_audit.get("claim_scope") != SOURCE_AUDIT_SCOPE:
        failures.append("combined audit claim_scope mismatch")
    if final_redacted.get("claim_scope") != FINAL_REDACTED_SCOPE:
        failures.append("final redacted audit claim_scope mismatch")

    final_summary = combined_audit.get("pool_summaries", {}).get("final", {})
    forbidden_final_summary = forbidden_combined_final_summary_keys(final_summary)
    if forbidden_final_summary:
        failures.append(
            "combined_audit.pool_summaries.final exposes forbidden keys: "
            + ", ".join(forbidden_final_summary)
        )
    forbidden_redacted_keys = forbidden_final_redacted_keys(final_redacted)
    if forbidden_redacted_keys:
        failures.append(
            "final_redacted exposes forbidden top-level keys: "
            + ", ".join(forbidden_redacted_keys)
        )

    forbidden_surfaces = {
        "combined_audit.pool_summaries.final": final_summary,
        "final_redacted": final_redacted,
    }
    for surface_name, surface in forbidden_surfaces.items():
        forbidden_paths = v10.find_forbidden_final_detail_paths(surface, prefix=surface_name)
        if forbidden_paths:
            failures.append(
                f"{surface_name} exposes forbidden final detail keys: "
                + ", ".join(forbidden_paths[:12])
            )

    if not combined_audit.get("passed", False):
        failures.append("combined source-pool audit did not pass")
    expected_eval_pool = "development" if phase == "development" else "final"
    pool_summaries = combined_audit.get("pool_summaries", {})
    expected_hashes = {
        "train": v1.sha256_file(train_path),
        expected_eval_pool: v1.sha256_file(eval_path),
    }
    for pool_name, actual_hash in expected_hashes.items():
        audit_hash = pool_summaries.get(pool_name, {}).get("pool_file_sha256")
        if audit_hash != actual_hash:
            failures.append(f"{pool_name} pool hash mismatch")
    if final_redacted.get("pool_file_sha256") != pool_summaries.get("final", {}).get(
        "pool_file_sha256"
    ):
        failures.append("final redacted pool hash does not match combined audit")
    if phase == "final":
        final_raw_hash = v1.sha256_file(DEFAULT_POOL_DIR / "final_subjects.json")
        if final_redacted.get("pool_file_sha256") != final_raw_hash:
            failures.append("final redacted pool hash does not match final raw file")
    for pool_name, expected_count in (("train", 64), (expected_eval_pool, 24)):
        counts = pool_summaries.get(pool_name, {}).get("accepted_counts_by_behavior", {})
        for pattern in PATTERNS:
            if counts.get(pattern) != expected_count:
                failures.append(f"{pool_name} accepted count for {pattern} is not {expected_count}")
    for name, value in combined_audit.get("overlap_counts", {}).items():
        if isinstance(value, (int, float)) and value != 0:
            failures.append(f"cross-pool overlap {name} is nonzero")
    if phase == "development":
        assert_no_forbidden_final_raw_paths(
            [train_path, eval_path, combined_audit_path, final_redacted_path],
            allow_v15_final=False,
        )
    return failures


def behavior_of_record(record: Mapping[str, Any]) -> str:
    if "behavior" in record:
        return str(record["behavior"])
    return v1.behavior_of(record)


def records_by_behavior(
    subjects: Sequence[Mapping[str, Any]],
) -> Dict[str, list[Mapping[str, Any]]]:
    return {
        pattern: sorted(
            [record for record in subjects if behavior_of_record(record) == pattern],
            key=lambda record: str(record["subject_id"]),
        )
        for pattern in PATTERNS
    }


def build_training_pair_table(
    subjects: Sequence[Mapping[str, Any]],
) -> list[Dict[str, str]]:
    by_behavior = records_by_behavior(subjects)
    pairs = []
    for source in PATTERNS:
        for source_record in by_behavior[source]:
            for target in PATTERNS:
                if target == source:
                    continue
                for target_record in by_behavior[target]:
                    pairs.append({
                        "source_behavior": source,
                        "source_subject_id": str(source_record["subject_id"]),
                        "target_behavior": target,
                        "target_subject_id": str(target_record["subject_id"]),
                    })
    return pairs


def deterministic_random_signature(
    *,
    key_parts: Sequence[Any],
    dim: int = SIGNATURE_DIM,
) -> torch.Tensor:
    key = stable_hash_json(list(key_parts))
    seed = int(key[:16], 16) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=generator, dtype=torch.float32)


if __name__ == "__main__":
    main()
