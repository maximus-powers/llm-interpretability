"""Four-behavior representation-space steering under a frozen preregistration."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
        "base_seed": 30300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 31300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 32300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}
TRAINING_CONFIG = {
    "classifier_epochs": 1000,
    "classifier_lr": 0.10,
    "classifier_seed": 20260621,
    "classifier_weight_decay": 0.0001,
    "edit_epochs": 500,
    "edit_grad_clip_norm": 10.0,
    "edit_lr": 0.05,
    "edit_seed": 20260620,
    "edit_train_random_controls": 8,
    "edit_weight_decay": 0.0,
}
THRESHOLDS = {
    "expected_record_count": 288,
    "expected_per_target_count": 72,
    "expected_per_direction_count": 24,
    "max_mean_source_primary_margin_change": -0.05,
    "min_aggregate_individual_pass_rate": 0.90,
    "min_direction_individual_pass_rate": 0.90,
    "min_direction_matched_minus_best_control_primary_target_margin": 0.0,
    "min_direction_matched_primary_target_margin": 0.10,
    "min_mean_matched_centroid_improvement": 0.15,
    "min_mean_matched_minus_best_control_centroid_improvement": 0.10,
    "min_mean_matched_minus_best_control_primary_target_margin": 0.15,
    "min_mean_matched_primary_target_margin": 0.20,
    "min_per_record_centroid_improvement": 0.0,
    "min_per_record_matched_minus_best_control_centroid_improvement": 0.0,
    "min_per_record_matched_minus_best_control_primary_target_margin": 0.0,
    "min_per_record_primary_target_margin": 0.10,
    "min_per_target_centroid_improvement": 0.10,
    "min_per_target_individual_pass_rate": 0.80,
    "min_per_target_matched_minus_best_control_primary_target_margin": 0.10,
    "min_per_target_primary_target_margin": 0.15,
    "per_record_source_primary_margin_change_must_be_below": 0.0,
}
DECODER_FINAL_RAW = (
    REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / "final_subjects.json"
)


class LinearSignatureEvaluator(nn.Module):
    """Single affine behavior classifier over normalized signatures."""

    def __init__(self, signature_dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(signature_dim, n_classes)

    def forward(self, signatures: torch.Tensor) -> torch.Tensor:
        return self.linear(signatures.float())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["generate-pools", "development", "final"],
        required=True,
    )
    parser.add_argument(
        "--pool-dir",
        default="runs/four_behavior_representation_steering_v1_pools",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/four_behavior_representation_steering_v1",
    )
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=160)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    parser.add_argument("--random-controls", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_dir = REPO_ROOT / args.pool_dir
    output_dir = REPO_ROOT / args.output_dir
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


def assert_no_forbidden_final_raw_paths(
    paths: Iterable[Path | str],
    *,
    allow_steering_final: bool,
) -> None:
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path == DECODER_FINAL_RAW.resolve():
            raise ValueError(f"sealed decoder final raw path is forbidden: {path}")
        if path.name != "final_subjects.json":
            continue
        if allow_steering_final:
            continue
        raise ValueError(f"steering final raw path is forbidden before final eval: {path}")


def build_seed_preflight(
    configs: Mapping[str, Mapping[str, int]],
    *,
    behavior_stride: int,
) -> Dict[str, Any]:
    ranges = []
    for pool_name, pool_config in configs.items():
        for pattern_index, pattern in enumerate(PATTERNS):
            start = int(pool_config["base_seed"]) + pattern_index * behavior_stride
            end = start + int(pool_config["max_attempts_per_behavior"]) - 1
            ranges.append({
                "end_seed": int(end),
                "max_attempts": int(pool_config["max_attempts_per_behavior"]),
                "pattern": pattern,
                "pool": pool_name,
                "start_seed": int(start),
            })
    failures = []
    for left_index, left in enumerate(ranges):
        for right in ranges[left_index + 1:]:
            disjoint = left["end_seed"] < right["start_seed"] or right["end_seed"] < left["start_seed"]
            if not disjoint:
                failures.append(
                    "seed range overlap: "
                    f"{left['pool']}/{left['pattern']} "
                    f"{left['start_seed']}..{left['end_seed']} vs "
                    f"{right['pool']}/{right['pattern']} "
                    f"{right['start_seed']}..{right['end_seed']}"
                )
    return {
        "failures": failures,
        "passed": not failures,
        "seed_behavior_stride": int(behavior_stride),
        "seed_ranges": ranges,
    }


def centroid_delta_control(
    centroids: Mapping[str, torch.Tensor],
    source_behavior: str,
    target_behavior: str,
) -> torch.Tensor:
    return centroids[target_behavior] - centroids[source_behavior]


def select_shuffled_vector_key(
    subject_id: str,
    source_behavior: str,
    target_behavior: str,
) -> tuple[str, str]:
    candidates = sorted(
        (candidate_source, candidate_target)
        for candidate_source in PATTERNS
        for candidate_target in PATTERNS
        if candidate_source != candidate_target
        and candidate_source != source_behavior
        and candidate_target != target_behavior
    )
    digest = stable_hash_json([
        subject_id,
        source_behavior,
        target_behavior,
        "representation_steering_v1_shuffled_vector",
        20260619,
    ])
    index = int(digest[:16], 16) % len(candidates)
    return candidates[index]


def generate_pools(args: argparse.Namespace, pool_dir: Path) -> Dict[str, Any]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_seed_preflight(
        POOL_CONFIGS,
        behavior_stride=SEED_BEHAVIOR_STRIDE,
    )
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
    pool_payloads = {}
    pool_summaries = {}
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
        payload["claim_scope"] = "four_behavior_representation_steering_source_pool"
        payload["config"]["base_seed"] = int(pool_config["base_seed"])
        payload["config"]["seed_behavior_stride"] = int(SEED_BEHAVIOR_STRIDE)
        payload["pool_redacted_payload_sha256"] = stable_hash_json(
            poolgen.redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_path = pool_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = poolgen.summarize_pool(payload)
        summary["pool_file_sha256"] = sha256_file(pool_path)
        summary["pool_redacted_payload_sha256"] = payload["pool_redacted_payload_sha256"]
        pool_summaries[pool_name] = summary

    final_redacted = poolgen.build_final_redacted_summary(pool_payloads["final"])
    final_redacted["claim_scope"] = "redacted_final_steering_source_pool_audit_surface_only"
    final_redacted["pool_file_sha256"] = pool_summaries["final"]["pool_file_sha256"]
    final_redacted["summary_payload_sha256"] = stable_hash_json(final_redacted)
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
    audit["claim_scope"] = "four_behavior_representation_steering_source_pool_construction"
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (pool_dir / "combined_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return {
        "combined_audit_path": rel(pool_dir / "combined_audit.json"),
        "final_redacted_audit_path": rel(pool_dir / "final_redacted_audit.json"),
        "passed": bool(audit["passed"]),
        "pool_summaries": audit["pool_summaries"],
        "seed_preflight": seed_preflight,
    }


def run_development(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    paths = development_input_paths(pool_dir)
    assert_no_forbidden_final_raw_paths(
        paths.values(),
        allow_steering_final=False,
    )
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


def development_input_paths(pool_dir: Path) -> Dict[str, Path]:
    return {
        "combined_audit": pool_dir / "combined_audit.json",
        "development": pool_dir / "development_subjects.json",
        "final_redacted_audit": pool_dir / "final_redacted_audit.json",
        "train": pool_dir / "train_subjects.json",
    }


def run_final(args: argparse.Namespace, pool_dir: Path, output_dir: Path) -> Dict[str, Any]:
    development_path = output_dir / "development_results.json"
    if not development_path.exists():
        raise FileNotFoundError("development_results.json is required before final eval")
    development = json.loads(development_path.read_text())
    if not development.get("passed", False):
        raise RuntimeError("development did not pass; final raw pool must remain sealed")
    eval_path = pool_dir / "final_subjects.json"
    assert_no_forbidden_final_raw_paths([DECODER_FINAL_RAW], allow_steering_final=False)
    assert_no_forbidden_final_raw_paths([eval_path], allow_steering_final=True)
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
        allow_steering_final=allow_eval_final_raw,
    )
    train_payload = load_json(train_path)
    eval_payload = load_json(eval_path)
    combined_audit = load_json(combined_audit_path)
    final_redacted = load_json(final_redacted_path)
    train_subjects = accepted_records(train_payload)
    eval_subjects = accepted_records(eval_payload)
    train_stats = fit_train_statistics(train_subjects)
    classifier, classifier_summary = fit_primary_classifier(
        train_stats["z_train"],
        train_stats["y_train"],
    )
    vectors_payload = train_edit_vectors(
        z_train=train_stats["z_train"],
        y_train=train_stats["y_train"],
        classifier=classifier,
        centroids=train_stats["centroids"],
    )
    vector_path = output_dir / "edit_vectors.pt"
    torch.save(vectors_payload, vector_path)
    eval_result = evaluate_subjects(
        subjects=eval_subjects,
        train_stats=train_stats,
        classifier=classifier,
        vectors=vectors_payload["edit_vectors"],
        phase=phase,
        random_controls=random_controls,
    )
    failures = list(eval_result["failures"])
    overlap_counts = combined_audit.get("overlap_counts", {})
    for key, value in overlap_counts.items():
        if int(value) != 0:
            failures.append(f"cross-pool accepted {key} overlap count {value}")
    if not combined_audit.get("passed", False):
        failures.append("combined source-pool audit did not pass")
    result = {
        **eval_result,
        "classifier_summary": classifier_summary,
        "claim_scope": (
            "four_behavior_representation_steering_development"
            if phase == "development"
            else "four_behavior_representation_steering_final"
        ),
        "combined_audit_path": rel(combined_audit_path),
        "combined_audit_sha256": sha256_file(combined_audit_path),
        "dirty_worktree_caveat": True,
        "edit_vector_initialization": "train_centroid_delta",
        "edit_vectors_path": rel(vector_path),
        "eval_pool_path": rel(eval_path) if allow_eval_final_raw else rel(eval_path),
        "eval_pool_sha256": sha256_file(eval_path),
        "final_redacted_audit_path": rel(final_redacted_path),
        "final_redacted_audit_sha256": sha256_file(final_redacted_path),
        "final_redacted_summary": final_redacted,
        "forbidden_decoder_final_raw_opened": False,
        "objective_correction_status": "corrected_no_edit_relative_centroid_improvement",
        "phase": phase,
        "pool_overlap_counts": overlap_counts,
        "probe_examples_hash": combined_audit.get("probe_examples_hash"),
        "source_pool_audit_passed": bool(combined_audit.get("passed", False)),
        "thresholds": THRESHOLDS,
        "train_only_statistics_hash": stable_hash_json({
            "signature_mean": tensor_to_float_list(train_stats["sig_mean"]),
            "signature_std": tensor_to_float_list(train_stats["sig_std"]),
            "centroids": {
                key: tensor_to_float_list(value)
                for key, value in train_stats["centroids"].items()
            },
        }),
        "train_pool_path": rel(train_path),
        "train_pool_sha256": sha256_file(train_path),
        "training_config": TRAINING_CONFIG,
        "supersedes_development_artifact": (
            "flawed_v1_centroid_objective_55_of_288_passes"
        ),
        "vector_norms": {
            key: float(value.norm().item())
            for key, value in vectors_payload["edit_vectors"].items()
        },
        "vector_training_summary": vectors_payload["summary"],
    }
    result["failures"] = failures
    result["passed"] = not failures
    return result


def accepted_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [record for record in payload["records"] if record["accepted"]]


def behavior_of(record: Mapping[str, Any]) -> str:
    return str(record.get("target_pattern", record.get("pattern")))


def fit_train_statistics(subjects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    signatures = torch.tensor(
        [record["signature"] for record in subjects],
        dtype=torch.float32,
    )
    labels = torch.tensor([PATTERNS.index(behavior_of(record)) for record in subjects])
    sig_mean = signatures.mean(dim=0)
    sig_std = signatures.std(dim=0, unbiased=False)
    sig_std = torch.where(sig_std < 1e-6, torch.ones_like(sig_std), sig_std)
    z_train = (signatures - sig_mean) / sig_std
    centroids = {}
    for pattern_index, pattern in enumerate(PATTERNS):
        centroids[pattern] = z_train[labels == pattern_index].mean(dim=0)
    return {
        "centroids": centroids,
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "y_train": labels,
        "z_train": z_train,
    }


def fit_primary_classifier(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
) -> tuple[LinearSignatureEvaluator, Dict[str, Any]]:
    torch.manual_seed(TRAINING_CONFIG["classifier_seed"])
    model = LinearSignatureEvaluator(z_train.size(1), len(PATTERNS))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["classifier_lr"],
        weight_decay=TRAINING_CONFIG["classifier_weight_decay"],
    )
    history = []
    for epoch in range(1, TRAINING_CONFIG["classifier_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(z_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == TRAINING_CONFIG["classifier_epochs"]:
            pred = logits.argmax(dim=1)
            history.append({
                "accuracy": float((pred == y_train).float().mean().item()),
                "epoch": epoch,
                "loss": float(loss.item()),
            })
    model.eval()
    with torch.no_grad():
        final_logits = model(z_train)
        final_pred = final_logits.argmax(dim=1)
    return model, {
        "final_train_accuracy": float((final_pred == y_train).float().mean().item()),
        "history": history,
    }


def train_edit_vectors(
    *,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    classifier: LinearSignatureEvaluator,
    centroids: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    torch.manual_seed(TRAINING_CONFIG["edit_seed"])
    vectors = nn.ParameterDict()
    for source in PATTERNS:
        for target in PATTERNS:
            if source == target:
                continue
            key = vector_key(source, target)
            vectors[key] = nn.Parameter(centroid_delta_control(centroids, source, target).clone())
    optimizer = torch.optim.AdamW(
        vectors.parameters(),
        lr=TRAINING_CONFIG["edit_lr"],
        weight_decay=TRAINING_CONFIG["edit_weight_decay"],
    )
    generator = torch.Generator().manual_seed(TRAINING_CONFIG["edit_seed"] + 17)
    best = {"epoch": None, "loss": float("inf"), "state": None}
    history_tail = []
    for epoch in range(1, TRAINING_CONFIG["edit_epochs"] + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        metrics = {}
        for source in PATTERNS:
            source_index = PATTERNS.index(source)
            source_z = z_train[y_train == source_index]
            for target in PATTERNS:
                if source == target:
                    continue
                key = vector_key(source, target)
                loss, direction_metrics = edit_direction_loss(
                    source_z=source_z,
                    source=source,
                    target=target,
                    vector=vectors[key],
                    classifier=classifier,
                    centroids=centroids,
                    generator=generator,
                )
                losses.append(loss)
                metrics.update(direction_metrics)
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(vectors.parameters(), TRAINING_CONFIG["edit_grad_clip_norm"])
        optimizer.step()
        if total_loss.item() < best["loss"]:
            best = {
                "epoch": epoch,
                "loss": float(total_loss.item()),
                "state": {
                    key: value.detach().clone()
                    for key, value in vectors.items()
                },
            }
        if epoch == 1 or epoch % 50 == 0 or epoch == TRAINING_CONFIG["edit_epochs"]:
            history_tail.append({
                "epoch": epoch,
                "train_objective": float(total_loss.item()),
                **metrics,
            })
            history_tail = history_tail[-20:]
    if best["state"] is None:
        raise RuntimeError("No edit-vector checkpoint selected")
    return {
        "edit_vectors": best["state"],
        "summary": {
            "best_epoch": best["epoch"],
            "best_train_objective": best["loss"],
            "history_tail": history_tail,
        },
    }


def edit_direction_loss(
    *,
    source_z: torch.Tensor,
    source: str,
    target: str,
    vector: torch.Tensor,
    classifier: LinearSignatureEvaluator,
    centroids: Mapping[str, torch.Tensor],
    generator: torch.Generator,
) -> tuple[torch.Tensor, Dict[str, float]]:
    steered = source_z + vector
    no_edit_scores = representation_scores(source_z, source, target, classifier, centroids)
    steered_scores = representation_scores(steered, source, target, classifier, centroids)
    random_margins = []
    for _ in range(TRAINING_CONFIG["edit_train_random_controls"]):
        random_vector = torch.randn(
            vector.shape,
            dtype=vector.dtype,
            generator=generator,
        )
        random_vector = random_vector / random_vector.norm().clamp_min(1e-12)
        random_vector = random_vector * vector.detach().norm().clamp_min(1e-12)
        random_scores = representation_scores(
            source_z + random_vector,
            source,
            target,
            classifier,
            centroids,
        )
        random_margins.append(random_scores["primary_target_margin"].detach())
    worst_random_target_margin = torch.stack(random_margins).max(dim=0).values
    target_margin = steered_scores["primary_target_margin"]
    target_improvement = target_margin - no_edit_scores["primary_target_margin"]
    source_change = (
        steered_scores["primary_source_margin"]
        - no_edit_scores["primary_source_margin"]
    )
    centroid_improvement = centroid_improvement_relative_to_no_edit(
        no_edit_z=source_z,
        candidate_z=steered,
        target_behavior=target,
        centroids=centroids,
    )
    random_delta = target_margin - worst_random_target_margin
    loss = (
        2.0 * F.relu(0.35 - target_margin).mean()
        + F.relu(0.25 - target_improvement).mean()
        + F.relu(source_change + 0.10).mean()
        + F.relu(0.25 - centroid_improvement).mean()
        + F.relu(0.20 - random_delta).mean()
        + 0.0001 * vector.pow(2).mean()
    )
    key = vector_key(source, target)
    return loss, {
        f"{key}/centroid_improvement": float(centroid_improvement.mean().item()),
        f"{key}/source_change": float(source_change.mean().item()),
        f"{key}/target_margin": float(target_margin.mean().item()),
        f"{key}/vector_norm": float(vector.detach().norm().item()),
    }


def representation_scores(
    z: torch.Tensor,
    source: str,
    target: str,
    classifier: LinearSignatureEvaluator,
    centroids: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if z.dim() == 1:
        z = z.unsqueeze(0)
    logits = classifier(z)
    source_index = PATTERNS.index(source)
    target_index = PATTERNS.index(target)
    target_margin = class_margin(logits, target_index)
    source_margin = class_margin(logits, source_index)
    target_distance = torch.linalg.norm(z - centroids[target], dim=1)
    return {
        "logits": logits,
        "primary_source_margin": source_margin,
        "primary_target_margin": target_margin,
        "centroid_distance_to_target": target_distance,
    }


def centroid_improvement_relative_to_no_edit(
    *,
    no_edit_z: torch.Tensor,
    candidate_z: torch.Tensor,
    target_behavior: str,
    centroids: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if no_edit_z.dim() == 1:
        no_edit_z = no_edit_z.unsqueeze(0)
    if candidate_z.dim() == 1:
        candidate_z = candidate_z.unsqueeze(0)
    target_centroid = centroids[target_behavior]
    no_edit_distance = torch.linalg.norm(no_edit_z - target_centroid, dim=1)
    candidate_distance = torch.linalg.norm(candidate_z - target_centroid, dim=1)
    return no_edit_distance - candidate_distance


def class_margin(logits: torch.Tensor, class_index: int) -> torch.Tensor:
    target = logits[:, class_index]
    masked = logits.clone()
    masked[:, class_index] = -torch.inf
    return target - masked.max(dim=1).values


def evaluate_subjects(
    *,
    subjects: Sequence[Mapping[str, Any]],
    train_stats: Mapping[str, Any],
    classifier: LinearSignatureEvaluator,
    vectors: Mapping[str, torch.Tensor],
    phase: str,
    random_controls: int,
) -> Dict[str, Any]:
    records = []
    for subject in subjects:
        source = behavior_of(subject)
        signature = torch.tensor(subject["signature"], dtype=torch.float32)
        z = (signature - train_stats["sig_mean"]) / train_stats["sig_std"]
        for target in PATTERNS:
            if target == source:
                continue
            records.append(evaluate_record(
                subject=subject,
                z=z,
                source=source,
                target=target,
                classifier=classifier,
                centroids=train_stats["centroids"],
                vectors=vectors,
                phase=phase,
                random_controls=random_controls,
            ))
    aggregate = summarize_records(records)
    by_target = {
        target: summarize_records([record for record in records if record["target_behavior"] == target])
        for target in PATTERNS
    }
    by_direction = {
        vector_key(source, target): summarize_records([
            record for record in records
            if record["source_behavior"] == source and record["target_behavior"] == target
        ])
        for source in PATTERNS
        for target in PATTERNS
        if source != target
    }
    individual_audit = individual_gate_audit(records)
    failures = gate_failures(
        aggregate=aggregate,
        by_target=by_target,
        by_direction=by_direction,
        individual_audit=individual_audit,
    )
    return {
        "aggregate": aggregate,
        "by_direction": by_direction,
        "by_target": by_target,
        "failures": failures,
        "individual_gate_audit": individual_audit,
        "passed": not failures,
        "records": records,
    }


def evaluate_record(
    *,
    subject: Mapping[str, Any],
    z: torch.Tensor,
    source: str,
    target: str,
    classifier: LinearSignatureEvaluator,
    centroids: Mapping[str, torch.Tensor],
    vectors: Mapping[str, torch.Tensor],
    phase: str,
    random_controls: int,
) -> Dict[str, Any]:
    no_edit = score_candidate(
        z=z,
        candidate_z=z,
        source=source,
        target=target,
        classifier=classifier,
        centroids=centroids,
    )
    matched_vector = vectors[vector_key(source, target)]
    matched = score_candidate(
        z=z,
        candidate_z=z + matched_vector,
        source=source,
        target=target,
        classifier=classifier,
        centroids=centroids,
    )
    controls = build_controls(
        subject_id=str(subject["subject_id"]),
        z=z,
        source=source,
        target=target,
        matched_norm=float(matched_vector.norm().item()),
        classifier=classifier,
        centroids=centroids,
        vectors=vectors,
        phase=phase,
        random_controls=random_controls,
    )
    for control in controls:
        control["matched_minus_control_primary_target_margin"] = (
            matched["primary_target_margin"] - control["primary_target_margin"]
        )
        control["matched_minus_control_centroid_improvement"] = (
            matched["centroid_improvement"] - control["centroid_improvement"]
        )
    best_primary = max(controls, key=lambda item: item["primary_target_margin"])
    best_centroid = max(controls, key=lambda item: item["centroid_improvement"])
    summary = {
        "best_centroid_control_type": best_centroid["control_type"],
        "best_control_centroid_improvement": best_centroid["centroid_improvement"],
        "best_control_primary_target_margin": best_primary["primary_target_margin"],
        "best_primary_control_type": best_primary["control_type"],
        "matched_centroid_improvement": matched["centroid_improvement"],
        "matched_minus_best_control_centroid_improvement": (
            matched["centroid_improvement"] - best_centroid["centroid_improvement"]
        ),
        "matched_minus_best_control_primary_target_margin": (
            matched["primary_target_margin"] - best_primary["primary_target_margin"]
        ),
        "matched_primary_target_margin": matched["primary_target_margin"],
        "source_primary_margin_change": (
            matched["primary_source_margin"] - no_edit["primary_source_margin"]
        ),
    }
    passed = record_passes(summary)
    return {
        "controls": controls,
        "individual_all_gates_passed": passed,
        "matched": matched,
        "random_control_count": int(random_controls),
        "source_behavior": source,
        "subject_id": str(subject["subject_id"]),
        "summary": summary,
        "target_behavior": target,
        "vector_key": vector_key(source, target),
        "vector_norm": float(matched_vector.norm().item()),
    }


def build_controls(
    *,
    subject_id: str,
    z: torch.Tensor,
    source: str,
    target: str,
    matched_norm: float,
    classifier: LinearSignatureEvaluator,
    centroids: Mapping[str, torch.Tensor],
    vectors: Mapping[str, torch.Tensor],
    phase: str,
    random_controls: int,
) -> list[Dict[str, Any]]:
    control_vectors: list[tuple[str, torch.Tensor, Dict[str, Any]]] = [
        ("no_edit", torch.zeros_like(z), {}),
        ("null_vector", torch.zeros_like(z), {}),
        ("reverse_edit_vector", vectors[vector_key(target, source)], {}),
        (
            "target_source_centroid_delta",
            centroid_delta_control(centroids, source, target),
            {},
        ),
    ]
    for other_target in PATTERNS:
        if other_target not in (source, target):
            control_vectors.append((
                "same_source_other_target_edit_vector",
                vectors[vector_key(source, other_target)],
                {"control_target_behavior": other_target},
            ))
    if phase == "final":
        shuffled_source, shuffled_target = select_shuffled_vector_key(subject_id, source, target)
        control_vectors.append((
            "shuffled_target_edit_vector",
            vectors[vector_key(shuffled_source, shuffled_target)],
            {
                "control_source_behavior": shuffled_source,
                "control_target_behavior": shuffled_target,
            },
        ))
    controls = []
    for control_type, vector, extra in control_vectors:
        controls.append({
            "control_type": control_type,
            **extra,
            **score_candidate(
                z=z,
                candidate_z=z + vector,
                source=source,
                target=target,
                classifier=classifier,
                centroids=centroids,
            ),
        })
    generator = torch.Generator().manual_seed(random_seed_for_record(subject_id, source, target))
    for index in range(random_controls):
        random_vector = torch.randn(z.shape, dtype=z.dtype, generator=generator)
        random_vector = random_vector / random_vector.norm().clamp_min(1e-12)
        random_vector = random_vector * matched_norm
        controls.append({
            "control_type": "random_norm_matched_vector",
            "random_index": index,
            **score_candidate(
                z=z,
                candidate_z=z + random_vector,
                source=source,
                target=target,
                classifier=classifier,
                centroids=centroids,
            ),
        })
    return controls


def score_candidate(
    *,
    z: torch.Tensor,
    candidate_z: torch.Tensor,
    source: str,
    target: str,
    classifier: LinearSignatureEvaluator,
    centroids: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    if candidate_z.dim() == 1:
        candidate_z_batch = candidate_z.unsqueeze(0)
    else:
        candidate_z_batch = candidate_z
    if z.dim() == 1:
        z_batch = z.unsqueeze(0)
    else:
        z_batch = z
    logits = classifier(candidate_z_batch).detach()
    no_edit_logits = classifier(z_batch).detach()
    target_index = PATTERNS.index(target)
    source_index = PATTERNS.index(source)
    centroid_distances = torch.stack([
        torch.linalg.norm(candidate_z_batch - centroids[pattern], dim=1)
        for pattern in PATTERNS
    ], dim=1)
    pred_index = int(logits.argmax(dim=1).item())
    centroid_pred_index = int(centroid_distances.argmin(dim=1).item())
    return {
        "centroid_improvement": float(
            centroid_improvement_relative_to_no_edit(
                no_edit_z=z_batch,
                candidate_z=candidate_z_batch,
                target_behavior=target,
                centroids=centroids,
            ).item()
        ),
        "centroid_predicted_behavior": PATTERNS[centroid_pred_index],
        "primary_predicted_behavior": PATTERNS[pred_index],
        "primary_source_margin": float(class_margin(logits, source_index).item()),
        "primary_source_margin_change": float(
            (
                class_margin(logits, source_index)
                - class_margin(no_edit_logits, source_index)
            ).item()
        ),
        "primary_target_margin": float(class_margin(logits, target_index).item()),
    }


def random_seed_for_record(subject_id: str, source: str, target: str) -> int:
    digest = stable_hash_json([
        subject_id,
        source,
        target,
        "representation_steering_v1_random",
        20260619,
    ])
    return int(digest[:16], 16) % (2 ** 31)


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries = [record["summary"] for record in records]
    return {
        "individual_all_gate_pass_count": sum(
            1 for record in records if record["individual_all_gates_passed"]
        ),
        "individual_all_gate_pass_rate": (
            sum(1 for record in records if record["individual_all_gates_passed"]) / len(records)
            if records else 0.0
        ),
        "mean_matched_centroid_improvement": mean([
            item["matched_centroid_improvement"] for item in summaries
        ]),
        "mean_matched_minus_best_control_centroid_improvement": mean([
            item["matched_minus_best_control_centroid_improvement"] for item in summaries
        ]),
        "mean_matched_minus_best_control_primary_target_margin": mean([
            item["matched_minus_best_control_primary_target_margin"] for item in summaries
        ]),
        "mean_matched_primary_target_margin": mean([
            item["matched_primary_target_margin"] for item in summaries
        ]),
        "mean_source_primary_margin_change": mean([
            item["source_primary_margin_change"] for item in summaries
        ]),
        "n": len(records),
    }


def individual_gate_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures = []
    for record in records:
        summary = record["summary"]
        failed = []
        checks = [
            (
                "matched_primary_target_margin",
                summary["matched_primary_target_margin"],
                THRESHOLDS["min_per_record_primary_target_margin"],
                ">=",
            ),
            (
                "matched_minus_best_control_primary_target_margin",
                summary["matched_minus_best_control_primary_target_margin"],
                THRESHOLDS["min_per_record_matched_minus_best_control_primary_target_margin"],
                ">",
            ),
            (
                "matched_centroid_improvement",
                summary["matched_centroid_improvement"],
                THRESHOLDS["min_per_record_centroid_improvement"],
                ">",
            ),
            (
                "matched_minus_best_control_centroid_improvement",
                summary["matched_minus_best_control_centroid_improvement"],
                THRESHOLDS["min_per_record_matched_minus_best_control_centroid_improvement"],
                ">",
            ),
            (
                "source_primary_margin_change",
                summary["source_primary_margin_change"],
                THRESHOLDS["per_record_source_primary_margin_change_must_be_below"],
                "<",
            ),
        ]
        for name, value, threshold, operator in checks:
            if not passes(value, threshold, operator):
                failed.append({
                    "check": name,
                    "operator": operator,
                    "threshold": threshold,
                    "value": value,
                })
        if failed:
            failures.append({
                "failed": failed,
                "source_behavior": record["source_behavior"],
                "subject_id": record["subject_id"],
                "target_behavior": record["target_behavior"],
            })
    return {
        "all_gate_pass_count": len(records) - len(failures),
        "all_gate_pass_rate": (
            (len(records) - len(failures)) / len(records) if records else 0.0
        ),
        "failed_records": failures,
        "n": len(records),
    }


def gate_failures(
    *,
    aggregate: Mapping[str, Any],
    by_target: Mapping[str, Mapping[str, Any]],
    by_direction: Mapping[str, Mapping[str, Any]],
    individual_audit: Mapping[str, Any],
) -> list[str]:
    failures = []
    aggregate_checks = [
        ("n", aggregate["n"], THRESHOLDS["expected_record_count"], "=="),
        (
            "mean_matched_primary_target_margin",
            aggregate["mean_matched_primary_target_margin"],
            THRESHOLDS["min_mean_matched_primary_target_margin"],
            ">=",
        ),
        (
            "mean_matched_minus_best_control_primary_target_margin",
            aggregate["mean_matched_minus_best_control_primary_target_margin"],
            THRESHOLDS["min_mean_matched_minus_best_control_primary_target_margin"],
            ">=",
        ),
        (
            "mean_matched_centroid_improvement",
            aggregate["mean_matched_centroid_improvement"],
            THRESHOLDS["min_mean_matched_centroid_improvement"],
            ">=",
        ),
        (
            "mean_matched_minus_best_control_centroid_improvement",
            aggregate["mean_matched_minus_best_control_centroid_improvement"],
            THRESHOLDS["min_mean_matched_minus_best_control_centroid_improvement"],
            ">=",
        ),
        (
            "mean_source_primary_margin_change",
            aggregate["mean_source_primary_margin_change"],
            THRESHOLDS["max_mean_source_primary_margin_change"],
            "<=",
        ),
    ]
    for name, value, threshold, operator in aggregate_checks:
        if not passes(float(value), float(threshold), operator):
            failures.append(format_failure("aggregate", name, value, threshold, operator))
    if individual_audit["all_gate_pass_rate"] < THRESHOLDS["min_aggregate_individual_pass_rate"]:
        failures.append(
            format_failure(
                "aggregate",
                "individual pass rate",
                individual_audit["all_gate_pass_rate"],
                THRESHOLDS["min_aggregate_individual_pass_rate"],
                ">=",
            )
        )
    for target, summary in by_target.items():
        target_checks = [
            ("n", summary["n"], THRESHOLDS["expected_per_target_count"], "=="),
            (
                "mean matched primary target margin",
                summary["mean_matched_primary_target_margin"],
                THRESHOLDS["min_per_target_primary_target_margin"],
                ">=",
            ),
            (
                "mean matched-minus-best-control primary target margin",
                summary["mean_matched_minus_best_control_primary_target_margin"],
                THRESHOLDS["min_per_target_matched_minus_best_control_primary_target_margin"],
                ">=",
            ),
            (
                "mean matched centroid improvement",
                summary["mean_matched_centroid_improvement"],
                THRESHOLDS["min_per_target_centroid_improvement"],
                ">=",
            ),
            (
                "individual pass rate",
                summary["individual_all_gate_pass_rate"],
                THRESHOLDS["min_per_target_individual_pass_rate"],
                ">=",
            ),
        ]
        for name, value, threshold, operator in target_checks:
            if not passes(float(value), float(threshold), operator):
                failures.append(format_failure(f"target {target}", name, value, threshold, operator))
    for direction, summary in by_direction.items():
        direction_checks = [
            ("n", summary["n"], THRESHOLDS["expected_per_direction_count"], "=="),
            (
                "mean matched primary target margin",
                summary["mean_matched_primary_target_margin"],
                THRESHOLDS["min_direction_matched_primary_target_margin"],
                ">=",
            ),
            (
                "mean matched-minus-best-control primary target margin",
                summary["mean_matched_minus_best_control_primary_target_margin"],
                THRESHOLDS["min_direction_matched_minus_best_control_primary_target_margin"],
                ">",
            ),
            (
                "individual pass rate",
                summary["individual_all_gate_pass_rate"],
                THRESHOLDS["min_direction_individual_pass_rate"],
                ">=",
            ),
        ]
        for name, value, threshold, operator in direction_checks:
            if not passes(float(value), float(threshold), operator):
                failures.append(format_failure(f"direction {direction}", name, value, threshold, operator))
    return failures


def format_failure(
    scope: str,
    name: str,
    value: float,
    threshold: float,
    operator: str,
) -> str:
    failure_operator = {
        "==": "!=",
        ">=": "<",
        ">": "<=",
        "<=": ">",
        "<": ">=",
    }[operator]
    return (
        f"{scope} {name} failed: observed {value} "
        f"{failure_operator} required {threshold}"
    )


def record_passes(summary: Mapping[str, float]) -> bool:
    return (
        summary["matched_primary_target_margin"]
        >= THRESHOLDS["min_per_record_primary_target_margin"]
        and summary["matched_minus_best_control_primary_target_margin"]
        > THRESHOLDS["min_per_record_matched_minus_best_control_primary_target_margin"]
        and summary["matched_centroid_improvement"]
        > THRESHOLDS["min_per_record_centroid_improvement"]
        and summary["matched_minus_best_control_centroid_improvement"]
        > THRESHOLDS["min_per_record_matched_minus_best_control_centroid_improvement"]
        and summary["source_primary_margin_change"]
        < THRESHOLDS["per_record_source_primary_margin_change_must_be_below"]
    )


def vector_key(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def passes(value: float, threshold: float, operator: str) -> bool:
    if operator == "==":
        return value == threshold
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    raise ValueError(f"Unsupported operator: {operator}")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
