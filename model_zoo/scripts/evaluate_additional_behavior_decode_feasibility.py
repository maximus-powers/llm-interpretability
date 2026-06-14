"""Feasibility check for additional-behavior stored-probe decoding.

This evaluates the already locked stored-probe decoder on freshly trained source
models for clean behaviors that were not part of the two-behavior decode proof.
It is deliberately labeled as feasibility: failure is useful evidence about the
current scope, while passing would require a separate preregistered final proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
sys.path.insert(0, str(MODEL_ZOO_ROOT))

from hypernet.behavior_suite import (  # noqa: E402
    PREDICATES,
    build_clean_behavior_suite,
    enumerate_sequence_universe,
)
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.models import SubjectNetwork  # noqa: E402
from hypernet.paired_contrast import (  # noqa: E402
    build_digit_probe_examples,
    extract_signature_with_stored_probes,
    signature_hash_stable_float_list,
)


PATTERNS = ("has_majority", "mountain_pattern")
ALL_CLEAN_PATTERNS = (
    "sorted_ascending",
    "sorted_descending",
    "has_majority",
    "mountain_pattern",
)
THRESHOLDS = {
    "min_mean_matched_target_margin": 0.20,
    "min_mean_matched_minus_noise_target_margin": 0.20,
    "min_individual_pass_rate": 0.90,
    "min_per_behavior_individual_pass_rate": 0.80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/stored_probe_additional_behavior_decode_feasibility_v1",
    )
    parser.add_argument(
        "--holdout-dir",
        default="runs/fresh_additional_behavior_decode_holdout_v1",
    )
    parser.add_argument("--decoder-path", default="runs/stored_probe_functional_decoder_v2_adaptive/model.pt")
    parser.add_argument("--n-per-behavior", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260930)
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--max-train-positives", type=int, default=512)
    parser.add_argument("--negative-multiple", type=int, default=2)
    parser.add_argument("--noise-controls", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260931)
    parser.add_argument("--force-regenerate-holdout", action="store_true")
    return parser.parse_args()


def load_decoder(decoder_path: Path) -> tuple[nn.Module, Mapping]:
    checkpoint = torch.load(decoder_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    in_dim = int(state_dict["0.weight"].shape[1])
    hidden_dim = int(state_dict["0.weight"].shape[0])
    out_dim = int(state_dict["8.weight"].shape[0])
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.0),
        nn.Linear(hidden_dim, out_dim),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def build_suite() -> Dict:
    return build_clean_behavior_suite(
        patterns=ALL_CLEAN_PATTERNS,
        support_per_class=32,
        heldout_per_class=64,
        seed=20260609,
    )


def sequence_tensor(sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.float32)


def behavior_margin(weights: torch.Tensor, pattern: str, suite: Mapping) -> float:
    model = SubjectNetwork.from_weights(weights.detach().cpu().float())
    model.eval()
    positive = sequence_tensor(suite["heldout"][pattern]["positive"])
    negative = sequence_tensor(suite["heldout"][pattern]["negative"])
    with torch.no_grad():
        pos = torch.sigmoid(model(positive)).mean()
        neg = torch.sigmoid(model(negative)).mean()
    return float((pos - neg).item())


def subject_margin(model: SubjectNetwork, pattern: str, suite: Mapping) -> float:
    positive = sequence_tensor(suite["heldout"][pattern]["positive"])
    negative = sequence_tensor(suite["heldout"][pattern]["negative"])
    model.eval()
    with torch.no_grad():
        pos = torch.sigmoid(model(positive)).mean()
        neg = torch.sigmoid(model(negative)).mean()
    return float((pos - neg).item())


def sample_training_cases(
    pattern: str,
    seed: int,
    max_train_positives: int,
    negative_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    predicate = PREDICATES[pattern]
    universe = enumerate_sequence_universe(seq_len=5, base=10)
    positives = [seq for seq in universe if predicate(seq)]
    negatives = [seq for seq in universe if not predicate(seq)]
    rng = random.Random(seed)
    sampled_positives = rng.sample(
        positives,
        min(int(max_train_positives), len(positives)),
    )
    sampled_negatives = rng.sample(
        negatives,
        min(int(negative_multiple) * len(sampled_positives), len(negatives)),
    )
    inputs = torch.tensor(sampled_positives + sampled_negatives, dtype=torch.float32)
    labels = torch.tensor(
        [1.0] * len(sampled_positives) + [0.0] * len(sampled_negatives),
        dtype=torch.float32,
    )
    order = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(seed))
    return inputs[order], labels[order], len(sampled_positives), len(sampled_negatives)


def train_subject(
    pattern: str,
    seed: int,
    epochs: int,
    lr: float,
    max_train_positives: int,
    negative_multiple: int,
) -> tuple[SubjectNetwork, Dict]:
    torch.manual_seed(seed)
    model = SubjectNetwork()
    inputs, labels, n_positive, n_negative = sample_training_cases(
        pattern,
        seed,
        max_train_positives=max_train_positives,
        negative_multiple=negative_multiple,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    return model, {
        "train_positive_count": int(n_positive),
        "train_negative_count": int(n_negative),
        "final_train_loss": float(loss.item()),
    }


def generate_holdout(args: argparse.Namespace, suite: Mapping, holdout_dir: Path) -> Dict:
    holdout_dir.mkdir(parents=True, exist_ok=True)
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    subjects: List[Dict] = []
    failures: List[Dict] = []
    for pattern_index, pattern in enumerate(PATTERNS):
        accepted = 0
        attempts = 0
        while accepted < args.n_per_behavior:
            seed = args.base_seed + pattern_index * 10000 + attempts
            attempts += 1
            model, train_info = train_subject(
                pattern,
                seed,
                args.train_epochs,
                args.lr,
                args.max_train_positives,
                args.negative_multiple,
            )
            heldout_margin = subject_margin(model, pattern, suite)
            if heldout_margin < args.source_margin_gate:
                failures.append({
                    "pattern": pattern,
                    "seed": int(seed),
                    "heldout_margin": heldout_margin,
                })
                if attempts > args.n_per_behavior * 30:
                    raise RuntimeError(f"Too many rejected subjects for {pattern}")
                continue
            weights = model.to_flat().detach().cpu().float()
            signature = extract_signature_with_stored_probes(weights, probe_examples)
            subjects.append({
                "subject_id": f"fresh_additional:{pattern}:{accepted}:seed:{seed}",
                "target_pattern": pattern,
                "seed": int(seed),
                "heldout_margin": heldout_margin,
                "train_positive_count": train_info["train_positive_count"],
                "train_negative_count": train_info["train_negative_count"],
                "final_train_loss": train_info["final_train_loss"],
                "weights": [float(value) for value in weights.tolist()],
                "weights_hash": stable_hash_json([float(value) for value in weights.tolist()]),
                "signature": [float(value) for value in signature.tolist()],
                "signature_hash": signature_hash_stable_float_list(signature.tolist()),
            })
            accepted += 1
    subjects_path = holdout_dir / "subjects.json"
    subjects_path.write_text(json.dumps({"subjects": subjects}, indent=2, sort_keys=True))
    subjects_sha = sha256_file(subjects_path)
    summary = {
        "subjects_path": str(subjects_path),
        "subjects_sha256": subjects_sha,
        "n_subjects": len(subjects),
        "n_per_behavior_requested": int(args.n_per_behavior),
        "patterns": list(PATTERNS),
        "seed": int(args.base_seed),
        "train_epochs": int(args.train_epochs),
        "lr": float(args.lr),
        "source_margin_gate": float(args.source_margin_gate),
        "max_train_positives": int(args.max_train_positives),
        "negative_multiple": int(args.negative_multiple),
        "probe_set_id": "stored_digit_probe_v1_seed_20260610_n256",
        "probe_examples_hash": stable_hash_json(probe_examples),
        "signature_hash_algorithm": "stable_hash_json_float_list_v1",
        "behavior_suite_metadata": suite["metadata"],
        "counts": count_by_pattern(subjects),
        "source_margin_summary": summarize_subjects_by_pattern(
            subjects,
            "heldout_margin",
        ),
        "failures": failures,
        "passed": len(subjects) == args.n_per_behavior * len(PATTERNS),
    }
    (holdout_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_or_generate_holdout(args: argparse.Namespace, suite: Mapping) -> tuple[Dict, Dict]:
    holdout_dir = Path(args.holdout_dir)
    subjects_path = holdout_dir / "subjects.json"
    summary_path = holdout_dir / "summary.json"
    if (
        subjects_path.exists()
        and summary_path.exists()
        and not args.force_regenerate_holdout
    ):
        summary = json.loads(summary_path.read_text())
        summary.setdefault("subjects_sha256", sha256_file(subjects_path))
        return json.loads(subjects_path.read_text()), summary
    summary = generate_holdout(args, suite, holdout_dir)
    return json.loads(subjects_path.read_text()), summary


def decode_from_signature(
    decoder: nn.Module,
    checkpoint: Mapping,
    signature: torch.Tensor,
) -> torch.Tensor:
    condition = (signature.float() - checkpoint["sig_mean"]) / checkpoint["sig_std"]
    with torch.no_grad():
        weights_norm = decoder(condition.unsqueeze(0)).squeeze(0)
    return weights_norm * checkpoint["weight_std"] + checkpoint["weight_mean"]


def evaluate(args: argparse.Namespace, subjects_payload: Mapping, summary: Mapping) -> Dict:
    suite = build_suite()
    decoder, checkpoint = load_decoder(Path(args.decoder_path))
    generator = torch.Generator().manual_seed(args.eval_seed)
    records = []
    for subject in subjects_payload["subjects"]:
        target = subject["target_pattern"]
        signature = torch.tensor(subject["signature"], dtype=torch.float32)
        matched_weights = decode_from_signature(decoder, checkpoint, signature)
        noise_margins = []
        for _ in range(args.noise_controls):
            noise_condition = torch.randn(
                checkpoint["sig_mean"].shape,
                generator=generator,
                dtype=torch.float32,
            )
            with torch.no_grad():
                noise_weights_norm = decoder(noise_condition.unsqueeze(0)).squeeze(0)
            noise_weights = noise_weights_norm * checkpoint["weight_std"] + checkpoint["weight_mean"]
            noise_margins.append(behavior_margin(noise_weights, target, suite))
        matched_margin = behavior_margin(matched_weights, target, suite)
        worst_noise_margin = max(noise_margins)
        deltas = {
            "matched_minus_worst_noise_target_margin": matched_margin - worst_noise_margin,
        }
        records.append({
            "subject_id": subject["subject_id"],
            "target_pattern": target,
            "source_heldout_margin": float(subject["heldout_margin"]),
            "margins": {
                "matched": matched_margin,
                "worst_noise": worst_noise_margin,
                "mean_noise": safe_mean(noise_margins),
            },
            "deltas": deltas,
            "noise_control_count": int(args.noise_controls),
        })
    aggregate = summarize_records(records)
    by_behavior = {
        pattern: summarize_records([
            record for record in records
            if record["target_pattern"] == pattern
        ])
        for pattern in PATTERNS
    }
    individual_audit = individual_gate_audit(records)
    failures = gate_failures(aggregate, by_behavior, individual_audit)
    return {
        "aggregate": aggregate,
        "behavior_suite_metadata": suite["metadata"],
        "by_behavior": by_behavior,
        "caveats": [
            "Feasibility result only, not a preregistered final proof.",
            "Uses locked decoder trained for the current stored-probe setup.",
            "No decoder or method training was performed on these fresh additional-behavior subjects.",
            "Tests additional behaviors outside the two-behavior decode/steering proof.",
            "Noise controls are worst-of-K normalized-signature noise controls.",
            "Source gate was lowered to 0.20 for this bounded feasibility run.",
        ],
        "claim_scope": "fresh_subject_additional_behavior_decode_feasibility_not_proof",
        "decoder_path": args.decoder_path,
        "development_status": "feasibility_additional_behavior_no_final_claim",
        "failures": failures,
        "holdout_subjects_path": summary["subjects_path"],
        "holdout_subjects_sha256": summary["subjects_sha256"],
        "holdout_summary": summary,
        "individual_gate_audit": individual_audit,
        "noise_control_count": int(args.noise_controls),
        "passed": not failures,
        "records": records,
        "source_margin_summary": summarize_records_by_key(
            records,
            "target_pattern",
            "source_heldout_margin",
        ),
        "thresholds": THRESHOLDS,
    }


def summarize_records(records: Sequence[Mapping]) -> Dict:
    return {
        "n": len(records),
        "mean_matched_target_margin": safe_mean([
            record["margins"]["matched"] for record in records
        ]),
        "mean_worst_noise_target_margin": safe_mean([
            record["margins"]["worst_noise"] for record in records
        ]),
        "mean_matched_minus_worst_noise_target_margin": safe_mean([
            record["deltas"]["matched_minus_worst_noise_target_margin"]
            for record in records
        ]),
    }


def individual_gate_audit(records: Sequence[Mapping]) -> Dict:
    failed_records = []
    for record in records:
        failed = []
        if record["margins"]["matched"] < THRESHOLDS["min_mean_matched_target_margin"]:
            failed.append({
                "check": "matched_target_margin",
                "value": record["margins"]["matched"],
                "threshold": THRESHOLDS["min_mean_matched_target_margin"],
                "operator": ">=",
            })
        if (
            record["deltas"]["matched_minus_worst_noise_target_margin"]
            < THRESHOLDS["min_mean_matched_minus_noise_target_margin"]
        ):
            failed.append({
                "check": "matched_minus_worst_noise_target_margin",
                "value": record["deltas"]["matched_minus_worst_noise_target_margin"],
                "threshold": THRESHOLDS["min_mean_matched_minus_noise_target_margin"],
                "operator": ">=",
            })
        if failed:
            failed_records.append({
                "subject_id": record["subject_id"],
                "target_pattern": record["target_pattern"],
                "failed": failed,
            })
    n = len(records)
    by_behavior = {}
    for pattern in PATTERNS:
        behavior_records = [
            record for record in records
            if record["target_pattern"] == pattern
        ]
        behavior_failures = [
            failure for failure in failed_records
            if failure["target_pattern"] == pattern
        ]
        by_behavior[pattern] = {
            "n": len(behavior_records),
            "all_gate_pass_count": len(behavior_records) - len(behavior_failures),
            "all_gate_pass_rate": (
                (len(behavior_records) - len(behavior_failures)) / len(behavior_records)
                if behavior_records else 0.0
            ),
        }
    return {
        "n": n,
        "all_gate_pass_count": n - len(failed_records),
        "all_gate_pass_rate": (n - len(failed_records)) / n if n else 0.0,
        "by_behavior": by_behavior,
        "failed_records": failed_records,
    }


def gate_failures(
    aggregate: Mapping,
    by_behavior: Mapping[str, Mapping],
    individual_audit: Mapping,
) -> List[str]:
    failures = []
    mean_checks = [
        (
            "mean_matched_target_margin",
            THRESHOLDS["min_mean_matched_target_margin"],
        ),
        (
            "mean_matched_minus_worst_noise_target_margin",
            THRESHOLDS["min_mean_matched_minus_noise_target_margin"],
        ),
    ]
    for metric_name, threshold in mean_checks:
        if aggregate[metric_name] < threshold:
            failures.append(f"aggregate {metric_name} failed: {aggregate[metric_name]} < {threshold}")
        for behavior, summary in by_behavior.items():
            if summary[metric_name] < threshold:
                failures.append(f"{behavior} {metric_name} failed: {summary[metric_name]} < {threshold}")
    if individual_audit["all_gate_pass_rate"] < THRESHOLDS["min_individual_pass_rate"]:
        failures.append(
            "individual pass rate failed: "
            f"{individual_audit['all_gate_pass_rate']} < {THRESHOLDS['min_individual_pass_rate']}"
        )
    for behavior, audit in individual_audit["by_behavior"].items():
        if audit["all_gate_pass_rate"] < THRESHOLDS["min_per_behavior_individual_pass_rate"]:
            failures.append(
                f"{behavior} individual pass rate failed: "
                f"{audit['all_gate_pass_rate']} < "
                f"{THRESHOLDS['min_per_behavior_individual_pass_rate']}"
            )
    return failures


def count_by_pattern(subjects: Sequence[Mapping]) -> Dict[str, int]:
    counts = {pattern: 0 for pattern in PATTERNS}
    for subject in subjects:
        counts[subject["target_pattern"]] += 1
    return counts


def summarize_subjects_by_pattern(subjects: Sequence[Mapping], value_key: str) -> Dict[str, Dict]:
    return summarize_records_by_key(subjects, "target_pattern", value_key)


def summarize_records_by_key(
    records: Sequence[Mapping],
    group_key: str,
    value_key: str,
) -> Dict[str, Dict]:
    summary = {}
    for pattern in PATTERNS:
        values = [
            float(record[value_key])
            for record in records
            if record[group_key] == pattern
        ]
        summary[pattern] = {
            "n": len(values),
            "min": min(values) if values else None,
            "mean": safe_mean(values),
            "max": max(values) if values else None,
        }
    return summary


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    suite = build_suite()
    subjects_payload, summary = load_or_generate_holdout(args, suite)
    result = evaluate(args, subjects_payload, summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "passed": result["passed"],
        "failures": result["failures"],
        "results_path": str(result_path),
        "holdout_subjects_sha256": result["holdout_subjects_sha256"],
        "aggregate": result["aggregate"],
        "individual_gate_audit": result["individual_gate_audit"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
