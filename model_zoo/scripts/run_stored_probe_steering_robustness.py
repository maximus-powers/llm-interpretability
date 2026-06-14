"""Run a stricter stored-probe signature steering robustness evaluation.

This script is intentionally self-contained for auditability. It can generate a
fresh same-architecture holdout of small subject networks, then evaluates locked
learned edit vectors through the locked stored-probe decoder with stronger
controls:

- no edit;
- reverse edit direction;
- noise signature;
- worst-of-K norm-matched random edit vectors;
- individual pass-rate gates in addition to aggregate/per-target mean gates.
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


PATTERNS = ("sorted_ascending", "sorted_descending")
OPPOSITE = {
    "sorted_ascending": "sorted_descending",
    "sorted_descending": "sorted_ascending",
}
THRESHOLDS = {
    "min_mean_steered_minus_no_edit_target_margin": 0.20,
    "min_mean_steered_minus_reverse_direction_target_margin": 0.20,
    "min_mean_steered_minus_noise_target_margin": 0.20,
    "min_mean_steered_minus_worst_random_norm_matched_target_margin": 0.20,
    "min_mean_steered_target_margin": 0.20,
    "max_mean_steered_source_margin_change": -0.05,
    "min_individual_all_gate_pass_rate": 0.95,
    "min_per_target_individual_all_gate_pass_rate": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/stored_probe_signature_edit_vectors_v1_robust_external_eval",
    )
    parser.add_argument(
        "--holdout-dir",
        default="runs/fresh_external_steering_holdout_v2_robust",
    )
    parser.add_argument("--decoder-path", default="runs/stored_probe_functional_decoder_v2_adaptive/model.pt")
    parser.add_argument(
        "--edit-vectors-path",
        default="runs/stored_probe_signature_edit_vectors_v1_development/edit_vectors.pt",
    )
    parser.add_argument("--n-per-behavior", type=int, default=24)
    parser.add_argument("--base-seed", type=int, default=20260630)
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--random-controls", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=20260631)
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
        patterns=PATTERNS,
        support_per_class=32,
        heldout_per_class=64,
        seed=20260609,
    )


def sequence_tensor(sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.float32)


def behavior_margin(
    weights: torch.Tensor,
    pattern: str,
    suite: Mapping,
) -> float:
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


def sample_training_cases(pattern: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    predicate = PREDICATES[pattern]
    universe = enumerate_sequence_universe(seq_len=5, base=10)
    positives = [seq for seq in universe if predicate(seq)]
    negatives = [seq for seq in universe if not predicate(seq)]
    rng = random.Random(seed)
    sampled_negatives = rng.sample(negatives, 4 * len(positives))
    inputs = torch.tensor(positives + sampled_negatives, dtype=torch.float32)
    labels = torch.tensor(
        [1.0] * len(positives) + [0.0] * len(sampled_negatives),
        dtype=torch.float32,
    )
    order = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(seed))
    return inputs[order], labels[order], len(positives), len(sampled_negatives)


def train_subject(pattern: str, seed: int, epochs: int, lr: float) -> tuple[SubjectNetwork, Dict]:
    torch.manual_seed(seed)
    model = SubjectNetwork()
    inputs, labels, n_positive, n_negative = sample_training_cases(pattern, seed)
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
            model, train_info = train_subject(pattern, seed, args.train_epochs, args.lr)
            heldout_margin = subject_margin(model, pattern, suite)
            opposite_margin = subject_margin(model, OPPOSITE[pattern], suite)
            if heldout_margin < args.source_margin_gate:
                failures.append({
                    "pattern": pattern,
                    "seed": int(seed),
                    "heldout_margin": heldout_margin,
                    "opposite_heldout_margin": opposite_margin,
                })
                if attempts > args.n_per_behavior * 20:
                    raise RuntimeError(f"Too many rejected subjects for {pattern}")
                continue
            weights = model.to_flat().detach().cpu().float()
            signature = extract_signature_with_stored_probes(weights, probe_examples)
            subjects.append({
                "subject_id": f"fresh2:{pattern}:{accepted}:seed:{seed}",
                "target_pattern": pattern,
                "seed": int(seed),
                "heldout_margin": heldout_margin,
                "opposite_heldout_margin": opposite_margin,
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
        "probe_set_id": "stored_digit_probe_v1_seed_20260610_n256",
        "probe_examples_hash": stable_hash_json(probe_examples),
        "signature_hash_algorithm": "stable_hash_json_float_list_v1",
        "behavior_suite_metadata": suite["metadata"],
        "counts": count_by_pattern(subjects),
        "heldout_margin_summary": summarize_by_pattern(
            subjects,
            "target_pattern",
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
        return (
            json.loads(subjects_path.read_text()),
            summary,
        )
    summary = generate_holdout(args, suite, holdout_dir)
    return (
        json.loads(subjects_path.read_text()),
        summary,
    )


def decode_from_normalized_condition(
    decoder: nn.Module,
    checkpoint: Mapping,
    condition_norm: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        weights_norm = decoder(condition_norm.float().unsqueeze(0)).squeeze(0)
    return weights_norm * checkpoint["weight_std"] + checkpoint["weight_mean"]


def evaluate(args: argparse.Namespace, subjects_payload: Mapping, holdout_summary: Mapping) -> Dict:
    suite = build_suite()
    decoder, decoder_checkpoint = load_decoder(Path(args.decoder_path))
    edit_payload = torch.load(args.edit_vectors_path, map_location="cpu")
    edit_vectors = edit_payload["edit_vectors"]
    generator = torch.Generator().manual_seed(args.eval_seed)
    records = []
    for subject in subjects_payload["subjects"]:
        source = subject["target_pattern"]
        target = OPPOSITE[source]
        edit_key = f"{source}_to_{target}"
        edit_vector = edit_vectors[edit_key].float()
        signature = torch.tensor(subject["signature"], dtype=torch.float32)
        condition = (signature - decoder_checkpoint["sig_mean"]) / decoder_checkpoint["sig_std"]
        no_edit_weights = decode_from_normalized_condition(decoder, decoder_checkpoint, condition)
        steered_weights = decode_from_normalized_condition(
            decoder,
            decoder_checkpoint,
            condition + edit_vector,
        )
        reverse_weights = decode_from_normalized_condition(
            decoder,
            decoder_checkpoint,
            condition - edit_vector,
        )
        noise_condition = torch.randn(
            condition.shape,
            generator=generator,
            dtype=condition.dtype,
        )
        noise_weights = decode_from_normalized_condition(
            decoder,
            decoder_checkpoint,
            noise_condition,
        )
        random_target_margins = []
        random_source_margins = []
        random_norms = []
        for _ in range(args.random_controls):
            random_vector = torch.randn(
                edit_vector.shape,
                generator=generator,
                dtype=edit_vector.dtype,
            )
            random_vector = random_vector / random_vector.norm().clamp_min(1e-12)
            random_vector = random_vector * edit_vector.norm()
            random_weights = decode_from_normalized_condition(
                decoder,
                decoder_checkpoint,
                condition + random_vector,
            )
            random_target_margins.append(behavior_margin(random_weights, target, suite))
            random_source_margins.append(behavior_margin(random_weights, source, suite))
            random_norms.append(float(random_vector.norm().item()))
        margins = {
            "source": {
                "no_edit": behavior_margin(no_edit_weights, source, suite),
                "steered": behavior_margin(steered_weights, source, suite),
                "reverse_direction": behavior_margin(reverse_weights, source, suite),
                "noise_signature": behavior_margin(noise_weights, source, suite),
                "worst_random_norm_matched": max(random_source_margins),
                "mean_random_norm_matched": safe_mean(random_source_margins),
            },
            "target": {
                "no_edit": behavior_margin(no_edit_weights, target, suite),
                "steered": behavior_margin(steered_weights, target, suite),
                "reverse_direction": behavior_margin(reverse_weights, target, suite),
                "noise_signature": behavior_margin(noise_weights, target, suite),
                "worst_random_norm_matched": max(random_target_margins),
                "mean_random_norm_matched": safe_mean(random_target_margins),
            },
        }
        deltas = {
            "steered_minus_no_edit_target_margin": (
                margins["target"]["steered"] - margins["target"]["no_edit"]
            ),
            "steered_minus_reverse_direction_target_margin": (
                margins["target"]["steered"] - margins["target"]["reverse_direction"]
            ),
            "steered_minus_noise_target_margin": (
                margins["target"]["steered"] - margins["target"]["noise_signature"]
            ),
            "steered_minus_worst_random_norm_matched_target_margin": (
                margins["target"]["steered"] - margins["target"]["worst_random_norm_matched"]
            ),
            "steered_source_margin_change": (
                margins["source"]["steered"] - margins["source"]["no_edit"]
            ),
        }
        records.append({
            "subject_id": subject["subject_id"],
            "source_pattern": source,
            "target_pattern": target,
            "source_heldout_margin": float(subject["heldout_margin"]),
            "source_opposite_heldout_margin": float(subject["opposite_heldout_margin"]),
            "edit_vector_norm": float(edit_vector.norm().item()),
            "random_control_count": int(args.random_controls),
            "random_norm_matched_vector_norm_min": min(random_norms),
            "random_norm_matched_vector_norm_max": max(random_norms),
            "margins": margins,
            "deltas": deltas,
        })

    aggregate = summarize_records(records)
    by_target = {
        target: summarize_records([record for record in records if record["target_pattern"] == target])
        for target in PATTERNS
    }
    individual_audit = individual_gate_audit(records)
    failed_subject_ids = {
        failure["subject_id"]
        for failure in individual_audit["failed_records"]
    }
    for record in records:
        record["individual_all_gates_passed"] = (
            record["subject_id"] not in failed_subject_ids
        )
    failures = gate_failures(aggregate, by_target, individual_audit)
    return {
        "aggregate": aggregate,
        "behavior_suite_metadata": suite["metadata"],
        "by_target": by_target,
        "caveats": [
            "Fresh external same-architecture subject networks, not HF/checkpoint-distribution rows.",
            "Scope is only sorted_ascending <-> sorted_descending steering.",
            "Locked decoder and locked learned edit vectors; no training or tuning on this evaluation result.",
            "Edit vectors are applied in normalized stored-probe signature coordinates.",
            "Worst-random control is worst-of-K per subject, where K is random_control_count.",
            "Does not prove larger models or additional behaviors.",
        ],
        "claim_scope": "robust_fresh_external_mean_and_pass_rate_steering_two_behavior_same_architecture",
        "decoder_path": args.decoder_path,
        "development_status": "fresh_robust_external_evaluation_no_tuning_after_result",
        "edit_vectors_path": args.edit_vectors_path,
        "eval_seed": int(args.eval_seed),
        "failures": failures,
        "holdout_subjects_path": holdout_summary["subjects_path"],
        "holdout_subjects_sha256": holdout_summary["subjects_sha256"],
        "holdout_summary": holdout_summary,
        "individual_gate_audit": individual_audit,
        "passed": not failures,
        "random_control_count": int(args.random_controls),
        "records": records,
        "thresholds": THRESHOLDS,
    }


def summarize_records(records: Sequence[Mapping]) -> Dict:
    return {
        "n": len(records),
        "mean_steered_target_margin": safe_mean([
            record["margins"]["target"]["steered"] for record in records
        ]),
        "mean_no_edit_target_margin": safe_mean([
            record["margins"]["target"]["no_edit"] for record in records
        ]),
        "mean_reverse_direction_target_margin": safe_mean([
            record["margins"]["target"]["reverse_direction"] for record in records
        ]),
        "mean_noise_target_margin": safe_mean([
            record["margins"]["target"]["noise_signature"] for record in records
        ]),
        "mean_worst_random_norm_matched_target_margin": safe_mean([
            record["margins"]["target"]["worst_random_norm_matched"] for record in records
        ]),
        "mean_steered_minus_no_edit_target_margin": safe_mean([
            record["deltas"]["steered_minus_no_edit_target_margin"] for record in records
        ]),
        "mean_steered_minus_reverse_direction_target_margin": safe_mean([
            record["deltas"]["steered_minus_reverse_direction_target_margin"]
            for record in records
        ]),
        "mean_steered_minus_noise_target_margin": safe_mean([
            record["deltas"]["steered_minus_noise_target_margin"] for record in records
        ]),
        "mean_steered_minus_worst_random_norm_matched_target_margin": safe_mean([
            record["deltas"]["steered_minus_worst_random_norm_matched_target_margin"]
            for record in records
        ]),
        "mean_steered_source_margin_change": safe_mean([
            record["deltas"]["steered_source_margin_change"] for record in records
        ]),
    }


def individual_gate_audit(records: Sequence[Mapping]) -> Dict:
    checks = {
        "steered_target_margin": (
            lambda record: record["margins"]["target"]["steered"],
            THRESHOLDS["min_mean_steered_target_margin"],
            ">=",
        ),
        "steered_minus_no_edit_target_margin": (
            lambda record: record["deltas"]["steered_minus_no_edit_target_margin"],
            THRESHOLDS["min_mean_steered_minus_no_edit_target_margin"],
            ">=",
        ),
        "steered_minus_reverse_direction_target_margin": (
            lambda record: record["deltas"]["steered_minus_reverse_direction_target_margin"],
            THRESHOLDS["min_mean_steered_minus_reverse_direction_target_margin"],
            ">=",
        ),
        "steered_minus_noise_target_margin": (
            lambda record: record["deltas"]["steered_minus_noise_target_margin"],
            THRESHOLDS["min_mean_steered_minus_noise_target_margin"],
            ">=",
        ),
        "steered_minus_worst_random_norm_matched_target_margin": (
            lambda record: record["deltas"][
                "steered_minus_worst_random_norm_matched_target_margin"
            ],
            THRESHOLDS[
                "min_mean_steered_minus_worst_random_norm_matched_target_margin"
            ],
            ">=",
        ),
        "steered_source_margin_change": (
            lambda record: record["deltas"]["steered_source_margin_change"],
            THRESHOLDS["max_mean_steered_source_margin_change"],
            "<=",
        ),
    }
    pass_counts = {name: 0 for name in checks}
    failed_records = []
    for record in records:
        failed = []
        for name, (getter, threshold, operator) in checks.items():
            value = float(getter(record))
            ok = value >= threshold if operator == ">=" else value <= threshold
            if ok:
                pass_counts[name] += 1
            else:
                failed.append({
                    "check": name,
                    "value": value,
                    "threshold": threshold,
                    "operator": operator,
                })
        if failed:
            failed_records.append({
                "subject_id": record["subject_id"],
                "source_pattern": record["source_pattern"],
                "target_pattern": record["target_pattern"],
                "failed": failed,
            })
    n = len(records)
    by_target = {}
    for target in PATTERNS:
        target_records = [record for record in records if record["target_pattern"] == target]
        target_failures = [
            failure for failure in failed_records if failure["target_pattern"] == target
        ]
        by_target[target] = {
            "n": len(target_records),
            "all_gate_pass_count": len(target_records) - len(target_failures),
            "all_gate_pass_rate": (
                (len(target_records) - len(target_failures)) / len(target_records)
                if target_records else 0.0
            ),
        }
    return {
        "n": n,
        "all_gate_pass_count": n - len(failed_records),
        "all_gate_pass_rate": (n - len(failed_records)) / n if n else 0.0,
        "by_target": by_target,
        "failed_records": failed_records,
        "per_check_pass_counts": pass_counts,
        "per_check_pass_rates": {
            name: (count / n if n else 0.0)
            for name, count in pass_counts.items()
        },
    }


def gate_failures(
    aggregate: Mapping,
    by_target: Mapping[str, Mapping],
    individual_audit: Mapping,
) -> List[str]:
    failures: List[str] = []
    mean_checks = [
        (
            "mean_steered_minus_no_edit_target_margin",
            THRESHOLDS["min_mean_steered_minus_no_edit_target_margin"],
            ">=",
        ),
        (
            "mean_steered_minus_reverse_direction_target_margin",
            THRESHOLDS["min_mean_steered_minus_reverse_direction_target_margin"],
            ">=",
        ),
        (
            "mean_steered_minus_noise_target_margin",
            THRESHOLDS["min_mean_steered_minus_noise_target_margin"],
            ">=",
        ),
        (
            "mean_steered_minus_worst_random_norm_matched_target_margin",
            THRESHOLDS[
                "min_mean_steered_minus_worst_random_norm_matched_target_margin"
            ],
            ">=",
        ),
        (
            "mean_steered_target_margin",
            THRESHOLDS["min_mean_steered_target_margin"],
            ">=",
        ),
        (
            "mean_steered_source_margin_change",
            THRESHOLDS["max_mean_steered_source_margin_change"],
            "<=",
        ),
    ]
    for name, threshold, operator in mean_checks:
        if not passes(float(aggregate[name]), threshold, operator):
            failures.append(f"aggregate {name} failed: {aggregate[name]} {operator} {threshold}")
        for target, summary in by_target.items():
            if not passes(float(summary[name]), threshold, operator):
                failures.append(f"{target} {name} failed: {summary[name]} {operator} {threshold}")
    pass_rate = individual_audit["all_gate_pass_rate"]
    if pass_rate < THRESHOLDS["min_individual_all_gate_pass_rate"]:
        failures.append(
            "individual all-gate pass rate failed: "
            f"{pass_rate} < {THRESHOLDS['min_individual_all_gate_pass_rate']} required"
        )
    for target, audit in individual_audit["by_target"].items():
        target_pass_rate = audit["all_gate_pass_rate"]
        if target_pass_rate < THRESHOLDS["min_per_target_individual_all_gate_pass_rate"]:
            failures.append(
                f"{target} individual all-gate pass rate failed: "
                f"{target_pass_rate} < "
                f"{THRESHOLDS['min_per_target_individual_all_gate_pass_rate']} required"
            )
    return failures


def passes(value: float, threshold: float, operator: str) -> bool:
    return value >= threshold if operator == ">=" else value <= threshold


def count_by_pattern(subjects: Sequence[Mapping]) -> Dict[str, int]:
    counts = {pattern: 0 for pattern in PATTERNS}
    for subject in subjects:
        counts[subject["target_pattern"]] += 1
    return counts


def summarize_by_pattern(
    subjects: Sequence[Mapping],
    pattern_key: str,
    value_key: str,
) -> Dict[str, Dict]:
    result = {}
    for pattern in PATTERNS:
        values = [float(subject[value_key]) for subject in subjects if subject[pattern_key] == pattern]
        result[pattern] = {
            "n": len(values),
            "min": min(values) if values else None,
            "mean": safe_mean(values),
            "max": max(values) if values else None,
        }
    return result


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
    subjects_payload, holdout_summary = load_or_generate_holdout(args, suite)
    result = evaluate(args, subjects_payload, holdout_summary)
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
        "individual_gate_audit": {
            "all_gate_pass_count": result["individual_gate_audit"]["all_gate_pass_count"],
            "all_gate_pass_rate": result["individual_gate_audit"]["all_gate_pass_rate"],
            "by_target": result["individual_gate_audit"]["by_target"],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
