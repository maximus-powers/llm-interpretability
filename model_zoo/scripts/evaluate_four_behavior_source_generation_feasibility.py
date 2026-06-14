"""Evaluate source-subject generation feasibility for the four-behavior protocol.

This does not train a decoder. It checks whether a source-training protocol can
produce subjects that clear the preregistered heldout-margin source gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
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


PATTERNS = (
    "sorted_ascending",
    "sorted_descending",
    "has_majority",
    "mountain_pattern",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/four_behavior_source_generation_feasibility_v1",
    )
    parser.add_argument("--n-per-behavior", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20261010)
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=32)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument(
        "--training-mode",
        choices=("support_only", "heldout_excluded_full_pool"),
        default="support_only",
    )
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    parser.add_argument(
        "--collection-mode",
        choices=("fixed_n", "accept_reject"),
        default="fixed_n",
    )
    parser.add_argument("--target-accepted-per-behavior", type=int, default=8)
    parser.add_argument("--max-attempts-per-behavior", type=int, default=32)
    parser.add_argument(
        "--claim-scope",
        default="source_generation_feasibility_only_not_decoder_evidence",
    )
    parser.add_argument(
        "--development-status",
        default="feasibility_check_before_decoder_training",
    )
    return parser.parse_args()


def build_suite(support_per_class: int, heldout_per_class: int) -> Dict:
    return build_clean_behavior_suite(
        patterns=PATTERNS,
        support_per_class=support_per_class,
        heldout_per_class=heldout_per_class,
        seed=20260609,
    )


def sequence_tensor(sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.float32)


def train_subject(
    pattern: str,
    seed: int,
    suite: Mapping,
    epochs: int,
    lr: float,
    args: argparse.Namespace,
    candidate_pools: Mapping[str, Mapping[str, List[tuple[int, ...]]]],
    heldout_sequences: set[tuple[int, ...]],
) -> tuple[SubjectNetwork, Dict]:
    torch.manual_seed(seed)
    model = SubjectNetwork()
    train_payload = build_training_payload(
        pattern=pattern,
        seed=seed,
        suite=suite,
        args=args,
        candidate_pools=candidate_pools,
        heldout_sequences=heldout_sequences,
    )
    inputs = sequence_tensor(train_payload["selected_positive"] + train_payload["selected_negative"])
    labels = torch.tensor(
        [1.0] * len(train_payload["selected_positive"])
        + [0.0] * len(train_payload["selected_negative"]),
        dtype=torch.float32,
    )
    order = torch.randperm(
        len(inputs),
        generator=torch.Generator().manual_seed(seed),
    )
    inputs = inputs[order]
    labels = labels[order]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(inputs), labels)
        loss.backward()
        optimizer.step()
    train_info = {
        "final_train_loss": float(loss.item()),
        "training_mode": args.training_mode,
        **train_payload["metadata"],
    }
    return model, train_info


def build_training_payload(
    pattern: str,
    seed: int,
    suite: Mapping,
    args: argparse.Namespace,
    candidate_pools: Mapping[str, Mapping[str, List[tuple[int, ...]]]],
    heldout_sequences: set[tuple[int, ...]],
) -> Dict:
    if args.training_mode == "support_only":
        selected_positive = [
            tuple(seq) for seq in suite["support"][pattern]["positive"]
        ]
        selected_hard_negative: List[tuple[int, ...]] = []
        selected_generic_negative = [
            tuple(seq) for seq in suite["support"][pattern]["negative"]
        ]
    else:
        pools = candidate_pools[pattern]
        selected_positive = sample_cases(
            pools["positive"],
            args.positive_cap,
            seed,
            pattern,
            "positive",
        )
        selected_hard_negative = sample_cases(
            pools["hard_negative"],
            args.hard_negative_cap,
            seed,
            pattern,
            "hard_negative",
        )
        selected_generic_negative = sample_cases(
            pools["generic_negative"],
            args.generic_negative_cap,
            seed,
            pattern,
            "generic_negative",
        )

    selected_negative = selected_hard_negative + selected_generic_negative
    selected_combined = selected_positive + selected_negative
    overlap_count = len(set(selected_combined) & heldout_sequences)
    metadata = {
        "selected_positive_count": len(selected_positive),
        "selected_hard_negative_count": len(selected_hard_negative),
        "selected_generic_negative_count": len(selected_generic_negative),
        "selected_negative_count": len(selected_negative),
        "selected_total_count": len(selected_combined),
        "selected_positive_hash": hash_sequences(selected_positive),
        "selected_hard_negative_hash": hash_sequences(selected_hard_negative),
        "selected_generic_negative_hash": hash_sequences(selected_generic_negative),
        "selected_train_cases_hash": hash_sequences(selected_combined),
        "selected_train_vs_heldout_overlap_count": int(overlap_count),
    }
    return {
        "metadata": metadata,
        "selected_negative": [list(seq) for seq in selected_negative],
        "selected_positive": [list(seq) for seq in selected_positive],
    }


def sample_cases(
    candidates: Sequence[tuple[int, ...]],
    cap: int,
    seed: int,
    pattern: str,
    category: str,
) -> List[tuple[int, ...]]:
    candidates = list(candidates)
    rng = __import__("random").Random(f"{seed}:{pattern}:{category}")
    rng.shuffle(candidates)
    return candidates[: min(int(cap), len(candidates))]


def build_heldout_sequences(suite: Mapping) -> set[tuple[int, ...]]:
    return {
        tuple(seq)
        for pattern in PATTERNS
        for split_class in ("positive", "negative")
        for seq in suite["heldout"][pattern][split_class]
    }


def build_candidate_pools(
    heldout_sequences: set[tuple[int, ...]],
) -> Dict[str, Dict[str, List[tuple[int, ...]]]]:
    universe = enumerate_sequence_universe(seq_len=5, base=10)
    pools: Dict[str, Dict[str, List[tuple[int, ...]]]] = {}
    for pattern in PATTERNS:
        predicate = PREDICATES[pattern]
        pools[pattern] = {
            "positive": [
                seq for seq in universe
                if seq not in heldout_sequences and predicate(seq)
            ],
            "hard_negative": [
                seq for seq in universe
                if seq not in heldout_sequences
                and not predicate(seq)
                and any(
                    PREDICATES[other_pattern](seq)
                    for other_pattern in PATTERNS
                    if other_pattern != pattern
                )
            ],
            "generic_negative": [
                seq for seq in universe
                if seq not in heldout_sequences
                and not any(PREDICATES[other_pattern](seq) for other_pattern in PATTERNS)
            ],
        }
    return pools


def summarize_candidate_pools(
    candidate_pools: Mapping[str, Mapping[str, Sequence[tuple[int, ...]]]],
) -> Dict[str, Dict]:
    return {
        pattern: {
            category: {
                "count": len(cases),
                "hash": hash_sequences(cases),
            }
            for category, cases in pools.items()
        }
        for pattern, pools in candidate_pools.items()
    }


def hash_sequences(sequences: Sequence[Sequence[int]]) -> str:
    return stable_hash_json([
        [int(value) for value in seq]
        for seq in sequences
    ])


def subject_margin(model: SubjectNetwork, pattern: str, split: str, suite: Mapping) -> float:
    positive = sequence_tensor(suite[split][pattern]["positive"])
    negative = sequence_tensor(suite[split][pattern]["negative"])
    model.eval()
    with torch.no_grad():
        pos = torch.sigmoid(model(positive)).mean()
        neg = torch.sigmoid(model(negative)).mean()
    return float((pos - neg).item())


def evaluate(args: argparse.Namespace) -> Dict:
    suite = build_suite(
        support_per_class=args.support_per_class,
        heldout_per_class=args.heldout_per_class,
    )
    heldout_sequences = build_heldout_sequences(suite)
    candidate_pools = build_candidate_pools(heldout_sequences)
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    records: List[Dict] = []
    for pattern_index, pattern in enumerate(PATTERNS):
        if args.collection_mode == "fixed_n":
            max_attempts = args.n_per_behavior
            target_accepted = args.n_per_behavior
        else:
            max_attempts = args.max_attempts_per_behavior
            target_accepted = args.target_accepted_per_behavior
        accepted_count = 0
        for attempt_index in range(max_attempts):
            if args.collection_mode == "accept_reject" and accepted_count >= target_accepted:
                break
            seed = args.base_seed + pattern_index * 10000 + attempt_index
            record = evaluate_attempt(
                pattern=pattern,
                pattern_index=pattern_index,
                attempt_index=attempt_index,
                seed=seed,
                suite=suite,
                args=args,
                candidate_pools=candidate_pools,
                heldout_sequences=heldout_sequences,
                probe_examples=probe_examples,
            )
            if record["accepted"]:
                accepted_count += 1
                record["accepted_index"] = accepted_count - 1
            records.append(record)
    by_behavior = summarize_by_behavior(records, args.source_margin_gate)
    aggregate = summarize_records(records, args.source_margin_gate)
    max_overlap_count = max(
        (
            int(record["train_info"]["selected_train_vs_heldout_overlap_count"])
            for record in records
        ),
        default=0,
    )
    failures = []
    target_count = (
        args.n_per_behavior
        if args.collection_mode == "fixed_n"
        else args.target_accepted_per_behavior
    )
    for pattern, summary in by_behavior.items():
        if summary["pass_count"] < target_count:
            failures.append(
                f"{pattern} source gate pass count "
                f"{summary['pass_count']}/{target_count}"
            )
    if max_overlap_count != 0:
        failures.append(
            "selected train/heldout overlap failed: "
            f"max overlap count {max_overlap_count}"
        )
    caveats = [
        "Pilot feasibility sample; n=8 per behavior is not an impossibility result.",
        "Tests source generation, not stored-probe decoding.",
        "A revised source-generation protocol would require a new preregistration before proof use.",
    ]
    if args.collection_mode == "accept_reject":
        interpretation = (
            "Under this heldout-excluded full-pool source-generation protocol "
            "with deterministic accept-reject collection, the pilot result is "
            "determined by accepted source counts, heldout source-margin gates, "
            "and selected-training-vs-heldout overlap gates."
        )
    elif args.training_mode == "support_only":
        interpretation = (
            "Under this support-only source-generation protocol, has_majority fails "
            "the preregistered heldout source-margin gate in this pilot."
        )
    else:
        interpretation = (
            "Under this heldout-excluded full-pool source-generation protocol, "
            "has_majority remains the limiting behavior for the preregistered "
            "heldout source-margin gate in this pilot."
        )
    return {
        "aggregate": aggregate,
        "behavior_suite_metadata": suite["metadata"],
        "by_behavior": by_behavior,
        "caveats": caveats,
        "claim_scope": args.claim_scope,
        "config": {
            "base_seed": int(args.base_seed),
            "collection_mode": args.collection_mode,
            "generic_negative_cap": int(args.generic_negative_cap),
            "hard_negative_cap": int(args.hard_negative_cap),
            "heldout_per_class": int(args.heldout_per_class),
            "lr": float(args.lr),
            "max_attempts_per_behavior": int(args.max_attempts_per_behavior),
            "n_per_behavior": int(args.n_per_behavior),
            "positive_cap": int(args.positive_cap),
            "patterns": list(PATTERNS),
            "probe_examples_hash": stable_hash_json(probe_examples),
            "probe_set_id": "stored_digit_probe_v1_seed_20260610_n256",
            "source_margin_gate": float(args.source_margin_gate),
            "support_per_class": int(args.support_per_class),
            "target_accepted_per_behavior": int(args.target_accepted_per_behavior),
            "train_epochs": int(args.train_epochs),
            "training_mode": args.training_mode,
        },
        "global_heldout_exclusion_count": len(heldout_sequences),
        "max_selected_train_vs_heldout_overlap_count": int(max_overlap_count),
        "candidate_pool_summary": summarize_candidate_pools(candidate_pools),
        "development_status": args.development_status,
        "failures": failures,
        "interpretation": build_interpretation(
            args=args,
            by_behavior=by_behavior,
            failures=failures,
            interpretation=interpretation,
        ),
        "passed": not failures,
        "acceptance_summary": summarize_acceptance(records),
        "records": records,
    }


def build_interpretation(
    args: argparse.Namespace,
    by_behavior: Mapping[str, Mapping],
    failures: Sequence[str],
    interpretation: str,
) -> str:
    if args.collection_mode != "accept_reject":
        return interpretation
    if not failures:
        return (
            "Under the heldout-excluded full-pool source-generation protocol "
            "with deterministic accept-reject collection, all four behaviors "
            f"produced {args.target_accepted_per_behavior} accepted pilot source "
            f"subjects under the {args.source_margin_gate:.2f} heldout "
            "source-margin gate. No rejection was required under this seed "
            "schedule; this is source-generation feasibility only."
        )
    limiting = [
        behavior for behavior, summary in by_behavior.items()
        if summary["pass_count"] < args.target_accepted_per_behavior
    ]
    return (
        "Under the heldout-excluded full-pool source-generation protocol "
        "with deterministic accept-reject collection, the pilot failed its "
        f"accepted-source gate for: {', '.join(limiting)}. This is "
        "source-generation feasibility only."
    )


def evaluate_attempt(
    pattern: str,
    pattern_index: int,
    attempt_index: int,
    seed: int,
    suite: Mapping,
    args: argparse.Namespace,
    candidate_pools: Mapping[str, Mapping[str, List[tuple[int, ...]]]],
    heldout_sequences: set[tuple[int, ...]],
    probe_examples: Sequence[Mapping],
    include_values: bool = False,
) -> Dict:
    model, train_info = train_subject(
        pattern=pattern,
        seed=seed,
        suite=suite,
        epochs=args.train_epochs,
        lr=args.lr,
        args=args,
        candidate_pools=candidate_pools,
        heldout_sequences=heldout_sequences,
    )
    weights = model.to_flat().detach().cpu().float()
    signature = extract_signature_with_stored_probes(weights, probe_examples)
    support_margin = subject_margin(model, pattern, "support", suite)
    heldout_margin = subject_margin(model, pattern, "heldout", suite)
    overlap_count = int(train_info["selected_train_vs_heldout_overlap_count"])
    accepted = heldout_margin >= args.source_margin_gate and overlap_count == 0
    record = {
        "accepted": accepted,
        "accepted_index": None,
        "attempt_index": int(attempt_index),
        "heldout_margin": heldout_margin,
        "passed_source_gate": heldout_margin >= args.source_margin_gate,
        "pattern": pattern,
        "pattern_index": int(pattern_index),
        "seed": int(seed),
        "signature_hash": signature_hash_stable_float_list(signature.tolist()),
        "source_margin_gate": float(args.source_margin_gate),
        "subject_id": (
            f"four_behavior_source_feasibility:{pattern}:attempt:{attempt_index}:seed:{seed}"
        ),
        "support_margin": support_margin,
        "train_info": train_info,
        "weights_hash": stable_hash_json([float(value) for value in weights.tolist()]),
    }
    if include_values:
        record["signature"] = [float(value) for value in signature.tolist()]
        record["weights"] = [float(value) for value in weights.tolist()]
    return record


def summarize_by_behavior(
    records: Sequence[Mapping],
    source_margin_gate: float,
) -> Dict[str, Dict]:
    return {
        pattern: summarize_records(
            [record for record in records if record["pattern"] == pattern],
            source_margin_gate,
        )
        for pattern in PATTERNS
    }


def summarize_records(
    records: Sequence[Mapping],
    source_margin_gate: float,
) -> Dict:
    margins = [float(record["heldout_margin"]) for record in records]
    support_margins = [float(record["support_margin"]) for record in records]
    pass_count = sum(
        1 for record in records
        if float(record["heldout_margin"]) >= source_margin_gate
    )
    return {
        "heldout_margin_max": max(margins) if margins else None,
        "heldout_margin_mean": safe_mean(margins),
        "heldout_margin_min": min(margins) if margins else None,
        "n": len(records),
        "pass_count": int(pass_count),
        "pass_rate": pass_count / len(records) if records else 0.0,
        "source_margin_gate": float(source_margin_gate),
        "support_margin_mean": safe_mean(support_margins),
    }


def summarize_acceptance(records: Sequence[Mapping]) -> Dict[str, Dict]:
    summary = {}
    for pattern in PATTERNS:
        pattern_records = [
            record for record in records
            if record["pattern"] == pattern
        ]
        accepted_records = [
            record for record in pattern_records
            if record["accepted"]
        ]
        rejected_records = [
            record for record in pattern_records
            if not record["accepted"]
        ]
        summary[pattern] = {
            "accepted_count": len(accepted_records),
            "accepted_subject_ids": [
                record["subject_id"] for record in accepted_records
            ],
            "acceptance_rate": (
                len(accepted_records) / len(pattern_records)
                if pattern_records else 0.0
            ),
            "attempts_used": len(pattern_records),
            "rejected_count": len(rejected_records),
            "rejected_subject_ids": [
                record["subject_id"] for record in rejected_records
            ],
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
    result = evaluate(args)
    result["result_payload_sha256"] = stable_hash_json(result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "aggregate": result["aggregate"],
        "by_behavior": result["by_behavior"],
        "failures": result["failures"],
        "passed": result["passed"],
        "results_path": str(result_path),
        "results_sha256": sha256_file(result_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
