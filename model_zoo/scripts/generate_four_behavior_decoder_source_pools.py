"""Generate disjoint source pools for the four-behavior decoder proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model_zoo" / "scripts"))

from evaluate_four_behavior_source_generation_feasibility import (  # noqa: E402
    PATTERNS,
    build_candidate_pools,
    build_heldout_sequences,
    build_suite,
    evaluate_attempt,
    summarize_candidate_pools,
)
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402


SEED_BEHAVIOR_STRIDE = 100000

POOL_CONFIGS = {
    "train": {
        "base_seed": 20300000,
        "target_accepted_per_behavior": 64,
        "max_attempts_per_behavior": 128,
    },
    "development": {
        "base_seed": 21300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
    "final": {
        "base_seed": 22300000,
        "target_accepted_per_behavior": 24,
        "max_attempts_per_behavior": 64,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/four_behavior_decoder_source_pools_v2",
    )
    parser.add_argument("--train-epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--source-margin-gate", type=float, default=0.40)
    parser.add_argument("--support-per-class", type=int, default=160)
    parser.add_argument("--heldout-per-class", type=int, default=64)
    parser.add_argument("--positive-cap", type=int, default=2048)
    parser.add_argument("--hard-negative-cap", type=int, default=1024)
    parser.add_argument("--generic-negative-cap", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_preflight = build_seed_preflight()
    if seed_preflight["failures"]:
        raise SystemExit(json.dumps(seed_preflight, indent=2, sort_keys=True))
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
    for pool_name, pool_config in POOL_CONFIGS.items():
        payload = generate_pool(
            args=args,
            pool_name=pool_name,
            pool_config=pool_config,
            suite=suite,
            heldout_sequences=heldout_sequences,
            candidate_pools=candidate_pools,
            candidate_pool_summary=candidate_pool_summary,
            probe_examples=probe_examples,
        )
        payload["pool_redacted_payload_sha256"] = stable_hash_json(
            redact_weights_and_signatures(payload)
        )
        pool_payloads[pool_name] = payload
        pool_summaries[pool_name] = summarize_pool(payload)
        pool_path = output_dir / f"{pool_name}_subjects.json"
        pool_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
        pool_summaries[pool_name]["pool_file_sha256"] = sha256_file(pool_path)
        pool_summaries[pool_name]["pool_redacted_payload_sha256"] = payload[
            "pool_redacted_payload_sha256"
        ]

    final_redacted = build_final_redacted_summary(pool_payloads["final"])
    final_redacted["pool_file_sha256"] = pool_summaries["final"]["pool_file_sha256"]
    final_redacted["summary_payload_sha256"] = stable_hash_json(final_redacted)
    (output_dir / "final_redacted_audit.json").write_text(
        json.dumps(final_redacted, indent=2, sort_keys=True)
    )

    audit = build_combined_audit(
        pool_payloads=pool_payloads,
        pool_summaries=pool_summaries,
        seed_preflight=seed_preflight,
        suite=suite,
        probe_examples=probe_examples,
    )
    audit["audit_payload_sha256"] = stable_hash_json(audit)
    (output_dir / "combined_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    print(json.dumps({
        "combined_audit_path": str(output_dir / "combined_audit.json"),
        "failures": audit["failures"],
        "passed": audit["passed"],
        "pool_summaries": {
            name: {
            "accepted_counts_by_behavior": summary["accepted_counts_by_behavior"],
            "attempt_counts_by_behavior": (
                summary["attempt_counts_by_behavior"]
                if name != "final"
                else "sealed"
            ),
            }
            for name, summary in pool_summaries.items()
        },
    }, indent=2, sort_keys=True))


def generate_pool(
    args: argparse.Namespace,
    pool_name: str,
    pool_config: Mapping,
    suite: Mapping,
    heldout_sequences: set[tuple[int, ...]],
    candidate_pools: Mapping,
    candidate_pool_summary: Mapping,
    probe_examples: Sequence[Mapping],
) -> Dict:
    records = []
    for pattern_index, pattern in enumerate(PATTERNS):
        accepted_count = 0
        for attempt_index in range(pool_config["max_attempts_per_behavior"]):
            if accepted_count >= pool_config["target_accepted_per_behavior"]:
                break
            seed = (
                pool_config["base_seed"]
                + pattern_index * SEED_BEHAVIOR_STRIDE
                + attempt_index
            )
            record = evaluate_attempt(
                pattern=pattern,
                pattern_index=pattern_index,
                attempt_index=attempt_index,
                seed=seed,
                suite=suite,
                args=source_args(args),
                candidate_pools=candidate_pools,
                heldout_sequences=heldout_sequences,
                probe_examples=probe_examples,
                include_values=True,
            )
            record["pool"] = pool_name
            if record["accepted"]:
                accepted_count += 1
                record["accepted_index"] = accepted_count - 1
            records.append(record)
    summary = summarize_records(records)
    return {
        "behavior_suite_metadata": suite["metadata"],
        "candidate_pool_summary": candidate_pool_summary,
        "claim_scope": "four_behavior_decoder_source_pool_construction_not_decoder_evidence",
        "config": {
            "generic_negative_cap": int(args.generic_negative_cap),
            "hard_negative_cap": int(args.hard_negative_cap),
            "heldout_per_class": int(args.heldout_per_class),
            "lr": float(args.lr),
            "positive_cap": int(args.positive_cap),
            "source_margin_gate": float(args.source_margin_gate),
            "support_per_class": int(args.support_per_class),
            "seed_behavior_stride": int(SEED_BEHAVIOR_STRIDE),
            "train_epochs": int(args.train_epochs),
            "training_mode": "heldout_excluded_full_pool",
            **pool_config,
        },
        "development_status": "source_pool_construction_before_decoder_training",
        "pool": pool_name,
        "records": records,
        "summary": summary,
    }


def build_seed_preflight() -> Dict:
    ranges = []
    for pool_name, pool_config in POOL_CONFIGS.items():
        for pattern_index, pattern in enumerate(PATTERNS):
            start = pool_config["base_seed"] + pattern_index * SEED_BEHAVIOR_STRIDE
            end = start + pool_config["max_attempts_per_behavior"] - 1
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
            if left["end_seed"] < right["start_seed"] or right["end_seed"] < left["start_seed"]:
                continue
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
        "seed_behavior_stride": int(SEED_BEHAVIOR_STRIDE),
        "seed_ranges": ranges,
    }


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        generic_negative_cap=args.generic_negative_cap,
        hard_negative_cap=args.hard_negative_cap,
        lr=args.lr,
        positive_cap=args.positive_cap,
        source_margin_gate=args.source_margin_gate,
        train_epochs=args.train_epochs,
        training_mode="heldout_excluded_full_pool",
    )


def summarize_records(records: Sequence[Mapping]) -> Dict:
    by_behavior = {}
    max_overlap = 0
    for pattern in PATTERNS:
        behavior_records = [
            record for record in records
            if record["pattern"] == pattern
        ]
        accepted = [record for record in behavior_records if record["accepted"]]
        rejected = [record for record in behavior_records if not record["accepted"]]
        margins = [float(record["heldout_margin"]) for record in accepted]
        by_behavior[pattern] = {
            "accepted_count": len(accepted),
            "accepted_subject_ids": [record["subject_id"] for record in accepted],
            "acceptance_rate": (
                len(accepted) / len(behavior_records) if behavior_records else 0.0
            ),
            "attempt_count": len(behavior_records),
            "heldout_margin_min": min(margins) if margins else None,
            "heldout_margin_mean": safe_mean(margins),
            "rejected_count": len(rejected),
            "rejected_subject_ids": [record["subject_id"] for record in rejected],
        }
    for record in records:
        max_overlap = max(
            max_overlap,
            int(record["train_info"]["selected_train_vs_heldout_overlap_count"]),
        )
    return {
        "accepted_counts_by_behavior": {
            pattern: values["accepted_count"]
            for pattern, values in by_behavior.items()
        },
        "attempt_counts_by_behavior": {
            pattern: values["attempt_count"]
            for pattern, values in by_behavior.items()
        },
        "by_behavior": by_behavior,
        "max_selected_train_vs_heldout_overlap_count": int(max_overlap),
        "record_count": len(records),
    }


def summarize_pool(payload: Mapping) -> Dict:
    return payload["summary"]


def build_final_redacted_summary(payload: Mapping) -> Dict:
    return {
        "behavior_suite_hashes": {
            "heldout_hash": payload["behavior_suite_metadata"]["heldout_hash"],
            "support_hash": payload["behavior_suite_metadata"]["support_hash"],
        },
        "candidate_pool_summary_hash": stable_hash_json(payload["candidate_pool_summary"]),
        "claim_scope": "redacted_final_source_pool_audit_surface_only",
        "config_hash": stable_hash_json(payload["config"]),
        "pool": payload["pool"],
        "pool_redacted_payload_sha256": payload["pool_redacted_payload_sha256"],
        "summary": {
            "accepted_counts_by_behavior": payload["summary"]["accepted_counts_by_behavior"],
            "max_selected_train_vs_heldout_overlap_count": payload["summary"][
                "max_selected_train_vs_heldout_overlap_count"
            ],
        },
    }


def build_combined_audit(
    pool_payloads: Mapping[str, Mapping],
    pool_summaries: Mapping[str, Mapping],
    seed_preflight: Mapping,
    suite: Mapping,
    probe_examples: Sequence[Mapping],
) -> Dict:
    failures = []
    required_counts = {
        "train": 64,
        "development": 24,
        "final": 24,
    }
    for pool_name, required_count in required_counts.items():
        summary = pool_summaries[pool_name]
        for pattern, count in summary["accepted_counts_by_behavior"].items():
            if count < required_count:
                failures.append(f"{pool_name}/{pattern} accepted count {count} < {required_count}")
        if summary["max_selected_train_vs_heldout_overlap_count"] != 0:
            failures.append(f"{pool_name} selected train/heldout overlap is nonzero")
        for record in pool_payloads[pool_name]["records"]:
            if record["accepted"] and float(record["heldout_margin"]) < 0.40:
                failures.append(f"{pool_name}/{record['subject_id']} accepted below margin gate")
            if int(record["train_info"]["selected_train_vs_heldout_overlap_count"]) != 0:
                failures.append(f"{pool_name}/{record['subject_id']} overlap is nonzero")

    overlap_counts = compute_cross_pool_overlaps(pool_payloads)
    for key, value in overlap_counts.items():
        if value:
            failures.append(f"cross-pool accepted {key} overlap count {value}")
    for failure in seed_preflight["failures"]:
        failures.append(f"seed preflight failed: {failure}")

    return {
        "behavior_suite_hashes": {
            "heldout_hash": suite["metadata"]["heldout_hash"],
            "support_hash": suite["metadata"]["support_hash"],
        },
        "claim_scope": "source_pool_construction_not_decoder_evidence",
        "failures": failures,
        "overlap_counts": overlap_counts,
        "passed": not failures,
        "pool_file_sha256": {
            pool_name: pool_summaries[pool_name]["pool_file_sha256"]
            for pool_name in pool_payloads
        },
        "pool_redacted_payload_hashes": {
            pool_name: payload["pool_redacted_payload_sha256"]
            for pool_name, payload in pool_payloads.items()
        },
        "pool_summaries": build_public_pool_summaries(pool_summaries),
        "probe_examples_hash": stable_hash_json(probe_examples),
        "required_counts": required_counts,
        "seed_preflight": seed_preflight,
    }


def build_public_pool_summaries(pool_summaries: Mapping[str, Mapping]) -> Dict:
    public = {}
    for pool_name, summary in pool_summaries.items():
        if pool_name == "final":
            public[pool_name] = {
                "accepted_counts_by_behavior": summary["accepted_counts_by_behavior"],
                "max_selected_train_vs_heldout_overlap_count": summary[
                    "max_selected_train_vs_heldout_overlap_count"
                ],
                "pool_file_sha256": summary["pool_file_sha256"],
                "pool_redacted_payload_sha256": summary["pool_redacted_payload_sha256"],
            }
        else:
            public[pool_name] = summary
    return public


def compute_cross_pool_overlaps(pool_payloads: Mapping[str, Mapping]) -> Dict[str, int]:
    fields = {
        "seed": lambda record: record["seed"],
        "signature_hash": lambda record: record["signature_hash"],
        "subject_id": lambda record: record["subject_id"],
        "weights_hash": lambda record: record["weights_hash"],
    }
    accepted = {
        pool_name: [record for record in payload["records"] if record["accepted"]]
        for pool_name, payload in pool_payloads.items()
    }
    pool_names = list(accepted)
    overlaps = {}
    for index, left_name in enumerate(pool_names):
        for right_name in pool_names[index + 1:]:
            left_records = accepted[left_name]
            right_records = accepted[right_name]
            for field_name, getter in fields.items():
                left_values = {getter(record) for record in left_records}
                right_values = {getter(record) for record in right_records}
                key = f"{left_name}__{right_name}__{field_name}"
                overlaps[key] = len(left_values & right_values)
    return overlaps


def redact_weights_and_signatures(payload: Mapping) -> Dict:
    redacted = json.loads(json.dumps(payload))
    for record in redacted["records"]:
        record.pop("weights", None)
        record.pop("signature", None)
    return redacted


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
