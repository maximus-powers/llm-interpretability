"""Train/development V3: per-subject stored-probe signature inversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
SCRIPT_ROOT = MODEL_ZOO_ROOT / "scripts"
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from hypernet.behavior_suite import build_clean_behavior_suite  # noqa: E402
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402
import train_four_behavior_decoder_development as v1  # noqa: E402


PATTERNS = v1.PATTERNS
FORBIDDEN_FINAL_RAW_NAME = v1.FORBIDDEN_FINAL_RAW_NAME
SEALED_FINAL_RAW_PATH = v1.SEALED_FINAL_RAW_PATH
THRESHOLDS = {
    **v1.THRESHOLDS,
    "min_inferred_behavior_accuracy": 0.90,
    "min_per_behavior_inferred_behavior_accuracy": 0.80,
}

assert_no_final_raw_paths = v1.assert_no_final_raw_paths
best_control_metrics = v1.best_control_metrics
select_train_control = v1.select_train_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/four_behavior_decoder_development_v3_signature_inversion",
    )
    parser.add_argument(
        "--train-pool",
        default="runs/four_behavior_decoder_source_pools_v2/train_subjects.json",
    )
    parser.add_argument(
        "--development-pool",
        default="runs/four_behavior_decoder_source_pools_v2/development_subjects.json",
    )
    parser.add_argument(
        "--combined-audit",
        default="runs/four_behavior_decoder_source_pools_v2/combined_audit.json",
    )
    parser.add_argument(
        "--final-redacted-audit",
        default="runs/four_behavior_decoder_source_pools_v2/final_redacted_audit.json",
    )
    parser.add_argument("--optimization-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--noise-base-seed", type=int, default=20260617)
    return parser.parse_args()


def main() -> None:
    assert_no_final_raw_paths(sys.argv[1:])
    args = parse_args()
    opened_paths: List[str] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [
        Path(args.train_pool),
        Path(args.development_pool),
        Path(args.combined_audit),
        Path(args.final_redacted_audit),
    ]
    assert_no_final_raw_paths(input_paths)
    train_payload = v1.read_json(Path(args.train_pool), opened_paths)
    development_payload = v1.read_json(Path(args.development_pool), opened_paths)
    combined_audit = v1.read_json(Path(args.combined_audit), opened_paths)
    final_redacted_audit = v1.read_json(Path(args.final_redacted_audit), opened_paths)
    assert_no_final_raw_paths(opened_paths)

    train_records = v1.accepted_records(train_payload)
    development_records = v1.accepted_records(development_payload)
    suite = build_suite()
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    data = v1.build_tensors(train_records, development_records)
    train_centroids = build_train_centroids(data)
    query_plan = build_query_plan(
        args=args,
        data=data,
        train_records=train_records,
        development_records=development_records,
        train_centroids=train_centroids,
    )
    optimized = invert_query_signatures(
        args=args,
        query_plan=query_plan,
        data=data,
        suite=suite,
        probe_examples=probe_examples,
    )
    eval_result = evaluate_development(
        data=data,
        suite=suite,
        train_records=train_records,
        development_records=development_records,
        query_plan=query_plan,
        optimized_weights=optimized["weights"],
        optimized_signature_mse=optimized["signature_mse"],
    )
    checkpoint_path = output_dir / "decoded_weights.pt"
    torch.save(
        {
            "decoded_weights": optimized["weights"],
            "query_plan": query_plan,
            "sig_mean": data["sig_mean"],
            "sig_std": data["sig_std"],
            "train_centroids": train_centroids,
        },
        checkpoint_path,
    )
    result = build_result(
        args=args,
        opened_paths=opened_paths,
        train_records=train_records,
        development_records=development_records,
        combined_audit=combined_audit,
        final_redacted_audit=final_redacted_audit,
        probe_examples=probe_examples,
        suite=suite,
        data=data,
        train_centroids=train_centroids,
        query_plan=query_plan,
        optimized=optimized,
        eval_result=eval_result,
        checkpoint_path=checkpoint_path,
    )
    result["result_payload_sha256"] = stable_hash_json(result)
    result_text = json.dumps(result, sort_keys=True)
    result["result_text_excludes_final_subjects_json"] = (
        FORBIDDEN_FINAL_RAW_NAME not in result_text
    )
    if not result["result_text_excludes_final_subjects_json"]:
        result["failures"].append("result artifact names final_subjects.json")
        result["passed"] = False
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "aggregate": result["aggregate"],
        "failures": result["failures"],
        "passed": result["passed"],
        "results_path": str(results_path),
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def build_suite() -> Dict:
    return build_clean_behavior_suite(
        patterns=PATTERNS,
        support_per_class=160,
        heldout_per_class=64,
        seed=20260609,
    )


def differentiable_signature(
    flat_weights: torch.Tensor,
    probe_examples: Sequence[Mapping],
) -> torch.Tensor:
    return differentiable_signature_batch(flat_weights.unsqueeze(0), probe_examples)[0]


def differentiable_signature_batch(
    flat_weights: torch.Tensor,
    probe_examples: Sequence[Mapping],
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
            fft_mag = torch.abs(torch.fft.fft(neuron_acts, dim=1))[:, : max(1, n_samples // 2)]
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
) -> List[torch.Tensor]:
    x = inputs.unsqueeze(0).expand(flat_weights.shape[0], -1, -1)
    offset = 0
    activations = []
    for out_dim, in_dim in [(8, 5), (8, 8), (8, 8), (8, 8), (8, 8)]:
        size = out_dim * in_dim
        weight = flat_weights[:, offset:offset + size].view(-1, out_dim, in_dim)
        offset += size
        bias = flat_weights[:, offset:offset + out_dim].view(-1, out_dim)
        offset += out_dim
        x = torch.einsum("bni,boi->bno", x, weight) + bias.unsqueeze(1)
        x = F.gelu(x)
        activations.append(x)
    return activations


def safe_corrcoef_batch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean()
    denom = left_centered.norm(dim=1) * right_centered.norm()
    value = (left_centered * right_centered.unsqueeze(0)).sum(dim=1) / denom.clamp_min(1e-12)
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def build_train_centroids(data: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    centroids = {}
    for pattern in PATTERNS:
        mask = torch.tensor(
            [name == pattern for name in data["train_pattern_names"]],
            dtype=torch.bool,
        )
        centroids[pattern] = data["train_signatures_norm"][mask].mean(dim=0)
    return centroids


def infer_behavior_from_centroids(
    query_norm: torch.Tensor,
    centroids: Mapping[str, torch.Tensor],
) -> str:
    distances = {
        pattern: float(torch.sum((query_norm - centroid) ** 2).item())
        for pattern, centroid in centroids.items()
    }
    return min(distances, key=distances.get)


def build_query_plan(
    args: argparse.Namespace,
    data: Mapping[str, torch.Tensor],
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
    train_centroids: Mapping[str, torch.Tensor],
) -> Dict:
    query_entries = []
    train_norm = data["train_signatures_norm"]
    global_centroid = data["train_signatures_norm"].mean(dim=0)
    for dev_index, record in enumerate(development_records):
        query_entries.append(make_query_entry(
            data=data,
            train_records=train_records,
            train_norm=train_norm,
            train_centroids=train_centroids,
            dev_index=dev_index,
            dev_subject_id=record["subject_id"],
            control_type="matched",
            query_norm=data["development_signatures_norm"][dev_index],
        ))
        centroid_queries = [
            ("null_signature", torch.zeros_like(global_centroid)),
            ("train_global_centroid", global_centroid),
            (f"same_label_train_centroid:{record['pattern']}", train_centroids[record["pattern"]]),
        ]
        for pattern in PATTERNS:
            if pattern != record["pattern"]:
                centroid_queries.append((
                    f"other_label_train_centroid:{pattern}",
                    train_centroids[pattern],
                ))
        for control_type, query_norm in centroid_queries:
            query_entries.append(make_query_entry(
                data=data,
                train_records=train_records,
                train_norm=train_norm,
                train_centroids=train_centroids,
                dev_index=dev_index,
                dev_subject_id=record["subject_id"],
                control_type=control_type,
                query_norm=query_norm,
            ))
        seed_hash = stable_hash_json([record["subject_id"], "v3_noise", args.noise_base_seed])
        seed = int(seed_hash[:16], 16) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        for noise_index in range(32):
            query_entries.append(make_query_entry(
                data=data,
                train_records=train_records,
                train_norm=train_norm,
                train_centroids=train_centroids,
                dev_index=dev_index,
                dev_subject_id=record["subject_id"],
                control_type=f"noise_signature:{noise_index:02d}",
                query_norm=torch.randn(560, generator=generator),
                noise_index=noise_index,
                noise_seed=seed,
            ))
    init_indices = [entry["nearest_train_index"] for entry in query_entries]
    query_norm = torch.stack([entry["query_norm"] for entry in query_entries])
    init_weights = data["train_weights"][torch.tensor(init_indices, dtype=torch.long)]
    return {
        "entries": query_entries,
        "init_weights": init_weights,
        "query_norm": query_norm,
    }


def make_query_entry(
    data: Mapping[str, torch.Tensor],
    train_records: Sequence[Mapping],
    train_norm: torch.Tensor,
    train_centroids: Mapping[str, torch.Tensor],
    dev_index: int,
    dev_subject_id: str,
    control_type: str,
    query_norm: torch.Tensor,
    noise_index: int | None = None,
    noise_seed: int | None = None,
) -> Dict:
    distances = torch.sum((train_norm - query_norm.unsqueeze(0)) ** 2, dim=1)
    nearest_index = int(torch.argmin(distances).item())
    inferred = infer_behavior_from_centroids(query_norm, train_centroids)
    entry = {
        "control_type": control_type,
        "dev_index": int(dev_index),
        "dev_subject_id_hash": stable_hash_json(dev_subject_id),
        "inferred_behavior": inferred,
        "nearest_train_behavior": train_records[nearest_index]["pattern"],
        "nearest_train_distance": float(distances[nearest_index].item()),
        "nearest_train_index": nearest_index,
        "nearest_train_subject_id_hash": stable_hash_json(
            train_records[nearest_index]["subject_id"]
        ),
        "query_norm": query_norm.detach().clone(),
    }
    if noise_index is not None:
        entry["noise_index"] = int(noise_index)
        entry["noise_seed"] = int(noise_seed)
    return entry


def invert_query_signatures(
    args: argparse.Namespace,
    query_plan: Mapping,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    probe_examples: Sequence[Mapping],
) -> Dict:
    torch.manual_seed(args.seed)
    weights = query_plan["init_weights"].detach().clone().requires_grad_(True)
    init_weights = query_plan["init_weights"].detach().clone()
    query_norm = query_plan["query_norm"].detach()
    optimizer = torch.optim.Adam([weights], lr=args.lr)
    inferred_patterns = [entry["inferred_behavior"] for entry in query_plan["entries"]]
    for _ in range(args.optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        signature = differentiable_signature_batch(weights, probe_examples)
        signature_norm = (signature - data["sig_mean"]) / data["sig_std"]
        signature_loss = F.mse_loss(signature_norm, query_norm)
        behavior = batch_behavior_loss(weights, inferred_patterns, suite)
        init_l2 = F.mse_loss(weights, init_weights)
        loss = 5.0 * signature_loss + 0.5 * behavior["bce"] + behavior["margin_hinge"] + 0.01 * init_l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weights], args.grad_clip_norm)
        optimizer.step()
    with torch.no_grad():
        signature = differentiable_signature_batch(weights, probe_examples)
        signature_norm = (signature - data["sig_mean"]) / data["sig_std"]
        signature_mse = ((signature_norm - query_norm) ** 2).mean(dim=1)
    return {
        "signature_mse": signature_mse.detach().cpu(),
        "weights": weights.detach().cpu(),
    }


def batch_behavior_loss(
    flat_weights: torch.Tensor,
    patterns: Sequence[str],
    suite: Mapping,
) -> Dict[str, torch.Tensor]:
    bces = []
    hinges = []
    for pattern in PATTERNS:
        indices = [index for index, value in enumerate(patterns) if value == pattern]
        if not indices:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=flat_weights.device)
        positive = v1.sequence_tensor(suite["support"][pattern]["positive"]).to(flat_weights.device)
        negative = v1.sequence_tensor(suite["support"][pattern]["negative"]).to(flat_weights.device)
        inputs = torch.cat([positive, negative], dim=0)
        labels = torch.cat([
            torch.ones(len(positive), device=flat_weights.device),
            torch.zeros(len(negative), device=flat_weights.device),
        ])
        logits = v1.subject_forward_flat_batch(flat_weights[idx], inputs)
        bces.append(F.binary_cross_entropy_with_logits(
            logits,
            labels.unsqueeze(0).expand(logits.shape[0], -1),
        ))
        pos_prob = torch.sigmoid(logits[:, :len(positive)]).mean(dim=1)
        neg_prob = torch.sigmoid(logits[:, len(positive):]).mean(dim=1)
        hinges.append(F.relu(torch.tensor(0.40, device=flat_weights.device) - (pos_prob - neg_prob)).mean())
    return {
        "bce": torch.stack(bces).mean(),
        "margin_hinge": torch.stack(hinges).mean(),
    }


def evaluate_development(
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
    query_plan: Mapping,
    optimized_weights: torch.Tensor,
    optimized_signature_mse: torch.Tensor,
) -> Dict:
    entries_by_dev: Dict[int, List[tuple[int, Mapping]]] = {}
    for query_index, entry in enumerate(query_plan["entries"]):
        entries_by_dev.setdefault(entry["dev_index"], []).append((query_index, entry))
    train_by_behavior = {
        pattern: [record for record in train_records if record["pattern"] == pattern]
        for pattern in PATTERNS
    }
    records = []
    for dev_index, dev_record in enumerate(development_records):
        pattern = dev_record["pattern"]
        source_weights = data["development_weights"][dev_index]
        subject_inputs = v1.heldout_inputs_for(pattern, suite)
        query_entries = entries_by_dev[dev_index]
        matched_index, matched_entry = next(
            (idx, entry)
            for idx, entry in query_entries
            if entry["control_type"] == "matched"
        )
        matched_weights = optimized_weights[matched_index]
        controls = []
        for query_index, entry in query_entries:
            if entry["control_type"] == "matched":
                continue
            controls.append(evaluate_weight_control(
                control_behavior=entry["inferred_behavior"],
                control_type=f"v3_inversion:{entry['control_type']}",
                weights=optimized_weights[query_index],
                source_weights=source_weights,
                subject_inputs=subject_inputs,
                target_pattern=pattern,
                suite=suite,
                extra={
                    "inferred_behavior": entry["inferred_behavior"],
                    "nearest_train_behavior": entry["nearest_train_behavior"],
                    "nearest_train_distance": entry["nearest_train_distance"],
                },
            ))
        nearest_train_weights = data["train_weights"][matched_entry["nearest_train_index"]]
        controls.append(evaluate_weight_control(
            control_behavior=matched_entry["nearest_train_behavior"],
            control_type="nearest_train_signature_neighbor",
            weights=nearest_train_weights,
            source_weights=source_weights,
            subject_inputs=subject_inputs,
            target_pattern=pattern,
            suite=suite,
            extra={
                "nearest_train_distance": matched_entry["nearest_train_distance"],
                "nearest_train_subject_id_hash": matched_entry["nearest_train_subject_id_hash"],
            },
        ))
        same = select_train_control(
            train_by_behavior[pattern],
            development_subject_id=dev_record["subject_id"],
            control_family="same_label_other_subject",
            control_behavior=pattern,
        )
        controls.append(evaluate_weight_control(
            control_behavior=pattern,
            control_type="same_label_other_subject",
            weights=torch.tensor(same["weights"], dtype=torch.float32),
            source_weights=source_weights,
            subject_inputs=subject_inputs,
            target_pattern=pattern,
            suite=suite,
        ))
        for other in PATTERNS:
            if other == pattern:
                continue
            selected = select_train_control(
                train_by_behavior[other],
                development_subject_id=dev_record["subject_id"],
                control_family="different_label_other_subject",
                control_behavior=other,
            )
            controls.append(evaluate_weight_control(
                control_behavior=other,
                control_type=f"different_label_other_subject:{other}",
                weights=torch.tensor(selected["weights"], dtype=torch.float32),
                source_weights=source_weights,
                subject_inputs=subject_inputs,
                target_pattern=pattern,
                suite=suite,
            ))
        matched_subject_output_mse = v1.subject_output_mse(
            matched_weights,
            source_weights,
            subject_inputs,
        )
        matched_target_margin = v1.target_margin(matched_weights, pattern, suite)
        for control in controls:
            control["matched_minus_control_target_margin"] = (
                matched_target_margin - control["target_margin"]
            )
            control["control_minus_matched_subject_output_mse"] = (
                control["subject_output_mse"] - matched_subject_output_mse
            )
        best = best_control_metrics(controls)
        record = {
            "best_control_minus_matched_subject_output_mse": (
                best["best_subject_output_mse"]
                - matched_subject_output_mse
            ),
            "best_subject_output_mse": best["best_subject_output_mse"],
            "best_subject_output_mse_control_type": best[
                "best_subject_output_mse_control_type"
            ],
            "best_target_margin": best["best_target_margin"],
            "best_target_margin_control_type": best["best_target_margin_control_type"],
            "controls": controls,
            "inferred_behavior": matched_entry["inferred_behavior"],
            "inferred_behavior_correct": matched_entry["inferred_behavior"] == pattern,
            "matched_signature_mse": float(optimized_signature_mse[matched_index].item()),
            "matched_subject_output_mse": matched_subject_output_mse,
            "matched_target_margin": matched_target_margin,
            "nearest_train_behavior": matched_entry["nearest_train_behavior"],
            "nearest_train_distance": matched_entry["nearest_train_distance"],
            "nearest_train_subject_id_hash": matched_entry["nearest_train_subject_id_hash"],
            "pattern": pattern,
            "subject_id_hash": stable_hash_json(dev_record["subject_id"]),
        }
        record["matched_minus_best_control_target_margin"] = (
            record["matched_target_margin"] - record["best_target_margin"]
        )
        record["individual_passed"] = individual_passed_v3(record)
        records.append(record)
    return summarize_records_v3(records)


def evaluate_weight_control(
    control_behavior: str,
    control_type: str,
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    subject_inputs: torch.Tensor,
    target_pattern: str,
    suite: Mapping,
    extra: Mapping | None = None,
) -> Dict:
    payload = {
        "control_behavior": control_behavior,
        "control_type": control_type,
        "subject_output_mse": v1.subject_output_mse(weights, source_weights, subject_inputs),
        "target_margin": v1.target_margin(weights, target_pattern, suite),
    }
    if extra:
        payload.update(extra)
    return payload


def individual_passed_v3(record: Mapping) -> bool:
    return (
        record["inferred_behavior_correct"]
        and record["matched_target_margin"] >= THRESHOLDS["min_individual_matched_target_margin"]
        and record["matched_minus_best_control_target_margin"]
        >= THRESHOLDS["min_individual_matched_minus_best_control_target_margin"]
        and record["best_control_minus_matched_subject_output_mse"]
        > THRESHOLDS["min_individual_best_control_minus_matched_subject_output_mse"]
    )


def summarize_records_v3(records: Sequence[Mapping]) -> Dict:
    aggregate = summarize_metric_records_v3(records)
    by_behavior = {
        pattern: summarize_metric_records_v3([
            record for record in records if record["pattern"] == pattern
        ])
        for pattern in PATTERNS
    }
    failures = []
    require_metric(
        failures,
        aggregate["inferred_behavior_accuracy"],
        THRESHOLDS["min_inferred_behavior_accuracy"],
        "aggregate inferred behavior accuracy",
    )
    require_metric(
        failures,
        aggregate["mean_matched_target_margin"],
        THRESHOLDS["min_mean_matched_target_margin"],
        "aggregate matched target margin",
    )
    require_metric(
        failures,
        aggregate["mean_matched_minus_best_control_target_margin"],
        THRESHOLDS["min_mean_matched_minus_best_control_target_margin"],
        "aggregate matched-minus-best-control target margin",
    )
    require_metric(
        failures,
        aggregate["mean_best_control_minus_matched_subject_output_mse"],
        THRESHOLDS["min_mean_best_control_minus_matched_subject_output_mse"],
        "aggregate best-control-minus-matched subject MSE",
    )
    require_metric(
        failures,
        aggregate["individual_all_gate_pass_rate"],
        THRESHOLDS["min_individual_all_gate_pass_rate"],
        "aggregate individual pass rate",
    )
    for pattern, summary in by_behavior.items():
        if summary["n"] < 24:
            failures.append(f"{pattern} development count below 24")
        require_metric(
            failures,
            summary["inferred_behavior_accuracy"],
            THRESHOLDS["min_per_behavior_inferred_behavior_accuracy"],
            f"{pattern} inferred behavior accuracy",
        )
        require_metric(
            failures,
            summary["mean_matched_target_margin"],
            THRESHOLDS["min_per_behavior_matched_target_margin"],
            f"{pattern} matched target margin",
        )
        require_metric(
            failures,
            summary["mean_matched_minus_best_control_target_margin"],
            THRESHOLDS["min_per_behavior_matched_minus_best_control_target_margin"],
            f"{pattern} matched-minus-best-control target margin",
        )
        require_metric(
            failures,
            summary["mean_best_control_minus_matched_subject_output_mse"],
            THRESHOLDS["min_per_behavior_best_control_minus_matched_subject_output_mse"],
            f"{pattern} best-control-minus-matched subject MSE",
        )
        require_metric(
            failures,
            summary["individual_all_gate_pass_rate"],
            THRESHOLDS["min_per_behavior_individual_all_gate_pass_rate"],
            f"{pattern} individual pass rate",
        )
    return {
        "aggregate": aggregate,
        "by_behavior": by_behavior,
        "development_records": records,
        "failures": failures,
        "inferred_behavior_confusion": inferred_behavior_confusion(records),
        "inverted_control_inferred_behavior_counts": control_inferred_counts(records),
        "passed": not failures,
    }


def summarize_metric_records_v3(records: Sequence[Mapping]) -> Dict:
    n = len(records)
    passed = sum(1 for record in records if record["individual_passed"])
    inferred = sum(1 for record in records if record["inferred_behavior_correct"])
    return {
        "individual_all_gate_pass_count": int(passed),
        "individual_all_gate_pass_rate": float(passed / n) if n else 0.0,
        "inferred_behavior_accuracy": float(inferred / n) if n else 0.0,
        "mean_best_control_minus_matched_subject_output_mse": mean(
            record["best_control_minus_matched_subject_output_mse"]
            for record in records
        ),
        "mean_matched_minus_best_control_target_margin": mean(
            record["matched_minus_best_control_target_margin"]
            for record in records
        ),
        "mean_matched_signature_mse": mean(
            record["matched_signature_mse"]
            for record in records
        ),
        "mean_matched_subject_output_mse": mean(
            record["matched_subject_output_mse"]
            for record in records
        ),
        "mean_matched_target_margin": mean(
            record["matched_target_margin"]
            for record in records
        ),
        "n": int(n),
    }


def inferred_behavior_confusion(records: Sequence[Mapping]) -> Dict[str, Dict[str, int]]:
    matrix = {pattern: {other: 0 for other in PATTERNS} for pattern in PATTERNS}
    for record in records:
        matrix[record["pattern"]][record["inferred_behavior"]] += 1
    return matrix


def control_inferred_counts(records: Sequence[Mapping]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Counter] = {}
    for record in records:
        for control in record["controls"]:
            if not control["control_type"].startswith("v3_inversion:"):
                continue
            key = control["control_type"].split(":", 1)[0]
            counts.setdefault(key, Counter())[control["inferred_behavior"]] += 1
    return {key: dict(value) for key, value in counts.items()}


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def require_metric(
    failures: List[str],
    value: float,
    threshold: float,
    label: str,
) -> None:
    if value < threshold:
        failures.append(f"{label} {value:.6f} < {threshold:.6f}")


def build_result(
    args: argparse.Namespace,
    opened_paths: Sequence[str],
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
    combined_audit: Mapping,
    final_redacted_audit: Mapping,
    probe_examples: Sequence[Mapping],
    suite: Mapping,
    data: Mapping[str, torch.Tensor],
    train_centroids: Mapping[str, torch.Tensor],
    query_plan: Mapping,
    optimized: Mapping,
    eval_result: Mapping,
    checkpoint_path: Path,
) -> Dict:
    input_audit = {
        "argv": sys.argv,
        "no_opened_path_endswith_final_subjects_json": not any(
            path.endswith(FORBIDDEN_FINAL_RAW_NAME) for path in opened_paths
        ),
        "opened_paths": list(opened_paths),
        "sealed_final_raw_path_not_opened": str(SEALED_FINAL_RAW_PATH) not in opened_paths,
    }
    failures = list(eval_result["failures"])
    if not input_audit["no_opened_path_endswith_final_subjects_json"]:
        failures.append("opened final_subjects.json")
    if not input_audit["sealed_final_raw_path_not_opened"]:
        failures.append("opened sealed final raw path")
    return {
        "aggregate": eval_result["aggregate"],
        "by_behavior": eval_result["by_behavior"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "claim_scope": "four_behavior_decoder_development_v3_signature_inversion_not_final_proof",
        "combined_audit_hash": stable_hash_json(combined_audit),
        "config": {
            "grad_clip_norm": float(args.grad_clip_norm),
            "lr": float(args.lr),
            "noise_base_seed": int(args.noise_base_seed),
            "optimization_steps": int(args.optimization_steps),
            "seed": int(args.seed),
            "loss_weights": {
                "initialization_l2": 0.01,
                "signature_mse": 5.0,
                "support_bce": 0.5,
                "support_margin_hinge": 1.0,
            },
            "support_margin_target": 0.40,
        },
        "development_records": eval_result["development_records"],
        "development_status": "adaptive_v3_signature_inversion_train_development_only_final_pool_sealed",
        "development_subject_counts_by_behavior": v1.count_by_behavior(development_records),
        "evidence_interpretation": (
            "adaptive V3 signature-inversion development eligibility only; "
            "not stored-probe final decoder proof"
        ),
        "failures": failures,
        "final_redacted_audit_hash": stable_hash_json(final_redacted_audit),
        "inferred_behavior_confusion": eval_result["inferred_behavior_confusion"],
        "input_path_audit": input_audit,
        "inverted_control_inferred_behavior_counts": eval_result[
            "inverted_control_inferred_behavior_counts"
        ],
        "normalization_hashes": {
            "sig_mean": v1.tensor_hash(data["sig_mean"]),
            "sig_std": v1.tensor_hash(data["sig_std"]),
            "weight_mean": v1.tensor_hash(data["weight_mean"]),
            "weight_std": v1.tensor_hash(data["weight_std"]),
        },
        "optimized_signature_mse_mean": float(optimized["signature_mse"].mean().item()),
        "overlap_counts": v1.train_development_overlap_counts(
            train_records,
            development_records,
        ),
        "passed": not failures,
        "probe_examples_hash": stable_hash_json(probe_examples),
        "query_count": len(query_plan["entries"]),
        "source_pool_hashes": {
            "development": combined_audit["pool_file_sha256"]["development"],
            "train": combined_audit["pool_file_sha256"]["train"],
        },
        "suite_hashes": {
            "heldout": suite["metadata"]["heldout_hash"],
            "support": suite["metadata"]["support_hash"],
        },
        "thresholds": THRESHOLDS,
        "train_centroid_hash": stable_hash_json({
            key: [float(value) for value in tensor.tolist()]
            for key, tensor in train_centroids.items()
        }),
        "train_subject_counts_by_behavior": v1.count_by_behavior(train_records),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
