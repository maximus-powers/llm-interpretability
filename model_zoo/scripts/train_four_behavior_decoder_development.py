"""Train/evaluate the four-behavior stored-probe decoder development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO_ROOT = REPO_ROOT / "model_zoo"
sys.path.insert(0, str(MODEL_ZOO_ROOT))

from hypernet.behavior_suite import build_clean_behavior_suite  # noqa: E402
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402


PATTERNS = (
    "sorted_ascending",
    "sorted_descending",
    "has_majority",
    "mountain_pattern",
)
FORBIDDEN_FINAL_RAW_NAME = "final_subjects.json"
SEALED_FINAL_RAW_PATH = (
    REPO_ROOT / "runs" / "four_behavior_decoder_source_pools_v2" / FORBIDDEN_FINAL_RAW_NAME
).resolve()
THRESHOLDS = {
    "min_mean_matched_target_margin": 0.20,
    "min_mean_matched_minus_best_control_target_margin": 0.20,
    "min_mean_best_control_minus_matched_subject_output_mse": 0.05,
    "min_individual_all_gate_pass_rate": 0.90,
    "min_per_behavior_matched_target_margin": 0.20,
    "min_per_behavior_matched_minus_best_control_target_margin": 0.15,
    "min_per_behavior_best_control_minus_matched_subject_output_mse": 0.02,
    "min_per_behavior_individual_all_gate_pass_rate": 0.80,
    "min_individual_matched_target_margin": 0.20,
    "min_individual_matched_minus_best_control_target_margin": 0.10,
    "min_individual_best_control_minus_matched_subject_output_mse": 0.00,
}


class SignatureToWeightsDecoder(nn.Module):
    """Direct MLP decoder from normalized signature to normalized flat weights."""

    def __init__(self, input_dim: int = 560, output_dim: int = 345):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in (512, 512, 512):
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, signatures_norm: torch.Tensor) -> torch.Tensor:
        return self.net(signatures_norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/four_behavior_decoder_development_v1",
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
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--noise-base-seed", type=int, default=20260612)
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
    train_payload = read_json(Path(args.train_pool), opened_paths)
    development_payload = read_json(Path(args.development_pool), opened_paths)
    combined_audit = read_json(Path(args.combined_audit), opened_paths)
    final_redacted_audit = read_json(Path(args.final_redacted_audit), opened_paths)
    assert_no_final_raw_paths(opened_paths)

    train_records = accepted_records(train_payload)
    development_records = accepted_records(development_payload)
    suite = build_clean_behavior_suite(
        patterns=PATTERNS,
        support_per_class=160,
        heldout_per_class=64,
        seed=20260609,
    )
    probe_examples = build_digit_probe_examples(
        n_examples=256,
        seed=20260610,
        seq_len=5,
        base=10,
    )
    data = build_tensors(train_records, development_records)
    model, training_history, best_checkpoint = train_decoder(
        args,
        data,
        suite,
        train_records,
        development_records,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    eval_result = evaluate_development(
        model=model,
        args=args,
        data=data,
        suite=suite,
        train_records=train_records,
        development_records=development_records,
    )

    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sig_mean": data["sig_mean"],
            "sig_std": data["sig_std"],
            "weight_mean": data["weight_mean"],
            "weight_std": data["weight_std"],
            "best_epoch": best_checkpoint["epoch"],
            "architecture": {
                "input_dim": 560,
                "hidden_layers": [512, 512, 512],
                "activation": "GELU",
                "normalization": "LayerNorm",
                "dropout": 0.0,
                "output_dim": 345,
            },
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
        training_history=training_history,
        best_checkpoint=best_checkpoint,
        eval_result=eval_result,
        checkpoint_path=checkpoint_path,
    )
    results_path = output_dir / "results.json"
    result["result_payload_sha256"] = stable_hash_json(result)
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "failures": result["failures"],
        "passed": result["passed"],
        "results_path": str(results_path),
        "aggregate": result["aggregate"],
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def assert_no_final_raw_paths(paths: Iterable[object]) -> None:
    for item in paths:
        path_text = str(item)
        if path_text.endswith(FORBIDDEN_FINAL_RAW_NAME):
            raise ValueError(f"forbidden final raw path: {path_text}")
        try:
            resolved = Path(path_text).resolve()
        except OSError:
            continue
        if resolved == SEALED_FINAL_RAW_PATH:
            raise ValueError(f"forbidden sealed final raw path: {path_text}")


def read_json(path: Path, opened_paths: List[str]) -> Dict:
    resolved = path.resolve()
    assert_no_final_raw_paths([resolved])
    opened_paths.append(str(resolved))
    return json.loads(resolved.read_text())


def accepted_records(payload: Mapping) -> List[Dict]:
    return [
        dict(record)
        for record in payload["records"]
        if record["accepted"]
    ]


def build_tensors(
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
) -> Dict[str, torch.Tensor]:
    train_signatures = torch.tensor(
        [record["signature"] for record in train_records],
        dtype=torch.float32,
    )
    train_weights = torch.tensor(
        [record["weights"] for record in train_records],
        dtype=torch.float32,
    )
    development_signatures = torch.tensor(
        [record["signature"] for record in development_records],
        dtype=torch.float32,
    )
    development_weights = torch.tensor(
        [record["weights"] for record in development_records],
        dtype=torch.float32,
    )
    sig_mean = train_signatures.mean(dim=0)
    sig_std = clamp_std(train_signatures.std(dim=0, unbiased=False))
    weight_mean = train_weights.mean(dim=0)
    weight_std = clamp_std(train_weights.std(dim=0, unbiased=False))
    return {
        "development_pattern_names": [
            str(record["pattern"]) for record in development_records
        ],
        "train_signatures": train_signatures,
        "train_signatures_norm": (train_signatures - sig_mean) / sig_std,
        "train_weights": train_weights,
        "train_weights_norm": (train_weights - weight_mean) / weight_std,
        "train_pattern_names": [
            str(record["pattern"]) for record in train_records
        ],
        "development_signatures": development_signatures,
        "development_signatures_norm": (development_signatures - sig_mean) / sig_std,
        "development_weights": development_weights,
        "sig_mean": sig_mean,
        "sig_std": sig_std,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
    }


def clamp_std(std: torch.Tensor) -> torch.Tensor:
    return torch.where(std < 1e-6, torch.ones_like(std), std)


def train_decoder(
    args: argparse.Namespace,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
) -> tuple[SignatureToWeightsDecoder, List[Dict], Dict]:
    torch.manual_seed(args.seed)
    model = SignatureToWeightsDecoder()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    n_train = int(data["train_signatures_norm"].shape[0])
    generator = torch.Generator().manual_seed(args.seed)
    history: List[Dict] = []
    best_checkpoint: Dict | None = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n_train, generator=generator)
        model.train()
        for start in range(0, n_train, args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            sig_batch = data["train_signatures_norm"][batch_idx]
            weight_batch = data["train_weights_norm"][batch_idx]
            target_patterns = [
                PATTERNS.index(data["train_patterns"][int(index)])
                for index in batch_idx.tolist()
            ] if "train_patterns" in data else None
            del target_patterns
            pred_norm = model(sig_batch)
            pred_weights = denormalize_weights(pred_norm, data)
            recon = F.mse_loss(pred_norm, weight_batch)
            behavior = batch_behavior_loss(
                pred_weights,
                [
                    data["train_pattern_names"][int(index)]
                    for index in batch_idx.tolist()
                ],
                suite,
            )
            loss = recon + 0.2 * behavior["bce"] + 0.5 * behavior["margin_hinge"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            dev_eval = evaluate_development(
                model=model,
                args=args,
                data=data,
                suite=suite,
                train_records=train_records,
                development_records=development_records,
            )
            dev_metrics = checkpoint_metrics(dev_eval)
            history.append({
                "epoch": epoch,
                "aggregate": dev_eval["aggregate"],
                "by_behavior": dev_eval["by_behavior"],
                **dev_metrics,
            })
            candidate = {
                "epoch": epoch,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                **dev_metrics,
            }
            if best_checkpoint is None or checkpoint_is_better(candidate, best_checkpoint):
                best_checkpoint = candidate
    if best_checkpoint is None:
        raise RuntimeError("no checkpoint evaluated")
    return model, history, best_checkpoint


def denormalize_weights(pred_norm: torch.Tensor, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return pred_norm * data["weight_std"] + data["weight_mean"]


def batch_behavior_loss(
    flat_weights: torch.Tensor,
    patterns: Sequence[str],
    suite: Mapping,
) -> Dict[str, torch.Tensor]:
    bces = []
    hinges = []
    for index, pattern in enumerate(patterns):
        positive = sequence_tensor(suite["support"][pattern]["positive"])
        negative = sequence_tensor(suite["support"][pattern]["negative"])
        inputs = torch.cat([positive, negative], dim=0)
        labels = torch.cat([
            torch.ones(len(positive)),
            torch.zeros(len(negative)),
        ])
        logits = subject_forward_flat_batch(flat_weights[index:index + 1], inputs)[0]
        bces.append(F.binary_cross_entropy_with_logits(logits, labels))
        pos_prob = torch.sigmoid(logits[:len(positive)]).mean()
        neg_prob = torch.sigmoid(logits[len(positive):]).mean()
        hinges.append(F.relu(torch.tensor(0.30) - (pos_prob - neg_prob)))
    return {
        "bce": torch.stack(bces).mean(),
        "margin_hinge": torch.stack(hinges).mean(),
    }


def sequence_tensor(sequences: Sequence[Sequence[int]]) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.float32)


def subject_forward_flat_batch(flat_weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Run SubjectNetwork forward for a batch of flat weights and shared inputs."""
    x = inputs.unsqueeze(0).expand(flat_weights.shape[0], -1, -1)
    offset = 0
    shapes = [
        (8, 5),
        (8, 8),
        (8, 8),
        (8, 8),
        (8, 8),
    ]
    for out_dim, in_dim in shapes:
        size = out_dim * in_dim
        weight = flat_weights[:, offset:offset + size].view(-1, out_dim, in_dim)
        offset += size
        bias = flat_weights[:, offset:offset + out_dim].view(-1, out_dim)
        offset += out_dim
        x = torch.einsum("bni,boi->bno", x, weight) + bias.unsqueeze(1)
        x = F.gelu(x)
    weight = flat_weights[:, offset:offset + 8].view(-1, 1, 8)
    offset += 8
    bias = flat_weights[:, offset:offset + 1].view(-1, 1)
    output = torch.einsum("bni,boi->bno", x, weight) + bias.unsqueeze(1)
    return output.squeeze(-1).squeeze(-1)


def checkpoint_metrics(dev_eval: Mapping) -> Dict:
    aggregate = dev_eval["aggregate"]
    score = (
        aggregate["mean_matched_minus_best_control_target_margin"]
        + 0.1 * aggregate["mean_best_control_minus_matched_subject_output_mse"]
    )
    return {
        "development_reconstruction_mse": aggregate["mean_matched_reconstruction_mse"],
        "development_score": float(score),
        "mean_development_best_control_minus_matched_subject_output_mse": aggregate[
            "mean_best_control_minus_matched_subject_output_mse"
        ],
        "mean_development_matched_minus_best_control_target_margin": aggregate[
            "mean_matched_minus_best_control_target_margin"
        ],
        "mean_development_matched_target_margin": aggregate["mean_matched_target_margin"],
    }


def checkpoint_is_better(candidate: Mapping, incumbent: Mapping) -> bool:
    if candidate["development_score"] != incumbent["development_score"]:
        return candidate["development_score"] > incumbent["development_score"]
    if candidate["development_reconstruction_mse"] != incumbent["development_reconstruction_mse"]:
        return candidate["development_reconstruction_mse"] < incumbent[
            "development_reconstruction_mse"
        ]
    return candidate["epoch"] < incumbent["epoch"]


def evaluate_development(
    model: SignatureToWeightsDecoder,
    args: argparse.Namespace,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
) -> Dict:
    with torch.no_grad():
        matched_norm = model(data["development_signatures_norm"])
        matched_weights = denormalize_weights(matched_norm, data)
    train_by_behavior = {
        pattern: [record for record in train_records if record["pattern"] == pattern]
        for pattern in PATTERNS
    }
    train_sig = data["train_signatures"]
    train_pattern_names = data["train_pattern_names"]
    global_centroid = train_sig.mean(dim=0)
    centroids = {
        pattern: train_sig[
            torch.tensor([name == pattern for name in train_pattern_names], dtype=torch.bool)
        ].mean(dim=0)
        for pattern in PATTERNS
    }
    records = []
    for index, record in enumerate(development_records):
        pattern = record["pattern"]
        source_weights = data["development_weights"][index]
        matched = matched_weights[index]
        subject_inputs = heldout_inputs_for(pattern, suite)
        matched_record = {
            "subject_id": record["subject_id"],
            "pattern": pattern,
            "matched_reconstruction_mse": float(
                F.mse_loss(matched_norm[index], (
                    source_weights - data["weight_mean"]
                ) / data["weight_std"]).item()
            ),
            "matched_subject_output_mse": subject_output_mse(
                matched,
                source_weights,
                subject_inputs,
            ),
            "matched_target_margin": target_margin(matched, pattern, suite),
        }
        controls = build_controls_for_subject(
            args=args,
            model=model,
            data=data,
            suite=suite,
            pattern=pattern,
            development_record=record,
            source_weights=source_weights,
            train_by_behavior=train_by_behavior,
            global_centroid=global_centroid,
            centroids=centroids,
            subject_inputs=subject_inputs,
        )
        best = best_control_metrics(controls)
        matched_record.update(best)
        matched_record["matched_minus_best_control_target_margin"] = (
            matched_record["matched_target_margin"] - best["best_target_margin"]
        )
        matched_record["best_control_minus_matched_subject_output_mse"] = (
            best["best_subject_output_mse"] - matched_record["matched_subject_output_mse"]
        )
        matched_record["individual_passed"] = individual_passed(matched_record)
        matched_record["controls"] = controls
        records.append(matched_record)
    return summarize_development_records(records)


def heldout_inputs_for(pattern: str, suite: Mapping) -> torch.Tensor:
    return torch.cat([
        sequence_tensor(suite["heldout"][pattern]["positive"]),
        sequence_tensor(suite["heldout"][pattern]["negative"]),
    ])


def build_controls_for_subject(
    args: argparse.Namespace,
    model: SignatureToWeightsDecoder,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    pattern: str,
    development_record: Mapping,
    source_weights: torch.Tensor,
    train_by_behavior: Mapping[str, Sequence[Mapping]],
    global_centroid: torch.Tensor,
    centroids: Mapping[str, torch.Tensor],
    subject_inputs: torch.Tensor,
) -> List[Dict]:
    raw_signatures = [
        ("null_signature", "null", torch.zeros_like(global_centroid)),
        ("train_global_centroid", "global", global_centroid),
        ("same_label_train_centroid", pattern, centroids[pattern]),
    ]
    for other in PATTERNS:
        if other != pattern:
            raw_signatures.append((f"other_label_train_centroid:{other}", other, centroids[other]))
    same = select_train_control(
        train_by_behavior[pattern],
        development_subject_id=development_record["subject_id"],
        control_family="same_label_other_subject",
        control_behavior=pattern,
    )
    raw_signatures.append((
        "same_label_other_subject",
        pattern,
        torch.tensor(same["signature"], dtype=torch.float32),
    ))
    for other in PATTERNS:
        if other == pattern:
            continue
        selected = select_train_control(
            train_by_behavior[other],
            development_subject_id=development_record["subject_id"],
            control_family="different_label_other_subject",
            control_behavior=other,
        )
        raw_signatures.append((
            f"different_label_other_subject:{other}",
            other,
            torch.tensor(selected["signature"], dtype=torch.float32),
        ))

    controls = [
        evaluate_control_signature(
            model=model,
            data=data,
            suite=suite,
            source_weights=source_weights,
            subject_inputs=subject_inputs,
            target_pattern=pattern,
            control_type=control_type,
            control_behavior=control_behavior,
            raw_signature=raw_signature,
        )
        for control_type, control_behavior, raw_signature in raw_signatures
    ]
    controls.extend(
        build_noise_controls(
            model=model,
            args=args,
            data=data,
            suite=suite,
            source_weights=source_weights,
            subject_inputs=subject_inputs,
            target_pattern=pattern,
            development_subject_id=development_record["subject_id"],
        )
    )
    return controls


def evaluate_control_signature(
    model: SignatureToWeightsDecoder,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    source_weights: torch.Tensor,
    subject_inputs: torch.Tensor,
    target_pattern: str,
    control_type: str,
    control_behavior: str,
    raw_signature: torch.Tensor,
) -> Dict:
    with torch.no_grad():
        norm = (raw_signature - data["sig_mean"]) / data["sig_std"]
        weights = denormalize_weights(model(norm.unsqueeze(0)).squeeze(0), data)
    return {
        "control_behavior": control_behavior,
        "control_type": control_type,
        "subject_output_mse": subject_output_mse(weights, source_weights, subject_inputs),
        "target_margin": target_margin(weights, target_pattern, suite),
    }


def build_noise_controls(
    model: SignatureToWeightsDecoder,
    args: argparse.Namespace,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    source_weights: torch.Tensor,
    subject_inputs: torch.Tensor,
    target_pattern: str,
    development_subject_id: str,
) -> List[Dict]:
    seed_hash = stable_hash_json([development_subject_id, "noise", args.noise_base_seed])
    seed = int(seed_hash[:16], 16) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    controls = []
    for index in range(32):
        norm = torch.randn(560, generator=generator)
        with torch.no_grad():
            weights = denormalize_weights(model(norm.unsqueeze(0)).squeeze(0), data)
        controls.append({
            "control_behavior": "noise",
            "control_type": f"noise_signature:{index:02d}",
            "noise_index": index,
            "noise_seed": int(seed),
            "subject_output_mse": subject_output_mse(weights, source_weights, subject_inputs),
            "target_margin": target_margin(weights, target_pattern, suite),
        })
    return controls


def select_train_control(
    records: Sequence[Mapping],
    development_subject_id: str,
    control_family: str,
    control_behavior: str,
) -> Mapping:
    if not records:
        raise ValueError(f"no train control candidates for {control_behavior}")
    candidates = sorted(
        records,
        key=lambda record: (
            record["weights_hash"],
            record["signature_hash"],
            record["subject_id"],
        ),
    )
    key = stable_hash_json([
        development_subject_id,
        control_family,
        control_behavior,
    ])
    index = int(key[:16], 16) % len(candidates)
    return candidates[index]


def target_margin(weights: torch.Tensor, pattern: str, suite: Mapping) -> float:
    positive = sequence_tensor(suite["heldout"][pattern]["positive"])
    negative = sequence_tensor(suite["heldout"][pattern]["negative"])
    with torch.no_grad():
        pos = torch.sigmoid(subject_forward_flat_batch(weights.unsqueeze(0), positive)[0]).mean()
        neg = torch.sigmoid(subject_forward_flat_batch(weights.unsqueeze(0), negative)[0]).mean()
    return float((pos - neg).item())


def subject_output_mse(
    weights: torch.Tensor,
    source_weights: torch.Tensor,
    inputs: torch.Tensor,
) -> float:
    with torch.no_grad():
        decoded = subject_forward_flat_batch(weights.unsqueeze(0), inputs)[0]
        source = subject_forward_flat_batch(source_weights.unsqueeze(0), inputs)[0]
    return float(F.mse_loss(decoded, source).item())


def best_control_metrics(controls: Sequence[Mapping]) -> Dict:
    if not controls:
        raise ValueError("controls must not be empty")
    best_margin = max(controls, key=lambda record: float(record["target_margin"]))
    best_mse = min(controls, key=lambda record: float(record["subject_output_mse"]))
    return {
        "best_target_margin": float(best_margin["target_margin"]),
        "best_target_margin_control_type": best_margin["control_type"],
        "best_subject_output_mse": float(best_mse["subject_output_mse"]),
        "best_subject_output_mse_control_type": best_mse["control_type"],
    }


def individual_passed(record: Mapping) -> bool:
    return (
        record["matched_target_margin"]
        >= THRESHOLDS["min_individual_matched_target_margin"]
        and record["matched_minus_best_control_target_margin"]
        >= THRESHOLDS["min_individual_matched_minus_best_control_target_margin"]
        and record["best_control_minus_matched_subject_output_mse"]
        > THRESHOLDS["min_individual_best_control_minus_matched_subject_output_mse"]
    )


def summarize_development_records(records: Sequence[Mapping]) -> Dict:
    aggregate = summarize_metric_records(records)
    by_behavior = {
        pattern: summarize_metric_records([
            record for record in records if record["pattern"] == pattern
        ])
        for pattern in PATTERNS
    }
    failures = []
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
        "passed": not failures,
    }


def summarize_metric_records(records: Sequence[Mapping]) -> Dict:
    n = len(records)
    passed = sum(1 for record in records if record["individual_passed"])
    return {
        "individual_all_gate_pass_count": int(passed),
        "individual_all_gate_pass_rate": float(passed / n) if n else 0.0,
        "mean_best_control_minus_matched_subject_output_mse": mean(
            record["best_control_minus_matched_subject_output_mse"]
            for record in records
        ),
        "mean_matched_minus_best_control_target_margin": mean(
            record["matched_minus_best_control_target_margin"]
            for record in records
        ),
        "mean_matched_subject_output_mse": mean(
            record["matched_subject_output_mse"]
            for record in records
        ),
        "mean_matched_reconstruction_mse": mean(
            record["matched_reconstruction_mse"]
            for record in records
        ),
        "mean_matched_target_margin": mean(
            record["matched_target_margin"]
            for record in records
        ),
        "n": int(n),
    }


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
    training_history: Sequence[Mapping],
    best_checkpoint: Mapping,
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
        "best_epoch": int(best_checkpoint["epoch"]),
        "by_behavior": eval_result["by_behavior"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "claim_scope": "four_behavior_decoder_development_not_final_proof",
        "combined_audit_hash": stable_hash_json(combined_audit),
        "config": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "eval_every": int(args.eval_every),
            "lr": float(args.lr),
            "noise_base_seed": int(args.noise_base_seed),
            "seed": int(args.seed),
            "weight_decay": float(args.weight_decay),
        },
        "development_status": "train_development_only_final_pool_sealed",
        "development_records": eval_result["development_records"],
        "development_subject_counts_by_behavior": count_by_behavior(development_records),
        "evidence_interpretation": (
            "development eligibility only; not stored-probe final decoder proof"
        ),
        "failures": failures,
        "final_redacted_audit_hash": stable_hash_json(final_redacted_audit),
        "input_path_audit": input_audit,
        "normalization_hashes": {
            "sig_mean": tensor_hash(data["sig_mean"]),
            "sig_std": tensor_hash(data["sig_std"]),
            "weight_mean": tensor_hash(data["weight_mean"]),
            "weight_std": tensor_hash(data["weight_std"]),
        },
        "overlap_counts": train_development_overlap_counts(train_records, development_records),
        "passed": not failures,
        "probe_examples_hash": stable_hash_json(probe_examples),
        "source_pool_hashes": {
            "development": combined_audit["pool_file_sha256"]["development"],
            "train": combined_audit["pool_file_sha256"]["train"],
        },
        "suite_hashes": {
            "heldout": suite["metadata"]["heldout_hash"],
            "support": suite["metadata"]["support_hash"],
        },
        "thresholds": THRESHOLDS,
        "train_subject_counts_by_behavior": count_by_behavior(train_records),
        "training_history": training_history,
    }


def count_by_behavior(records: Sequence[Mapping]) -> Dict[str, int]:
    return {
        pattern: sum(1 for record in records if record["pattern"] == pattern)
        for pattern in PATTERNS
    }


def train_development_overlap_counts(
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
) -> Dict[str, int]:
    fields = {
        "seed": "seed",
        "signature_hash": "signature_hash",
        "subject_id": "subject_id",
        "weights_hash": "weights_hash",
    }
    return {
        field_name: len(
            {record[field] for record in train_records}
            & {record[field] for record in development_records}
        )
        for field_name, field in fields.items()
    }


def tensor_hash(tensor: torch.Tensor) -> str:
    return stable_hash_json([float(value) for value in tensor.detach().cpu().view(-1).tolist()])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
