"""Train/evaluate four-behavior decoder development V2 with functional distillation."""

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
SCRIPT_ROOT = MODEL_ZOO_ROOT / "scripts"
sys.path.insert(0, str(MODEL_ZOO_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from hypernet.behavior_suite import (  # noqa: E402
    build_clean_behavior_suite,
    enumerate_sequence_universe,
)
from hypernet.dataset_provenance import stable_hash_json  # noqa: E402
from hypernet.paired_contrast import build_digit_probe_examples  # noqa: E402
import train_four_behavior_decoder_development as v1  # noqa: E402


PATTERNS = v1.PATTERNS
THRESHOLDS = v1.THRESHOLDS
FORBIDDEN_FINAL_RAW_NAME = v1.FORBIDDEN_FINAL_RAW_NAME
SEALED_FINAL_RAW_PATH = v1.SEALED_FINAL_RAW_PATH

assert_no_final_raw_paths = v1.assert_no_final_raw_paths
best_control_metrics = v1.best_control_metrics
select_train_control = v1.select_train_control


class SignatureToWeightsDecoderV2(nn.Module):
    """Larger direct MLP decoder for V2 functional distillation."""

    def __init__(self, input_dim: int = 560, output_dim: int = 345):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in (1024, 1024, 1024, 512):
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
        default="runs/four_behavior_decoder_development_v2",
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
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--distillation-cases-per-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--noise-base-seed", type=int, default=20260615)
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
    distillation_cases = build_distillation_cases(suite)
    distillation_inputs = v1.sequence_tensor(distillation_cases)
    data = v1.build_tensors(train_records, development_records)
    distillation_logits = v1.subject_forward_flat_batch(
        data["train_weights"],
        distillation_inputs,
    ).detach()

    model, training_history, best_checkpoint = train_decoder(
        args=args,
        data=data,
        suite=suite,
        train_records=train_records,
        development_records=development_records,
        distillation_inputs=distillation_inputs,
        distillation_logits=distillation_logits,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    eval_result = v1.evaluate_development(
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
                "hidden_layers": [1024, 1024, 1024, 512],
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
        distillation_cases=distillation_cases,
        distillation_logits=distillation_logits,
        training_history=training_history,
        best_checkpoint=best_checkpoint,
        eval_result=eval_result,
        checkpoint_path=checkpoint_path,
    )
    result["result_payload_sha256"] = stable_hash_json(result)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "aggregate": result["aggregate"],
        "best_epoch": result["best_epoch"],
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


def build_distillation_cases(suite: Mapping) -> List[List[int]]:
    heldout = {
        tuple(seq)
        for pattern in PATTERNS
        for split_name in ("positive", "negative")
        for seq in suite["heldout"][pattern][split_name]
    }
    candidates = [
        seq for seq in enumerate_sequence_universe(seq_len=5, base=10)
        if seq not in heldout
    ]
    rng = random.Random(20260613)
    rng.shuffle(candidates)
    return [[int(value) for value in seq] for seq in candidates[:4096]]


def hash_sequences(sequences: Sequence[Sequence[int]]) -> str:
    return stable_hash_json([
        [int(value) for value in sequence]
        for sequence in sequences
    ])


def train_decoder(
    args: argparse.Namespace,
    data: Mapping[str, torch.Tensor],
    suite: Mapping,
    train_records: Sequence[Mapping],
    development_records: Sequence[Mapping],
    distillation_inputs: torch.Tensor,
    distillation_logits: torch.Tensor,
) -> tuple[SignatureToWeightsDecoderV2, List[Dict], Dict]:
    torch.manual_seed(args.seed)
    model = SignatureToWeightsDecoderV2()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    n_train = int(data["train_signatures_norm"].shape[0])
    n_cases = int(distillation_inputs.shape[0])
    generator = torch.Generator().manual_seed(args.seed)
    history: List[Dict] = []
    best_checkpoint: Dict | None = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n_train, generator=generator)
        model.train()
        for start in range(0, n_train, args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            case_idx = torch.randperm(n_cases, generator=generator)[
                : args.distillation_cases_per_batch
            ]
            sig_batch = data["train_signatures_norm"][batch_idx]
            weight_batch = data["train_weights_norm"][batch_idx]
            pred_norm = model(sig_batch)
            pred_weights = v1.denormalize_weights(pred_norm, data)
            recon = F.mse_loss(pred_norm, weight_batch)
            behavior = batch_behavior_loss_v2(
                pred_weights,
                [
                    data["train_pattern_names"][int(index)]
                    for index in batch_idx.tolist()
                ],
                suite,
            )
            pred_logits = v1.subject_forward_flat_batch(
                pred_weights,
                distillation_inputs[case_idx],
            )
            target_logits = distillation_logits[batch_idx][:, case_idx]
            distill = F.mse_loss(pred_logits, target_logits)
            loss = distill + 0.5 * behavior["bce"] + behavior["margin_hinge"] + 0.05 * recon
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            dev_eval = v1.evaluate_development(
                model=model,
                args=args,
                data=data,
                suite=suite,
                train_records=train_records,
                development_records=development_records,
            )
            dev_metrics = checkpoint_metrics_v2(dev_eval)
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
            if best_checkpoint is None or v1.checkpoint_is_better(candidate, best_checkpoint):
                best_checkpoint = candidate
    if best_checkpoint is None:
        raise RuntimeError("no checkpoint evaluated")
    return model, history, best_checkpoint


def batch_behavior_loss_v2(
    flat_weights: torch.Tensor,
    patterns: Sequence[str],
    suite: Mapping,
) -> Dict[str, torch.Tensor]:
    bces = []
    hinges = []
    for index, pattern in enumerate(patterns):
        positive = v1.sequence_tensor(suite["support"][pattern]["positive"])
        negative = v1.sequence_tensor(suite["support"][pattern]["negative"])
        inputs = torch.cat([positive, negative], dim=0)
        labels = torch.cat([
            torch.ones(len(positive)),
            torch.zeros(len(negative)),
        ])
        logits = v1.subject_forward_flat_batch(flat_weights[index:index + 1], inputs)[0]
        bces.append(F.binary_cross_entropy_with_logits(logits, labels))
        pos_prob = torch.sigmoid(logits[:len(positive)]).mean()
        neg_prob = torch.sigmoid(logits[len(positive):]).mean()
        hinges.append(F.relu(torch.tensor(0.40) - (pos_prob - neg_prob)))
    return {
        "bce": torch.stack(bces).mean(),
        "margin_hinge": torch.stack(hinges).mean(),
    }


def checkpoint_metrics_v2(dev_eval: Mapping) -> Dict:
    metrics = v1.checkpoint_metrics(dev_eval)
    return {
        **metrics,
        "selection_rule": (
            "mean_matched_minus_best_control_target_margin + "
            "0.1 * mean_best_control_minus_matched_subject_output_mse"
        ),
    }


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
    distillation_cases: Sequence[Sequence[int]],
    distillation_logits: torch.Tensor,
    training_history: Sequence[Mapping],
    best_checkpoint: Mapping,
    eval_result: Mapping,
    checkpoint_path: Path,
) -> Dict:
    result = v1.build_result(
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
    result["claim_scope"] = "four_behavior_decoder_development_v2_not_final_proof"
    result["development_status"] = "adaptive_v2_train_development_only_final_pool_sealed"
    result["config"] = {
        "batch_size": int(args.batch_size),
        "distillation_cases_per_batch": int(args.distillation_cases_per_batch),
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "lr": float(args.lr),
        "noise_base_seed": int(args.noise_base_seed),
        "seed": int(args.seed),
        "weight_decay": float(args.weight_decay),
        "loss_weights": {
            "distillation_mse": 1.0,
            "reconstruction_mse": 0.05,
            "support_bce": 0.5,
            "support_margin_hinge": 1.0,
        },
        "support_margin_target": 0.40,
    }
    result["distillation_case_hash"] = hash_sequences(distillation_cases)
    result["distillation_case_count"] = len(distillation_cases)
    result["train_distillation_logits_hash"] = tensor_hash(distillation_logits)
    result["evidence_interpretation"] = (
        "adaptive V2 development eligibility only; not stored-probe final decoder proof"
    )
    result["result_text_excludes_final_subjects_json"] = (
        FORBIDDEN_FINAL_RAW_NAME not in json.dumps(result, sort_keys=True)
    )
    if not result["result_text_excludes_final_subjects_json"]:
        result["failures"].append("result artifact names final_subjects.json")
        result["passed"] = False
    return result


def tensor_hash(tensor: torch.Tensor) -> str:
    return stable_hash_json([
        float(value)
        for value in tensor.detach().cpu().reshape(-1).tolist()
    ])


if __name__ == "__main__":
    main()
