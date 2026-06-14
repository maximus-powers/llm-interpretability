"""Train robust normalized-signature edit vectors without using proof holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_stored_probe_steering_robustness as robust  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/stored_probe_signature_edit_vectors_v2_robust_development",
    )
    parser.add_argument(
        "--train-pool-dir",
        default="runs/fresh_robust_edit_v2_train_pool",
    )
    parser.add_argument("--decoder-path", default="runs/stored_probe_functional_decoder_v2_adaptive/model.pt")
    parser.add_argument(
        "--init-edit-vectors-path",
        default="runs/stored_probe_signature_edit_vectors_v1_development/edit_vectors.pt",
    )
    parser.add_argument("--n-per-behavior", type=int, default=40)
    parser.add_argument("--train-per-behavior", type=int, default=32)
    parser.add_argument("--pool-base-seed", type=int, default=20260730)
    parser.add_argument("--subject-train-epochs", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--random-train-controls", type=int, default=2)
    parser.add_argument("--eval-random-controls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--force-regenerate-train-pool", action="store_true")
    return parser.parse_args()


def flat_forward(flat_weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if flat_weights.dim() == 1:
        flat_weights = flat_weights.unsqueeze(0)
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0).expand(flat_weights.size(0), -1, -1)
    batch_size = flat_weights.size(0)
    x = inputs.to(flat_weights.device)
    offset = 0
    for layer_idx in range(5):
        in_dim = 5 if layer_idx == 0 else 8
        out_dim = 8
        weight_count = out_dim * in_dim
        weight = flat_weights[:, offset:offset + weight_count].view(batch_size, out_dim, in_dim)
        offset += weight_count
        bias = flat_weights[:, offset:offset + out_dim]
        offset += out_dim
        x = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
        x = F.gelu(x)
    weight = flat_weights[:, offset:offset + 8].view(batch_size, 1, 8)
    offset += 8
    bias = flat_weights[:, offset:offset + 1]
    offset += 1
    if offset != flat_weights.size(1):
        raise ValueError(f"Expected {offset} flat weights, got {flat_weights.size(1)}")
    return (torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)).squeeze(-1)


def differentiable_margin(
    flat_weights: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    pos = torch.sigmoid(flat_forward(flat_weights, positive)).mean(dim=1)
    neg = torch.sigmoid(flat_forward(flat_weights, negative)).mean(dim=1)
    return pos - neg


def decode_conditions(
    decoder: torch.nn.Module,
    checkpoint: Mapping,
    conditions: torch.Tensor,
) -> torch.Tensor:
    weights_norm = decoder(conditions.float())
    return weights_norm * checkpoint["weight_std"] + checkpoint["weight_mean"]


def split_subjects(subjects: Sequence[Mapping], train_per_behavior: int) -> tuple[list, list]:
    train = []
    val = []
    for pattern in robust.PATTERNS:
        pattern_subjects = [
            subject for subject in subjects
            if subject["target_pattern"] == pattern
        ]
        train.extend(pattern_subjects[:train_per_behavior])
        val.extend(pattern_subjects[train_per_behavior:])
    return train, val


def build_condition_tensor(subjects: Sequence[Mapping], checkpoint: Mapping) -> torch.Tensor:
    signatures = torch.tensor(
        [subject["signature"] for subject in subjects],
        dtype=torch.float32,
    )
    return (signatures - checkpoint["sig_mean"]) / checkpoint["sig_std"]


def direction_loss(
    direction_key: str,
    vector: torch.Tensor,
    conditions: torch.Tensor,
    source_pattern: str,
    target_pattern: str,
    no_edit_source_margin: torch.Tensor,
    no_edit_target_margin: torch.Tensor,
    decoder: torch.nn.Module,
    checkpoint: Mapping,
    suite: Mapping,
    generator: torch.Generator,
    random_controls: int,
) -> tuple[torch.Tensor, Dict[str, float]]:
    target_pos = torch.tensor(suite["support"][target_pattern]["positive"], dtype=torch.float32)
    target_neg = torch.tensor(suite["support"][target_pattern]["negative"], dtype=torch.float32)
    source_pos = torch.tensor(suite["support"][source_pattern]["positive"], dtype=torch.float32)
    source_neg = torch.tensor(suite["support"][source_pattern]["negative"], dtype=torch.float32)

    steered_weights = decode_conditions(decoder, checkpoint, conditions + vector)
    target_margin = differentiable_margin(steered_weights, target_pos, target_neg)
    source_margin = differentiable_margin(steered_weights, source_pos, source_neg)

    random_target_margins = []
    for _ in range(random_controls):
        random_vector = torch.randn(
            vector.shape,
            generator=generator,
            dtype=vector.dtype,
        )
        random_vector = random_vector / random_vector.norm().clamp_min(1e-12)
        random_vector = random_vector * vector.detach().norm()
        random_weights = decode_conditions(decoder, checkpoint, conditions + random_vector)
        random_target_margins.append(
            differentiable_margin(random_weights, target_pos, target_neg).detach()
        )
    worst_random_target = torch.stack(random_target_margins, dim=0).max(dim=0).values

    target_loss = F.relu(0.35 - target_margin).mean()
    improvement_loss = F.relu(0.30 - (target_margin - no_edit_target_margin)).mean()
    random_delta_loss = F.relu(0.25 - (target_margin - worst_random_target)).mean()
    source_ceiling_loss = F.relu(source_margin - 0.0).mean()
    source_change_loss = F.relu(source_margin - no_edit_source_margin + 0.10).mean()
    l2_loss = vector.pow(2).mean()
    loss = (
        2.0 * target_loss
        + improvement_loss
        + random_delta_loss
        + 1.5 * source_ceiling_loss
        + source_change_loss
        + 0.0001 * l2_loss
    )
    return loss, {
        f"{direction_key}/target_margin": float(target_margin.mean().item()),
        f"{direction_key}/source_margin": float(source_margin.mean().item()),
        f"{direction_key}/target_loss": float(target_loss.item()),
        f"{direction_key}/improvement_loss": float(improvement_loss.item()),
        f"{direction_key}/random_delta_loss": float(random_delta_loss.item()),
        f"{direction_key}/source_ceiling_loss": float(source_ceiling_loss.item()),
        f"{direction_key}/source_change_loss": float(source_change_loss.item()),
        f"{direction_key}/norm": float(vector.detach().norm().item()),
    }


def compute_no_edit_margins(
    subjects: Sequence[Mapping],
    decoder: torch.nn.Module,
    checkpoint: Mapping,
    suite: Mapping,
) -> Dict[str, Dict[str, torch.Tensor]]:
    result = {}
    for source in robust.PATTERNS:
        target = robust.OPPOSITE[source]
        selected = [subject for subject in subjects if subject["target_pattern"] == source]
        conditions = build_condition_tensor(selected, checkpoint)
        weights = decode_conditions(decoder, checkpoint, conditions).detach()
        source_pos = torch.tensor(suite["support"][source]["positive"], dtype=torch.float32)
        source_neg = torch.tensor(suite["support"][source]["negative"], dtype=torch.float32)
        target_pos = torch.tensor(suite["support"][target]["positive"], dtype=torch.float32)
        target_neg = torch.tensor(suite["support"][target]["negative"], dtype=torch.float32)
        result[source] = {
            "source": differentiable_margin(weights, source_pos, source_neg).detach(),
            "target": differentiable_margin(weights, target_pos, target_neg).detach(),
        }
    return result


def evaluate_pool(
    args: argparse.Namespace,
    subjects: Sequence[Mapping],
    pool_summary: Mapping,
    edit_vectors_path: Path,
    output_dir: Path,
    name: str,
) -> Dict:
    eval_args = SimpleNamespace(
        decoder_path=args.decoder_path,
        edit_vectors_path=str(edit_vectors_path),
        eval_seed=args.seed + 500,
        random_controls=args.eval_random_controls,
    )
    summary = dict(pool_summary)
    summary["subjects_path"] = f"{pool_summary['subjects_path']}::{name}"
    summary["subjects_sha256"] = pool_summary["subjects_sha256"]
    result = robust.evaluate(eval_args, {"subjects": list(subjects)}, summary)
    path = output_dir / f"{name}_results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = robust.build_suite()
    pool_args = SimpleNamespace(
        holdout_dir=args.train_pool_dir,
        force_regenerate_holdout=args.force_regenerate_train_pool,
        n_per_behavior=args.n_per_behavior,
        base_seed=args.pool_base_seed,
        train_epochs=args.subject_train_epochs,
        lr=0.003,
        source_margin_gate=0.40,
    )
    pool_payload, pool_summary = robust.load_or_generate_holdout(pool_args, suite)
    train_subjects, val_subjects = split_subjects(
        pool_payload["subjects"],
        args.train_per_behavior,
    )

    decoder, checkpoint = robust.load_decoder(Path(args.decoder_path))
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    decoder.eval()

    init_payload = torch.load(args.init_edit_vectors_path, map_location="cpu")
    vectors = torch.nn.ParameterDict({
        key: torch.nn.Parameter(value.detach().clone().float())
        for key, value in init_payload["edit_vectors"].items()
    })
    optimizer = torch.optim.AdamW(
        vectors.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    no_edit = compute_no_edit_margins(train_subjects, decoder, checkpoint, suite)
    train_conditions = {
        source: build_condition_tensor(
            [subject for subject in train_subjects if subject["target_pattern"] == source],
            checkpoint,
        )
        for source in robust.PATTERNS
    }

    history = []
    best = {"epoch": None, "loss": float("inf"), "state": None}
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        metrics = {}
        for source in robust.PATTERNS:
            target = robust.OPPOSITE[source]
            key = f"{source}_to_{target}"
            loss, direction_metrics = direction_loss(
                key,
                vectors[key],
                train_conditions[source],
                source,
                target,
                no_edit[source]["source"],
                no_edit[source]["target"],
                decoder,
                checkpoint,
                suite,
                generator,
                args.random_train_controls,
            )
            losses.append(loss)
            metrics.update(direction_metrics)
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        optimizer.step()
        entry = {"epoch": epoch, "train_loss": float(total_loss.item()), **metrics}
        history.append(entry)
        if total_loss.item() < best["loss"]:
            best = {
                "epoch": epoch,
                "loss": float(total_loss.item()),
                "state": {
                    key: value.detach().clone()
                    for key, value in vectors.items()
                },
            }

    if best["state"] is None:
        raise RuntimeError("No best edit-vector state recorded")
    edit_vectors_path = output_dir / "edit_vectors.pt"
    torch.save({
        "edit_vectors": best["state"],
        "history": history,
        "best_epoch": best["epoch"],
        "best_train_loss": best["loss"],
        "seed": args.seed,
        "training_pool_summary": pool_summary,
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(val_subjects),
        "objective": {
            "target_margin_hinge": 0.35,
            "target_improvement_hinge": 0.30,
            "random_delta_hinge": 0.25,
            "source_margin_ceiling": 0.0,
            "source_change_hinge": -0.10,
            "random_train_controls": args.random_train_controls,
        },
    }, edit_vectors_path)

    train_result = evaluate_pool(
        args,
        train_subjects,
        pool_summary,
        edit_vectors_path,
        output_dir,
        "train_pool",
    )
    val_result = evaluate_pool(
        args,
        val_subjects,
        pool_summary,
        edit_vectors_path,
        output_dir,
        "validation_pool",
    )
    result = {
        "best_epoch": best["epoch"],
        "best_train_loss": best["loss"],
        "decoder_path": args.decoder_path,
        "edit_vectors_path": str(edit_vectors_path),
        "init_edit_vectors_path": args.init_edit_vectors_path,
        "passed": bool(train_result["passed"] and val_result["passed"]),
        "train_pool_result_path": str(output_dir / "train_pool_results.json"),
        "validation_pool_result_path": str(output_dir / "validation_pool_results.json"),
        "train_pool_passed": train_result["passed"],
        "validation_pool_passed": val_result["passed"],
        "train_pool_failures": train_result["failures"],
        "validation_pool_failures": val_result["failures"],
        "train_pool_summary": {
            "aggregate": train_result["aggregate"],
            "individual_gate_audit": train_result["individual_gate_audit"],
        },
        "validation_pool_summary": {
            "aggregate": val_result["aggregate"],
            "individual_gate_audit": val_result["individual_gate_audit"],
        },
        "training_pool_subjects_sha256": pool_summary["subjects_sha256"],
        "training_pool_subjects_path": pool_summary["subjects_path"],
        "history_tail": history[-10:],
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
