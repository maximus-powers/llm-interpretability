import torch
import numpy as np
import logging
from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_reconstruction_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: Dict[str, Any],
):
    if predicted.dim() == 3:
        batch_size, max_tokens, token_dim = predicted.shape
        predicted = predicted.view(-1, token_dim)
        target = target.view(-1, token_dim)
        mask = mask.view(-1)

    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    mask_bool = mask_np > 0
    pred_real = pred_np[mask_bool]
    target_real = target_np[mask_bool]

    if len(pred_real) == 0:
        logger.warning("No real tokens found (all padding). Returning zero metrics.")
        return {
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "cosine_similarity": 0.0,
            "relative_error": 0.0,
            "r2_score": 0.0,
        }

    metrics = {}

    mse = np.mean((pred_real - target_real) ** 2)
    metrics["mse"] = float(mse)
    metrics["mae"] = float(np.mean(np.abs(pred_real - target_real)))
    rmse = np.sqrt(mse)
    metrics["rmse"] = float(rmse)
    
    # Per-token cosine similarity (original method)
    cos_sims = []
    for i in range(len(pred_real)):
        pred_vec = pred_real[i : i + 1]
        target_vec = target_real[i : i + 1]
        pred_norm = np.linalg.norm(pred_vec)
        target_norm = np.linalg.norm(target_vec)
        if pred_norm > 1e-8 and target_norm > 1e-8:
            cos_sim = cosine_similarity(pred_vec, target_vec)[0, 0]
            cos_sims.append(cos_sim)
    metrics["cosine_similarity"] = float(np.mean(cos_sims)) if cos_sims else 0.0
    
    # Batch-wide cosine similarity (flattened, more meaningful for weight reconstruction)
    # This matches the cosine_batch loss and our evaluation target
    pred_flat = pred_real.flatten()
    target_flat = target_real.flatten()
    pred_norm = np.linalg.norm(pred_flat)
    target_norm = np.linalg.norm(target_flat)
    if pred_norm > 1e-8 and target_norm > 1e-8:
        batch_cosine = np.dot(pred_flat, target_flat) / (pred_norm * target_norm)
        metrics["cosine_batch"] = float(batch_cosine)
    else:
        metrics["cosine_batch"] = 0.0
    
    epsilon = 1e-8
    relative_errors = np.abs(pred_real - target_real) / (np.abs(target_real) + epsilon)
    metrics["relative_error"] = float(np.mean(relative_errors))

    ss_res = np.sum((target_real - pred_real) ** 2)
    ss_tot = np.sum((target_real - np.mean(target_real)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    metrics["r2_score"] = float(r2)

    return metrics


def compute_weight_level_metrics(
    reconstructed_weights: Dict[str, torch.Tensor],
    original_weights: Dict[str, torch.Tensor],
):
    """
    Compute per-layer weight reconstruction metrics.
    
    Returns metrics including MSE, MAE, max error, and cosine similarity
    for each layer, plus aggregated statistics across all layers.
    """
    metrics = {}
    layer_metrics = []

    for layer_name in original_weights.keys():
        if layer_name not in reconstructed_weights:
            logger.warning(f"Layer {layer_name} not found in reconstructed weights")
            continue
        orig = original_weights[layer_name].detach().cpu().numpy()
        recon = reconstructed_weights[layer_name].detach().cpu().numpy()
        if orig.shape != recon.shape:
            logger.warning(
                f"Shape mismatch for {layer_name}: orig={orig.shape}, recon={recon.shape}"
            )
            continue
        
        # Existing metrics
        layer_mse = np.mean((orig - recon) ** 2)
        layer_mae = np.mean(np.abs(orig - recon))
        layer_max_error = np.max(np.abs(orig - recon))
        
        # NEW: Per-layer cosine similarity
        orig_flat = orig.flatten()
        recon_flat = recon.flatten()
        orig_norm = np.linalg.norm(orig_flat)
        recon_norm = np.linalg.norm(recon_flat)
        if orig_norm > 1e-8 and recon_norm > 1e-8:
            layer_cosine = float(np.dot(orig_flat, recon_flat) / (orig_norm * recon_norm))
        else:
            layer_cosine = 0.0
        
        layer_metrics.append(
            {
                "layer": layer_name,
                "mse": float(layer_mse),
                "mae": float(layer_mae),
                "max_error": float(layer_max_error),
                "cosine_similarity": layer_cosine,
                "num_params": int(np.prod(orig.shape)),
            }
        )

    if not layer_metrics:
        logger.warning("No valid layers found for weight-level metrics")
        return {"per_layer": []}

    # Aggregate metrics
    metrics["layer_mse_mean"] = np.mean([m["mse"] for m in layer_metrics])
    metrics["layer_mse_std"] = np.std([m["mse"] for m in layer_metrics])
    metrics["layer_mae_mean"] = np.mean([m["mae"] for m in layer_metrics])
    metrics["layer_mae_std"] = np.std([m["mae"] for m in layer_metrics])
    metrics["layer_max_error_mean"] = np.mean([m["max_error"] for m in layer_metrics])
    
    # NEW: Aggregate cosine similarity metrics
    metrics["layer_cosine_mean"] = float(np.mean([m["cosine_similarity"] for m in layer_metrics]))
    metrics["layer_cosine_std"] = float(np.std([m["cosine_similarity"] for m in layer_metrics]))

    metrics["per_layer"] = layer_metrics

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    if prefix:
        logger.info(f"\n{prefix.upper()} Metrics:")
    else:
        logger.info("\nMetrics:")

    main_metrics = [
        "loss",
        "mse",
        "mae",
        "rmse",
        "cosine_similarity",
        "relative_error",
        "r2_score",
    ]

    for metric_name in main_metrics:
        if metric_name in metrics:
            value = metrics[metric_name]
            logger.info(f"  {metric_name}: {value:.6f}")
    if "layer_mse_mean" in metrics:
        logger.info("\n  Layer-Level Metrics:")
        logger.info(
            f"    MSE (mean ± std): {metrics['layer_mse_mean']:.6f} ± {metrics.get('layer_mse_std', 0):.6f}"
        )
        logger.info(
            f"    MAE (mean ± std): {metrics['layer_mae_mean']:.6f} ± {metrics.get('layer_mae_std', 0):.6f}"
        )
        logger.info(
            f"    Max Error (mean): {metrics.get('layer_max_error_mean', 0):.6f}"
        )
        # NEW: Show per-layer cosine similarity aggregate
        if "layer_cosine_mean" in metrics:
            logger.info(
                f"    Cosine Sim (mean ± std): {metrics['layer_cosine_mean']:.6f} ± {metrics.get('layer_cosine_std', 0):.6f}"
            )
    if "per_layer" in metrics and metrics["per_layer"]:
        logger.info("\n  Per-Layer Breakdown:")
        for layer_metric in metrics["per_layer"][:10]:
            # Include cosine similarity in per-layer breakdown
            cosine_str = f", cos={layer_metric['cosine_similarity']:.4f}" if "cosine_similarity" in layer_metric else ""
            logger.info(
                f"    {layer_metric['layer']}: "
                f"MSE={layer_metric['mse']:.6f}, "
                f"MAE={layer_metric['mae']:.6f}, "
                f"max_err={layer_metric['max_error']:.6f}"
                f"{cosine_str}"
            )
        if len(metrics["per_layer"]) > 10:
            logger.info(f"    ... and {len(metrics['per_layer']) - 10} more layers")


def format_metrics_for_logging(metrics: Dict[str, float], prefix: str = ""):
    parts = [prefix]
    for metric_name in ["loss", "mse", "mae", "rmse", "cosine_similarity", "cosine_batch"]:
        if metric_name in metrics:
            parts.append(f"{metric_name}={metrics[metric_name]:.4f}")
    return ", ".join(parts)
