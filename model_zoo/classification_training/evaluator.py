import torch
import numpy as np
import logging
from typing import Dict, Any, List
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

logger = logging.getLogger(__name__)


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, config: Dict[str, Any], all_patterns: List[str] = None):
    metrics = {}

    metrics['accuracy_exact_match'] = (predictions == labels).all(dim=1).float().mean().item()
    metrics['accuracy_hamming'] = (predictions == labels).float().mean().item()

    preds_np = predictions.numpy()
    labels_np = labels.numpy()

    # macros
    metrics['precision_macro'] = precision_score(labels_np, preds_np, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    # micro
    metrics['precision_micro'] = precision_score(labels_np, preds_np, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(labels_np, preds_np, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(labels_np, preds_np, average='micro', zero_division=0)

    metrics['hamming_loss'] = hamming_loss(labels_np, preds_np)

    # pattern-wise
    if config['evaluation'].get('per_pattern_metrics', False) and all_patterns:
        per_pattern_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)
        per_pattern_precision = precision_score(labels_np, preds_np, average=None, zero_division=0)
        per_pattern_recall = recall_score(labels_np, preds_np, average=None, zero_division=0)
        for i, pattern_name in enumerate(all_patterns):
            metrics[f'pattern_{pattern_name}_f1'] = per_pattern_f1[i]
            metrics[f'pattern_{pattern_name}_precision'] = per_pattern_precision[i]
            metrics[f'pattern_{pattern_name}_recall'] = per_pattern_recall[i]

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    main_metrics = [
        'accuracy_exact_match',
        'accuracy_hamming',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'precision_micro',
        'recall_micro',
        'f1_micro',
        'hamming_loss',
        'loss'
    ]
    logger.info(f"{prefix}Metrics:")
    for metric_name in main_metrics:
        if metric_name in metrics:
            logger.info(f"  {metric_name}: {metrics[metric_name]:.4f}")
    pattern_metrics = {k: v for k, v in metrics.items() if k.startswith('pattern_')}
    if pattern_metrics:
        logger.info("  Per-pattern F1 scores:")
        for pattern_name, f1_score in sorted(pattern_metrics.items()):
            pattern = pattern_name.replace('pattern_', '').replace('_f1', '')
            logger.info(f"    {pattern}: {f1_score:.4f}")


def compute_class_weights(train_loader, num_patterns: int = 15) -> torch.Tensor:
    logger.info("Computing class weights from training data...")
    pattern_counts = torch.zeros(num_patterns)
    total_samples = 0
    for _, labels in train_loader:
        pattern_counts += labels.sum(dim=0)
        total_samples += labels.size(0)
    num_positive = pattern_counts
    num_negative = total_samples - pattern_counts
    pos_weight = num_negative / (num_positive + 1e-8)
    logger.info(f"Class weights computed (range: {pos_weight.min():.2f} - {pos_weight.max():.2f})")
    return pos_weight
