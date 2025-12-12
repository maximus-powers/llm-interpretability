import torch
import logging
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
from torch.utils.data import DataLoader

from model_zoo.dataset_generation.models import SubjectModel, SequenceDataset

logger = logging.getLogger(__name__)


class RepresentationEvaluator:
    def __init__(
        self, benchmark_dataset_path: str, device: str = "cpu", batch_size: int = 32
    ):
        self.device = device
        self.batch_size = batch_size
        with open(benchmark_dataset_path, "r") as f:
            self.benchmark_dataset = json.load(f)
        metadata = self.benchmark_dataset["metadata"]
        self.vocab = metadata["vocab"]
        self.vocab_size = len(self.vocab)
        self.sequence_length = metadata["sequence_length"]

        logger.info(f"Loaded benchmark from {benchmark_dataset_path}")
        logger.info(f"Patterns: {metadata['pattern_names']}")

    def compute_pattern_metrics(
        self,
        model: SubjectModel,
        target_pattern: str,
        model_trained_patterns: List[str],
        vocab: Optional[List[str]] = None,
    ):
        if vocab is None:
            vocab = self.vocab

        positive_sequences = []
        negative_sequences = []
        for example in self.benchmark_dataset["examples"]:
            seq = example["sequence"]
            pattern = example["pattern"]
            if pattern == target_pattern:
                positive_sequences.append(seq)
            elif pattern != target_pattern:
                negative_sequences.append((seq, pattern))

        clean_negatives = []
        for seq, source_pattern in negative_sequences:
            if source_pattern not in model_trained_patterns:
                clean_negatives.append(seq)

        model.eval()
        with torch.no_grad():
            if positive_sequences:
                pos_dataset = SequenceDataset(
                    [
                        {"sequence": seq, "label": 1, "pattern": target_pattern}
                        for seq in positive_sequences
                    ],
                    vocab_size=self.vocab_size,
                )
                pos_loader = DataLoader(
                    pos_dataset, batch_size=self.batch_size, shuffle=False
                )
                pos_predictions = []
                for sequences, labels, patterns in pos_loader:
                    sequences = sequences.to(self.device)
                    outputs = model(sequences).squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    pos_predictions.extend(preds.cpu().tolist())
                tp = sum(1 for p in pos_predictions if p == 1.0)
                fn = sum(1 for p in pos_predictions if p == 0.0)
            else:
                tp = fn = 0

            if clean_negatives:
                neg_dataset = SequenceDataset(
                    [
                        {"sequence": seq, "label": 0, "pattern": "negative"}
                        for seq in clean_negatives
                    ],
                    vocab_size=self.vocab_size,
                )
                neg_loader = DataLoader(
                    neg_dataset, batch_size=self.batch_size, shuffle=False
                )
                neg_predictions = []
                for sequences, labels, patterns in neg_loader:
                    sequences = sequences.to(self.device)
                    outputs = model(sequences).squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    neg_predictions.extend(preds.cpu().tolist())
                fp = sum(1 for p in neg_predictions if p == 1.0)
                tn = sum(1 for p in neg_predictions if p == 0.0)
            else:
                fp = tn = 0

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "n_positive_examples": len(positive_sequences),
            "n_clean_negatives": len(clean_negatives),
        }

    def evaluate_modification(
        self,
        original_weights: Dict[str, torch.Tensor],
        modified_weights: Dict[str, torch.Tensor],
        original_patterns: List[str],
        all_patterns: List[str],
        metadata: Dict[str, Any],
    ):
        logger.info(f"Evaluating modification: original_patterns={original_patterns}")
        logger.info(f"Evaluating on all {len(all_patterns)} patterns (observational)")

        model_config = metadata.get("model_config", {})
        subject_architecture = {
            "vocab_size": model_config.get("vocab_size", 7),
            "sequence_length": model_config.get("sequence_length", 7),
            "num_layers": model_config.get("num_layers", 3),
            "neurons_per_layer": model_config.get("neurons_per_layer", 64),
            "activation_type": model_config.get("activation_type", "relu"),
            "dropout_rate": model_config.get("dropout_rate", 0.0),
            "precision": model_config.get("precision", "float32"),
        }

        original_model = SubjectModel(**subject_architecture)
        original_model.load_state_dict(original_weights)
        original_model.to(self.device)

        modified_model = SubjectModel(**subject_architecture)
        modified_model.load_state_dict(modified_weights)
        modified_model.to(self.device)

        vocab = [chr(ord("A") + i) for i in range(subject_architecture["vocab_size"])]

        # Evaluate original model on ALL patterns
        logger.info("Evaluating original model on all patterns...")
        original_metrics = {}
        for pattern in all_patterns:
            metrics = self.compute_pattern_metrics(
                model=original_model,
                target_pattern=pattern,
                model_trained_patterns=original_patterns,
                vocab=vocab,
            )
            original_metrics[pattern] = metrics

        # Compute cumulative metrics for original
        original_metrics["cumulative"] = self._compute_cumulative_metrics(
            original_metrics
        )

        # Evaluate modified model on ALL patterns
        logger.info("Evaluating modified model on all patterns...")
        modified_metrics = {}
        # Note: For modified model, we still use original_patterns as trained patterns
        # because contamination filtering is based on what the model was originally trained on
        for pattern in all_patterns:
            metrics = self.compute_pattern_metrics(
                model=modified_model,
                target_pattern=pattern,
                model_trained_patterns=original_patterns,  # Still filter by original trained patterns
                vocab=vocab,
            )
            modified_metrics[pattern] = metrics

        # Compute cumulative metrics for modified
        modified_metrics["cumulative"] = self._compute_cumulative_metrics(
            modified_metrics
        )

        results = {
            "original_metrics": original_metrics,
            "modified_metrics": modified_metrics,
        }

        # Log summary
        self._log_evaluation_summary_new(results, original_patterns, all_patterns)

        return results

    def compute_aggregate_statistics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics across multiple evaluations.

        Args:
            results: List of evaluation results from evaluate_modification()
                    Each result has structure: {original_metrics, modified_metrics}

        Returns:
            Dictionary with aggregate statistics
        """
        # Filter out failed evaluations
        valid_results = [r for r in results if "error" not in r and "original_metrics" in r and "modified_metrics" in r]

        if not valid_results:
            logger.warning("No valid results to aggregate")
            return {}

        # Compute deltas across all patterns
        all_macro_f1_deltas = []
        all_macro_accuracy_deltas = []
        all_micro_f1_deltas = []
        all_micro_accuracy_deltas = []

        # Per-pattern deltas
        all_pattern_f1_deltas = defaultdict(list)
        all_pattern_accuracy_deltas = defaultdict(list)

        for r in valid_results:
            original = r["original_metrics"]
            modified = r["modified_metrics"]

            # Cumulative deltas
            if "cumulative" in original and "cumulative" in modified:
                all_macro_f1_deltas.append(
                    modified["cumulative"]["macro_f1"] - original["cumulative"]["macro_f1"]
                )
                all_macro_accuracy_deltas.append(
                    modified["cumulative"]["macro_accuracy"] - original["cumulative"]["macro_accuracy"]
                )
                all_micro_f1_deltas.append(
                    modified["cumulative"]["micro_f1"] - original["cumulative"]["micro_f1"]
                )
                all_micro_accuracy_deltas.append(
                    modified["cumulative"]["micro_accuracy"] - original["cumulative"]["micro_accuracy"]
                )

            # Per-pattern deltas (excluding cumulative)
            for pattern in original.keys():
                if pattern == "cumulative":
                    continue
                if pattern in modified:
                    all_pattern_f1_deltas[pattern].append(
                        modified[pattern]["f1"] - original[pattern]["f1"]
                    )
                    all_pattern_accuracy_deltas[pattern].append(
                        modified[pattern]["accuracy"] - original[pattern]["accuracy"]
                    )

        aggregate = {
            "n_evaluations": len(valid_results),
            "n_failed": len(results) - len(valid_results),
            "mean_macro_f1_delta": sum(all_macro_f1_deltas) / len(all_macro_f1_deltas) if all_macro_f1_deltas else 0.0,
            "mean_macro_accuracy_delta": sum(all_macro_accuracy_deltas) / len(all_macro_accuracy_deltas) if all_macro_accuracy_deltas else 0.0,
            "mean_micro_f1_delta": sum(all_micro_f1_deltas) / len(all_micro_f1_deltas) if all_micro_f1_deltas else 0.0,
            "mean_micro_accuracy_delta": sum(all_micro_accuracy_deltas) / len(all_micro_accuracy_deltas) if all_micro_accuracy_deltas else 0.0,
            "pattern_f1_delta_means": {
                pattern: sum(deltas) / len(deltas)
                for pattern, deltas in all_pattern_f1_deltas.items()
            },
            "pattern_f1_delta_stds": {
                pattern: torch.tensor(deltas).std().item()
                for pattern, deltas in all_pattern_f1_deltas.items()
            },
            "pattern_accuracy_delta_means": {
                pattern: sum(deltas) / len(deltas)
                for pattern, deltas in all_pattern_accuracy_deltas.items()
            },
            "pattern_accuracy_delta_stds": {
                pattern: torch.tensor(deltas).std().item()
                for pattern, deltas in all_pattern_accuracy_deltas.items()
            },
        }

        logger.info("=" * 60)
        logger.info("AGGREGATE STATISTICS")
        logger.info("=" * 60)
        logger.info(
            f"Evaluations: {aggregate['n_evaluations']} (failed: {aggregate['n_failed']})"
        )
        logger.info(f"\nMacro-Averaged Deltas:")
        logger.info(f"  Mean F1 Delta: {aggregate['mean_macro_f1_delta']:+.3f}")
        logger.info(f"  Mean Accuracy Delta: {aggregate['mean_macro_accuracy_delta']:+.3f}")
        logger.info(f"\nMicro-Averaged Deltas:")
        logger.info(f"  Mean F1 Delta: {aggregate['mean_micro_f1_delta']:+.3f}")
        logger.info(f"  Mean Accuracy Delta: {aggregate['mean_micro_accuracy_delta']:+.3f}")

        if aggregate["pattern_f1_delta_means"]:
            logger.info(f"\nPer-Pattern F1 Delta Means:")
            for pattern in sorted(aggregate["pattern_f1_delta_means"].keys()):
                mean = aggregate["pattern_f1_delta_means"][pattern]
                std = aggregate["pattern_f1_delta_stds"][pattern]
                logger.info(f"  {pattern}: {mean:+.3f} (±{std:.3f})")

        logger.info("=" * 60)

        return aggregate

    def _compute_cumulative_metrics(
        self, pattern_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute cumulative metrics across all patterns.
        Returns both macro (unweighted) and micro (weighted by sample size) averages.

        Args:
            pattern_metrics: Dictionary mapping pattern names to their metrics

        Returns:
            Dictionary with macro and micro averages for f1, precision, recall, accuracy
        """
        # Filter out 'cumulative' if it exists
        metrics_list = [m for p, m in pattern_metrics.items() if p != "cumulative"]

        if not metrics_list:
            return {
                "macro_accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "micro_accuracy": 0.0,
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
            }

        # Macro-average (equal weight per pattern)
        macro = {
            "macro_accuracy": sum(m["accuracy"] for m in metrics_list) / len(metrics_list),
            "macro_precision": sum(m["precision"] for m in metrics_list) / len(metrics_list),
            "macro_recall": sum(m["recall"] for m in metrics_list) / len(metrics_list),
            "macro_f1": sum(m["f1"] for m in metrics_list) / len(metrics_list),
        }

        # Micro-average (pool TP/FP/TN/FN, then compute metrics)
        total_tp = sum(m["tp"] for m in metrics_list)
        total_fp = sum(m["fp"] for m in metrics_list)
        total_fn = sum(m["fn"] for m in metrics_list)
        total_tn = sum(m["tn"] for m in metrics_list)

        total = total_tp + total_fp + total_fn + total_tn
        micro_accuracy = (total_tp + total_tn) / total if total > 0 else 0.0
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        micro = {
            "micro_accuracy": float(micro_accuracy),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
        }

        # Combine and return both
        return {**macro, **micro}

    def _log_evaluation_summary_new(
        self,
        results: Dict[str, Any],
        original_patterns: List[str],
        all_patterns: List[str],
    ):
        """Log evaluation summary with new observational format."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY (OBSERVATIONAL)")
        logger.info("=" * 60)

        original_metrics = results["original_metrics"]
        modified_metrics = results["modified_metrics"]

        # Cumulative metrics - Macro-averaged
        logger.info(f"\nCumulative Metrics (Macro-averaged):")
        logger.info(
            f"  Original - F1: {original_metrics['cumulative']['macro_f1']:.3f}, "
            f"Acc: {original_metrics['cumulative']['macro_accuracy']:.3f}, "
            f"Prec: {original_metrics['cumulative']['macro_precision']:.3f}, "
            f"Rec: {original_metrics['cumulative']['macro_recall']:.3f}"
        )
        logger.info(
            f"  Modified - F1: {modified_metrics['cumulative']['macro_f1']:.3f}, "
            f"Acc: {modified_metrics['cumulative']['macro_accuracy']:.3f}, "
            f"Prec: {modified_metrics['cumulative']['macro_precision']:.3f}, "
            f"Rec: {modified_metrics['cumulative']['macro_recall']:.3f}"
        )

        # Cumulative metrics - Micro-averaged
        logger.info(f"\nCumulative Metrics (Micro-averaged):")
        logger.info(
            f"  Original - F1: {original_metrics['cumulative']['micro_f1']:.3f}, "
            f"Acc: {original_metrics['cumulative']['micro_accuracy']:.3f}, "
            f"Prec: {original_metrics['cumulative']['micro_precision']:.3f}, "
            f"Rec: {original_metrics['cumulative']['micro_recall']:.3f}"
        )
        logger.info(
            f"  Modified - F1: {modified_metrics['cumulative']['micro_f1']:.3f}, "
            f"Acc: {modified_metrics['cumulative']['micro_accuracy']:.3f}, "
            f"Prec: {modified_metrics['cumulative']['micro_precision']:.3f}, "
            f"Rec: {modified_metrics['cumulative']['micro_recall']:.3f}"
        )

        # Per-pattern metrics for trained patterns
        logger.info(f"\nTrained Patterns ({len(original_patterns)}):")
        for pattern in original_patterns:
            if pattern in original_metrics and pattern in modified_metrics:
                orig = original_metrics[pattern]
                mod = modified_metrics[pattern]
                logger.info(f"  {pattern}:")
                logger.info(
                    f"    Original - F1: {orig['f1']:.3f}, Acc: {orig['accuracy']:.3f}"
                )
                logger.info(
                    f"    Modified - F1: {mod['f1']:.3f}, Acc: {mod['accuracy']:.3f}"
                )
                logger.info(
                    f"    Delta    - F1: {mod['f1'] - orig['f1']:+.3f}, Acc: {mod['accuracy'] - orig['accuracy']:+.3f}"
                )

        # Sample of untrained patterns (show first 3)
        untrained_patterns = [p for p in all_patterns if p not in original_patterns]
        if untrained_patterns:
            logger.info(
                f"\nSample Untrained Patterns (showing 3 of {len(untrained_patterns)}):"
            )
            for pattern in untrained_patterns[:3]:
                if pattern in original_metrics and pattern in modified_metrics:
                    orig = original_metrics[pattern]
                    mod = modified_metrics[pattern]
                    logger.info(f"  {pattern}:")
                    logger.info(
                        f"    Original - F1: {orig['f1']:.3f}, Acc: {orig['accuracy']:.3f}"
                    )
                    logger.info(
                        f"    Modified - F1: {mod['f1']:.3f}, Acc: {mod['accuracy']:.3f}"
                    )
                    logger.info(
                        f"    Delta    - F1: {mod['f1'] - orig['f1']:+.3f}, Acc: {mod['accuracy'] - orig['accuracy']:+.3f}"
                    )

        logger.info("=" * 60)
