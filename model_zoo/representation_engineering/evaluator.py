import torch
import logging
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
from torch.utils.data import DataLoader

from model_zoo.dataset_generation.models import SubjectModel, SequenceDataset

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Matching Utility Functions
# ============================================================================


def matches_pattern(
    sequence: List[Any], pattern: str, vocab: Optional[List[str]] = None
) -> bool:
    """
    Check if a sequence matches a specific pattern.

    Args:
        sequence: Sequence to check (list of tokens/strings)
        pattern: Pattern name
        vocab: Vocabulary for vowel/consonant checks

    Returns:
        True if sequence matches pattern, False otherwise
    """
    if not sequence:
        return False

    # Convert to list if tuple
    if isinstance(sequence, tuple):
        sequence = list(sequence)

    if pattern == "all_same":
        return len(set(sequence)) == 1

    elif pattern == "palindrome":
        return sequence == sequence[::-1]

    elif pattern == "sorted_ascending":
        return sequence == sorted(sequence)

    elif pattern == "sorted_descending":
        return sequence == sorted(sequence, reverse=True)

    elif pattern == "alternating":
        if len(sequence) < 2:
            return False
        # Check if exactly two unique values that alternate
        unique = list(set(sequence))
        if len(unique) != 2:
            return False
        return all(sequence[i] != sequence[i + 1] for i in range(len(sequence) - 1))

    elif pattern == "contains_abc":
        # Check if contains consecutive subsequence A, B, C
        if vocab is None:
            vocab = ["A", "B", "C", "D", "E", "F", "G"]
        target = vocab[:3]  # ['A', 'B', 'C']
        for i in range(len(sequence) - 2):
            if sequence[i : i + 3] == target:
                return True
        return False

    elif pattern == "starts_with":
        # Pattern requires starting with a specific token (implementation detail: any token satisfies this)
        return True  # All sequences start with something

    elif pattern == "ends_with":
        # Pattern requires ending with a specific token (implementation detail: any token satisfies this)
        return True  # All sequences end with something

    elif pattern == "no_repeats":
        return len(sequence) == len(set(sequence))

    elif pattern == "has_majority":
        # One token appears more than 50% of positions
        if not sequence:
            return False
        from collections import Counter

        counts = Counter(sequence)
        max_count = max(counts.values())
        majority_threshold = len(sequence) // 2 + 1
        return max_count >= majority_threshold

    elif pattern == "increasing_pairs":
        # Each adjacent pair in alphabetical order
        return all(sequence[i] <= sequence[i + 1] for i in range(len(sequence) - 1))

    elif pattern == "decreasing_pairs":
        # Each adjacent pair in reverse alphabetical order
        return all(sequence[i] >= sequence[i + 1] for i in range(len(sequence) - 1))

    elif pattern == "vowel_consonant":
        # Alternates between vowels and consonants
        if vocab is None:
            vocab = ["A", "B", "C", "D", "E", "F", "G"]
        vowels = ["A", "E", "I", "O", "U"]
        vowels = [v for v in vowels if v in vocab]
        consonants = [c for c in vocab if c not in vowels]

        if not vowels or not consonants:
            return False

        # Check if alternates between vowel and consonant
        for i in range(len(sequence) - 1):
            curr_is_vowel = sequence[i] in vowels
            next_is_vowel = sequence[i + 1] in vowels
            if curr_is_vowel == next_is_vowel:  # Both vowels or both consonants
                return False
        return True

    elif pattern == "first_last_match":
        return len(sequence) >= 2 and sequence[0] == sequence[-1]

    elif pattern == "mountain_pattern":
        # Increases then decreases
        if len(sequence) < 3:
            return False
        max_val = max(sequence)
        max_idx = sequence.index(max_val)
        # Check increasing up to max
        increasing = all(sequence[i] <= sequence[i + 1] for i in range(max_idx))
        # Check decreasing after max
        decreasing = all(
            sequence[i] >= sequence[i + 1] for i in range(max_idx, len(sequence) - 1)
        )
        return increasing and decreasing

    else:
        logger.warning(f"Unknown pattern: {pattern}")
        return False


class RepresentationEvaluator:
    def __init__(self, benchmark_dataset_path: str, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size

        with open(benchmark_dataset_path, 'r') as f:
            self.benchmark_dataset = json.load(f)

        metadata = self.benchmark_dataset['metadata']
        self.vocab = metadata['vocab']
        self.vocab_size = len(self.vocab)
        self.sequence_length = metadata['sequence_length']

        logger.info(f"Loaded benchmark from {benchmark_dataset_path}")
        logger.info(f"Benchmark contains {len(self.benchmark_dataset['examples'])} examples")
        logger.info(f"Patterns: {metadata['pattern_names']}")

    def compute_pattern_metrics(
        self,
        model: SubjectModel,
        target_pattern: str,
        model_trained_patterns: List[str],
        all_patterns: List[str],
        vocab: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if vocab is None:
            vocab = self.vocab

        positive_sequences = []
        negative_sequences = []
        for example in self.benchmark_dataset['examples']:
            seq = example['sequence']
            pattern = example['pattern']
            if pattern == target_pattern:
                positive_sequences.append(seq)
            elif pattern != target_pattern:
                negative_sequences.append((seq, pattern))

        clean_negatives = []
        for seq, source_pattern in negative_sequences:
            matches_trained = any(
                matches_pattern(seq, trained_pattern, vocab)
                for trained_pattern in model_trained_patterns
                if trained_pattern != target_pattern
            )
            if not matches_trained:
                clean_negatives.append(seq)

        model.eval()
        with torch.no_grad():
            if positive_sequences:
                pos_dataset = SequenceDataset(
                    [{"sequence": seq, "label": 1, "pattern": target_pattern} for seq in positive_sequences],
                    vocab_size=self.vocab_size,
                )
                pos_loader = DataLoader(pos_dataset, batch_size=self.batch_size, shuffle=False)
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
                    [{"sequence": seq, "label": 0, "pattern": "negative"} for seq in clean_negatives],
                    vocab_size=self.vocab_size,
                )
                neg_loader = DataLoader(neg_dataset, batch_size=self.batch_size, shuffle=False)
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
    ) -> Dict[str, Any]:
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
                all_patterns=all_patterns,
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
                all_patterns=all_patterns,
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

    def _generate_benchmark(self, patterns: List[str]) -> Dict[str, Any]:
        """Generate benchmark dataset for evaluation."""
        logger.info(f"Generating benchmark for patterns: {patterns}")

        benchmark = self.pattern_sampler.create_dataset(
            include_patterns=patterns,
            samples_per_pattern=self.samples_per_pattern,
            negative_ratio=self.negative_ratio,
        )

        logger.info(f"Generated benchmark with {len(benchmark['examples'])} examples")
        return benchmark

    def _evaluate_weights(
        self, weights: Dict[str, torch.Tensor], benchmark: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load weights into SubjectModel and compute accuracy.

        Args:
            weights: Model state_dict
            benchmark: Benchmark dataset

        Returns:
            Dictionary with accuracy metrics
        """
        # Create model
        model = SubjectModel(**self.subject_architecture)
        model.load_state_dict(weights)
        model.to(self.device)
        model.eval()

        # Create dataset and dataloader
        dataset = SequenceDataset(
            benchmark["examples"],
            vocab_size=self.subject_architecture.get("vocab_size", 7),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Compute accuracy
        correct = 0
        total = 0
        pattern_correct = defaultdict(int)
        pattern_total = defaultdict(int)

        with torch.no_grad():
            for sequences, labels, patterns in loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = model(sequences).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                predictions = (torch.sigmoid(outputs) > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Track per-pattern accuracy
                for pred, label, pattern in zip(
                    predictions.cpu(), labels.cpu(), patterns
                ):
                    pattern_total[pattern] += 1
                    if pred.item() == label.item():
                        pattern_correct[pattern] += 1

        overall_accuracy = correct / total if total > 0 else 0.0

        pattern_accuracy = {
            pattern: pattern_correct[pattern] / pattern_total[pattern]
            for pattern in pattern_total.keys()
        }

        results = {
            "overall_accuracy": overall_accuracy,
            "pattern_accuracy": pattern_accuracy,
            "total_examples": total,
            "correct_predictions": correct,
        }

        return results

    def _compute_comparison(
        self,
        original_results: Dict[str, Any],
        modified_results: Dict[str, Any],
        original_patterns: List[str],
        target_patterns: List[str],
        operation: str,
    ) -> Dict[str, Any]:
        """Compute comparison metrics between original and modified models."""
        # Overall accuracy delta
        accuracy_delta = (
            modified_results["overall_accuracy"] - original_results["overall_accuracy"]
        )

        # Per-pattern deltas
        pattern_deltas = {}
        for pattern in set(original_patterns + target_patterns):
            orig_acc = original_results["pattern_accuracy"].get(pattern, 0.0)
            mod_acc = modified_results["pattern_accuracy"].get(pattern, 0.0)
            pattern_deltas[pattern] = mod_acc - orig_acc

        # Categorize patterns
        target_pattern_deltas = {
            p: pattern_deltas[p] for p in target_patterns if p in pattern_deltas
        }
        original_pattern_deltas = {
            p: pattern_deltas[p] for p in original_patterns if p in pattern_deltas
        }

        # Compute success metrics
        if operation == "add":
            # Target patterns should improve
            target_success = all(
                delta > 0.05 for delta in target_pattern_deltas.values()
            )
        elif operation == "remove":
            # Target patterns should decrease
            target_success = all(
                delta < -0.05 for delta in target_pattern_deltas.values()
            )
        else:
            target_success = False

        # Original patterns should be largely preserved (within 10% drop)
        preservation_success = all(
            abs(delta) < 0.10 for delta in original_pattern_deltas.values()
        )

        comparison = {
            "accuracy_delta": accuracy_delta,
            "pattern_deltas": pattern_deltas,
            "target_pattern_deltas": target_pattern_deltas,
            "original_pattern_deltas": original_pattern_deltas,
            "target_success": target_success,
            "preservation_success": preservation_success,
            "overall_success": target_success and preservation_success,
        }

        return comparison

    def _log_evaluation_summary(self, results: Dict[str, Any]):
        """Log evaluation summary."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        original = results["original_model"]
        modified = results["modified_model"]
        comparison = results["comparison"]

        logger.info(f"Overall Accuracy:")
        logger.info(f"  Original: {original['overall_accuracy']:.1%}")
        logger.info(f"  Modified: {modified['overall_accuracy']:.1%}")
        logger.info(f"  Delta: {comparison['accuracy_delta']:+.1%}")

        logger.info(f"\nPer-Pattern Accuracy:")
        for pattern in sorted(comparison["pattern_deltas"].keys()):
            orig_acc = original["pattern_accuracy"].get(pattern, 0.0)
            mod_acc = modified["pattern_accuracy"].get(pattern, 0.0)
            delta = comparison["pattern_deltas"][pattern]

            pattern_type = ""
            if pattern in comparison["target_pattern_deltas"]:
                pattern_type = " [TARGET]"
            elif pattern in comparison["original_pattern_deltas"]:
                pattern_type = " [ORIGINAL]"

            logger.info(f"  {pattern}{pattern_type}:")
            logger.info(
                f"    Original: {orig_acc:.1%}, Modified: {mod_acc:.1%}, Delta: {delta:+.1%}"
            )

        logger.info(f"\nSuccess Metrics:")
        logger.info(f"  Target Success: {comparison['target_success']}")
        logger.info(f"  Preservation Success: {comparison['preservation_success']}")
        logger.info(f"  Overall Success: {comparison['overall_success']}")
        logger.info("=" * 60)

    def evaluate_batch(
        self, modifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of modifications.

        Args:
            modifications: List of dicts with keys:
                - 'original_weights'
                - 'modified_weights'
                - 'original_patterns'
                - 'target_patterns'
                - 'operation'
                - 'metadata' (optional)

        Returns:
            List of evaluation results
        """
        results = []

        for idx, mod in enumerate(modifications):
            logger.info(f"Evaluating modification {idx + 1}/{len(modifications)}")

            try:
                result = self.evaluate_modification(
                    original_weights=mod["original_weights"],
                    modified_weights=mod["modified_weights"],
                    original_patterns=mod["original_patterns"],
                    target_patterns=mod["target_patterns"],
                    operation=mod["operation"],
                )

                # Add metadata if provided
                if "metadata" in mod:
                    result["metadata"] = mod["metadata"]

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to evaluate modification {idx}: {e}")
                results.append({"error": str(e), "metadata": mod.get("metadata")})

        return results

    def compute_aggregate_statistics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics across multiple evaluations.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with aggregate statistics
        """
        # Filter out failed evaluations
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            logger.warning("No valid results to aggregate")
            return {}

        # Aggregate metrics
        accuracy_deltas = [r["comparison"]["accuracy_delta"] for r in valid_results]
        target_successes = [r["comparison"]["target_success"] for r in valid_results]
        preservation_successes = [
            r["comparison"]["preservation_success"] for r in valid_results
        ]
        overall_successes = [r["comparison"]["overall_success"] for r in valid_results]

        # Per-pattern deltas
        all_pattern_deltas = defaultdict(list)
        for r in valid_results:
            for pattern, delta in r["comparison"]["pattern_deltas"].items():
                all_pattern_deltas[pattern].append(delta)

        aggregate = {
            "n_evaluations": len(valid_results),
            "n_failed": len(results) - len(valid_results),
            "mean_accuracy_delta": sum(accuracy_deltas) / len(accuracy_deltas),
            "target_success_rate": sum(target_successes) / len(target_successes),
            "preservation_success_rate": sum(preservation_successes)
            / len(preservation_successes),
            "overall_success_rate": sum(overall_successes) / len(overall_successes),
            "pattern_delta_means": {
                pattern: sum(deltas) / len(deltas)
                for pattern, deltas in all_pattern_deltas.items()
            },
            "pattern_delta_stds": {
                pattern: torch.tensor(deltas).std().item()
                for pattern, deltas in all_pattern_deltas.items()
            },
        }

        logger.info("=" * 60)
        logger.info("AGGREGATE STATISTICS")
        logger.info("=" * 60)
        logger.info(
            f"Evaluations: {aggregate['n_evaluations']} (failed: {aggregate['n_failed']})"
        )
        logger.info(f"Mean Accuracy Delta: {aggregate['mean_accuracy_delta']:+.1%}")
        logger.info(f"Target Success Rate: {aggregate['target_success_rate']:.1%}")
        logger.info(
            f"Preservation Success Rate: {aggregate['preservation_success_rate']:.1%}"
        )
        logger.info(f"Overall Success Rate: {aggregate['overall_success_rate']:.1%}")
        logger.info("=" * 60)

        return aggregate

    def _compute_cumulative_metrics(
        self, pattern_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute cumulative (average) metrics across all patterns.

        Args:
            pattern_metrics: Dictionary mapping pattern names to their metrics

        Returns:
            Dictionary with cumulative f1, precision, recall, accuracy
        """
        # Filter out 'cumulative' if it exists
        metrics_list = [m for p, m in pattern_metrics.items() if p != "cumulative"]

        if not metrics_list:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        cumulative = {
            "accuracy": sum(m["accuracy"] for m in metrics_list) / len(metrics_list),
            "precision": sum(m["precision"] for m in metrics_list) / len(metrics_list),
            "recall": sum(m["recall"] for m in metrics_list) / len(metrics_list),
            "f1": sum(m["f1"] for m in metrics_list) / len(metrics_list),
        }

        return cumulative

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

        # Cumulative metrics
        logger.info(f"\nCumulative Metrics:")
        logger.info(
            f"  Original - F1: {original_metrics['cumulative']['f1']:.3f}, "
            f"Acc: {original_metrics['cumulative']['accuracy']:.3f}, "
            f"Prec: {original_metrics['cumulative']['precision']:.3f}, "
            f"Rec: {original_metrics['cumulative']['recall']:.3f}"
        )
        logger.info(
            f"  Modified - F1: {modified_metrics['cumulative']['f1']:.3f}, "
            f"Acc: {modified_metrics['cumulative']['accuracy']:.3f}, "
            f"Prec: {modified_metrics['cumulative']['precision']:.3f}, "
            f"Rec: {modified_metrics['cumulative']['recall']:.3f}"
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
