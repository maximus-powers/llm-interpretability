import torch
import logging
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    hamming_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np

logger = logging.getLogger(__name__)


class EncoderHeadEvaluator:
    def __init__(self, task_type: str, task_config: Dict[str, Any]):
        self.task_type = task_type
        self.task_config = task_config

    def evaluate(self, predictions: Any, targets: Any):
        if self.task_type == "pattern_classification":
            return self._evaluate_pattern_classification(predictions, targets)
        elif self.task_type == "accuracy_prediction":
            return self._evaluate_accuracy_prediction(predictions, targets)
        elif self.task_type == "hyperparameter_prediction":
            return self._evaluate_hyperparameter_prediction(predictions, targets)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _evaluate_pattern_classification(
        self, logits: torch.Tensor, targets: torch.Tensor
    ):
        targets_np = targets.detach().cpu().numpy()

        # apply sigmoid and threshold
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        predictions = (probs > 0.5).astype(int)

        metrics = {}
        metrics["hamming_loss"] = hamming_loss(targets_np, predictions)
        metrics["subset_accuracy"] = accuracy_score(targets_np, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets_np, predictions, average=None, zero_division=0
        )
        metrics["micro_f1"] = precision_recall_fscore_support(
            targets_np, predictions, average="micro", zero_division=0
        )[2]
        metrics["macro_f1"] = precision_recall_fscore_support(
            targets_np, predictions, average="macro", zero_division=0
        )[2]
        metrics["macro_precision"] = precision_recall_fscore_support(
            targets_np, predictions, average="macro", zero_division=0
        )[0]
        metrics["macro_recall"] = precision_recall_fscore_support(
            targets_np, predictions, average="macro", zero_division=0
        )[1]
        try:
            metrics["roc_auc_macro"] = roc_auc_score(targets_np, probs, average="macro")
        except ValueError:
            logger.warning("Could not compute ROC-AUC (not enough positive samples)")
            metrics["roc_auc_macro"] = 0.0

        # pattern-wise metrics
        pattern_names = self.task_config.get("patterns", [])
        for i, pattern_name in enumerate(pattern_names):
            metrics[f"f1_{pattern_name}"] = f1[i]
            metrics[f"precision_{pattern_name}"] = precision[i]
            metrics[f"recall_{pattern_name}"] = recall[i]

        return metrics

    def _evaluate_accuracy_prediction(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ):
        predictions_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        metrics = {}

        # regression metrics
        metrics["mse"] = mean_squared_error(targets_np, predictions_np)
        metrics["mae"] = mean_absolute_error(targets_np, predictions_np)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["r2"] = r2_score(targets_np, predictions_np)
        # relative error
        relative_errors = np.abs(predictions_np - targets_np) / (
            np.abs(targets_np) + 1e-8
        )
        metrics["mean_relative_error"] = np.mean(relative_errors)
        metrics["median_relative_error"] = np.median(relative_errors)
        # accuracy within thresholds
        within_5pct = np.mean(np.abs(predictions_np - targets_np) < 0.05)
        within_10pct = np.mean(np.abs(predictions_np - targets_np) < 0.10)
        metrics["accuracy_within_5pct"] = within_5pct
        metrics["accuracy_within_10pct"] = within_10pct

        return metrics

    def _evaluate_hyperparameter_prediction(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ):
        metrics = {}

        # evaluate continuous targets
        continuous_targets = self.task_config.get("continuous_targets", {})
        for name in continuous_targets.keys():
            pred_key = f"continuous_{name}"
            if pred_key not in predictions:
                continue
            pred = predictions[pred_key].detach().cpu().numpy().flatten()
            target = targets[pred_key].detach().cpu().numpy().flatten()
            # regression metrics
            metrics[f"{name}_mse"] = mean_squared_error(target, pred)
            metrics[f"{name}_mae"] = mean_absolute_error(target, pred)
            metrics[f"{name}_r2"] = r2_score(target, pred)
            # relative error
            rel_error = np.mean(np.abs(pred - target) / (np.abs(target) + 1e-8))
            metrics[f"{name}_relative_error"] = rel_error

        # evaluate classification targets
        discrete_targets = self.task_config.get("discrete_targets", {})
        for name, config in discrete_targets.items():
            pred_key = f"discrete_{name}"
            if pred_key not in predictions:
                continue
            logits = predictions[pred_key].detach().cpu()
            target = targets[pred_key].detach().cpu().numpy()
            pred_classes = torch.argmax(logits, dim=1).numpy()
            metrics[f"{name}_accuracy"] = accuracy_score(target, pred_classes)

            # classwise metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                target, pred_classes, average="macro", zero_division=0
            )
            metrics[f"{name}_precision"] = precision
            metrics[f"{name}_recall"] = recall
            metrics[f"{name}_f1"] = f1

        # average mse across continuous targets
        continuous_mses = [v for k, v in metrics.items() if k.endswith("_mse")]
        if continuous_mses:
            metrics["avg_continuous_mse"] = np.mean(continuous_mses)
        # average accuracy across discrete targets
        discrete_accs = [v for k, v in metrics.items() if k.endswith("_accuracy")]
        if discrete_accs:
            metrics["avg_discrete_accuracy"] = np.mean(discrete_accs)

        return metrics


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    task_type: str,
    task_config: Dict[str, Any],
    device: str = "cpu",
):
    model.eval()
    evaluator = EncoderHeadEvaluator(task_type, task_config)
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if "latent" in batch:
                latent = batch["latent"].to(device)
                outputs = model.prediction_head(latent)
            else:
                tokens = batch["tokens"].to(device)
                mask = batch["mask"].to(device)
                outputs = model(tokens, mask)
            targets = batch["target"]

            if isinstance(outputs, dict):
                # hparam prediction
                if not all_predictions:
                    all_predictions = {k: [] for k in outputs.keys()}
                    all_targets = {k: [] for k in targets.keys()}
                for key in outputs.keys():
                    all_predictions[key].append(outputs[key].cpu())
                    all_targets[key].append(targets[key].cpu())
            else:
                # pattern classification or accuracy prediction
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

    # concat results
    if isinstance(all_predictions, dict):
        # hparam prediction
        predictions = {k: torch.cat(v, dim=0) for k, v in all_predictions.items()}
        targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
    else:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

    metrics = evaluator.evaluate(predictions, targets)

    return metrics
