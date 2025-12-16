import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

from .encoder_decoder_model import ProjectionHead

logger = logging.getLogger(__name__)


class ReconstructionLoss:
    def __init__(self, config: Dict[str, Any], loss_type: str = "mse"):
        self.config = config
        self.loss_type = loss_type
        if loss_type not in ["mse", "mae", "cosine"]:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Must be 'mse', 'mae', or 'cosine'"
            )

    def compute(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            token_error = ((predicted - target) ** 2).mean(dim=-1)
        elif self.loss_type == "mae":
            token_error = torch.abs(predicted - target).mean(dim=-1)
        elif self.loss_type == "cosine":
            pred_norm = F.normalize(predicted, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1)
            token_error = 1 - cos_sim
        masked_error = token_error * mask
        loss = masked_error.sum() / mask.sum().clamp(min=1)
        return loss


class CombinedReconstructionLoss:
    """Combines multiple reconstruction losses with weighted sum"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.losses = {}
        self.weights = {}
        for loss_spec in config["loss"]["components"]:
            loss_type = loss_spec["type"]
            weight = loss_spec["weight"]
            if loss_type in ["mse", "mae", "cosine"]:
                self.losses[loss_type] = ReconstructionLoss(config, loss_type=loss_type)
            else:
                raise ValueError(f"Unknown loss type in combined loss: {loss_type}")
            self.weights[loss_type] = weight

        logger.info(
            f"Combined loss initialized with components: {list(self.losses.keys())}"
        )

    def compute(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ):
        total_loss = 0
        loss_components = {}
        for loss_name, loss_fn in self.losses.items():
            component_loss = loss_fn.compute(predicted, target, mask)
            loss_components[f"loss_{loss_name}"] = component_loss.item()
            total_loss += self.weights[loss_name] * component_loss

        return total_loss, loss_components


class SupervisedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float,
        device: str,
        projection_head_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self._log_counter = 0

        if projection_head_config is not None:
            self.projection_head = ProjectionHead(
                input_dim=projection_head_config["input_dim"],
                hidden_dim=projection_head_config["hidden_dim"],
                output_dim=projection_head_config["output_dim"],
            ).to(device)
        else:
            self.projection_head = None

    def _create_behavior_positive_mask(self, behavior_labels, device):
        batch_size = len(behavior_labels)

        # get all unique behaviors
        all_behaviors = set()
        for labels in behavior_labels:
            all_behaviors.update(labels)
        behavior_to_idx = {b: i for i, b in enumerate(all_behaviors)}
        num_behaviors = len(all_behaviors)

        # create binary label matrix: [batch_size, num_behaviors]
        label_matrix = torch.zeros(batch_size, num_behaviors, device=device)
        for i, labels in enumerate(behavior_labels):
            for label in labels:
                label_matrix[i, behavior_to_idx[label]] = 1.0

        # positive mask: samples share at least one behavior
        # (label_matrix @ label_matrix.T) > 0 means overlap exists
        overlap = torch.mm(label_matrix, label_matrix.T)  # [batch, batch]
        positive_mask = (overlap > 0).float()

        # remove self-connections
        positive_mask = positive_mask - torch.eye(batch_size, device=device)
        positive_mask = positive_mask.clamp(min=0)

        return positive_mask

    def forward(self, latents: torch.Tensor, behavior_labels) -> torch.Tensor:
        if behavior_labels is None:
            raise ValueError("behavior_labels required for supervised contrastive loss")

        if self.projection_head is not None:
            latents = self.projection_head(latents)

        device = latents.device
        batch_size = latents.size(0)

        # L2 normalize features (standard for contrastive learning)
        features = F.normalize(latents, p=2, dim=1)

        # compute similarity matrix: [batch, batch]
        sim_matrix = torch.mm(features, features.T) / self.temperature

        # create positive mask from behavior labels
        positive_mask = self._create_behavior_positive_mask(behavior_labels, device)

        # for numerical stability: subtract max from each row
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # compute log-sum-exp for denominator (all pairs except self)
        self_mask = torch.eye(batch_size, device=device)
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)  # mask out self
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # log probabilities: log(exp(sim_ij/T) / sum_k≠i exp(sim_ik/T))
        log_prob = sim_matrix - log_sum_exp

        # mean log prob over positive pairs for each anchor
        # mask: only consider positive pairs
        num_positives_per_anchor = positive_mask.sum(dim=1)  # [batch]

        # compute mean log prob for positives (avoiding division by zero)
        # for anchors with no positives, we'll handle separately
        has_positives = num_positives_per_anchor > 0

        if not has_positives.any():
            logger.warning("No positive pairs found in batch!")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # sum of log_prob for positive pairs, divided by num positives
        masked_log_prob = log_prob * positive_mask
        sum_log_prob_pos = masked_log_prob.sum(dim=1)  # [batch]

        # only average over anchors that have positives
        mean_log_prob_pos = sum_log_prob_pos[has_positives] / num_positives_per_anchor[has_positives]

        # loss is negative mean log probability
        loss = -mean_log_prob_pos.mean()

        return loss


class GammaContrastReconLoss(nn.Module):
    """Combined contrastive + reconstruction loss: L = gamma * L_contrastive + (1 - gamma) * L_reconstruction"""

    def __init__(self, config: Dict[str, Any], device: str):
        super().__init__()
        loss_cfg = config["loss"]
        self.gamma = loss_cfg.get("gamma", 0.5)
        assert 0 <= self.gamma <= 1, f"gamma must be in [0, 1], got {self.gamma}"
        self.device = device

        temperature = loss_cfg.get("temperature", 0.1)

        projection_head_config = None
        if loss_cfg.get("projection_head") is not None:
            proj_cfg = loss_cfg["projection_head"]
            projection_head_config = {
                "input_dim": proj_cfg["input_dim"],
                "hidden_dim": proj_cfg["hidden_dim"],
                "output_dim": proj_cfg["output_dim"],
            }

        logger.info("Using supervised contrastive loss")
        self.loss_contrast = SupervisedContrastiveLoss(
            temperature=temperature,
            device=device,
            projection_head_config=projection_head_config,
        )

        recon_type = loss_cfg.get("reconstruction_type", "mse")
        if recon_type in ["mse", "mae", "cosine"]:
            self.loss_recon = ReconstructionLoss(config, loss_type=recon_type)
        else:
            raise ValueError(f"Unknown reconstruction type: {recon_type}")

        logger.info(f"GammaContrastReconLoss: gamma={self.gamma}, recon={recon_type}")

    def forward(
        self,
        latents: torch.Tensor,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        behavior_labels,
    ):
        if self.gamma < 1e-10:
            loss_recon = self.loss_recon.compute(reconstructed, target, mask)
            return loss_recon, torch.tensor(0.0, device=self.device), loss_recon

        elif abs(1.0 - self.gamma) < 1e-10:
            loss_contrast = self.loss_contrast(latents, behavior_labels)
            return loss_contrast, loss_contrast, torch.tensor(0.0, device=self.device)

        else:
            loss_contrast = self.loss_contrast(latents, behavior_labels)
            loss_recon = self.loss_recon.compute(reconstructed, target, mask)
            total_loss = self.gamma * loss_contrast + (1 - self.gamma) * loss_recon
            return total_loss, loss_contrast, loss_recon
