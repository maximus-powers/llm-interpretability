import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, Any, Optional

from .encoder_decoder_model import ProjectionHead

logger = logging.getLogger(__name__)


class TemperatureScheduler:
    """Schedules temperature from high (soft clustering) to low (hard clustering)."""

    def __init__(
        self,
        initial_temp: float = 0.3,
        final_temp: float = 0.08,
        warmup_epochs: int = 20,
        decay_type: str = "linear",
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.warmup_epochs = warmup_epochs
        self.decay_type = decay_type

    def get_temperature(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.final_temp

        progress = epoch / max(1, self.warmup_epochs)
        if self.decay_type == "linear":
            return self.initial_temp - (self.initial_temp - self.final_temp) * progress
        elif self.decay_type == "cosine":
            return self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1 + math.cos(math.pi * progress)
            )
        return self.initial_temp


class GammaScheduler:
    """Schedules gamma from reconstruction-focused to contrastive-focused."""

    def __init__(
        self,
        initial_gamma: float = 0.1,
        final_gamma: float = 0.5,
        warmup_epochs: int = 30,
        decay_type: str = "linear",
    ):
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.warmup_epochs = warmup_epochs
        self.decay_type = decay_type

    def get_gamma(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.final_gamma

        progress = epoch / max(1, self.warmup_epochs)
        if self.decay_type == "linear":
            return self.initial_gamma + (self.final_gamma - self.initial_gamma) * progress
        elif self.decay_type == "cosine":
            return self.initial_gamma + (self.final_gamma - self.initial_gamma) * (
                1 - 0.5 * (1 + math.cos(math.pi * progress))
            )
        return self.initial_gamma


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
        temperature_schedule: Optional[Dict[str, Any]] = None,
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
    ):
        super().__init__()
        self.base_temperature = temperature
        self.temperature = temperature
        self.device = device
        self._log_counter = 0

        # VICReg-style regularization to prevent representation collapse
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        if variance_weight > 0:
            logger.info(f"Variance regularization enabled: weight={variance_weight}")
        if covariance_weight > 0:
            logger.info(f"Covariance regularization enabled: weight={covariance_weight}")

        # temperature scheduling
        self.temp_scheduler = None
        if temperature_schedule is not None and temperature_schedule.get("enabled", False):
            self.temp_scheduler = TemperatureScheduler(
                initial_temp=temperature_schedule.get("initial", 0.3),
                final_temp=temperature_schedule.get("final", temperature),
                warmup_epochs=temperature_schedule.get("warmup_epochs", 20),
                decay_type=temperature_schedule.get("decay_type", "linear"),
            )
            logger.info(
                f"Temperature scheduling enabled: {self.temp_scheduler.initial_temp} -> "
                f"{self.temp_scheduler.final_temp} over {self.temp_scheduler.warmup_epochs} epochs"
            )

        if projection_head_config is not None:
            self.projection_head = ProjectionHead(
                input_dim=projection_head_config["input_dim"],
                hidden_dim=projection_head_config["hidden_dim"],
                output_dim=projection_head_config["output_dim"],
            ).to(device)
        else:
            self.projection_head = None

    def update_temperature(self, epoch: int) -> float:
        """Update temperature based on scheduler. Call at start of each epoch."""
        if self.temp_scheduler is not None:
            self.temperature = self.temp_scheduler.get_temperature(epoch)
        return self.temperature

    def _variance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """VICReg variance term: penalize low std to prevent collapse."""
        std = embeddings.std(dim=0)
        # Hinge loss: penalize if std < 1
        return F.relu(1 - std).mean()

    def _covariance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """VICReg covariance term: decorrelate embedding dimensions."""
        batch_size, dim = embeddings.shape
        embeddings = embeddings - embeddings.mean(dim=0)
        cov = (embeddings.T @ embeddings) / (batch_size - 1)
        # Zero out diagonal (we only want off-diagonal)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / dim

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

        # VICReg regularization (computed BEFORE normalization to prevent collapse)
        var_loss = torch.tensor(0.0, device=device)
        cov_loss = torch.tensor(0.0, device=device)
        if self.variance_weight > 0:
            var_loss = self._variance_loss(latents)
        if self.covariance_weight > 0:
            cov_loss = self._covariance_loss(latents)

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
            # Still apply regularization even if no positives
            return self.variance_weight * var_loss + self.covariance_weight * cov_loss

        # sum of log_prob for positive pairs, divided by num positives
        masked_log_prob = log_prob * positive_mask
        sum_log_prob_pos = masked_log_prob.sum(dim=1)  # [batch]

        # only average over anchors that have positives
        mean_log_prob_pos = sum_log_prob_pos[has_positives] / num_positives_per_anchor[has_positives]

        # loss is negative mean log probability + regularization
        contrastive_loss = -mean_log_prob_pos.mean()
        loss = contrastive_loss + self.variance_weight * var_loss + self.covariance_weight * cov_loss

        return loss


class GammaContrastReconLoss(nn.Module):
    """Combined contrastive + reconstruction loss: L = gamma * L_contrastive + (1 - gamma) * L_reconstruction"""

    def __init__(self, config: Dict[str, Any], device: str):
        super().__init__()
        loss_cfg = config["loss"]
        self.base_gamma = loss_cfg.get("gamma", 0.5)
        self.gamma = self.base_gamma
        assert 0 <= self.gamma <= 1, f"gamma must be in [0, 1], got {self.gamma}"
        self.device = device

        # gamma scheduling
        self.gamma_scheduler = None
        gamma_schedule = loss_cfg.get("gamma_schedule")
        if gamma_schedule is not None and gamma_schedule.get("enabled", False):
            self.gamma_scheduler = GammaScheduler(
                initial_gamma=gamma_schedule.get("initial", 0.1),
                final_gamma=gamma_schedule.get("final", self.base_gamma),
                warmup_epochs=gamma_schedule.get("warmup_epochs", 30),
                decay_type=gamma_schedule.get("decay_type", "linear"),
            )
            logger.info(
                f"Gamma scheduling enabled: {self.gamma_scheduler.initial_gamma} -> "
                f"{self.gamma_scheduler.final_gamma} over {self.gamma_scheduler.warmup_epochs} epochs"
            )

        temperature = loss_cfg.get("temperature", 0.1)
        temperature_schedule = loss_cfg.get("temperature_schedule")

        # VICReg-style regularization to prevent representation collapse
        variance_weight = loss_cfg.get("variance_weight", 1.0)  # Default ON
        covariance_weight = loss_cfg.get("covariance_weight", 0.04)  # Default ON

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
            temperature_schedule=temperature_schedule,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
        )

        recon_type = loss_cfg.get("reconstruction_type", "mse")
        if recon_type in ["mse", "mae", "cosine"]:
            self.loss_recon = ReconstructionLoss(config, loss_type=recon_type)
        else:
            raise ValueError(f"Unknown reconstruction type: {recon_type}")

        logger.info(f"GammaContrastReconLoss: gamma={self.gamma}, recon={recon_type}")

    def update_schedule(self, epoch: int):
        """Update gamma and temperature schedules. Call at start of each epoch."""
        if self.gamma_scheduler is not None:
            self.gamma = self.gamma_scheduler.get_gamma(epoch)
        self.loss_contrast.update_temperature(epoch)
        return self.gamma, self.loss_contrast.temperature

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
