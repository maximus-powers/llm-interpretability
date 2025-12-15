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
        self.similarity_f = nn.CosineSimilarity(dim=2)

        if projection_head_config is not None:
            self.projection_head = ProjectionHead(
                input_dim=projection_head_config["input_dim"],
                hidden_dim=projection_head_config["hidden_dim"],
                output_dim=projection_head_config["output_dim"],
            ).to(device)
        else:
            self.projection_head = None

    def _create_behavior_positive_mask(self, behavior_labels):
        batch_size = len(behavior_labels)
        positive_mask = torch.zeros((batch_size, batch_size), dtype=bool)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # positive if any pattern overlap
                    if set(behavior_labels[i]) & set(behavior_labels[j]):
                        positive_mask[i, j] = True

        return positive_mask

    def forward(self, latents: torch.Tensor, behavior_labels) -> torch.Tensor:
        if behavior_labels is None:
            raise ValueError("behavior_labels required for supervised contrastive loss")

        if self.projection_head is not None:
            latents = self.projection_head(latents)

        batch_size = latents.size(0)
        sim = (
            self.similarity_f(latents.unsqueeze(1), latents.unsqueeze(0))
            / self.temperature
        )

        # create positive mask from behavior labels
        positive_mask = self._create_behavior_positive_mask(behavior_labels).to(
            latents.device
        )

        # supervised contrastive loss
        # for each anchor, compute loss over its positive pairs
        loss = 0.0
        num_valid_anchors = 0

        for i in range(batch_size):
            positives_i = positive_mask[i]
            num_positives = positives_i.sum().item()

            if num_positives == 0:
                continue  # skip samples with no positive pairs

            # numerator: exp(sim) for positive pairs
            pos_sim = sim[i][positives_i]

            # denominator: sum of exp(sim) for all pairs except self
            all_except_self = torch.ones(batch_size, dtype=bool, device=latents.device)
            all_except_self[i] = False
            denom = sim[i][all_except_self].exp().sum()

            # loss for this anchor: -log(exp(pos) / denom) for each positive
            loss_i = -torch.log(pos_sim.exp() / denom.clamp(min=1e-8)).mean()
            loss += loss_i
            num_valid_anchors += 1

        if num_valid_anchors > 0:
            loss /= num_valid_anchors

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
