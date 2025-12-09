import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

from .encoder_decoder_model import ProjectionHead

logger = logging.getLogger(__name__)


class ReconstructionLoss:
    """Unified reconstruction loss supporting MSE, MAE, and Cosine variants"""
    def __init__(self, config: Dict[str, Any], loss_type: str = 'mse'):
        self.config = config
        self.loss_type = loss_type
        if loss_type not in ['mse', 'mae', 'cosine']:
            raise ValueError(f"Unknown loss type: {loss_type}. Must be 'mse', 'mae', or 'cosine'")

    def compute(self, predicted: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            token_error = ((predicted - target) ** 2).mean(dim=-1)
        elif self.loss_type == 'mae':
            token_error = torch.abs(predicted - target).mean(dim=-1)
        elif self.loss_type == 'cosine':
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
        for loss_spec in config['loss']['components']:
            loss_type = loss_spec['type']
            weight = loss_spec['weight']
            if loss_type in ['mse', 'mae', 'cosine']:
                self.losses[loss_type] = ReconstructionLoss(config, loss_type=loss_type)
            else:
                raise ValueError(f"Unknown loss type in combined loss: {loss_type}")
            self.weights[loss_type] = weight

        logger.info(f"Combined loss initialized with components: {list(self.losses.keys())}")

    def compute(self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        total_loss = 0
        loss_components = {}
        for loss_name, loss_fn in self.losses.items():
            component_loss = loss_fn.compute(predicted, target, mask)
            loss_components[f'loss_{loss_name}'] = component_loss.item()
            total_loss += self.weights[loss_name] * component_loss

        return total_loss, loss_components


class NT_Xent_Loss(nn.Module):
    """Unified NT-Xent loss supporting full SimCLR and positive-only variants"""

    def __init__(self, batch_size: int, temperature: float, device: str,
                 projection_head_config: Optional[Dict[str, Any]] = None,
                 variant: str = 'simclr'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.variant = variant

        if variant not in ['simclr', 'positive']:
            raise ValueError(f"Unknown NT-Xent variant: {variant}. Must be 'simclr' or 'positive'")
        if variant == 'simclr':
            self.mask = self._mask_correlated_samples(batch_size)
            self.criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.criterion = nn.MSELoss(reduction='mean')

        self.similarity_f = nn.CosineSimilarity(dim=2)

        if projection_head_config is not None:
            self.projection_head = ProjectionHead(
                input_dim=projection_head_config['input_dim'],
                hidden_dim=projection_head_config['hidden_dim'],
                output_dim=projection_head_config['output_dim']
            ).to(device)
        else:
            self.projection_head = None

    def _mask_correlated_samples(self, batch_size: int) -> torch.Tensor:
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if self.projection_head is not None:
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        if self.variant == 'simclr':
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask.to(sim.device)].reshape(N, -1)

            labels = torch.zeros(N, device=positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)

            loss = self.criterion(logits, labels)
            loss /= N
        else:  # positive-only variant
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).unsqueeze(1)
            labels = torch.zeros_like(positive_samples)
            loss = self.criterion(positive_samples, labels)
            loss /= N

        return loss


class GammaContrastReconLoss(nn.Module):
    """Combined contrastive + reconstruction loss: L = gamma * L_contrastive + (1 - gamma) * L_reconstruction"""

    def __init__(self, config: Dict[str, Any], batch_size: int, device: str):
        super().__init__()
        loss_cfg = config['loss']
        self.gamma = loss_cfg.get('gamma', 0.5)
        assert 0 <= self.gamma <= 1, f"gamma must be in [0, 1], got {self.gamma}"
        self.device = device

        contrast_type = loss_cfg.get('contrast_type', 'simclr')
        temperature = loss_cfg.get('temperature', 0.1)

        projection_head_config = None
        if loss_cfg.get('projection_head') is not None:
            proj_cfg = loss_cfg['projection_head']
            projection_head_config = {
                'input_dim': proj_cfg['input_dim'],
                'hidden_dim': proj_cfg['hidden_dim'],
                'output_dim': proj_cfg['output_dim']
            }

        if contrast_type in ['simclr', 'positive']:
            logger.info(f"Using {contrast_type} NT-Xent contrastive loss")
            self.loss_contrast = NT_Xent_Loss(
                batch_size=batch_size,
                temperature=temperature,
                device=device,
                projection_head_config=projection_head_config,
                variant=contrast_type
            )
        else:
            raise ValueError(f"Unknown contrast type: {contrast_type}")

        recon_type = loss_cfg.get('reconstruction_type', 'mse')
        if recon_type in ['mse', 'mae', 'cosine']:
            self.loss_recon = ReconstructionLoss(config, loss_type=recon_type)
        else:
            raise ValueError(f"Unknown reconstruction type: {recon_type}")

        logger.info(f"GammaContrastReconLoss: gamma={self.gamma}, contrast={contrast_type}, recon={recon_type}")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, y: torch.Tensor, t: torch.Tensor, mask: torch.Tensor):
        if self.gamma < 1e-10:
            loss_recon = self.loss_recon.compute(y, t, mask)
            return loss_recon, torch.tensor(0.0, device=self.device), loss_recon

        elif abs(1.0 - self.gamma) < 1e-10:
            loss_contrast = self.loss_contrast(z_i, z_j)
            return loss_contrast, loss_contrast, torch.tensor(0.0, device=self.device)

        else:
            loss_contrast = self.loss_contrast(z_i, z_j)
            loss_recon = self.loss_recon.compute(y, t, mask)
            total_loss = self.gamma * loss_contrast + (1 - self.gamma) * loss_recon
            return total_loss, loss_contrast, loss_recon
