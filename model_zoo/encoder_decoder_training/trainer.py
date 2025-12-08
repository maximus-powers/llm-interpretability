import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import copy
import json
import yaml
import subprocess
import atexit
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

from .evaluator import compute_reconstruction_metrics, print_metrics, format_metrics_for_logging
from .tokenizer import WeightTokenizer
from .encoder_decoder_model import ProjectionHead
from .data_loader import augment_tokenized_weights

logger = logging.getLogger(__name__)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ReconstructionLoss:
    """Base class for reconstruction losses"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def compute(self, predicted: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MSEReconstructionLoss(ReconstructionLoss):
    def compute(self, predicted: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        squared_error = (predicted - target) ** 2
        token_error = squared_error.mean(dim=-1)
        masked_error = token_error * mask
        loss = masked_error.sum() / mask.sum().clamp(min=1)
        return loss


class MAEReconstructionLoss(ReconstructionLoss):
    def compute(self, predicted: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        absolute_error = torch.abs(predicted - target)
        token_error = absolute_error.mean(dim=-1)
        masked_error = token_error * mask
        loss = masked_error.sum() / mask.sum().clamp(min=1)
        return loss


class CosineReconstructionLoss(ReconstructionLoss):
    def compute(self, predicted: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(predicted, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        token_loss = 1 - cos_sim
        masked_loss = token_loss * mask
        loss = masked_loss.sum() / mask.sum().clamp(min=1)
        return loss


class CombinedReconstructionLoss(ReconstructionLoss):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        loss_cfg = config['loss']

        self.losses = {}
        self.weights = {}

        for loss_spec in loss_cfg['components']:
            loss_type = loss_spec['type']
            weight = loss_spec['weight']

            if loss_type == 'mse':
                self.losses['mse'] = MSEReconstructionLoss(config)
            elif loss_type == 'mae':
                self.losses['mae'] = MAEReconstructionLoss(config)
            elif loss_type == 'cosine':
                self.losses['cosine'] = CosineReconstructionLoss(config)
            else:
                raise ValueError(f"Unknown loss type in combined loss: {loss_type}")

            self.weights[loss_type] = weight

        logger.info(f"Combined loss initialized with components: {list(self.losses.keys())}, "
                   f"weights: {self.weights}")

    def compute(self, predicted: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        loss_components = {}

        for loss_name, loss_fn in self.losses.items():
            component_loss = loss_fn.compute(predicted, target, mask)
            loss_components[f'loss_{loss_name}'] = component_loss.item()
            total_loss += self.weights[loss_name] * component_loss

        return total_loss, loss_components


class NT_Xent_Loss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR."""

    def __init__(self, batch_size: int, temperature: float, device: str,
                 projection_head_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self._mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
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
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask.to(sim.device)].reshape(N, -1)

        labels = torch.zeros(N, device=positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class NT_Xent_pos_Loss(nn.Module):
    """Positive-only variant of NT-Xent loss."""

    def __init__(self, batch_size: int, temperature: float, device: str,
                 projection_head_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

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

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if self.projection_head is not None:
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
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

        if contrast_type == 'simclr':
            logger.info("Using SimCLR NT-Xent contrastive loss")
            self.loss_contrast = NT_Xent_Loss(
                batch_size=batch_size,
                temperature=temperature,
                device=device,
                projection_head_config=projection_head_config
            )
        elif contrast_type == 'positive':
            logger.info("Using positive-only contrastive loss")
            self.loss_contrast = NT_Xent_pos_Loss(
                batch_size=batch_size,
                temperature=temperature,
                device=device,
                projection_head_config=projection_head_config
            )
        else:
            raise ValueError(f"Unknown contrast type: {contrast_type}")

        recon_type = loss_cfg.get('reconstruction_type', 'mse')
        if recon_type == 'mse':
            self.loss_recon = MSEReconstructionLoss(config)
        elif recon_type == 'mae':
            self.loss_recon = MAEReconstructionLoss(config)
        elif recon_type == 'cosine':
            self.loss_recon = CosineReconstructionLoss(config)
        else:
            raise ValueError(f"Unknown reconstruction type: {recon_type}")

        logger.info(f"GammaContrastReconLoss: gamma={self.gamma}, contrast={contrast_type}, recon={recon_type}")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor,
                y: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def create_loss_function(config: Dict[str, Any], batch_size: int = None,
                         device: str = None) -> ReconstructionLoss:
    loss_type = config['loss']['type']

    if loss_type == 'mse':
        return MSEReconstructionLoss(config)
    elif loss_type == 'mae':
        return MAEReconstructionLoss(config)
    elif loss_type == 'cosine':
        return CosineReconstructionLoss(config)
    elif loss_type == 'combined':
        return CombinedReconstructionLoss(config)
    elif loss_type == 'contrastive':
        if batch_size is None or device is None:
            raise ValueError("batch_size and device are required for contrastive loss")
        return GammaContrastReconLoss(config, batch_size, device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# TRAINER
# ============================================================================

class EncoderDecoderTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 train_loader, val_loader, device: str,
                 tokenizer: WeightTokenizer, test_loader=None):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.tokenizer = tokenizer

        batch_size = config['training']['batch_size']
        self.criterion = create_loss_function(config, batch_size=batch_size, device=device)
        self.is_combined_loss = isinstance(self.criterion, CombinedReconstructionLoss)
        self.is_contrastive_loss = isinstance(self.criterion, GammaContrastReconLoss)

        if self.is_contrastive_loss:
            loss_cfg = config['loss']
            self.augmentation_type = loss_cfg.get('augmentation_type', 'noise')
            self.noise_std = loss_cfg.get('noise_std', 0.01)
            self.dropout_prob = loss_cfg.get('dropout_prob', 0.1)
            logger.info(f"Contrastive loss enabled with augmentation: {self.augmentation_type}")

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.scheduler = None
        if config['training']['lr_scheduler']['enabled']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=config['training']['early_stopping']['mode'],
                patience=config['training']['lr_scheduler']['patience'],
                factor=config['training']['lr_scheduler']['factor'],
                min_lr=config['training']['lr_scheduler']['min_lr']
            )

        run_dir = Path(config.get('run_dir', '.'))

        self.writer = None
        self.tensorboard_process = None
        self.log_dir = None
        if config['logging']['tensorboard']['enabled']:
            self.log_dir = run_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            if config['logging']['tensorboard'].get('auto_launch', False):
                self.tensorboard_process = self._start_tensorboard_server()
        self.early_stopping_config = config['training']['early_stopping']
        self.monitor_metric = self.early_stopping_config['monitor']
        self.mode = self.early_stopping_config['mode']
        self.patience = self.early_stopping_config['patience']
        if self.mode == 'max':
            self.best_metric = float('-inf')
        else:
            self.best_metric = float('inf')
        self.patience_counter = 0

        self.checkpoint_config = config['logging']['checkpoint']
        if self.checkpoint_config['enabled']:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = None

        self.global_step = 0

        self.hub_config = config.get('hub', {})
        self.hub_enabled = self.hub_config.get('enabled', False)
        self.hub_repo = None
        self.hf_api = None

        if self.hub_enabled:
            repo_id = self.hub_config.get('repo_id')
            if not repo_id:
                logger.error("Hub enabled but repo_id not specified")
                self.hub_enabled = False
            else:
                token = self.hub_config.get('token') or os.environ.get('HF_TOKEN')
                if not token:
                    logger.warning("No HuggingFace token found. Hub disabled.")
                    self.hub_enabled = False
                else:
                    self.hf_api = HfApi(token=token)
                    self.hub_repo = repo_id

                    try:
                        create_repo(
                            repo_id=repo_id,
                            token=token,
                            private=self.hub_config.get('private', False),
                            exist_ok=True
                        )
                        logger.info(f"HuggingFace Hub repo ready: {repo_id}")
                    except Exception as e:
                        logger.warning(f"Failed to create Hub repo: {e}")
                        self.hub_enabled = False

        logger.info("EncoderDecoderTrainer initialized")

    def _start_tensorboard_server(self):
        try:
            port = self.config['logging']['tensorboard'].get('port', 6006)
            process = subprocess.Popen(
                ['tensorboard', '--logdir', str(self.log_dir), '--port', str(port), '--bind_all'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            atexit.register(lambda: process.kill())
            logger.info(f"TensorBoard server started at http://localhost:{port}")
            return process
        except Exception as e:
            logger.warning(f"Failed to start TensorBoard server: {e}")
            return None

    def stop_tensorboard(self):
        if self.tensorboard_process:
            self.tensorboard_process.kill()
            self.tensorboard_process = None

    def train(self):
        logger.info("Starting training")
        logger.info(f"Total epochs: {self.config['training']['epochs']}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")

        for epoch in range(self.config['training']['epochs']):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            logger.info(f"{'='*70}")

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            self._log_epoch(epoch, train_metrics, val_metrics)

            if self.scheduler:
                monitor_value = val_metrics.get('loss')
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Learning rate: {current_lr:.6f}")

            if self._is_best(val_metrics):
                logger.info(f"New best model! {self.monitor_metric}={val_metrics.get('loss', 0):.6f}")
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        test_metrics = None
        if self.test_loader is not None:
            logger.info("\n" + "="*70)
            logger.info("Evaluating on test set...")
            logger.info("="*70)
            test_metrics = self._evaluate_test_set()
            print_metrics(test_metrics, prefix="test_")

        # Push encoder to Hub
        if self.hub_enabled and self.hub_config.get('push_model', True):
            self._finalize_hub_push(test_metrics)

        if self.writer:
            self.writer.close()

        logger.info("\nTraining complete!")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_loss_components = {}

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            tokens = batch['tokenized_weights'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            batch_size = tokens.size(0)

            self.optimizer.zero_grad()

            if self.is_contrastive_loss:
                tokens_i, mask_i = augment_tokenized_weights(
                    tokens, mask,
                    augmentation_type=self.augmentation_type,
                    noise_std=self.noise_std,
                    dropout_prob=self.dropout_prob
                )
                tokens_j, mask_j = augment_tokenized_weights(
                    tokens, mask,
                    augmentation_type=self.augmentation_type,
                    noise_std=self.noise_std,
                    dropout_prob=self.dropout_prob
                )

                z_i = self.model.encode(tokens_i, mask_i)
                z_j = self.model.encode(tokens_j, mask_j)

                y_i = self.model.decode(z_i, tokens.size(1))
                y_j = self.model.decode(z_j, tokens.size(1))

                y = torch.cat([y_i, y_j], dim=0)
                t = torch.cat([tokens_i, tokens_j], dim=0)
                mask_combined = torch.cat([mask_i, mask_j], dim=0)

                loss, loss_contrast, loss_recon = self.criterion(z_i, z_j, y, t, mask_combined)
                loss_components = {
                    'loss_contrast': loss_contrast.item(),
                    'loss_recon': loss_recon.item()
                }

                for key, value in loss_components.items():
                    if key not in all_loss_components:
                        all_loss_components[key] = 0
                    all_loss_components[key] += value * batch_size

            elif self.is_combined_loss:
                reconstructed = self.model(tokens, mask)
                loss, loss_components = self.criterion.compute(reconstructed, tokens, mask)
                for key, value in loss_components.items():
                    if key not in all_loss_components:
                        all_loss_components[key] = 0
                    all_loss_components[key] += value * batch_size
            else:
                reconstructed = self.model(tokens, mask)
                loss = self.criterion.compute(reconstructed, tokens, mask)
                loss_components = {}

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            progress_bar.set_postfix({'loss': loss.item()})

            if self.writer and batch_idx % self.config['logging']['tensorboard'].get('log_interval', 10) == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
                for component_name, component_value in loss_components.items():
                    self.writer.add_scalar(f'train/{component_name}', component_value, self.global_step)

            self.global_step += 1

        metrics = {'loss': total_loss / total_samples}

        for key, value in all_loss_components.items():
            metrics[key] = value / total_samples

        return metrics

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_samples = 0

        all_reconstructed = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Epoch {epoch+1}"):
                tokens = batch['tokenized_weights'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                batch_size = tokens.size(0)

                if self.is_contrastive_loss:
                    tokens_i, mask_i = augment_tokenized_weights(
                        tokens, mask,
                        augmentation_type=self.augmentation_type,
                        noise_std=self.noise_std,
                        dropout_prob=self.dropout_prob
                    )
                    tokens_j, mask_j = augment_tokenized_weights(
                        tokens, mask,
                        augmentation_type=self.augmentation_type,
                        noise_std=self.noise_std,
                        dropout_prob=self.dropout_prob
                    )

                    z_i = self.model.encode(tokens_i, mask_i)
                    z_j = self.model.encode(tokens_j, mask_j)

                    y_i = self.model.decode(z_i, tokens.size(1))
                    y_j = self.model.decode(z_j, tokens.size(1))

                    y = torch.cat([y_i, y_j], dim=0)
                    t = torch.cat([tokens_i, tokens_j], dim=0)
                    mask_combined = torch.cat([mask_i, mask_j], dim=0)

                    loss, _, _ = self.criterion(z_i, z_j, y, t, mask_combined)

                    reconstructed = y_i
                    all_reconstructed.append(reconstructed.cpu())
                    all_targets.append(tokens.cpu())
                    all_masks.append(mask.cpu())

                elif self.is_combined_loss:
                    reconstructed = self.model(tokens, mask)
                    loss, _ = self.criterion.compute(reconstructed, tokens, mask)
                    all_reconstructed.append(reconstructed.cpu())
                    all_targets.append(tokens.cpu())
                    all_masks.append(mask.cpu())

                else:
                    reconstructed = self.model(tokens, mask)
                    loss = self.criterion.compute(reconstructed, tokens, mask)
                    all_reconstructed.append(reconstructed.cpu())
                    all_targets.append(tokens.cpu())
                    all_masks.append(mask.cpu())

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        all_reconstructed = torch.cat(all_reconstructed, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        metrics = compute_reconstruction_metrics(
            all_reconstructed, all_targets, all_masks, self.config
        )
        metrics['loss'] = total_loss / total_samples

        return metrics

    def _evaluate_test_set(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_samples = 0

        all_reconstructed = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Set Evaluation"):
                tokens = batch['tokenized_weights'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                batch_size = tokens.size(0)

                reconstructed = self.model(tokens, mask)

                if self.is_combined_loss:
                    loss, _ = self.criterion.compute(reconstructed, tokens, mask)
                else:
                    loss = self.criterion.compute(reconstructed, tokens, mask)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_reconstructed.append(reconstructed.cpu())
                all_targets.append(tokens.cpu())
                all_masks.append(mask.cpu())

        all_reconstructed = torch.cat(all_reconstructed, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        metrics = compute_reconstruction_metrics(
            all_reconstructed, all_targets, all_masks, self.config
        )
        metrics['loss'] = total_loss / total_samples

        return metrics

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float]):
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Train - {format_metrics_for_logging(train_metrics, 'train_')}")
        logger.info(f"  Val   - {format_metrics_for_logging(val_metrics, 'val_')}")

        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train_epoch/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val_epoch/{key}', value, epoch)

    def _is_best(self, val_metrics: Dict[str, float]) -> bool:
        metric_value = val_metrics.get('loss')
        if metric_value is None:
            return False

        is_best = False
        if self.mode == 'max':
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self.patience_counter = 0
                is_best = True
        else:
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                self.patience_counter = 0
                is_best = True

        return is_best

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        if not self.early_stopping_config['enabled']:
            return False

        metric_value = val_metrics.get('loss')
        if metric_value is None:
            return False

        improved = False
        if self.mode == 'max':
            improved = metric_value > self.best_metric
        else:
            improved = metric_value < self.best_metric

        if not improved:
            self.patience_counter += 1
            logger.info(f"No improvement for {self.patience_counter} epochs (patience={self.patience})")

            if self.patience_counter >= self.patience:
                return True

        return False

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")

    def _finalize_hub_push(self, test_metrics: Optional[Dict[str, float]] = None):
        if not self.hub_enabled or not self.hub_repo:
            return

        try:
            logger.info("Finalizing HuggingFace Hub push (encoder only)...")

            operations = []

            self._create_model_card(test_metrics)
            readme_path = self.checkpoint_dir / 'README.md'
            if readme_path.exists():
                operations.append(
                    CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(readme_path))
                )

            if self.hub_config.get('push_encoder_only', True):
                encoder_checkpoint = {
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'config': self.config,
                    'tokenizer_config': self.tokenizer.get_config(),
                    'latent_dim': self.model.latent_dim,
                    'architecture_type': self.config['architecture']['type']
                }

                encoder_path = self.checkpoint_dir / 'encoder.pt'
                torch.save(encoder_checkpoint, encoder_path)

                operations.append(
                    CommitOperationAdd(path_in_repo='encoder.pt', path_or_fileobj=str(encoder_path))
                )
                logger.info("Added encoder.pt to upload queue")

            config_sanitized = copy.deepcopy(self.config)
            if 'hub' in config_sanitized and 'token' in config_sanitized['hub']:
                config_sanitized['hub']['token'] = '<REDACTED>'
            config_path = self.checkpoint_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_sanitized, f, default_flow_style=False)
            operations.append(
                CommitOperationAdd(path_in_repo='config.yaml', path_or_fileobj=str(config_path))
            )

            tokenizer_config = self.tokenizer.get_config()
            tokenizer_path = self.checkpoint_dir / 'tokenizer_config.json'
            with open(tokenizer_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            operations.append(
                CommitOperationAdd(path_in_repo='tokenizer_config.json', path_or_fileobj=str(tokenizer_path))
            )

            if operations:
                logger.info(f"Uploading {len(operations)} files...")
                self.hf_api.create_commit(
                    repo_id=self.hub_repo,
                    repo_type='model',
                    operations=operations,
                    commit_message="Upload weight-space encoder and configuration"
                )
                logger.info(f"Successfully uploaded encoder to https://huggingface.co/{self.hub_repo}")

        except Exception as e:
            logger.warning(f"Failed to finalize HuggingFace Hub push: {e}", exc_info=True)

    def _create_model_card(self, test_metrics: Optional[Dict[str, float]] = None):
        dataset_name = self.config['dataset']['hf_dataset']
        arch_type = self.config['architecture']['type']

        model_card = f"""---
tags:
- weight-space-learning
- neural-network-encoder
- autoencoder
- {arch_type}
datasets:
- {dataset_name}
---

# Weight-Space Encoder ({arch_type.upper()})

This model is a weight-space encoder trained on neural network weights following the SANE methodology.
It learns to encode neural network weights into a compact latent representation.

## Model Description

- **Architecture**: {arch_type.capitalize()} encoder-decoder
- **Training Dataset**: {dataset_name}
- **Input Mode**: {self.config['dataset']['input_mode']}
- **Latent Dimension**: {self.model.latent_dim}

## Tokenization

- **Chunk Size**: {self.tokenizer.chunk_size} weight values per token
- **Max Tokens**: {self.tokenizer.max_tokens}
- **Metadata**: {self.tokenizer.include_metadata}

## Training Configuration

- **Loss Function**: {self.config['loss']['type']}
- **Optimizer**: {self.config['training']['optimizer']}
- **Learning Rate**: {self.config['training']['learning_rate']}
- **Batch Size**: {self.config['training']['batch_size']}
"""

        if test_metrics:
            model_card += f"""
## Performance Metrics (Test Set)

- **MSE**: {test_metrics.get('mse', 0):.6f}
- **MAE**: {test_metrics.get('mae', 0):.6f}
- **RMSE**: {test_metrics.get('rmse', 0):.6f}
- **Cosine Similarity**: {test_metrics.get('cosine_similarity', 0):.4f}
- **R² Score**: {test_metrics.get('r2_score', 0):.4f}
"""

        model_card += """
## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Load encoder
encoder_path = hf_hub_download(repo_id="your-repo-id", filename="encoder.pt")
checkpoint = torch.load(encoder_path)

encoder_state_dict = checkpoint['encoder_state_dict']
latent_dim = checkpoint['latent_dim']

# Reconstruct encoder model (need architecture)
# ... (architecture reconstruction code)

# Encode weights
with torch.no_grad():
    latent = encoder(tokenized_weights, attention_mask)
```

## Citation

```bibtex
@software{weight_space_encoder,
  title={Weight-Space Encoder for Neural Network Representation Learning},
  year={2025},
  url={https://huggingface.co/""" + self.hub_repo + """}
}
```
"""

        model_card_path = self.checkpoint_dir / 'README.md'
        with open(model_card_path, 'w') as f:
            f.write(model_card)


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Best metric: {checkpoint.get('best_metric', 'unknown')}")

    return checkpoint
