import torch
import torch.nn as nn
import logging
import os
import copy
import json
import yaml
import subprocess
import atexit
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, Any, Optional, List
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

from .evaluator import (
    compute_reconstruction_metrics,
    print_metrics,
    format_metrics_for_logging,
)
from .tokenizer import WeightTokenizer
from .losses import (
    ReconstructionLoss,
    CombinedReconstructionLoss,
    GammaContrastReconLoss,
    FunctionalReconstructionLoss,
    VarianceRegularizationLoss,
)

logger = logging.getLogger(__name__)


class EncoderDecoderTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader,
        val_loader,
        device: str,
        tokenizer: WeightTokenizer,
        test_loader=None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.tokenizer = tokenizer
        self.input_mode = config["dataset"]["input_mode"]

        # loss
        loss_config = config["loss"]
        self.loss_weights = {}
        # reconstruction loss
        self.reconstruction_loss = None
        recon_config = loss_config.get("reconstruction", {})
        if recon_config.get("enabled", False):
            recon_type = recon_config.get("type", "mse")
            recon_weight = recon_config.get("weight", 1.0)
            mask_padding = recon_config.get("mask_padding", True)
            if recon_type in ["mse", "mae", "cosine"]:
                self.reconstruction_loss = ReconstructionLoss(config, loss_type=recon_type, mask_padding=mask_padding)
            elif recon_type == "combined":
                self.reconstruction_loss = CombinedReconstructionLoss(config, mask_padding=mask_padding)
            self.loss_weights["reconstruction"] = recon_weight
            logger.info(f"ReconstructionLoss initialized: type={recon_type}, mask_padding={mask_padding}")
        # contrastive loss
        self.contrastive_loss = None
        contrast_config = loss_config.get("contrastive", {})
        if contrast_config.get("enabled", False):
            contrast_weight = contrast_config.get("weight", 0.3)
            self.contrastive_loss = GammaContrastReconLoss(config, device)
            self.loss_weights["contrastive"] = contrast_weight
        # functional loss
        self.functional_loss = None
        func_config = loss_config.get("functional", {})
        if func_config.get("enabled", False):
            func_weight = func_config.get("weight", 0.5)
            self.functional_loss = FunctionalReconstructionLoss(config, device)
            self.loss_weights["functional"] = func_weight

        # variance regularization loss (prevents decoder collapse)
        self.variance_loss = None
        variance_config = loss_config.get("variance", {})
        if variance_config.get("enabled", False):
            variance_weight = variance_config.get("weight", 0.1)
            target_variance = variance_config.get("target_variance", 0.01)
            self.variance_loss = VarianceRegularizationLoss(target_variance=target_variance)
            self.loss_weights["variance"] = variance_weight

        self.is_contrastive_loss = self.contrastive_loss is not None
        self.is_combined_loss = isinstance(self.reconstruction_loss, CombinedReconstructionLoss)
        # normalize weights
        total_weight = sum(self.loss_weights.values())
        if total_weight > 0:
            self.loss_weights = {k: v / total_weight for k, v in self.loss_weights.items()}
        for loss_name, weight in self.loss_weights.items():
            logger.info(f"{loss_name.capitalize()} loss enabled: weight={weight:.1%}")
        logger.info(f"Normalized loss weights: {self.loss_weights}")

        # optimizer and scheduler
        base_lr = config["training"]["learning_rate"]
        weight_decay = config["training"]["weight_decay"]

        # add projection head parameters with separate learning rate if using contrastive loss
        if self.is_contrastive_loss:
            if (
                hasattr(self.contrastive_loss, "loss_fn")
                and hasattr(self.contrastive_loss.loss_fn, "projection_head")
                and self.contrastive_loss.loss_fn.projection_head is not None
            ):
                proj_head = self.contrastive_loss.loss_fn.projection_head
                proj_head_lr = config["loss"]["contrastive"].get("projection_head_lr", base_lr)

                # use parameter groups for different learning rates
                param_groups = [
                    {"params": model.parameters(), "lr": base_lr},
                    {"params": proj_head.parameters(), "lr": proj_head_lr},
                ]
                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=weight_decay,
                )
                logger.info(
                    f"Optimizer: model lr={base_lr}, projection_head lr={proj_head_lr}"
                )
            else:
                logger.warning("Contrastive loss enabled but projection head not found or is None")
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=base_lr,
                    weight_decay=weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
            )
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = config["training"].get(
            "gradient_accumulation_steps", 1
        )

        # scheduler setup
        self.scheduler = None
        lr_scheduler_config = config["training"].get("lr_scheduler", {})
        if lr_scheduler_config.get("enabled", False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=config["training"]["early_stopping"]["mode"],
                patience=lr_scheduler_config.get("patience", 5),
                factor=lr_scheduler_config.get("factor", 0.5),
                min_lr=lr_scheduler_config.get("min_lr", 1e-6),
            )
            logger.info("Using ReduceLROnPlateau scheduler")

        # logs
        run_dir = Path(config.get("run_dir", "."))
        self.writer = None
        self.tensorboard_process = None
        self.log_dir = None
        if config["logging"]["tensorboard"]["enabled"]:
            self.log_dir = run_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

            # custom scalars layout for organized multi-line plots
            layout = {
                "Loss Components": {
                    "all_losses": ["Multiline", ["train/reconstruction", "train/contrastive", "train/functional"]],
                },
                "Train vs Val": {
                    "loss": ["Multiline", ["train_epoch/loss", "val_epoch/loss"]],
                    "mse": ["Multiline", ["train_epoch/mse", "val_epoch/mse"]],
                    "cosine_similarity": ["Multiline", ["train_epoch/cosine_similarity", "val_epoch/cosine_similarity"]],
                },
                "Reconstruction Quality": {
                    "metrics": ["Multiline", ["val_epoch/mse", "val_epoch/mae", "val_epoch/rmse"]],
                },
            }
            self.writer.add_custom_scalars(layout)

            if config["logging"]["tensorboard"].get("auto_launch", False):
                self.tensorboard_process = self._start_tensorboard_server()

        # checkpointing
        self.early_stopping_config = config["training"]["early_stopping"]
        self.monitor_metric = self.early_stopping_config["monitor"]
        self.mode = self.early_stopping_config["mode"]
        self.patience = self.early_stopping_config["patience"]
        if self.mode == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")
        self.patience_counter = 0
        self.checkpoint_config = config["logging"]["checkpoint"]
        if self.checkpoint_config["enabled"]:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = None
        self.global_step = 0

        # hf hub
        self.hub_config = config.get("hub", {})
        self.hub_enabled = self.hub_config.get("enabled", False)
        self.hub_repo = None
        self.hf_api = None
        if self.hub_enabled:
            repo_id = self.hub_config.get("repo_id")

            if not repo_id:
                logger.error("Hub enabled but repo_id not specified")
                self.hub_enabled = False
            else:
                token = self.hub_config.get("token") or os.environ.get("HF_TOKEN")
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
                            private=self.hub_config.get("private", False),
                            exist_ok=True,
                        )
                        logger.info(f"HuggingFace Hub repo ready: {repo_id}")
                    except Exception as e:
                        logger.warning(f"Failed to create Hub repo: {e}")
                        self.hub_enabled = False

        logger.info("EncoderDecoderTrainer initialized")

    def _log_embeddings_to_projector(self, embeddings: torch.Tensor, labels: list, epoch: int):
        if self.log_dir is None or self.writer is None:
            return

        metadata = labels if labels and len(labels) == embeddings.shape[0] else None

        # add_embedding creates projector/ dir with tensors.tsv and metadata.tsv
        self.writer.add_embedding(
            mat=embeddings,
            metadata=metadata,
            global_step=epoch,
            tag="latent_space"
        )

        # flush to ensure files are written before TensorBoard reads them
        self.writer.flush()

        logger.info(f"Logged {embeddings.shape[0]} embeddings to projector (epoch {epoch})")

    def _get_filter_normalized_direction(self, model: nn.Module) -> List[torch.Tensor]:
        """
        Generate a random direction in weight space with filter-wise normalization.

        Filter normalization scales each filter/layer to have the same norm as the
        corresponding filter in the model weights. This makes the direction more
        meaningful for visualization (see Li et al. 2018).
        """
        direction = []
        for param in model.parameters():
            # Random direction
            d = torch.randn_like(param)
            # Filter-wise normalization: scale to match parameter norm
            param_norm = param.norm()
            if param_norm > 0:
                d = d / d.norm() * param_norm
            direction.append(d)
        return direction

    def _set_weights_along_direction(
        self,
        model: nn.Module,
        origin_weights: List[torch.Tensor],
        direction1: List[torch.Tensor],
        direction2: List[torch.Tensor],
        alpha: float,
        beta: float,
    ):
        """
        Set model weights to: origin + alpha * direction1 + beta * direction2
        """
        with torch.no_grad():
            for param, origin, d1, d2 in zip(model.parameters(), origin_weights, direction1, direction2):
                param.copy_(origin + alpha * d1 + beta * d2)

    def _compute_loss_at_point(self, model: nn.Module, data_batch: Dict[str, torch.Tensor]) -> float:
        """
        Compute loss at current model weights using a single batch.
        """
        model.eval()
        with torch.no_grad():
            encoder_input = data_batch["encoder_input"].to(self.device)
            encoder_mask = data_batch["encoder_mask"].to(self.device)
            decoder_target = data_batch["decoder_target"].to(self.device)
            decoder_mask = data_batch["decoder_mask"].to(self.device)
            arch_specs = data_batch.get("arch_spec", None)

            reconstructed, _ = model(encoder_input, encoder_mask, arch_specs, decoder_target.size(1))

            # Use simple MSE for landscape visualization
            token_error = ((reconstructed - decoder_target) ** 2).mean(dim=-1)
            masked_error = token_error * decoder_mask
            loss = masked_error.sum() / decoder_mask.sum().clamp(min=1)

        return loss.item()

    def _compute_loss_landscape(
        self,
        steps: int = 21,
        distance: float = 1.0,
        data_batch: Dict[str, torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute a 2D loss landscape around current weights.

        Args:
            steps: Number of points along each direction (creates steps x steps grid)
            distance: How far to perturb in each direction (scaled by filter norm)
            data_batch: Batch of data to evaluate loss on

        Returns:
            2D numpy array of loss values
        """
        if data_batch is None:
            return None

        # Store original weights
        origin_weights = [param.clone() for param in self.model.parameters()]

        # Generate two random orthogonal directions
        direction1 = self._get_filter_normalized_direction(self.model)
        direction2 = self._get_filter_normalized_direction(self.model)

        # Create coordinate grid
        coords = np.linspace(-distance, distance, steps)
        landscape = np.zeros((steps, steps))

        # Evaluate loss at each point
        for i, alpha in enumerate(coords):
            for j, beta in enumerate(coords):
                self._set_weights_along_direction(
                    self.model, origin_weights, direction1, direction2, alpha, beta
                )
                landscape[i, j] = self._compute_loss_at_point(self.model, data_batch)

        # Restore original weights
        with torch.no_grad():
            for param, origin in zip(self.model.parameters(), origin_weights):
                param.copy_(origin)

        return landscape, coords

    def _log_loss_landscape(self, epoch: int, data_batch: Dict[str, torch.Tensor] = None):
        """
        Compute and log loss landscape visualization to TensorBoard.
        """
        if self.writer is None or data_batch is None:
            return

        viz_config = self.config["logging"]["tensorboard"].get("visualizations", {})
        landscape_config = viz_config.get("loss_landscape", {})

        steps = landscape_config.get("resolution", 15)
        distance = landscape_config.get("distance", 0.5)

        logger.info(f"Computing loss landscape ({steps}x{steps} grid)...")

        try:
            landscape, coords = self._compute_loss_landscape(
                steps=steps,
                distance=distance,
                data_batch=data_batch,
            )

            if landscape is None:
                return

            # Create 3D surface plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            X, Y = np.meshgrid(coords, coords)
            surf = ax.plot_surface(X, Y, landscape, cmap='viridis', alpha=0.8)

            # Mark the center (original weights)
            center_idx = len(coords) // 2
            center_loss = landscape[center_idx, center_idx]
            ax.scatter([0], [0], [center_loss], color='red', s=100, label='Current weights')

            ax.set_xlabel('Direction 1')
            ax.set_ylabel('Direction 2')
            ax.set_zlabel('Loss')
            ax.set_title(f'Loss Landscape (Epoch {epoch})')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.legend()

            # Log to tensorboard
            self.writer.add_figure('loss_landscape/3d_surface', fig, epoch)

            # Also create 2D contour plot
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            contour = ax2.contourf(X, Y, landscape, levels=20, cmap='viridis')
            ax2.scatter([0], [0], color='red', s=100, marker='x', label='Current weights')
            ax2.set_xlabel('Direction 1')
            ax2.set_ylabel('Direction 2')
            ax2.set_title(f'Loss Landscape Contour (Epoch {epoch})')
            fig2.colorbar(contour)
            ax2.legend()

            self.writer.add_figure('loss_landscape/contour', fig2, epoch)

            # Log scalar metrics about the landscape
            self.writer.add_scalar('loss_landscape/center_loss', center_loss, epoch)
            self.writer.add_scalar('loss_landscape/min_loss', landscape.min(), epoch)
            self.writer.add_scalar('loss_landscape/max_loss', landscape.max(), epoch)
            self.writer.add_scalar('loss_landscape/sharpness', landscape.max() - landscape.min(), epoch)

            plt.close('all')
            logger.info(f"Loss landscape logged (center={center_loss:.4f}, range={landscape.min():.4f}-{landscape.max():.4f})")

        except Exception as e:
            logger.warning(f"Failed to compute loss landscape: {e}")

    def _start_tensorboard_server(self):
        try:
            port = self.config["logging"]["tensorboard"].get("port", 6006)
            process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir",
                    str(self.log_dir),
                    "--port",
                    str(port),
                    "--bind_all",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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

    def _calculate_loss(
        self,
        encoder_input,
        encoder_mask,
        decoder_target,
        decoder_mask,
        arch_specs=None,
        behavior_labels=None,
        original_shapes=None,
        model_configs=None,
        test_inputs=None,
    ):
        batch_size = encoder_input.size(0)

        # Use first arch_spec for batch (all samples in batch should have same architecture)
        # This is a simplification - for variable architectures, batch-per-architecture is needed
        arch_spec = arch_specs[0] if arch_specs else None

        # forward pass with architecture bypass
        latent = self.model.encode(encoder_input, encoder_mask)
        reconstructed = self.model.decode(latent, arch_spec, decoder_target.size(1))

        # === Decoder Health Diagnostics ===
        # These metrics detect decoder collapse (all positions outputting identical values)
        decoder_health = {}
        with torch.no_grad():
            # Variance across token positions (should be > 0, collapse = 0)
            decoder_health["decoder_token_variance"] = reconstructed.var(dim=1).mean().item()
            # Variance across features (less critical but useful)
            decoder_health["decoder_feature_variance"] = reconstructed.var(dim=2).mean().item()

            # FiLM layer statistics (gamma should start ~1, learn to vary)
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "film_layers"):
                gamma_values = []
                for film_layer in self.model.decoder.film_layers:
                    # Get gamma from bias (first half of film_generator bias)
                    bias = film_layer.film_generator.bias
                    half = bias.size(0) // 2
                    gamma_values.append(bias[:half].detach())
                if gamma_values:
                    all_gamma = torch.cat(gamma_values)
                    decoder_health["film_gamma_mean"] = all_gamma.mean().item()
                    decoder_health["film_gamma_std"] = all_gamma.std().item()

        # Decoder always outputs weights only (no metadata - architecture bypasses latent)
        weights_only_dim = getattr(self.model, "weights_only_dim", None)
        if weights_only_dim is not None:
            decoder_target_for_loss = decoder_target[:, :, :weights_only_dim]
        else:
            decoder_target_for_loss = decoder_target

        # accumulate losses
        total_loss = 0.0
        loss_components = {}

        # reconstruction loss
        if self.reconstruction_loss is not None:
            recon_weight = self.loss_weights.get("reconstruction", 1.0)
            if self.is_combined_loss:
                recon_loss, recon_components = self.reconstruction_loss.compute(
                    reconstructed, decoder_target_for_loss, decoder_mask
                )
                for k, v in recon_components.items():
                    loss_components[f"recon_{k}"] = v
            else:
                recon_loss = self.reconstruction_loss.compute(
                    reconstructed, decoder_target_for_loss, decoder_mask
                )
            weighted_recon = recon_weight * recon_loss
            total_loss = total_loss + weighted_recon
            loss_components["reconstruction"] = (
                recon_loss.item() if hasattr(recon_loss, "item") else recon_loss
            )

        # contrastive loss
        if self.contrastive_loss is not None and behavior_labels is not None:
            contrast_weight = self.loss_weights.get("contrastive", 0.3)
            contrast_loss = self.contrastive_loss(latent, behavior_labels)
            weighted_contrast = contrast_weight * contrast_loss
            total_loss = total_loss + weighted_contrast
            loss_components["contrastive"] = (
                contrast_loss.item() if hasattr(contrast_loss, "item") else contrast_loss
            )

        # functional loss
        if (
            self.functional_loss is not None
            and original_shapes is not None
            and model_configs is not None
            and test_inputs is not None
        ):
            func_weight = self.loss_weights.get("functional", 0.5)
            batch_func_loss = 0.0

            for i in range(batch_size):
                # detokenize original weights (from decoder_target)
                orig_weights = self.tokenizer.detokenize_differentiable(
                    decoder_target[i], decoder_mask[i], original_shapes[i]
                )
                # detokenize reconstructed weights (preserves gradients)
                recon_weights = self.tokenizer.detokenize_differentiable(
                    reconstructed[i], decoder_mask[i], original_shapes[i]
                )
                # compute functional loss for this sample (weight is applied inside)
                func_loss = self.functional_loss(
                    orig_weights, recon_weights, model_configs[i], test_inputs[i]
                )
                batch_func_loss = batch_func_loss + func_loss

            batch_func_loss = batch_func_loss / batch_size
            weighted_func = func_weight * batch_func_loss
            total_loss = total_loss + weighted_func
            loss_components["functional"] = (
                batch_func_loss.item() if hasattr(batch_func_loss, "item") else batch_func_loss
            )

        # variance regularization loss (prevents decoder collapse)
        if self.variance_loss is not None:
            variance_weight = self.loss_weights.get("variance", 0.1)
            var_loss = self.variance_loss.compute(reconstructed, decoder_mask)
            weighted_var = variance_weight * var_loss
            total_loss = total_loss + weighted_var
            loss_components["variance"] = (
                var_loss.item() if hasattr(var_loss, "item") else var_loss
            )

        # Add decoder health metrics (for TensorBoard monitoring)
        loss_components.update(decoder_health)

        return total_loss, reconstructed, loss_components

    def train(self):
        logger.info("Starting training")

        for epoch in range(self.config["training"]["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            self._log_epoch(epoch, train_metrics, val_metrics)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # update scheduler
            if self.scheduler:
                monitor_value = val_metrics.get(
                    self.monitor_metric, val_metrics.get("loss")
                )
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
                    new_lr = self.optimizer.param_groups[0]["lr"]
                    if new_lr != current_lr:
                        logger.info(
                            f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}"
                        )
                    current_lr = new_lr

            if self.writer:
                self.writer.add_scalar("train/learning_rate", current_lr, epoch)
            is_best = self._is_best(val_metrics)
            if is_best:
                metric_value = val_metrics.get(self.monitor_metric, val_metrics.get("loss"))
                logger.info(
                    f"New best model! {self.monitor_metric}={metric_value:.6f}"
                )
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                if self._check_early_stopping(val_metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        test_metrics = None
        if self.test_loader is not None:
            logger.info("Evaluating on test set")
            test_metrics = self._evaluate_test_set()
            print_metrics(test_metrics, prefix="test_")

        # push encoder to hub
        if self.hub_enabled and self.hub_config.get("push_model", True):
            self._finalize_hub_push(test_metrics)

        if self.writer:
            self.writer.close()

        logger.info("\nTraining complete")

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_loss_components = {}
        accumulation_steps = self.gradient_accumulation_steps

        self.optimizer.zero_grad()  # zero gradients at start of epoch

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(progress_bar):
            encoder_input = batch["encoder_input"].to(self.device)
            encoder_mask = batch["encoder_mask"].to(self.device)
            decoder_target = batch["decoder_target"].to(self.device)
            decoder_mask = batch["decoder_mask"].to(self.device)
            behavior_labels = batch.get("behavior_labels", None)
            batch_size = encoder_input.size(0)

            # extract functional loss data if available
            original_shapes = batch.get("original_shapes", None)
            model_configs = batch.get("model_config", None)
            test_inputs = batch.get("test_inputs", None)
            arch_specs = batch.get("arch_spec", None)
            if test_inputs is not None:
                test_inputs = test_inputs.to(self.device)

            loss, _, loss_components = self._calculate_loss(
                encoder_input,
                encoder_mask,
                decoder_target,
                decoder_mask,
                arch_specs=arch_specs,
                behavior_labels=behavior_labels,
                original_shapes=original_shapes,
                model_configs=model_configs,
                test_inputs=test_inputs,
            )

            # scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            # accumulate loss components
            for key, value in loss_components.items():
                if key not in all_loss_components:
                    all_loss_components[key] = 0
                all_loss_components[key] += value * batch_size

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # only step optimizer every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # log gradient norms for debugging
                if self.writer:
                    global_step = epoch * len(self.train_loader) + batch_idx

                    # encoder gradient norm
                    encoder_grad_norm = 0.0
                    for p in self.model.encoder.parameters():
                        if p.grad is not None:
                            encoder_grad_norm += p.grad.data.norm(2).item() ** 2
                    encoder_grad_norm = encoder_grad_norm**0.5
                    self.writer.add_scalar(
                        "train/encoder_grad_norm", encoder_grad_norm, global_step
                    )

                    # decoder gradient norm
                    decoder_grad_norm = 0.0
                    for p in self.model.decoder.parameters():
                        if p.grad is not None:
                            decoder_grad_norm += p.grad.data.norm(2).item() ** 2
                    decoder_grad_norm = decoder_grad_norm**0.5
                    self.writer.add_scalar(
                        "train/decoder_grad_norm", decoder_grad_norm, global_step
                    )

                    # projection head gradient norm (for contrastive learning)
                    if self.is_contrastive_loss:
                        proj_head = getattr(
                            getattr(self.contrastive_loss, "loss_fn", None),
                            "projection_head",
                            None,
                        )
                        if proj_head is not None:
                            proj_grad_norm = 0.0
                            for p in proj_head.parameters():
                                if p.grad is not None:
                                    proj_grad_norm += p.grad.data.norm(2).item() ** 2
                            proj_grad_norm = proj_grad_norm**0.5
                            self.writer.add_scalar(
                                "train/proj_head_grad_norm", proj_grad_norm, global_step
                            )

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

            if (
                self.writer
                and batch_idx
                % self.config["logging"]["tensorboard"].get("log_interval", 10)
                == 0
            ):
                self.writer.add_scalar(
                    "train/batch_loss", loss.item(), self.global_step
                )
                for component_name, component_value in loss_components.items():
                    self.writer.add_scalar(
                        f"train/{component_name}", component_value, self.global_step
                    )

            self.global_step += 1

        # handle remaining gradients if batch count isn't divisible by accumulation_steps
        if len(self.train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        metrics = {"loss": total_loss / total_samples}
        for key, value in all_loss_components.items():
            metrics[key] = value / total_samples

        return metrics

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_reconstructed = []
        all_targets = []
        all_masks = []
        all_latents = []
        all_behavior_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Epoch {epoch + 1}"):
                encoder_input = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                decoder_target = batch["decoder_target"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                behavior_labels = batch.get("behavior_labels", None)
                batch_size = encoder_input.size(0)

                # extract functional loss data if available
                original_shapes = batch.get("original_shapes", None)
                model_configs = batch.get("model_config", None)
                test_inputs = batch.get("test_inputs", None)
                arch_specs = batch.get("arch_spec", None)
                if test_inputs is not None:
                    test_inputs = test_inputs.to(self.device)

                # calculate loss using helper method
                loss, reconstructed, _ = self._calculate_loss(
                    encoder_input,
                    encoder_mask,
                    decoder_target,
                    decoder_mask,
                    arch_specs=arch_specs,
                    behavior_labels=behavior_labels,
                    original_shapes=original_shapes,
                    model_configs=model_configs,
                    test_inputs=test_inputs,
                )

                # collect latents for tensorboard visualizations
                latent = self.model.encode(encoder_input, encoder_mask)
                all_latents.append(latent.cpu())
                if behavior_labels is not None:
                    # behavior_labels is list of lists - join each to single string for projector
                    for labels in behavior_labels:
                        if isinstance(labels, list):
                            all_behavior_labels.append(", ".join(labels) if labels else "none")
                        else:
                            all_behavior_labels.append(str(labels) if labels else "none")

                # Decoder always outputs weights only (architecture bypasses latent)
                weights_only_dim = getattr(self.model, "weights_only_dim", None)
                if weights_only_dim is not None:
                    decoder_target_for_metrics = decoder_target[:, :, :weights_only_dim]
                else:
                    decoder_target_for_metrics = decoder_target

                for i in range(reconstructed.size(0)):
                    all_reconstructed.append(reconstructed[i : i + 1].cpu())
                    all_targets.append(decoder_target_for_metrics[i : i + 1].cpu())
                    all_masks.append(decoder_mask[i : i + 1].cpu())

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # pad all tensors to consistent dimension before concatenating
        max_token_dim = max(t.size(2) for t in all_reconstructed)

        padded_reconstructed = []
        padded_targets = []
        for recon, target in zip(all_reconstructed, all_targets):
            if recon.size(2) < max_token_dim:
                pad_size = max_token_dim - recon.size(2)
                recon = torch.nn.functional.pad(recon, (0, pad_size))
                target = torch.nn.functional.pad(target, (0, pad_size))
            padded_reconstructed.append(recon)
            padded_targets.append(target)

        all_reconstructed = torch.cat(padded_reconstructed, dim=0)
        all_targets = torch.cat(padded_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        metrics = compute_reconstruction_metrics(
            all_reconstructed, all_targets, all_masks, self.config
        )
        metrics["loss"] = total_loss / total_samples

        # tensorboard advanced visualizations
        if self.writer:
            tb_config = self.config["logging"]["tensorboard"]
            viz_config = tb_config.get("visualizations", {})

            if viz_config.get("enabled", True):
                log_interval = viz_config.get("log_interval", 5)

                if epoch % log_interval == 0:
                    # concatenate all latents
                    all_latents_tensor = torch.cat(all_latents, dim=0)

                    # histograms
                    self.writer.add_histogram("latent/values", all_latents_tensor, epoch)
                    token_errors = ((all_reconstructed - all_targets) ** 2).mean(dim=-1)
                    masked_errors = token_errors * all_masks
                    self.writer.add_histogram("reconstruction/token_errors", masked_errors[all_masks > 0], epoch)
                    self.writer.add_histogram("weights/original", all_targets[all_masks.unsqueeze(-1).expand_as(all_targets) > 0], epoch)
                    self.writer.add_histogram("weights/reconstructed", all_reconstructed[all_masks.unsqueeze(-1).expand_as(all_reconstructed) > 0], epoch)

                    # embedding projector
                    self._log_embeddings_to_projector(all_latents_tensor, all_behavior_labels, epoch)

                    # weight heatmap images
                    num_samples = viz_config.get("num_image_samples", 4)
                    for i in range(min(num_samples, all_reconstructed.size(0))):
                        orig = all_targets[i]
                        recon = all_reconstructed[i]
                        mask = all_masks[i]

                        valid_len = int(mask.sum().item())
                        orig = orig[:valid_len]
                        recon = recon[:valid_len]

                        def normalize_for_viz(x):
                            x_min, x_max = x.min(), x.max()
                            if x_max - x_min > 0:
                                return (x - x_min) / (x_max - x_min)
                            return x * 0

                        orig_norm = normalize_for_viz(orig)
                        recon_norm = normalize_for_viz(recon)
                        diff = torch.abs(orig - recon)
                        diff_norm = normalize_for_viz(diff)

                        self.writer.add_image(f"weights/sample_{i}/original", orig_norm.T.unsqueeze(0), epoch)
                        self.writer.add_image(f"weights/sample_{i}/reconstructed", recon_norm.T.unsqueeze(0), epoch)
                        self.writer.add_image(f"weights/sample_{i}/error", diff_norm.T.unsqueeze(0), epoch)

                    # Loss landscape visualization
                    landscape_config = viz_config.get("loss_landscape", {})
                    if landscape_config.get("enabled", False):
                        landscape_interval = landscape_config.get("log_interval", 10)
                        if epoch % landscape_interval == 0:
                            # Get a batch for landscape computation
                            landscape_batch = next(iter(self.val_loader))
                            self._log_loss_landscape(epoch, landscape_batch)

        return metrics

    def _evaluate_test_set(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_reconstructed = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Set Evaluation"):
                encoder_input = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                decoder_target = batch["decoder_target"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                behavior_labels = batch.get("behavior_labels", None)
                batch_size = encoder_input.size(0)

                # extract functional loss data if available
                original_shapes = batch.get("original_shapes", None)
                model_configs = batch.get("model_config", None)
                test_inputs = batch.get("test_inputs", None)
                arch_specs = batch.get("arch_spec", None)
                if test_inputs is not None:
                    test_inputs = test_inputs.to(self.device)

                # calculate loss using helper method
                loss, reconstructed, _ = self._calculate_loss(
                    encoder_input,
                    encoder_mask,
                    decoder_target,
                    decoder_mask,
                    arch_specs=arch_specs,
                    behavior_labels=behavior_labels,
                    original_shapes=original_shapes,
                    model_configs=model_configs,
                    test_inputs=test_inputs,
                )

                # Decoder always outputs weights only (architecture bypasses latent)
                weights_only_dim = getattr(self.model, "weights_only_dim", None)
                if weights_only_dim is not None:
                    decoder_target_for_metrics = decoder_target[:, :, :weights_only_dim]
                else:
                    decoder_target_for_metrics = decoder_target

                for i in range(reconstructed.size(0)):
                    all_reconstructed.append(reconstructed[i : i + 1].cpu())
                    all_targets.append(decoder_target_for_metrics[i : i + 1].cpu())
                    all_masks.append(decoder_mask[i : i + 1].cpu())

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # pad all tensors to consistent dimension before concatenating
        max_token_dim = max(t.size(2) for t in all_reconstructed)

        padded_reconstructed = []
        padded_targets = []
        for recon, target in zip(all_reconstructed, all_targets):
            if recon.size(2) < max_token_dim:
                pad_size = max_token_dim - recon.size(2)
                recon = torch.nn.functional.pad(recon, (0, pad_size))
                target = torch.nn.functional.pad(target, (0, pad_size))
            padded_reconstructed.append(recon)
            padded_targets.append(target)

        all_reconstructed = torch.cat(padded_reconstructed, dim=0)
        all_targets = torch.cat(padded_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        metrics = compute_reconstruction_metrics(
            all_reconstructed, all_targets, all_masks, self.config
        )
        metrics["loss"] = total_loss / total_samples

        return metrics

    def _log_epoch(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train - {format_metrics_for_logging(train_metrics, 'train_')}")
        logger.info(f"  Val   - {format_metrics_for_logging(val_metrics, 'val_')}")

        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train_epoch/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val_epoch/{key}", value, epoch)

    def _is_best(self, val_metrics: Dict[str, float]):
        metric_value = val_metrics.get(self.monitor_metric, val_metrics.get("loss"))
        if metric_value is None:
            return False
        is_best = False
        if self.mode == "max":
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

    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        if not self.early_stopping_config["enabled"]:
            return False
        metric_value = val_metrics.get(self.monitor_metric, val_metrics.get("loss"))
        if metric_value is None:
            return False
        improved = False
        if self.mode == "max":
            improved = metric_value > self.best_metric
        else:
            improved = metric_value < self.best_metric
        if not improved:
            self.patience_counter += 1
            logger.info(
                f"No improvement in {self.monitor_metric} for {self.patience_counter} epochs (patience={self.patience})"
            )
            if self.patience_counter >= self.patience:
                return True
        return False

    def _save_checkpoint(
        self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False
    ):
        if not self.checkpoint_dir:
            return
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
            "best_metric": self.best_metric,
            "config": self.config,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best checkpoint to {checkpoint_path}")

    def _finalize_hub_push(self, test_metrics: Optional[Dict[str, float]] = None):
        if not self.hub_enabled or not self.hub_repo:
            return
        try:
            logger.info("Finalizing HuggingFace Hub push (encoder and decoder)...")
            operations = []

            self._create_model_card(test_metrics)
            readme_path = self.checkpoint_dir / "README.md"
            if readme_path.exists():
                operations.append(
                    CommitOperationAdd(
                        path_in_repo="README.md", path_or_fileobj=str(readme_path)
                    )
                )

            arch_type = self.config["architecture"]["type"]

            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                # save encoder
                encoder_checkpoint = {
                    "encoder_state_dict": self.model.encoder.state_dict(),
                    "config": self.config,
                    "tokenizer_config": self.tokenizer.get_config(),
                    "latent_dim": self.model.latent_dim,
                    "architecture_type": arch_type,
                }
                encoder_path = self.checkpoint_dir / "encoder.pt"
                torch.save(encoder_checkpoint, encoder_path)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo="encoder.pt", path_or_fileobj=str(encoder_path)
                    )
                )
                logger.info("Added encoder.pt to upload queue")

                # save decoder
                decoder_checkpoint = {
                    "decoder_state_dict": self.model.decoder.state_dict(),
                    "config": self.config,
                    "tokenizer_config": self.tokenizer.get_config(),
                    "latent_dim": self.model.latent_dim,
                    "architecture_type": arch_type,
                }
                decoder_path = self.checkpoint_dir / "decoder.pt"
                torch.save(decoder_checkpoint, decoder_path)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo="decoder.pt", path_or_fileobj=str(decoder_path)
                    )
                )
                logger.info("Added decoder.pt to upload queue")
            else:
                logger.warning(
                    "Model does not have separate encoder/decoder attributes"
                )

            # config
            config_sanitized = copy.deepcopy(self.config)
            if "hub" in config_sanitized and "token" in config_sanitized["hub"]:
                config_sanitized["hub"]["token"] = "<REDACTED>"
            config_path = self.checkpoint_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_sanitized, f, default_flow_style=False)
            operations.append(
                CommitOperationAdd(
                    path_in_repo="config.yaml", path_or_fileobj=str(config_path)
                )
            )

            # tokenizer
            tokenizer_config = self.tokenizer.get_config()
            tokenizer_path = self.checkpoint_dir / "tokenizer_config.json"
            with open(tokenizer_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            operations.append(
                CommitOperationAdd(
                    path_in_repo="tokenizer_config.json",
                    path_or_fileobj=str(tokenizer_path),
                )
            )

            # push all files
            if operations:
                self.hf_api.create_commit(
                    repo_id=self.hub_repo,
                    repo_type="model",
                    operations=operations,
                    commit_message="Upload weight-space autoencoder (encoder + decoder) and configuration",
                )

            # upload tb logs
            if self.hub_config.get("push_logs", True):
                if self.log_dir and self.log_dir.exists():
                    self.hf_api.upload_folder(
                        folder_path=str(self.log_dir),
                        path_in_repo="runs",
                        repo_id=self.hub_repo,
                        repo_type="model",
                        ignore_patterns=["*.tmp", "*.lock"],
                    )

            logger.info(
                f"Completed HuggingFace Hub push to: https://huggingface.co/{self.hub_repo}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to finalize HuggingFace Hub push: {e}", exc_info=True
            )

    def _create_model_card(self, test_metrics: Optional[Dict[str, float]] = None):
        dataset_name = self.config["dataset"]["hf_dataset"]
        arch_type = self.config["architecture"]["type"]

        model_card = f"""---
tags:
- weight-space-learning
- neural-network-autoencoder
- autoencoder
- {arch_type}
datasets:
- {dataset_name}
---

# Weight-Space Autoencoder ({arch_type.upper()})

This model is a weight-space autoencoder trained on neural network activation weights/signatures.
It includes both an encoder (compresses weights into latent representations) and a decoder (reconstructs weights from latent codes).

## Model Description

- **Architecture**: {arch_type.capitalize()} encoder-decoder
- **Training Dataset**: {dataset_name}
- **Input Mode**: {self.config["dataset"]["input_mode"]}
- **Latent Dimension**: {self.model.latent_dim}

## Tokenization

- **Granularity**: {self.tokenizer.granularity}
- **Max Tokens**: {self.tokenizer.max_tokens}

## Training Config

- **Loss Functions**: {', '.join(self.loss_weights.keys())}
- **Optimizer**: {self.config["training"]["optimizer"]}
- **Learning Rate**: {self.config["training"]["learning_rate"]}
- **Batch Size**: {self.config["training"]["batch_size"]}
"""

        if test_metrics:
            model_card += f"""
## Performance Metrics (Test Set)

- **MSE**: {test_metrics.get("mse", 0):.6f}
- **MAE**: {test_metrics.get("mae", 0):.6f}
- **RMSE**: {test_metrics.get("rmse", 0):.6f}
- **Cosine Similarity**: {test_metrics.get("cosine_similarity", 0):.4f}
- **R² Score**: {test_metrics.get("r2_score", 0):.4f}
"""

        model_card_path = self.checkpoint_dir / "README.md"
        with open(model_card_path, "w") as f:
            f.write(model_card)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint
