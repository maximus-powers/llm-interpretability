import torch
import torch.nn as nn
import logging
import os
import copy
import json
import yaml
import subprocess
import atexit
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

from .evaluator import (
    compute_reconstruction_metrics,
    print_metrics,
    format_metrics_for_logging,
)
from .tokenizer import WeightTokenizer
from .data_loader import augment_tokenized_weights
from .losses import (
    ReconstructionLoss,
    CombinedReconstructionLoss,
    GammaContrastReconLoss,
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
        batch_size = config["training"]["batch_size"]

        # loss
        loss_type = config["loss"]["type"]
        self.is_contrastive_loss = False
        self.is_combined_loss = False
        if loss_type in ["mse", "mae", "cosine"]:
            self.criterion = ReconstructionLoss(config, loss_type=loss_type)
        elif loss_type == "combined":
            self.criterion = CombinedReconstructionLoss(config)
            self.is_combined_loss = True
        elif loss_type == "contrastive":
            if batch_size is None or device is None:
                raise ValueError(
                    "batch_size and device are required for contrastive loss"
                )
            self.criterion = GammaContrastReconLoss(config, batch_size, device)
            self.is_contrastive_loss = True
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        if self.is_contrastive_loss:
            loss_cfg = config["loss"]
            self.augmentation_type = loss_cfg.get("augmentation_type", "noise")
            self.noise_std = loss_cfg.get("noise_std", 0.01)
            self.dropout_prob = loss_cfg.get("dropout_prob", 0.1)
            logger.info(
                f"Contrastive loss enabled with augmentation: {self.augmentation_type}"
            )

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.scheduler = None
        if config["training"]["lr_scheduler"]["enabled"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=config["training"]["early_stopping"]["mode"],
                patience=config["training"]["lr_scheduler"]["patience"],
                factor=config["training"]["lr_scheduler"]["factor"],
                min_lr=config["training"]["lr_scheduler"]["min_lr"],
            )

        # logs
        run_dir = Path(config.get("run_dir", "."))
        self.writer = None
        self.tensorboard_process = None
        self.log_dir = None
        if config["logging"]["tensorboard"]["enabled"]:
            self.log_dir = run_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
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

    def train(self):
        logger.info("Starting training")

        for epoch in range(self.config["training"]["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            self._log_epoch(epoch, train_metrics, val_metrics)
            if self.scheduler:
                monitor_value = val_metrics.get("loss")
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(f"Learning rate: {current_lr:.6f}")
            if self._is_best(val_metrics):
                logger.info(
                    f"New best model! {self.monitor_metric}={val_metrics.get('loss', 0):.6f}"
                )
                self._save_checkpoint(epoch, val_metrics, is_best=True)
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

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            tokens = batch["tokenized_weights"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            batch_size = tokens.size(0)
            self.optimizer.zero_grad()

            if self.is_contrastive_loss:
                tokens_i, mask_i = augment_tokenized_weights(
                    tokens,
                    mask,
                    augmentation_type=self.augmentation_type,
                    noise_std=self.noise_std,
                    dropout_prob=self.dropout_prob,
                )
                tokens_j, mask_j = augment_tokenized_weights(
                    tokens,
                    mask,
                    augmentation_type=self.augmentation_type,
                    noise_std=self.noise_std,
                    dropout_prob=self.dropout_prob,
                )
                z_i = self.model.encode(tokens_i, mask_i)
                z_j = self.model.encode(tokens_j, mask_j)
                y_i = self.model.decode(z_i, tokens.size(1))
                y_j = self.model.decode(z_j, tokens.size(1))
                y = torch.cat([y_i, y_j], dim=0)
                t = torch.cat([tokens_i, tokens_j], dim=0)
                mask_combined = torch.cat([mask_i, mask_j], dim=0)
                loss, loss_contrast, loss_recon = self.criterion(
                    z_i, z_j, y, t, mask_combined
                )
                loss_components = {
                    "loss_contrast": loss_contrast.item(),
                    "loss_recon": loss_recon.item(),
                }
                for key, value in loss_components.items():
                    if key not in all_loss_components:
                        all_loss_components[key] = 0
                    all_loss_components[key] += value * batch_size

            elif self.is_combined_loss:
                reconstructed = self.model(tokens, mask)
                loss, loss_components = self.criterion.compute(
                    reconstructed, tokens, mask
                )
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

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Epoch {epoch + 1}"):
                tokens = batch["tokenized_weights"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                batch_size = tokens.size(0)

                if self.is_contrastive_loss:
                    tokens_i, mask_i = augment_tokenized_weights(
                        tokens,
                        mask,
                        augmentation_type=self.augmentation_type,
                        noise_std=self.noise_std,
                        dropout_prob=self.dropout_prob,
                    )
                    tokens_j, mask_j = augment_tokenized_weights(
                        tokens,
                        mask,
                        augmentation_type=self.augmentation_type,
                        noise_std=self.noise_std,
                        dropout_prob=self.dropout_prob,
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
        metrics["loss"] = total_loss / total_samples

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
                tokens = batch["tokenized_weights"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                batch_size = tokens.size(0)

                if self.is_contrastive_loss:
                    tokens_i, mask_i = augment_tokenized_weights(
                        tokens,
                        mask,
                        augmentation_type=self.augmentation_type,
                        noise_std=self.noise_std,
                        dropout_prob=self.dropout_prob,
                    )
                    tokens_j, mask_j = augment_tokenized_weights(
                        tokens,
                        mask,
                        augmentation_type=self.augmentation_type,
                        noise_std=self.noise_std,
                        dropout_prob=self.dropout_prob,
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

                else:
                    reconstructed = self.model(tokens, mask)
                    if self.is_combined_loss:
                        loss, _ = self.criterion.compute(reconstructed, tokens, mask)
                    else:
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
        metric_value = val_metrics.get("loss")
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
        metric_value = val_metrics.get("loss")
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
                f"No improvement for {self.patience_counter} epochs (patience={self.patience})"
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

- **Chunk Size**: {self.tokenizer.chunk_size} weight values per token
- **Max Tokens**: {self.tokenizer.max_tokens}
- **Metadata**: {self.tokenizer.include_metadata}

## Training Config

- **Loss Function**: {self.config["loss"]["type"]}
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
