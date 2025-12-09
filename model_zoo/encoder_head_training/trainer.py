import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import yaml
from huggingface_hub import HfApi, create_repo

from .model import EncoderWithHead
from .evaluator import evaluate_model

logger = logging.getLogger(__name__)


class EncoderHeadTrainer:
    def __init__(self, config: Dict[str, Any], model: EncoderWithHead, train_loader, val_loader, test_loader,
                 device: str = 'cpu', output_dir: Path = Path('output')):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_config = config['training']
        self.task_type = config['task']['type']
        self.task_config = config['task']
        self.hub_config = config.get('huggingface_hub', {})

        # optimizer
        opt_config = self.training_config['optimizer']
        opt_type = opt_config['type']
        lr = opt_config['lr']
        if self.model.freeze_encoder:
            params = self.model.prediction_head.parameters()
            logger.info("Optimizing head only (frozen encoder)")
        else:
            encoder_lr = opt_config.get('encoder_lr_multiplier', 0.1) * lr
            head_lr = lr
            params = [
                {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
                {'params': self.model.prediction_head.parameters(), 'lr': head_lr}
            ]
            logger.info(f"Differential LR - Encoder: {encoder_lr}, Head: {head_lr}")
        if opt_type == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=opt_config.get('weight_decay', 0.0))
        elif opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
        elif opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=opt_config.get('momentum', 0.9), weight_decay=opt_config.get('weight_decay', 0.0))
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        # scheduler
        sched_config = self.training_config.get('scheduler', {})
        sched_type = sched_config.get('type', 'none')
        if sched_type == 'none':
            self.scheduler = None
        elif sched_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['epochs']
            )
        elif sched_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self._is_lower_better() else 'max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
        
        # loss
        if self.task_type == 'pattern_classification':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == 'accuracy_prediction':
            self.criterion = nn.MSELoss()
        elif self.task_type == 'hyperparameter_prediction':
            self.criterion = None # handled in _compute_loss
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # tb
        log_dir = self.output_dir / 'runs'
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        logger.info(f"TensorBoard logs: {log_dir}")

        # state
        self.current_epoch = 0
        self.best_val_metric = float('inf') if self._is_lower_better() else float('-inf')
        self.patience_counter = 0

        # hf
        if self.hub_config.get('push_to_hub', False):
            self.hf_api = HfApi()
            self.hub_repo = self.hub_config['repo_id']

    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        if self.task_type in ['pattern_classification', 'accuracy_prediction']:
            return self.criterion(outputs, targets)

        elif self.task_type == 'hyperparameter_prediction':
            # multi-task loss
            total_loss = 0.0
            loss_weights = self.task_config.get('loss_weights', {})
            # continuous (mse):
            for name in self.task_config.get('continuous_targets', {}).keys():
                pred_key = f'continuous_{name}'
                pred = outputs[pred_key]
                target = targets[pred_key]
                loss = F.mse_loss(pred, target)
                weight = loss_weights.get(name, 1.0)
                total_loss += weight * loss
            # discrete targets (cross-entropy):
            for name in self.task_config.get('discrete_targets', {}).keys():
                pred_key = f'discrete_{name}'
                logits = outputs[pred_key]
                target = targets[pred_key]
                loss = F.cross_entropy(logits, target)
                weight = loss_weights.get(name, 1.0)
                total_loss += weight * loss
            return total_loss

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _is_lower_better(self):
        metric = self.training_config.get('early_stopping_metric', 'loss')
        return 'loss' in metric or 'error' in metric or 'mse' in metric

    def train(self):
        logger.info("Starting training...")
        epochs = self.training_config['epochs']

        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch()
            val_metrics = self._validate()
            self._log_metrics(train_loss, val_metrics, epoch)
            # step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            # early stopping check
            should_stop = self._check_early_stopping(val_metrics)
            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # final eval on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(self.model, self.test_loader, self.task_type, self.task_config, self.device)
        self._log_test_metrics(test_metrics)

        # save final checkpoint
        self._save_checkpoint('final')

        # upload to hub
        if self.hub_config.get('push_to_hub', False):
            self._push_to_hub()

        logger.info("Training complete!")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in progress_bar:
            if 'latent' in batch: # encodings cached
                latent = batch['latent'].to(self.device)
                targets = self._move_targets_to_device(batch['target'])
                outputs = self.model.prediction_head(latent)
            else:
                tokens = batch['tokens'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = self._move_targets_to_device(batch['target'])
                outputs = self.model(tokens, mask)

            # backward pass
            loss = self._compute_loss(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            if self.training_config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config['grad_clip'])

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if 'latent' in batch:
                    latent = batch['latent'].to(self.device)
                    targets = self._move_targets_to_device(batch['target'])
                    outputs = self.model.prediction_head(latent)
                else:
                    tokens = batch['tokens'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    targets = self._move_targets_to_device(batch['target'])
                    outputs = self.model(tokens, mask)

                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        metrics = evaluate_model(self.model, self.val_loader, self.task_type, self.task_config, self.device)
        metrics['loss'] = total_loss / num_batches

        return metrics

    def _move_targets_to_device(self, targets):
        if isinstance(targets, dict):
            return {k: v.to(self.device) for k, v in targets.items()}
        else:
            return targets.to(self.device)

    def _log_metrics(self, train_loss: float, val_metrics: Dict[str, float], epoch: int):
        # tb
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        for key, value in val_metrics.items():
            if key != 'loss':
                self.writer.add_scalar(f'Val/{key}', value, epoch)
        # console
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}")
        if self.task_type == 'pattern_classification':
            logger.info(f"  Val Metrics - Hamming: {val_metrics.get('hamming_loss', 0):.4f}, "
                       f"Subset Acc: {val_metrics.get('subset_accuracy', 0):.4f}, "
                       f"Macro F1: {val_metrics.get('macro_f1', 0):.4f}")
        elif self.task_type == 'accuracy_prediction':
            logger.info(f"  Val Metrics - MSE: {val_metrics.get('mse', 0):.4f}, "
                       f"MAE: {val_metrics.get('mae', 0):.4f}, "
                       f"R²: {val_metrics.get('r2', 0):.4f}")

    def _log_test_metrics(self, test_metrics: Dict[str, float]):
        logger.info("Test Set Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            self.writer.add_scalar(f'Test/{key}', value, 0)

    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        patience = self.training_config.get('patience', 10)
        metric_name = self.training_config.get('early_stopping_metric', 'loss')
        current_metric = val_metrics[metric_name]

        if self._is_lower_better():
            improved = current_metric < self.best_val_metric
        else:
            improved = current_metric > self.best_val_metric

        if improved:
            self.best_val_metric = current_metric
            self.patience_counter = 0
            checkpoint_path = self.output_dir / 'best_checkpoint.pt'
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'prediction_head_state_dict': self.model.prediction_head.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_metric': self.best_val_metric,
                'config': self.config
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"New best {metric_name}: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement ({self.patience_counter}/{patience})")

        return self.patience_counter >= patience

    def _save_checkpoint(self, checkpoint_name: str):
        checkpoint_path = self.output_dir / f'{checkpoint_name}_checkpoint.pt'
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {checkpoint_name} checkpoint to {checkpoint_path}")

    def _push_to_hub(self):
        logger.info("Uploading to HuggingFace Hub...")
        try:
            create_repo(self.hub_repo, repo_type='model', exist_ok=True)
            logger.info(f"Repository ready: {self.hub_repo}")
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            return

        # Full model (encoder + prediction head)
        model_path = self.output_dir / 'full_model.pt'

        # Prepare checkpoint with all metadata
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # Complete EncoderWithHead
            'encoder_state_dict': self.model.encoder.state_dict(),  # Separate for convenience
            'prediction_head_state_dict': self.model.prediction_head.state_dict(),  # Separate for convenience
            'task_type': self.task_type,
            'task_config': self.task_config,
            'freeze_encoder': self.model.freeze_encoder,
            'encoder_repo_id': self.config['encoder']['repo_id']  # Attribution
        }

        # Add encoder metadata if available in config
        encoder_config = self.config.get('encoder', {})
        if 'latent_dim' in encoder_config:
            checkpoint['latent_dim'] = encoder_config['latent_dim']
        if 'architecture_type' in encoder_config:
            checkpoint['architecture_type'] = encoder_config['architecture_type']
        if 'tokenizer_config' in encoder_config:
            checkpoint['tokenizer_config'] = encoder_config['tokenizer_config']
        if 'full_config' in encoder_config:
            checkpoint['encoder_full_config'] = encoder_config['full_config']

        torch.save(checkpoint, model_path)
        self.hf_api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo='full_model.pt',
            repo_id=self.hub_repo,
            repo_type='model'
        )
        logger.info("Uploaded full_model.pt (encoder + prediction head)")

        # config
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        self.hf_api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo='config.yaml',
            repo_id=self.hub_repo,
            repo_type='model'
        )
        logger.info("Uploaded config.yaml")

        # tb logs
        if self.hub_config.get('push_logs', True):
            if self.log_dir and self.log_dir.exists():
                self.hf_api.upload_folder(
                    folder_path=str(self.log_dir),
                    path_in_repo='runs',
                    repo_id=self.hub_repo,
                    repo_type='model',
                    ignore_patterns=['*.tmp', '*.lock']
                )
                logger.info("TensorBoard logs uploaded")

        self._create_model_card()

        logger.info(f"Successfully uploaded to: https://huggingface.co/{self.hub_repo}")

    def _create_model_card(self):
        encoder_repo = self.config['encoder']['repo_id']
        freeze_status = "frozen" if self.model.freeze_encoder else "fine-tuned"
        readme_content = f"""---
tags:
- weight-space
- neural-network-analysis
- {self.task_type}
- encoder-with-head
datasets:
- {self.config['data']['dataset_repo_id']}
---

# {self.task_type.replace('_', ' ').title()} Model (Full Encoder + Head)

Complete model containing both the weight-space encoder and task-specific prediction head.

## Model Details

- **Task Type**: {self.task_type}
- **Base Encoder**: [{encoder_repo}](https://huggingface.co/{encoder_repo}) ({freeze_status})
- **Training Dataset**: {self.config['data']['dataset_repo_id']}
- **Model Components**: Complete EncoderWithHead (encoder + prediction head)

## Task Configuration

```yaml
{yaml.dump(self.task_config, default_flow_style=False)}
```

## Usage

```python
from model_zoo.encoder_head_training import EncoderWithHead, create_tokenizer_from_config
import torch

# Load full model checkpoint
checkpoint = torch.load('full_model.pt')

# The checkpoint contains:
# - model_state_dict: Complete EncoderWithHead state
# - encoder_state_dict: Encoder only (for convenience)
# - prediction_head_state_dict: Head only (for convenience)
# - task_type, task_config: Task information
# - tokenizer_config: Tokenizer configuration (if available)

# To use, you'll need to reconstruct the encoder and head architecture
# then load the model_state_dict

# Example inference (after reconstructing model):
# model.eval()
# with torch.no_grad():
#     predictions = model(tokens, mask)
```

## Training Configuration

```yaml
{yaml.dump(self.training_config, default_flow_style=False)}
```

## Notes

This model contains the complete encoder and prediction head. You can:
- Load the full `model_state_dict` to get both components together
- Load individual `encoder_state_dict` or `prediction_head_state_dict` if needed
- Original encoder available at: [{encoder_repo}](https://huggingface.co/{encoder_repo})
"""

        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        self.hf_api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo='README.md',
            repo_id=self.hub_repo,
            repo_type='model'
        )
        logger.info("Uploaded README.md")
