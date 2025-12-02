import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo
import subprocess
import atexit
import yaml
import copy

from .evaluator import compute_metrics, print_metrics, compute_class_weights

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], train_loader, val_loader, device: str, all_patterns: list = None, test_loader=None):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.all_patterns = all_patterns or config['dataset']['patterns']
        pos_weight = None
        if config['training'].get('pos_weight') is None:
            pos_weight = compute_class_weights(train_loader, config['model']['output']['num_patterns']).to(device)
        elif config['training']['pos_weight'] is not None:
            pos_weight = torch.tensor(config['training']['pos_weight'], dtype=torch.float32).to(device)

        # loss fn (val loss doesn't use pos weight, train does)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.val_criterion = nn.BCEWithLogitsLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # lr scheduler
        self.scheduler = None
        if config['training']['lr_scheduler']['enabled']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=config['training']['early_stopping']['mode'],
                patience=config['training']['lr_scheduler']['patience'],
                factor=config['training']['lr_scheduler']['factor'],
                min_lr=config['training']['lr_scheduler']['min_lr']
            )

        # tensorboard
        self.writer = None
        self.tensorboard_process = None
        if config['logging']['tensorboard']['enabled']:
            log_dir = Path(config['logging']['tensorboard']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.tensorboard_process = self._start_tensorboard_server()

        # early stopping
        self.early_stopping_config = config['training']['early_stopping']
        self.monitor_metric = self.early_stopping_config['monitor']
        self.mode = self.early_stopping_config['mode']
        self.patience = self.early_stopping_config['patience']
        if self.mode == 'max':
            self.best_metric = float('-inf')
        else:
            self.best_metric = float('inf')
        self.patience_counter = 0

        # checkpointing
        self.checkpoint_config = config['logging']['checkpoint']
        if self.checkpoint_config['enabled']:
            checkpoint_dir = Path(self.checkpoint_config['save_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = None
        self.global_step = 0

        # hf hub
        self.hub_config = config.get('hub', {})
        self.hub_enabled = self.hub_config.get('enabled', False)
        self.hub_repo = None
        if self.hub_enabled:
            try:
                repo_id = self.hub_config.get('repo_id')
                if not repo_id:
                    logger.error("HuggingFace Hub enabled but repo_id not specified in config")
                    self.hub_enabled = False
                else:
                    token = self.hub_config.get('token') or os.environ.get('HF_TOKEN')
                    if not token:
                        logger.warning("No HuggingFace token found. Set hub.token in config", exc_info=True)
                        self.hub_enabled = False
                    else:
                        self.hf_api = HfApi(token=token)
                        try:
                            create_repo(repo_id=repo_id, token=token, private=self.hub_config.get('private', True), exist_ok=True)
                            self.hub_repo = repo_id
                            logger.info(f"HuggingFace Hub integration enabled: {repo_id}")
                        except Exception as e:
                            logger.error(f"Failed to create HuggingFace repo: {e}", exc_info=True)
                            self.hub_enabled = False
            except ImportError:
                logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
                self.hub_enabled = False

    def train(self):
        logger.info("Starting training")
        for epoch in range(self.config['training']['epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            # train
            train_metrics = self._train_epoch(epoch)
            # val
            val_metrics = self._validate_epoch(epoch)
            # logs
            self._log_epoch(epoch, train_metrics, val_metrics)

            # lr schedule step
            if self.scheduler:
                metric_key = self.monitor_metric.replace('val_', '')
                monitor_value = val_metrics.get(metric_key)
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Learning rate: {current_lr:.6f}")

            # checkpoints
            if self._is_best(val_metrics):
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # eval on test set
        test_metrics = None
        if self.test_loader is not None:
            logger.info("\nEvaluating on test set...")
            test_metrics = self._evaluate_test_set()
            logger.info("\nTest Set Metrics:")
            print_metrics(test_metrics, prefix="test_")

        # push to hub
        if self.hub_enabled:
            self._finalize_hub_push(test_metrics)
        if self.writer:
            self.writer.close()

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}")

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            # forward
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            # backward pass
            loss.backward()
            self.optimizer.step()
            # metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = (torch.sigmoid(logits) >
                    self.config['evaluation']['decision_threshold']).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            # progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            # tb logging
            if self.writer and batch_idx % self.config['logging']['tensorboard'].get('log_interval', 10) == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
            self.global_step += 1
        # metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_preds, all_labels, self.config, self.all_patterns)
        metrics['loss'] = total_loss / total_samples # weighted

        return metrics

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc=f"Val Epoch {epoch+1}"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                batch_size = labels.size(0)
                logits = self.model(inputs)
                loss = self.val_criterion(logits, labels) # unweighted loss for val
                total_loss += loss.item() * batch_size # weighted by batch size
                total_samples += batch_size
                preds = (torch.sigmoid(logits) > self.config['evaluation']['decision_threshold']).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_preds, all_labels, self.config, self.all_patterns)
        metrics['loss'] = total_loss / total_samples
        return metrics

    def _evaluate_test_set(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Test Evaluation"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                batch_size = labels.size(0)
                logits = self.model(inputs)
                loss = self.val_criterion(logits, labels) # unweighted loss for val
                total_loss += loss.item() * batch_size # weighted by batch size
                total_samples += batch_size
                preds = (torch.sigmoid(logits) >
                        self.config['evaluation']['decision_threshold']).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_preds, all_labels, self.config, self.all_patterns)
        metrics['loss'] = total_loss / total_samples

        return metrics

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        # logger.info("----Training Metrics:")
        # print_metrics(train_metrics, prefix="train_")
        # logger.info("----Validation Metrics:")
        # print_metrics(val_metrics, prefix="val_")

        # tb logging
        if self.writer:
            for metric_name, value in train_metrics.items():
                if not metric_name.startswith('pattern_'):
                    self.writer.add_scalar(f'train/{metric_name}', value, epoch)
            for metric_name, value in val_metrics.items():
                if not metric_name.startswith('pattern_'):
                    self.writer.add_scalar(f'val/{metric_name}', value, epoch)

    def _is_best(self, val_metrics: Dict[str, float]):
        metric_key = self.monitor_metric.replace('val_', '')
        current_value = val_metrics.get(metric_key)
        if current_value is None:
            logger.warning(f"Monitor metric '{metric_key}' not found in validation metrics")
            return False
        is_best = False
        if self.mode == 'max':
            if current_value > self.best_metric:
                self.best_metric = current_value
                is_best = True
        else:  # min mode
            if current_value < self.best_metric:
                self.best_metric = current_value
                is_best = True
        if is_best:
            logger.info(f"New best {self.monitor_metric}: {self.best_metric:.4f}")
        return is_best

    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        if not self.early_stopping_config['enabled']:
            return False
        metric_key = self.monitor_metric.replace('val_', '')
        current_value = val_metrics.get(metric_key)
        if current_value is None:
            return False

        # check if improved this epoch
        improved = False
        if self.mode == 'max':
            improved = current_value >= self.best_metric
        else:
            improved = current_value <= self.best_metric
        if not improved:
            self.patience_counter += 1
            logger.info(f"Early stopping patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                return True
        else:
            self.patience_counter = 0 # reset

        return False

    def _start_tensorboard_server(self):
        if not self.config['logging']['tensorboard'].get('auto_launch', False):
            return None
        log_dir = Path(self.config['logging']['tensorboard']['log_dir'])
        port = self.config['logging']['tensorboard'].get('port', 6006)
        try:
            tensorboard_process = subprocess.Popen(
                ['tensorboard', '--logdir', str(log_dir.absolute()), '--port', str(port), '--bind_all'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            def cleanup_tensorboard():
                if tensorboard_process.poll() is None:
                    tensorboard_process.terminate()
                    tensorboard_process.wait()
            atexit.register(cleanup_tensorboard)
            logger.info(f"TensorBoard server started! Access at: http://localhost:{port}")
            return tensorboard_process
        except FileNotFoundError:
            logger.warning("TensorBoard not installed. Install with: pip install tensorboard")
            return None
        except Exception as e:
            logger.warning(f"Failed to start TensorBoard server: {e}")
            return None

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        if not self.checkpoint_config['enabled']:
            return
        if self.checkpoint_config.get('save_best_only', True) and not is_best:
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
            logger.info(f"Saved best model checkpoint to: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to: {checkpoint_path}")

    def _create_model_card(self, test_metrics: Optional[Dict[str, float]] = None):
        if not self.hub_enabled or not self.hub_repo:
            return
        dataset_name = self.config['dataset']['hf_dataset']
        dataset_link = f"https://huggingface.co/datasets/{dataset_name}"
        model_card = f"""---
tags:
- pattern-classification
- multi-label-classification
datasets:
- {dataset_name}
---

# Pattern Classifier

This model was trained to classify which patterns a subject model was trained on, based on neuron activation signatures.

## Dataset

- **Training Dataset**: [{dataset_name}]({dataset_link})
- **Input Mode**: {self.config['dataset']['input_mode']}
- **Number of Patterns**: {len(self.all_patterns)}

## Patterns

The model predicts which of the following {len(self.all_patterns)} patterns the subject model was trained to classify as positive:

"""
        for i, pattern in enumerate(self.all_patterns, 1):
            model_card += f"{i}. `{pattern}`\n"

        model_card += "\n## Model Architecture\n\n"
        model_card += f"- **Signature Encoder**: {self.config['model']['signature_encoder'].get('hidden_dims', 'auto')}\n"
        model_card += f"- **Activation**: {self.config['model']['signature_encoder'].get('activation', 'relu')}\n"
        model_card += f"- **Dropout**: {self.config['model']['signature_encoder'].get('dropout', 0.3)}\n"
        model_card += f"- **Batch Normalization**: {self.config['model'].get('use_batch_norm', False)}\n"

        model_card += "\n## Training Configuration\n\n"
        model_card += f"- **Optimizer**: {self.config['training']['optimizer']}\n"
        model_card += f"- **Learning Rate**: {self.config['training']['learning_rate']}\n"
        model_card += f"- **Batch Size**: {self.config['training']['batch_size']}\n"
        model_card += "- **Loss Function**: BCE with Logits (with pos_weight for training, unweighted for validation)\n"

        if test_metrics:
            model_card += "\n## Test Set Performance\n\n"
            model_card += f"- **F1 Macro**: {test_metrics.get('f1_macro', 0):.4f}\n"
            model_card += f"- **F1 Micro**: {test_metrics.get('f1_micro', 0):.4f}\n"
            model_card += f"- **Hamming Accuracy**: {test_metrics.get('accuracy_hamming', 0):.4f}\n"
            model_card += f"- **Exact Match Accuracy**: {test_metrics.get('accuracy_exact_match', 0):.4f}\n"
            model_card += f"- **BCE Loss**: {test_metrics.get('loss', 0):.4f}\n"

            model_card += "\n### Per-Pattern Performance (Test Set)\n\n"
            model_card += "| Pattern | Precision | Recall | F1 Score |\n"
            model_card += "|---------|-----------|--------|----------|\n"
            for pattern in self.all_patterns:
                precision_key = f'pattern_{pattern}_precision'
                recall_key = f'pattern_{pattern}_recall'
                f1_key = f'pattern_{pattern}_f1'
                precision = test_metrics.get(precision_key, 0.0)
                recall = test_metrics.get(recall_key, 0.0)
                f1 = test_metrics.get(f1_key, 0.0)
                model_card += f"| {pattern} | {precision*100:.1f}% | {recall*100:.1f}% | {f1*100:.1f}% |\n"

        model_card_path = Path(self.checkpoint_dir) / 'README.md'
        with open(model_card_path, 'w') as f:
            f.write(model_card)

    def _finalize_hub_push(self, test_metrics: Optional[Dict[str, float]] = None):
        if not self.hub_enabled or not self.hub_repo:
            return
        try:
            from huggingface_hub import CommitOperationAdd

            logger.info("Finalizing HuggingFace Hub push...")

            # Prepare all files to upload in a single commit
            operations = []

            # 1. Model card
            self._create_model_card(test_metrics)
            readme_path = Path(self.checkpoint_dir) / 'README.md'
            if readme_path.exists():
                operations.append(
                    CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(readme_path))
                )
                logger.info("Added README.md to upload queue")

            # 2. Best model checkpoint
            if self.hub_config.get('push_model', True):
                best_checkpoint = Path(self.checkpoint_dir) / 'best_model.pt'
                if best_checkpoint.exists():
                    operations.append(
                        CommitOperationAdd(path_in_repo='best_model.pt', path_or_fileobj=str(best_checkpoint))
                    )
                    logger.info("Added best_model.pt to upload queue")

            # 3. Sanitized config file
            config_sanitized = copy.deepcopy(self.config)
            if 'hub' in config_sanitized and 'token' in config_sanitized['hub']:
                config_sanitized['hub']['token'] = '<REDACTED>'
            config_path = Path(self.checkpoint_dir) / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_sanitized, f)
            operations.append(
                CommitOperationAdd(path_in_repo='config.yaml', path_or_fileobj=str(config_path))
            )
            logger.info("Added config.yaml to upload queue")

            # Upload all files in a single commit to avoid LFS issues
            if operations:
                logger.info(f"Uploading {len(operations)} files in a single commit...")
                self.hf_api.create_commit(
                    repo_id=self.hub_repo,
                    repo_type='model',
                    operations=operations,
                    commit_message="Upload model, config, and documentation"
                )
                logger.info("Successfully uploaded files to Hub")

            # 4. TensorBoard logs (separate upload via folder)
            if self.hub_config.get('push_logs', True):
                log_dir = Path(self.config['logging']['tensorboard']['log_dir'])
                if log_dir.exists():
                    logger.info("Uploading TensorBoard logs...")
                    self.hf_api.upload_folder(
                        folder_path=str(log_dir),
                        path_in_repo='runs',
                        repo_id=self.hub_repo,
                        repo_type='model',
                        ignore_patterns=['*.tmp', '*.lock']
                    )
                    logger.info("TensorBoard logs uploaded to Hub")

            logger.info(f"Completed HuggingFace Hub push to: https://huggingface.co/{self.hub_repo}")

        except Exception as e:
            logger.warning(f"Failed to finalize HuggingFace Hub push: {e}", exc_info=True)


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return {
        'epoch': checkpoint['epoch'],
        'val_metrics': checkpoint.get('val_metrics', {}),
        'best_metric': checkpoint.get('best_metric')
    }
