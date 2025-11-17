import torch
import torch.nn as nn
import logging
import copy
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

logger = logging.getLogger(__name__)

class SubjectModel(nn.Module):
    """
    Simple neural network for sequence binary classification.
    Takes integer token indices (e.g. 7 positions with vocab_size possible values each) and outputs binary classification preds.
    """
    def __init__(self, vocab_size: int = 7, sequence_length: int = 7, num_layers: int = 6, neurons_per_layer: int = 25, activation_type: str = 'relu', dropout_rate: float = 0.1, precision: str = 'float32'):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.input_size = sequence_length 
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.precision = precision

        # set dtype
        if precision == 'float16':
            self.dtype = torch.float16
        elif precision == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # set activation fn
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activations.get(activation_type.lower(), nn.ReLU())
        
        # build network
        layers = []
        # input layer
        layers.append(nn.Linear(self.input_size, neurons_per_layer))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        # ouput layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.network = nn.Sequential(*layers)
        # apply precision setting
        self.to(dtype=self.dtype)

        # store config
        self.config = {
            'vocab_size': vocab_size,
            'sequence_length': sequence_length,
            'num_layers': num_layers,
            'neurons_per_layer': neurons_per_layer,
            'activation_type': activation_type,
            'dropout_rate': dropout_rate,
            'precision': precision,
            'input_size': self.input_size,
            'input_format': 'integer_indices'
        }
        
        logger.debug(f"Created SubjectModel: {num_layers} layers, " f"{neurons_per_layer} neurons/layer, {activation_type} activation")
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def quantize_weights(self, quantization_type: str):
        if quantization_type == 'int8': # scale to [-127, 127] range and round
            for param in self.parameters():
                if param.requires_grad:
                    max_val = param.abs().max()
                    if max_val > 0:
                        scale = 127.0 / max_val
                        param.data = torch.round(param * scale).clamp(-127, 127) / scale
        
        elif quantization_type == 'int4': # scale to [-7, 7] range and round
            for param in self.parameters():
                if param.requires_grad:
                    max_val = param.abs().max()
                    if max_val > 0:
                        scale = 7.0 / max_val
                        param.data = torch.round(param * scale).clamp(-7, 7) / scale

        elif quantization_type == 'ternary': # -1, 0, or +1
            for param in self.parameters():
                if param.requires_grad:
                    threshold = param.abs().max() * 0.2 # if it's 20% of max away from 0, set to 0
                    param.data = torch.where(param.abs() < threshold, torch.zeros_like(param), torch.sign(param))                    

        elif quantization_type == 'binary':
            for param in self.parameters(): # use sign to get +1/-1
                if param.requires_grad:
                    param.data = torch.sign(param.data)
                    param.data[param.data == 0] = 1.0 # 0 gets +1

        else:
            logger.warning(f"Unknown quantization type '{quantization_type}', skipping quantization")
        logger.info(f"Applied {quantization_type} quantization to model weights")
    
    def get_layer_activations(self, x, layer_names: Optional[list] = None):
        # for feature extraction
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        hooks = []
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                if layer_names is None or name in layer_names:
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        with torch.no_grad():
            _ = self.forward(x)
        for hook in hooks:
            hook.remove()
        return activations
    
    def save_model(self, path: str):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': 'SubjectModel'
        }
        torch.save(save_dict, path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"Loaded model from {path}")
        return model

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, examples, vocab=None, vocab_size=None):
        self.examples = examples
        
        if vocab is not None:
            self.vocab = vocab
        elif vocab_size is not None:
            self.vocab = [chr(ord('A') + i) for i in range(vocab_size)]
        else:
            self.vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # default
            
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        logger.debug(f"Created SequenceDataset with {len(examples)} examples, vocab: {self.vocab}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        sequence = example['sequence']
        label = float(example['label'])
        pattern = example.get('pattern', example.get('excluded_pattern', 'unknown'))
        sequence_tensor = self._sequence_to_tensor(sequence)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return sequence_tensor, label_tensor, pattern
    
    def _sequence_to_tensor(self, sequence):
        indices = []
        for token in sequence:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                indices.append(0)  # default to first token if unknown
        return torch.tensor(indices, dtype=torch.float32)

class SubjectModelTrainer:
    def __init__(self, device: str = 'auto', quantization_type: str = 'none'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.quantization_type = quantization_type
        logger.info(f"Initialized trainer with device: {self.device}, quantization: {quantization_type}")
    
    def train_and_evaluate(self, model: SubjectModel, train_loader, val_loader,
                   num_epochs: int = 30, learning_rate: float = 0.001, early_stopping_patience: int = 10,
                   save_path: str = None, log_dir: str = None, selected_patterns: List[str] = None,
                   target_pattern: str = None, stage: str = None, tensorboard_config: Dict[str, Any] = None,
                   checkpoint_config: Dict[str, Any] = None, epoch_offset: int = 0,
                   wait_for_first_improvement: bool = False, min_improvement_threshold: float = 0.0) -> Dict[str, Any]:

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_accuracy_checkpoint = None
        patience_counter = 0
        training_history = []

        # dynamic stage switching
        initial_val_loss = None
        first_improvement_epoch = None
        improved_at_least_once = False

        # calculate pattern combination for metric prefixing
        pattern_combo = '+'.join(sorted(selected_patterns))

        # init tb writer (if log_dir)
        writer = None
        if log_dir and tensorboard_config and tensorboard_config.get('enabled', False):
            try:
                writer = SummaryWriter(log_dir=log_dir)

                if epoch_offset == 0:
                    # log hyperparams at start
                    hparams = {
                        'num_layers': model.config['num_layers'],
                        'neurons_per_layer': model.config['neurons_per_layer'],
                        'activation_type': model.config['activation_type'],
                        'learning_rate': learning_rate,
                        'dropout_rate': model.config['dropout_rate'],
                    }
                    if selected_patterns:
                        hparams['selected_patterns'] = ','.join(selected_patterns)
                    if target_pattern:
                        hparams['target_corruption_pattern'] = target_pattern

                    writer.add_text('hyperparameters', str(hparams), 0)
                    logger.info(f"TensorBoard logging enabled: {log_dir}")
                else:
                    # marker for stage transition
                    writer.add_text(f'{'+'.join(sorted(selected_patterns))}/stage_transition', f'Transition from DEGRADED to IMPROVED stage at epoch {epoch_offset}', epoch_offset)
                    writer.add_scalar(f'{'+'.join(sorted(selected_patterns))}/Stage/transition_marker', 1.0, epoch_offset)
                    logger.info(f"Continuing TensorBoard logging from epoch {epoch_offset}")

            except ImportError:
                logger.warning("TensorBoard not available, skipping logging")
                writer = None

        # checkpoint saving config
        save_checkpoints = False
        save_optimizer_state = False
        if checkpoint_config and log_dir:
            save_checkpoints = checkpoint_config.get('save_every_epoch', False)
            save_optimizer_state = checkpoint_config.get('save_optimizer_state', False)

        # train phase - wrap in try-finally to ensure writer cleanup
        # initialize per_pattern_accuracy to track across epochs
        per_pattern_accuracy = {}

        try:
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                # track predictions for first epoch to debug
                if epoch == 0:
                    all_predictions = []
                    all_targets = []

                for batch_idx, (data, targets, patterns) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)
                    # patterns is a tuple of pattern strings, one per example in the batch
                    optimizer.zero_grad()
                    outputs = model(data).squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    if targets.dim() == 0:
                        targets = targets.unsqueeze(0)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # check for gradient issues on first batch of first epoch
                    if epoch == 0 and batch_idx == 0:
                        total_grad_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_grad_norm += p.grad.data.norm(2).item() ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        logger.info(f"First batch gradient norm: {total_grad_norm:.6f}")

                    optimizer.step()
                    train_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (predicted == targets).sum().item()
                    train_total += targets.size(0)

                    if epoch == 0:
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())

                # log prediction distribution for first epoch
                if epoch == 0:
                    unique, counts = np.unique(all_predictions, return_counts=True)
                    pred_dist = dict(zip(unique, counts))
                    logger.info(f"First epoch prediction distribution: {pred_dist}")
                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total if train_total > 0 else 0

                # val phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                # track per-pattern accuracy
                per_pattern_stats = {pattern: {'correct': 0, 'total': 0} for pattern in selected_patterns}
                per_pattern_stats['negative'] = {'correct': 0, 'total': 0}  # for negative examples

                with torch.no_grad():
                    for data, targets, patterns in val_loader:
                        data, targets = data.to(self.device), targets.to(self.device)
                        outputs = model(data).squeeze()
                        if outputs.dim() == 0:
                            outputs = outputs.unsqueeze(0)
                        if targets.dim() == 0:
                            targets = targets.unsqueeze(0)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        val_correct += (predicted == targets).sum().item()
                        val_total += targets.size(0)

                        # track per-pattern accuracy
                        for i, (pred, target, pattern) in enumerate(zip(predicted, targets, patterns)):
                            is_correct = (pred == target).item()
                            # positive examples belong to specific pattern, negative examples are labeled with excluded_pattern
                            if target.item() == 1.0 and pattern in per_pattern_stats:
                                per_pattern_stats[pattern]['correct'] += int(is_correct)
                                per_pattern_stats[pattern]['total'] += 1
                            elif target.item() == 0.0:
                                # negative examples - track separately
                                per_pattern_stats['negative']['correct'] += int(is_correct)
                                per_pattern_stats['negative']['total'] += 1

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total if val_total > 0 else 0

                # calculate per-pattern accuracies
                per_pattern_accuracy = {}
                for pattern, stats in per_pattern_stats.items():
                    if stats['total'] > 0:
                        per_pattern_accuracy[pattern] = stats['correct'] / stats['total']
                    else:
                        per_pattern_accuracy[pattern] = 0.0

                if epoch == 0:
                    initial_val_loss = avg_val_loss
                    logger.info(f"Initial validation loss: {initial_val_loss:.4f}")

                # check for first improvement
                if wait_for_first_improvement and initial_val_loss is not None and not improved_at_least_once:
                    improvement_pct = (initial_val_loss - avg_val_loss) / initial_val_loss if initial_val_loss > 0 else 0.0

                    if improvement_pct >= min_improvement_threshold:
                        first_improvement_epoch = epoch
                        improved_at_least_once = True
                        logger.info(f"First significant improvement detected at epoch {epoch+1} "
                                   f"Val loss: {avg_val_loss:.4f} < {initial_val_loss:.4f} "
                                   f"({improvement_pct*100:.1f}% improvement, threshold: {min_improvement_threshold*100:.1f}%)")
                        if writer:
                            global_epoch_temp = epoch + epoch_offset
                            writer.add_scalar(f'{pattern_combo}/Markers/first_improvement', 1.0, global_epoch_temp)
                            writer.add_text(f'{pattern_combo}/first_improvement',
                                           f'First significant improvement: {avg_val_loss:.4f} < {initial_val_loss:.4f} ({improvement_pct*100:.1f}%)',
                                           global_epoch_temp)
                    elif avg_val_loss < initial_val_loss:
                        logger.info(f"Small improvement at epoch {epoch+1}: {improvement_pct*100:.1f}% "
                                   f"(below {min_improvement_threshold*100:.1f}% threshold)")

                # best val accuracy checkpoint
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_accuracy_checkpoint = copy.deepcopy(model.state_dict())
                    logger.info(f"New best accuracy: {val_acc:.4f} at epoch {epoch+1}")
                    if writer:
                        global_epoch_temp = epoch + epoch_offset
                        writer.add_scalar(f'{pattern_combo}/Markers/best_accuracy', val_acc, global_epoch_temp)
                        writer.add_text(f'{pattern_combo}/best_accuracy',
                                       f'New best validation accuracy: {val_acc:.4f}',
                                       global_epoch_temp)

                global_epoch = epoch + epoch_offset # offset = degraded epochs if in improvement stage

                epoch_metrics = {
                    'epoch': epoch,
                    'global_epoch': global_epoch,
                    'train_loss': avg_train_loss,
                    'train_acc': train_acc,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc
                }
                training_history.append(epoch_metrics)

                # log to tb
                if writer:
                    # prefix all metrics with pattern combination for grouping in tensorboard
                    writer.add_scalar(f'{pattern_combo}/Loss/train', avg_train_loss, global_epoch)
                    writer.add_scalar(f'{pattern_combo}/Loss/val', avg_val_loss, global_epoch)
                    writer.add_scalar(f'{pattern_combo}/Accuracy/train', train_acc, global_epoch)
                    writer.add_scalar(f'{pattern_combo}/Accuracy/val', val_acc, global_epoch)
                    writer.add_scalar(f'{pattern_combo}/Learning_rate', learning_rate, global_epoch)
                    stage_value = 0.0 if stage == 'degraded' else 1.0
                    writer.add_scalar(f'{pattern_combo}/Stage/current_stage', stage_value, global_epoch)

                    # log per-pattern accuracies
                    for pattern, accuracy in per_pattern_accuracy.items():
                        writer.add_scalar(f'{pattern_combo}/Accuracy/pattern-{pattern}', accuracy, global_epoch)

                # checkpoint saving
                if save_checkpoints and log_dir:
                    checkpoint_subdir = f"checkpoints_{stage}" if stage else "checkpoints"
                    checkpoint_dir = Path(log_dir) / checkpoint_subdir
                    checkpoint_path = checkpoint_dir / f"epoch_{global_epoch+1}.pt"
                    checkpoint_data = {
                        'epoch': epoch,
                        'global_epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'model_config': model.config,
                        'train_loss': avg_train_loss,
                        'train_acc': train_acc,
                        'val_loss': avg_val_loss,
                        'val_acc': val_acc
                    }
                    if save_optimizer_state:
                        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()

                    torch.save(checkpoint_data, checkpoint_path)

                # log every epoch
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if save_path:
                        model.save_model(save_path)
                else:
                    patience_counter += 1

                # Check for early return if waiting for first improvement and it was detected
                if wait_for_first_improvement and improved_at_least_once:
                    logger.info(f"Returning early after first improvement at epoch {epoch+1}")
                    break

                # early stopping check
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            if self.quantization_type != 'none':
                model.quantize_weights(self.quantization_type)
                if save_path:
                    model.save_model(save_path)

            # log hyperparameters and final metrics to tb
            if writer and training_history:
                try:
                    # prepare hyperparameters (use pattern_combo calculated earlier)
                    hparam_dict = {
                        'pattern_combination': pattern_combo,
                        'target_corruption_pattern': target_pattern,
                        'num_patterns': len(selected_patterns),
                        'stage': stage,
                        'learning_rate': learning_rate,
                        'num_layers': model.num_layers,
                        'neurons_per_layer': model.neurons_per_layer,
                        'activation_type': model.activation_type,
                    }

                    # prepare final metrics
                    final_metrics_dict = {
                        'hparam/accuracy_val': training_history[-1]['val_acc'],
                        'hparam/loss_val': training_history[-1]['val_loss'],
                        'hparam/accuracy_train': training_history[-1]['train_acc'],
                        'hparam/loss_train': training_history[-1]['train_loss'],
                        'hparam/best_val_accuracy': best_val_accuracy,
                    }

                    # add per-pattern final accuracies to metrics
                    # use the last per_pattern_accuracy calculated in the final epoch
                    if per_pattern_accuracy:
                        for pattern, accuracy in per_pattern_accuracy.items():
                            final_metrics_dict[f'hparam/accuracy_pattern-{pattern}'] = accuracy

                    # log to hparams
                    writer.add_hparams(hparam_dict, final_metrics_dict)
                    logger.info(f"Logged hyperparameters to TensorBoard: {pattern_combo} (stage: {stage})")
                except Exception as e:
                    logger.warning(f"Error logging hyperparameters to TensorBoard: {e}")

        finally:
            # ensure tb writer is always closed
            if writer:
                try:
                    writer.flush()
                    writer.close()
                except Exception as e:
                    logger.warning(f"Error closing TensorBoard writer: {e}")

        final_metrics = training_history[-1] if training_history else {}

        # use best accuracy checkpoint
        if best_accuracy_checkpoint is None:
            best_accuracy_checkpoint = model.state_dict()

        results = {
            'training_history': training_history,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'best_accuracy_checkpoint': best_accuracy_checkpoint,
            'epochs_trained': len(training_history),
            'early_stopped': patience_counter >= early_stopping_patience,
            'improved_at_least_once': improved_at_least_once,
            'first_improvement_epoch': first_improvement_epoch,
            'initial_val_loss': initial_val_loss,
        }
        return results

    def train_staged_improvement(self, model: SubjectModel,
                                corrupted_train_loader, corrupted_val_loader,
                                clean_train_loader, clean_val_loader,
                                max_degraded_epochs: int, improvement_epochs: int,
                                learning_rate: float, improvement_lr_factor: float = 0.1,
                                early_stopping_patience: int = 10,
                                degraded_log_dir: str = None, improved_log_dir: str = None,
                                selected_patterns: List[str] = None, target_pattern: str = None,
                                tensorboard_config: Dict[str, Any] = None,
                                checkpoint_config: Dict[str, Any] = None,
                                min_improvement_threshold: float = 0.0) -> Dict[str, Any]:
        model = model.to(self.device)

        logger.info(f"Stage 1: Training on corrupted data (waiting for {min_improvement_threshold*100:.1f}% improvement)...")
        degraded_results = self.train_and_evaluate(
            model=model,
            train_loader=corrupted_train_loader,
            val_loader=corrupted_val_loader,
            num_epochs=max_degraded_epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            save_path=None,
            log_dir=degraded_log_dir,
            selected_patterns=selected_patterns,
            target_pattern=target_pattern,
            stage='degraded',
            tensorboard_config=tensorboard_config,
            checkpoint_config=checkpoint_config,
            epoch_offset=0,
            wait_for_first_improvement=True,
            min_improvement_threshold=min_improvement_threshold
        )

        # check if model improved during degraded stage
        if not degraded_results['improved_at_least_once']:
            logger.warning(f"Model never improved during degraded stage ({degraded_results['epochs_trained']} epochs)")
            return {
                'success': False,
                'reason': 'no_improvement_in_degraded_stage',
                'epochs_waited': degraded_results['epochs_trained'],
                'initial_val_loss': degraded_results['initial_val_loss'],
            }

        first_improvement_epoch = degraded_results['first_improvement_epoch']
        degraded_weights = copy.deepcopy(model.state_dict())
        degraded_acc = degraded_results['final_metrics'].get('val_acc', 0)
        logger.info(f"Stage 1 complete - First improvement at epoch {first_improvement_epoch + 1}, Val accuracy: {degraded_acc:.4f}")

        logger.info("Stage 2: Continuing training on clean data...")
        improved_results = self.train_and_evaluate(
            model=model,
            train_loader=clean_train_loader,
            val_loader=clean_val_loader,
            num_epochs=improvement_epochs,
            learning_rate=learning_rate * improvement_lr_factor,
            early_stopping_patience=early_stopping_patience,
            save_path=None,
            log_dir=improved_log_dir,
            selected_patterns=selected_patterns,
            target_pattern=target_pattern,
            stage='improved',
            tensorboard_config=tensorboard_config,
            checkpoint_config=checkpoint_config,
            epoch_offset=first_improvement_epoch + 1,  # start from first improvement epoch
            wait_for_first_improvement=False
        )

        # best accuracy checkpoint for final weights
        improved_weights = improved_results['best_accuracy_checkpoint']
        improved_acc = improved_results['best_val_accuracy']
        best_epoch_offset = None
        for idx, metrics in enumerate(improved_results['training_history']):
            if abs(metrics.get('val_acc', 0) - improved_acc) < 0.0001:  # Float comparison tolerance
                best_epoch_offset = metrics['global_epoch']
                break

        improvement = improved_acc - degraded_acc
        logger.info(f"Stage 2 complete - Best validation accuracy: {improved_acc:.4f} (epoch {best_epoch_offset}), Improvement: {improvement:+.4f}")

        return {
            'success': True,
            'degraded': {
                'weights': degraded_weights,
                'metrics': degraded_results,
                'accuracy': degraded_acc
            },
            'improved': {
                'weights': improved_weights,
                'metrics': improved_results,
                'accuracy': improved_acc,
                'best_epoch': best_epoch_offset
            },
            'improvement': improvement,
            'stages_completed': 2,
            'first_improvement_epoch': first_improvement_epoch,
        }


def create_subject_model(model_id: str, num_layers: int = 6, neurons_per_layer: int = 25, activation_type: str = 'relu',
                        random_seed: int = 42, dropout_rate: float = 0.1, precision: str = 'float32',
                        vocab_size: int = 7, sequence_length: int = 7) -> Tuple[SubjectModel, Dict[str, Any]]:
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    config = {
        'model_id': model_id,
        'num_layers': num_layers,
        'neurons_per_layer': neurons_per_layer,
        'activation_type': activation_type,
        'random_seed': random_seed,
        'dropout_rate': dropout_rate,
        'precision': precision,
        'vocab_size': vocab_size,
        'sequence_length': sequence_length
    }

    model = SubjectModel(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        num_layers=num_layers,
        neurons_per_layer=neurons_per_layer,
        activation_type=activation_type,
        dropout_rate=dropout_rate,
        precision=precision
    )
    
    logger.info(f"Initialized subject model '{model_id}' with {num_layers} layers, {neurons_per_layer} neurons/layer")
    
    return model, config


def create_data_loaders(examples: list, batch_size: int = 32, train_ratio: float = 0.8, random_seed: int = 42, num_workers: int = 0, pin_memory: bool = False, vocab_size: int = None):
    dataset = SequenceDataset(examples, vocab_size=vocab_size)
    
    # split dataset
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.debug(f"Created data loaders: {train_size} train, {val_size} val examples")
    
    return train_loader, val_loader