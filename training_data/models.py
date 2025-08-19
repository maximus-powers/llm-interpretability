import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SubjectModel(nn.Module):
    """
    Simple neural network for sequence binary classification.
    Takes one-hot encoded sequences (7 tokens × 7 positions = 49 features) and outputs binary classification preds.
    """
    
    def __init__(self, 
                 vocab_size: int = 7,
                 sequence_length: int = 7,
                 num_layers: int = 6,
                 neurons_per_layer: int = 25,
                 activation_type: str = 'relu',
                 dropout_rate: float = 0.1):

        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.input_size = vocab_size * sequence_length 
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.activation = self._get_activation(activation_type)
        
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
        
        # store config
        self.config = {
            'vocab_size': vocab_size,
            'sequence_length': sequence_length,
            'num_layers': num_layers,
            'neurons_per_layer': neurons_per_layer,
            'activation_type': activation_type,
            'dropout_rate': dropout_rate,
            'input_size': self.input_size
        }
        
        logger.debug(f"Created SubjectModel: {num_layers} layers, " f"{neurons_per_layer} neurons/layer, {activation_type} activation")
    
    def _get_activation(self, activation_type: str):
        # only using relu and gelu now
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation_type.lower(), nn.ReLU())
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def predict_classes(self, x, threshold: float = 0.5):
        probabilities = self.predict(x)
        return (probabilities > threshold).float()
    
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
    
    def get_weight_statistics(self):
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'shape': list(param.shape),
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'norm': param.norm().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'num_params': param.numel()
                }
        
        return stats
    
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
    """
    Dataset class for sequence classification.
    Handles conversion from list of examples to PyTorch tensors.
    """
    
    def __init__(self, examples, vocab=None):
        self.examples = examples
        self.vocab = vocab or ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        logger.debug(f"Created SequenceDataset with {len(examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        sequence = example['sequence']
        label = float(example['label'])
        sequence_tensor = self._sequence_to_tensor(sequence)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return sequence_tensor, label_tensor
    
    def _sequence_to_tensor(self, sequence):
        seq_length = len(sequence)
        one_hot = torch.zeros(seq_length, self.vocab_size)
        for i, token in enumerate(sequence):
            if token in self.token_to_idx:
                token_idx = self.token_to_idx[token]
                one_hot[i, token_idx] = 1.0
        return one_hot.view(-1)

class SubjectModelTrainer:
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def train_model(self,
                   model: SubjectModel,
                   train_loader,
                   val_loader,
                   num_epochs: int = 30,
                   learning_rate: float = 0.001,
                   early_stopping_patience: int = 10,
                   save_path: str = None,
                   verbose: bool = True) -> Dict[str, Any]:

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        if verbose:
            logger.info(f"Starting training for {num_epochs} epochs")
        
        # train phase
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(data).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            # val phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, targets in val_loader:
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
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc
            }
            training_history.append(epoch_metrics)
            
            if verbose and (epoch % 10 == 0 or epoch < 5):
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                           f"train_acc={train_acc:.4f}, val_loss={avg_val_loss:.4f}, "
                           f"val_acc={val_acc:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if save_path:
                    model.save_model(save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
        
        final_metrics = training_history[-1] if training_history else {}
        
        results = {
            'training_history': training_history,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(training_history),
            'early_stopped': patience_counter >= early_stopping_patience
        }
        
        if verbose:
            logger.info(f"Training completed: {len(training_history)} epochs, "
                       f"best_val_loss={best_val_loss:.4f}")
        
        return results


def create_subject_model(model_id: str,
                        num_layers: int = 6,
                        neurons_per_layer: int = 25,
                        activation_type: str = 'relu',
                        random_seed: int = 42,
                        dropout_rate: float = 0.1) -> Tuple[SubjectModel, Dict[str, Any]]:
    """
    Create a subject model with given config.
    """
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    config = {
        'model_id': model_id,
        'num_layers': num_layers,
        'neurons_per_layer': neurons_per_layer,
        'activation_type': activation_type,
        'random_seed': random_seed,
        'dropout_rate': dropout_rate
    }
    
    model = SubjectModel(
        num_layers=num_layers,
        neurons_per_layer=neurons_per_layer,
        activation_type=activation_type,
        dropout_rate=dropout_rate
    )
    
    logger.info(f"Created model '{model_id}' with {num_layers} layers, "
               f"{neurons_per_layer} neurons/layer")
    
    return model, config


def create_data_loaders(examples: list, batch_size: int = 32, train_ratio: float = 0.8, random_seed: int = 42):
    """
    Create training and validation data loaders from examples.
    """
    dataset = SequenceDataset(examples)
    
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
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.debug(f"Created data loaders: {train_size} train, {val_size} val examples")
    
    return train_loader, val_loader