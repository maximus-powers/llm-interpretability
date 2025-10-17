import torch
import torch.nn as nn
import logging
import copy
from typing import Dict, Any, Tuple, Optional

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
        sequence_tensor = self._sequence_to_tensor(sequence)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return sequence_tensor, label_tensor
    
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
                   save_path: str = None) -> Dict[str, Any]:

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
                
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
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if save_path:
                    model.save_model(save_path)
            else:
                patience_counter += 1
        
        if self.quantization_type != 'none':
            model.quantize_weights(self.quantization_type)
            if save_path:
                model.save_model(save_path)
        
        final_metrics = training_history[-1] if training_history else {}
        
        results = {
            'training_history': training_history,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(training_history),
            'early_stopped': patience_counter >= early_stopping_patience
        }
                
        return results

    def train_staged_improvement(self, model: SubjectModel,
                                corrupted_train_loader, corrupted_val_loader,
                                clean_train_loader, clean_val_loader,
                                degraded_epochs: int, improvement_epochs: int,
                                learning_rate: float, improvement_lr_factor: float = 0.1,
                                early_stopping_patience: int = 10) -> Dict[str, Any]:
        model = model.to(self.device)

        logger.info("Stage 1: Training on corrupted data...")
        degraded_results = self.train_and_evaluate(
            model=model,
            train_loader=corrupted_train_loader,
            val_loader=corrupted_val_loader,
            num_epochs=degraded_epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            save_path=None
        )
        degraded_weights = copy.deepcopy(model.state_dict())

        logger.info("Stage 2: Continuing training on clean data...")
        improved_results = self.train_and_evaluate(
            model=model,
            train_loader=clean_train_loader,
            val_loader=clean_val_loader,
            num_epochs=improvement_epochs,
            learning_rate=learning_rate * improvement_lr_factor,
            early_stopping_patience=early_stopping_patience,
            save_path=None
        )

        improved_weights = model.state_dict()

        degraded_acc = degraded_results['final_metrics'].get('val_acc', 0)
        improved_acc = improved_results['final_metrics'].get('val_acc', 0)
        improvement = improved_acc - degraded_acc

        return {
            'degraded': {
                'weights': degraded_weights,
                'metrics': degraded_results,
                'accuracy': degraded_acc
            },
            'improved': {
                'weights': improved_weights,
                'metrics': improved_results,
                'accuracy': improved_acc
            },
            'improvement': improvement,
            'stages_completed': 2
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