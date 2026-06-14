"""
Experiment 6: Dual-Encoder Architecture

MOTIVATION:
- Behavioral-only features (9 dims) fail completely -> 100% model collapse
- Input_correlations ~= weights (0.94 cosine at layer 0) - they're "architectural", not "behavioral"
- We need BOTH for reconstruction, but want Z_behavior to be interpretable

SOLUTION:
Dual-encoder that separates behavioral and weight information:
- BehavioralEncoder: [mean, std, fourier, pre_act] -> Z_behavior (interpretable)
- WeightEncoder: weights -> Z_weight (detailed)
- DualLatentDecoder: [Z_behavior, Z_weight, position] -> weights

TRAINING LOSSES:
1. Functional Loss: MSE on network outputs (PRIMARY)
2. Alignment Loss: cosine(Z_behavior, Z_weight.detach()) (encourages behavioral to capture info)
3. Reconstruction Loss: MSE on weights (SECONDARY)

ABLATION STUDY:
- Combination modes: concat, add, gated
- Detach strategies: alignment_only, full_detach, no_detach
- Alignment lambda: 0.1, 0.5, 1.0

SUCCESS CRITERIA:
- Output correlation >= 0.87 (match Exp4 functional-loss baseline)
- Model collapse rate = 0%
- Z_behavior captures meaningful behavioral info for rep-eng
"""

import sys
import time
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.components.encoders import WeightEncoder, BehavioralEncoder, BEHAVIORAL_INDICES
from hypernet.components.decoders import DualLatentDecoder
from hypernet.utils.data import create_dataloaders
from hypernet.functional_eval import (
    SubjectModel,
    load_weights_into_model,
    create_test_inputs,
    compute_functional_agreement,
    compute_output_correlation,
)


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""
    name: str
    combination_mode: str  # 'concat', 'add', 'gated'
    detach_mode: str       # 'alignment_only', 'full_detach', 'no_detach'
    lambda_align: float    # Alignment loss weight
    lambda_func: float = 1.0   # Functional loss weight
    lambda_recon: float = 0.1  # Reconstruction loss weight
    
    # Model hyperparameters
    latent_dim: int = 64
    hidden_dim: int = 256
    encoder_layers: int = 3
    decoder_layers: int = 4
    dropout: float = 0.1
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Data
    train_samples: int = 8000
    val_samples: int = 1000


# Define ablation configurations
ABLATION_CONFIGS = {
    # Main ablations: combination mode
    "6a_concat_detach_align_0.5": ExperimentConfig(
        name="6a_concat_detach_align_0.5",
        combination_mode="concat",
        detach_mode="alignment_only",
        lambda_align=0.5,
    ),
    "6b_add_detach_align_0.5": ExperimentConfig(
        name="6b_add_detach_align_0.5",
        combination_mode="add",
        detach_mode="alignment_only",
        lambda_align=0.5,
    ),
    "6c_gated_detach_align_0.5": ExperimentConfig(
        name="6c_gated_detach_align_0.5",
        combination_mode="gated",
        detach_mode="alignment_only",
        lambda_align=0.5,
    ),
    # Detach mode ablations
    "6d_concat_full_detach_0.5": ExperimentConfig(
        name="6d_concat_full_detach_0.5",
        combination_mode="concat",
        detach_mode="full_detach",
        lambda_align=0.5,
    ),
    "6e_concat_no_detach_0.5": ExperimentConfig(
        name="6e_concat_no_detach_0.5",
        combination_mode="concat",
        detach_mode="no_detach",
        lambda_align=0.5,
    ),
    # Lambda ablations
    "6f_concat_detach_align_0.1": ExperimentConfig(
        name="6f_concat_detach_align_0.1",
        combination_mode="concat",
        detach_mode="alignment_only",
        lambda_align=0.1,
    ),
    "6g_concat_detach_align_1.0": ExperimentConfig(
        name="6g_concat_detach_align_1.0",
        combination_mode="concat",
        detach_mode="alignment_only",
        lambda_align=1.0,
    ),
}


class DualEncoderSystem(nn.Module):
    """
    Dual-encoder system for MUAT.
    
    Components:
    - BehavioralEncoder: signatures (behavioral only) -> Z_behavior
    - WeightEncoder: weights -> Z_weight
    - DualLatentDecoder: [Z_behavior, Z_weight, position] -> weights
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        self.config = config
        
        # Behavioral encoder (9-feature behavioral signatures)
        self.behavioral_encoder = BehavioralEncoder(
            behavioral_dim=9,  # len(BEHAVIORAL_INDICES)
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        # Weight encoder
        self.weight_encoder = WeightEncoder(
            input_dim=9,  # max_fan_in + 1
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        # Dual latent decoder
        self.decoder = DualLatentDecoder(
            behavioral_latent_dim=config.latent_dim,
            weight_latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=9,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
            combination_mode=config.combination_mode,
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual-encoder system.
        
        Args:
            signatures: [batch, num_neurons, 17] full signatures
            weights: [batch, num_neurons, 9] weight tokens
            positions: [batch, num_neurons, 3] positions
            mask: Optional validity mask
            
        Returns:
            pred_weights: [batch, num_neurons, 9] predicted weights
            z_behavior: [batch, num_neurons, latent_dim] behavioral latent
            z_weight: [batch, num_neurons, latent_dim] weight latent
        """
        # Encode behavioral signatures (auto-extracts behavioral features)
        z_behavior = self.behavioral_encoder(signatures, mask)
        
        # Encode weights
        z_weight = self.weight_encoder(weights, mask)
        
        # Decode
        pred_weights = self.decoder(z_behavior, z_weight, positions, mask)
        
        return pred_weights, z_behavior, z_weight
    
    def encode_behavior(self, signatures: torch.Tensor) -> torch.Tensor:
        """Encode only behavioral features (for rep-eng)."""
        return self.behavioral_encoder(signatures)
    
    def encode_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Encode weights."""
        return self.weight_encoder(weights)
    
    def decode(
        self,
        z_behavior: torch.Tensor,
        z_weight: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from latents."""
        return self.decoder(z_behavior, z_weight, positions)


class FunctionalLoss(nn.Module):
    """
    Functional loss: measures behavior similarity, not weight similarity.
    
    Computes MSE between network outputs on a set of test inputs.
    
    Optimization: Only computes for a subset of samples to reduce overhead.
    """
    
    def __init__(
        self, 
        n_test_samples: int = 20,  # Reduced from 50 for speed
        sequence_length: int = 5, 
        vocab_size: int = 10,
        max_models_per_batch: int = 4,  # Only compute for first N models per batch
    ):
        super().__init__()
        self.n_test_samples = n_test_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.max_models_per_batch = max_models_per_batch
        
        # Pre-generate test inputs (same for all evaluations)
        self.register_buffer(
            'test_inputs',
            torch.randint(0, vocab_size, (n_test_samples, sequence_length)).float()
        )
    
    def forward(
        self,
        pred_weights: torch.Tensor,  # [batch, num_neurons, 9]
        true_weights: torch.Tensor,  # [batch, num_neurons, 9]
        positions: torch.Tensor,     # [batch, num_neurons, 3]
        mask: torch.Tensor,          # [batch, num_neurons]
    ) -> torch.Tensor:
        """
        Compute functional loss by building networks and comparing outputs.
        
        This is expensive but critical for functional accuracy.
        """
        batch_size = pred_weights.shape[0]
        total_loss = 0.0
        valid_count = 0
        
        # Only compute for first N models to save time
        for b in range(min(batch_size, self.max_models_per_batch)):
            # Extract valid neurons
            valid_mask = mask[b].bool()
            if valid_mask.sum() == 0:
                continue
            
            pred_w = pred_weights[b][valid_mask]  # [num_valid, 9]
            true_w = true_weights[b][valid_mask]  # [num_valid, 9]
            pos = positions[b][valid_mask]        # [num_valid, 3]
            
            # Build networks from weights
            try:
                pred_outputs = self._run_network(pred_w, pos)
                true_outputs = self._run_network(true_w, pos)
                
                # MSE on outputs
                loss = F.mse_loss(pred_outputs, true_outputs)
                total_loss += loss
                valid_count += 1
            except Exception:
                # Skip if network construction fails
                continue
        
        if valid_count == 0:
            return torch.tensor(0.0, device=pred_weights.device, requires_grad=True)
        
        return total_loss / valid_count
    
    def _run_network(self, weights: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Build and run a network from weight tokens."""
        # Group neurons by layer
        layer_neurons = defaultdict(list)
        for i in range(len(weights)):
            layer_idx = int(positions[i, 0].item())
            neuron_idx = int(positions[i, 1].item())
            fan_in = int(positions[i, 2].item())
            layer_neurons[layer_idx].append((neuron_idx, weights[i], fan_in))
        
        # Build layer weights
        x = self.test_inputs
        
        for layer_idx in sorted(layer_neurons.keys()):
            neurons = sorted(layer_neurons[layer_idx], key=lambda x: x[0])
            fan_in = neurons[0][2]
            num_neurons = len(neurons)
            
            # Extract weight matrix and biases
            W = torch.stack([n[1][:fan_in] for n in neurons])  # [num_neurons, fan_in]
            b = torch.stack([n[1][fan_in:fan_in+1] for n in neurons]).squeeze(-1)  # [num_neurons]
            
            # Linear + GELU
            x = F.linear(x, W, b)
            x = F.gelu(x)
        
        return x


def compute_alignment_loss(
    z_behavior: torch.Tensor,
    z_weight: torch.Tensor,
    mask: torch.Tensor,
    detach_mode: str,
) -> torch.Tensor:
    """
    Compute alignment loss between behavioral and weight latents.
    
    Args:
        z_behavior: [batch, num_neurons, latent_dim]
        z_weight: [batch, num_neurons, latent_dim]
        mask: [batch, num_neurons]
        detach_mode: 'alignment_only', 'full_detach', 'no_detach'
        
    Returns:
        alignment_loss: scalar
    """
    # Apply detach based on mode
    if detach_mode == 'alignment_only':
        # Gradients flow to behavioral encoder, not weight encoder
        z_weight_for_align = z_weight.detach()
        z_behavior_for_align = z_behavior
    elif detach_mode == 'full_detach':
        # No gradients through alignment (just monitoring)
        z_weight_for_align = z_weight.detach()
        z_behavior_for_align = z_behavior.detach()
    else:  # no_detach
        # Gradients flow to both encoders
        z_weight_for_align = z_weight
        z_behavior_for_align = z_behavior
    
    # Compute cosine similarity
    mask_expanded = mask.unsqueeze(-1)  # [batch, num_neurons, 1]
    
    # Normalize
    z_b_norm = F.normalize(z_behavior_for_align, dim=-1)
    z_w_norm = F.normalize(z_weight_for_align, dim=-1)
    
    # Cosine similarity per neuron
    cos_sim = (z_b_norm * z_w_norm).sum(dim=-1)  # [batch, num_neurons]
    
    # Masked mean (we want to MAXIMIZE similarity, so loss = 1 - sim)
    alignment_loss = (1 - cos_sim) * mask
    alignment_loss = alignment_loss.sum() / mask.sum()
    
    return alignment_loss


def train_epoch(
    model: DualEncoderSystem,
    loader,
    optimizer,
    config: ExperimentConfig,
    device: str,
    functional_loss_fn: FunctionalLoss,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_func = 0.0
    total_align = 0.0
    total_recon = 0.0
    total_cosine = 0.0
    num_batches = 0
    
    for batch in loader:
        signatures = batch['signatures'].to(device)
        weights = batch['weights'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred_weights, z_behavior, z_weight = model(signatures, weights, positions, mask)
        
        # Losses
        # 1. Reconstruction loss (weight MSE)
        mask_expanded = mask.unsqueeze(-1)
        recon_loss = ((pred_weights - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
        
        # 2. Functional loss
        func_loss = functional_loss_fn(pred_weights, weights, positions, mask)
        
        # 3. Alignment loss
        align_loss = compute_alignment_loss(z_behavior, z_weight, mask, config.detach_mode)
        
        # Combined loss
        loss = (
            config.lambda_func * func_loss +
            config.lambda_align * align_loss +
            config.lambda_recon * recon_loss
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            cosine = compute_batch_cosine(pred_weights, weights, mask)
        
        total_loss += loss.item()
        total_func += func_loss.item()
        total_align += align_loss.item()
        total_recon += recon_loss.item()
        total_cosine += cosine
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'func_loss': total_func / num_batches,
        'align_loss': total_align / num_batches,
        'recon_loss': total_recon / num_batches,
        'cosine': total_cosine / num_batches,
    }


def compute_batch_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute masked cosine similarity."""
    batch_size = pred.shape[0]
    cos_sims = []
    
    for b in range(batch_size):
        valid_mask = mask[b].bool()
        if valid_mask.sum() == 0:
            continue
        
        pred_valid = pred[b][valid_mask].flatten()
        target_valid = target[b][valid_mask].flatten()
        
        cos_sim = F.cosine_similarity(
            pred_valid.unsqueeze(0),
            target_valid.unsqueeze(0)
        )
        cos_sims.append(cos_sim.item())
    
    return sum(cos_sims) / len(cos_sims) if cos_sims else 0.0


@torch.no_grad()
def validate(
    model: DualEncoderSystem,
    loader,
    config: ExperimentConfig,
    device: str,
    functional_loss_fn: FunctionalLoss,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_func = 0.0
    total_align = 0.0
    total_recon = 0.0
    total_cosine = 0.0
    total_alignment_cosine = 0.0
    num_batches = 0
    
    for batch in loader:
        signatures = batch['signatures'].to(device)
        weights = batch['weights'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        pred_weights, z_behavior, z_weight = model(signatures, weights, positions, mask)
        
        # Losses
        mask_expanded = mask.unsqueeze(-1)
        recon_loss = ((pred_weights - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
        func_loss = functional_loss_fn(pred_weights, weights, positions, mask)
        align_loss = compute_alignment_loss(z_behavior, z_weight, mask, 'no_detach')
        
        loss = (
            config.lambda_func * func_loss +
            config.lambda_align * align_loss +
            config.lambda_recon * recon_loss
        )
        
        # Metrics
        cosine = compute_batch_cosine(pred_weights, weights, mask)
        
        # Alignment cosine (Z_behavior vs Z_weight)
        z_b_norm = F.normalize(z_behavior, dim=-1)
        z_w_norm = F.normalize(z_weight, dim=-1)
        align_cos = ((z_b_norm * z_w_norm).sum(dim=-1) * mask).sum() / mask.sum()
        
        total_loss += loss.item()
        total_func += func_loss.item()
        total_align += align_loss.item()
        total_recon += recon_loss.item()
        total_cosine += cosine
        total_alignment_cosine += align_cos.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'func_loss': total_func / num_batches,
        'align_loss': total_align / num_batches,
        'recon_loss': total_recon / num_batches,
        'cosine': total_cosine / num_batches,
        'z_alignment': total_alignment_cosine / num_batches,
    }


@torch.no_grad()
def evaluate_functional_accuracy(
    model: DualEncoderSystem,
    loader,
    device: str,
    n_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate functional accuracy: does the reconstructed network behave like the original?
    
    Returns:
        - balanced_accuracy: agreement rate on binary predictions
        - output_correlation: correlation between logits
        - collapse_rate: fraction of models that always predict same class
    """
    model.eval()
    
    from datasets import load_dataset as hf_load_dataset
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    
    accuracies = []
    correlations = []
    collapse_count = 0
    total_models = 0
    
    for batch in loader:
        signatures = batch['signatures'].to(device)
        weights = batch['weights'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        pred_weights, _, _ = model(signatures, weights, positions, mask)
        
        batch_size = pred_weights.shape[0]
        
        for b in range(min(batch_size, n_samples - total_models)):
            if total_models >= n_samples:
                break
            
            valid_mask = mask[b].bool()
            if valid_mask.sum() == 0:
                continue
            
            # Get original config (we need it to build SubjectModel)
            # For simplicity, use standard config
            config = {
                'vocab_size': 10,
                'sequence_length': 5,
                'num_layers': 4,  # Approximate
                'neurons_per_layer': 6,  # Approximate
            }
            
            # Build original and reconstructed models
            pred_w = pred_weights[b][valid_mask].cpu()
            true_w = weights[b][valid_mask].cpu()
            pos = positions[b][valid_mask].cpu()
            
            try:
                # Build state dicts
                pred_state = weights_to_state_dict(pred_w, pos)
                true_state = weights_to_state_dict(true_w, pos)
                
                # Infer config from positions
                # Layer indices are 0, 2, 4, 6, 8, 10, 12 (Linear layers only, GELU layers are 1, 3, 5, etc.)
                # The last layer (max_layer) is the output layer with 1 neuron
                # num_layers for SubjectModel is the number of HIDDEN layers (not including output)
                layer_indices = sorted(set(int(p[0].item()) for p in pos))
                max_layer = max(layer_indices)
                num_hidden_layers = len(layer_indices) - 1  # Exclude output layer
                
                neurons_per_layer = int((pos[:, 0] == 0).sum().item())
                fan_in_first = int(pos[0, 2].item())
                
                config = {
                    'vocab_size': 10,
                    'sequence_length': fan_in_first,
                    'num_layers': num_hidden_layers,
                    'neurons_per_layer': neurons_per_layer,
                }
                
                original = SubjectModel(config)
                reconstructed = SubjectModel(config)
                
                load_weights_into_model(original, true_state)
                load_weights_into_model(reconstructed, pred_state)
                
                test_inputs = create_test_inputs(config, 200)
                
                agreement = compute_functional_agreement(original, reconstructed, test_inputs)
                correlation = compute_output_correlation(original, reconstructed, test_inputs)
                
                # Check for collapse (always predicts same class)
                with torch.no_grad():
                    preds = reconstructed.predict(test_inputs)
                    if preds.std() < 0.01:  # All same prediction
                        collapse_count += 1
                
                accuracies.append(agreement)
                correlations.append(correlation)
                total_models += 1
                
            except Exception as e:
                continue
        
        if total_models >= n_samples:
            break
    
    return {
        'balanced_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
        'output_correlation': sum(correlations) / len(correlations) if correlations else 0.0,
        'collapse_rate': collapse_count / total_models if total_models > 0 else 1.0,
        'n_evaluated': total_models,
    }


def weights_to_state_dict(weights: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert flat weight predictions to state_dict format."""
    state_dict = {}
    
    # Group by layer
    layer_neurons = defaultdict(list)
    for i in range(len(weights)):
        layer_idx = int(positions[i, 0].item())
        neuron_idx = int(positions[i, 1].item())
        layer_neurons[layer_idx].append((neuron_idx, weights[i]))
    
    for layer_idx in sorted(layer_neurons.keys()):
        neurons = sorted(layer_neurons[layer_idx], key=lambda x: x[0])
        fan_in = int(positions[positions[:, 0] == layer_idx][0, 2].item())
        
        W = torch.stack([n[1][:fan_in] for n in neurons])
        b = torch.stack([n[1][fan_in:fan_in+1] for n in neurons]).squeeze(-1)
        
        state_dict[f'network.{layer_idx}.weight'] = W
        state_dict[f'network.{layer_idx}.bias'] = b
    
    return state_dict


def run_experiment(
    config: ExperimentConfig, 
    device: str = "auto",
    use_tensorboard: bool = True,
    save_checkpoints: bool = True,
    run_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run a single ablation experiment with TensorBoard logging."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Setup run directory
    if run_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp6_{config.name}_{timestamp}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*60}")
    print(f"Combination mode: {config.combination_mode}")
    print(f"Detach mode: {config.detach_mode}")
    print(f"Lambda align: {config.lambda_align}")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print()
    
    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        # Log hyperparameters
        hparams = {
            'combination_mode': config.combination_mode,
            'detach_mode': config.detach_mode,
            'lambda_align': config.lambda_align,
            'lambda_func': config.lambda_func,
            'lambda_recon': config.lambda_recon,
            'latent_dim': config.latent_dim,
            'hidden_dim': config.hidden_dim,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'train_samples': config.train_samples,
        }
        writer.add_text('config', json.dumps(hparams, indent=2))
        print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        batch_size=config.batch_size,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
    )
    
    # Create model
    model = DualEncoderSystem(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    
    # Functional loss
    functional_loss_fn = FunctionalLoss(n_test_samples=20, max_models_per_batch=4).to(device)
    
    # Training loop
    best_val_cosine = 0.0
    best_val_corr = 0.0
    best_epoch = 0
    global_step = 0
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        epoch_train_metrics = defaultdict(float)
        num_batches = 0
        
        for batch in train_loader:
            signatures = batch['signatures'].to(device)
            weights = batch['weights'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_weights, z_behavior, z_weight = model(signatures, weights, positions, mask)
            
            # Losses
            mask_expanded = mask.unsqueeze(-1)
            recon_loss = ((pred_weights - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
            func_loss = functional_loss_fn(pred_weights, weights, positions, mask)
            align_loss = compute_alignment_loss(z_behavior, z_weight, mask, config.detach_mode)
            
            loss = (
                config.lambda_func * func_loss +
                config.lambda_align * align_loss +
                config.lambda_recon * recon_loss
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                cosine = compute_batch_cosine(pred_weights, weights, mask)
            
            epoch_train_metrics['loss'] += loss.item()
            epoch_train_metrics['func_loss'] += func_loss.item()
            epoch_train_metrics['align_loss'] += align_loss.item()
            epoch_train_metrics['recon_loss'] += recon_loss.item()
            epoch_train_metrics['cosine'] += cosine
            num_batches += 1
            global_step += 1
            
            # Log to TensorBoard (every 10 batches)
            if writer and global_step % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/func_loss', func_loss.item(), global_step)
                writer.add_scalar('train/align_loss', align_loss.item(), global_step)
                writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
                writer.add_scalar('train/cosine', cosine, global_step)
        
        # Average train metrics
        for k in epoch_train_metrics:
            epoch_train_metrics[k] /= num_batches
        
        # Validate
        val_metrics = validate(model, val_loader, config, device, functional_loss_fn)
        scheduler.step()
        
        # Log epoch metrics to TensorBoard
        if writer:
            writer.add_scalar('epoch/train_loss', epoch_train_metrics['loss'], epoch)
            writer.add_scalar('epoch/train_cosine', epoch_train_metrics['cosine'], epoch)
            writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            writer.add_scalar('epoch/val_cosine', val_metrics['cosine'], epoch)
            writer.add_scalar('epoch/val_func_loss', val_metrics['func_loss'], epoch)
            writer.add_scalar('epoch/z_alignment', val_metrics['z_alignment'], epoch)
            writer.add_scalar('epoch/lr', scheduler.get_last_lr()[0], epoch)
        
        # Track best
        if val_metrics['cosine'] > best_val_cosine:
            best_val_cosine = val_metrics['cosine']
            best_epoch = epoch + 1
            
            # Save best checkpoint
            if save_checkpoints:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_cosine': val_metrics['cosine'],
                    'config': asdict(config),
                }
                torch.save(checkpoint, run_dir / "best_model.pt")
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Loss: {epoch_train_metrics['loss']:.4f} | "
                  f"Val Cosine: {val_metrics['cosine']:.4f} | "
                  f"Z-Align: {val_metrics['z_alignment']:.4f}")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    print(f"Best val cosine: {best_val_cosine:.4f} (epoch {best_epoch})")
    
    # Final functional evaluation
    print("\nRunning functional evaluation...")
    func_metrics = evaluate_functional_accuracy(model, val_loader, device, n_samples=100)
    
    print(f"\nFUNCTIONAL METRICS:")
    print(f"  Balanced Accuracy: {func_metrics['balanced_accuracy']:.4f}")
    print(f"  Output Correlation: {func_metrics['output_correlation']:.4f}")
    print(f"  Collapse Rate: {func_metrics['collapse_rate']:.2%}")
    print(f"  Models Evaluated: {func_metrics['n_evaluated']}")
    
    # Log final metrics to TensorBoard
    if writer:
        writer.add_scalar('final/balanced_accuracy', func_metrics['balanced_accuracy'], 0)
        writer.add_scalar('final/output_correlation', func_metrics['output_correlation'], 0)
        writer.add_scalar('final/collapse_rate', func_metrics['collapse_rate'], 0)
        writer.add_scalar('final/best_val_cosine', best_val_cosine, 0)
        
        # Log hparams with metrics
        writer.add_hparams(
            hparams,
            {
                'hparam/best_val_cosine': best_val_cosine,
                'hparam/output_correlation': func_metrics['output_correlation'],
                'hparam/balanced_accuracy': func_metrics['balanced_accuracy'],
                'hparam/collapse_rate': func_metrics['collapse_rate'],
            }
        )
        writer.close()
    
    # Compile results
    results = {
        'config': config.name,
        'best_val_cosine': best_val_cosine,
        'best_epoch': best_epoch,
        'final_z_alignment': val_metrics['z_alignment'],
        'balanced_accuracy': func_metrics['balanced_accuracy'],
        'output_correlation': func_metrics['output_correlation'],
        'collapse_rate': func_metrics['collapse_rate'],
        'train_time_minutes': train_time / 60,
        'run_dir': str(run_dir),
    }
    
    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")
    
    return results


def run_all_ablations(device: str = "auto", quick: bool = False) -> Dict[str, Dict[str, float]]:
    """Run all ablation experiments."""
    all_results = {}
    
    configs_to_run = ABLATION_CONFIGS
    
    if quick:
        # Quick mode: reduce samples and epochs
        for name, config in configs_to_run.items():
            config.train_samples = 1000
            config.val_samples = 200
            config.epochs = 20
    
    for name, config in configs_to_run.items():
        try:
            results = run_experiment(config, device)
            all_results[name] = results
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            all_results[name] = {'error': str(e)}
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Config':<35} {'Cosine':>8} {'Z-Align':>8} {'Acc':>8} {'OutCorr':>8} {'Collapse':>8}")
    print("-"*80)
    
    for name, results in all_results.items():
        if 'error' in results:
            print(f"{name:<35} ERROR: {results['error'][:30]}")
        else:
            print(f"{name:<35} "
                  f"{results['best_val_cosine']:>8.4f} "
                  f"{results['final_z_alignment']:>8.4f} "
                  f"{results['balanced_accuracy']:>8.4f} "
                  f"{results['output_correlation']:>8.4f} "
                  f"{results['collapse_rate']:>7.1%}")
    
    print("="*80)
    
    # Find best config
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        best_by_corr = max(valid_results.items(), key=lambda x: x[1]['output_correlation'])
        print(f"\nBest by Output Correlation: {best_by_corr[0]} ({best_by_corr[1]['output_correlation']:.4f})")
        
        best_by_cosine = max(valid_results.items(), key=lambda x: x[1]['best_val_cosine'])
        print(f"Best by Weight Cosine: {best_by_cosine[0]} ({best_by_cosine[1]['best_val_cosine']:.4f})")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run dual-encoder ablation experiments")
    parser.add_argument("--config", type=str, default=None, help="Run specific config (e.g., 6a_concat_detach_align_0.5)")
    parser.add_argument("--all", action="store_true", help="Run all ablations")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples, epochs)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    if args.all:
        results = run_all_ablations(device=args.device, quick=args.quick)
    elif args.config:
        if args.config not in ABLATION_CONFIGS:
            print(f"Unknown config: {args.config}")
            print(f"Available: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)
        
        config = ABLATION_CONFIGS[args.config]
        if args.quick:
            config.train_samples = 1000
            config.val_samples = 200
            config.epochs = 20
        
        results = run_experiment(config, device=args.device)
        print(f"\nResults: {results}")
    else:
        # Default: run first config
        config = ABLATION_CONFIGS["6a_concat_detach_align_0.5"]
        if args.quick:
            config.train_samples = 1000
            config.val_samples = 200
            config.epochs = 20
        
        results = run_experiment(config, device=args.device)
        print(f"\nResults: {results}")
