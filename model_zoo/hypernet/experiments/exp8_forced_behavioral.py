"""
Experiment 8: Forced Behavioral Dependence

PROBLEM IDENTIFIED:
In Exp6, the gated decoder learned to ignore Z_behavior (gate α ≈ 0.01).
This defeats the purpose of having an interpretable behavioral latent.

SOLUTION:
Force the decoder to depend on Z_behavior by:
1. Using Z_behavior as the PRIMARY input (not optional via gate)
2. Using Z_weight only as a RESIDUAL/REFINEMENT signal
3. Adding Z_weight dropout during training (force behavioral reliance)

NEW ARCHITECTURE:
```
Z_behavior ──────────────────► Main Decoder Path ──► Base Weights
                                      │
Z_weight ──► Dropout(p=0.5) ──► Residual MLP ──────► Weight Residual
                                      │
                              Base + Residual ──────► Final Weights
```

This ensures:
- Z_behavior MUST contain enough info for reasonable reconstruction
- Z_weight provides fine-grained corrections
- At test time, modifying Z_behavior changes the base prediction
"""

import sys
import time
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.components.encoders import WeightEncoder, BehavioralEncoder, ResidualMLPBlock
from hypernet.components.decoders import SinusoidalPositionEncoder, FiLMBlock
from hypernet.utils.data import create_dataloaders
from hypernet.functional_eval import SubjectModel, load_weights_into_model, create_test_inputs


@dataclass
class ForcedBehavioralConfig:
    """Configuration for forced behavioral experiment."""
    name: str = "forced_behavioral"
    
    # Architecture
    latent_dim: int = 64
    hidden_dim: int = 256
    encoder_layers: int = 3
    decoder_layers: int = 4
    dropout: float = 0.1
    
    # KEY: Z_weight dropout during training
    z_weight_dropout: float = 0.5  # High dropout forces behavioral reliance
    
    # Loss weights
    lambda_func: float = 1.0
    lambda_recon: float = 0.1
    lambda_behavior_recon: float = 0.5  # NEW: Loss on behavior-only reconstruction
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Data
    train_samples: int = 8000
    val_samples: int = 1000


class BehavioralPrimaryDecoder(nn.Module):
    """
    Decoder where Z_behavior is PRIMARY and Z_weight is RESIDUAL.
    
    Architecture:
    1. Main path: Z_behavior + position → Base weights (must work alone!)
    2. Residual path: Z_weight → Weight corrections (dropped out during training)
    3. Output: Base + Residual
    
    This forces Z_behavior to carry the main reconstruction signal.
    """
    
    def __init__(
        self,
        behavioral_dim: int = 64,
        weight_dim: int = 64,
        position_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 9,
        num_layers: int = 4,
        dropout: float = 0.1,
        z_weight_dropout: float = 0.5,
    ):
        super().__init__()
        
        self.z_weight_dropout = z_weight_dropout
        self.output_dim = output_dim
        
        # Position encoder
        self.position_encoder = SinusoidalPositionEncoder(
            input_dim=3,
            output_dim=position_dim
        )
        
        # === MAIN PATH: Z_behavior → Base Weights ===
        # This path must be able to produce reasonable weights alone
        self.main_input_proj = nn.Linear(behavioral_dim + position_dim, hidden_dim)
        self.main_norm = nn.LayerNorm(hidden_dim)
        
        self.main_blocks = nn.ModuleList([
            FiLMBlock(behavioral_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.main_output_norm = nn.LayerNorm(hidden_dim)
        self.main_output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize main output to small values (not zero - we want it to learn)
        nn.init.xavier_uniform_(self.main_output_proj.weight, gain=0.1)
        nn.init.zeros_(self.main_output_proj.bias)
        
        # === RESIDUAL PATH: Z_weight → Weight Corrections ===
        # This provides fine-grained corrections but is dropped out during training
        self.residual_proj = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize residual output to zero (start with no correction)
        nn.init.zeros_(self.residual_proj[-1].weight)
        nn.init.zeros_(self.residual_proj[-1].bias)
        
        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        z_behavior: torch.Tensor,
        z_weight: torch.Tensor,
        positions: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_behavior: [batch, neurons, behavioral_dim]
            z_weight: [batch, neurons, weight_dim]
            positions: [batch, neurons, 3]
            training: Whether to apply z_weight dropout
            
        Returns:
            final_weights: [batch, neurons, output_dim]
            base_weights: [batch, neurons, output_dim] (from behavior only)
        """
        # Handle unbatched
        squeeze = False
        if z_behavior.dim() == 2:
            z_behavior = z_behavior.unsqueeze(0)
            z_weight = z_weight.unsqueeze(0)
            positions = positions.unsqueeze(0)
            squeeze = True
        
        # Position encoding
        pos_enc = self.position_encoder(positions)
        
        # === MAIN PATH ===
        main_input = torch.cat([z_behavior, pos_enc], dim=-1)
        x = self.main_input_proj(main_input)
        x = self.main_norm(x)
        x = F.gelu(x)
        
        for block in self.main_blocks:
            x = block(x, z_behavior)
        
        x = self.main_output_norm(x)
        base_weights = self.main_output_proj(x)
        
        # === RESIDUAL PATH ===
        # Apply dropout to z_weight during training
        if training and self.z_weight_dropout > 0:
            # Dropout entire z_weight vectors (not individual elements)
            dropout_mask = torch.bernoulli(
                torch.ones(z_weight.shape[0], z_weight.shape[1], 1, device=z_weight.device) 
                * (1 - self.z_weight_dropout)
            )
            z_weight_dropped = z_weight * dropout_mask / (1 - self.z_weight_dropout + 1e-8)
        else:
            z_weight_dropped = z_weight
        
        residual = self.residual_proj(z_weight_dropped)
        residual = residual * self.residual_scale
        
        # Combine
        final_weights = base_weights + residual
        
        if squeeze:
            final_weights = final_weights.squeeze(0)
            base_weights = base_weights.squeeze(0)
        
        return final_weights, base_weights


class ForcedBehavioralSystem(nn.Module):
    """
    Full system with forced behavioral dependence.
    """
    
    def __init__(self, config: ForcedBehavioralConfig):
        super().__init__()
        self.config = config
        
        self.behavioral_encoder = BehavioralEncoder(
            behavioral_dim=9,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        self.weight_encoder = WeightEncoder(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        self.decoder = BehavioralPrimaryDecoder(
            behavioral_dim=config.latent_dim,
            weight_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=9,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
            z_weight_dropout=config.z_weight_dropout,
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_weights: Final predicted weights
            base_weights: Weights from behavioral path only
            z_behavior: Behavioral latent
            z_weight: Weight latent
        """
        z_behavior = self.behavioral_encoder(signatures)
        z_weight = self.weight_encoder(weights)
        
        pred_weights, base_weights = self.decoder(
            z_behavior, z_weight, positions, training=training
        )
        
        return pred_weights, base_weights, z_behavior, z_weight
    
    def decode_behavior_only(
        self,
        z_behavior: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Decode using ONLY behavioral latent (for rep-eng)."""
        z_weight_zeros = torch.zeros(
            z_behavior.shape[0], z_behavior.shape[1], self.config.latent_dim,
            device=z_behavior.device
        )
        final, base = self.decoder(z_behavior, z_weight_zeros, positions, training=False)
        return base  # Return base weights (behavior only)


def train_forced_behavioral(
    config: ForcedBehavioralConfig,
    device: str = "auto",
    use_tensorboard: bool = True,
) -> Dict:
    """Train the forced behavioral system."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp8_{config.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 8: Forced Behavioral Dependence")
    print(f"{'='*60}")
    print(f"Z_weight dropout: {config.z_weight_dropout}")
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if use_tensorboard else None
    if writer:
        writer.add_text('config', json.dumps(asdict(config), indent=2))
        print(f"\nTensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Data
    train_loader, val_loader = create_dataloaders(
        batch_size=config.batch_size,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
    )
    
    # Model
    model = ForcedBehavioralSystem(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    
    # Training
    best_val_cosine = 0
    global_step = 0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0
        
        for batch in train_loader:
            signatures = batch['signatures'].to(device)
            weights = batch['weights'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward (with z_weight dropout)
            pred_weights, base_weights, z_behavior, z_weight = model(
                signatures, weights, positions, training=True
            )
            
            # Losses
            mask_exp = mask.unsqueeze(-1)
            
            # 1. Full reconstruction loss
            recon_loss = ((pred_weights - weights) ** 2 * mask_exp).sum() / mask_exp.sum()
            
            # 2. Behavior-only reconstruction loss (KEY: forces main path to work)
            behavior_recon_loss = ((base_weights - weights) ** 2 * mask_exp).sum() / mask_exp.sum()
            
            # Combined loss
            loss = (
                config.lambda_recon * recon_loss +
                config.lambda_behavior_recon * behavior_recon_loss
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_losses['loss'] += loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['behavior_recon'] += behavior_recon_loss.item()
            n_batches += 1
            global_step += 1
            
            if writer and global_step % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
                writer.add_scalar('train/behavior_recon_loss', behavior_recon_loss.item(), global_step)
        
        # Validation
        model.eval()
        val_cosines = []
        val_behavior_cosines = []
        
        with torch.no_grad():
            for batch in val_loader:
                signatures = batch['signatures'].to(device)
                weights = batch['weights'].to(device)
                positions = batch['positions'].to(device)
                mask = batch['mask'].to(device)
                
                pred_weights, base_weights, _, _ = model(
                    signatures, weights, positions, training=False
                )
                
                # Compute cosine similarities
                for b in range(len(mask)):
                    valid = mask[b].bool()
                    if valid.sum() == 0:
                        continue
                    
                    pred_flat = pred_weights[b][valid].flatten()
                    base_flat = base_weights[b][valid].flatten()
                    true_flat = weights[b][valid].flatten()
                    
                    cos_full = F.cosine_similarity(pred_flat.unsqueeze(0), true_flat.unsqueeze(0)).item()
                    cos_base = F.cosine_similarity(base_flat.unsqueeze(0), true_flat.unsqueeze(0)).item()
                    
                    val_cosines.append(cos_full)
                    val_behavior_cosines.append(cos_base)
        
        val_cosine = sum(val_cosines) / len(val_cosines)
        val_behavior_cosine = sum(val_behavior_cosines) / len(val_behavior_cosines)
        
        scheduler.step()
        
        if writer:
            writer.add_scalar('epoch/val_cosine', val_cosine, epoch)
            writer.add_scalar('epoch/val_behavior_cosine', val_behavior_cosine, epoch)
            writer.add_scalar('epoch/residual_scale', model.decoder.residual_scale.item(), epoch)
        
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'val_cosine': val_cosine,
                'val_behavior_cosine': val_behavior_cosine,
            }, run_dir / "best_model.pt")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Loss: {epoch_losses['loss']/n_batches:.4f} | "
                  f"Val Cosine: {val_cosine:.4f} | "
                  f"Behavior-Only Cosine: {val_behavior_cosine:.4f} | "
                  f"Residual Scale: {model.decoder.residual_scale.item():.4f}")
    
    if writer:
        writer.close()
    
    print(f"\nBest val cosine: {best_val_cosine:.4f}")
    print(f"Results saved to: {run_dir}")
    
    return {
        'best_val_cosine': best_val_cosine,
        'run_dir': str(run_dir),
    }


if __name__ == "__main__":
    config = ForcedBehavioralConfig(
        name="forced_behavioral_v1",
        z_weight_dropout=0.5,
        epochs=100,
        train_samples=8000,
        val_samples=1000,
    )
    
    results = train_forced_behavioral(config)
