"""
Experiment 9: Pure Behavioral Representation

GOAL: Create a Z_behavior that is PURELY behavioral (no weight/architectural info)
and can still decode to functional weights.

TWO DESIGNS TESTED:

Design A: Direct Behavioral Mapping
- Z_behavior (from behavioral sigs only) + position → weights
- Auxiliary Z_weight used ONLY for alignment loss during training
- Decoder never sees actual weights

Design B: Weight Prior + Behavioral Residual  
- Learned weight prior provides "base weights" per position
- Z_behavior → residual decoder → weight adjustments
- Final = prior + residual

CRITICAL: Z_behavior uses ONLY behavioral features:
- mean, std, fourier_0-4, pre_activation_mean, pre_activation_std (9 features)
- NO input_correlations (indices 7-14) - they leak weight information!
"""

import sys
import time
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.components.encoders import WeightEncoder, ResidualMLPBlock
from hypernet.components.decoders import SinusoidalPositionEncoder, FiLMBlock, FiLMLayer
from hypernet.utils.data import create_dataloaders
from hypernet.functional_eval import SubjectModel, load_weights_into_model, create_test_inputs

# CRITICAL: Only behavioral features, NO input_correlations
BEHAVIORAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 15, 16]  # 9 features


@dataclass
class PureBehavioralConfig:
    name: str = "pure_behavioral"
    design: str = "A"  # "A" (direct) or "B" (prior)
    
    # Architecture
    latent_dim: int = 64
    hidden_dim: int = 256
    encoder_layers: int = 3
    decoder_layers: int = 4
    dropout: float = 0.1
    
    # Design A specific
    lambda_align: float = 0.5  # Alignment loss weight
    
    # Design B specific
    prior_type: str = "deterministic"  # "deterministic" or "stochastic"
    lambda_residual: float = 0.01  # Residual regularization
    
    # Training
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Data
    train_samples: int = 8000
    val_samples: int = 1000


class PureBehavioralEncoder(nn.Module):
    """
    Encoder that takes ONLY behavioral features.
    Explicitly excludes input_correlations to prevent information leakage.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.behavioral_dim = len(BEHAVIORAL_INDICES)  # 9
        self.latent_dim = latent_dim
        
        self.input_proj = nn.Linear(self.behavioral_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, signatures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signatures: [batch, neurons, 17] full signatures
        Returns:
            z_behavior: [batch, neurons, latent_dim]
        """
        # CRITICAL: Extract ONLY behavioral features
        if signatures.shape[-1] == 17:
            behavioral = signatures[..., BEHAVIORAL_INDICES]
        else:
            behavioral = signatures  # Already extracted
        
        x = self.input_proj(behavioral)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        return x


# ============================================================================
# DESIGN A: Direct Behavioral Mapping
# ============================================================================

class DirectBehavioralDecoder(nn.Module):
    """
    Decoder that maps Z_behavior + position directly to weights.
    NO access to actual weights - must rely entirely on behavioral information.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        position_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 9,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.position_encoder = SinusoidalPositionEncoder(
            input_dim=3,
            output_dim=position_dim
        )
        
        self.input_proj = nn.Linear(latent_dim + position_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.blocks = nn.ModuleList([
            FiLMBlock(latent_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize output to small values
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        z_behavior: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_behavior: [batch, neurons, latent_dim]
            positions: [batch, neurons, 3]
        Returns:
            weights: [batch, neurons, 9]
        """
        squeeze = False
        if z_behavior.dim() == 2:
            z_behavior = z_behavior.unsqueeze(0)
            positions = positions.unsqueeze(0)
            squeeze = True
        
        pos_enc = self.position_encoder(positions)
        
        x = torch.cat([z_behavior, pos_enc], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x, z_behavior)
        
        x = self.output_norm(x)
        weights = self.output_proj(x)
        
        if squeeze:
            weights = weights.squeeze(0)
        
        return weights


class DesignA_DirectBehavioral(nn.Module):
    """
    Design A: Direct mapping from behavioral features to weights.
    
    Training uses auxiliary Z_weight for alignment loss only.
    Inference uses only Z_behavior.
    """
    
    def __init__(self, config: PureBehavioralConfig):
        super().__init__()
        self.config = config
        
        # Main encoder (behavioral features only)
        self.behavioral_encoder = PureBehavioralEncoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        # Decoder (Z_behavior + position → weights)
        self.decoder = DirectBehavioralDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=9,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
        )
        
        # Auxiliary weight encoder (for alignment loss only, not used in inference)
        self.weight_encoder = WeightEncoder(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_weights: Predicted weights from behavioral info
            z_behavior: Behavioral latent
            z_weight: Weight latent (for alignment loss)
        """
        z_behavior = self.behavioral_encoder(signatures)
        z_weight = self.weight_encoder(weights)
        pred_weights = self.decoder(z_behavior, positions)
        
        return pred_weights, z_behavior, z_weight
    
    def encode(self, signatures: torch.Tensor) -> torch.Tensor:
        """Encode behavioral features only."""
        return self.behavioral_encoder(signatures)
    
    def decode(self, z_behavior: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Decode from behavioral latent."""
        return self.decoder(z_behavior, positions)


# ============================================================================
# DESIGN B: Weight Prior + Behavioral Residual
# ============================================================================

class WeightPrior(nn.Module):
    """
    Learned weight prior conditioned on position.
    Provides "base weights" that represent typical weights for each position.
    """
    
    def __init__(
        self,
        position_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 9,
        prior_type: str = "deterministic",
    ):
        super().__init__()
        
        self.prior_type = prior_type
        self.output_dim = output_dim
        
        self.position_encoder = SinusoidalPositionEncoder(
            input_dim=3,
            output_dim=position_dim
        )
        
        # MLP to predict prior parameters
        self.prior_net = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        if prior_type == "deterministic":
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:  # stochastic
            self.mean_layer = nn.Linear(hidden_dim, output_dim)
            self.logvar_layer = nn.Linear(hidden_dim, output_dim)
            # Initialize logvar to produce std ≈ 0.35 (from data analysis)
            nn.init.zeros_(self.logvar_layer.weight)
            nn.init.constant_(self.logvar_layer.bias, -2.0)  # exp(-2) ≈ 0.14, but we'll learn
    
    def forward(self, positions: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Args:
            positions: [batch, neurons, 3]
            sample: If True and stochastic, sample from distribution
        Returns:
            w_prior: [batch, neurons, output_dim]
        """
        squeeze = False
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            squeeze = True
        
        pos_enc = self.position_encoder(positions)
        h = self.prior_net(pos_enc)
        
        if self.prior_type == "deterministic":
            w_prior = self.output_layer(h)
        else:
            mean = self.mean_layer(h)
            logvar = self.logvar_layer(h)
            
            if sample and self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                w_prior = mean + eps * std
            else:
                w_prior = mean
        
        if squeeze:
            w_prior = w_prior.squeeze(0)
        
        return w_prior


class BehavioralResidualDecoder(nn.Module):
    """
    Decoder that outputs weight RESIDUALS based on Z_behavior.
    Final weights = prior + residual
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        position_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 9,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.position_encoder = SinusoidalPositionEncoder(
            input_dim=3,
            output_dim=position_dim
        )
        
        self.input_proj = nn.Linear(latent_dim + position_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.blocks = nn.ModuleList([
            FiLMBlock(latent_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize output to ZERO (start with no residual)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        z_behavior: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Returns weight residual."""
        squeeze = False
        if z_behavior.dim() == 2:
            z_behavior = z_behavior.unsqueeze(0)
            positions = positions.unsqueeze(0)
            squeeze = True
        
        pos_enc = self.position_encoder(positions)
        
        x = torch.cat([z_behavior, pos_enc], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x, z_behavior)
        
        x = self.output_norm(x)
        residual = self.output_proj(x)
        
        if squeeze:
            residual = residual.squeeze(0)
        
        return residual


class DesignB_WeightPrior(nn.Module):
    """
    Design B: Weight prior + behavioral residual.
    
    Prior provides base weights (position-dependent, learned).
    Z_behavior modulates via residual.
    """
    
    def __init__(self, config: PureBehavioralConfig):
        super().__init__()
        self.config = config
        
        # Behavioral encoder
        self.behavioral_encoder = PureBehavioralEncoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        # Weight prior (position → base weights)
        self.weight_prior = WeightPrior(
            hidden_dim=config.hidden_dim // 2,
            output_dim=9,
            prior_type=config.prior_type,
        )
        
        # Residual decoder (Z_behavior → weight adjustments)
        self.residual_decoder = BehavioralResidualDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=9,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_weights: Final predicted weights (prior + residual)
            w_prior: Base weights from prior
            residual: Weight adjustments from behavioral decoder
        """
        z_behavior = self.behavioral_encoder(signatures)
        w_prior = self.weight_prior(positions)
        residual = self.residual_decoder(z_behavior, positions)
        
        pred_weights = w_prior + residual
        
        return pred_weights, w_prior, residual
    
    def encode(self, signatures: torch.Tensor) -> torch.Tensor:
        """Encode behavioral features only."""
        return self.behavioral_encoder(signatures)
    
    def decode(self, z_behavior: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Decode from behavioral latent."""
        w_prior = self.weight_prior(positions, sample=False)
        residual = self.residual_decoder(z_behavior, positions)
        return w_prior + residual


# ============================================================================
# TRAINING
# ============================================================================

def compute_alignment_loss(
    z_behavior: torch.Tensor,
    z_weight: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Encourage Z_behavior to capture information in Z_weight."""
    # Detach z_weight so gradients only flow to behavioral encoder
    z_w = z_weight.detach()
    
    # Normalize
    z_b_norm = F.normalize(z_behavior, dim=-1)
    z_w_norm = F.normalize(z_w, dim=-1)
    
    # Cosine similarity
    cos_sim = (z_b_norm * z_w_norm).sum(dim=-1)
    
    # Masked mean (loss = 1 - similarity)
    loss = (1 - cos_sim) * mask
    return loss.sum() / mask.sum()


def train_design_a(
    config: PureBehavioralConfig,
    device: str,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[DesignA_DirectBehavioral, Dict]:
    """Train Design A: Direct Behavioral Mapping."""
    
    print("\n" + "="*60)
    print("DESIGN A: Direct Behavioral Mapping")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(
        batch_size=config.batch_size,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
    )
    
    model = DesignA_DirectBehavioral(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    
    best_val_cosine = 0
    history = {'train_loss': [], 'val_cosine': [], 'val_align': []}
    global_step = 0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            signatures = batch['signatures'].to(device)
            weights = batch['weights'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            pred_weights, z_behavior, z_weight = model(signatures, weights, positions)
            
            # Reconstruction loss
            mask_exp = mask.unsqueeze(-1)
            recon_loss = ((pred_weights - weights) ** 2 * mask_exp).sum() / mask_exp.sum()
            
            # Alignment loss
            align_loss = compute_alignment_loss(z_behavior, z_weight, mask)
            
            loss = recon_loss + config.lambda_align * align_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            if writer and global_step % 20 == 0:
                writer.add_scalar('A/train_loss', loss.item(), global_step)
                writer.add_scalar('A/recon_loss', recon_loss.item(), global_step)
                writer.add_scalar('A/align_loss', align_loss.item(), global_step)
        
        # Validation
        model.eval()
        val_cosines = []
        val_aligns = []
        
        with torch.no_grad():
            for batch in val_loader:
                signatures = batch['signatures'].to(device)
                weights = batch['weights'].to(device)
                positions = batch['positions'].to(device)
                mask = batch['mask'].to(device)
                
                pred_weights, z_behavior, z_weight = model(signatures, weights, positions)
                
                for b in range(len(mask)):
                    valid = mask[b].bool()
                    if valid.sum() == 0:
                        continue
                    
                    cos = F.cosine_similarity(
                        pred_weights[b][valid].flatten().unsqueeze(0),
                        weights[b][valid].flatten().unsqueeze(0)
                    ).item()
                    val_cosines.append(cos)
                    
                    # Z alignment
                    z_b = z_behavior[b][valid]
                    z_w = z_weight[b][valid]
                    align = F.cosine_similarity(
                        z_b.flatten().unsqueeze(0),
                        z_w.flatten().unsqueeze(0)
                    ).item()
                    val_aligns.append(align)
        
        val_cosine = sum(val_cosines) / len(val_cosines)
        val_align = sum(val_aligns) / len(val_aligns)
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_cosine'].append(val_cosine)
        history['val_align'].append(val_align)
        
        if writer:
            writer.add_scalar('A/val_cosine', val_cosine, epoch)
            writer.add_scalar('A/val_align', val_align, epoch)
        
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Loss: {epoch_loss/n_batches:.4f} | "
                  f"Val Cosine: {val_cosine:.4f} | "
                  f"Z-Align: {val_align:.4f}")
    
    print(f"\nDesign A Best Val Cosine: {best_val_cosine:.4f}")
    
    return model, {'best_val_cosine': best_val_cosine, 'history': history}


def train_design_b(
    config: PureBehavioralConfig,
    device: str,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[DesignB_WeightPrior, Dict]:
    """Train Design B: Weight Prior + Behavioral Residual."""
    
    print("\n" + "="*60)
    print(f"DESIGN B: Weight Prior ({config.prior_type}) + Behavioral Residual")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(
        batch_size=config.batch_size,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
    )
    
    model = DesignB_WeightPrior(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    
    best_val_cosine = 0
    history = {'train_loss': [], 'val_cosine': [], 'prior_cosine': [], 'residual_norm': []}
    global_step = 0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            signatures = batch['signatures'].to(device)
            weights = batch['weights'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            pred_weights, w_prior, residual = model(signatures, positions)
            
            mask_exp = mask.unsqueeze(-1)
            
            # Reconstruction loss
            recon_loss = ((pred_weights - weights) ** 2 * mask_exp).sum() / mask_exp.sum()
            
            # Residual regularization (encourage small residuals)
            residual_loss = (residual ** 2 * mask_exp).sum() / mask_exp.sum()
            
            loss = recon_loss + config.lambda_residual * residual_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            if writer and global_step % 20 == 0:
                writer.add_scalar('B/train_loss', loss.item(), global_step)
                writer.add_scalar('B/recon_loss', recon_loss.item(), global_step)
                writer.add_scalar('B/residual_loss', residual_loss.item(), global_step)
        
        # Validation
        model.eval()
        val_cosines = []
        prior_cosines = []
        residual_norms = []
        
        with torch.no_grad():
            for batch in val_loader:
                signatures = batch['signatures'].to(device)
                weights = batch['weights'].to(device)
                positions = batch['positions'].to(device)
                mask = batch['mask'].to(device)
                
                pred_weights, w_prior, residual = model(signatures, positions)
                
                for b in range(len(mask)):
                    valid = mask[b].bool()
                    if valid.sum() == 0:
                        continue
                    
                    # Full prediction cosine
                    cos = F.cosine_similarity(
                        pred_weights[b][valid].flatten().unsqueeze(0),
                        weights[b][valid].flatten().unsqueeze(0)
                    ).item()
                    val_cosines.append(cos)
                    
                    # Prior-only cosine (how good is prior alone?)
                    prior_cos = F.cosine_similarity(
                        w_prior[b][valid].flatten().unsqueeze(0),
                        weights[b][valid].flatten().unsqueeze(0)
                    ).item()
                    prior_cosines.append(prior_cos)
                    
                    # Residual norm
                    res_norm = residual[b][valid].norm().item()
                    residual_norms.append(res_norm)
        
        val_cosine = sum(val_cosines) / len(val_cosines)
        prior_cosine = sum(prior_cosines) / len(prior_cosines)
        residual_norm = sum(residual_norms) / len(residual_norms)
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_cosine'].append(val_cosine)
        history['prior_cosine'].append(prior_cosine)
        history['residual_norm'].append(residual_norm)
        
        if writer:
            writer.add_scalar('B/val_cosine', val_cosine, epoch)
            writer.add_scalar('B/prior_cosine', prior_cosine, epoch)
            writer.add_scalar('B/residual_norm', residual_norm, epoch)
        
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Loss: {epoch_loss/n_batches:.4f} | "
                  f"Val Cosine: {val_cosine:.4f} | "
                  f"Prior Cosine: {prior_cosine:.4f} | "
                  f"Residual Norm: {residual_norm:.4f}")
    
    print(f"\nDesign B Best Val Cosine: {best_val_cosine:.4f}")
    
    return model, {'best_val_cosine': best_val_cosine, 'history': history}


# ============================================================================
# CAUSALITY TEST
# ============================================================================

@torch.no_grad()
def test_causality(model, device: str, design: str, n_samples: int = 50) -> Dict:
    """
    Test if modifying Z_behavior causes output changes.
    This is the KEY test for interpretability.
    """
    print("\n" + "="*60)
    print(f"CAUSALITY TEST (Design {design})")
    print("="*60)
    
    _, val_loader = create_dataloaders(batch_size=8, train_samples=100, val_samples=100)
    
    model.eval()
    
    noise_scales = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    effects = []
    
    samples_tested = 0
    
    for batch in val_loader:
        if samples_tested >= n_samples:
            break
        
        signatures = batch['signatures'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        # Encode
        z_behavior = model.encode(signatures)
        
        for b in range(len(mask)):
            if samples_tested >= n_samples:
                break
            
            valid = mask[b].bool()
            if valid.sum() == 0:
                continue
            
            z_b = z_behavior[b:b+1]
            pos = positions[b:b+1]
            
            # Original prediction
            pred_orig = model.decode(z_b, pos)[0][valid]
            
            sample_effects = []
            for noise_scale in noise_scales:
                if noise_scale == 0:
                    sample_effects.append(0.0)
                    continue
                
                # Add noise to Z_behavior
                noise = torch.randn_like(z_b) * noise_scale
                z_b_mod = z_b + noise
                
                # Decode modified
                pred_mod = model.decode(z_b_mod, pos)[0][valid]
                
                # Measure change
                change = 1 - F.cosine_similarity(
                    pred_orig.flatten().unsqueeze(0),
                    pred_mod.flatten().unsqueeze(0)
                ).item()
                sample_effects.append(change)
            
            effects.append(sample_effects)
            samples_tested += 1
    
    # Average effects
    effects = torch.tensor(effects)
    mean_effects = effects.mean(dim=0).tolist()
    
    print(f"\nZ_behavior Modification Effects:")
    print(f"{'Noise Scale':<15} {'Output Change':>15}")
    print("-" * 32)
    for scale, effect in zip(noise_scales, mean_effects):
        print(f"{scale:<15.1f} {effect:>15.4f}")
    
    # Sensitivity = effect per unit noise
    sensitivity = mean_effects[3] / (noise_scales[3] + 1e-8)  # at noise=0.5
    print(f"\nSensitivity (effect @ noise=0.5): {mean_effects[3]:.4f}")
    print(f"(Higher = Z_behavior modifications have more effect)")
    
    return {
        'noise_scales': noise_scales,
        'mean_effects': mean_effects,
        'sensitivity': sensitivity,
    }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison(device: str = "auto"):
    """Run both designs and compare."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp9_comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 9: Pure Behavioral Representation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    print(f"\nTensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")
    
    # Config for both designs
    config_a = PureBehavioralConfig(
        name="design_a_direct",
        design="A",
        lambda_align=0.5,
        epochs=50,
        train_samples=8000,
        val_samples=1000,
    )
    
    config_b_det = PureBehavioralConfig(
        name="design_b_deterministic",
        design="B",
        prior_type="deterministic",
        lambda_residual=0.01,
        epochs=50,
        train_samples=8000,
        val_samples=1000,
    )
    
    config_b_sto = PureBehavioralConfig(
        name="design_b_stochastic",
        design="B",
        prior_type="stochastic",
        lambda_residual=0.01,
        epochs=50,
        train_samples=8000,
        val_samples=1000,
    )
    
    results = {}
    
    # Train Design A
    model_a, results_a = train_design_a(config_a, device, writer)
    results['A'] = results_a
    causality_a = test_causality(model_a, device, "A")
    results['A']['causality'] = causality_a
    
    # Save Design A
    torch.save({
        'model_state_dict': model_a.state_dict(),
        'config': asdict(config_a),
        'results': results_a,
        'causality': causality_a,
    }, run_dir / "design_a.pt")
    
    # Train Design B (deterministic prior)
    model_b_det, results_b_det = train_design_b(config_b_det, device, writer)
    results['B_det'] = results_b_det
    causality_b_det = test_causality(model_b_det, device, "B_det")
    results['B_det']['causality'] = causality_b_det
    
    torch.save({
        'model_state_dict': model_b_det.state_dict(),
        'config': asdict(config_b_det),
        'results': results_b_det,
        'causality': causality_b_det,
    }, run_dir / "design_b_deterministic.pt")
    
    # Train Design B (stochastic prior)
    model_b_sto, results_b_sto = train_design_b(config_b_sto, device, writer)
    results['B_sto'] = results_b_sto
    causality_b_sto = test_causality(model_b_sto, device, "B_sto")
    results['B_sto']['causality'] = causality_b_sto
    
    torch.save({
        'model_state_dict': model_b_sto.state_dict(),
        'config': asdict(config_b_sto),
        'results': results_b_sto,
        'causality': causality_b_sto,
    }, run_dir / "design_b_stochastic.pt")
    
    writer.close()
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Design':<25} {'Val Cosine':>12} {'Sensitivity':>12}")
    print("-"*50)
    print(f"{'A: Direct':<25} {results['A']['best_val_cosine']:>12.4f} {results['A']['causality']['sensitivity']:>12.4f}")
    print(f"{'B: Prior (deterministic)':<25} {results['B_det']['best_val_cosine']:>12.4f} {results['B_det']['causality']['sensitivity']:>12.4f}")
    print(f"{'B: Prior (stochastic)':<25} {results['B_sto']['best_val_cosine']:>12.4f} {results['B_sto']['causality']['sensitivity']:>12.4f}")
    
    # Determine winner
    # Prioritize causality (sensitivity) since that's key for interpretability
    sensitivities = {
        'A': results['A']['causality']['sensitivity'],
        'B_det': results['B_det']['causality']['sensitivity'],
        'B_sto': results['B_sto']['causality']['sensitivity'],
    }
    winner = max(sensitivities, key=sensitivities.get)
    
    print(f"\n*** WINNER (highest causality): {winner} ***")
    print(f"Sensitivity: {sensitivities[winner]:.4f}")
    
    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump({
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in results.items()},
            'winner': winner,
        }, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")
    
    return results, winner


if __name__ == "__main__":
    results, winner = run_comparison()
