"""
Experiment 2: Direct Signature -> Weight Prediction

Question: How well can signatures directly predict weights (no latent)?

This gives us a baseline for signature predictive power.
Compare to the empirical input_correlation analysis (~0.37 cosine).

Architecture:
    Signatures -> MLP -> Weights

Success Criterion: Beat 0.37 cosine (input_correlation baseline)
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.components.encoders import ResidualMLPBlock
from hypernet.utils.data import create_dataloaders


class DirectPredictor(nn.Module):
    """
    Direct signature -> weight predictor.
    
    Simple MLP that predicts weights from signatures.
    Does NOT use position information to test pure signature predictive power.
    """
    
    def __init__(
        self,
        signature_dim: int = 17,
        output_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(signature_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
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
            signatures: [batch, num_neurons, signature_dim]
            
        Returns:
            weights: [batch, num_neurons, output_dim]
        """
        x = self.input_proj(signatures)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        return self.output_proj(x)


class PositionAwarePredictor(nn.Module):
    """
    Signature + Position -> Weight predictor.
    
    Uses both signature and position info to predict weights.
    This should do better than DirectPredictor.
    """
    
    def __init__(
        self,
        signature_dim: int = 17,
        position_dim: int = 3,
        output_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        from hypernet.components.decoders import SinusoidalPositionEncoder
        
        # Position encoder
        self.pos_encoder = SinusoidalPositionEncoder(
            input_dim=3, output_dim=32
        )
        
        # Input projection (signature + position encoding)
        self.input_proj = nn.Linear(signature_dim + 32, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, signatures: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signatures: [batch, num_neurons, signature_dim]
            positions: [batch, num_neurons, 3]
            
        Returns:
            weights: [batch, num_neurons, output_dim]
        """
        pos_enc = self.pos_encoder(positions)
        x = torch.cat([signatures, pos_enc], dim=-1)
        
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_norm(x)
        return self.output_proj(x)


def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
        cos_sims.append(cos_sim)
    
    if len(cos_sims) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    return torch.stack(cos_sims).mean()


def train_epoch(model, loader, optimizer, device, use_position=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cosine = 0
    num_batches = 0
    
    for batch in loader:
        signatures = batch['signatures'].to(device)
        weights = batch['weights'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        if use_position:
            pred = model(signatures, positions)
        else:
            pred = model(signatures)
        
        # Masked MSE loss
        mask_expanded = mask.unsqueeze(-1)
        mse_loss = ((pred - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
        
        with torch.no_grad():
            cos_sim = compute_cosine_similarity(pred, weights, mask)
        
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += mse_loss.item()
        total_cosine += cos_sim.item()
        num_batches += 1
    
    return total_loss / num_batches, total_cosine / num_batches


def validate(model, loader, device, use_position=False):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_cosine = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            signatures = batch['signatures'].to(device)
            weights = batch['weights'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            if use_position:
                pred = model(signatures, positions)
            else:
                pred = model(signatures)
            
            mask_expanded = mask.unsqueeze(-1)
            mse_loss = ((pred - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
            cos_sim = compute_cosine_similarity(pred, weights, mask)
            
            total_loss += mse_loss.item()
            total_cosine += cos_sim.item()
            num_batches += 1
    
    return total_loss / num_batches, total_cosine / num_batches


def run_experiment(
    train_samples: int = 8000,
    val_samples: int = 1000,
    batch_size: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 4,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "auto",
):
    """Run the direct prediction experiment."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("=" * 60)
    print("EXPERIMENT 2: Direct Signature -> Weight Prediction")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    
    results = {}
    
    # Test 1: Direct prediction (no position)
    print("\n" + "=" * 60)
    print("TEST A: Signature -> Weights (NO position)")
    print("=" * 60)
    
    model_direct = DirectPredictor(
        signature_dim=17,
        output_dim=9,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model_direct.parameters()):,}")
    
    optimizer = AdamW(model_direct.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    best_cosine_direct = 0
    start = time.time()
    
    for epoch in range(epochs):
        train_loss, train_cos = train_epoch(model_direct, train_loader, optimizer, device, use_position=False)
        val_loss, val_cos = validate(model_direct, val_loader, device, use_position=False)
        scheduler.step()
        
        if val_cos > best_cosine_direct:
            best_cosine_direct = val_cos
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | Val Cosine: {val_cos:.4f}")
    
    print(f"\nDirect prediction (no position): {best_cosine_direct:.4f}")
    results['direct_no_position'] = best_cosine_direct
    
    # Test 2: Position-aware prediction
    print("\n" + "=" * 60)
    print("TEST B: Signature + Position -> Weights")
    print("=" * 60)
    
    model_pos = PositionAwarePredictor(
        signature_dim=17,
        output_dim=9,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model_pos.parameters()):,}")
    
    optimizer = AdamW(model_pos.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    best_cosine_pos = 0
    
    for epoch in range(epochs):
        train_loss, train_cos = train_epoch(model_pos, train_loader, optimizer, device, use_position=True)
        val_loss, val_cos = validate(model_pos, val_loader, device, use_position=True)
        scheduler.step()
        
        if val_cos > best_cosine_pos:
            best_cosine_pos = val_cos
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | Val Cosine: {val_cos:.4f}")
    
    print(f"\nPosition-aware prediction: {best_cosine_pos:.4f}")
    results['position_aware'] = best_cosine_pos
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Empirical baseline (input_correlations): ~0.37")
    print(f"Direct prediction (no position):         {best_cosine_direct:.4f}")
    print(f"Position-aware prediction:               {best_cosine_pos:.4f}")
    print(f"Weight autoencoder upper bound:          ~0.9998")
    print()
    
    if best_cosine_pos > 0.37:
        print("SUCCESS: Signatures + position beat input_correlation baseline!")
    else:
        print("NOTE: Did not beat input_correlation baseline yet.")
    
    if best_cosine_pos > 0.90:
        print("EXCELLENT: Signatures + position achieve >0.90 cosine!")
    
    return results


if __name__ == "__main__":
    results = run_experiment(
        train_samples=1000,
        val_samples=200,
        epochs=30,
    )
