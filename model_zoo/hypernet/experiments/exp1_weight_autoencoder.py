"""
Experiment 1: Weight Autoencoder Baseline

Question: Can we encode/decode weights through a latent with high fidelity?

This establishes the upper bound on reconstruction quality.
If we can't achieve >0.95 cosine similarity here, something is fundamentally broken.

Architecture:
    Weights -> WeightEncoder -> Latent -> HypernetDecoder + Position -> Weights'

Success Criterion: >= 0.95 cosine similarity on validation set
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.components.decoders import WeightAutoencoder
from hypernet.utils.data import create_dataloaders


def compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between predictions and targets, respecting mask.
    
    Args:
        pred: [batch, num_neurons, dim]
        target: [batch, num_neurons, dim]
        mask: [batch, num_neurons]
    
    Returns:
        mean cosine similarity across batch
    """
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


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cosine = 0
    num_batches = 0
    
    for batch in loader:
        weights = batch['weights'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        reconstructed, latent = model(weights, positions, mask)
        
        # Masked MSE loss
        mask_expanded = mask.unsqueeze(-1)
        mse_loss = ((reconstructed - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
        
        # Cosine similarity (for monitoring)
        with torch.no_grad():
            cos_sim = compute_cosine_similarity(reconstructed, weights, mask)
        
        # Backward
        mse_loss.backward()
        
        # Gradient clipping (hypernetwork best practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += mse_loss.item()
        total_cosine += cos_sim.item()
        num_batches += 1
    
    return total_loss / num_batches, total_cosine / num_batches


def validate(model, loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_cosine = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            weights = batch['weights'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            reconstructed, latent = model(weights, positions, mask)
            
            mask_expanded = mask.unsqueeze(-1)
            mse_loss = ((reconstructed - weights) ** 2 * mask_expanded).sum() / mask_expanded.sum()
            cos_sim = compute_cosine_similarity(reconstructed, weights, mask)
            
            total_loss += mse_loss.item()
            total_cosine += cos_sim.item()
            num_batches += 1
    
    return total_loss / num_batches, total_cosine / num_batches


def run_experiment(
    # Data params
    train_samples: int = 8000,
    val_samples: int = 1000,
    batch_size: int = 64,
    # Model params
    latent_dim: int = 64,
    hidden_dim: int = 256,
    encoder_layers: int = 3,
    decoder_layers: int = 4,
    dropout: float = 0.1,
    # Training params
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    # Other
    device: str = "auto",
    verbose: bool = True,
):
    """
    Run the weight autoencoder experiment.
    
    Returns:
        dict with final metrics
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"=" * 60)
    print("EXPERIMENT 1: Weight Autoencoder Baseline")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Train samples: {train_samples}, Val samples: {val_samples}")
    print(f"Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")
    print(f"Encoder layers: {encoder_layers}, Decoder layers: {decoder_layers}")
    print(f"Learning rate: {lr}, Epochs: {epochs}")
    print()
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = WeightAutoencoder(
        input_dim=9,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        dropout=dropout,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    best_cosine = 0
    best_epoch = 0
    
    print()
    print("Training...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_cosine = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_cosine = validate(model, val_loader, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        if val_cosine > best_cosine:
            best_cosine = val_cosine
            best_epoch = epoch
        
        if verbose or epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Cos: {train_cosine:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Cos: {val_cosine:.4f} | "
                  f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s")
    print()
    print(f"RESULTS:")
    print(f"  Best validation cosine: {best_cosine:.4f} (epoch {best_epoch+1})")
    print(f"  Final validation cosine: {val_cosine:.4f}")
    print()
    
    if best_cosine >= 0.95:
        print("SUCCESS: Achieved >= 0.95 cosine similarity!")
        print("Weight autoencoder can reconstruct weights through latent.")
    elif best_cosine >= 0.90:
        print("PARTIAL SUCCESS: Achieved >= 0.90 cosine similarity.")
        print("Consider training longer or increasing model capacity.")
    else:
        print(f"NEEDS WORK: Only achieved {best_cosine:.4f} cosine similarity.")
        print("Check model architecture or training setup.")
    
    return {
        'best_cosine': best_cosine,
        'best_epoch': best_epoch,
        'final_cosine': val_cosine,
        'final_loss': val_loss,
        'total_time': total_time,
        'model': model,
    }


if __name__ == "__main__":
    results = run_experiment(
        train_samples=8000,
        val_samples=1000,
        batch_size=64,
        epochs=50,
        lr=1e-3,
        verbose=True,
    )
