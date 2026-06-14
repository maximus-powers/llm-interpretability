"""
Experiment 12B: Learned Bottleneck for Behavioral Representation

Since UWH doesn't hold for our input_correlations (they're behavioral, not raw weights),
we'll learn the optimal bottleneck dimensionality empirically.

Approach:
1. Train encoder-decoder with varying bottleneck sizes (8, 16, 32, 64, 128)
2. Joint training with classification loss
3. Find the minimum bottleneck that maintains good classification

The hypothesis: There exists a low-dimensional behavioral manifold, even if it's
not the same as the UWH weight manifold.
"""

import sys
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import load_dataset as hf_load_dataset

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# All 17 features (best performance from Exp11)
ALL_INDICES = list(range(17))

PATTERN_LABELS = [
    'contains_abc', 'palindrome', 'alternating', 'sorted_descending',
    'sorted_ascending', 'ends_with', 'decreasing_pairs', 'mountain_pattern',
    'increasing_pairs', 'first_last_match', 'starts_with', 'no_repeats',
    'has_majority', 'vowel_consonant'
]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_LABELS)}
NUM_CLASSES = len(PATTERN_LABELS)


@dataclass
class BottleneckConfig:
    name: str = "bottleneck"
    
    # Architecture
    bottleneck_dim: int = 16  # The key variable we're testing
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.2
    max_neurons: int = 64
    
    # Training
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    
    # Loss weights
    lambda_class: float = 1.0
    lambda_recon: float = 0.1  # Reconstruction of signatures
    
    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.1


class SignatureDataset(Dataset):
    """Dataset returning full signature tensors."""
    
    def __init__(self, hf_dataset, config: BottleneckConfig):
        self.hf_dataset = hf_dataset
        self.config = config
        self.feature_dim = 17
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.hf_dataset[idx]
        
        pattern = sample['classification_completion']
        label = PATTERN_TO_IDX.get(pattern, -1)
        
        sig_data = json.loads(sample['improved_signature'])
        neuron_activations = sig_data['neuron_activations']
        
        signatures = []
        layer_ids = []
        neuron_ids = []
        
        layer_indices = sorted([int(k) for k in neuron_activations.keys()])
        
        for layer_idx in layer_indices:
            layer_data = neuron_activations.get(str(layer_idx), {})
            neuron_profiles = layer_data.get('neuron_profiles', {})
            
            for neuron_idx in sorted([int(k) for k in neuron_profiles.keys()]):
                profile = neuron_profiles[str(neuron_idx)]
                sig = self._extract_signature(profile)
                signatures.append(sig)
                layer_ids.append(layer_idx // 2)
                neuron_ids.append(neuron_idx)
        
        num_real = len(signatures)
        
        while len(signatures) < self.config.max_neurons:
            signatures.append(torch.zeros(17))
            layer_ids.append(0)
            neuron_ids.append(0)
        
        signatures = signatures[:self.config.max_neurons]
        layer_ids = layer_ids[:self.config.max_neurons]
        neuron_ids = neuron_ids[:self.config.max_neurons]
        
        signatures = torch.stack(signatures)
        
        mask = torch.zeros(self.config.max_neurons)
        mask[:min(num_real, self.config.max_neurons)] = 1.0
        
        return {
            'signatures': signatures,
            'mask': mask,
            'layer_ids': torch.tensor(layer_ids, dtype=torch.long),
            'neuron_ids': torch.tensor(neuron_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'num_neurons': min(num_real, self.config.max_neurons),
        }
    
    def _extract_signature(self, profile: Dict) -> torch.Tensor:
        sig = torch.zeros(17)
        sig[0] = profile.get('mean', 0)
        sig[1] = profile.get('std', 0)
        
        fourier = profile.get('fourier', [0] * 5)
        for i, f in enumerate(fourier[:5]):
            sig[2 + i] = f
        
        input_corr = profile.get('input_correlations', [0] * 8)
        for i, c in enumerate(input_corr[:8]):
            sig[7 + i] = c
        
        sig[15] = profile.get('pre_activation_mean', 0)
        sig[16] = profile.get('pre_activation_std', 0)
        
        return sig


class BottleneckEncoder(nn.Module):
    """Transformer encoder with bottleneck output."""
    
    def __init__(self, config: BottleneckConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(17, config.hidden_dim)
        
        # Position embeddings
        self.layer_embedding = nn.Embedding(10, config.hidden_dim // 2)
        self.neuron_embedding = nn.Embedding(config.max_neurons, config.hidden_dim // 2)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Bottleneck projection
        self.bottleneck_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
        )
    
    def forward(self, signatures, mask, layer_ids, neuron_ids):
        batch_size = signatures.shape[0]
        
        x = self.input_proj(signatures)
        
        layer_emb = self.layer_embedding(layer_ids.clamp(0, 9))
        neuron_emb = self.neuron_embedding(neuron_ids.clamp(0, self.config.max_neurons - 1))
        pos_emb = torch.cat([layer_emb, neuron_emb], dim=-1)
        x = x + pos_emb
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        cls_mask = torch.ones(batch_size, 1, device=mask.device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        attn_mask = (extended_mask == 0)
        
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        cls_output = x[:, 0]
        
        # Project to bottleneck
        z = self.bottleneck_proj(cls_output)
        
        return z, cls_output  # Return both bottleneck and full representation


class BottleneckDecoder(nn.Module):
    """Decoder that reconstructs signature statistics from bottleneck."""
    
    def __init__(self, config: BottleneckConfig):
        super().__init__()
        self.config = config
        
        # Expand bottleneck
        self.expand = nn.Sequential(
            nn.Linear(config.bottleneck_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Reconstruct aggregated signature statistics
        # Instead of per-neuron reconstruction, we reconstruct summary stats
        self.recon_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 17 * 3),  # mean, std, max per feature
        )
    
    def forward(self, z):
        h = self.expand(z)
        recon = self.recon_head(h)
        return recon.view(-1, 17, 3)  # [batch, 17 features, 3 stats]


class BottleneckModel(nn.Module):
    """Combined encoder-decoder with classification head."""
    
    def __init__(self, config: BottleneckConfig):
        super().__init__()
        self.config = config
        
        self.encoder = BottleneckEncoder(config)
        self.decoder = BottleneckDecoder(config)
        
        # Classification from bottleneck
        self.classifier = nn.Sequential(
            nn.Linear(config.bottleneck_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, NUM_CLASSES),
        )
    
    def forward(self, signatures, mask, layer_ids, neuron_ids):
        z, _ = self.encoder(signatures, mask, layer_ids, neuron_ids)
        logits = self.classifier(z)
        recon = self.decoder(z)
        return logits, z, recon
    
    def compute_target_stats(self, signatures, mask):
        """Compute target statistics for reconstruction loss."""
        # Masked mean, std, max per feature
        mask_expanded = mask.unsqueeze(-1)  # [batch, neurons, 1]
        masked_sigs = signatures * mask_expanded
        
        # Sum and count for mean
        sig_sum = masked_sigs.sum(dim=1)  # [batch, 17]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        sig_mean = sig_sum / counts
        
        # Std
        diff_sq = ((signatures - sig_mean.unsqueeze(1)) ** 2) * mask_expanded
        sig_var = diff_sq.sum(dim=1) / counts
        sig_std = sig_var.sqrt()
        
        # Max (use large negative for masked positions)
        masked_for_max = signatures.clone()
        masked_for_max[mask == 0] = -1e9
        sig_max = masked_for_max.max(dim=1)[0]
        
        return torch.stack([sig_mean, sig_std, sig_max], dim=-1)  # [batch, 17, 3]


def create_dataloaders(config: BottleneckConfig):
    """Create train/val/test dataloaders."""
    print("Loading dataset...")
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    
    total = len(hf_ds)
    train_size = int(total * config.train_ratio)
    val_size = int(total * config.val_ratio)
    
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_ds = SignatureDataset(hf_ds.select(train_indices), config)
    val_ds = SignatureDataset(hf_ds.select(val_indices), config)
    test_ds = SignatureDataset(hf_ds.select(test_indices), config)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def train_bottleneck(config: BottleneckConfig, device: str = "auto") -> Dict:
    """Train bottleneck model and evaluate."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Training: bottleneck_dim={config.bottleneck_dim}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    model = BottleneckModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    recon_criterion = nn.MSELoss()
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            signatures = batch['signatures'].to(device)
            mask = batch['mask'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            neuron_ids = batch['neuron_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits, z, recon = model(signatures, mask, layer_ids, neuron_ids)
            
            # Classification loss
            loss_class = class_criterion(logits, labels)
            
            # Reconstruction loss
            target_stats = model.compute_target_stats(signatures, mask)
            loss_recon = recon_criterion(recon, target_stats)
            
            # Combined loss
            loss = config.lambda_class * loss_class + config.lambda_recon * loss_recon
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                signatures = batch['signatures'].to(device)
                mask = batch['mask'].to(device)
                layer_ids = batch['layer_ids'].to(device)
                neuron_ids = batch['neuron_ids'].to(device)
                labels = batch['label'].to(device)
                
                logits, _, _ = model(signatures, mask, layer_ids, neuron_ids)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_z = []
    
    with torch.no_grad():
        for batch in test_loader:
            signatures = batch['signatures'].to(device)
            mask = batch['mask'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            neuron_ids = batch['neuron_ids'].to(device)
            labels = batch['label'].to(device)
            
            logits, z, _ = model(signatures, mask, layer_ids, neuron_ids)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_z.append(z.cpu())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_z = torch.cat(all_z, dim=0)
    
    test_acc = (all_preds == all_labels).mean()
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Bottleneck dim: {config.bottleneck_dim}")
    
    # Analyze latent space
    z_std = all_z.std(dim=0)
    z_usage = (z_std > 0.1).sum().item()  # Dimensions with meaningful variance
    
    print(f"Latent dimensions with std > 0.1: {z_usage}/{config.bottleneck_dim}")
    
    return {
        'bottleneck_dim': config.bottleneck_dim,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'n_params': n_params,
        'z_usage': z_usage,
        'z_std': z_std.tolist(),
    }


def run_bottleneck_ablation():
    """Test different bottleneck sizes."""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp12_bottleneck_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 12B: Learned Bottleneck Ablation")
    print("="*60)
    
    bottleneck_sizes = [8, 16, 32, 64, 128, 256]
    results = []
    
    for dim in bottleneck_sizes:
        config = BottleneckConfig(
            name=f"bottleneck_{dim}",
            bottleneck_dim=dim,
            epochs=50,
        )
        
        result = train_bottleneck(config)
        results.append(result)
        
        # Save intermediate results
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("BOTTLENECK ABLATION RESULTS")
    print("="*60)
    print(f"{'Dim':>6} | {'Val Acc':>10} | {'Test Acc':>10} | {'Z Usage':>10}")
    print("-"*50)
    
    for r in results:
        print(f"{r['bottleneck_dim']:6d} | {r['best_val_acc']:10.4f} | {r['test_acc']:10.4f} | {r['z_usage']:10d}")
    
    # Find optimal
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\n*** BEST: dim={best['bottleneck_dim']} with {best['test_acc']:.4f} test accuracy ***")
    
    # Find minimum dim that achieves >95% of best
    threshold = best['test_acc'] * 0.95
    for r in results:
        if r['test_acc'] >= threshold:
            print(f"*** MINIMUM EFFICIENT: dim={r['bottleneck_dim']} achieves {r['test_acc']:.4f} (>95% of best) ***")
            break
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="Run full ablation")
    parser.add_argument("--dim", type=int, default=16, help="Single bottleneck dim to test")
    
    args = parser.parse_args()
    
    if args.ablation:
        run_bottleneck_ablation()
    else:
        config = BottleneckConfig(
            bottleneck_dim=args.dim,
            epochs=50,
        )
        train_bottleneck(config)
