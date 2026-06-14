"""
Experiment 11: Improved Behavior Classification

Building on Exp10 (41% accuracy), trying several improvements:

1. ARCHITECTURE: Deeper/wider models, transformer encoder
2. FEATURES: Compare behavioral-only (9) vs full (17) features  
3. REGULARIZATION: Dropout, label smoothing, mixup
4. AGGREGATION: Hierarchical (per-layer then cross-layer)

Goal: Maximize classification accuracy on the 14 behavior patterns.
"""

import sys
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import load_dataset as hf_load_dataset

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Feature indices
BEHAVIORAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 15, 16]  # 9 features (no input_corr)
ALL_INDICES = list(range(17))  # All 17 features

PATTERN_LABELS = [
    'contains_abc', 'palindrome', 'alternating', 'sorted_descending',
    'sorted_ascending', 'ends_with', 'decreasing_pairs', 'mountain_pattern',
    'increasing_pairs', 'first_last_match', 'starts_with', 'no_repeats',
    'has_majority', 'vowel_consonant'
]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_LABELS)}
NUM_CLASSES = len(PATTERN_LABELS)


@dataclass 
class ImprovedConfig:
    name: str = "improved"
    
    # Features
    use_all_features: bool = False  # If True, use all 17; if False, use 9 behavioral
    
    # Architecture
    model_type: str = "transformer"  # "mlp", "transformer"
    neuron_latent_dim: int = 128
    model_latent_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.2
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    
    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_neurons: int = 64


class BehaviorDataset(Dataset):
    """Dataset with configurable features."""
    
    def __init__(self, hf_dataset, config: ImprovedConfig):
        self.hf_dataset = hf_dataset
        self.config = config
        self.feature_indices = ALL_INDICES if config.use_all_features else BEHAVIORAL_INDICES
        self.feature_dim = len(self.feature_indices)
    
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
                layer_ids.append(layer_idx // 2)  # Normalize layer idx (0, 2, 4 -> 0, 1, 2)
                neuron_ids.append(neuron_idx)
        
        num_real = len(signatures)
        
        # Pad
        while len(signatures) < self.config.max_neurons:
            signatures.append(torch.zeros(17))
            layer_ids.append(0)
            neuron_ids.append(0)
        
        signatures = signatures[:self.config.max_neurons]
        layer_ids = layer_ids[:self.config.max_neurons]
        neuron_ids = neuron_ids[:self.config.max_neurons]
        
        signatures = torch.stack(signatures)
        
        # Extract selected features
        signatures = signatures[..., self.feature_indices]
        
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


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for neuron sequences."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_neurons: int = 64,
        max_layers: int = 10,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable position embeddings
        self.layer_embedding = nn.Embedding(max_layers, hidden_dim // 2)
        self.neuron_embedding = nn.Embedding(max_neurons, hidden_dim // 2)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        mask: torch.Tensor,
        layer_ids: torch.Tensor,
        neuron_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            signatures: [batch, neurons, input_dim]
            mask: [batch, neurons]
            layer_ids: [batch, neurons]
            neuron_ids: [batch, neurons]
        Returns:
            model_repr: [batch, output_dim]
        """
        batch_size = signatures.shape[0]
        
        # Input projection
        x = self.input_proj(signatures)
        
        # Add position embeddings
        layer_emb = self.layer_embedding(layer_ids.clamp(0, 9))
        neuron_emb = self.neuron_embedding(neuron_ids.clamp(0, 63))
        pos_emb = torch.cat([layer_emb, neuron_emb], dim=-1)
        x = x + pos_emb
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Update mask for CLS token (always valid)
        cls_mask = torch.ones(batch_size, 1, device=mask.device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        
        # Create attention mask (True = ignore)
        attn_mask = (extended_mask == 0)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Take CLS token output
        cls_output = x[:, 0]
        
        # Project to output dim
        model_repr = self.output_proj(cls_output)
        
        return model_repr


class MLPEncoder(nn.Module):
    """Simple MLP encoder with attention aggregation (baseline)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.encoder = nn.Sequential(*layers)
        
        # Attention aggregation
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = signatures.shape[0]
        
        # Encode each neuron
        x = self.encoder(signatures)  # [batch, neurons, hidden]
        
        # Attention aggregation
        query = self.attn_query.expand(batch_size, -1, -1)
        keys = self.attn_proj(x)
        
        scores = torch.bmm(query, keys.transpose(1, 2)) / (keys.shape[-1] ** 0.5)
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        aggregated = torch.bmm(attn, x).squeeze(1)
        
        return self.output_proj(aggregated)


class ImprovedClassifier(nn.Module):
    """Improved behavior classifier."""
    
    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config
        
        input_dim = 17 if config.use_all_features else 9
        
        if config.model_type == "transformer":
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.model_latent_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_neurons=config.max_neurons,
            )
        else:
            self.encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.model_latent_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.model_latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, NUM_CLASSES),
        )
    
    def forward(self, signatures, mask, layer_ids=None, neuron_ids=None):
        if self.config.model_type == "transformer":
            model_repr = self.encoder(signatures, mask, layer_ids, neuron_ids)
        else:
            model_repr = self.encoder(signatures, mask)
        
        logits = self.classifier(model_repr)
        return logits, model_repr


def create_dataloaders(config: ImprovedConfig):
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
    
    train_ds = BehaviorDataset(hf_ds.select(train_indices), config)
    val_ds = BehaviorDataset(hf_ds.select(val_indices), config)
    test_ds = BehaviorDataset(hf_ds.select(test_indices), config)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Feature dim: {train_ds.feature_dim}")
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def train_model(config: ImprovedConfig, device: str = "auto") -> Tuple[nn.Module, Dict]:
    """Train and evaluate model."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp11_{config.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    feature_type = "all_17" if config.use_all_features else "behavioral_9"
    
    print("="*60)
    print(f"Training: {config.name}")
    print("="*60)
    print(f"Model: {config.model_type}, Features: {feature_type}")
    print(f"Hidden: {config.hidden_dim}, Layers: {config.num_layers}")
    print(f"Device: {device}")
    
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    model = ImprovedClassifier(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.epochs
    pct_start = min(config.warmup_epochs / config.epochs, 0.3)  # Cap at 30%
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Train
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
            
            logits, _ = model(signatures, mask, layer_ids, neuron_ids)
            loss = criterion(logits, labels)
            
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
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                signatures = batch['signatures'].to(device)
                mask = batch['mask'].to(device)
                layer_ids = batch['layer_ids'].to(device)
                neuron_ids = batch['neuron_ids'].to(device)
                labels = batch['label'].to(device)
                
                logits, _ = model(signatures, mask, layer_ids, neuron_ids)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'val_acc': val_acc,
            }, run_dir / "best_model.pt")
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
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            signatures = batch['signatures'].to(device)
            mask = batch['mask'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            neuron_ids = batch['neuron_ids'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(signatures, mask, layer_ids, neuron_ids)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=PATTERN_LABELS, digits=3))
    
    writer.close()
    
    results = {
        'name': config.name,
        'model_type': config.model_type,
        'use_all_features': config.use_all_features,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'n_params': n_params,
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return model, results


def run_ablation():
    """Run ablation study comparing different configurations."""
    
    configs = [
        # Baseline: MLP with behavioral features
        ImprovedConfig(
            name="mlp_behavioral",
            model_type="mlp",
            use_all_features=False,
            hidden_dim=256,
            num_layers=3,
            epochs=50,
        ),
        # MLP with all features
        ImprovedConfig(
            name="mlp_all_features",
            model_type="mlp",
            use_all_features=True,
            hidden_dim=256,
            num_layers=3,
            epochs=50,
        ),
        # Transformer with behavioral features
        ImprovedConfig(
            name="transformer_behavioral",
            model_type="transformer",
            use_all_features=False,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            epochs=80,
        ),
        # Transformer with all features
        ImprovedConfig(
            name="transformer_all_features",
            model_type="transformer",
            use_all_features=True,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            epochs=80,
        ),
        # Large transformer with all features
        ImprovedConfig(
            name="transformer_large",
            model_type="transformer",
            use_all_features=True,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            model_latent_dim=512,
            epochs=100,
            dropout=0.3,
        ),
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'#'*60}")
        print(f"# Running: {config.name}")
        print(f"{'#'*60}")
        
        try:
            _, result = train_model(config)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            results.append({'name': config.name, 'error': str(e)})
    
    # Summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Config':<30} {'Model':<12} {'Features':<10} {'Val Acc':>10} {'Test Acc':>10}")
    print("-"*80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<30} ERROR: {r['error'][:40]}")
        else:
            feat = "all_17" if r['use_all_features'] else "behav_9"
            print(f"{r['name']:<30} {r['model_type']:<12} {feat:<10} {r['best_val_acc']:>10.4f} {r['test_acc']:>10.4f}")
    
    # Find best
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['test_acc'])
        print(f"\n*** BEST: {best['name']} with {best['test_acc']:.4f} test accuracy ***")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation()
    elif args.quick:
        config = ImprovedConfig(
            name="quick_test",
            model_type="transformer",
            use_all_features=True,
            hidden_dim=256,
            num_layers=3,
            epochs=10,
        )
        train_model(config)
    else:
        # Default: best config
        config = ImprovedConfig(
            name="transformer_best",
            model_type="transformer",
            use_all_features=True,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            model_latent_dim=512,
            epochs=100,
            dropout=0.3,
        )
        train_model(config)
