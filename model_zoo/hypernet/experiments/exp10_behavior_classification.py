"""
Experiment 10: Behavioral Classification

GOAL: Train Z_behavior to accurately classify what behavior/pattern a network
was trained to detect.

This is the RIGHT metric for interpretability:
- We don't care if weights match exactly
- We care if Z_behavior captures the FUNCTION of the network

DATASET:
- 10,000 subject models
- 14 distinct patterns: contains_abc, palindrome, alternating, sorted_descending, etc.
- Each model was trained to detect one pattern

APPROACH:
1. Encode each model's neurons using PureBehavioralEncoder (9 behavioral features only)
2. Aggregate neuron-level Z_behavior into model-level representation
3. Train classifier: model_representation → pattern label

METRICS:
- Classification accuracy (14-way)
- Per-pattern precision/recall
- Confusion matrix analysis
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import load_dataset as hf_load_dataset

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.components.encoders import ResidualMLPBlock

# Behavioral features only - NO input_correlations
BEHAVIORAL_INDICES = [0, 1, 2, 3, 4, 5, 6, 15, 16]  # 9 features

# Pattern labels
PATTERN_LABELS = [
    'contains_abc', 'palindrome', 'alternating', 'sorted_descending',
    'sorted_ascending', 'ends_with', 'decreasing_pairs', 'mountain_pattern',
    'increasing_pairs', 'first_last_match', 'starts_with', 'no_repeats',
    'has_majority', 'vowel_consonant'
]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_LABELS)}


@dataclass
class ClassificationConfig:
    name: str = "behavior_classification"
    
    # Encoder
    neuron_latent_dim: int = 64
    model_latent_dim: int = 128
    hidden_dim: int = 256
    encoder_layers: int = 3
    dropout: float = 0.1
    
    # Aggregation
    aggregation: str = "attention"  # "mean", "max", "attention"
    
    # Training
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    
    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class BehaviorClassificationDataset(Dataset):
    """Dataset for behavior pattern classification."""
    
    def __init__(
        self,
        hf_dataset,
        max_neurons: int = 64,
        signature_dim: int = 17,
    ):
        self.hf_dataset = hf_dataset
        self.max_neurons = max_neurons
        self.signature_dim = signature_dim
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.hf_dataset[idx]
        
        # Get pattern label
        pattern = sample['classification_completion']
        label = PATTERN_TO_IDX.get(pattern, -1)
        
        # Parse signature
        sig_data = json.loads(sample['improved_signature'])
        neuron_activations = sig_data['neuron_activations']
        
        # Extract behavioral signatures for each neuron
        signatures = []
        layer_indices = sorted([int(k) for k in neuron_activations.keys()])
        
        for layer_idx in layer_indices:
            layer_data = neuron_activations.get(str(layer_idx), {})
            neuron_profiles = layer_data.get('neuron_profiles', {})
            
            for neuron_idx in sorted([int(k) for k in neuron_profiles.keys()]):
                profile = neuron_profiles[str(neuron_idx)]
                sig = self._extract_signature(profile)
                signatures.append(sig)
        
        # Pad/truncate to max_neurons
        num_real = len(signatures)
        while len(signatures) < self.max_neurons:
            signatures.append(torch.zeros(self.signature_dim))
        signatures = signatures[:self.max_neurons]
        
        signatures = torch.stack(signatures)
        mask = torch.zeros(self.max_neurons)
        mask[:min(num_real, self.max_neurons)] = 1.0
        
        return {
            'signatures': signatures,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long),
            'num_neurons': min(num_real, self.max_neurons),
        }
    
    def _extract_signature(self, profile: Dict) -> torch.Tensor:
        sig = torch.zeros(self.signature_dim)
        
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


class NeuronEncoder(nn.Module):
    """Encodes individual neurons from behavioral signatures."""
    
    def __init__(
        self,
        input_dim: int = 9,  # behavioral features only
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, signatures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signatures: [batch, neurons, 17] full signatures
        Returns:
            z: [batch, neurons, latent_dim]
        """
        # Extract behavioral features only
        if signatures.shape[-1] == 17:
            behavioral = signatures[..., BEHAVIORAL_INDICES]
        else:
            behavioral = signatures
        
        x = self.input_proj(behavioral)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        return x


class AttentionAggregator(nn.Module):
    """Aggregates neuron representations into model representation using attention."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        # Learnable "model" query token
        self.model_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
    
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, neurons, input_dim]
            mask: [batch, neurons]
        Returns:
            model_repr: [batch, output_dim]
        """
        batch_size = z.shape[0]
        
        # Expand model query for batch
        query = self.model_query.expand(batch_size, -1, -1)  # [batch, 1, dim]
        
        # Project neurons
        keys = self.key(z)    # [batch, neurons, dim]
        values = self.value(z)  # [batch, neurons, dim]
        
        # Attention scores
        scores = torch.bmm(query, keys.transpose(1, 2))  # [batch, 1, neurons]
        scores = scores / (keys.shape[-1] ** 0.5)
        
        # Mask invalid neurons
        mask_expanded = mask.unsqueeze(1)  # [batch, 1, neurons]
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax and aggregate
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(mask_expanded == 0, 0)  # Zero out for safety
        
        aggregated = torch.bmm(attn, values)  # [batch, 1, dim]
        aggregated = aggregated.squeeze(1)  # [batch, dim]
        
        # Output projection
        output = self.output_proj(aggregated)
        output = self.norm(output)
        
        return output


class BehaviorClassifier(nn.Module):
    """
    Full model: signatures → neuron encodings → model representation → pattern class
    """
    
    def __init__(self, config: ClassificationConfig):
        super().__init__()
        self.config = config
        
        # Neuron encoder
        self.neuron_encoder = NeuronEncoder(
            input_dim=len(BEHAVIORAL_INDICES),
            latent_dim=config.neuron_latent_dim,
            hidden_dim=config.hidden_dim // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        
        # Aggregator
        if config.aggregation == "attention":
            self.aggregator = AttentionAggregator(
                config.neuron_latent_dim,
                config.model_latent_dim,
            )
        else:
            self.aggregator = None
            self.agg_proj = nn.Linear(config.neuron_latent_dim, config.model_latent_dim)
            self.agg_norm = nn.LayerNorm(config.model_latent_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.model_latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, len(PATTERN_LABELS)),
        )
    
    def forward(
        self,
        signatures: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [batch, num_classes]
            model_repr: [batch, model_latent_dim] for analysis
        """
        # Encode neurons
        z_neurons = self.neuron_encoder(signatures)  # [batch, neurons, neuron_latent]
        
        # Aggregate to model level
        if self.aggregator is not None:
            model_repr = self.aggregator(z_neurons, mask)
        else:
            # Simple mean/max pooling
            mask_expanded = mask.unsqueeze(-1)
            if self.config.aggregation == "mean":
                model_repr = (z_neurons * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:  # max
                z_neurons_masked = z_neurons.masked_fill(mask_expanded == 0, float('-inf'))
                model_repr = z_neurons_masked.max(dim=1)[0]
            
            model_repr = self.agg_proj(model_repr)
            model_repr = self.agg_norm(model_repr)
        
        # Classify
        logits = self.classifier(model_repr)
        
        return logits, model_repr
    
    def get_model_representation(
        self,
        signatures: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get model-level representation for analysis."""
        with torch.no_grad():
            _, model_repr = self.forward(signatures, mask)
        return model_repr


def create_dataloaders(
    config: ClassificationConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    
    print("Loading dataset...")
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    
    # Split
    total = len(hf_ds)
    train_size = int(total * config.train_ratio)
    val_size = int(total * config.val_ratio)
    test_size = total - train_size - val_size
    
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_ds = BehaviorClassificationDataset(hf_ds.select(train_indices))
    val_ds = BehaviorClassificationDataset(hf_ds.select(val_indices))
    test_ds = BehaviorClassificationDataset(hf_ds.select(test_indices))
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def train_classifier(
    config: ClassificationConfig,
    device: str = "auto",
) -> Tuple[BehaviorClassifier, Dict]:
    """Train the behavior classifier."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp10_{config.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 10: Behavior Classification")
    print("="*60)
    print(f"Device: {device}")
    print(f"Aggregation: {config.aggregation}")
    print(f"Run dir: {run_dir}")
    
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    print(f"\nTensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Model
    model = BehaviorClassifier(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            signatures = batch['signatures'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(signatures, mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
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
                labels = batch['label'].to(device)
                
                logits, _ = model(signatures, mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'val_acc': val_acc,
            }, run_dir / "best_model.pt")
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    
    # Test evaluation
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            signatures = batch['signatures'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(signatures, mask)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Random baseline: {1/len(PATTERN_LABELS):.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=PATTERN_LABELS, digits=3))
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'random_baseline': 1/len(PATTERN_LABELS),
        'history': history,
    }
    
    with open(run_dir / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != 'history'}, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    np.save(run_dir / "confusion_matrix.npy", cm)
    
    writer.close()
    
    print(f"\nResults saved to: {run_dir}")
    
    return model, results


def compare_aggregations(device: str = "auto"):
    """Compare different aggregation methods."""
    
    results = {}
    
    for agg in ["mean", "max", "attention"]:
        print(f"\n{'='*60}")
        print(f"Testing aggregation: {agg}")
        print("="*60)
        
        config = ClassificationConfig(
            name=f"agg_{agg}",
            aggregation=agg,
            epochs=30,
        )
        
        model, result = train_classifier(config, device)
        results[agg] = result
    
    print("\n" + "="*60)
    print("AGGREGATION COMPARISON")
    print("="*60)
    print(f"{'Method':<15} {'Val Acc':>10} {'Test Acc':>10}")
    print("-"*40)
    for agg, result in results.items():
        print(f"{agg:<15} {result['best_val_acc']:>10.4f} {result['test_acc']:>10.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare aggregation methods")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_aggregations(args.device)
    else:
        config = ClassificationConfig(
            name="attention_full",
            aggregation="attention",
            epochs=args.epochs,
        )
        model, results = train_classifier(config, args.device)
