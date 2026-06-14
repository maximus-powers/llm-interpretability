"""
Generate a validated dataset of subject networks.

Each network is trained and validated to ACTUALLY exhibit its labeled behavior
with a strong margin before being included in the dataset.

Usage:
    python -m hypernet.generate_validated_dataset --n-samples 10000 --output validated_dataset.pt
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import multiprocessing as mp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n_samples: int = 10000
    samples_per_pattern: Optional[int] = None  # If None, distribute evenly
    
    # Network architecture (matches hypernet expectations)
    num_layers: int = 5
    neurons_per_layer: int = 8
    input_dim: int = 5  # 5 numbers in sequence
    
    # Training
    epochs: int = 200
    lr: float = 0.01
    batch_size: int = 32
    train_samples: int = 500
    val_samples: int = 100
    
    # Validation thresholds
    min_accuracy: float = 0.85  # Must get 85%+ on test set
    min_margin: float = 0.05    # Average margin must be > 0.05
    max_attempts: int = 5       # Retry training this many times
    
    # Signature
    n_probe_samples: int = 200  # Samples for signature extraction
    
    # Output
    output_path: str = "validated_dataset.pt"


# =============================================================================
# Patterns - Numeric sequences (5 numbers, values 0-9)
# =============================================================================

PATTERNS = [
    'sorted_ascending',
    'sorted_descending', 
    'palindrome',
    'alternating',
    'first_last_match',
    'mountain_pattern',
    'increasing_pairs',
    'decreasing_pairs',
    'no_repeats',
    'has_majority',
    'all_same',
]


def check_pattern(seq: List[int], pattern: str) -> bool:
    """Check if a sequence matches a pattern."""
    if pattern == 'sorted_ascending':
        return all(seq[i] < seq[i+1] for i in range(len(seq)-1))
    
    elif pattern == 'sorted_descending':
        return all(seq[i] > seq[i+1] for i in range(len(seq)-1))
    
    elif pattern == 'palindrome':
        return seq == seq[::-1]
    
    elif pattern == 'alternating':
        if len(set(seq)) != 2:
            return False
        return all(seq[i] != seq[i+1] for i in range(len(seq)-1))
    
    elif pattern == 'first_last_match':
        return seq[0] == seq[-1]
    
    elif pattern == 'mountain_pattern':
        # Increases then decreases, peak in middle
        n = len(seq)
        peak_idx = n // 2
        increasing = all(seq[i] < seq[i+1] for i in range(peak_idx))
        decreasing = all(seq[i] > seq[i+1] for i in range(peak_idx, n-1))
        return increasing and decreasing
    
    elif pattern == 'increasing_pairs':
        # Each adjacent pair is increasing
        return all(seq[i] < seq[i+1] for i in range(len(seq)-1))
    
    elif pattern == 'decreasing_pairs':
        return all(seq[i] > seq[i+1] for i in range(len(seq)-1))
    
    elif pattern == 'no_repeats':
        return len(set(seq)) == len(seq)
    
    elif pattern == 'has_majority':
        from collections import Counter
        counts = Counter(seq)
        return counts.most_common(1)[0][1] > len(seq) // 2
    
    elif pattern == 'all_same':
        return len(set(seq)) == 1
    
    return False


def generate_positive_sample(pattern: str, seq_len: int = 5) -> List[int]:
    """Generate a sequence that matches the pattern."""
    max_attempts = 1000
    
    for _ in range(max_attempts):
        if pattern == 'sorted_ascending':
            # Pick 5 distinct values and sort them
            vals = random.sample(range(10), seq_len)
            return sorted(vals)
        
        elif pattern == 'sorted_descending':
            vals = random.sample(range(10), seq_len)
            return sorted(vals, reverse=True)
        
        elif pattern == 'palindrome':
            half = [random.randint(0, 9) for _ in range((seq_len + 1) // 2)]
            if seq_len % 2 == 0:
                return half + half[::-1]
            else:
                return half + half[-2::-1]
        
        elif pattern == 'alternating':
            a, b = random.sample(range(10), 2)
            return [a if i % 2 == 0 else b for i in range(seq_len)]
        
        elif pattern == 'first_last_match':
            seq = [random.randint(0, 9) for _ in range(seq_len)]
            seq[-1] = seq[0]
            return seq
        
        elif pattern == 'mountain_pattern':
            peak_idx = seq_len // 2
            peak_val = random.randint(5, 9)
            seq = []
            for i in range(seq_len):
                if i < peak_idx:
                    seq.append(random.randint(0, peak_val - peak_idx + i))
                elif i == peak_idx:
                    seq.append(peak_val)
                else:
                    seq.append(random.randint(0, peak_val - (i - peak_idx)))
            # Ensure strictly increasing then decreasing
            for i in range(peak_idx):
                seq[i] = i
            for i in range(peak_idx + 1, seq_len):
                seq[i] = peak_val - (i - peak_idx)
            return seq
        
        elif pattern == 'increasing_pairs':
            vals = random.sample(range(10), seq_len)
            return sorted(vals)
        
        elif pattern == 'decreasing_pairs':
            vals = random.sample(range(10), seq_len)
            return sorted(vals, reverse=True)
        
        elif pattern == 'no_repeats':
            return random.sample(range(10), seq_len)
        
        elif pattern == 'has_majority':
            majority_val = random.randint(0, 9)
            majority_count = seq_len // 2 + 1
            seq = [majority_val] * majority_count
            seq += [random.randint(0, 9) for _ in range(seq_len - majority_count)]
            random.shuffle(seq)
            return seq
        
        elif pattern == 'all_same':
            val = random.randint(0, 9)
            return [val] * seq_len
        
        # Fallback: random sequence, check if it matches
        seq = [random.randint(0, 9) for _ in range(seq_len)]
        if check_pattern(seq, pattern):
            return seq
    
    raise ValueError(f"Could not generate positive sample for {pattern}")


def generate_negative_sample(pattern: str, seq_len: int = 5) -> List[int]:
    """Generate a sequence that does NOT match the pattern."""
    max_attempts = 1000
    
    for _ in range(max_attempts):
        seq = [random.randint(0, 9) for _ in range(seq_len)]
        if not check_pattern(seq, pattern):
            return seq
    
    raise ValueError(f"Could not generate negative sample for {pattern}")


def generate_training_data(
    pattern: str,
    n_train: int = 500,
    n_val: int = 100,
    seq_len: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate balanced training data for a pattern."""
    
    def make_dataset(n_samples):
        X, y = [], []
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos
        
        for _ in range(n_pos):
            seq = generate_positive_sample(pattern, seq_len)
            X.append(seq)
            y.append(1)
        
        for _ in range(n_neg):
            seq = generate_negative_sample(pattern, seq_len)
            X.append(seq)
            y.append(0)
        
        # Shuffle
        combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    X_train, y_train = make_dataset(n_train)
    X_val, y_val = make_dataset(n_val)
    
    return X_train, y_train, X_val, y_val


# =============================================================================
# Subject Network
# =============================================================================

class SubjectNetwork(nn.Module):
    """Simple MLP for pattern classification."""
    
    def __init__(
        self,
        input_dim: int = 5,
        num_layers: int = 5,
        hidden_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = 1 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize with small weights for stable training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
    
    def get_activations(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get activations at each layer."""
        activations = {}
        current = x
        layer_idx = 0
        
        for module in self.network:
            current = module(current)
            if isinstance(module, nn.Linear) and layer_idx < self.num_layers - 1:
                activations[layer_idx] = current.clone()
                layer_idx += 1
        
        return activations
    
    def to_flat(self) -> torch.Tensor:
        """Flatten all parameters to a single tensor."""
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)
    
    def from_flat(self, flat: torch.Tensor):
        """Load parameters from flattened tensor."""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat[idx:idx + numel].view(p.shape)
            idx += numel
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Signature Extraction
# =============================================================================

def extract_signature(
    model: SubjectNetwork,
    n_samples: int = 200,
    seq_len: int = 5,
) -> torch.Tensor:
    """
    Extract behavioral signature from a trained model.
    
    For each neuron, computes:
    - mean, std (2)
    - fourier coefficients (5)
    - input correlations (input_dim)
    - pre-activation mean, std (2)
    
    Total: (2 + 5 + input_dim + 2) * num_neurons
    """
    model.eval()
    
    # Generate probe inputs (random sequences)
    probe_inputs = torch.rand(n_samples, seq_len) * 9  # Values 0-9
    
    with torch.no_grad():
        activations = model.get_activations(probe_inputs)
    
    features = []
    
    for layer_idx in sorted(activations.keys()):
        acts = activations[layer_idx]  # [n_samples, hidden_dim]
        
        for neuron_idx in range(acts.shape[1]):
            neuron_acts = acts[:, neuron_idx]  # [n_samples]
            
            # Basic stats
            features.append(neuron_acts.mean().item())
            features.append(neuron_acts.std().item())
            
            # Fourier (top 5 coefficients)
            fft = torch.fft.fft(neuron_acts)
            fft_mag = torch.abs(fft)[:n_samples // 2]
            top_k = min(5, len(fft_mag))
            fourier_features = fft_mag[:top_k].tolist()
            fourier_features += [0.0] * (5 - top_k)
            features.extend(fourier_features)
            
            # Input correlations
            for input_idx in range(seq_len):
                corr = torch.corrcoef(
                    torch.stack([neuron_acts, probe_inputs[:, input_idx]])
                )[0, 1].item()
                features.append(corr if not torch.isnan(torch.tensor(corr)) else 0.0)
            
            # Pre-activation stats (approximate - use activation as proxy)
            features.append(neuron_acts.mean().item())
            features.append(neuron_acts.std().item())
    
    return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# Training & Validation
# =============================================================================

def train_subject_network(
    pattern: str,
    config: DatasetConfig,
) -> Tuple[Optional[SubjectNetwork], Optional[Dict]]:
    """
    Train a subject network and validate it exhibits the correct behavior.
    
    Returns (model, metrics) if successful, (None, None) if failed.
    """
    # Generate training data
    X_train, y_train, X_val, y_val = generate_training_data(
        pattern,
        n_train=config.train_samples,
        n_val=config.val_samples,
        seq_len=config.input_dim,
    )
    
    # Create model
    model = SubjectNetwork(
        input_dim=config.input_dim,
        num_layers=config.num_layers,
        hidden_dim=config.neurons_per_layer,
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20
    )
    criterion = nn.BCEWithLogitsLoss()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    max_patience = 30
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final validation with behavior-specific test
    metrics = validate_behavior(model, pattern, config.input_dim)
    
    if metrics['accuracy'] >= config.min_accuracy and metrics['margin'] >= config.min_margin:
        return model, metrics
    
    return None, None


def validate_behavior(
    model: SubjectNetwork,
    pattern: str,
    seq_len: int = 5,
    n_test: int = 100,
) -> Dict:
    """
    Validate that model exhibits the correct behavior.
    
    Returns metrics including accuracy and margin.
    """
    model.eval()
    
    # Generate test samples
    positives = [generate_positive_sample(pattern, seq_len) for _ in range(n_test)]
    negatives = [generate_negative_sample(pattern, seq_len) for _ in range(n_test)]
    
    X_pos = torch.tensor(positives, dtype=torch.float32)
    X_neg = torch.tensor(negatives, dtype=torch.float32)
    
    with torch.no_grad():
        pos_out = torch.sigmoid(model(X_pos))
        neg_out = torch.sigmoid(model(X_neg))
    
    # Accuracy: positives should be > 0.5, negatives should be < 0.5
    pos_correct = (pos_out > 0.5).float().mean().item()
    neg_correct = (neg_out < 0.5).float().mean().item()
    accuracy = (pos_correct + neg_correct) / 2
    
    # Margin: average difference between positive and negative outputs
    margin = (pos_out.mean() - neg_out.mean()).item()
    
    # Output variance (check for collapse)
    all_out = torch.cat([pos_out, neg_out])
    output_std = all_out.std().item()
    
    return {
        'accuracy': accuracy,
        'margin': margin,
        'pos_mean': pos_out.mean().item(),
        'neg_mean': neg_out.mean().item(),
        'output_std': output_std,
        'collapsed': output_std < 0.05,
    }


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_single_sample(
    args: Tuple[str, DatasetConfig, int]
) -> Optional[Dict]:
    """Generate a single validated sample (for multiprocessing)."""
    pattern, config, seed = args
    
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    for attempt in range(config.max_attempts):
        model, metrics = train_subject_network(pattern, config)
        
        if model is not None:
            # Extract signature
            signature = extract_signature(
                model,
                n_samples=config.n_probe_samples,
                seq_len=config.input_dim,
            )
            
            return {
                'pattern': pattern,
                'weights': model.to_flat(),
                'signature': signature,
                'metrics': metrics,
            }
    
    return None


def generate_dataset(config: DatasetConfig) -> Dict:
    """Generate the full validated dataset."""
    
    # Determine samples per pattern
    if config.samples_per_pattern is not None:
        samples_per_pattern = {p: config.samples_per_pattern for p in PATTERNS}
    else:
        base_count = config.n_samples // len(PATTERNS)
        remainder = config.n_samples % len(PATTERNS)
        samples_per_pattern = {p: base_count for p in PATTERNS}
        for i, p in enumerate(PATTERNS[:remainder]):
            samples_per_pattern[p] += 1
    
    logger.info(f"Generating {config.n_samples} samples across {len(PATTERNS)} patterns")
    logger.info(f"Samples per pattern: {samples_per_pattern}")
    
    all_samples = []
    failed_counts = {p: 0 for p in PATTERNS}
    
    # Generate samples for each pattern
    for pattern in PATTERNS:
        n_samples = samples_per_pattern[pattern]
        logger.info(f"\nGenerating {n_samples} samples for {pattern}...")
        
        pattern_samples = []
        attempts = 0
        max_total_attempts = n_samples * config.max_attempts * 2
        
        pbar = tqdm(total=n_samples, desc=pattern)
        
        while len(pattern_samples) < n_samples and attempts < max_total_attempts:
            seed = random.randint(0, 2**31)
            result = generate_single_sample((pattern, config, seed))
            
            if result is not None:
                pattern_samples.append(result)
                pbar.update(1)
            else:
                failed_counts[pattern] += 1
            
            attempts += 1
        
        pbar.close()
        
        if len(pattern_samples) < n_samples:
            logger.warning(f"Only generated {len(pattern_samples)}/{n_samples} for {pattern}")
        
        all_samples.extend(pattern_samples)
    
    # Compile dataset
    logger.info(f"\nCompiling dataset with {len(all_samples)} samples...")
    
    weights = torch.stack([s['weights'] for s in all_samples])
    signatures = torch.stack([s['signature'] for s in all_samples])
    patterns = [s['pattern'] for s in all_samples]
    pattern_to_idx = {p: i for i, p in enumerate(PATTERNS)}
    labels = torch.tensor([pattern_to_idx[p] for p in patterns])
    
    # Compute statistics
    accuracies = [s['metrics']['accuracy'] for s in all_samples]
    margins = [s['metrics']['margin'] for s in all_samples]
    
    dataset = {
        'weights': weights,
        'signatures': signatures,
        'labels': labels,
        'patterns': PATTERNS,
        'pattern_to_idx': pattern_to_idx,
        'config': asdict(config),
        'stats': {
            'n_samples': len(all_samples),
            'samples_per_pattern': {p: sum(1 for s in all_samples if s['pattern'] == p) for p in PATTERNS},
            'failed_per_pattern': failed_counts,
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'mean_margin': sum(margins) / len(margins),
            'min_accuracy': min(accuracies),
            'min_margin': min(margins),
        },
    }
    
    return dataset


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate validated dataset of subject networks"
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=10000,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="validated_dataset.pt",
        help="Output file path"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.85,
        help="Minimum accuracy threshold (default: 0.85)"
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help="Minimum margin threshold (default: 0.05)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs per network (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    config = DatasetConfig(
        n_samples=args.n_samples,
        output_path=args.output,
        min_accuracy=args.min_accuracy,
        min_margin=args.min_margin,
        epochs=args.epochs,
    )
    
    logger.info("=" * 60)
    logger.info("Validated Dataset Generation")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")
    
    # Generate dataset
    dataset = generate_dataset(config)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {dataset['stats']['n_samples']}")
    logger.info(f"Mean accuracy: {dataset['stats']['mean_accuracy']:.4f}")
    logger.info(f"Mean margin: {dataset['stats']['mean_margin']:.4f}")
    logger.info(f"Min accuracy: {dataset['stats']['min_accuracy']:.4f}")
    logger.info(f"Min margin: {dataset['stats']['min_margin']:.4f}")
    logger.info(f"Saved to: {output_path}")
    
    logger.info("\nSamples per pattern:")
    for pattern, count in dataset['stats']['samples_per_pattern'].items():
        failed = dataset['stats']['failed_per_pattern'][pattern]
        logger.info(f"  {pattern}: {count} (failed: {failed})")


if __name__ == "__main__":
    main()
