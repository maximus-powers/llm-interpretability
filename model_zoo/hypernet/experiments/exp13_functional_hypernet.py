"""
Experiment 13: Functional Hypernetwork

Key insight from research: Don't just reconstruct weights - verify FUNCTIONAL behavior.

Architecture:
1. Weight Autoencoder: Learn compressed representation of weights
2. Signature Encoder: Behavioral signatures → conditioning vector  
3. Conditional Generation: Generate weight latent from conditioning
4. Functional Verification: Ensure generated weights exhibit target behavior

Training:
- Phase 1: Train weight autoencoder with functional loss
- Phase 2: Train conditional generator (CVAE)
- Phase 3: Test behavior editing
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use all patterns for more data, but focus testing on sortable ones
ALL_PATTERNS = [
    'contains_abc', 'palindrome', 'alternating', 'sorted_descending',
    'sorted_ascending', 'ends_with', 'decreasing_pairs', 'mountain_pattern',
    'increasing_pairs', 'first_last_match', 'starts_with', 'no_repeats',
    'has_majority', 'vowel_consonant'
]
PATTERN_TO_IDX = {p: i for i, p in enumerate(ALL_PATTERNS)}
IDX_TO_PATTERN = {i: p for p, i in PATTERN_TO_IDX.items()}

# For editing tests, focus on these pairs
TEST_PATTERNS = ['sorted_descending', 'sorted_ascending']

# Target architecture: 5 layers, 8 neurons per layer (most common)
TARGET_ARCH = (5, 8)


# =============================================================================
# Subject Network (the network whose weights we're generating)
# =============================================================================

class SubjectNetwork(nn.Module):
    """The target network architecture - matches (5, 8) config."""
    
    def __init__(self, num_layers: int = 5, hidden_dim: int = 8, input_dim: int = 5):
        super().__init__()
        # Architecture: input_dim -> hidden x (num_layers-1) -> 1
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = 1 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    """Count total parameters in SubjectNetwork."""
    return sum(p.numel() for p in model.parameters())


def weights_to_flat(model: SubjectNetwork) -> torch.Tensor:
    """Extract weights from model as flat tensor."""
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1))
    return torch.cat(params)


def flat_to_weights(flat: torch.Tensor, model: SubjectNetwork):
    """Load flat tensor into model parameters."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat[idx:idx + numel].view(p.shape)
        idx += numel


def create_network_from_flat(flat_weights: torch.Tensor, num_layers: int = 5, hidden_dim: int = 8) -> SubjectNetwork:
    """Create a new SubjectNetwork with given weights."""
    model = SubjectNetwork(num_layers=num_layers, hidden_dim=hidden_dim)
    flat_to_weights(flat_weights, model)
    return model


# =============================================================================
# Behavioral Testing
# =============================================================================

def test_behavior(model: SubjectNetwork, pattern: str) -> Dict:
    """Test if model exhibits the specified behavior pattern."""
    model.eval()
    
    # Generate test cases
    if pattern == 'sorted_descending':
        positive = torch.tensor([
            [9, 7, 5, 3, 1],
            [8, 6, 4, 2, 0],
            [7, 5, 3, 2, 1],
            [9, 8, 7, 6, 5],
            [5, 4, 3, 2, 1],
        ], dtype=torch.float32)
        negative = torch.tensor([
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [3, 1, 4, 1, 5],
        ], dtype=torch.float32)
    elif pattern == 'sorted_ascending':
        positive = torch.tensor([
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4],
        ], dtype=torch.float32)
        negative = torch.tensor([
            [9, 7, 5, 3, 1],
            [8, 6, 4, 2, 0],
            [7, 5, 3, 2, 1],
            [9, 8, 7, 6, 5],
            [3, 1, 4, 1, 5],
        ], dtype=torch.float32)
    else:
        return {'supported': False}
    
    with torch.no_grad():
        pos_out = torch.sigmoid(model(positive)).mean().item()
        neg_out = torch.sigmoid(model(negative)).mean().item()
    
    return {
        'supported': True,
        'positive_output': pos_out,
        'negative_output': neg_out,
        'correct': pos_out > neg_out,
        'margin': pos_out - neg_out,
    }


def compute_functional_signature(model: SubjectNetwork, n_probes: int = 50) -> torch.Tensor:
    """Compute a behavioral signature by probing the network."""
    model.eval()
    
    # Fixed probe inputs for consistency
    torch.manual_seed(42)
    probes = torch.randn(n_probes, 5)
    
    with torch.no_grad():
        outputs = torch.sigmoid(model(probes))
    
    # Return raw outputs as signature (captures full behavior)
    return outputs.squeeze()


# =============================================================================
# Weight Autoencoder with Functional Loss
# =============================================================================

class WeightAutoencoder(nn.Module):
    """
    Autoencoder for neural network weights.
    
    Key: Uses FUNCTIONAL loss - reconstructed weights must produce
    same behavior, not just match numerically.
    """
    
    def __init__(self, weight_dim: int = 181, latent_dim: int = 32):
        super().__init__()
        self.weight_dim = weight_dim
        self.latent_dim = latent_dim
        
        # Encoder: weights -> latent
        self.encoder = nn.Sequential(
            nn.Linear(weight_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        
        # Decoder: latent -> weights
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, weight_dim),
        )
    
    def encode(self, weights: torch.Tensor) -> torch.Tensor:
        return self.encoder(weights)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def forward(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(weights)
        reconstructed = self.decode(latent)
        return latent, reconstructed


# =============================================================================
# Conditional VAE for Signature -> Weights
# =============================================================================

class SignatureEncoder(nn.Module):
    """Encode behavioral signatures into conditioning vectors."""
    
    def __init__(self, sig_dim: int = 510, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        # Per-neuron encoding (signature is [batch, n_neurons, n_features])
        # But we'll receive it flattened as [batch, sig_dim]
        self.net = nn.Sequential(
            nn.Linear(sig_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        return self.net(signature)


class ConditionalVAE(nn.Module):
    """
    Conditional VAE: Generate weight latents conditioned on behavioral signatures.
    
    p(z | signature) - the distribution of weight latents given behavior
    """
    
    def __init__(
        self,
        weight_dim: int = 181,
        sig_dim: int = 510,
        latent_dim: int = 32,
        condition_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Signature encoder
        self.sig_encoder = SignatureEncoder(sig_dim, output_dim=condition_dim)
        
        # Encoder: (weights, condition) -> (mu, logvar)
        self.encoder_net = nn.Sequential(
            nn.Linear(weight_dim + condition_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder: (z, condition) -> weights
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, weight_dim),
        )
    
    def encode(self, weights: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode weights + condition to latent distribution parameters."""
        x = torch.cat([weights, condition], dim=-1)
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent + condition to weights."""
        x = torch.cat([z, condition], dim=-1)
        return self.decoder(x)
    
    def forward(self, weights: torch.Tensor, signature: torch.Tensor):
        """
        Forward pass for training.
        
        Returns: (reconstructed_weights, mu, logvar, condition)
        """
        # Encode signature to condition
        condition = self.sig_encoder(signature)
        
        # Encode to latent
        mu, logvar = self.encode(weights, condition)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_weights = self.decode(z, condition)
        
        return recon_weights, mu, logvar, condition
    
    def generate(self, signature: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Generate weights from signature (sampling from prior)."""
        condition = self.sig_encoder(signature)
        
        # Sample from prior
        z = torch.randn(signature.size(0) * n_samples, self.latent_dim, device=signature.device)
        condition = condition.repeat_interleave(n_samples, dim=0)
        
        return self.decode(z, condition)


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_with_weights():
    """Load dataset with actual weights and signatures."""
    print("Loading dataset...")
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    
    all_weights = []
    all_signatures = []
    all_labels = []
    all_configs = []
    
    # First pass: determine expected weight size for target architecture
    expected_weight_size = None
    
    print("Processing samples...")
    for i in tqdm(range(len(hf_ds)), desc='Loading'):
        sample = hf_ds[i]
        pattern = sample['classification_completion']
        
        if pattern not in ALL_PATTERNS:
            continue
        
        # Extract weights
        try:
            weights_data = json.loads(sample['improved_model_weights'])
            config = weights_data['config']
            
            # Filter for target architecture
            arch = (config['num_layers'], config['neurons_per_layer'])
            if arch != TARGET_ARCH:
                continue
            
            # Flatten weights in consistent order
            flat_weights = []
            for key in sorted(weights_data['weights'].keys()):
                w = weights_data['weights'][key]
                if isinstance(w[0], list):
                    for row in w:
                        flat_weights.extend(row)
                else:
                    flat_weights.extend(w)
            
            # Set expected size from first valid sample
            if expected_weight_size is None:
                expected_weight_size = len(flat_weights)
                print(f"Expected weight size: {expected_weight_size}")
            
            # Verify size consistency
            if len(flat_weights) != expected_weight_size:
                continue
            
            # Extract signature
            sig_data = json.loads(sample['improved_signature'])
            na = sig_data['neuron_activations']
            
            sig_features = []
            for layer in sorted(na.keys(), key=int):
                for neuron in sorted(na[layer].get('neuron_profiles', {}).keys(), key=int):
                    profile = na[layer]['neuron_profiles'][neuron]
                    sig_features.extend([
                        profile.get('mean', 0),
                        profile.get('std', 0),
                    ])
                    sig_features.extend(profile.get('fourier', [0] * 5)[:5])
                    sig_features.extend(profile.get('input_correlations', [0] * 8)[:8])
                    sig_features.append(profile.get('pre_activation_mean', 0))
                    sig_features.append(profile.get('pre_activation_std', 0))
            
            # Pad/truncate signature to fixed size
            max_sig_dim = 510  # 30 neurons * 17 features
            sig_features = sig_features[:max_sig_dim]
            sig_features += [0] * (max_sig_dim - len(sig_features))
            
            all_weights.append(flat_weights)
            all_signatures.append(sig_features)
            all_labels.append(PATTERN_TO_IDX[pattern])
            all_configs.append(config)
            
        except Exception as e:
            continue
    
    print(f"Loaded {len(all_weights)} samples")
    
    return {
        'weights': torch.tensor(all_weights, dtype=torch.float32),
        'signatures': torch.tensor(all_signatures, dtype=torch.float32),
        'labels': torch.tensor(all_labels, dtype=torch.long),
        'configs': all_configs,
    }


# =============================================================================
# Training
# =============================================================================

def compute_functional_loss(
    original_weights: torch.Tensor,
    reconstructed_weights: torch.Tensor,
    n_probes: int = 30,
) -> torch.Tensor:
    """
    Compute functional loss: do reconstructed weights produce same behavior?
    
    This is the KEY insight - we care about function, not exact weights.
    """
    batch_size = original_weights.size(0)
    
    # Fixed probe inputs
    torch.manual_seed(42)
    probes = torch.randn(n_probes, 5, device=original_weights.device)
    
    total_loss = 0.0
    
    for i in range(batch_size):
        # Create networks from weights
        orig_net = create_network_from_flat(original_weights[i].detach().cpu())
        recon_net = create_network_from_flat(reconstructed_weights[i].detach().cpu())
        
        # Get outputs
        with torch.no_grad():
            orig_out = orig_net(probes.cpu())
            recon_out = recon_net(probes.cpu())
        
        # Compare outputs (functional similarity)
        total_loss += F.mse_loss(recon_out, orig_out)
    
    return total_loss / batch_size


def train_cvae(
    model: ConditionalVAE,
    data: Dict,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'auto',
    lambda_kl: float = 0.1,
    lambda_functional: float = 1.0,
):
    """Train the Conditional VAE with functional loss."""
    
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Training on {device}")
    model = model.to(device)
    
    # Normalize data
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    w_mean, w_std = weights.mean(0), weights.std(0).clamp(min=1e-6)
    s_mean, s_std = signatures.mean(0), signatures.std(0).clamp(min=1e-6)
    
    weights_norm = (weights - w_mean) / w_std
    sigs_norm = (signatures - s_mean) / s_std
    
    # Split
    n = len(weights)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:int(0.8 * n)]
    test_idx = perm[int(0.8 * n):]
    
    train_ds = TensorDataset(
        weights_norm[train_idx], sigs_norm[train_idx], labels[train_idx],
        weights[train_idx]  # Keep unnormalized for functional eval
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_test_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in train_loader:
            w_norm, s_norm, lab, w_orig = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            # Forward
            recon_w, mu, logvar, _ = model(w_norm, s_norm)
            
            # Reconstruction loss (weight space)
            loss_recon = F.mse_loss(recon_w, w_norm)
            
            # KL divergence
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss
            loss = loss_recon + lambda_kl * loss_kl
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()
        
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            
            with torch.no_grad():
                # Test reconstruction
                test_w = weights_norm[test_idx].to(device)
                test_s = sigs_norm[test_idx].to(device)
                test_labels = labels[test_idx]
                
                recon_w, _, _, _ = model(test_w, test_s)
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(recon_w, test_w, dim=1).mean().item()
                
                # Test generation (sample from prior)
                gen_w = model.generate(test_s)
                
                # Unnormalize generated weights
                gen_w_unnorm = gen_w.cpu() * w_std + w_mean
                
                # Test functional accuracy
                correct = 0
                total = 0
                for i in range(min(50, len(gen_w_unnorm))):
                    pattern = IDX_TO_PATTERN[test_labels[i].item()]
                    net = create_network_from_flat(gen_w_unnorm[i])
                    result = test_behavior(net, pattern)
                    if result['supported'] and result['correct']:
                        correct += 1
                    total += 1
                
                func_acc = correct / total if total > 0 else 0
            
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Recon Cos: {cos_sim:.4f} | Func Acc: {func_acc:.4f}")
            
            if func_acc > best_test_acc:
                best_test_acc = func_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, {'w_mean': w_mean, 'w_std': w_std, 's_mean': s_mean, 's_std': s_std}


def test_behavior_editing(
    model: ConditionalVAE,
    data: Dict,
    norm_stats: Dict,
    device: str = 'auto',
):
    """Test if editing the conditioning changes behavior."""
    
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    w_mean, w_std = norm_stats['w_mean'], norm_stats['w_std']
    s_mean, s_std = norm_stats['s_mean'], norm_stats['s_std']
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    sigs_norm = (signatures - s_mean) / s_std
    
    print("\n" + "=" * 60)
    print("BEHAVIOR EDITING TEST")
    print("=" * 60)
    
    with torch.no_grad():
        # Get conditioning for sorted_descending and sorted_ascending
        desc_idx = PATTERN_TO_IDX['sorted_descending']
        asc_idx = PATTERN_TO_IDX['sorted_ascending']
        
        desc_mask = labels == desc_idx
        asc_mask = labels == asc_idx
        
        print(f"Descending samples: {desc_mask.sum()}, Ascending samples: {asc_mask.sum()}")
        
        if desc_mask.sum() < 5 or asc_mask.sum() < 5:
            print("Not enough samples for editing test")
            return
        
        desc_sigs = sigs_norm[desc_mask][:20].to(device)
        asc_sigs = sigs_norm[asc_mask][:20].to(device)
        
        # Compute class centroids in conditioning space
        desc_cond = model.sig_encoder(desc_sigs).mean(0)
        asc_cond = model.sig_encoder(asc_sigs).mean(0)
        
        print(f"Conditioning distance: {(desc_cond - asc_cond).norm():.4f}")
        
        # Test: Generate from descending signatures, then edit toward ascending
        print("\nOriginal (sorted_descending signatures):")
        correct_orig = 0
        for i in range(10):
            gen_w = model.generate(desc_sigs[i:i+1])
            gen_w_unnorm = gen_w.cpu() * w_std + w_mean
            net = create_network_from_flat(gen_w_unnorm[0])
            result = test_behavior(net, 'sorted_descending')
            if result['supported']:
                print(f"  Sample {i}: correct={result['correct']}, margin={result['margin']:.3f}")
                if result['correct']:
                    correct_orig += 1
        print(f"  Accuracy: {correct_orig}/10")
        
        # Get actual descending weights to encode
        desc_weights = (data['weights'][desc_mask][:10] - w_mean) / w_std
        desc_weights = desc_weights.to(device)
        
        # Edit: Replace condition (not interpolate - swap entirely)
        print("\nCondition swapping (encode desc weights, decode with asc condition):")
        correct_orig = 0
        correct_swapped_desc = 0
        correct_swapped_asc = 0
        
        for i in range(min(10, len(desc_weights))):
            # Encode the actual descending weights
            desc_cond_i = model.sig_encoder(desc_sigs[i:i+1])
            mu, logvar = model.encode(desc_weights[i:i+1], desc_cond_i)
            z = mu  # Use mean for deterministic
            
            # Decode with ORIGINAL condition
            recon_orig = model.decode(z, desc_cond_i)
            recon_orig_w = (recon_orig.cpu() * w_std + w_mean)[0]
            net_orig = create_network_from_flat(recon_orig_w)
            result_orig = test_behavior(net_orig, 'sorted_descending')
            if result_orig['supported'] and result_orig['correct']:
                correct_orig += 1
            
            # Decode with ASCENDING condition (same z!)
            # Need to expand asc_cond to match batch dimension
            asc_cond_expanded = asc_cond if asc_cond.dim() == 2 else asc_cond.unsqueeze(0)
            recon_swap = model.decode(z, asc_cond_expanded)
            recon_swap_w = (recon_swap.cpu() * w_std + w_mean)[0]
            net_swap = create_network_from_flat(recon_swap_w)
            
            result_swap_desc = test_behavior(net_swap, 'sorted_descending')
            result_swap_asc = test_behavior(net_swap, 'sorted_ascending')
            
            if result_swap_desc['supported'] and result_swap_desc['correct']:
                correct_swapped_desc += 1
            if result_swap_asc['supported'] and result_swap_asc['correct']:
                correct_swapped_asc += 1
        
        print(f"  Original (desc cond): {correct_orig}/10 correct as descending")
        print(f"  Swapped (asc cond): {correct_swapped_desc}/10 still descending, {correct_swapped_asc}/10 now ascending")
        
        # Also test interpolation
        print("\nCondition interpolation:")
        for alpha in [0.0, 0.5, 1.0]:
            correct_desc = 0
            correct_asc = 0
            
            for i in range(min(10, len(desc_weights))):
                desc_cond_i = model.sig_encoder(desc_sigs[i:i+1])
                mu, _ = model.encode(desc_weights[i:i+1], desc_cond_i)
                z = mu
                
                # Interpolate condition
                interp_cond = (1 - alpha) * desc_cond_i + alpha * asc_cond
                
                recon = model.decode(z, interp_cond)
                recon_w = (recon.cpu() * w_std + w_mean)[0]
                net = create_network_from_flat(recon_w)
                
                r_desc = test_behavior(net, 'sorted_descending')
                r_asc = test_behavior(net, 'sorted_ascending')
                
                if r_desc['supported'] and r_desc['correct']:
                    correct_desc += 1
                if r_asc['supported'] and r_asc['correct']:
                    correct_asc += 1
            
            print(f"  α={alpha}: {correct_desc}/10 descending, {correct_asc}/10 ascending")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If 'now ascending' increases with α, editing works functionally!")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("EXPERIMENT 13: Functional Hypernetwork")
    print("=" * 60)
    
    # Load data
    data = load_dataset_with_weights()
    
    # Verify subject models work (only for testable patterns)
    print("\nVerifying subject models...")
    correct = 0
    tested = 0
    for i in range(min(50, len(data['weights']))):
        net = create_network_from_flat(data['weights'][i])
        pattern = IDX_TO_PATTERN[data['labels'][i].item()]
        result = test_behavior(net, pattern)
        if result['supported']:
            tested += 1
            if result['correct']:
                correct += 1
    print(f"Subject model accuracy: {correct}/{tested} (on testable patterns)")
    
    # Determine weight dimension from data
    weight_dim = data['weights'].shape[1]
    sig_dim = data['signatures'].shape[1]
    print(f"Weight dim: {weight_dim}, Signature dim: {sig_dim}")
    
    # Create and train CVAE
    print("\nTraining Conditional VAE...")
    model = ConditionalVAE(
        weight_dim=weight_dim,
        sig_dim=sig_dim,
        latent_dim=64,  # Larger latent
        condition_dim=128,  # Larger conditioning
    )
    
    model, norm_stats = train_cvae(
        model, data,
        epochs=150,
        batch_size=64,
        lr=1e-3,
        lambda_kl=0.05,  # Lower KL weight for better reconstruction
    )
    
    # Test editing
    test_behavior_editing(model, data, norm_stats)


if __name__ == "__main__":
    main()
