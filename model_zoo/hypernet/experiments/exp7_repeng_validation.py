"""
Experiment 7: Representation Engineering Validation

Tests whether the dual-encoder architecture enables meaningful representation engineering:

1. Z_behavior Structure: Do neurons with similar functions cluster?
2. Z_behavior Causality: Does modifying Z_behavior change network behavior?
3. Interpolation: Can we smoothly interpolate between model behaviors?
4. Steering Vectors: Can we find directions that correspond to behavioral changes?
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import asdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hypernet.experiments.exp6_dual_encoder import (
    DualEncoderSystem, ExperimentConfig, weights_to_state_dict
)
from hypernet.utils.data import create_dataloaders
from hypernet.functional_eval import (
    SubjectModel, load_weights_into_model, create_test_inputs,
    compute_functional_agreement, compute_output_correlation
)


def load_trained_model(checkpoint_path: str, device: str = "auto") -> Tuple[DualEncoderSystem, ExperimentConfig]:
    """Load a trained dual-encoder model from checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ExperimentConfig(**checkpoint['config'])
    
    model = DualEncoderSystem(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


@torch.no_grad()
def extract_latents(
    model: DualEncoderSystem,
    loader,
    device: str,
    n_samples: int = 100,
) -> Dict[str, torch.Tensor]:
    """Extract Z_behavior and Z_weight for a set of models."""
    
    all_z_behavior = []
    all_z_weight = []
    all_positions = []
    all_weights = []
    all_signatures = []
    all_masks = []
    
    total = 0
    for batch in loader:
        if total >= n_samples:
            break
            
        signatures = batch['signatures'].to(device)
        weights = batch['weights'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        
        # Get latents
        _, z_behavior, z_weight = model(signatures, weights, positions, mask)
        
        all_z_behavior.append(z_behavior.cpu())
        all_z_weight.append(z_weight.cpu())
        all_positions.append(positions.cpu())
        all_weights.append(weights.cpu())
        all_signatures.append(signatures.cpu())
        all_masks.append(mask.cpu())
        
        total += signatures.shape[0]
    
    return {
        'z_behavior': torch.cat(all_z_behavior, dim=0)[:n_samples],
        'z_weight': torch.cat(all_z_weight, dim=0)[:n_samples],
        'positions': torch.cat(all_positions, dim=0)[:n_samples],
        'weights': torch.cat(all_weights, dim=0)[:n_samples],
        'signatures': torch.cat(all_signatures, dim=0)[:n_samples],
        'masks': torch.cat(all_masks, dim=0)[:n_samples],
    }


def test_latent_structure(data: Dict[str, torch.Tensor], save_dir: Path) -> Dict[str, float]:
    """
    Test 1: Is Z_behavior structured?
    
    - Neurons in same layer should cluster (they have similar functions)
    - Z_behavior should have lower intra-layer variance than Z_weight
    """
    print("\n" + "="*60)
    print("TEST 1: Latent Space Structure")
    print("="*60)
    
    z_behavior = data['z_behavior']  # [n_models, n_neurons, latent_dim]
    z_weight = data['z_weight']
    positions = data['positions']
    masks = data['masks']
    
    # Flatten across models, keeping only valid neurons
    z_b_flat = []
    z_w_flat = []
    layer_labels = []
    
    for i in range(len(z_behavior)):
        valid = masks[i].bool()
        z_b_flat.append(z_behavior[i][valid])
        z_w_flat.append(z_weight[i][valid])
        layer_labels.extend(positions[i][valid, 0].int().tolist())
    
    z_b_flat = torch.cat(z_b_flat, dim=0).numpy()
    z_w_flat = torch.cat(z_w_flat, dim=0).numpy()
    layer_labels = np.array(layer_labels)
    
    print(f"Total neurons: {len(z_b_flat)}")
    print(f"Unique layers: {np.unique(layer_labels)}")
    
    # Compute intra-layer variance
    def compute_intra_layer_variance(z, layers):
        variances = []
        for layer in np.unique(layers):
            layer_z = z[layers == layer]
            if len(layer_z) > 1:
                var = np.var(layer_z, axis=0).mean()
                variances.append(var)
        return np.mean(variances)
    
    var_behavior = compute_intra_layer_variance(z_b_flat, layer_labels)
    var_weight = compute_intra_layer_variance(z_w_flat, layer_labels)
    
    print(f"\nIntra-layer variance:")
    print(f"  Z_behavior: {var_behavior:.4f}")
    print(f"  Z_weight:   {var_weight:.4f}")
    print(f"  Ratio:      {var_behavior/var_weight:.4f} (lower = more structured)")
    
    # PCA visualization
    pca = PCA(n_components=2)
    z_b_pca = pca.fit_transform(z_b_flat)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_b_pca[:, 0], z_b_pca[:, 1], c=layer_labels, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, label='Layer Index')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Z_behavior PCA (colored by layer)')
    plt.savefig(save_dir / 'z_behavior_pca.png', dpi=150)
    plt.close()
    print(f"\nSaved PCA plot to {save_dir / 'z_behavior_pca.png'}")
    
    return {
        'intra_layer_var_behavior': var_behavior,
        'intra_layer_var_weight': var_weight,
        'variance_ratio': var_behavior / var_weight,
    }


def test_modification_causality(
    model: DualEncoderSystem,
    data: Dict[str, torch.Tensor],
    device: str,
    save_dir: Path,
) -> Dict[str, float]:
    """
    Test 2: Does modifying Z_behavior change network behavior?
    
    - Add noise to Z_behavior, measure behavior change
    - Add noise to Z_weight, measure behavior change
    - Z_behavior modifications should have larger effect per unit norm
    """
    print("\n" + "="*60)
    print("TEST 2: Modification Causality")
    print("="*60)
    
    model.eval()
    
    z_behavior = data['z_behavior'].to(device)
    z_weight = data['z_weight'].to(device)
    positions = data['positions'].to(device)
    weights = data['weights'].to(device)
    masks = data['masks'].to(device)
    
    noise_scales = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    behavior_effects = []
    weight_effects = []
    
    n_test = min(20, len(z_behavior))
    
    for noise_scale in noise_scales:
        behavior_correlations = []
        weight_correlations = []
        
        for i in range(n_test):
            valid = masks[i].bool()
            z_b = z_behavior[i:i+1]
            z_w = z_weight[i:i+1]
            pos = positions[i:i+1]
            true_w = weights[i][valid].cpu()
            
            # Original prediction
            pred_orig = model.decode(z_b, z_w, pos)[0][valid].cpu()
            
            # Modify Z_behavior
            noise_b = torch.randn_like(z_b) * noise_scale
            z_b_mod = z_b + noise_b
            pred_b_mod = model.decode(z_b_mod, z_w, pos)[0][valid].cpu()
            
            # Modify Z_weight  
            noise_w = torch.randn_like(z_w) * noise_scale
            z_w_mod = z_w + noise_w
            pred_w_mod = model.decode(z_b, z_w_mod, pos)[0][valid].cpu()
            
            # Measure change (cosine distance from original prediction)
            def cosine_dist(a, b):
                return 1 - F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
            
            behavior_correlations.append(cosine_dist(pred_orig, pred_b_mod))
            weight_correlations.append(cosine_dist(pred_orig, pred_w_mod))
        
        behavior_effects.append(np.mean(behavior_correlations))
        weight_effects.append(np.mean(weight_correlations))
        
        print(f"Noise scale {noise_scale:.1f}: Z_behavior effect={np.mean(behavior_correlations):.4f}, Z_weight effect={np.mean(weight_correlations):.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(noise_scales, behavior_effects, 'b-o', label='Z_behavior modification')
    plt.plot(noise_scales, weight_effects, 'r-o', label='Z_weight modification')
    plt.xlabel('Noise Scale')
    plt.ylabel('Output Change (cosine distance)')
    plt.title('Effect of Latent Modifications on Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'modification_causality.png', dpi=150)
    plt.close()
    print(f"\nSaved causality plot to {save_dir / 'modification_causality.png'}")
    
    return {
        'behavior_effect_0.5': behavior_effects[3],
        'weight_effect_0.5': weight_effects[3],
        'behavior_sensitivity': behavior_effects[3] / (noise_scales[3] + 1e-8),
    }


def test_interpolation(
    model: DualEncoderSystem,
    data: Dict[str, torch.Tensor],
    device: str,
    save_dir: Path,
) -> Dict[str, float]:
    """
    Test 3: Can we smoothly interpolate between model behaviors?
    
    - Take two models, interpolate their Z_behavior
    - Check that decoded weights produce intermediate behaviors
    """
    print("\n" + "="*60)
    print("TEST 3: Behavior Interpolation")
    print("="*60)
    
    model.eval()
    
    z_behavior = data['z_behavior'].to(device)
    z_weight = data['z_weight'].to(device)
    positions = data['positions'].to(device)
    weights = data['weights'].to(device)
    masks = data['masks'].to(device)
    
    # Find two models with same architecture (same number of neurons)
    neuron_counts = masks.sum(dim=1).cpu()
    # Find most common count manually (MPS doesn't support mode)
    unique_counts, counts = torch.unique(neuron_counts, return_counts=True)
    common_count = unique_counts[counts.argmax()].item()
    same_arch_indices = (neuron_counts == common_count).nonzero().squeeze()
    
    if len(same_arch_indices) < 2:
        print("Not enough models with same architecture for interpolation test")
        return {}
    
    idx1, idx2 = same_arch_indices[0].item(), same_arch_indices[1].item()
    
    print(f"Interpolating between model {idx1} and model {idx2}")
    print(f"Both have {int(common_count)} neurons")
    
    # Get latents for both models
    z_b1, z_w1 = z_behavior[idx1:idx1+1], z_weight[idx1:idx1+1]
    z_b2, z_w2 = z_behavior[idx2:idx2+1], z_weight[idx2:idx2+1]
    pos = positions[idx1:idx1+1]  # Same architecture
    mask = masks[idx1]
    valid = mask.bool()
    
    # Interpolation
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    interpolated_weights = []
    
    for alpha in alphas:
        z_b_interp = (1 - alpha) * z_b1 + alpha * z_b2
        # Keep z_weight from model 1 (we're testing Z_behavior interpolation)
        pred_w = model.decode(z_b_interp, z_w1, pos)[0][valid].cpu()
        interpolated_weights.append(pred_w)
    
    # Measure smoothness: consecutive predictions should be similar
    smoothness_scores = []
    for i in range(len(alphas) - 1):
        cos_sim = F.cosine_similarity(
            interpolated_weights[i].flatten().unsqueeze(0),
            interpolated_weights[i+1].flatten().unsqueeze(0)
        ).item()
        smoothness_scores.append(cos_sim)
    
    avg_smoothness = np.mean(smoothness_scores)
    print(f"\nInterpolation smoothness (cosine between consecutive): {smoothness_scores}")
    print(f"Average smoothness: {avg_smoothness:.4f}")
    
    # Also check distance from endpoints
    dist_from_start = []
    dist_from_end = []
    for i, w in enumerate(interpolated_weights):
        d_start = 1 - F.cosine_similarity(w.flatten().unsqueeze(0), interpolated_weights[0].flatten().unsqueeze(0)).item()
        d_end = 1 - F.cosine_similarity(w.flatten().unsqueeze(0), interpolated_weights[-1].flatten().unsqueeze(0)).item()
        dist_from_start.append(d_start)
        dist_from_end.append(d_end)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, dist_from_start, 'b-o', label='Distance from model 1')
    plt.plot(alphas, dist_from_end, 'r-o', label='Distance from model 2')
    plt.xlabel('Interpolation alpha')
    plt.ylabel('Cosine distance')
    plt.title('Z_behavior Interpolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'interpolation.png', dpi=150)
    plt.close()
    print(f"\nSaved interpolation plot to {save_dir / 'interpolation.png'}")
    
    return {
        'interpolation_smoothness': avg_smoothness,
        'monotonic_from_start': all(dist_from_start[i] <= dist_from_start[i+1] for i in range(len(dist_from_start)-1)),
        'monotonic_from_end': all(dist_from_end[i] >= dist_from_end[i+1] for i in range(len(dist_from_end)-1)),
    }


def test_functional_modification(
    model: DualEncoderSystem,
    data: Dict[str, torch.Tensor],
    device: str,
    save_dir: Path,
) -> Dict[str, float]:
    """
    Test 4: Do Z_behavior modifications produce functionally different but valid networks?
    
    - Modify Z_behavior
    - Build the resulting network
    - Check it still produces valid outputs (not collapsed/NaN)
    - Check outputs are different from original
    """
    print("\n" + "="*60)
    print("TEST 4: Functional Modification")
    print("="*60)
    
    model.eval()
    
    z_behavior = data['z_behavior'].to(device)
    z_weight = data['z_weight'].to(device)
    positions = data['positions'].to(device)
    weights = data['weights'].to(device)
    masks = data['masks'].to(device)
    
    n_test = min(20, len(z_behavior))
    
    valid_networks = 0
    different_behaviors = 0
    functional_agreements = []
    
    for i in range(n_test):
        try:
            valid = masks[i].bool()
            z_b = z_behavior[i:i+1]
            z_w = z_weight[i:i+1]
            pos = positions[i:i+1]
            true_w = weights[i][valid].cpu()
            pos_valid = positions[i][valid].cpu()
            
            # Original prediction
            pred_orig = model.decode(z_b, z_w, pos)[0][valid].cpu()
            
            # Modified Z_behavior (add meaningful perturbation)
            # Use the mean direction of Z_behavior as a steering vector
            z_b_mean = z_behavior[:, valid, :].mean(dim=(0, 1), keepdim=True)
            z_b_mod = z_b + 0.5 * (z_b_mean.to(device) - z_b.mean(dim=1, keepdim=True))
            pred_mod = model.decode(z_b_mod, z_w, pos)[0][valid].cpu()
            
            # Build networks
            orig_state = weights_to_state_dict(pred_orig, pos_valid)
            mod_state = weights_to_state_dict(pred_mod, pos_valid)
            
            # Infer config
            layer_indices = sorted(set(int(p[0].item()) for p in pos_valid))
            num_hidden = len(layer_indices) - 1
            neurons_per_layer = int((pos_valid[:, 0] == 0).sum().item())
            fan_in = int(pos_valid[0, 2].item())
            
            config = {
                'vocab_size': 10,
                'sequence_length': fan_in,
                'num_layers': num_hidden,
                'neurons_per_layer': neurons_per_layer,
            }
            
            orig_model = SubjectModel(config)
            mod_model = SubjectModel(config)
            
            load_weights_into_model(orig_model, orig_state)
            load_weights_into_model(mod_model, mod_state)
            
            # Test
            test_input = create_test_inputs(config, 100)
            
            orig_out = orig_model(test_input)
            mod_out = mod_model(test_input)
            
            # Check validity
            if not (torch.isnan(mod_out).any() or torch.isinf(mod_out).any()):
                valid_networks += 1
                
                # Check if behavior is different
                agreement = compute_functional_agreement(orig_model, mod_model, test_input)
                functional_agreements.append(agreement)
                
                if agreement < 0.95:  # Meaningfully different
                    different_behaviors += 1
                    
        except Exception as e:
            continue
    
    print(f"\nResults:")
    print(f"  Valid networks: {valid_networks}/{n_test} ({valid_networks/n_test:.1%})")
    print(f"  Different behaviors: {different_behaviors}/{valid_networks} ({different_behaviors/max(1,valid_networks):.1%})")
    if functional_agreements:
        print(f"  Mean functional agreement: {np.mean(functional_agreements):.4f}")
    
    return {
        'valid_network_rate': valid_networks / n_test,
        'different_behavior_rate': different_behaviors / max(1, valid_networks),
        'mean_functional_agreement': np.mean(functional_agreements) if functional_agreements else 0.0,
    }


def run_validation(checkpoint_path: str, device: str = "auto"):
    """Run all rep-eng validation tests."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("="*60)
    print("REPRESENTATION ENGINEERING VALIDATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Setup save directory
    checkpoint_dir = Path(checkpoint_path).parent
    save_dir = checkpoint_dir / "repeng_validation"
    save_dir.mkdir(exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model, config = load_trained_model(checkpoint_path, device)
    print(f"Loaded model: {config.name}")
    
    # Load data
    print("\nLoading data...")
    _, val_loader = create_dataloaders(
        batch_size=32,
        train_samples=100,
        val_samples=200,
    )
    
    # Extract latents
    print("\nExtracting latents...")
    data = extract_latents(model, val_loader, device, n_samples=100)
    print(f"Extracted latents for {len(data['z_behavior'])} models")
    
    # Run tests
    results = {}
    
    results['structure'] = test_latent_structure(data, save_dir)
    results['causality'] = test_modification_causality(model, data, device, save_dir)
    results['interpolation'] = test_interpolation(model, data, device, save_dir)
    results['functional'] = test_functional_modification(model, data, device, save_dir)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print("\n1. Latent Structure:")
    print(f"   Variance ratio (behavior/weight): {results['structure']['variance_ratio']:.4f}")
    print(f"   (Lower = Z_behavior more structured by layer)")
    
    print("\n2. Modification Causality:")
    print(f"   Z_behavior sensitivity: {results['causality']['behavior_sensitivity']:.4f}")
    print(f"   (Higher = modifications have larger effect)")
    
    print("\n3. Interpolation:")
    print(f"   Smoothness: {results['interpolation']['interpolation_smoothness']:.4f}")
    print(f"   Monotonic: {results['interpolation']['monotonic_from_start']}")
    
    print("\n4. Functional Modification:")
    print(f"   Valid networks: {results['functional']['valid_network_rate']:.1%}")
    print(f"   Different behaviors: {results['functional']['different_behavior_rate']:.1%}")
    
    # Save results
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {save_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    results = run_validation(args.checkpoint, args.device)
