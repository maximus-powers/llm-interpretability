"""
Experiment 12 Phase 1: Universal Weight Hypothesis Analysis

GOAL: Verify if the Universal Weight Hypothesis holds for our subject models.
- Extract all subject model weights from the dataset
- Apply HOSVD to find universal basis vectors
- Measure variance explained by top-k components
- Express each model as k coefficients

If UWH holds, ~16 coefficients should explain >90% of weight variance.
"""

import sys
import json
import pickle
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class UWHConfig:
    max_components: int = 64  # Max basis vectors to compute
    variance_threshold: float = 0.99  # Target variance explained
    max_models: int = 10000  # Max models to analyze
    

def extract_weights_from_signature(sample: Dict) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract flattened weight vector from a sample's signature data.
    
    Returns dict with:
    - 'weights': flattened weight tensor
    - 'weight_structure': list of (layer_idx, neuron_idx, fan_in) for reconstruction
    """
    try:
        sig_data = json.loads(sample['improved_signature'])
        neuron_activations = sig_data['neuron_activations']
        
        all_weights = []
        weight_structure = []
        
        layer_indices = sorted([int(k) for k in neuron_activations.keys()])
        
        for layer_idx in layer_indices:
            layer_data = neuron_activations.get(str(layer_idx), {})
            neuron_profiles = layer_data.get('neuron_profiles', {})
            
            for neuron_idx in sorted([int(k) for k in neuron_profiles.keys()]):
                profile = neuron_profiles[str(neuron_idx)]
                
                # Get input correlations (these ARE the weight proxies)
                input_corr = profile.get('input_correlations', [])
                if input_corr:
                    all_weights.extend(input_corr)
                    weight_structure.append({
                        'layer': layer_idx,
                        'neuron': neuron_idx,
                        'fan_in': len(input_corr)
                    })
        
        if not all_weights:
            return None
            
        return {
            'weights': torch.tensor(all_weights, dtype=torch.float32),
            'structure': weight_structure,
            'pattern': sample.get('classification_completion', 'unknown')
        }
    except Exception as e:
        return None


def collect_all_weights(config: UWHConfig) -> Tuple[torch.Tensor, List[Dict], List[str]]:
    """
    Collect weights from all models in the dataset.
    
    Returns:
    - weight_matrix: [num_models, weight_dim] tensor
    - structures: list of weight structures (for reconstruction)
    - patterns: list of behavior patterns
    """
    print("Loading dataset...")
    hf_ds = hf_load_dataset('maximuspowers/muat-sigs-with-input-correlations', split='train')
    
    num_models = min(len(hf_ds), config.max_models)
    print(f"Processing {num_models} models...")
    
    weights_list = []
    structures = []
    patterns = []
    
    # First pass: determine consistent weight dimension
    weight_dims = []
    for i in range(min(100, num_models)):
        result = extract_weights_from_signature(hf_ds[i])
        if result:
            weight_dims.append(len(result['weights']))
    
    if not weight_dims:
        raise ValueError("Could not extract weights from any samples")
    
    # Use the most common weight dimension
    from collections import Counter
    dim_counts = Counter(weight_dims)
    target_dim = dim_counts.most_common(1)[0][0]
    print(f"Target weight dimension: {target_dim}")
    
    # Second pass: collect all weights with matching dimension
    for i in tqdm(range(num_models), desc="Extracting weights"):
        result = extract_weights_from_signature(hf_ds[i])
        if result and len(result['weights']) == target_dim:
            weights_list.append(result['weights'])
            structures.append(result['structure'])
            patterns.append(result['pattern'])
    
    print(f"Collected {len(weights_list)} models with dimension {target_dim}")
    
    weight_matrix = torch.stack(weights_list)
    return weight_matrix, structures, patterns


def compute_uwh_basis(weight_matrix: torch.Tensor, config: UWHConfig) -> Dict:
    """
    Compute Universal Weight Hypothesis basis via SVD.
    
    Following the UWH paper:
    1. Zero-center the weights
    2. Compute SVD
    3. Analyze variance explained by top-k components
    """
    print(f"\nWeight matrix shape: {weight_matrix.shape}")
    num_models, weight_dim = weight_matrix.shape
    
    # Step 1: Zero-center
    mean_weights = weight_matrix.mean(dim=0)
    centered_weights = weight_matrix - mean_weights
    
    print(f"Mean weight norm: {mean_weights.norm():.4f}")
    print(f"Centered weight std: {centered_weights.std():.4f}")
    
    # Step 2: SVD (truncated for efficiency)
    print("\nComputing SVD...")
    k = min(config.max_components, num_models, weight_dim)
    
    # Use torch.linalg.svd for better numerical stability
    # svd_lowrank returns V not Vh, need to transpose
    if num_models > 1000:
        # Use randomized SVD for efficiency
        print(f"Using randomized SVD with k={k}")
        U, S, V = torch.svd_lowrank(centered_weights, q=k)
        Vh = V.T  # [k, weight_dim]
    else:
        U, S, Vh = torch.linalg.svd(centered_weights, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
    
    print(f"SVD complete. U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
    
    # Step 3: Analyze variance explained
    total_variance = (centered_weights ** 2).sum()
    explained_variance = S ** 2
    cumulative_variance = torch.cumsum(explained_variance, dim=0) / total_variance
    
    # Find number of components for target variance
    variance_thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
    components_needed = {}
    for thresh in variance_thresholds:
        n_components = (cumulative_variance < thresh).sum().item() + 1
        components_needed[thresh] = min(n_components, k)
    
    print("\n" + "="*60)
    print("VARIANCE ANALYSIS")
    print("="*60)
    for thresh, n in components_needed.items():
        actual_var = cumulative_variance[min(n-1, len(cumulative_variance)-1)].item()
        print(f"{thresh*100:5.1f}% variance: {n:3d} components (actual: {actual_var*100:.2f}%)")
    
    # Detailed breakdown for first 32 components
    print("\nTop-32 component breakdown:")
    print(f"{'k':>4} | {'Var %':>8} | {'Cumul %':>8} | {'Singular':>10}")
    print("-" * 40)
    for i in range(min(32, len(S))):
        var_i = (explained_variance[i] / total_variance * 100).item()
        cum_i = (cumulative_variance[i] * 100).item()
        print(f"{i+1:4d} | {var_i:8.3f} | {cum_i:8.3f} | {S[i].item():10.4f}")
    
    # Step 4: Compute coefficients for each model
    # coefficients[i] = (W[i] - mean) @ V.T = U[i] * S
    coefficients = U * S.unsqueeze(0)  # [num_models, k]
    
    # Verify reconstruction
    print("\nVerifying reconstruction quality...")
    for n_comp in [4, 8, 16, 32, min(64, k)]:
        if n_comp > k:
            continue
        # Reconstruct using top n_comp components
        recon = mean_weights + coefficients[:, :n_comp] @ Vh[:n_comp, :]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            weight_matrix, recon, dim=1
        ).mean()
        
        # MSE
        mse = ((weight_matrix - recon) ** 2).mean()
        
        print(f"  k={n_comp:2d}: Cosine={cos_sim:.4f}, MSE={mse:.6f}")
    
    return {
        'mean_weights': mean_weights,
        'singular_values': S,
        'basis_vectors': Vh,  # [k, weight_dim] - each row is a basis vector
        'coefficients': coefficients,  # [num_models, k]
        'cumulative_variance': cumulative_variance,
        'components_needed': components_needed,
        'num_models': num_models,
        'weight_dim': weight_dim,
    }


def analyze_coefficient_behavior_correlation(
    coefficients: torch.Tensor,
    patterns: List[str],
    top_k: int = 16
) -> Dict:
    """
    Analyze how UWH coefficients correlate with behavior patterns.
    
    If coefficients encode behavior, different patterns should have
    different coefficient distributions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print("\n" + "="*60)
    print("COEFFICIENT-BEHAVIOR CORRELATION")
    print("="*60)
    
    # Encode patterns
    le = LabelEncoder()
    labels = le.fit_transform(patterns)
    num_classes = len(le.classes_)
    print(f"Number of behavior classes: {num_classes}")
    print(f"Classes: {list(le.classes_)}")
    
    # Use top-k coefficients
    X = coefficients[:, :top_k].numpy()
    y = labels
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    # Test different k values
    for k in [4, 8, 16, min(32, top_k)]:
        if k > top_k:
            continue
            
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train[:, :k], y_train)
        
        train_acc = clf.score(X_train[:, :k], y_train)
        test_acc = clf.score(X_test[:, :k], y_test)
        
        print(f"\nLogistic Regression with k={k} coefficients:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        
        results[k] = {'train_acc': train_acc, 'test_acc': test_acc}
    
    # Compute per-class mean coefficients
    print("\nPer-class coefficient means (top 8):")
    class_means = {}
    for i, cls in enumerate(le.classes_):
        mask = labels == i
        class_coeffs = coefficients[mask, :8].mean(dim=0)
        class_means[cls] = class_coeffs
        print(f"  {cls:20s}: {class_coeffs[:4].numpy().round(3)}")
    
    return {
        'classification_results': results,
        'class_means': {k: v.tolist() for k, v in class_means.items()},
        'num_classes': num_classes,
    }


def run_uwh_analysis():
    """Main analysis function."""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent.parent.parent.parent / "runs" / f"exp12_uwh_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENT 12: Universal Weight Hypothesis Analysis")
    print("="*60)
    
    config = UWHConfig()
    
    # Step 1: Collect weights
    weight_matrix, structures, patterns = collect_all_weights(config)
    
    # Step 2: Compute UWH basis
    uwh_results = compute_uwh_basis(weight_matrix, config)
    
    # Step 3: Analyze behavior correlation
    behavior_results = analyze_coefficient_behavior_correlation(
        uwh_results['coefficients'],
        patterns,
        top_k=32
    )
    
    # Save results
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Key finding: how many components for 90% variance?
    k_90 = uwh_results['components_needed'].get(0.90, -1)
    k_95 = uwh_results['components_needed'].get(0.95, -1)
    
    print(f"Components for 90% variance: {k_90}")
    print(f"Components for 95% variance: {k_95}")
    
    # Best classification from coefficients
    best_k = max(behavior_results['classification_results'].keys())
    best_acc = behavior_results['classification_results'][best_k]['test_acc']
    print(f"Best classification from {best_k} coefficients: {best_acc:.4f}")
    
    # Does UWH hold?
    uwh_holds = k_90 <= 20  # If 90% variance in <=20 components
    print(f"\nUWH holds for our models: {uwh_holds}")
    
    if uwh_holds:
        print(">>> Proceeding with UWH-constrained learning is justified!")
    else:
        print(">>> May need more components or different approach")
    
    # Save everything
    save_data = {
        'config': asdict(config),
        'num_models': uwh_results['num_models'],
        'weight_dim': uwh_results['weight_dim'],
        'components_needed': uwh_results['components_needed'],
        'singular_values': uwh_results['singular_values'][:64].tolist(),
        'cumulative_variance': uwh_results['cumulative_variance'][:64].tolist(),
        'behavior_results': behavior_results,
        'uwh_holds': uwh_holds,
    }
    
    with open(run_dir / "analysis_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    
    # Save the basis for Phase 2
    torch.save({
        'mean_weights': uwh_results['mean_weights'],
        'basis_vectors': uwh_results['basis_vectors'],
        'singular_values': uwh_results['singular_values'],
        'coefficients': uwh_results['coefficients'],
        'patterns': patterns,
        'structures': structures[0] if structures else None,  # Save one structure as reference
    }, run_dir / "uwh_basis.pt")
    
    print(f"\nResults saved to: {run_dir}")
    
    return uwh_results, behavior_results, run_dir


if __name__ == "__main__":
    run_uwh_analysis()
