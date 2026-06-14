"""
Training script for FunctionalHyperNetwork.

Trains a Conditional VAE that generates neural network weights
conditioned on behavioral signatures.

Usage:
    python -m hypernet.train --config configs/hypernet/default.yaml
    python -m hypernet.train --epochs 150 --latent-dim 64
    python -m hypernet.train --use-functional-loss
"""

import argparse
import json
import logging
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypernet.models import FunctionalHyperNetwork, SubjectNetwork, BehaviorEditor
from hypernet.models.functional_hypernetwork import HyperNetConfig
from hypernet.behavior_suite import (
    CLEAN_PROOF_PATTERNS,
    behavior_cases_for_training,
    build_clean_behavior_suite,
)
from hypernet.dataset_provenance import (
    deduplicate_fingerprints,
    make_sample_fingerprint,
)

# Optional TensorBoard
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None  # type: ignore
    HAS_TENSORBOARD = False


# =============================================================================
# Constants
# =============================================================================

ALL_PATTERNS = [
    'contains_abc', 'palindrome', 'alternating', 'sorted_descending',
    'sorted_ascending', 'ends_with', 'decreasing_pairs', 'mountain_pattern',
    'increasing_pairs', 'first_last_match', 'starts_with', 'no_repeats',
    'has_majority', 'vowel_consonant'
]
PATTERN_TO_IDX = {p: i for i, p in enumerate(ALL_PATTERNS)}
IDX_TO_PATTERN = {i: p for p, i in PATTERN_TO_IDX.items()}
TARGET_ARCH = (5, 8)

logger = logging.getLogger(__name__)


# =============================================================================
# Behavior Testing - All 14 Patterns
# =============================================================================

def get_test_cases(pattern: str) -> Optional[Dict]:
    """Get positive and negative test cases for a pattern."""
    test_cases = {
        'sorted_descending': {
            'positive': [[9, 7, 5, 3, 1], [8, 6, 4, 2, 0], [7, 5, 3, 2, 1],
                        [9, 8, 7, 6, 5], [5, 4, 3, 2, 1]],
            'negative': [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8], [1, 2, 3, 4, 5],
                        [5, 6, 7, 8, 9], [3, 1, 4, 1, 5]],
        },
        'sorted_ascending': {
            'positive': [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8], [1, 2, 3, 4, 5],
                        [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
            'negative': [[9, 7, 5, 3, 1], [8, 6, 4, 2, 0], [7, 5, 3, 2, 1],
                        [9, 8, 7, 6, 5], [3, 1, 4, 1, 5]],
        },
        'palindrome': {
            'positive': [[1, 2, 3, 2, 1], [5, 5, 5, 5, 5], [1, 0, 0, 0, 1],
                        [3, 2, 1, 2, 3], [7, 8, 9, 8, 7]],
            'negative': [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 1, 1, 1, 2],
                        [3, 2, 1, 2, 4], [7, 8, 9, 8, 6]],
        },
        'alternating': {
            'positive': [[1, 2, 1, 2, 1], [5, 6, 5, 6, 5], [0, 1, 0, 1, 0],
                        [3, 4, 3, 4, 3], [9, 8, 9, 8, 9]],
            'negative': [[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1],
                        [1, 2, 2, 1, 2], [9, 9, 8, 8, 7]],
        },
        'first_last_match': {
            'positive': [[1, 2, 3, 4, 1], [5, 0, 0, 0, 5], [3, 1, 2, 1, 3],
                        [7, 8, 9, 8, 7], [0, 5, 5, 5, 0]],
            'negative': [[1, 2, 3, 4, 5], [5, 0, 0, 0, 6], [3, 1, 2, 1, 4],
                        [7, 8, 9, 8, 6], [0, 5, 5, 5, 1]],
        },
        'mountain_pattern': {
            'positive': [[1, 3, 5, 3, 1], [0, 2, 4, 2, 0], [2, 4, 6, 4, 2],
                        [1, 2, 3, 2, 1], [3, 5, 7, 5, 3]],
            'negative': [[5, 3, 1, 3, 5], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5],
                        [5, 4, 3, 2, 1], [1, 3, 2, 4, 1]],
        },
        'increasing_pairs': {
            'positive': [[1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [2, 3, 4, 5, 6],
                        [1, 3, 5, 7, 9], [0, 2, 4, 6, 8]],
            'negative': [[5, 4, 3, 2, 1], [2, 1, 4, 3, 5], [1, 1, 1, 1, 1],
                        [5, 3, 4, 2, 1], [9, 7, 5, 3, 1]],
        },
        'decreasing_pairs': {
            'positive': [[5, 4, 3, 2, 1], [9, 7, 5, 3, 1], [8, 6, 4, 2, 0],
                        [6, 5, 4, 3, 2], [7, 5, 3, 1, 0]],
            'negative': [[1, 2, 3, 4, 5], [1, 3, 5, 7, 9], [0, 2, 4, 6, 8],
                        [1, 1, 1, 1, 1], [2, 4, 3, 5, 1]],
        },
        'no_repeats': {
            'positive': [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 3, 5, 7, 9],
                        [0, 2, 4, 6, 8], [9, 7, 5, 3, 1]],
            'negative': [[1, 1, 2, 3, 4], [1, 2, 2, 3, 4], [1, 2, 3, 3, 4],
                        [1, 2, 3, 4, 4], [5, 5, 5, 5, 5]],
        },
        'has_majority': {
            'positive': [[1, 1, 1, 2, 3], [2, 2, 2, 1, 3], [5, 5, 5, 5, 1],
                        [3, 3, 3, 1, 2], [0, 0, 0, 1, 2]],
            'negative': [[1, 2, 3, 4, 5], [1, 1, 2, 2, 3], [1, 2, 1, 2, 3],
                        [5, 4, 3, 2, 1], [0, 1, 2, 3, 4]],
        },
        # These patterns use character/string logic - harder to test numerically
        'contains_abc': {
            'positive': [[1, 2, 3, 4, 5], [0, 1, 2, 3, 0], [1, 2, 3, 0, 0],
                        [0, 0, 1, 2, 3], [2, 1, 2, 3, 4]],
            'negative': [[5, 4, 3, 2, 1], [1, 3, 5, 7, 9], [0, 0, 0, 0, 0],
                        [9, 8, 7, 6, 5], [1, 1, 1, 1, 1]],
        },
        'starts_with': {
            'positive': [[1, 0, 0, 0, 0], [1, 2, 3, 4, 5], [1, 1, 1, 1, 1],
                        [1, 5, 5, 5, 5], [1, 9, 8, 7, 6]],
            'negative': [[0, 1, 2, 3, 4], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1],
                        [0, 0, 0, 0, 1], [9, 8, 7, 6, 5]],
        },
        'ends_with': {
            'positive': [[0, 0, 0, 0, 1], [5, 4, 3, 2, 1], [1, 1, 1, 1, 1],
                        [9, 8, 7, 6, 1], [0, 1, 2, 3, 1]],
            'negative': [[1, 0, 0, 0, 0], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0],
                        [1, 5, 5, 5, 5], [5, 4, 3, 2, 0]],
        },
        'vowel_consonant': {
            'positive': [[1, 2, 1, 2, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1],
                        [2, 1, 2, 1, 2], [0, 2, 0, 2, 0]],
            'negative': [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 2, 3, 4, 5],
                        [2, 2, 2, 2, 2], [1, 1, 2, 2, 1]],
        },
    }
    return test_cases.get(pattern)


def test_behavior(model: SubjectNetwork, pattern: str) -> Dict:
    """Test if model exhibits the specified behavior pattern."""
    model.eval()
    
    cases = get_test_cases(pattern)
    if cases is None:
        return {'supported': False}
    
    positive = torch.tensor(cases['positive'], dtype=torch.float32)
    negative = torch.tensor(cases['negative'], dtype=torch.float32)
    
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


def test_all_behaviors(model: SubjectNetwork) -> Dict[str, Dict]:
    """Test a model against ALL behavior patterns.
    
    Returns dict mapping pattern -> {positive_output, negative_output, correct, margin}
    """
    results = {}
    for pattern in ALL_PATTERNS:
        result = test_behavior(model, pattern)
        if result.get('supported', False):
            results[pattern] = result
    return results


def evaluate_pattern_behaviors(
    model: FunctionalHyperNetwork,
    data: Dict,
    n_samples_per_pattern: int = 10,
) -> Dict[str, Dict]:
    """
    Evaluate reconstructed networks' behavior on all patterns.
    
    For each pattern P, take n networks trained on P, reconstruct them,
    and test how they respond to ALL patterns' test cases.
    
    Returns:
        {
            'sorted_ascending': {
                'own_positive': 0.85,      # Output on own positive examples
                'own_negative': 0.12,      # Output on own negative examples  
                'own_correct': 0.9,        # Fraction correctly discriminating own pattern
                'own_margin': 0.73,        # Average margin on own pattern
                'cross_pattern': {         # How it responds to OTHER patterns
                    'sorted_descending': {'positive': 0.3, 'negative': 0.7, 'margin': -0.4},
                    ...
                }
            },
            ...
        }
    """
    model.eval()
    device = next(model.parameters()).device
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    results = {}
    
    # Get unique patterns in the dataset
    unique_labels = torch.unique(labels).tolist()
    patterns_in_data = [IDX_TO_PATTERN[l] for l in unique_labels if l in IDX_TO_PATTERN]
    
    for pattern in patterns_in_data:
        pattern_idx = PATTERN_TO_IDX.get(pattern)
        if pattern_idx is None:
            continue
            
        mask = labels == pattern_idx
        indices = torch.where(mask)[0][:n_samples_per_pattern]
        
        if len(indices) == 0:
            continue
        
        # Aggregate metrics for this pattern
        own_pos_outputs = []
        own_neg_outputs = []
        own_correct_count = 0
        cross_results = {p: {'pos': [], 'neg': []} for p in patterns_in_data if p != pattern}
        
        for idx in indices:
            w = weights[idx:idx+1].to(device)
            s = signatures[idx:idx+1].to(device)
            
            # Reconstruct
            with torch.no_grad():
                recon, _, _, _ = model(w, s)
            
            # Create network from reconstructed weights
            recon_net = SubjectNetwork.from_weights(recon[0].cpu())
            
            # Test on own pattern
            own_result = test_behavior(recon_net, pattern)
            if own_result.get('supported', False):
                own_pos_outputs.append(own_result['positive_output'])
                own_neg_outputs.append(own_result['negative_output'])
                if own_result['correct']:
                    own_correct_count += 1
            
            # Test on all other patterns
            for other_pattern in cross_results.keys():
                other_result = test_behavior(recon_net, other_pattern)
                if other_result.get('supported', False):
                    cross_results[other_pattern]['pos'].append(other_result['positive_output'])
                    cross_results[other_pattern]['neg'].append(other_result['negative_output'])
        
        # Aggregate
        n_tested = len(own_pos_outputs)
        if n_tested > 0:
            results[pattern] = {
                'own_positive': sum(own_pos_outputs) / n_tested,
                'own_negative': sum(own_neg_outputs) / n_tested,
                'own_correct': own_correct_count / n_tested,
                'own_margin': (sum(own_pos_outputs) - sum(own_neg_outputs)) / n_tested,
                'n_tested': n_tested,
                'cross_pattern': {}
            }
            
            for other_pattern, other_data in cross_results.items():
                if len(other_data['pos']) > 0:
                    avg_pos = sum(other_data['pos']) / len(other_data['pos'])
                    avg_neg = sum(other_data['neg']) / len(other_data['neg'])
                    results[pattern]['cross_pattern'][other_pattern] = {
                        'positive': avg_pos,
                        'negative': avg_neg,
                        'margin': avg_pos - avg_neg,
                    }
    
    return results


def create_weight_heatmaps(
    model: FunctionalHyperNetwork,
    data: Dict,
    n_samples: int = 4,
) -> Dict[str, plt.Figure]:
    """
    Create heatmap visualizations of original vs reconstructed weights.
    
    Returns dict of matplotlib figures for TensorBoard logging.
    """
    model.eval()
    device = next(model.parameters()).device
    
    weights = data['weights'][:n_samples].to(device)
    signatures = data['signatures'][:n_samples].to(device)
    labels = data['labels'][:n_samples]
    
    with torch.no_grad():
        recon, _, _, _ = model(weights, signatures)
    
    weights_np = weights.cpu().numpy()
    recon_np = recon.cpu().numpy()
    
    figures = {}
    
    # 1. Weight comparison heatmap (original vs reconstructed)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        pattern = IDX_TO_PATTERN.get(labels[i].item(), 'unknown')
        
        # Reshape weights to approximate square for visualization
        # 345 weights -> 15x23 grid
        orig = weights_np[i].reshape(15, 23)
        rec = recon_np[i].reshape(15, 23)
        diff = orig - rec
        
        vmax = max(abs(orig).max(), abs(rec).max())
        
        im0 = axes[i, 0].imshow(orig, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[i, 0].set_title(f'Original ({pattern})')
        axes[i, 0].set_ylabel(f'Sample {i}')
        plt.colorbar(im0, ax=axes[i, 0])
        
        im1 = axes[i, 1].imshow(rec, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[i, 1].set_title('Reconstructed')
        plt.colorbar(im1, ax=axes[i, 1])
        
        im2 = axes[i, 2].imshow(diff, cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title(f'Difference (MSE={np.mean(diff**2):.4f})')
        plt.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    figures['weight_comparison'] = fig
    
    # 2. Network output comparison on test inputs
    fig2, axes2 = plt.subplots(n_samples, 1, figsize=(10, 2.5 * n_samples))
    if n_samples == 1:
        axes2 = [axes2]
    
    # Create test inputs (range of digit sequences)
    test_inputs = torch.tensor([
        [0, 1, 2, 3, 4],  # ascending
        [4, 3, 2, 1, 0],  # descending
        [1, 2, 1, 2, 1],  # alternating
        [1, 2, 3, 2, 1],  # palindrome
        [0, 2, 4, 2, 0],  # mountain
        [5, 5, 5, 5, 5],  # constant
        [1, 1, 2, 2, 3],  # increasing pairs
        [3, 3, 2, 2, 1],  # decreasing pairs
        [1, 2, 3, 4, 1],  # first_last_match
        [1, 2, 3, 4, 5],  # no_repeats
    ], dtype=torch.float32)
    
    test_labels = ['asc', 'desc', 'alt', 'palin', 'mount', 'const', 'inc_p', 'dec_p', 'fl_m', 'no_rep']
    
    for i in range(n_samples):
        pattern = IDX_TO_PATTERN.get(labels[i].item(), 'unknown')
        
        # Create networks
        orig_net = SubjectNetwork.from_weights(weights[i].cpu())
        recon_net = SubjectNetwork.from_weights(recon[i].cpu())
        
        with torch.no_grad():
            orig_out = torch.sigmoid(orig_net(test_inputs)).numpy()
            recon_out = torch.sigmoid(recon_net(test_inputs)).numpy()
        
        x = np.arange(len(test_labels))
        width = 0.35
        
        axes2[i].bar(x - width/2, orig_out, width, label='Original', alpha=0.8)
        axes2[i].bar(x + width/2, recon_out, width, label='Reconstructed', alpha=0.8)
        axes2[i].set_ylabel('Output')
        axes2[i].set_title(f'Network outputs - trained on: {pattern}')
        axes2[i].set_xticks(x)
        axes2[i].set_xticklabels(test_labels, rotation=45)
        axes2[i].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes2[i].legend()
        axes2[i].set_ylim(0, 1)
    
    plt.tight_layout()
    figures['output_comparison'] = fig2
    
    # 3. Per-layer weight distribution comparison
    fig3, axes3 = plt.subplots(2, 3, figsize=(12, 8))
    
    # Weight indices for each layer (5 layers + 1 output = 6 linear layers)
    # Layer 0: 5*8 + 8 = 48 params (input)
    # Layers 1-4: 8*8 + 8 = 72 params each (hidden)
    # Layer 5: 8*1 + 1 = 9 params (output)
    layer_sizes = [48, 72, 72, 72, 72, 9]  # Total = 345
    layer_starts = [0]
    for s in layer_sizes[:-1]:
        layer_starts.append(layer_starts[-1] + s)
    
    # Aggregate across samples
    for layer_idx in range(6):
        ax = axes3[layer_idx // 3, layer_idx % 3]
        start = layer_starts[layer_idx]
        end = start + layer_sizes[layer_idx]
        
        orig_layer = weights_np[:, start:end].flatten()
        recon_layer = recon_np[:, start:end].flatten()
        
        ax.hist(orig_layer, bins=30, alpha=0.5, label='Original', density=True)
        ax.hist(recon_layer, bins=30, alpha=0.5, label='Reconstructed', density=True)
        ax.set_title(f'Layer {layer_idx} weights')
        ax.legend()
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Density')
    
    plt.tight_layout()
    figures['layer_distributions'] = fig3
    
    return figures


# =============================================================================
# Data Loading
# =============================================================================

def apply_deduplication(
    weights: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    fingerprints: List[Dict],
) -> Dict:
    """Deduplicate exact weight/signature rows before train/validation splitting."""
    keep_indices, summary = deduplicate_fingerprints(fingerprints)
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    return {
        "weights": weights[keep_tensor],
        "signatures": signatures[keep_tensor],
        "labels": labels[keep_tensor],
        "fingerprints": [fingerprints[i] for i in keep_indices],
        "deduplication": summary,
    }


def load_data(
    max_samples: Optional[int] = None,
    include_patterns: Optional[List[str]] = None,
) -> Dict:
    """Load dataset with weights and signatures from HuggingFace."""
    logger.info("Loading dataset from HuggingFace...")
    hf_ds = hf_load_dataset('maximuspowers/hypernet_validated', split='train')
    include_pattern_set = set(include_patterns) if include_patterns else None
    
    all_weights = []
    all_signatures = []
    all_labels = []
    fingerprints = []
    expected_weight_size = None
    
    # Get dataset length safely
    ds_len = len(hf_ds) if hasattr(hf_ds, '__len__') else 10000  # type: ignore
    
    for i in tqdm(range(ds_len), desc='Processing'):
        if max_samples and len(all_weights) >= max_samples:
            break
            
        sample = hf_ds[i]  # type: ignore
        pattern = sample['classification_completion']
        
        if pattern not in ALL_PATTERNS:
            continue
        if include_pattern_set is not None and pattern not in include_pattern_set:
            continue
        
        try:
            weights_data = json.loads(str(sample['improved_model_weights']))
            config = weights_data['config']
            
            arch = (config['num_layers'], config['neurons_per_layer'])
            if arch != TARGET_ARCH:
                continue
            
            # Flatten weights - MUST use numerical order, not alphabetical!
            # Keys like "network.0.weight" must come before "network.2.weight"
            def weight_key_order(key):
                # Extract layer number from keys like "network.0.weight" or "network.10.bias"
                parts = key.split('.')
                layer_num = int(parts[1])
                # Weights come before biases within same layer
                param_type = 0 if 'weight' in key else 1
                return (layer_num, param_type)
            
            flat_weights = []
            for key in sorted(weights_data['weights'].keys(), key=weight_key_order):
                w = weights_data['weights'][key]
                if isinstance(w[0], list):
                    for row in w:
                        flat_weights.extend(row)
                else:
                    flat_weights.extend(w)
            
            if expected_weight_size is None:
                expected_weight_size = len(flat_weights)
            if len(flat_weights) != expected_weight_size:
                continue
            
            # Extract signature
            sig_data = json.loads(str(sample['improved_signature']))
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
            
            max_sig_dim = 510
            sig_features = sig_features[:max_sig_dim]
            sig_features += [0] * (max_sig_dim - len(sig_features))
            
            all_weights.append(flat_weights)
            all_signatures.append(sig_features)
            label = PATTERN_TO_IDX[pattern]
            all_labels.append(label)
            fingerprints.append(
                make_sample_fingerprint(
                    row_index=i,
                    sample=dict(sample),
                    flat_weights=flat_weights,
                    signature_features=sig_features,
                    label=label,
                )
            )
            
        except Exception:
            continue
    
    weights_tensor = torch.tensor(all_weights, dtype=torch.float32)
    signatures_tensor = torch.tensor(all_signatures, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    deduped = apply_deduplication(
        weights_tensor,
        signatures_tensor,
        labels_tensor,
        fingerprints,
    )

    logger.info(
        "Loaded %s samples, %s after deduplication",
        len(all_weights),
        len(deduped["weights"]),
    )
    
    # Count per pattern
    label_counts = {}
    for l in deduped["labels"].tolist():
        p = IDX_TO_PATTERN[int(l)]
        label_counts[p] = label_counts.get(p, 0) + 1
    logger.info(f"Pattern distribution: {label_counts}")
    logger.info(f"Deduplication summary: {deduped['deduplication']}")

    dataset_provenance = {
        "dataset_id": "maximuspowers/hypernet_validated",
        "split": "train",
        "fingerprint": getattr(hf_ds, "_fingerprint", None),
        "include_patterns": sorted(include_pattern_set) if include_pattern_set else None,
        "source_count": len(all_weights),
        "deduplicated_count": int(len(deduped["weights"])),
        "deduplication": deduped["deduplication"],
        "row_indices": [
            int(fingerprint["row_index"])
            for fingerprint in deduped["fingerprints"]
        ],
        "row_hashes": [
            str(fingerprint["row_hash"])
            for fingerprint in deduped["fingerprints"]
        ],
        "weight_hashes": [
            str(fingerprint["weight_hash"])
            for fingerprint in deduped["fingerprints"]
        ],
        "signature_hashes": [
            str(fingerprint["signature_hash"])
            for fingerprint in deduped["fingerprints"]
        ],
        "probe_provenance": {
            "status": "not_embedded_in_hf_rows",
            "claim_scope": "fixed_signature_column",
        },
    }
    
    return {
        'weights': deduped["weights"],
        'signatures': deduped["signatures"],
        'labels': deduped["labels"],
        'include_patterns': sorted(include_pattern_set) if include_pattern_set else None,
        'fingerprints': deduped["fingerprints"],
        'dataset_provenance': dataset_provenance,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_reconstruction(
    model: FunctionalHyperNetwork,
    weights: torch.Tensor,
    signatures: torch.Tensor,
    n_samples: int = 50,
) -> Dict[str, float]:
    """Evaluate reconstruction quality."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        test_w = weights[:n_samples].to(device)
        test_s = signatures[:n_samples].to(device)
        
        recon, _, _, _ = model(test_w, test_s)
        cos_sim = F.cosine_similarity(recon, test_w, dim=1).mean().item()
        mse = F.mse_loss(recon, test_w).item()
    
    return {'cosine_similarity': cos_sim, 'mse': mse}


def evaluate_editing(
    model: FunctionalHyperNetwork,
    data: Dict,
    source_pattern: str = 'sorted_descending',
    target_pattern: str = 'sorted_ascending',
    n_samples: int = 10,
) -> Dict:
    """Evaluate behavior editing quality."""
    model.eval()
    editor = BehaviorEditor(model)
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    source_idx = PATTERN_TO_IDX.get(source_pattern)
    target_idx = PATTERN_TO_IDX.get(target_pattern)
    
    if source_idx is None or target_idx is None:
        return {'error': f'Unknown pattern: {source_pattern} or {target_pattern}'}
    
    source_mask = labels == source_idx
    target_mask = labels == target_idx
    
    if source_mask.sum() < n_samples or target_mask.sum() < n_samples:
        return {'error': f'Not enough samples for {source_pattern}->{target_pattern}'}
    
    # Get samples
    source_indices = torch.where(source_mask)[0][:n_samples]
    target_indices = torch.where(target_mask)[0][:n_samples]
    
    # Compute average target signature
    target_sig = signatures[target_indices].mean(0)
    
    results = {
        'source_pattern': source_pattern,
        'target_pattern': target_pattern,
        'original_correct': 0,
        'edited_correct_source': 0,
        'edited_correct_target': 0,
        'total': 0,
    }
    
    for idx in source_indices:
        orig_weights = weights[idx]
        source_sig = signatures[idx]
        
        # Test original
        orig_net = SubjectNetwork.from_weights(orig_weights)
        orig_result = test_behavior(orig_net, source_pattern)
        
        if not orig_result['supported']:
            continue
            
        results['total'] += 1
        if orig_result['correct']:
            results['original_correct'] += 1
        
        # Edit toward target
        edited_net = editor.create_edited_network(
            orig_weights, source_sig, target_sig
        )
        
        source_result = test_behavior(edited_net, source_pattern)
        target_result = test_behavior(edited_net, target_pattern)
        
        if source_result['supported'] and source_result['correct']:
            results['edited_correct_source'] += 1
        if target_result['supported'] and target_result['correct']:
            results['edited_correct_target'] += 1
    
    return results


def evaluate_all_editing_pairs(
    model: FunctionalHyperNetwork,
    data: Dict,
    n_samples: int = 10,
) -> List[Dict]:
    """Evaluate editing on multiple pattern pairs."""
    # Test pairs that make semantic sense
    test_pairs = [
        ('sorted_descending', 'sorted_ascending'),
        ('sorted_ascending', 'sorted_descending'),
        ('increasing_pairs', 'decreasing_pairs'),
        ('decreasing_pairs', 'increasing_pairs'),
    ]
    
    results = []
    for source, target in test_pairs:
        result = evaluate_editing(model, data, source, target, n_samples)
        if 'error' not in result:
            results.append(result)
    
    return results


# =============================================================================
# Training
# =============================================================================

class HyperNetTrainer:
    """Trainer with TensorBoard logging and comprehensive evaluation."""
    
    def __init__(
        self,
        model: FunctionalHyperNetwork,
        data: Dict,
        run_dir: Path,
        use_tensorboard: bool = True,
        auto_launch_tensorboard: bool = False,
        tensorboard_port: int = 6006,
    ):
        self.model = model
        self.data = data
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_process = None
        
        self.writer = None
        if use_tensorboard and HAS_TENSORBOARD and SummaryWriter is not None:
            tb_log_dir = str(run_dir / 'tensorboard')
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard logging to {tb_log_dir}")
            
            if auto_launch_tensorboard:
                try:
                    self.tb_process = subprocess.Popen(
                        ['tensorboard', '--logdir', tb_log_dir, '--port', str(tensorboard_port)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    logger.info(f"TensorBoard started at http://localhost:{tensorboard_port}")
                    webbrowser.open(f'http://localhost:{tensorboard_port}')
                except Exception as e:
                    logger.warning(f"Failed to auto-launch TensorBoard: {e}")
    
    def train(
        self,
        epochs: int = 150,
        batch_size: int = 64,
        lr: float = 1e-3,
        lambda_kl: float = 0.1,
        lambda_functional: float = 0.5,
        use_functional_loss: bool = False,
        functional_loss_start_epoch: int = 50,
        device: str = 'auto',
        eval_every: int = 10,
        early_stopping_patience: Optional[int] = None,
        behavior_cases: Optional[Dict[str, Dict[str, List[List[int]]]]] = None,
    ) -> Dict:
        """Train the model with logging and evaluation."""
        
        weights = self.data['weights']
        signatures = self.data['signatures']
        labels = self.data['labels']
        
        def callback(epoch: int, metrics: Dict):
            if self.writer:
                for k, v in metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, epoch)
            
            # Periodic evaluation
            if (epoch + 1) % eval_every == 0:
                recon_metrics = evaluate_reconstruction(
                    self.model, weights, signatures
                )
                if self.writer:
                    for k, v in recon_metrics.items():
                        self.writer.add_scalar(f'eval/{k}', v, epoch)
                
                # Detailed per-pattern behavior evaluation
                pattern_metrics = evaluate_pattern_behaviors(
                    self.model, self.data, n_samples_per_pattern=5
                )
                if self.writer:
                    for pattern, pdata in pattern_metrics.items():
                        # Own pattern performance
                        self.writer.add_scalar(
                            f'behavior/{pattern}/own_positive', 
                            pdata['own_positive'], epoch
                        )
                        self.writer.add_scalar(
                            f'behavior/{pattern}/own_negative', 
                            pdata['own_negative'], epoch
                        )
                        self.writer.add_scalar(
                            f'behavior/{pattern}/own_margin', 
                            pdata['own_margin'], epoch
                        )
                        self.writer.add_scalar(
                            f'behavior/{pattern}/own_accuracy', 
                            pdata['own_correct'], epoch
                        )
                        
                        # Cross-pattern responses (how this pattern's networks respond to other patterns)
                        for other_pattern, cross_data in pdata.get('cross_pattern', {}).items():
                            self.writer.add_scalar(
                                f'cross/{pattern}_on_{other_pattern}/positive',
                                cross_data['positive'], epoch
                            )
                            self.writer.add_scalar(
                                f'cross/{pattern}_on_{other_pattern}/negative',
                                cross_data['negative'], epoch
                            )
                            self.writer.add_scalar(
                                f'cross/{pattern}_on_{other_pattern}/margin',
                                cross_data['margin'], epoch
                            )
                
                # Generate weight heatmaps every 20 epochs
                if self.writer and (epoch + 1) % 20 == 0:
                    try:
                        heatmap_figs = create_weight_heatmaps(self.model, self.data, n_samples=4)
                        for name, fig in heatmap_figs.items():
                            self.writer.add_figure(f'heatmaps/{name}', fig, epoch)
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Failed to create heatmaps: {e}")
        
        history = self.model.fit(
            weights, signatures, labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_kl=lambda_kl,
            lambda_functional=lambda_functional,
            use_functional_loss=use_functional_loss,
            functional_loss_start_epoch=functional_loss_start_epoch,
            device=device,
            verbose=True,
            callback=callback,
            early_stopping_patience=early_stopping_patience,
            behavior_cases=behavior_cases,
        )
        
        # Final evaluation
        logger.info("Running final evaluation...")
        
        recon_metrics = evaluate_reconstruction(self.model, weights, signatures)
        logger.info(f"Reconstruction - Cosine: {recon_metrics['cosine_similarity']:.4f}, "
                   f"MSE: {recon_metrics['mse']:.4f}")
        
        edit_results = evaluate_all_editing_pairs(self.model, self.data)
        for r in edit_results:
            logger.info(f"Editing {r['source_pattern']} -> {r['target_pattern']}: "
                       f"Original {r['original_correct']}/{r['total']}, "
                       f"Edited to target {r['edited_correct_target']}/{r['total']}")
        
        # Save model
        model_path = self.run_dir / 'model.pt'
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save results
        results = {
            'reconstruction': recon_metrics,
            'editing': edit_results,
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        }
        
        results_path = self.run_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return results


def build_hypernet_config(
    weight_dim: int,
    sig_dim: int,
    latent_dim: int,
    condition_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_kl: float,
    lambda_functional: float,
    lambda_condition_specificity: float,
    lambda_calibrated_behavior_margin: float,
    matched_behavior_min_margin: float,
    matched_mountain_target_weight: float,
    lambda_control_behavior_penalty: float,
    lambda_control_hard_negative_penalty: float,
    control_max_allowed_margin: float,
    train_centroid_control_weight: float,
    condition_ablation_control_weight: float,
    noise_control_weight: float,
    shuffled_control_weight: float,
    control_sorted_descending_target_weight: float,
    control_has_majority_target_weight: float,
    sorted_descending_specificity_weight: float,
    lambda_edit_behavior: float,
    lambda_edit_margin_delta: float,
    use_condition_residual_decoder: bool,
    condition_residual_scale: float,
    lambda_shuffled_residual_contrastive: float,
    shuffled_residual_min_delta: float,
    functional_loss_start_epoch: int,
    functional_loss_samples: int,
) -> HyperNetConfig:
    """Build the auditable model config used by the training entrypoint."""
    return HyperNetConfig(
        weight_dim=weight_dim,
        sig_dim=sig_dim,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_kl=lambda_kl,
        lambda_functional=lambda_functional,
        lambda_condition_specificity=lambda_condition_specificity,
        lambda_calibrated_behavior_margin=lambda_calibrated_behavior_margin,
        matched_behavior_min_margin=matched_behavior_min_margin,
        matched_mountain_target_weight=matched_mountain_target_weight,
        lambda_control_behavior_penalty=lambda_control_behavior_penalty,
        lambda_control_hard_negative_penalty=lambda_control_hard_negative_penalty,
        control_max_allowed_margin=control_max_allowed_margin,
        train_centroid_control_weight=train_centroid_control_weight,
        condition_ablation_control_weight=condition_ablation_control_weight,
        noise_control_weight=noise_control_weight,
        shuffled_control_weight=shuffled_control_weight,
        control_sorted_descending_target_weight=control_sorted_descending_target_weight,
        control_has_majority_target_weight=control_has_majority_target_weight,
        sorted_descending_specificity_weight=sorted_descending_specificity_weight,
        lambda_edit_behavior=lambda_edit_behavior,
        lambda_edit_margin_delta=lambda_edit_margin_delta,
        use_condition_residual_decoder=use_condition_residual_decoder,
        condition_residual_scale=condition_residual_scale,
        lambda_shuffled_residual_contrastive=lambda_shuffled_residual_contrastive,
        shuffled_residual_min_delta=shuffled_residual_min_delta,
        functional_loss_start_epoch=functional_loss_start_epoch,
        functional_loss_samples=functional_loss_samples,
    )


def train(
    config_path: Optional[str] = None,
    epochs: int = 150,
    batch_size: int = 64,
    latent_dim: int = 128,  # Increased for more capacity
    condition_dim: int = 128,
    hidden_dim: int = 512,  # Increased for more capacity
    lr: float = 1e-3,
    lambda_kl: float = 0.01,  # Reduced to allow more reconstruction freedom
    lambda_functional: float = 10.0,  # Increased to prioritize functional behavior!
    lambda_condition_specificity: float = 1.0,
    lambda_calibrated_behavior_margin: float = 0.0,
    matched_behavior_min_margin: float = 0.02,
    matched_mountain_target_weight: float = 1.0,
    lambda_control_behavior_penalty: float = 1.0,
    lambda_control_hard_negative_penalty: float = 0.0,
    control_max_allowed_margin: float = -0.05,
    train_centroid_control_weight: float = 3.0,
    condition_ablation_control_weight: float = 1.0,
    noise_control_weight: float = 1.0,
    shuffled_control_weight: float = 1.0,
    control_sorted_descending_target_weight: float = 1.0,
    control_has_majority_target_weight: float = 1.0,
    sorted_descending_specificity_weight: float = 2.0,
    lambda_edit_behavior: float = 1.0,
    lambda_edit_margin_delta: float = 1.0,
    use_condition_residual_decoder: bool = False,
    condition_residual_scale: float = 1.0,
    lambda_shuffled_residual_contrastive: float = 0.0,
    shuffled_residual_min_delta: float = 0.05,
    use_functional_loss: bool = False,
    functional_loss_start_epoch: int = 0,  # Start immediately
    functional_loss_samples: int = 16,
    device: str = 'auto',
    max_samples: Optional[int] = None,
    run_dir: Optional[str] = None,
    use_tensorboard: bool = True,
    auto_launch_tensorboard: bool = False,
    tensorboard_port: int = 6006,
    early_stopping_patience: Optional[int] = None,
    include_patterns: Optional[List[str]] = None,
):
    """Train the FunctionalHyperNetwork."""
    
    # Load config if provided
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        epochs = cfg.get('epochs', epochs)
        batch_size = cfg.get('batch_size', batch_size)
        latent_dim = cfg.get('latent_dim', latent_dim)
        condition_dim = cfg.get('condition_dim', condition_dim)
        hidden_dim = cfg.get('hidden_dim', hidden_dim)
        lr = cfg.get('lr', lr)
        lambda_kl = cfg.get('lambda_kl', lambda_kl)
        lambda_functional = cfg.get('lambda_functional', lambda_functional)
        lambda_condition_specificity = cfg.get(
            'lambda_condition_specificity',
            lambda_condition_specificity,
        )
        lambda_calibrated_behavior_margin = cfg.get(
            'lambda_calibrated_behavior_margin',
            lambda_calibrated_behavior_margin,
        )
        matched_behavior_min_margin = cfg.get(
            'matched_behavior_min_margin',
            matched_behavior_min_margin,
        )
        matched_mountain_target_weight = cfg.get(
            'matched_mountain_target_weight',
            matched_mountain_target_weight,
        )
        lambda_control_behavior_penalty = cfg.get(
            'lambda_control_behavior_penalty',
            lambda_control_behavior_penalty,
        )
        lambda_control_hard_negative_penalty = cfg.get(
            'lambda_control_hard_negative_penalty',
            lambda_control_hard_negative_penalty,
        )
        control_max_allowed_margin = cfg.get(
            'control_max_allowed_margin',
            control_max_allowed_margin,
        )
        train_centroid_control_weight = cfg.get(
            'train_centroid_control_weight',
            train_centroid_control_weight,
        )
        condition_ablation_control_weight = cfg.get(
            'condition_ablation_control_weight',
            condition_ablation_control_weight,
        )
        noise_control_weight = cfg.get(
            'noise_control_weight',
            noise_control_weight,
        )
        shuffled_control_weight = cfg.get(
            'shuffled_control_weight',
            shuffled_control_weight,
        )
        control_sorted_descending_target_weight = cfg.get(
            'control_sorted_descending_target_weight',
            control_sorted_descending_target_weight,
        )
        control_has_majority_target_weight = cfg.get(
            'control_has_majority_target_weight',
            control_has_majority_target_weight,
        )
        sorted_descending_specificity_weight = cfg.get(
            'sorted_descending_specificity_weight',
            sorted_descending_specificity_weight,
        )
        lambda_edit_behavior = cfg.get('lambda_edit_behavior', lambda_edit_behavior)
        lambda_edit_margin_delta = cfg.get(
            'lambda_edit_margin_delta',
            lambda_edit_margin_delta,
        )
        use_condition_residual_decoder = cfg.get(
            'use_condition_residual_decoder',
            use_condition_residual_decoder,
        )
        condition_residual_scale = cfg.get(
            'condition_residual_scale',
            condition_residual_scale,
        )
        lambda_shuffled_residual_contrastive = cfg.get(
            'lambda_shuffled_residual_contrastive',
            lambda_shuffled_residual_contrastive,
        )
        shuffled_residual_min_delta = cfg.get(
            'shuffled_residual_min_delta',
            shuffled_residual_min_delta,
        )
        use_functional_loss = cfg.get('use_functional_loss', use_functional_loss)
        functional_loss_start_epoch = cfg.get('functional_loss_start_epoch', functional_loss_start_epoch)
        functional_loss_samples = cfg.get(
            'functional_loss_samples',
            functional_loss_samples,
        )
        device = cfg.get('device', device)
        max_samples = cfg.get('max_samples', max_samples)
        include_patterns = cfg.get('include_patterns', include_patterns)
        if isinstance(include_patterns, str):
            include_patterns = [p.strip() for p in include_patterns.split(',') if p.strip()]
        use_tensorboard = cfg.get('use_tensorboard', use_tensorboard)
    
    # Load data
    if include_patterns is None and use_functional_loss:
        include_patterns = list(CLEAN_PROOF_PATTERNS)
    data = load_data(max_samples=max_samples, include_patterns=include_patterns)
    
    weights = data['weights']
    signatures = data['signatures']
    
    weight_dim = weights.shape[1]
    sig_dim = signatures.shape[1]
    
    logger.info(f"Data: {len(weights)} samples, {weight_dim} weight dims, {sig_dim} sig dims")
    
    # Create model
    config = build_hypernet_config(
        weight_dim=weight_dim,
        sig_dim=sig_dim,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_kl=lambda_kl,
        lambda_functional=lambda_functional,
        lambda_condition_specificity=lambda_condition_specificity,
        lambda_calibrated_behavior_margin=lambda_calibrated_behavior_margin,
        matched_behavior_min_margin=matched_behavior_min_margin,
        matched_mountain_target_weight=matched_mountain_target_weight,
        lambda_control_behavior_penalty=lambda_control_behavior_penalty,
        lambda_control_hard_negative_penalty=lambda_control_hard_negative_penalty,
        control_max_allowed_margin=control_max_allowed_margin,
        train_centroid_control_weight=train_centroid_control_weight,
        condition_ablation_control_weight=condition_ablation_control_weight,
        noise_control_weight=noise_control_weight,
        shuffled_control_weight=shuffled_control_weight,
        control_sorted_descending_target_weight=control_sorted_descending_target_weight,
        control_has_majority_target_weight=control_has_majority_target_weight,
        sorted_descending_specificity_weight=sorted_descending_specificity_weight,
        lambda_edit_behavior=lambda_edit_behavior,
        lambda_edit_margin_delta=lambda_edit_margin_delta,
        use_condition_residual_decoder=use_condition_residual_decoder,
        condition_residual_scale=condition_residual_scale,
        lambda_shuffled_residual_contrastive=lambda_shuffled_residual_contrastive,
        shuffled_residual_min_delta=shuffled_residual_min_delta,
        functional_loss_start_epoch=functional_loss_start_epoch,
        functional_loss_samples=functional_loss_samples,
    )
    model = FunctionalHyperNetwork(config=config)
    model._dataset_patterns = data.get('include_patterns')
    model._dataset_provenance = data.get('dataset_provenance')
    behavior_suite = build_clean_behavior_suite()
    model._behavior_suite_metadata = behavior_suite["metadata"]
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_path = Path(__file__).parent.parent.parent / "runs" / f"hypernet_{timestamp}"
    else:
        run_dir_path = Path(run_dir)
    
    # Train
    trainer = HyperNetTrainer(
        model, data, run_dir_path,
        use_tensorboard=use_tensorboard,
        auto_launch_tensorboard=auto_launch_tensorboard,
        tensorboard_port=tensorboard_port,
    )
    results = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_kl=lambda_kl,
        lambda_functional=lambda_functional,
        use_functional_loss=use_functional_loss,
        functional_loss_start_epoch=functional_loss_start_epoch,
        device=device,
        early_stopping_patience=early_stopping_patience,
        behavior_cases=behavior_cases_for_training(behavior_suite),
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir_path}")
    logger.info(f"Final reconstruction cosine: {results['reconstruction']['cosine_similarity']:.4f}")
    
    success_count = sum(1 for r in results['editing'] if r['edited_correct_target'] > r['total'] // 2)
    logger.info(f"Editing success: {success_count}/{len(results['editing'])} pairs")
    
    return model, results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train FunctionalHyperNetwork for behavioral weight generation"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dimension (default: 64)"
    )
    parser.add_argument(
        "--condition-dim",
        type=int,
        default=128,
        help="Conditioning dimension (default: 128)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension (default: 256)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.1,
        help="KL divergence weight (default: 0.1)"
    )
    parser.add_argument(
        "--lambda-functional",
        type=float,
        default=0.5,
        help="Functional loss weight (default: 0.5)"
    )
    parser.add_argument(
        "--lambda-condition-specificity",
        type=float,
        default=1.0,
        help="Condition-specificity loss weight (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-calibrated-behavior-margin",
        type=float,
        default=0.0,
        help="Matched calibrated sigmoid-margin loss weight (default: 0.0)"
    )
    parser.add_argument(
        "--matched-behavior-min-margin",
        type=float,
        default=0.02,
        help="Minimum calibrated matched behavior margin (default: 0.02)"
    )
    parser.add_argument(
        "--matched-mountain-target-weight",
        type=float,
        default=1.0,
        help="Calibrated margin weight for mountain_pattern (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-control-behavior-penalty",
        type=float,
        default=1.0,
        help="Control behavior-prior penalty weight (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-control-hard-negative-penalty",
        type=float,
        default=0.0,
        help="Worst-target control penalty weight (default: 0.0)"
    )
    parser.add_argument(
        "--control-max-allowed-margin",
        type=float,
        default=-0.05,
        help="Maximum allowed sigmoid margin for control decodes (default: -0.05)"
    )
    parser.add_argument(
        "--train-centroid-control-weight",
        type=float,
        default=3.0,
        help="Extra weight for train-centroid zero-latent control penalty (default: 3.0)"
    )
    parser.add_argument(
        "--condition-ablation-control-weight",
        type=float,
        default=1.0,
        help="Extra weight for zero-condition control penalty (default: 1.0)"
    )
    parser.add_argument(
        "--control-sorted-descending-target-weight",
        type=float,
        default=1.0,
        help="All-target control loss weight for sorted_descending (default: 1.0)"
    )
    parser.add_argument(
        "--control-has-majority-target-weight",
        type=float,
        default=1.0,
        help="All-target control loss weight for has_majority (default: 1.0)"
    )
    parser.add_argument(
        "--sorted-descending-specificity-weight",
        type=float,
        default=2.0,
        help="Subject-specificity weight for sorted_descending samples (default: 2.0)"
    )
    parser.add_argument(
        "--lambda-edit-behavior",
        type=float,
        default=1.0,
        help="Edit-path target behavior loss weight (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-edit-margin-delta",
        type=float,
        default=1.0,
        help="Edit-path target margin-delta loss weight (default: 1.0)"
    )
    parser.add_argument(
        "--use-functional-loss",
        action="store_true",
        help="Enable functional loss during training"
    )
    parser.add_argument(
        "--functional-loss-start-epoch",
        type=int,
        default=50,
        help="Epoch to start functional loss (default: 50)"
    )
    parser.add_argument(
        "--functional-loss-samples",
        type=int,
        default=16,
        help="Number of functional probes/samples used in losses (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on (default: auto)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (for testing)"
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default=None,
        help="Comma-separated behavior patterns to include"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to save run outputs"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--auto-launch-tensorboard",
        action="store_true",
        help="Auto-launch TensorBoard and open in browser"
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="Port for TensorBoard (default: 6006)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    train(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        lambda_kl=args.lambda_kl,
        lambda_functional=args.lambda_functional,
        lambda_condition_specificity=args.lambda_condition_specificity,
        lambda_calibrated_behavior_margin=args.lambda_calibrated_behavior_margin,
        matched_behavior_min_margin=args.matched_behavior_min_margin,
        matched_mountain_target_weight=args.matched_mountain_target_weight,
        lambda_control_behavior_penalty=args.lambda_control_behavior_penalty,
        lambda_control_hard_negative_penalty=args.lambda_control_hard_negative_penalty,
        control_max_allowed_margin=args.control_max_allowed_margin,
        train_centroid_control_weight=args.train_centroid_control_weight,
        condition_ablation_control_weight=args.condition_ablation_control_weight,
        control_sorted_descending_target_weight=args.control_sorted_descending_target_weight,
        control_has_majority_target_weight=args.control_has_majority_target_weight,
        sorted_descending_specificity_weight=args.sorted_descending_specificity_weight,
        lambda_edit_behavior=args.lambda_edit_behavior,
        lambda_edit_margin_delta=args.lambda_edit_margin_delta,
        use_functional_loss=args.use_functional_loss,
        functional_loss_start_epoch=args.functional_loss_start_epoch,
        functional_loss_samples=args.functional_loss_samples,
        device=args.device,
        max_samples=args.max_samples,
        include_patterns=[p.strip() for p in args.patterns.split(',')] if args.patterns else None,
        run_dir=args.run_dir,
        use_tensorboard=not args.no_tensorboard,
        auto_launch_tensorboard=args.auto_launch_tensorboard,
        tensorboard_port=args.tensorboard_port,
    )


if __name__ == "__main__":
    main()
