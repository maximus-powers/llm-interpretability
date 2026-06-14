"""
Reconstruction quality metrics.

Computes:
- Weight-level: MSE, cosine similarity
- Functional: Output MSE on probe inputs
- Behavioral: Does reconstructed network still classify correctly?
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PatternReconstructionMetrics:
    """Reconstruction metrics for a single pattern."""
    pattern_name: str
    n_samples: int
    weight_cosine_mean: float
    weight_cosine_std: float
    weight_mse_mean: float
    weight_mse_std: float
    functional_mse_mean: float
    functional_mse_std: float
    behavioral_accuracy: float
    margin_mean: float
    margin_std: float
    margin_min: float
    margin_max: float


@dataclass
class ReconstructionMetrics:
    """Aggregated reconstruction metrics."""
    overall_weight_cosine: float
    overall_weight_mse: float
    overall_functional_mse: float
    overall_behavioral_accuracy: float
    overall_margin_mean: float
    n_samples: int
    n_patterns: int
    per_pattern: Dict[str, PatternReconstructionMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'overall': {
                'weight_cosine': self.overall_weight_cosine,
                'weight_mse': self.overall_weight_mse,
                'functional_mse': self.overall_functional_mse,
                'behavioral_accuracy': self.overall_behavioral_accuracy,
                'margin_mean': self.overall_margin_mean,
                'n_samples': self.n_samples,
                'n_patterns': self.n_patterns,
            },
            'per_pattern': {
                name: {
                    'n_samples': m.n_samples,
                    'weight_cosine_mean': m.weight_cosine_mean,
                    'weight_mse_mean': m.weight_mse_mean,
                    'functional_mse_mean': m.functional_mse_mean,
                    'behavioral_accuracy': m.behavioral_accuracy,
                    'margin_mean': m.margin_mean,
                    'margin_min': m.margin_min,
                    'margin_max': m.margin_max,
                }
                for name, m in self.per_pattern.items()
            }
        }


def compute_reconstruction_metrics(
    model,
    weights: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    idx_to_pattern: Dict[int, str],
    test_behavior_fn,
    subject_network_cls,
    n_probes: int = 30,
) -> ReconstructionMetrics:
    """
    Compute comprehensive reconstruction metrics.
    
    Args:
        model: FunctionalHyperNetwork model
        weights: Test set weights [N, weight_dim]
        signatures: Test set signatures [N, sig_dim]
        labels: Test set pattern labels [N]
        idx_to_pattern: Mapping from label index to pattern name
        test_behavior_fn: Function to test network behavior
        subject_network_cls: SubjectNetwork class
        n_probes: Number of probe inputs for functional loss
    
    Returns:
        ReconstructionMetrics dataclass
    """
    model.eval()
    device = next(model.parameters()).device
    
    weights_d = weights.to(device)
    signatures_d = signatures.to(device)
    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)
    
    logger.info(f"Computing reconstruction metrics for {len(weights)} samples")
    
    # Generate probe inputs for functional comparison
    torch.manual_seed(42)  # Deterministic probes
    probes = torch.randint(0, 10, (n_probes, 5)).float().to(device)
    
    # Collect per-sample metrics
    all_cosines = []
    all_mses = []
    all_func_mses = []
    all_behavioral_correct = []
    all_margins = []
    
    # Per-pattern collections
    pattern_metrics = {int(l): {
        'cosines': [], 'mses': [], 'func_mses': [], 
        'behavioral_correct': [], 'margins': []
    } for l in unique_labels}
    
    # Process in batches for efficiency
    batch_size = 64
    n_batches = (len(weights) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Reconstruction eval"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(weights))
        
        w_batch = weights_d[start:end]
        s_batch = signatures_d[start:end]
        l_batch = labels_np[start:end]
        
        with torch.no_grad():
            recon, _, _, _ = model(w_batch, s_batch)
        
        # Per-sample metrics
        for i in range(len(w_batch)):
            orig_w = w_batch[i]
            recon_w = recon[i]
            label = int(l_batch[i])
            pattern = idx_to_pattern.get(label, f'pattern_{label}')
            
            # Weight cosine similarity
            cosine = F.cosine_similarity(
                orig_w.unsqueeze(0), recon_w.unsqueeze(0)
            ).item()
            
            # Weight MSE
            mse = F.mse_loss(recon_w, orig_w).item()
            
            # Functional MSE (output comparison on probes)
            orig_net = subject_network_cls.from_weights(orig_w.cpu())
            recon_net = subject_network_cls.from_weights(recon_w.cpu())
            
            with torch.no_grad():
                orig_out = orig_net(probes.cpu())
                recon_out = recon_net(probes.cpu())
            func_mse = F.mse_loss(recon_out, orig_out).item()
            
            # Behavioral test
            behavior_result = test_behavior_fn(recon_net, pattern)
            is_correct = behavior_result.get('correct', False) if behavior_result.get('supported', False) else False
            margin = behavior_result.get('margin', 0.0)
            
            # Store metrics
            all_cosines.append(cosine)
            all_mses.append(mse)
            all_func_mses.append(func_mse)
            all_behavioral_correct.append(is_correct)
            all_margins.append(margin)
            
            pattern_metrics[label]['cosines'].append(cosine)
            pattern_metrics[label]['mses'].append(mse)
            pattern_metrics[label]['func_mses'].append(func_mse)
            pattern_metrics[label]['behavioral_correct'].append(is_correct)
            pattern_metrics[label]['margins'].append(margin)
    
    # Aggregate per-pattern
    per_pattern_results = {}
    for label, metrics in pattern_metrics.items():
        pattern_name = idx_to_pattern.get(label, f'pattern_{label}')
        if len(metrics['cosines']) == 0:
            continue
        
        cosines = np.array(metrics['cosines'])
        mses = np.array(metrics['mses'])
        func_mses = np.array(metrics['func_mses'])
        margins = np.array(metrics['margins'])
        
        per_pattern_results[pattern_name] = PatternReconstructionMetrics(
            pattern_name=pattern_name,
            n_samples=len(cosines),
            weight_cosine_mean=float(np.mean(cosines)),
            weight_cosine_std=float(np.std(cosines)),
            weight_mse_mean=float(np.mean(mses)),
            weight_mse_std=float(np.std(mses)),
            functional_mse_mean=float(np.mean(func_mses)),
            functional_mse_std=float(np.std(func_mses)),
            behavioral_accuracy=float(np.mean(metrics['behavioral_correct'])),
            margin_mean=float(np.mean(margins)),
            margin_std=float(np.std(margins)),
            margin_min=float(np.min(margins)),
            margin_max=float(np.max(margins)),
        )
    
    # Overall metrics
    return ReconstructionMetrics(
        overall_weight_cosine=float(np.mean(all_cosines)),
        overall_weight_mse=float(np.mean(all_mses)),
        overall_functional_mse=float(np.mean(all_func_mses)),
        overall_behavioral_accuracy=float(np.mean(all_behavioral_correct)),
        overall_margin_mean=float(np.mean(all_margins)),
        n_samples=len(all_cosines),
        n_patterns=len(unique_labels),
        per_pattern=per_pattern_results,
    )
