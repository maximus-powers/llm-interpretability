"""
Editing quality metrics.

Computes:
- Full NxN pattern editing matrix
- Success rates at different thresholds
- Cross-pattern response (after editing to X, how does it respond to all patterns?)
- Optimal threshold finding
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EditingPairMetrics:
    """Metrics for a single source→target editing pair."""
    source_pattern: str
    target_pattern: str
    n_samples: int
    
    # Success at threshold 0.5
    success_rate_05: float
    target_pos_acc_05: float
    target_neg_acc_05: float
    
    # Success at optimal threshold
    optimal_threshold: float
    success_rate_optimal: float
    target_pos_acc_optimal: float
    target_neg_acc_optimal: float
    
    # Margin statistics
    margin_mean: float
    margin_std: float
    margin_min: float
    margin_max: float
    margin_success_rate: float
    
    # Cross-pattern response (edited network tested on all patterns)
    cross_pattern_response: Dict[str, float] = field(default_factory=dict)


@dataclass
class EditingMetrics:
    """Aggregated editing metrics."""
    n_pairs: int
    n_samples_total: int
    
    # Overall success rates
    overall_success_rate_05: float
    overall_success_rate_optimal: float
    global_optimal_threshold: float
    overall_margin_success_rate: float
    
    # Matrix of success rates [source x target]
    success_matrix_05: Dict[str, Dict[str, float]] = field(default_factory=dict)
    success_matrix_optimal: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-pair detailed metrics
    pair_metrics: List[EditingPairMetrics] = field(default_factory=list)
    
    # Best and worst pairs
    best_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    worst_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'overall': {
                'n_pairs': self.n_pairs,
                'n_samples_total': self.n_samples_total,
                'success_rate_05': self.overall_success_rate_05,
                'success_rate_optimal': self.overall_success_rate_optimal,
                'global_optimal_threshold': self.global_optimal_threshold,
                'margin_success_rate': self.overall_margin_success_rate,
            },
            'success_matrix_05': self.success_matrix_05,
            'success_matrix_optimal': self.success_matrix_optimal,
            'best_pairs': [{'source': s, 'target': t, 'rate': r} for s, t, r in self.best_pairs],
            'worst_pairs': [{'source': s, 'target': t, 'rate': r} for s, t, r in self.worst_pairs],
            'pair_details': [
                {
                    'source': p.source_pattern,
                    'target': p.target_pattern,
                    'n_samples': p.n_samples,
                    'success_rate_05': p.success_rate_05,
                    'success_rate_optimal': p.success_rate_optimal,
                    'optimal_threshold': p.optimal_threshold,
                    'target_pos_acc_05': p.target_pos_acc_05,
                    'target_neg_acc_05': p.target_neg_acc_05,
                    'target_pos_acc_optimal': p.target_pos_acc_optimal,
                    'target_neg_acc_optimal': p.target_neg_acc_optimal,
                    'margin_mean': p.margin_mean,
                    'margin_std': p.margin_std,
                    'margin_min': p.margin_min,
                    'margin_max': p.margin_max,
                    'margin_success_rate': p.margin_success_rate,
                    'cross_pattern_response': p.cross_pattern_response,
                }
                for p in self.pair_metrics
            ]
        }


def find_optimal_threshold(
    positive_outputs: np.ndarray,
    negative_outputs: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Find optimal threshold that maximizes accuracy.
    
    Returns:
        (optimal_threshold, pos_accuracy, neg_accuracy, total_accuracy)
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    best_acc = 0.0
    best_thresh = 0.5
    best_pos_acc = 0.0
    best_neg_acc = 0.0
    
    for thresh in thresholds:
        pos_acc = (positive_outputs > thresh).mean()
        neg_acc = (negative_outputs < thresh).mean()
        total_acc = (pos_acc + neg_acc) / 2
        
        if total_acc > best_acc:
            best_acc = total_acc
            best_thresh = thresh
            best_pos_acc = pos_acc
            best_neg_acc = neg_acc
    
    return best_thresh, best_pos_acc, best_neg_acc, best_acc


def threshold_success_rate(
    positive_outputs: np.ndarray,
    negative_outputs: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of edited networks whose target positives and negatives both pass."""
    return float(((positive_outputs > threshold) & (negative_outputs < threshold)).mean())


def compute_editing_metrics(
    model,
    editor,
    weights: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    idx_to_pattern: Dict[int, str],
    pattern_to_idx: Dict[str, int],
    test_behavior_fn,
    get_test_cases_fn,
    subject_network_cls,
) -> EditingMetrics:
    """
    Compute comprehensive editing metrics.
    
    Args:
        model: FunctionalHyperNetwork model
        editor: BehaviorEditor instance
        weights: Test set weights [N, weight_dim]
        signatures: Test set signatures [N, sig_dim]
        labels: Test set pattern labels [N]
        idx_to_pattern: Mapping from label index to pattern name
        pattern_to_idx: Mapping from pattern name to label index
        test_behavior_fn: Function to test network behavior
        get_test_cases_fn: Function to get test cases for a pattern
        subject_network_cls: SubjectNetwork class
    
    Returns:
        EditingMetrics dataclass
    """
    model.eval()
    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)
    patterns_in_data = [idx_to_pattern[int(l)] for l in unique_labels]
    
    logger.info(f"Computing editing metrics for {len(patterns_in_data)} patterns")
    logger.info(f"Total pairs: {len(patterns_in_data) * (len(patterns_in_data) - 1)}")
    
    all_pair_metrics = []
    all_margins = []
    all_pos_outputs = []
    all_neg_outputs = []
    
    success_matrix_05 = {p: {} for p in patterns_in_data}
    success_matrix_optimal = {p: {} for p in patterns_in_data}
    
    # For each source pattern
    for source_pattern in tqdm(patterns_in_data, desc="Editing eval (source)"):
        source_idx = pattern_to_idx[source_pattern]
        source_mask = labels_np == source_idx
        source_indices = np.where(source_mask)[0]
        
        if len(source_indices) == 0:
            continue
        
        # For each target pattern
        for target_pattern in patterns_in_data:
            if target_pattern == source_pattern:
                continue
            
            target_idx = pattern_to_idx[target_pattern]
            target_mask = labels_np == target_idx
            target_indices = np.where(target_mask)[0]
            
            if len(target_indices) == 0:
                continue
            
            # Get test cases for target pattern
            test_cases = get_test_cases_fn(target_pattern)
            if test_cases is None:
                logger.warning(f"No test cases for {target_pattern}")
                continue
            
            pos_inputs = torch.tensor(test_cases['positive'], dtype=torch.float32)
            neg_inputs = torch.tensor(test_cases['negative'], dtype=torch.float32)
            
            # Compute target signature centroid
            target_sigs = signatures[target_indices]
            target_sig = target_sigs.mean(0)
            
            # Collect metrics for this pair
            pair_margins = []
            pair_pos_outputs = []
            pair_neg_outputs = []
            cross_pattern_responses = {p: [] for p in patterns_in_data}
            
            # Test each source sample
            for src_idx in source_indices:
                orig_weights = weights[src_idx]
                source_sig = signatures[src_idx]
                
                # Edit the network
                edited_net = editor.create_edited_network(
                    orig_weights, source_sig, target_sig
                )
                
                # Test on target pattern
                with torch.no_grad():
                    pos_out = torch.sigmoid(edited_net(pos_inputs)).mean().item()
                    neg_out = torch.sigmoid(edited_net(neg_inputs)).mean().item()
                
                margin = pos_out - neg_out
                pair_margins.append(margin)
                pair_pos_outputs.append(pos_out)
                pair_neg_outputs.append(neg_out)
                all_margins.append(margin)
                all_pos_outputs.append(pos_out)
                all_neg_outputs.append(neg_out)
                
                # Cross-pattern response: test on ALL patterns
                for other_pattern in patterns_in_data:
                    other_cases = get_test_cases_fn(other_pattern)
                    if other_cases is None:
                        continue
                    other_pos = torch.tensor(other_cases['positive'], dtype=torch.float32)
                    other_neg = torch.tensor(other_cases['negative'], dtype=torch.float32)
                    
                    with torch.no_grad():
                        other_pos_out = torch.sigmoid(edited_net(other_pos)).mean().item()
                        other_neg_out = torch.sigmoid(edited_net(other_neg)).mean().item()
                    
                    other_margin = other_pos_out - other_neg_out
                    cross_pattern_responses[other_pattern].append(other_margin)
            
            # Compute metrics for this pair
            pair_pos_arr = np.array(pair_pos_outputs)
            pair_neg_arr = np.array(pair_neg_outputs)
            pair_margins_arr = np.array(pair_margins)
            
            # Success at 0.5 requires both positive and negative sides to clear threshold.
            success_05 = threshold_success_rate(pair_pos_arr, pair_neg_arr, 0.5)
            pos_acc_05 = (pair_pos_arr > 0.5).mean()
            neg_acc_05 = (pair_neg_arr < 0.5).mean()
            
            # Optimal threshold
            opt_thresh, opt_pos_acc, opt_neg_acc, _ = find_optimal_threshold(
                pair_pos_arr, pair_neg_arr
            )
            success_optimal = threshold_success_rate(
                pair_pos_arr,
                pair_neg_arr,
                opt_thresh,
            )
            margin_success = (pair_margins_arr > 0).mean()
            
            # Cross-pattern response means
            cross_response_means = {
                p: float(np.mean(v)) if v else 0.0 
                for p, v in cross_pattern_responses.items()
            }
            
            pair_metric = EditingPairMetrics(
                source_pattern=source_pattern,
                target_pattern=target_pattern,
                n_samples=len(pair_margins),
                success_rate_05=float(success_05),
                target_pos_acc_05=float(pos_acc_05),
                target_neg_acc_05=float(neg_acc_05),
                optimal_threshold=float(opt_thresh),
                success_rate_optimal=float(success_optimal),
                target_pos_acc_optimal=float(opt_pos_acc),
                target_neg_acc_optimal=float(opt_neg_acc),
                margin_mean=float(np.mean(pair_margins_arr)),
                margin_std=float(np.std(pair_margins_arr)),
                margin_min=float(np.min(pair_margins_arr)),
                margin_max=float(np.max(pair_margins_arr)),
                margin_success_rate=float(margin_success),
                cross_pattern_response=cross_response_means,
            )
            
            all_pair_metrics.append(pair_metric)
            success_matrix_05[source_pattern][target_pattern] = float(success_05)
            success_matrix_optimal[source_pattern][target_pattern] = float(success_optimal)
    
    # Global optimal threshold
    all_pos_arr = np.array(all_pos_outputs)
    all_neg_arr = np.array(all_neg_outputs)
    global_opt_thresh, _, _, _ = find_optimal_threshold(all_pos_arr, all_neg_arr)
    
    # Overall success rates
    all_margins_arr = np.array(all_margins)
    overall_success_05 = threshold_success_rate(all_pos_arr, all_neg_arr, 0.5)
    overall_success_optimal = threshold_success_rate(
        all_pos_arr,
        all_neg_arr,
        global_opt_thresh,
    )
    overall_margin_success = (all_margins_arr > 0).mean()
    
    # Find best and worst pairs
    pair_rates = [(p.source_pattern, p.target_pattern, p.success_rate_optimal) 
                  for p in all_pair_metrics]
    pair_rates_sorted = sorted(pair_rates, key=lambda x: x[2], reverse=True)
    
    best_pairs = pair_rates_sorted[:5]
    worst_pairs = pair_rates_sorted[-5:]
    
    return EditingMetrics(
        n_pairs=len(all_pair_metrics),
        n_samples_total=len(all_margins),
        overall_success_rate_05=float(overall_success_05),
        overall_success_rate_optimal=float(overall_success_optimal),
        global_optimal_threshold=float(global_opt_thresh),
        overall_margin_success_rate=float(overall_margin_success),
        success_matrix_05=success_matrix_05,
        success_matrix_optimal=success_matrix_optimal,
        pair_metrics=all_pair_metrics,
        best_pairs=best_pairs,
        worst_pairs=worst_pairs,
    )


def export_editing_matrices(
    metrics: EditingMetrics,
    output_dir: str,
    patterns: List[str],
) -> Tuple[str, str]:
    """
    Export editing success matrices as TSV files.
    
    Returns:
        Tuple of (matrix_05_path, matrix_optimal_path)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Success matrix at 0.5
    path_05 = os.path.join(output_dir, 'editing_matrix_05.tsv')
    with open(path_05, 'w') as f:
        # Header
        f.write('source\t' + '\t'.join(patterns) + '\n')
        for source in patterns:
            row = [source]
            for target in patterns:
                if source == target:
                    row.append('-')
                else:
                    rate = metrics.success_matrix_05.get(source, {}).get(target, 0.0)
                    row.append(f'{rate:.3f}')
            f.write('\t'.join(row) + '\n')
    
    # Success matrix at optimal threshold
    path_optimal = os.path.join(output_dir, 'editing_matrix_optimal.tsv')
    with open(path_optimal, 'w') as f:
        f.write('source\t' + '\t'.join(patterns) + '\n')
        for source in patterns:
            row = [source]
            for target in patterns:
                if source == target:
                    row.append('-')
                else:
                    rate = metrics.success_matrix_optimal.get(source, {}).get(target, 0.0)
                    row.append(f'{rate:.3f}')
            f.write('\t'.join(row) + '\n')
    
    # Cross-pattern response matrix
    path_cross = os.path.join(output_dir, 'cross_pattern_response.tsv')
    with open(path_cross, 'w') as f:
        f.write('source\ttarget\t' + '\t'.join(f'resp_{p}' for p in patterns) + '\n')
        for pm in metrics.pair_metrics:
            row = [pm.source_pattern, pm.target_pattern]
            for p in patterns:
                resp = pm.cross_pattern_response.get(p, 0.0)
                row.append(f'{resp:.4f}')
            f.write('\t'.join(row) + '\n')
    
    logger.info(f"Exported editing matrices to {output_dir}")
    
    return path_05, path_optimal
