"""
Latent space analysis metrics.

Computes:
- Silhouette score (clustering quality)
- Adjusted Rand Index (clustering vs true labels)
- Intra-class and inter-class distances
- Linear separability (logistic regression accuracy)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class LatentMetrics:
    """Results from latent space analysis."""
    silhouette_score: float
    adjusted_rand_index: float
    intra_class_distance: float
    inter_class_distance: float
    inter_intra_ratio: float
    linear_separability: float
    n_samples: int
    n_patterns: int
    
    # Per-pattern centroids for later use
    pattern_centroids: Dict[int, np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'silhouette_score': self.silhouette_score,
            'adjusted_rand_index': self.adjusted_rand_index,
            'intra_class_distance': self.intra_class_distance,
            'inter_class_distance': self.inter_class_distance,
            'inter_intra_ratio': self.inter_intra_ratio,
            'linear_separability': self.linear_separability,
            'n_samples': self.n_samples,
            'n_patterns': self.n_patterns,
        }


def compute_latent_metrics(
    model,
    weights: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
) -> LatentMetrics:
    """
    Compute comprehensive latent space metrics.
    
    Args:
        model: FunctionalHyperNetwork model
        weights: Test set weights [N, weight_dim]
        signatures: Test set signatures [N, sig_dim]
        labels: Test set pattern labels [N]
    
    Returns:
        LatentMetrics dataclass with all computed metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get latent representations
    weights_d = weights.to(device)
    signatures_d = signatures.to(device)
    
    with torch.no_grad():
        # Encode to get mu (latent mean)
        _, mu, _, _ = model(weights_d, signatures_d)
        latents = mu.cpu().numpy()
    
    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)
    n_patterns = len(unique_labels)
    
    logger.info(f"Computing latent metrics for {len(latents)} samples, {n_patterns} patterns")
    
    # 1. Silhouette Score
    if n_patterns > 1 and len(latents) > n_patterns:
        sil_score = silhouette_score(latents, labels_np)
    else:
        sil_score = 0.0
        logger.warning("Not enough samples/patterns for silhouette score")
    
    # 2. Adjusted Rand Index (cluster vs true labels)
    if n_patterns > 1:
        kmeans = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latents)
        ari = adjusted_rand_score(labels_np, cluster_labels)
    else:
        ari = 0.0
    
    # 3. Intra-class and Inter-class distances
    intra_distances = []
    centroids = {}
    
    for label in unique_labels:
        mask = labels_np == label
        class_latents = latents[mask]
        centroid = class_latents.mean(axis=0)
        centroids[int(label)] = centroid
        
        # Intra-class: distance from each point to its centroid
        distances = np.linalg.norm(class_latents - centroid, axis=1)
        intra_distances.extend(distances)
    
    intra_class_dist = np.mean(intra_distances) if intra_distances else 0.0
    
    # Inter-class: distance between centroids
    centroid_list = list(centroids.values())
    inter_distances = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            dist = np.linalg.norm(centroid_list[i] - centroid_list[j])
            inter_distances.append(dist)
    
    inter_class_dist = np.mean(inter_distances) if inter_distances else 0.0
    inter_intra_ratio = inter_class_dist / (intra_class_dist + 1e-8)
    
    # 4. Linear Separability (logistic regression)
    if n_patterns > 1:
        scaler = StandardScaler()
        latents_scaled = scaler.fit_transform(latents)
        
        clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        clf.fit(latents_scaled, labels_np)
        linear_sep = clf.score(latents_scaled, labels_np)
    else:
        linear_sep = 1.0
    
    return LatentMetrics(
        silhouette_score=float(sil_score),
        adjusted_rand_index=float(ari),
        intra_class_distance=float(intra_class_dist),
        inter_class_distance=float(inter_class_dist),
        inter_intra_ratio=float(inter_intra_ratio),
        linear_separability=float(linear_sep),
        n_samples=len(latents),
        n_patterns=n_patterns,
        pattern_centroids=centroids,
    )


def export_latents_for_projector(
    model,
    weights: torch.Tensor,
    signatures: torch.Tensor,
    labels: torch.Tensor,
    idx_to_pattern: Dict[int, str],
    output_dir: str,
) -> Tuple[str, str]:
    """
    Export latent vectors and metadata as TSV files for TensorBoard Projector.
    
    Args:
        model: FunctionalHyperNetwork model
        weights: Weights tensor [N, weight_dim]
        signatures: Signatures tensor [N, sig_dim]
        labels: Pattern labels [N]
        idx_to_pattern: Mapping from label index to pattern name
        output_dir: Directory to save TSV files
    
    Returns:
        Tuple of (vectors_path, metadata_path)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    weights_d = weights.to(device)
    signatures_d = signatures.to(device)
    
    with torch.no_grad():
        _, mu, _, _ = model(weights_d, signatures_d)
        latents = mu.cpu().numpy()
    
    labels_np = labels.numpy()
    
    # Write vectors TSV (no header, tab-separated floats)
    vectors_path = os.path.join(output_dir, 'latent_vectors.tsv')
    with open(vectors_path, 'w') as f:
        for row in latents:
            f.write('\t'.join(f'{x:.6f}' for x in row) + '\n')
    
    # Write metadata TSV (with header)
    metadata_path = os.path.join(output_dir, 'metadata.tsv')
    with open(metadata_path, 'w') as f:
        f.write('pattern\tpattern_idx\tsample_idx\n')
        for i, label in enumerate(labels_np):
            pattern_name = idx_to_pattern.get(int(label), f'unknown_{label}')
            f.write(f'{pattern_name}\t{int(label)}\t{i}\n')
    
    logger.info(f"Exported {len(latents)} latent vectors to {vectors_path}")
    logger.info(f"Exported metadata to {metadata_path}")
    
    return vectors_path, metadata_path
