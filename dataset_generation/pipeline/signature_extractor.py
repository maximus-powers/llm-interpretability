import torch
import numpy as np
import logging
from typing import Dict, List, Any
from .models import SubjectModel, SequenceDataset
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class ActivationSignatureExtractor:
    """
    Extracts activation signatures (aka model fingerprints) by processing the signature dataset through subject models.
    """
    
    def __init__(self, device: str = 'auto', signature_dataset_path: str = None, neuron_profile_config: Dict[str, Dict] = None):
        if not signature_dataset_path.exists():
            raise FileNotFoundError(f"Signature dataset not found at {signature_dataset_path}. Create one using pattern_sampler.create_signature_dataset_file(), and make sure you hold onto it. This is the element used for interpretability, and the same dataset must be used for all training examples and inference.")
        with open(signature_dataset_path, 'r') as f:
            logger.info(f"Signature dataset found: {signature_dataset_path}")
            self.signature_dataset = json.load(f)
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.neuron_profile_config = neuron_profile_config or {
            'mean': {},
            'std': {}
        }
        
        logger.info(f"Initialized Signature Extractor: device={self.device}, profile_methods={list(self.neuron_profile_config.keys())}")
    
    def extract(self, model: SubjectModel, batch_size: int = 32) -> Dict[str, Any]:
        """
        Extract activation features by processing signature dataset through the model.
        """
        model = model.to(self.device)
        model.eval() # need to disable dropout/batchnorm
        logger.info(f"Extracting features from model using {len(self.signature_dataset['examples'])} baseline examples")
        
        examples = self.signature_dataset['examples']
        formatted_examples = []
        example_patterns = []
        for example in examples:
            formatted_examples.append({
                'sequence': example['sequence'],
                'label': 1.0 # dummy, isn't used in anything here, dataset class just extracts it so we're avoiding type err
            })
            example_patterns.append(example.get('pattern', None))
        dataset = SequenceDataset(formatted_examples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False) # no shuffling for consistency in activations, shouldn't matter tho
        
        layer_activations = {}
        predictions = []
        prediction_confidences = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                logits = model(data)
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy().flatten())
                prediction_confidences.extend(torch.abs(probs - 0.5).cpu().numpy().flatten())
                batch_activations = model.get_layer_activations(data) # get's 
                # accumulate activations by layer
                for layer_name, activation in batch_activations.items():
                    if layer_name not in layer_activations:
                        layer_activations[layer_name] = []
                    layer_activations[layer_name].append(activation.cpu().numpy())
        
        # aggregate activations with pattern information
        processed_features = self._process_layer_activations(layer_activations, example_patterns)
        features = {
            'neuron_activations': processed_features,
            'model_config': model.config,
        }
        
        logger.info("Computed activation signature")
        
        return features
    
    #### METHODS TO TRY FOR NEURON PROFILING ####
    def _compute_mean(self, neuron_activations: np.ndarray) -> float:
        """Neuron's typical activation level - high values indicate generally responsive neurons"""
        return float(np.mean(neuron_activations))
    
    def _compute_std(self, neuron_activations: np.ndarray) -> float:
        """Activation variability across examples - high values indicate context-sensitive neurons"""
        return float(np.std(neuron_activations))
        
    def _compute_max(self, neuron_activations: np.ndarray) -> float:
        """Peak activation strength - reveals neuron's maximum response capability"""
        return float(np.max(neuron_activations))
        
    def _compute_min(self, neuron_activations: np.ndarray) -> float:
        """Minimum activation level - negative values indicate inhibitory capabilities"""
        return float(np.min(neuron_activations))
    
    def _compute_pca(self, all_layer_activations: np.ndarray, neuron_id: int, components: int = 1) -> List[float]:
        """Neuron's contribution to layer's main activation patterns - reveals computational role"""
        if all_layer_activations.shape[0] <= components: # not enough samples for pca
            return [0.0] * components
        try:
            # fit PCA on the layer activations (examples as rows, neurons as columns)
            n_components_actual = min(components, all_layer_activations.shape[0] - 1, all_layer_activations.shape[1])
            pca = PCA(n_components=n_components_actual)
            pca.fit(all_layer_activations)
            # get loadings (how much each neuron contributes to each component)
            # component shape: [n_components, n_neurons]
            loadings = pca.components_[:, neuron_id] 
            result = loadings[:components].tolist()
            while len(result) < components:
                result.append(0.0)
            return result
        except Exception as e:
            logger.warning(f"PCA failed: {e}, returning zeros")
            return [0.0] * components
        
    def _compute_entropy(self, neuron_activations: np.ndarray, bins: int = 20) -> float:
        """Shannon entropy of activation distribution - measures response predictability"""
        if np.std(neuron_activations) == 0:
            return 0.0  # 0 entropy
        hist, _ = np.histogram(neuron_activations, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return float(-np.sum(hist * np.log2(hist + 1e-10)))
        
    def _compute_clustering(self, neuron_activations: np.ndarray, n_clusters: int = 2) -> List[float]:
        """K-means cluster centers - reveals distinct activation states (e.g., on/off modes)"""
        if len(np.unique(neuron_activations)) < n_clusters: # not enough unique values to form clusters
            unique_vals = np.unique(neuron_activations)
            result = unique_vals.tolist()
            while len(result) < n_clusters:
                result.append(result[-1] if result else 0.0)
            return result
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(neuron_activations.reshape(-1, 1))
            return sorted(kmeans.cluster_centers_.flatten().tolist())
        except Exception:
            return [float(np.min(neuron_activations)), float(np.max(neuron_activations))]
            
    def _compute_svd(self, all_layer_activations: np.ndarray, neuron_id: int, components: int = 1) -> List[float]:
        """SVD coefficients - neuron's importance in low-rank approximations of layer activity"""
        try:
            U, s, Vt = np.linalg.svd(all_layer_activations, full_matrices=False)
            result = Vt[:components, neuron_id].tolist() # coefficients for this neuron in the top singular vectors
            return result
        except Exception:
            return [0.0] * components
            
    def _compute_fourier(self, neuron_activations: np.ndarray, n_frequencies: int = 1) -> List[float]:
        """Dominant frequency magnitudes - detects periodic/rhythmic activation patterns"""
        if len(neuron_activations) < 2:
            return [0.0] * n_frequencies
        try:
            fft = np.fft.fft(neuron_activations)
            mags = np.abs(fft[:len(fft)//2])  # only positive frequencies
            top_indices = np.argsort(mags)[-n_frequencies:]
            return mags[top_indices].tolist()
        except Exception:
            return [0.0] * n_frequencies
    
    def _compute_pattern_wise(self, neuron_activations: np.ndarray, example_patterns: List[str]) -> List[float]:
        """Mean activation for each pattern in the signature dataset"""
        if len(neuron_activations) != len(example_patterns):
            logger.warning(f"Mismatch between activations ({len(neuron_activations)}) and patterns ({len(example_patterns)})")
            return []
        
        pattern_activations = {}
        for activation, pattern in zip(neuron_activations, example_patterns):
            if pattern is None:
                continue
            if pattern not in pattern_activations:
                pattern_activations[pattern] = []
            pattern_activations[pattern].append(activation)
        all_patterns = sorted(pattern_activations.keys())
        
        pattern_means = []
        for pattern in all_patterns:
            if pattern_activations[pattern]:
                mean_activation = float(np.mean(pattern_activations[pattern]))
                pattern_means.append(mean_activation)
            else:
                pattern_means.append(0.0)
        
        return pattern_means
    
    def _process_layer_activations(self, layer_activations: Dict[str, List[np.ndarray]], example_patterns: List[str]) -> Dict[str, Any]:
        """Orchestrates the computation of neuron profiles for each layer."""
        layer_profiles = {}
        
        for layer_name, activation_list in layer_activations.items():
            all_activations = np.concatenate(activation_list, axis=0) # [num_examples, num_neurons]
            num_examples, num_neurons = all_activations.shape
            
            # create neuron profiles
            neuron_profiles = {}
            for neuron_id in range(num_neurons):
                neuron_activations = all_activations[:, neuron_id]
                profile = {}
                
                for method, params in self.neuron_profile_config.items():
                    if method == 'mean':
                        profile['mean'] = self._compute_mean(neuron_activations)
                    elif method == 'std':
                        profile['std'] = self._compute_std(neuron_activations)
                    elif method == 'max':
                        profile['max'] = self._compute_max(neuron_activations)
                    elif method == 'min':
                        profile['min'] = self._compute_min(neuron_activations)
                    elif method == 'pca':
                        components = params.get('components', 1)
                        profile['pca'] = self._compute_pca(all_activations, neuron_id, components)
                    elif method == 'entropy':
                        bins = params.get('bins', 20)
                        profile['entropy'] = self._compute_entropy(neuron_activations, bins)
                    elif method == 'clustering':
                        n_clusters = params.get('n_clusters', 2)
                        profile['clustering'] = self._compute_clustering(neuron_activations, n_clusters)
                    elif method == 'svd':
                        components = params.get('components', 1)
                        profile['svd'] = self._compute_svd(all_activations, neuron_id, components)
                    elif method == 'fourier':
                        n_frequencies = params.get('n_frequencies', 1)
                        profile['fourier'] = self._compute_fourier(neuron_activations, n_frequencies)
                    elif method == 'pattern_wise':
                        profile['pattern_wise'] = self._compute_pattern_wise(neuron_activations, example_patterns)
                    else:
                        logger.warning(f"Unknown profiling method: {method}")
                        
                neuron_profiles[neuron_id] = profile
            
            layer_profiles[layer_name] = {
                'neuron_profiles': neuron_profiles,
                'layer_info': {
                    'num_neurons': num_neurons,
                    'num_examples': num_examples,
                    'profile_methods': list(self.neuron_profile_config.keys())
                }
            }
        
        return layer_profiles