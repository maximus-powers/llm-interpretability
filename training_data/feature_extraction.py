import torch
import numpy as np
import logging
from typing import Dict, List, Any
from models import SubjectModel, SequenceDataset

logger = logging.getLogger(__name__)


class BaselineFeatureExtractor:
    """
    Extracts features by processing baseline dataset through models.
    
    These features serve as a consistent "fingerprint" that helps the interpreter
    understand what any model's weights do by seeing how the model processes
    known examples.
    """
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initialized BaselineFeatureExtractor with device: {self.device}")
    
    def extract_features(self, model: SubjectModel, baseline_dataset: Dict[str, Any], batch_size: int = 32) -> Dict[str, Any]:
        """
        Extract features by processing baseline dataset through the model.
        """
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Extracting features from model using {len(baseline_dataset['examples'])} baseline examples")
        
        examples = baseline_dataset['examples']
        formatted_examples = []
        for example in examples:
            formatted_examples.append({
                'sequence': example['sequence'],
                'label': 1.0 
            })
        
        dataset = SequenceDataset(formatted_examples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        layer_activations = {}
        predictions = []
        prediction_confidences = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(loader):
                data = data.to(self.device)
                logits = model(data)
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy().flatten())
                prediction_confidences.extend(torch.abs(probs - 0.5).cpu().numpy().flatten())
                batch_activations = model.get_layer_activations(data)
                # accumulate activations by layer
                for layer_name, activation in batch_activations.items():
                    if layer_name not in layer_activations:
                        layer_activations[layer_name] = []
                    layer_activations[layer_name].append(activation.cpu().numpy())
        
        # aggregate activations
        processed_features = self._process_layer_activations(layer_activations)
        weight_stats = model.get_weight_statistics()
        prediction_stats = self._calculate_prediction_stats(
            predictions, prediction_confidences, baseline_dataset
        )
        features = {
            'layer_activations': processed_features['layer_stats'],
            'activation_patterns': processed_features['pattern_stats'],
            'weight_statistics': weight_stats,
            'prediction_statistics': prediction_stats,
            'model_config': model.config,
            'baseline_info': {
                'num_examples': len(examples),
                'pattern_coverage': baseline_dataset.get('pattern_coverage', {}),
                'dataset_name': baseline_dataset.get('name', 'unknown')
            }
        }
        
        logger.info(f"Extracted features: {len(processed_features['layer_stats'])} layers, "
                   f"{len(weight_stats)} weight tensors")
        
        return features
    
    def _process_layer_activations(self, layer_activations: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
        """
        Process and summarize layer activations.
        """
        layer_stats = {}
        pattern_stats = {}
        
        for layer_name, activation_list in layer_activations.items():
            # concat all batches
            all_activations = np.concatenate(activation_list, axis=0)  # [num_examples, layer_size]
            
            layer_stats[layer_name] = {
                'mean_activation': np.mean(all_activations, axis=0).tolist(),  # Average per neuron
                'std_activation': np.std(all_activations, axis=0).tolist(),   # Std per neuron
                'max_activation': np.max(all_activations, axis=0).tolist(),   # Max per neuron
                'min_activation': np.min(all_activations, axis=0).tolist(),   # Min per neuron
                'sparsity': np.mean(all_activations == 0, axis=0).tolist(),   # Fraction of zeros per neuron
                'activation_norm': np.linalg.norm(all_activations, axis=1).tolist()  # L2 norm per example
            }
            
            pattern_stats[layer_name] = {
                'mean_norm': float(np.mean(layer_stats[layer_name]['activation_norm'])),
                'std_norm': float(np.std(layer_stats[layer_name]['activation_norm'])),
                'overall_sparsity': float(np.mean(all_activations == 0)),
                'active_neurons': int(np.sum(np.any(all_activations != 0, axis=0))),
                'total_neurons': all_activations.shape[1],
                'activation_range': float(np.max(all_activations) - np.min(all_activations))
            }
        
        return {
            'layer_stats': layer_stats,
            'pattern_stats': pattern_stats
        }
    
    def _calculate_prediction_stats(self, predictions: List[float], confidences: List[float], baseline_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate prediction-related statistics.
        """
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        pattern_predictions = {}
        examples = baseline_dataset['examples']
        for i, example in enumerate(examples):
            for pattern in example.get('patterns', []):
                if pattern not in pattern_predictions:
                    pattern_predictions[pattern] = []
                pattern_predictions[pattern].append(predictions[i])

        pattern_stats = {}
        for pattern, preds in pattern_predictions.items():
            preds_array = np.array(preds)
            pattern_stats[pattern] = {
                'mean_prediction': float(np.mean(preds_array)),
                'std_prediction': float(np.std(preds_array)),
                'num_examples': len(preds_array),
                'confidence': float(np.mean([confidences[i] for i, ex in enumerate(examples) 
                                          if pattern in ex.get('patterns', [])]))
            }
        
        prediction_stats = {
            'overall_mean': float(np.mean(predictions)),
            'overall_std': float(np.std(predictions)),
            'overall_confidence': float(np.mean(confidences)),
            'prediction_distribution': {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions)),
                'q25': float(np.percentile(predictions, 25)),
                'q75': float(np.percentile(predictions, 75))
            },
            'pattern_predictions': pattern_stats,
            'high_confidence_examples': int(np.sum(confidences > 0.4)),
            'low_confidence_examples': int(np.sum(confidences < 0.1)),
        }
        
        return prediction_stats