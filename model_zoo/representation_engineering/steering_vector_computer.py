import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class SteeringVectorComputer:
    def __init__(
        self,
        pattern_clusters: Dict[str, Dict[str, List[torch.Tensor]]],
        device: str = "cpu",
        cache_dir: Path = Path("steering_vectors_cache"),
        normalize_vectors: bool = False,
    ):
        self.pattern_clusters = pattern_clusters
        self.device = device
        self.cache_dir = cache_dir
        self.normalize_vectors = normalize_vectors
        self.steering_vectors = {}
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized SteeringVectorComputer - {len(pattern_clusters)} patterns")

    def compute_steering_vector(self, pattern: str) -> torch.Tensor:
        if pattern not in self.pattern_clusters:
            raise ValueError(f"Pattern '{pattern}' not found")

        with_pattern = self.pattern_clusters[pattern]["with"]
        without_pattern = self.pattern_clusters[pattern]["without"]

        if not with_pattern:
            raise ValueError(f"No models with pattern '{pattern}'")
        if not without_pattern:
            raise ValueError(f"No models without pattern '{pattern}'")

        with_pattern_tensor = torch.stack(with_pattern).to(self.device)
        without_pattern_tensor = torch.stack(without_pattern).to(self.device)
        mean_with = with_pattern_tensor.mean(dim=0)
        mean_without = without_pattern_tensor.mean(dim=0)

        steering_vector = mean_with - mean_without

        if self.normalize_vectors:
            steering_vector = steering_vector / steering_vector.norm()
        self.steering_vectors[pattern] = {
            "vector": steering_vector.cpu(),
            "n_with": len(with_pattern),
            "n_without": len(without_pattern),
            "mean_with": mean_with.cpu(),
            "mean_without": mean_without.cpu(),
            "norm": steering_vector.norm().item(),
            "normalized": self.normalize_vectors,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Computed '{pattern}': {len(with_pattern)} with, {len(without_pattern)} without, norm={steering_vector.norm().item():.4f}")
        return steering_vector.cpu()

    def compute_all_steering_vectors(self, patterns: List[str]) -> Dict[str, torch.Tensor]:
        logger.info(f"Computing {len(patterns)} steering vectors...")
        steering_vectors = {}
        for pattern in patterns:
            try:
                vector = self.compute_steering_vector(pattern)
                steering_vectors[pattern] = vector
            except Exception as e:
                logger.error(f"Failed to compute '{pattern}': {e}")
                continue
        logger.info(f"Computed {len(steering_vectors)}/{len(patterns)} steering vectors")
        return steering_vectors

    def save_cache(self, filename: str = "steering_vectors.pt", metadata: Optional[Dict[str, Any]] = None):
        cache_path = self.cache_dir / filename
        cache_data = {
            "steering_vectors": self.steering_vectors,
            "normalize_vectors": self.normalize_vectors,
            "metadata": {
                "steering_dataset_path": metadata.get("steering_dataset_path")
                if metadata
                else None,
                "encoder_repo_id": metadata.get("encoder_repo_id")
                if metadata
                else None,
                "latent_dim": metadata.get("latent_dim") if metadata else None,
                "normalize_vectors": metadata.get("normalize_vectors")
                if metadata
                else self.normalize_vectors,
                "n_models": metadata.get("n_models") if metadata else None,
                "patterns": sorted(list(self.steering_vectors.keys())),
                "created_at": datetime.now().isoformat(),
                "cache_version": "1.0",
            },
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(cache_data, cache_path)
        logger.info(f"Saved {len(self.steering_vectors)} steering vectors to cache")

    def load_cache(self, filename: str = "steering_vectors.pt", expected_metadata: Optional[Dict[str, Any]] = None, validate: bool = True) -> bool:
        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            logger.info(f"Cache not found: {cache_path}")
            return False

        try:
            cache_data = torch.load(cache_path, map_location="cpu")
            if validate and expected_metadata:
                cached_meta = cache_data.get("metadata", {})
                critical_keys = ["steering_dataset_path", "encoder_repo_id", "latent_dim", "normalize_vectors"]
                for key in critical_keys:
                    if cached_meta.get(key) != expected_metadata.get(key):
                        logger.warning(f"Cache validation failed: {key} mismatch")
                        return False
            self.steering_vectors = cache_data["steering_vectors"]
            logger.info(f"Loaded {len(self.steering_vectors)} steering vectors from cache")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def get_steering_vector(self, pattern: str) -> torch.Tensor:
        if pattern in self.steering_vectors:
            return self.steering_vectors[pattern]["vector"]
        return self.compute_steering_vector(pattern)

    def get_vector_statistics(self, pattern: str) -> Dict[str, Any]:
        if pattern not in self.steering_vectors:
            raise ValueError(f"Steering vector for '{pattern}' not computed")
        metadata = self.steering_vectors[pattern]
        vector = metadata["vector"]
        return {
            "pattern": pattern,
            "n_with": metadata["n_with"],
            "n_without": metadata["n_without"],
            "norm": metadata["norm"],
            "normalized": metadata["normalized"],
            "min_value": vector.min().item(),
            "max_value": vector.max().item(),
            "mean_value": vector.mean().item(),
            "std_value": vector.std().item(),
            "timestamp": metadata["timestamp"],
        }

    def compare_steering_vectors(self, pattern1: str, pattern2: str) -> Dict[str, float]:
        if pattern1 not in self.steering_vectors or pattern2 not in self.steering_vectors:
            raise ValueError("Both patterns must have computed steering vectors")
        vec1 = self.steering_vectors[pattern1]["vector"]
        vec2 = self.steering_vectors[pattern2]["vector"]
        return {
            "cosine_similarity": torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item(),
            "euclidean_distance": (vec1 - vec2).norm().item(),
            "correlation": torch.corrcoef(torch.stack([vec1, vec2]))[0, 1].item(),
        }
