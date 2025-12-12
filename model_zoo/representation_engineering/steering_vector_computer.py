import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class SteeringVectorComputer:
    def __init__(
        self,
        pattern_clusters: Dict[str, List[torch.Tensor]],
        device: str = "cpu",
        cache_dir: Path = Path("steering_vectors_cache"),
        normalize_vectors: bool = False,
    ):
        self.pattern_clusters = pattern_clusters
        self.device = device
        self.cache_dir = cache_dir
        self.normalize_vectors = normalize_vectors
        self.steering_vectors = {}  # nested: {from_pattern: {to_pattern: vector_data}}
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_patterns = list(pattern_clusters.keys())
        n_pairwise = len(all_patterns) * (len(all_patterns) - 1)
        logger.info(
            f"Initialized SteeringVectorComputer - {len(all_patterns)} patterns, "
            f"{n_pairwise} possible pairwise vectors"
        )

    def compute_pairwise_steering_vector(
        self, from_pattern: str, to_pattern: str
    ) -> torch.Tensor:
        if from_pattern not in self.pattern_clusters:
            raise ValueError(f"Pattern '{from_pattern}' not found")
        if to_pattern not in self.pattern_clusters:
            raise ValueError(f"Pattern '{to_pattern}' not found")
        if from_pattern == to_pattern:
            raise ValueError("from_pattern and to_pattern must be different")

        from_cluster = self.pattern_clusters[from_pattern]
        to_cluster = self.pattern_clusters[to_pattern]

        if not from_cluster:
            raise ValueError(f"No models with pattern '{from_pattern}'")
        if not to_cluster:
            raise ValueError(f"No models with pattern '{to_pattern}'")

        from_tensor = torch.stack(from_cluster).to(self.device)
        to_tensor = torch.stack(to_cluster).to(self.device)
        mean_from = from_tensor.mean(dim=0)
        mean_to = to_tensor.mean(dim=0)

        steering_vector = mean_to - mean_from

        if self.normalize_vectors:
            steering_vector = steering_vector / steering_vector.norm()

        # store in nested dict structure
        if from_pattern not in self.steering_vectors:
            self.steering_vectors[from_pattern] = {}

        self.steering_vectors[from_pattern][to_pattern] = {
            "vector": steering_vector.cpu(),
            "n_from": len(from_cluster),
            "n_to": len(to_cluster),
            "mean_from": mean_from.cpu(),
            "mean_to": mean_to.cpu(),
            "norm": steering_vector.norm().item(),
            "normalized": self.normalize_vectors,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Computed '{from_pattern}'→'{to_pattern}': "
            f"{len(from_cluster)} from, {len(to_cluster)} to, "
            f"norm={steering_vector.norm().item():.4f}"
        )
        return steering_vector.cpu()

    def compute_all_pairwise_vectors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        all_patterns = list(self.pattern_clusters.keys())
        logger.info(
            f"Computing all pairwise vectors for {len(all_patterns)} patterns..."
        )

        computed = 0
        failed = 0
        for from_pattern in all_patterns:
            for to_pattern in all_patterns:
                if from_pattern == to_pattern:
                    continue
                try:
                    self.compute_pairwise_steering_vector(from_pattern, to_pattern)
                    computed += 1
                except Exception as e:
                    logger.error(f"Failed '{from_pattern}'→'{to_pattern}': {e}")
                    failed += 1

        logger.info(f"Computed {computed} pairwise vectors, {failed} failed")
        return self.steering_vectors

    def save_cache(
        self,
        filename: str = "steering_vectors.pt",
        metadata: Optional[Dict[str, Any]] = None,
    ):
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

    def load_cache(
        self,
        filename: str = "steering_vectors.pt",
        expected_metadata: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> bool:
        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            logger.info(f"Cache not found: {cache_path}")
            return False

        try:
            cache_data = torch.load(cache_path, map_location="cpu")
            if validate and expected_metadata:
                cached_meta = cache_data.get("metadata", {})
                critical_keys = [
                    "steering_dataset_path",
                    "encoder_repo_id",
                    "latent_dim",
                    "normalize_vectors",
                ]
                for key in critical_keys:
                    if cached_meta.get(key) != expected_metadata.get(key):
                        logger.warning(f"Cache validation failed: {key} mismatch")
                        return False
            self.steering_vectors = cache_data["steering_vectors"]
            logger.info(
                f"Loaded {len(self.steering_vectors)} steering vectors from cache"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def find_nearest_pattern(self, latent_vector: torch.Tensor) -> str:
        latent = latent_vector.to(self.device)

        min_distance = float("inf")
        nearest_pattern = None

        for pattern, cluster_latents in self.pattern_clusters.items():
            # compute cluster centroid
            cluster_tensor = torch.stack(cluster_latents).to(self.device)
            centroid = cluster_tensor.mean(dim=0)

            # compute euclidean distance
            distance = (latent - centroid).norm().item()

            if distance < min_distance:
                min_distance = distance
                nearest_pattern = pattern

        logger.debug(
            f"Nearest pattern: '{nearest_pattern}' (distance: {min_distance:.4f})"
        )
        return nearest_pattern

    def get_steering_vector(self, from_pattern: str, to_pattern: str) -> torch.Tensor:
        if from_pattern in self.steering_vectors:
            if to_pattern in self.steering_vectors[from_pattern]:
                return self.steering_vectors[from_pattern][to_pattern]["vector"]

        return self.compute_pairwise_steering_vector(from_pattern, to_pattern)

    def get_vector_statistics(
        self, from_pattern: str, to_pattern: str
    ) -> Dict[str, Any]:
        if from_pattern not in self.steering_vectors:
            raise ValueError(f"Steering vector for '{from_pattern}' not computed")
        if to_pattern not in self.steering_vectors[from_pattern]:
            raise ValueError(
                f"Steering vector for '{from_pattern}'→'{to_pattern}' not computed"
            )

        metadata = self.steering_vectors[from_pattern][to_pattern]
        vector = metadata["vector"]
        return {
            "from_pattern": from_pattern,
            "to_pattern": to_pattern,
            "transformation": f"{from_pattern}→{to_pattern}",
            "n_from": metadata["n_from"],
            "n_to": metadata["n_to"],
            "norm": metadata["norm"],
            "normalized": metadata["normalized"],
            "min_value": vector.min().item(),
            "max_value": vector.max().item(),
            "mean_value": vector.mean().item(),
            "std_value": vector.std().item(),
            "timestamp": metadata["timestamp"],
        }

    def compare_steering_vectors(
        self, from1: str, to1: str, from2: str, to2: str
    ) -> Dict[str, float]:
        if (
            from1 not in self.steering_vectors
            or to1 not in self.steering_vectors[from1]
        ):
            raise ValueError(f"Steering vector '{from1}'→'{to1}' not computed")
        if (
            from2 not in self.steering_vectors
            or to2 not in self.steering_vectors[from2]
        ):
            raise ValueError(f"Steering vector '{from2}'→'{to2}' not computed")

        vec1 = self.steering_vectors[from1][to1]["vector"]
        vec2 = self.steering_vectors[from2][to2]["vector"]
        return {
            "transformation1": f"{from1}→{to1}",
            "transformation2": f"{from2}→{to2}",
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                vec1.unsqueeze(0), vec2.unsqueeze(0)
            ).item(),
            "euclidean_distance": (vec1 - vec2).norm().item(),
            "correlation": torch.corrcoef(torch.stack([vec1, vec2]))[0, 1].item(),
        }
