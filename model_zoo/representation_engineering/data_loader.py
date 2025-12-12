import torch
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from huggingface_hub import hf_hub_download

from model_zoo.encoder_decoder_training import WeightTokenizer
from model_zoo.encoder_decoder_training import (
    MLPEncoderDecoder,
    TransformerEncoderDecoder,
)

logger = logging.getLogger(__name__)


class RepresentationDatasetLoader:
    def __init__(
        self,
        hf_dataset_path: str,
        encoder_repo_id: str,
        decoder_repo_id: Optional[str],
        tokenizer_config: Dict[str, Any],
        latent_dim: int,
        device: str = "auto",
        cache_latents: bool = True,
        cache_dir: str = "latent_cache",
        max_models: Optional[int] = None,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.encoder_repo_id = encoder_repo_id
        self.decoder_repo_id = decoder_repo_id or encoder_repo_id
        self.tokenizer_config = tokenizer_config
        self.latent_dim = latent_dim
        self.cache_latents = cache_latents
        self.cache_dir = Path(cache_dir)
        self.max_models = max_models

        # setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if cache_latents:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # load model and set up tokenizer
        self.encoder, self.decoder = self._load_encoder_decoder()
        self.tokenizer = WeightTokenizer(
            chunk_size=tokenizer_config["chunk_size"],
            max_tokens=tokenizer_config["max_tokens"],
            include_metadata=tokenizer_config.get("include_metadata", True),
        )

        self.models_data = []
        self.pattern_clusters = defaultdict(lambda: {"with": [], "without": []})
        logger.info("Initialized RepresentationDatasetLoader")

    def _load_encoder_decoder(self):
        logger.info(f"Loading encoder from {self.encoder_repo_id}...")

        encoder_path = hf_hub_download(
            repo_id=self.encoder_repo_id, filename="encoder.pt"
        )
        checkpoint = torch.load(encoder_path, map_location="cpu")
        encoder_state_dict = checkpoint["encoder_state_dict"]
        config = checkpoint["config"]
        architecture_type = checkpoint["architecture_type"]

        if architecture_type == "mlp":
            full_model = MLPEncoderDecoder(config)
        elif architecture_type == "transformer":
            full_model = TransformerEncoderDecoder(config)
        else:
            raise ValueError(f"Unknown architecture: {architecture_type}")

        full_model.load_state_dict(
            {
                "encoder." + k if not k.startswith("encoder.") else k: v
                for k, v in encoder_state_dict.items()
            },
            strict=False,
        )
        encoder = full_model.encoder
        encoder.eval()
        encoder.to(self.device)

        if self.decoder_repo_id == self.encoder_repo_id:
            decoder_state_dict = checkpoint.get("decoder_state_dict")
            if decoder_state_dict:
                full_model.load_state_dict(
                    {
                        "decoder." + k if not k.startswith("decoder.") else k: v
                        for k, v in decoder_state_dict.items()
                    },
                    strict=False,
                )
        else:
            decoder_path = hf_hub_download(
                repo_id=self.decoder_repo_id, filename="decoder.pt"
            )
            decoder_checkpoint = torch.load(decoder_path, map_location="cpu")
            decoder_state_dict = decoder_checkpoint.get("decoder_state_dict")
            if decoder_state_dict:
                full_model.load_state_dict(
                    {
                        "decoder." + k if not k.startswith("decoder.") else k: v
                        for k, v in decoder_state_dict.items()
                    },
                    strict=False,
                )

        decoder = full_model.decoder
        decoder.eval()
        decoder.to(self.device)

        logger.info("Loaded encoder and decoder")
        return encoder, decoder

    def load_and_encode_models(self) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any], torch.Tensor]]:
        cache_key = hashlib.md5(
            f"{self.hf_dataset_path}_{self.encoder_repo_id}_{self.max_models}".encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"latents_{cache_key}.pt"

        if self.cache_latents and cache_path.exists():
            logger.info("Loading from cache...")
            cached_data = torch.load(cache_path, map_location="cpu")
            self.models_data = cached_data["models_data"]
            logger.info(f"Loaded {len(self.models_data)} models from cache")
            return self.models_data

        logger.info(f"Loading dataset from {self.hf_dataset_path}...")
        dataset = load_dataset(self.hf_dataset_path)["train"]
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        models_encoded = 0
        for idx, example in enumerate(dataset):
            if self.max_models and models_encoded >= self.max_models:
                logger.info(f"Reached max_models limit: {self.max_models}")
                break
            if "improved_model_weights" not in example:
                continue

            improved_weights_json = example["improved_model_weights"]
            improved_weights = json.loads(improved_weights_json)
            weights_dict = improved_weights["weights"]
            metadata = json.loads(example["metadata"]) if isinstance(example["metadata"], str) else example["metadata"]

            try:
                tokenized = self.tokenizer.tokenize(weights_dict)
                tokens = tokenized["tokens"].unsqueeze(0).to(self.device)
                mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    latent = self.encoder(tokens, mask)

                latent_vector = latent.squeeze(0).cpu()
                self.models_data.append((weights_dict, metadata, latent_vector))
                models_encoded += 1
                if models_encoded % 50 == 0:
                    logger.info(f"Encoded {models_encoded} models...")
                    if models_encoded % 32 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to encode model {idx}: {e}")
                continue

        logger.info(f"Encoded {len(self.models_data)} models")
        if self.cache_latents:
            torch.save({"models_data": self.models_data}, cache_path)
            logger.info("Saved to cache")
        return self.models_data

    def group_by_patterns(self) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        if not self.models_data:
            logger.warning("No models loaded")
            return {}

        logger.info("Grouping models by patterns...")
        all_patterns = set()
        for _, metadata, _ in self.models_data:
            all_patterns.update(metadata.get("selected_patterns", []))

        logger.info(f"Found {len(all_patterns)} patterns: {sorted(all_patterns)}")

        for pattern in all_patterns:
            self.pattern_clusters[pattern] = {"with": [], "without": []}

        for _, metadata, latent_vector in self.models_data:
            model_patterns = set(metadata.get("selected_patterns", []))
            for pattern in all_patterns:
                if pattern in model_patterns:
                    self.pattern_clusters[pattern]["with"].append(latent_vector)
                else:
                    self.pattern_clusters[pattern]["without"].append(latent_vector)

        logger.info("Pattern statistics:")
        for pattern in sorted(all_patterns):
            n_with = len(self.pattern_clusters[pattern]["with"])
            n_without = len(self.pattern_clusters[pattern]["without"])
            logger.info(f"  {pattern}: {n_with} with, {n_without} without")
            if n_with < 10:
                logger.warning(f"    '{pattern}' has only {n_with} positive examples")
            if n_without < 10:
                logger.warning(f"    '{pattern}' has only {n_without} negative examples")
        return dict(self.pattern_clusters)

    def get_subject_models(
        self,
        filter_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
        if not self.models_data:
            logger.warning("No models loaded")
            return []

        filtered_models = []
        for weights_dict, metadata, _ in self.models_data:
            model_patterns = set(metadata.get("selected_patterns", []))

            if filter_patterns and not all(
                p in model_patterns for p in filter_patterns
            ):
                continue
            if exclude_patterns and any(p in model_patterns for p in exclude_patterns):
                continue

            filtered_models.append((weights_dict, metadata))

        logger.info(f"Filtered to {len(filtered_models)} models")

        if sample_size and len(filtered_models) > sample_size:
            import random
            filtered_models = random.sample(filtered_models, sample_size)
            logger.info(f"Sampled {sample_size} models")
        return filtered_models

    def get_all_patterns(self) -> List[str]:
        if not self.pattern_clusters:
            logger.warning("No pattern clusters available")
            return []
        return sorted(list(self.pattern_clusters.keys()))
