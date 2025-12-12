import torch
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset
from collections import defaultdict
from huggingface_hub import hf_hub_download
import random

from model_zoo.encoder_decoder_training import WeightTokenizer
from model_zoo.encoder_decoder_training import (
    MLPEncoderDecoder,
    TransformerEncoderDecoder,
)
from model_zoo.encoder_decoder_training.data_loader import (
    compute_dimensions_from_config,
    infer_signature_dimensions,
    preprocess_signature,
)

logger = logging.getLogger(__name__)


class RepresentationDatasetLoader:
    def __init__(
        self,
        hf_dataset_path: str,
        encoder_repo_id: str,
        decoder_repo_id: Optional[str],
        input_mode: str = "weights",
        max_dimensions: Optional[Dict[str, int]] = None,
        method_names: Optional[List[str]] = None,
        device: str = "auto",
        max_models: Optional[int] = None,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.encoder_repo_id = encoder_repo_id
        self.decoder_repo_id = decoder_repo_id or encoder_repo_id
        self.input_mode = input_mode
        self.method_names = method_names or []
        self.max_models = max_models
        self.max_dims = None

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

        # compute max dimensions for signatures if needed
        if input_mode in ["signature", "both"]:
            if not max_dimensions:
                raise ValueError(
                    "max_dimensions required for signature/both input mode"
                )
            if not method_names:
                raise ValueError("method_names required for signature/both input mode")
            config_with_max_dims = {"dataset": {"max_dimensions": max_dimensions}}
            self.max_dims = compute_dimensions_from_config(config_with_max_dims)

        # load encoder/decoder and tokenizer from checkpoint
        self.encoder, self.decoder, self.tokenizer, self.latent_dim = (
            self._load_encoder_decoder()
        )

        self.models_data = []
        self.pattern_clusters = defaultdict(lambda: {"with": [], "without": []})
        logger.info(
            f"Initialized RepresentationDatasetLoader (input_mode={input_mode})"
        )

    def _load_encoder_decoder(self):
        logger.info(f"Loading encoder from {self.encoder_repo_id}...")

        encoder_path = hf_hub_download(
            repo_id=self.encoder_repo_id, filename="encoder.pt"
        )
        checkpoint = torch.load(encoder_path, map_location="cpu")
        encoder_state_dict = checkpoint["encoder_state_dict"]
        config = checkpoint["config"]
        architecture_type = checkpoint["architecture_type"]
        tokenizer_config = checkpoint["tokenizer_config"]
        latent_dim = checkpoint["latent_dim"]

        # validate input mode matches
        checkpoint_input_mode = config.get("dataset", {}).get("input_mode", "weights")
        if checkpoint_input_mode != self.input_mode:
            raise ValueError(
                f"Input mode mismatch: encoder was trained with input_mode='{checkpoint_input_mode}' "
                f"but pipeline is configured for input_mode='{self.input_mode}'. "
                f"Please use input_mode='{checkpoint_input_mode}' in your config or load a different encoder."
            )
        logger.info(f"Validated input_mode: {checkpoint_input_mode}")

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
            strict=True,
        )
        encoder = full_model.encoder
        encoder.eval()
        encoder.to(self.device)

        if self.decoder_repo_id == self.encoder_repo_id:
            decoder_state_dict = checkpoint.get("decoder_state_dict")
            if not decoder_state_dict:
                raise ValueError(
                    f"Encoder checkpoint at {self.encoder_repo_id} missing 'decoder_state_dict'. "
                    "Cannot load decoder from this checkpoint."
                )
            full_model.load_state_dict(
                {
                    "decoder." + k if not k.startswith("decoder.") else k: v
                    for k, v in decoder_state_dict.items()
                },
                strict=True,
            )
        else:
            decoder_path = hf_hub_download(
                repo_id=self.decoder_repo_id, filename="decoder.pt"
            )
            decoder_checkpoint = torch.load(decoder_path, map_location="cpu")
            decoder_state_dict = decoder_checkpoint.get("decoder_state_dict")
            if not decoder_state_dict:
                raise ValueError(
                    f"Decoder checkpoint at {self.decoder_repo_id} missing 'decoder_state_dict'. "
                    "Cannot load decoder from this checkpoint."
                )
            full_model.load_state_dict(
                {
                    "decoder." + k if not k.startswith("decoder.") else k: v
                    for k, v in decoder_state_dict.items()
                },
                strict=True,
            )

        decoder = full_model.decoder
        decoder.eval()
        decoder.to(self.device)

        # load tokenizer from checkpoint config
        tokenizer = WeightTokenizer.from_config(tokenizer_config)
        logger.info(f"Loaded encoder, decoder, and tokenizer (latent_dim={latent_dim})")

        return encoder, decoder, tokenizer, latent_dim

    def load_and_encode_models(
        self,
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any], torch.Tensor]]:
        logger.info(f"Loading dataset from {self.hf_dataset_path}...")
        dataset = load_dataset(self.hf_dataset_path)["train"]
        logger.info(f"Dataset loaded: {len(dataset)} examples")

        # infer signature dimensions from first example if needed
        if self.input_mode in ["signature", "both"] and len(dataset) > 0:
            first_example = dataset[0]
            if "improved_signature" in first_example:
                inferred_dims = infer_signature_dimensions(
                    first_example["improved_signature"], self.method_names
                )
                self.max_dims["signature_features_per_neuron"] = inferred_dims[
                    "signature_features_per_neuron"
                ]
                logger.info(
                    f"Inferred signature dimensions: {inferred_dims['signature_features_per_neuron']} features/neuron"
                )

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
            metadata = (
                json.loads(example["metadata"])
                if isinstance(example["metadata"], str)
                else example["metadata"]
            )

            try:
                encoder_input = None
                encoder_mask = None

                if self.input_mode == "weights":
                    tokenized = self.tokenizer.tokenize(weights_dict)
                    encoder_input = tokenized["tokens"].unsqueeze(0).to(self.device)
                    encoder_mask = (
                        tokenized["attention_mask"].unsqueeze(0).to(self.device)
                    )

                elif self.input_mode == "signature":
                    if "improved_signature" not in example:
                        logger.warning(f"Model {idx} missing signature, skipping")
                        continue
                    signature_features, signature_mask = preprocess_signature(
                        example["improved_signature"], self.max_dims, self.method_names
                    )
                    encoder_input = (
                        torch.from_numpy(signature_features)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    encoder_mask = (
                        torch.from_numpy(signature_mask).unsqueeze(0).to(self.device)
                    )

                elif self.input_mode == "both":
                    tokenized = self.tokenizer.tokenize(weights_dict)
                    tokens = tokenized["tokens"]

                    if "improved_signature" not in example:
                        logger.warning(f"Model {idx} missing signature, skipping")
                        continue
                    signature_features, signature_mask = preprocess_signature(
                        example["improved_signature"], self.max_dims, self.method_names
                    )
                    signature = torch.from_numpy(signature_features)

                    tokens_flat = tokens.flatten()
                    combined = torch.cat([tokens_flat, signature], dim=0)
                    encoder_input = combined.unsqueeze(0).to(self.device)

                    token_mask_flat = tokenized["attention_mask"].repeat_interleave(
                        tokens.size(1)
                    )
                    sig_mask = torch.from_numpy(signature_mask)
                    combined_mask = torch.cat([token_mask_flat, sig_mask], dim=0)
                    encoder_mask = combined_mask.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    latent = self.encoder(encoder_input, encoder_mask)

                latent_vector = latent.squeeze(0).cpu()

                # Store signature in metadata if needed for modification
                if self.input_mode in ["signature", "both"]:
                    metadata["signature"] = example.get("improved_signature")

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
        return self.models_data

    def group_by_patterns(self) -> Dict[str, List[torch.Tensor]]:
        if not self.models_data:
            logger.warning("No models loaded")
            return {}

        logger.info("Grouping models by patterns...")
        pattern_clusters = defaultdict(list)

        # add each model to ALL pattern clusters it belongs to
        for _, metadata, latent_vector in self.models_data:
            patterns = metadata.get("selected_patterns", [])

            # add to each pattern cluster
            for pattern in patterns:
                pattern_clusters[pattern].append(latent_vector)

        # log statistics
        logger.info("Pattern cluster statistics:")
        for pattern in sorted(pattern_clusters.keys()):
            n_models = len(pattern_clusters[pattern])
            logger.info(f"  {pattern}: {n_models} examples")
            if n_models < 10:
                logger.warning(f"    '{pattern}' has only {n_models} examples")

        self.pattern_clusters = dict(pattern_clusters)
        return self.pattern_clusters

    def get_subject_models(
        self,
        filter_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
    ) -> List[Tuple[int, Dict[str, torch.Tensor], Dict[str, Any], Optional[Any]]]:
        if not self.models_data:
            logger.warning("No models loaded")
            return []

        filtered_models = []
        for idx, (weights_dict, metadata, _) in enumerate(self.models_data):
            model_patterns = set(metadata.get("selected_patterns", []))

            if filter_patterns and not all(
                p in model_patterns for p in filter_patterns
            ):
                continue
            if exclude_patterns and any(p in model_patterns for p in exclude_patterns):
                continue

            signature = metadata.get("signature")
            filtered_models.append((idx, weights_dict, metadata, signature))

        logger.info(f"Filtered to {len(filtered_models)} models")

        if sample_size and len(filtered_models) > sample_size:
            filtered_models = random.sample(filtered_models, sample_size)
            logger.info(f"Sampled {sample_size} models")
        return filtered_models

    def get_all_patterns(self) -> List[str]:
        if not self.pattern_clusters:
            logger.warning("No pattern clusters available")
            return []
        return sorted(list(self.pattern_clusters.keys()))
