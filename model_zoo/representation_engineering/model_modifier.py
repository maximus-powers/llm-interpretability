import torch
import logging
from typing import Dict, List, Tuple

from model_zoo.encoder_decoder_training import WeightTokenizer

logger = logging.getLogger(__name__)


class ModelModifier:
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, tokenizer: WeightTokenizer, device: str = "cpu"):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.encoder.eval()
        self.decoder.eval()
        logger.info("Initialized ModelModifier")

    def modify_model(
        self,
        subject_weights: Dict[str, torch.Tensor],
        steering_vectors: List[Tuple[str, torch.Tensor]],
        operation: str,
        strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if operation not in ["add", "remove"]:
            raise ValueError(f"Operation must be 'add' or 'remove', got '{operation}'")

        logger.info(f"Modifying model: operation={operation}, patterns={[p for p, _ in steering_vectors]}, strength={strength}")

        tokenized = self.tokenizer.tokenize(subject_weights)
        tokens = tokenized["tokens"].unsqueeze(0).to(self.device)
        mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.encoder(tokens, mask)

        original_latent_norm = latent.norm().item()
        logger.debug(f"Original latent norm: {original_latent_norm:.4f}")

        for pattern_name, steering_vector in steering_vectors:
            steering_vector = steering_vector.to(self.device).unsqueeze(0)
            if operation == "add":
                latent = latent + (strength * steering_vector)
                logger.debug(f"Added steering vector for '{pattern_name}' (norm: {steering_vector.norm().item():.4f})")
            elif operation == "remove":
                latent = latent - (strength * steering_vector)
                logger.debug(f"Removed steering vector for '{pattern_name}' (norm: {steering_vector.norm().item():.4f})")

        modified_latent_norm = latent.norm().item()
        logger.debug(f"Modified latent norm: {modified_latent_norm:.4f} (delta: {modified_latent_norm - original_latent_norm:+.4f})")

        with torch.no_grad():
            reconstructed_tokens = self.decoder(latent, tokens.size(1))

        modified_weights = self.tokenizer.detokenize(
            reconstructed_tokens.squeeze(0),
            mask.squeeze(0),
            tokenized["original_shapes"],
        )

        logger.info(f"Successfully modified model with {len(steering_vectors)} steering vector(s)")
        return modified_weights

    def modify_with_individual_vectors(
        self,
        subject_weights: Dict[str, torch.Tensor],
        steering_vectors: List[Tuple[str, torch.Tensor]],
        operation: str,
        strength: float = 1.0,
    ) -> List[Dict[str, torch.Tensor]]:
        modified_models = []
        for pattern_name, steering_vector in steering_vectors:
            logger.info(f"Applying individual steering vector for '{pattern_name}'")
            modified_weights = self.modify_model(
                subject_weights=subject_weights,
                steering_vectors=[(pattern_name, steering_vector)],
                operation=operation,
                strength=strength,
            )
            modified_models.append(modified_weights)
        return modified_models

    def encode_to_latent(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokenized = self.tokenizer.tokenize(weights)
        tokens = tokenized["tokens"].unsqueeze(0).to(self.device)
        mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.encoder(tokens, mask)
        return latent.squeeze(0).cpu()

    def decode_from_latent(self, latent: torch.Tensor, original_shapes: List[Tuple[str, Tuple[int, ...]]]) -> Dict[str, torch.Tensor]:
        latent = latent.unsqueeze(0).to(self.device)
        total_params = sum(torch.prod(torch.tensor(shape)).item() for _, shape in original_shapes)
        chunk_size = self.tokenizer.chunk_size
        max_tokens = self.tokenizer.max_tokens
        num_tokens = min(int((total_params + chunk_size - 1) // chunk_size), max_tokens)
        with torch.no_grad():
            reconstructed_tokens = self.decoder(latent, num_tokens)
        mask = torch.ones(num_tokens)
        reconstructed_weights = self.tokenizer.detokenize(reconstructed_tokens.squeeze(0), mask, original_shapes)
        return reconstructed_weights

    def compute_reconstruction_error(
        self,
        original_weights: Dict[str, torch.Tensor],
        reconstructed_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        orig_flat = torch.cat([w.flatten() for w in original_weights.values()])
        recon_flat = torch.cat([w.flatten() for w in reconstructed_weights.values()])
        mse = torch.nn.functional.mse_loss(recon_flat, orig_flat).item()
        mae = torch.nn.functional.l1_loss(recon_flat, orig_flat).item()
        cosine_sim = torch.nn.functional.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)).item()
        relative_error = (recon_flat - orig_flat).abs() / (orig_flat.abs() + 1e-8)
        mean_relative_error = relative_error.mean().item()
        return {
            "mse": mse,
            "mae": mae,
            "rmse": mse**0.5,
            "cosine_similarity": cosine_sim,
            "mean_relative_error": mean_relative_error,
        }
