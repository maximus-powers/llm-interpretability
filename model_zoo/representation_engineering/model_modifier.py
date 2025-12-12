import torch
import logging
from typing import Dict, List, Tuple, Optional, Any

from model_zoo.encoder_decoder_training import WeightTokenizer

logger = logging.getLogger(__name__)


class ModelModifier:
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        tokenizer: WeightTokenizer,
        input_mode: str = "weights",
        max_dims: Optional[Dict[str, int]] = None,
        method_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_dims = max_dims
        self.method_names = method_names
        self.device = device
        self.encoder.eval()
        self.decoder.eval()
        logger.info(f"Initialized ModelModifier (input_mode={input_mode})")

    def modify_model(
        self,
        subject_weights: Dict[str, torch.Tensor],
        subject_signature: Optional[Any],
        steering_vectors: List[Tuple[str, torch.Tensor]],
        strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        logger.info(
            f"Modifying model: patterns={[p for p, _ in steering_vectors]}, strength={strength}"
        )

        # Encode to latent based on input_mode
        encoder_input = None
        encoder_mask = None
        original_shapes = None
        num_tokens = None

        if self.input_mode == "weights":
            tokenized = self.tokenizer.tokenize(subject_weights)
            encoder_input = tokenized["tokens"].unsqueeze(0).to(self.device)
            encoder_mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)
            original_shapes = tokenized["original_shapes"]
            num_tokens = encoder_input.size(1)

        elif self.input_mode == "signature":
            if subject_signature is None:
                raise ValueError("Signature required for input_mode='signature'")

            from model_zoo.encoder_decoder_training.data_loader import preprocess_signature
            signature_features, signature_mask = preprocess_signature(
                subject_signature, self.max_dims, self.method_names
            )
            encoder_input = torch.from_numpy(signature_features).unsqueeze(0).to(self.device)
            encoder_mask = torch.from_numpy(signature_mask).unsqueeze(0).to(self.device)

            # For decoder output, we still need weights info
            tokenized = self.tokenizer.tokenize(subject_weights)
            original_shapes = tokenized["original_shapes"]
            num_tokens = tokenized["tokens"].size(1)

        elif self.input_mode == "both":
            if subject_signature is None:
                raise ValueError("Signature required for input_mode='both'")

            tokenized = self.tokenizer.tokenize(subject_weights)
            tokens = tokenized["tokens"]

            from model_zoo.encoder_decoder_training.data_loader import preprocess_signature
            signature_features, signature_mask = preprocess_signature(
                subject_signature, self.max_dims, self.method_names
            )
            signature = torch.from_numpy(signature_features)

            tokens_flat = tokens.flatten()
            combined = torch.cat([tokens_flat, signature], dim=0)
            encoder_input = combined.unsqueeze(0).to(self.device)

            token_mask_flat = tokenized["attention_mask"].repeat_interleave(tokens.size(1))
            sig_mask = torch.from_numpy(signature_mask)
            combined_mask = torch.cat([token_mask_flat, sig_mask], dim=0)
            encoder_mask = combined_mask.unsqueeze(0).to(self.device)

            original_shapes = tokenized["original_shapes"]
            num_tokens = tokens.size(1)

        with torch.no_grad():
            latent = self.encoder(encoder_input, encoder_mask)

        original_latent_norm = latent.norm().item()
        logger.debug(f"Original latent norm: {original_latent_norm:.4f}")

        # Apply steering vectors
        for pattern_name, steering_vector in steering_vectors:
            steering_vector = steering_vector.to(self.device).unsqueeze(0)
            latent = latent + (strength * steering_vector)
            logger.debug(
                f"Added steering vector for '{pattern_name}' (norm: {steering_vector.norm().item():.4f})"
            )

        modified_latent_norm = latent.norm().item()
        logger.debug(
            f"Modified latent norm: {modified_latent_norm:.4f} (delta: {modified_latent_norm - original_latent_norm:+.4f})"
        )

        # Decode back to weights (decoder always outputs weight tokens)
        with torch.no_grad():
            reconstructed_tokens = self.decoder(latent, num_tokens)

        # Create appropriate mask for detokenization
        if self.input_mode == "weights":
            detokenize_mask = encoder_mask.squeeze(0)
        else:
            detokenize_mask = torch.ones(num_tokens)

        modified_weights = self.tokenizer.detokenize(
            reconstructed_tokens.squeeze(0),
            detokenize_mask,
            original_shapes,
        )

        logger.info(
            f"Successfully modified model with {len(steering_vectors)} steering vector(s)"
        )
        return modified_weights
