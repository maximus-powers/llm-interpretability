import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class EncoderWithHead(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        prediction_head: nn.Module,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.prediction_head = prediction_head
        self.freeze_encoder = freeze_encoder

        # freeze encoder parameters
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen - only training prediction head")
        else:
            logger.info("Encoder unfrozen - fine-tuning entire model")

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                latent = self.encoder(tokens, mask)
        else:
            latent = self.encoder(tokens, mask)
        output = self.prediction_head(latent)
        return output
