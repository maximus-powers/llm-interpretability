import torch
import logging
import os
from typing import Dict, Any
from huggingface_hub import hf_hub_download
from model_zoo.encoder_decoder_training import WeightTokenizer
from model_zoo.encoder_decoder_training import (
    MLPEncoderDecoder,
    TransformerEncoderDecoder
)

logger = logging.getLogger(__name__)


def load_encoder_from_hub(repo_id: str, token: str = None):
    if token is None:
        token = os.environ.get('HF_TOKEN')

    # download ckpt
    encoder_path = hf_hub_download(repo_id=repo_id, filename='encoder.pt', token=token)
    checkpoint = torch.load(encoder_path, map_location='cpu')

    # metadata from ckpt
    encoder_state_dict = checkpoint['encoder_state_dict']
    config = checkpoint['config']
    tokenizer_config = checkpoint['tokenizer_config']
    latent_dim = checkpoint['latent_dim']
    architecture_type = checkpoint['architecture_type']
    logger.info(f"Encoder architecture: {architecture_type}")
    logger.info(f"Latent dimension: {latent_dim}")

    # init full encoder-decoder model
    if architecture_type == 'mlp':
        full_model = MLPEncoderDecoder(config)
    elif architecture_type == 'transformer':
        full_model = TransformerEncoderDecoder(config)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    # load encoder state dict into the full model
    full_model.load_state_dict({'encoder.' + k if not k.startswith('encoder.') else k: v for k, v in encoder_state_dict.items()}, strict=False)

    encoder = full_model.encoder
    encoder.eval()
    logger.info("Successfully loaded encoder from Hub")

    return {
        'encoder': encoder,
        'latent_dim': latent_dim,
        'tokenizer_config': tokenizer_config,
        'full_config': config,
        'architecture_type': architecture_type
    }


def create_tokenizer_from_config(tokenizer_config: Dict[str, Any]):
    tokenizer = WeightTokenizer(
        chunk_size=tokenizer_config['chunk_size'],
        max_tokens=tokenizer_config['max_tokens'],
        include_metadata=tokenizer_config['include_metadata']
    )
    logger.info("Created tokenizer")
    return tokenizer
