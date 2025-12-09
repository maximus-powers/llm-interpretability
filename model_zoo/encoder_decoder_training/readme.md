# Weight-Space Autoencoder Training

Trains autoencoders to compress neural network weights into compact latent representations and reconstruct them.

## Purpose

Given tokenized neural network weights, the autoencoder learns to encode them into a low-dimensional latent space while preserving the ability to reconstruct the original weights. This enables efficient weight storage, interpolation, and analysis of neural network weight spaces.

## Components

**encoder_decoder_model.py**: Implements MLP and Transformer encoder-decoder architectures. Encoder compresses tokenized weights into fixed-size latent vectors. Decoder reconstructs tokenized weights from latent codes. Supports pooling strategies (mean, max, flatten) and positional encodings.

**tokenizer.py**: Converts raw neural network weight tensors into token sequences. Chunks weights into fixed-size groups, adds metadata (layer index, parameter type, position), handles variable-length sequences with attention masks, and manages padding.

**data_loader.py**: Loads weight datasets from HuggingFace, tokenizes weights on-the-fly, creates train/val/test splits, and provides data augmentation for contrastive learning (noise injection, dropout).

**losses.py**: Unified loss functions for autoencoder training. Includes reconstruction losses (MSE, MAE, Cosine), combined weighted losses, and contrastive losses (NT-Xent/SimCLR) with optional projection heads. Supports mixed contrastive + reconstruction objectives via gamma weighting.

**trainer.py**: Training loop with early stopping, learning rate scheduling, and HuggingFace Hub integration. Handles checkpointing, TensorBoard logging, and automatic upload of encoder/decoder models with documentation.

**evaluator.py**: Computes reconstruction quality metrics including MSE, MAE, RMSE, cosine similarity, relative error, and R² score. Supports per-layer analysis for detailed performance breakdown.

## Training Flow

1. **Load dataset**: Read generated subject models from HuggingFace Hub
2. **Tokenize weights**: Convert weight tensors into token sequences with metadata and attention masks
3. **Initialize model**: Configure encoder-decoder architecture (MLP or Transformer) with specified latent dimension
4. **Train**: Optimize autoencoder using reconstruction loss (and optionally contrastive loss for better latent structure)
5. **Evaluate**: Measure reconstruction quality on held-out test set using multiple metrics
6. **Upload to Hub**: Push trained encoder and decoder as separate files to HuggingFace for downstream use

## Architecture Options

**MLP**: Fast and simple. Token pooling ’ fully-connected layers ’ latent bottleneck ’ decoder layers ’ token reconstruction.

**Transformer**: More expressive. Self-attention over token sequences ’ latent pooling ’ decoder with cross-attention ’ per-token reconstruction.

Both architectures support separate file loading (encoder-only for inference, decoder-only for generation, or both for full reconstruction).
