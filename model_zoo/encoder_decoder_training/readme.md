# Cross-Modal Encoder-Decoder Training

Trains cross-modal encoder-decoders that encode neural network activation signatures into latent representations and decode them back to weights. Uses supervised contrastive learning to organize the latent space by behavior.

## Purpose

Given tokenized activation signatures (behavioral fingerprints), the encoder learns to compress them into a behavior-organized latent space. The decoder reconstructs the original network weights from these latent codes. This enables:
- **Behavior-based clustering**: Networks with similar behaviors cluster together in latent space
- **Representation engineering**: Push latent representations between behavior clusters to add/remove behaviors from generated weights

## Cross-Modal Architecture

The encoder-decoder operates across modalities:
- **Encoder input**: Activation signatures (behavioral fingerprints of neurons)
- **Latent space**: Behavior-organized representations
- **Decoder output**: Network weights that produce the encoded behavior

This cross-modal design allows manipulating behaviors in latent space and generating weights that exhibit desired behaviors.

## Components

**encoder_decoder_model.py**: Implements MLP and Transformer encoder-decoder architectures. Encoder compresses tokenized signatures into fixed-size latent vectors. Decoder reconstructs tokenized weights from latent codes. Supports pooling strategies (mean, max, flatten) and positional encodings.

**tokenizer.py**: Converts raw data into token sequences. For signatures, uses neuron-level tokenization where each neuron becomes one token. Adds metadata (layer index, parameter type, position, shape encoding, token index) and handles variable-length sequences with attention masks.

**data_loader.py**: Loads datasets from HuggingFace containing signatures and weights. Tokenizes signatures for encoder input and weights for decoder target. Extracts behavior labels from `selected_patterns` metadata for supervised contrastive learning. Includes behavior-aware batch sampling to guarantee positive pairs in every training batch.

**losses.py**: Unified loss functions including reconstruction losses (MSE, MAE, Cosine) and supervised contrastive loss. The contrastive loss uses behavior labels to define positive pairs—networks sharing any behavior pattern are pulled together in latent space. Supports mixed contrastive + reconstruction objectives via gamma weighting.

**trainer.py**: Training loop with early stopping, learning rate scheduling, and HuggingFace Hub integration. Handles checkpointing, TensorBoard logging, and automatic upload of encoder/decoder models.

**evaluator.py**: Computes reconstruction quality metrics including MSE, MAE, RMSE, cosine similarity, relative error, and R² score. Supports per-layer analysis for detailed performance breakdown.

## Training Flow

1. **Load dataset**: Read generated subject models with signatures and weights from HuggingFace Hub
2. **Tokenize**: Convert signatures to encoder tokens (neuron-level) and weights to decoder target tokens
3. **Initialize model**: Configure encoder-decoder architecture with specified latent dimension
4. **Train**: Optimize using combined loss: `L = γ * L_contrastive + (1-γ) * L_reconstruction`
   - Contrastive loss clusters networks by behavior (positive pairs = networks sharing any behavior pattern)
   - Reconstruction loss ensures decoder can recreate weights from latent codes
5. **Evaluate**: Measure reconstruction quality on held-out test set
6. **Upload to Hub**: Push trained encoder and decoder to HuggingFace for downstream use

## Architecture Options

**MLP**: Fast and simple. Token pooling → fully-connected layers → latent bottleneck → decoder layers → token reconstruction.

**Transformer**: More expressive. Self-attention over token sequences → latent pooling → decoder with cross-attention → per-token reconstruction.

Both architectures support separate file loading (encoder-only for inference, decoder-only for generation, or both for full reconstruction).

## Supervised Contrastive Learning

The contrastive loss uses behavior labels from dataset metadata (`selected_patterns`) to define positive pairs:

- Networks with `[A, B]` and `[A, C]` → **positive pair** (share pattern A)
- Networks with `[A, B]` and `[D, E]` → **negative pair** (no overlap)

This creates overlapping soft clusters where networks sharing behaviors are pulled together proportionally. The temperature parameter controls clustering hardness.

### Behavior-Aware Batch Sampling

To ensure contrastive learning is effective, the data loader uses a `BehaviorAwareBatchSampler` that pairs each sample with a "buddy" sharing at least one behavior pattern. This guarantees every sample has a positive pair in its batch, preventing zero-gradient scenarios from the contrastive loss.
