"""
Self-contained experiments to validate each stage of the architecture.

Each experiment answers a specific question:
- exp1_weight_autoencoder: Can we encode/decode weights through a latent?
- exp2_direct_prediction: How well do signatures directly predict weights?
- exp3_shared_latent: Can signatures and weights map to the same latent?
- exp4_decode_from_sig: Full pipeline - signatures to weights
"""
