"""
Verify functional equivalence of reconstructed models.

This script loads a trained encoder-decoder, reconstructs weights for test samples,
and measures actual functional equivalence using MULTIPLE methods to cross-validate.

Methods:
1. Binary prediction agreement (sigmoid > 0.5)
2. Raw logit correlation (Pearson)
3. Output MSE / MAE
4. Decision boundary agreement on structured inputs
5. Rank correlation of outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import sys
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_zoo.encoder_decoder_training.data_loader import load_dataset, create_dataloaders
from model_zoo.encoder_decoder_training.encoder_decoder_model import TransformerEncoderDecoder
from model_zoo.encoder_decoder_training.tokenizer import WeightTokenizer
from model_zoo.dataset_generation.models import SubjectModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_model_and_data(config_path: str, checkpoint_path: str, device: str):
    """Load trained model and test data."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load dataset (creates tokenizer internally)
    dataset_info = load_dataset(config)
    tokenizer = dataset_info["tokenizer"]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset_info, config)
    
    # Create model (only takes config)
    model = TransformerEncoderDecoder(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Test set size: {len(test_loader.dataset)}")
    
    return model, test_loader, tokenizer, config


def compute_all_metrics(orig_out: torch.Tensor, recon_out: torch.Tensor):
    """
    Compute multiple metrics between original and reconstructed outputs.
    All metrics should be consistent if functional equivalence is real.
    """
    orig_np = orig_out.cpu().numpy().flatten()
    recon_np = recon_out.cpu().numpy().flatten()
    
    metrics = {}
    
    # 1. Binary prediction agreement
    orig_preds = (torch.sigmoid(orig_out) > 0.5).float()
    recon_preds = (torch.sigmoid(recon_out) > 0.5).float()
    metrics["binary_agreement"] = (orig_preds == recon_preds).float().mean().item()
    metrics["binary_agreement_inverted"] = (orig_preds == (1 - recon_preds)).float().mean().item()
    
    # 2. Raw logit Pearson correlation
    if len(orig_np) > 1 and np.std(orig_np) > 1e-8 and np.std(recon_np) > 1e-8:
        corr, _ = stats.pearsonr(orig_np, recon_np)
        metrics["logit_correlation"] = corr
    else:
        metrics["logit_correlation"] = 0.0
    
    # 3. Output MSE and MAE (on logits)
    metrics["output_mse"] = F.mse_loss(recon_out, orig_out).item()
    metrics["output_mae"] = F.l1_loss(recon_out, orig_out).item()
    
    # 4. Probability MSE (after sigmoid)
    orig_probs = torch.sigmoid(orig_out)
    recon_probs = torch.sigmoid(recon_out)
    metrics["prob_mse"] = F.mse_loss(recon_probs, orig_probs).item()
    
    # 5. Spearman rank correlation (order preservation)
    if len(orig_np) > 1:
        spearman_corr, _ = stats.spearmanr(orig_np, recon_np)
        metrics["rank_correlation"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    else:
        metrics["rank_correlation"] = 0.0
    
    # 6. Sign agreement of logits (simpler than binary)
    orig_signs = (orig_out > 0).float()
    recon_signs = (recon_out > 0).float()
    metrics["sign_agreement"] = (orig_signs == recon_signs).float().mean().item()
    
    # 7. Cosine similarity of output vectors
    metrics["output_cosine"] = F.cosine_similarity(
        orig_out.flatten().unsqueeze(0),
        recon_out.flatten().unsqueeze(0)
    ).item()
    
    return metrics


def evaluate_functional_equivalence(
    model: nn.Module,
    test_loader,
    tokenizer: WeightTokenizer,
    config: dict,
    device: str,
    num_samples: int = 100,
    num_test_inputs: int = 100,
):
    """
    Evaluate functional equivalence using multiple metrics.
    """
    model.eval()
    
    # Aggregated metrics
    all_metrics = defaultdict(list)
    
    # Classification buckets
    results = {
        "perfect_agreement": 0,
        "inverted_agreement": 0,
        "partial_agreement": 0,
        "broken": 0,
        "total": 0,
    }
    
    # Per-sample detailed data
    sample_data = []
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if samples_processed >= num_samples:
                break
                
            encoder_input = batch["encoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            original_shapes = batch["original_shapes"]
            model_configs = batch["model_config"]
            arch_spec = batch.get("arch_spec")
            
            batch_size = encoder_input.size(0)
            
            # arch_spec is a list of dicts (one per sample in batch), use first one
            single_arch_spec = arch_spec[0] if arch_spec else None
            
            if hasattr(model, 'decode_film') and config.get("architecture", {}).get("film_decoder", {}).get("enabled", False):
                encoded_tokens, latent = model.encode_all(encoder_input, encoder_mask)
                reconstructed = model.decode_film(
                    latent=latent,
                    arch_spec=single_arch_spec,
                    num_tokens=decoder_target.size(1),
                    encoder_features=encoded_tokens if config["architecture"]["film_decoder"].get("encoder_features", {}).get("enabled", False) else None,
                )
            else:
                reconstructed, latent = model(
                    encoder_input, decoder_target, encoder_mask, decoder_mask
                )
            
            for i in range(batch_size):
                if samples_processed >= num_samples:
                    break
                    
                results["total"] += 1
                samples_processed += 1
                
                # Detokenize weights
                orig_weights = tokenizer.detokenize_differentiable(
                    decoder_target[i], decoder_mask[i], original_shapes[i]
                )
                recon_weights = tokenizer.detokenize_differentiable(
                    reconstructed[i], decoder_mask[i], original_shapes[i]
                )
                
                # Weight-level cosine similarity
                orig_flat = torch.cat([v.flatten() for v in orig_weights.values()])
                recon_flat = torch.cat([v.flatten() for v in recon_weights.values()])
                weight_cos_sim = F.cosine_similarity(
                    orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
                ).item()
                all_metrics["weight_cosine"].append(weight_cos_sim)
                
                # Weight MSE
                weight_mse = F.mse_loss(recon_flat, orig_flat).item()
                all_metrics["weight_mse"].append(weight_mse)
                
                # Create SubjectModel instances
                mc = model_configs[i]
                vocab_size = mc.get("vocab_size", 10)
                sequence_length = mc.get("sequence_length", 5)
                num_layers = mc.get("num_layers", 6)
                neurons_per_layer = mc.get("neurons_per_layer", 7)
                activation_type = mc.get("activation_type", "relu")
                
                orig_model = SubjectModel(
                    vocab_size=vocab_size,
                    sequence_length=sequence_length,
                    num_layers=num_layers,
                    neurons_per_layer=neurons_per_layer,
                    activation_type=activation_type,
                    dropout_rate=0.0,
                    precision="float32",
                ).to(device)
                
                recon_model = SubjectModel(
                    vocab_size=vocab_size,
                    sequence_length=sequence_length,
                    num_layers=num_layers,
                    neurons_per_layer=neurons_per_layer,
                    activation_type=activation_type,
                    dropout_rate=0.0,
                    precision="float32",
                ).to(device)
                
                # Load weights
                orig_state = {k: v.detach().to(device) for k, v in orig_weights.items()}
                recon_state = {k: v.detach().to(device) for k, v in recon_weights.items()}
                
                try:
                    orig_model.load_state_dict(orig_state)
                    recon_model.load_state_dict(recon_state)
                except Exception as e:
                    logger.warning(f"Sample {samples_processed}: Failed to load weights: {e}")
                    results["broken"] += 1
                    continue
                
                orig_model.eval()
                recon_model.eval()
                
                # Generate random test inputs
                test_inputs = torch.randint(0, vocab_size, (num_test_inputs, sequence_length)).float().to(device)
                
                # Get outputs
                orig_out = orig_model(test_inputs)
                recon_out = recon_model(test_inputs)
                
                # Check for broken outputs
                if torch.isnan(recon_out).any() or torch.isinf(recon_out).any():
                    results["broken"] += 1
                    continue
                
                # Compute all metrics
                metrics = compute_all_metrics(orig_out, recon_out)
                
                # Store metrics
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                # Store sample data for analysis
                sample_data.append({
                    "weight_cosine": weight_cos_sim,
                    "weight_mse": weight_mse,
                    **metrics
                })
                
                # Classify sample
                agreement = metrics["binary_agreement"]
                inverted_agreement = metrics["binary_agreement_inverted"]
                
                if agreement == 1.0:
                    results["perfect_agreement"] += 1
                elif inverted_agreement == 1.0:
                    results["inverted_agreement"] += 1
                elif agreement > 0.5 or inverted_agreement > 0.5:
                    results["partial_agreement"] += 1
                else:
                    results["broken"] += 1
                
                if samples_processed % 50 == 0:
                    logger.info(f"Processed {samples_processed}/{num_samples} samples")
    
    return results, all_metrics, sample_data


def print_results(results: dict, all_metrics: dict, sample_data: list):
    """Print comprehensive results with cross-validation."""
    
    total = results["total"]
    
    print("\n" + "="*70)
    print("FUNCTIONAL EQUIVALENCE EVALUATION - MULTI-METRIC VALIDATION")
    print("="*70)
    
    print(f"\nTotal samples evaluated: {total}")
    
    # === SECTION 1: Classification Results ===
    print("\n" + "-"*70)
    print("1. CLASSIFICATION RESULTS (Binary Prediction Agreement)")
    print("-"*70)
    print(f"  Perfect agreement (100%):    {results['perfect_agreement']:4d} ({100*results['perfect_agreement']/total:.1f}%)")
    print(f"  Inverted agreement (100%):   {results['inverted_agreement']:4d} ({100*results['inverted_agreement']/total:.1f}%)")
    print(f"  Partial agreement (50-99%):  {results['partial_agreement']:4d} ({100*results['partial_agreement']/total:.1f}%)")
    print(f"  Broken (NaN/Inf/<50%):       {results['broken']:4d} ({100*results['broken']/total:.1f}%)")
    
    functional_equiv = results['perfect_agreement'] + results['inverted_agreement']
    print(f"\n  >>> FUNCTIONAL EQUIVALENT:   {functional_equiv:4d} ({100*functional_equiv/total:.1f}%)")
    
    # === SECTION 2: Weight-Level Metrics ===
    print("\n" + "-"*70)
    print("2. WEIGHT-LEVEL METRICS")
    print("-"*70)
    for metric in ["weight_cosine", "weight_mse"]:
        if metric in all_metrics and all_metrics[metric]:
            vals = np.array(all_metrics[metric])
            print(f"  {metric:20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")
    
    # === SECTION 3: Output-Level Metrics ===
    print("\n" + "-"*70)
    print("3. OUTPUT-LEVEL METRICS (should correlate with functional equivalence)")
    print("-"*70)
    
    output_metrics = ["binary_agreement", "sign_agreement", "logit_correlation", 
                      "rank_correlation", "output_cosine", "output_mse", "prob_mse"]
    
    for metric in output_metrics:
        if metric in all_metrics and all_metrics[metric]:
            vals = np.array(all_metrics[metric])
            print(f"  {metric:20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")
    
    # === SECTION 4: Cross-Validation Analysis ===
    print("\n" + "-"*70)
    print("4. CROSS-VALIDATION: Correlation Between Metrics")
    print("-"*70)
    print("   (All metrics should be highly correlated if measuring the same thing)")
    
    if sample_data:
        df_dict = {k: [s.get(k, np.nan) for s in sample_data] for k in sample_data[0].keys()}
        
        # Compute correlations between key metrics
        key_pairs = [
            ("weight_cosine", "binary_agreement"),
            ("weight_cosine", "logit_correlation"),
            ("binary_agreement", "logit_correlation"),
            ("binary_agreement", "output_cosine"),
            ("logit_correlation", "rank_correlation"),
        ]
        
        for m1, m2 in key_pairs:
            if m1 in df_dict and m2 in df_dict:
                v1 = np.array(df_dict[m1])
                v2 = np.array(df_dict[m2])
                mask = ~(np.isnan(v1) | np.isnan(v2))
                if mask.sum() > 2:
                    corr, _ = stats.pearsonr(v1[mask], v2[mask])
                    print(f"  corr({m1}, {m2}): {corr:.3f}")
    
    # === SECTION 5: Distribution Analysis ===
    print("\n" + "-"*70)
    print("5. DISTRIBUTION OF BINARY AGREEMENT")
    print("-"*70)
    
    if "binary_agreement" in all_metrics:
        agreements = np.array(all_metrics["binary_agreement"])
        inverted = np.array(all_metrics["binary_agreement_inverted"])
        best_agreement = np.maximum(agreements, inverted)
        
        print(f"  Best agreement (normal or inverted):")
        print(f"    Mean:   {np.mean(best_agreement):.4f}")
        print(f"    Median: {np.median(best_agreement):.4f}")
        print(f"    Std:    {np.std(best_agreement):.4f}")
        print(f"\n  Thresholds:")
        for thresh in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
            count = np.sum(best_agreement >= thresh)
            pct = 100 * count / len(best_agreement)
            print(f"    >= {thresh:.2f}: {count:4d} ({pct:.1f}%)")
    
    # === SECTION 6: Sanity Checks ===
    print("\n" + "-"*70)
    print("6. SANITY CHECKS")
    print("-"*70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Weight cosine should correlate with binary agreement
    if "weight_cosine" in all_metrics and "binary_agreement" in all_metrics:
        wc = np.array(all_metrics["weight_cosine"])
        ba = np.array(all_metrics["binary_agreement"])
        corr, _ = stats.pearsonr(wc, ba)
        passed = corr > 0.3
        checks_total += 1
        checks_passed += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Weight cosine correlates with binary agreement: r={corr:.3f} (expect > 0.3)")
    
    # Check 2: High logit correlation should imply high binary agreement
    if "logit_correlation" in all_metrics and "binary_agreement" in all_metrics:
        lc = np.array(all_metrics["logit_correlation"])
        ba = np.array(all_metrics["binary_agreement"])
        # For samples with high logit correlation, check binary agreement
        high_lc_mask = lc > 0.8
        if high_lc_mask.sum() > 0:
            mean_ba_high_lc = np.mean(ba[high_lc_mask])
            passed = mean_ba_high_lc > 0.9
            checks_total += 1
            checks_passed += int(passed)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] High logit corr (>0.8) implies high binary agreement: {mean_ba_high_lc:.3f} (expect > 0.9)")
    
    # Check 3: Output cosine should be consistent with logit correlation
    if "output_cosine" in all_metrics and "logit_correlation" in all_metrics:
        oc = np.array(all_metrics["output_cosine"])
        lc = np.array(all_metrics["logit_correlation"])
        # Filter out NaN values
        mask = ~(np.isnan(oc) | np.isnan(lc))
        if mask.sum() > 2:
            corr, _ = stats.pearsonr(oc[mask], lc[mask])
            passed = corr > 0.8
            checks_total += 1
            checks_passed += int(passed)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] Output cosine correlates with logit correlation: r={corr:.3f} (expect > 0.8)")
    
    # Check 4: Perfect agreement samples should have very high logit correlation
    if sample_data:
        perfect_samples = [s for s in sample_data if s.get("binary_agreement", 0) == 1.0]
        if perfect_samples:
            mean_lc = np.mean([s["logit_correlation"] for s in perfect_samples])
            passed = mean_lc > 0.95
            checks_total += 1
            checks_passed += int(passed)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] Perfect agreement samples have high logit corr: {mean_lc:.3f} (expect > 0.95)")
    
    print(f"\n  Sanity checks: {checks_passed}/{checks_total} passed")
    
    # === SECTION 7: Stratified Analysis ===
    print("\n" + "-"*70)
    print("7. STRATIFIED ANALYSIS: Weight Cosine vs Functional Equivalence")
    print("-"*70)
    
    if sample_data:
        # Group by weight cosine buckets
        buckets = [
            (0.0, 0.2, "0.0-0.2"),
            (0.2, 0.4, "0.2-0.4"),
            (0.4, 0.6, "0.4-0.6"),
            (0.6, 0.8, "0.6-0.8"),
            (0.8, 1.0, "0.8-1.0"),
        ]
        
        print("\n  Weight Cosine Bucket -> Binary Agreement:")
        for low, high, label in buckets:
            bucket_samples = [s for s in sample_data if low <= s["weight_cosine"] < high]
            if bucket_samples:
                mean_agree = np.mean([max(s["binary_agreement"], s["binary_agreement_inverted"]) for s in bucket_samples])
                perfect = sum(1 for s in bucket_samples if s["binary_agreement"] == 1.0 or s["binary_agreement_inverted"] == 1.0)
                print(f"    {label}: n={len(bucket_samples):3d}, mean_agreement={mean_agree:.3f}, perfect={perfect}/{len(bucket_samples)}")
    
    # === SECTION 8: Example Samples ===
    print("\n" + "-"*70)
    print("8. EXAMPLE SAMPLES")
    print("-"*70)
    
    if sample_data:
        # Show a few high weight cosine samples
        sorted_by_wc = sorted(sample_data, key=lambda x: x["weight_cosine"], reverse=True)
        print("\n  HIGH weight cosine samples (should have high agreement):")
        for i, s in enumerate(sorted_by_wc[:5]):
            best_agree = max(s['binary_agreement'], s['binary_agreement_inverted'])
            print(f"    Sample: wc={s['weight_cosine']:.3f}, best_agree={best_agree:.3f}, "
                  f"output_cos={s['output_cosine']:.3f}")
        
        # Show a few low weight cosine but high agreement samples (if they exist)
        print("\n  LOW weight cosine but HIGH agreement (surprising cases):")
        surprising = [s for s in sample_data if s["weight_cosine"] < 0.3 and 
                     max(s["binary_agreement"], s["binary_agreement_inverted"]) > 0.95]
        for i, s in enumerate(surprising[:5]):
            best_agree = max(s['binary_agreement'], s['binary_agreement_inverted'])
            print(f"    Sample: wc={s['weight_cosine']:.3f}, best_agree={best_agree:.3f}, "
                  f"output_cos={s['output_cosine']:.3f}")
        if not surprising:
            print("    (none found)")
        
        # Show a few high weight cosine but low agreement samples (if they exist)
        print("\n  HIGH weight cosine but LOW agreement (concerning cases):")
        concerning = [s for s in sample_data if s["weight_cosine"] > 0.6 and 
                     max(s["binary_agreement"], s["binary_agreement_inverted"]) < 0.8]
        for i, s in enumerate(concerning[:5]):
            best_agree = max(s['binary_agreement'], s['binary_agreement_inverted'])
            print(f"    Sample: wc={s['weight_cosine']:.3f}, best_agree={best_agree:.3f}, "
                  f"output_cos={s['output_cosine']:.3f}")
        if not concerning:
            print("    (none found)")
    
    print("\n" + "="*70)


def debug_single_sample(model, batch, tokenizer, config, device, sample_idx=0):
    """Deep dive into a single sample to verify everything is working."""
    print("\n" + "="*70)
    print("DEBUG: SINGLE SAMPLE ANALYSIS")
    print("="*70)
    
    encoder_input = batch["encoder_input"].to(device)
    decoder_target = batch["decoder_target"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)
    original_shapes = batch["original_shapes"]
    model_configs = batch["model_config"]
    arch_spec = batch.get("arch_spec")
    
    single_arch_spec = arch_spec[0] if arch_spec else None
    
    # Forward pass
    with torch.no_grad():
        encoded_tokens, latent = model.encode_all(encoder_input, encoder_mask)
        reconstructed = model.decode_film(
            latent=latent,
            arch_spec=single_arch_spec,
            num_tokens=decoder_target.size(1),
            encoder_features=encoded_tokens if config["architecture"]["film_decoder"].get("encoder_features", {}).get("enabled", False) else None,
        )
    
    i = sample_idx
    
    # Detokenize
    orig_weights = tokenizer.detokenize_differentiable(
        decoder_target[i], decoder_mask[i], original_shapes[i]
    )
    recon_weights = tokenizer.detokenize_differentiable(
        reconstructed[i], decoder_mask[i], original_shapes[i]
    )
    
    print(f"\n1. WEIGHT SHAPES:")
    for k in orig_weights.keys():
        print(f"   {k}: orig={orig_weights[k].shape}, recon={recon_weights[k].shape}")
    
    print(f"\n2. WEIGHT VALUES (first layer weight, first 5 values):")
    first_key = list(orig_weights.keys())[0]
    print(f"   Original:      {orig_weights[first_key].flatten()[:5].tolist()}")
    print(f"   Reconstructed: {recon_weights[first_key].flatten()[:5].tolist()}")
    
    # Weight cosine
    orig_flat = torch.cat([v.flatten() for v in orig_weights.values()])
    recon_flat = torch.cat([v.flatten() for v in recon_weights.values()])
    weight_cos = F.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)).item()
    print(f"\n3. WEIGHT COSINE SIMILARITY: {weight_cos:.4f}")
    
    # Model config
    mc = model_configs[i]
    print(f"\n4. MODEL CONFIG:")
    print(f"   vocab_size={mc.get('vocab_size')}, seq_len={mc.get('sequence_length')}, "
          f"layers={mc.get('num_layers')}, neurons={mc.get('neurons_per_layer')}")
    
    # Create models
    orig_model = SubjectModel(
        vocab_size=mc.get("vocab_size", 10),
        sequence_length=mc.get("sequence_length", 5),
        num_layers=mc.get("num_layers", 6),
        neurons_per_layer=mc.get("neurons_per_layer", 7),
        activation_type=mc.get("activation_type", "relu"),
        dropout_rate=0.0,
        precision="float32",
    ).to(device)
    
    recon_model = SubjectModel(
        vocab_size=mc.get("vocab_size", 10),
        sequence_length=mc.get("sequence_length", 5),
        num_layers=mc.get("num_layers", 6),
        neurons_per_layer=mc.get("neurons_per_layer", 7),
        activation_type=mc.get("activation_type", "relu"),
        dropout_rate=0.0,
        precision="float32",
    ).to(device)
    
    # Load weights
    orig_state = {k: v.detach().to(device) for k, v in orig_weights.items()}
    recon_state = {k: v.detach().to(device) for k, v in recon_weights.items()}
    orig_model.load_state_dict(orig_state)
    recon_model.load_state_dict(recon_state)
    orig_model.eval()
    recon_model.eval()
    
    # Test inputs
    vocab_size = mc.get("vocab_size", 10)
    sequence_length = mc.get("sequence_length", 5)
    test_inputs = torch.randint(0, vocab_size, (20, sequence_length)).float().to(device)
    
    print(f"\n5. TEST INPUTS (first 3):")
    for j in range(3):
        print(f"   {test_inputs[j].tolist()}")
    
    # Get outputs
    with torch.no_grad():
        orig_out = orig_model(test_inputs)
        recon_out = recon_model(test_inputs)
    
    print(f"\n6. RAW OUTPUTS (first 5):")
    print(f"   Original:      {orig_out[:5].flatten().tolist()}")
    print(f"   Reconstructed: {recon_out[:5].flatten().tolist()}")
    
    # Predictions
    orig_preds = (torch.sigmoid(orig_out) > 0.5).float()
    recon_preds = (torch.sigmoid(recon_out) > 0.5).float()
    
    print(f"\n7. PREDICTIONS (first 10):")
    print(f"   Original:      {orig_preds[:10].flatten().int().tolist()}")
    print(f"   Reconstructed: {recon_preds[:10].flatten().int().tolist()}")
    
    agreement = (orig_preds == recon_preds).float().mean().item()
    print(f"\n8. BINARY AGREEMENT: {agreement:.4f}")
    
    # Class balance
    orig_class_balance = orig_preds.mean().item()
    recon_class_balance = recon_preds.mean().item()
    print(f"\n9. CLASS BALANCE (fraction predicting class 1):")
    print(f"   Original:      {orig_class_balance:.4f}")
    print(f"   Reconstructed: {recon_class_balance:.4f}")
    
    # What if we use a RANDOM model?
    random_model = SubjectModel(
        vocab_size=mc.get("vocab_size", 10),
        sequence_length=mc.get("sequence_length", 5),
        num_layers=mc.get("num_layers", 6),
        neurons_per_layer=mc.get("neurons_per_layer", 7),
        activation_type=mc.get("activation_type", "relu"),
        dropout_rate=0.0,
        precision="float32",
    ).to(device)
    random_model.eval()
    
    with torch.no_grad():
        random_out = random_model(test_inputs)
    random_preds = (torch.sigmoid(random_out) > 0.5).float()
    random_agreement = (orig_preds == random_preds).float().mean().item()
    
    print(f"\n10. RANDOM MODEL AGREEMENT (baseline): {random_agreement:.4f}")
    
    print("\n" + "="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-test-inputs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--debug", action="store_true", help="Run single sample debug")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model and data
    model, test_loader, tokenizer, config = load_model_and_data(
        args.config, args.checkpoint, device
    )
    
    # Debug mode: analyze multiple samples
    if args.debug:
        sample_count = 0
        class_balances = []
        random_agreements = []
        
        for batch in test_loader:
            if sample_count >= 10:
                break
            debug_single_sample(model, batch, tokenizer, config, device, sample_idx=sample_count % batch["encoder_input"].size(0))
            sample_count += 1
            
            # Quick check of class balance for all samples in batch
            encoder_input = batch["encoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            original_shapes = batch["original_shapes"]
            model_configs = batch["model_config"]
            
            for i in range(min(5, encoder_input.size(0))):
                mc = model_configs[i]
                orig_weights = tokenizer.detokenize_differentiable(
                    decoder_target[i], decoder_mask[i], original_shapes[i]
                )
                
                orig_model = SubjectModel(
                    vocab_size=mc.get("vocab_size", 10),
                    sequence_length=mc.get("sequence_length", 5),
                    num_layers=mc.get("num_layers", 6),
                    neurons_per_layer=mc.get("neurons_per_layer", 7),
                    activation_type=mc.get("activation_type", "relu"),
                    dropout_rate=0.0,
                    precision="float32",
                ).to(device)
                
                orig_state = {k: v.detach().to(device) for k, v in orig_weights.items()}
                orig_model.load_state_dict(orig_state)
                orig_model.eval()
                
                test_inputs = torch.randint(0, mc.get("vocab_size", 10), (100, mc.get("sequence_length", 5))).float().to(device)
                
                with torch.no_grad():
                    orig_out = orig_model(test_inputs)
                orig_preds = (torch.sigmoid(orig_out) > 0.5).float()
                class_balances.append(orig_preds.mean().item())
                
                # Random model
                random_model = SubjectModel(
                    vocab_size=mc.get("vocab_size", 10),
                    sequence_length=mc.get("sequence_length", 5),
                    num_layers=mc.get("num_layers", 6),
                    neurons_per_layer=mc.get("neurons_per_layer", 7),
                    activation_type=mc.get("activation_type", "relu"),
                    dropout_rate=0.0,
                    precision="float32",
                ).to(device)
                random_model.eval()
                with torch.no_grad():
                    random_out = random_model(test_inputs)
                random_preds = (torch.sigmoid(random_out) > 0.5).float()
                random_agreement = (orig_preds == random_preds).float().mean().item()
                random_agreements.append(random_agreement)
        
        print("\n" + "="*70)
        print("CLASS BALANCE ANALYSIS (fraction predicting class 1)")
        print("="*70)
        print(f"Samples analyzed: {len(class_balances)}")
        print(f"Class balance: mean={np.mean(class_balances):.3f}, min={np.min(class_balances):.3f}, max={np.max(class_balances):.3f}")
        print(f"Random model agreement: mean={np.mean(random_agreements):.3f}, min={np.min(random_agreements):.3f}, max={np.max(random_agreements):.3f}")
        print("\nInterpretation:")
        if np.mean(class_balances) > 0.8 or np.mean(class_balances) < 0.2:
            print("  WARNING: Highly imbalanced classes! Random models can achieve high agreement.")
        if np.mean(random_agreements) > 0.6:
            print("  WARNING: Random models achieve >60% agreement - task may be trivial.")
        return
    
    # Run evaluation
    results, all_metrics, sample_data = evaluate_functional_equivalence(
        model, test_loader, tokenizer, config, device,
        num_samples=args.num_samples,
        num_test_inputs=args.num_test_inputs,
    )
    
    # Print comprehensive results
    print_results(results, all_metrics, sample_data)


if __name__ == "__main__":
    main()
