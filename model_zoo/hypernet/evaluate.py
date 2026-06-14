"""
Evaluation and inference script for FunctionalHyperNetwork.

Usage:
    # Evaluate a trained model
    python -m hypernet.evaluate --model runs/hypernet_xxx/model.pt
    
    # Edit behavior interactively
    python -m hypernet.evaluate --model model.pt --edit descending ascending
    
    # Generate weights for a behavior
    python -m hypernet.evaluate --model model.pt --generate ascending --n-samples 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypernet.models import FunctionalHyperNetwork, SubjectNetwork, BehaviorEditor
from hypernet.train import (
    load_data, test_behavior, evaluate_reconstruction,
    evaluate_editing, evaluate_all_editing_pairs,
    ALL_PATTERNS, PATTERN_TO_IDX, IDX_TO_PATTERN,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    max_samples: Optional[int] = None,
) -> Dict:
    """Comprehensive evaluation of a trained model."""
    logger.info(f"Loading model from {model_path}")
    model = FunctionalHyperNetwork.load(model_path)
    
    logger.info("Loading evaluation data...")
    data = load_data(max_samples=max_samples)
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    # Reconstruction quality
    logger.info("Evaluating reconstruction...")
    recon = evaluate_reconstruction(model, weights, signatures)
    logger.info(f"Reconstruction - Cosine: {recon['cosine_similarity']:.4f}, MSE: {recon['mse']:.4f}")
    
    # Editing quality
    logger.info("Evaluating behavior editing...")
    edit_results = evaluate_all_editing_pairs(model, data)
    
    for r in edit_results:
        logger.info(f"  {r['source_pattern']} -> {r['target_pattern']}: "
                   f"Original {r['original_correct']}/{r['total']}, "
                   f"Edited {r['edited_correct_target']}/{r['total']}")
    
    # Per-pattern generation quality
    logger.info("Evaluating per-pattern generation...")
    pattern_results = {}
    
    for pattern in ALL_PATTERNS:
        idx = PATTERN_TO_IDX[pattern]
        mask = labels == idx
        
        if mask.sum() < 5:
            continue
        
        # Get signature centroid for this pattern
        pattern_sigs = signatures[mask]
        centroid = pattern_sigs.mean(0)
        
        # Generate weights
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            gen_weights = model.generate(centroid.unsqueeze(0).to(device), n_samples=10)
        
        # Test generated weights
        correct = 0
        for i in range(len(gen_weights)):
            net = SubjectNetwork.from_weights(gen_weights[i].cpu())
            result = test_behavior(net, pattern)
            if result.get('supported') and result.get('correct'):
                correct += 1
        
        pattern_results[pattern] = {
            'n_samples': int(mask.sum()),
            'generated_correct': correct,
            'generated_total': 10,
        }
    
    logger.info("Per-pattern generation results:")
    for pattern, res in pattern_results.items():
        if res['generated_total'] > 0:
            logger.info(f"  {pattern}: {res['generated_correct']}/{res['generated_total']} "
                       f"(from {res['n_samples']} training samples)")
    
    return {
        'reconstruction': recon,
        'editing': edit_results,
        'generation': pattern_results,
    }


def edit_behavior(
    model_path: str,
    source_pattern: str,
    target_pattern: str,
    n_samples: int = 5,
    max_data_samples: Optional[int] = None,
) -> None:
    """Demonstrate behavior editing."""
    logger.info(f"Loading model from {model_path}")
    model = FunctionalHyperNetwork.load(model_path)
    editor = BehaviorEditor(model)
    
    logger.info("Loading data...")
    data = load_data(max_samples=max_data_samples)
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    source_idx = PATTERN_TO_IDX.get(source_pattern)
    target_idx = PATTERN_TO_IDX.get(target_pattern)
    
    if source_idx is None:
        logger.error(f"Unknown source pattern: {source_pattern}")
        logger.info(f"Available: {ALL_PATTERNS}")
        return
    
    if target_idx is None:
        logger.error(f"Unknown target pattern: {target_pattern}")
        logger.info(f"Available: {ALL_PATTERNS}")
        return
    
    source_mask = labels == source_idx
    target_mask = labels == target_idx
    
    if source_mask.sum() < n_samples:
        logger.error(f"Not enough {source_pattern} samples")
        return
    
    if target_mask.sum() < 1:
        logger.error(f"No {target_pattern} samples for target signature")
        return
    
    # Get target signature centroid
    target_sig = signatures[target_mask].mean(0)
    
    # Get source samples
    source_indices = torch.where(source_mask)[0][:n_samples]
    
    logger.info(f"\nEditing {source_pattern} -> {target_pattern}")
    logger.info("=" * 60)
    
    for i, idx in enumerate(source_indices):
        orig_weights = weights[idx]
        source_sig = signatures[idx]
        
        # Test original
        orig_net = SubjectNetwork.from_weights(orig_weights)
        orig_source = test_behavior(orig_net, source_pattern)
        orig_target = test_behavior(orig_net, target_pattern)
        
        # Edit
        edited_net = editor.create_edited_network(
            orig_weights, source_sig, target_sig
        )
        edited_source = test_behavior(edited_net, source_pattern)
        edited_target = test_behavior(edited_net, target_pattern)
        
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Original as {source_pattern}: {orig_source.get('correct', 'N/A')} "
                   f"(margin: {orig_source.get('margin', 0):.3f})")
        logger.info(f"  Original as {target_pattern}: {orig_target.get('correct', 'N/A')} "
                   f"(margin: {orig_target.get('margin', 0):.3f})")
        logger.info(f"  Edited as {source_pattern}: {edited_source.get('correct', 'N/A')} "
                   f"(margin: {edited_source.get('margin', 0):.3f})")
        logger.info(f"  Edited as {target_pattern}: {edited_target.get('correct', 'N/A')} "
                   f"(margin: {edited_target.get('margin', 0):.3f})")


def generate_weights(
    model_path: str,
    pattern: str,
    n_samples: int = 5,
    output_path: Optional[str] = None,
    max_data_samples: Optional[int] = None,
) -> None:
    """Generate weights for a target behavior."""
    logger.info(f"Loading model from {model_path}")
    model = FunctionalHyperNetwork.load(model_path)
    
    logger.info("Loading data for target signature...")
    data = load_data(max_samples=max_data_samples)
    
    signatures = data['signatures']
    labels = data['labels']
    
    idx = PATTERN_TO_IDX.get(pattern)
    if idx is None:
        logger.error(f"Unknown pattern: {pattern}")
        logger.info(f"Available: {ALL_PATTERNS}")
        return
    
    mask = labels == idx
    if mask.sum() < 1:
        logger.error(f"No samples for pattern {pattern}")
        return
    
    # Get signature centroid
    target_sig = signatures[mask].mean(0)
    
    logger.info(f"\nGenerating {n_samples} networks for pattern: {pattern}")
    logger.info("=" * 60)
    
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        gen_weights = model.generate(target_sig.unsqueeze(0).to(device), n_samples=n_samples)
    
    results = []
    for i in range(n_samples):
        net = SubjectNetwork.from_weights(gen_weights[i].cpu())
        result = test_behavior(net, pattern)
        
        status = "PASS" if result.get('correct') else "FAIL"
        margin = result.get('margin', 0)
        
        logger.info(f"  Network {i+1}: {status} (margin: {margin:.4f})")
        
        results.append({
            'weights': gen_weights[i].cpu().tolist(),
            'correct': result.get('correct', False),
            'margin': margin,
        })
    
    correct = sum(1 for r in results if r['correct'])
    logger.info(f"\nSuccess rate: {correct}/{n_samples}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'pattern': pattern,
                'n_samples': n_samples,
                'results': results,
            }, f, indent=2)
        logger.info(f"Saved to {output_path}")


def interpolate_behavior(
    model_path: str,
    source_pattern: str,
    target_pattern: str,
    steps: int = 5,
    max_data_samples: Optional[int] = None,
) -> None:
    """Show interpolation between two behaviors."""
    logger.info(f"Loading model from {model_path}")
    model = FunctionalHyperNetwork.load(model_path)
    editor = BehaviorEditor(model)
    
    logger.info("Loading data...")
    data = load_data(max_samples=max_data_samples)
    
    weights = data['weights']
    signatures = data['signatures']
    labels = data['labels']
    
    source_idx = PATTERN_TO_IDX.get(source_pattern)
    target_idx = PATTERN_TO_IDX.get(target_pattern)
    
    if source_idx is None or target_idx is None:
        logger.error(f"Unknown pattern(s)")
        return
    
    source_mask = labels == source_idx
    target_mask = labels == target_idx
    
    if source_mask.sum() < 1 or target_mask.sum() < 1:
        logger.error("Not enough samples")
        return
    
    # Get a source sample and target signature
    source_idx_sample = torch.where(source_mask)[0][0]
    orig_weights = weights[source_idx_sample]
    source_sig = signatures[source_idx_sample]
    target_sig = signatures[target_mask].mean(0)
    
    logger.info(f"\nInterpolating {source_pattern} -> {target_pattern}")
    logger.info("=" * 60)
    
    for i in range(steps + 1):
        alpha = i / steps
        edited_weights = editor.edit(
            orig_weights, source_sig, target_sig, interpolation=alpha
        )
        
        net = SubjectNetwork.from_weights(edited_weights.cpu())
        source_result = test_behavior(net, source_pattern)
        target_result = test_behavior(net, target_pattern)
        
        logger.info(f"  alpha={alpha:.2f}: "
                   f"{source_pattern}={source_result.get('correct', 'N/A')} "
                   f"(margin {source_result.get('margin', 0):.3f}), "
                   f"{target_pattern}={target_result.get('correct', 'N/A')} "
                   f"(margin {target_result.get('margin', 0):.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and use trained FunctionalHyperNetwork"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum data samples to load"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Full model evaluation")
    
    # edit command
    edit_parser = subparsers.add_parser("edit", help="Edit behavior")
    edit_parser.add_argument("source", help="Source pattern")
    edit_parser.add_argument("target", help="Target pattern")
    edit_parser.add_argument("-n", "--n-samples", type=int, default=5)
    
    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate weights")
    gen_parser.add_argument("pattern", help="Target pattern")
    gen_parser.add_argument("-n", "--n-samples", type=int, default=5)
    gen_parser.add_argument("-o", "--output", help="Output JSON file")
    
    # interpolate command
    interp_parser = subparsers.add_parser("interpolate", help="Interpolate behaviors")
    interp_parser.add_argument("source", help="Source pattern")
    interp_parser.add_argument("target", help="Target pattern")
    interp_parser.add_argument("--steps", type=int, default=5)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if args.command == "evaluate" or args.command is None:
        evaluate_model(args.model, max_samples=args.max_samples)
    elif args.command == "edit":
        edit_behavior(
            args.model, args.source, args.target,
            n_samples=args.n_samples,
            max_data_samples=args.max_samples,
        )
    elif args.command == "generate":
        generate_weights(
            args.model, args.pattern,
            n_samples=args.n_samples,
            output_path=args.output,
            max_data_samples=args.max_samples,
        )
    elif args.command == "interpolate":
        interpolate_behavior(
            args.model, args.source, args.target,
            steps=args.steps,
            max_data_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
