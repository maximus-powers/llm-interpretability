#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path
import yaml
import torch
from .classification.data_loader import load_dataset, create_dataloaders, compute_model_architecture
from .classification.classifier_model import PatternClassifierMLP
from .classification.trainer import ClassifierTrainer, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_classifier(args):
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_info = load_dataset(config)

    # model architecture
    num_patterns = len(dataset_info['all_patterns'])
    computed_arch = compute_model_architecture(dataset_info['input_dims'], num_patterns, config)
    if 'model' not in config:
        config['model'] = {}
    for encoder_name in ['signature_encoder', 'weight_encoder', 'fusion']:
        if encoder_name in config['model']:
            if 'hidden_dims' not in config['model'][encoder_name]:
                if computed_arch.get(encoder_name):
                    config['model'][encoder_name]['hidden_dims'] = computed_arch[encoder_name]['hidden_dims']
        else:
            if computed_arch.get(encoder_name):
                config['model'][encoder_name] = {
                    **computed_arch[encoder_name],
                    'dropout': 0.3,
                    'activation': 'relu'
                }
    config['model']['output'] = computed_arch['output']

    # dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset_info, config)

    # device
    if config['device']['type'] == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = config['device']['type']

    # init model
    model = PatternClassifierMLP(config, dataset_info['input_dims'])

    # checkpoint
    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model)
        logger.info(f"Resuming from epoch {checkpoint_info['epoch'] + 1}")

    logger.info("Starting training")
    trainer = ClassifierTrainer(model, config, train_loader, val_loader, device, dataset_info['all_patterns'], test_loader)
    trainer.train()
    logger.info("Training complete")


def main():
    parser = argparse.ArgumentParser(
        prog='experiments',
        description='Unified CLI for running experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='experiment_type', help='Experiment type')

    classifier_parser = subparsers.add_parser(
        'classifier',
        help='Pattern classifier experiments'
    )
    classifier_subparsers = classifier_parser.add_subparsers(
        dest='classifier_command',
        help='Classifier commands'
    )
    classifier_train = classifier_subparsers.add_parser(
        'train',
        help='Train the classifier'
    )
    classifier_train.add_argument(
        '--config',
        required=True,
        help='Path to config YAML file'
    )
    classifier_train.add_argument(
        '--resume',
        help='Resume from checkpoint (path to .pt file)'
    )

    args = parser.parse_args()
    if not args.experiment_type:
        parser.print_help()
        sys.exit(1)

    try:
        if args.experiment_type == 'classifier':
            if not args.classifier_command:
                classifier_parser.print_help()
                sys.exit(1)
            if args.classifier_command == 'train':
                train_classifier(args)
            else:
                classifier_parser.print_help()
                sys.exit(1)
        else:
            logger.error(f"Unknown experiment type: {args.experiment_type}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
