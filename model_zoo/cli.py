#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path
import yaml
import threading
import subprocess
import atexit
import torch
import questionary
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import time
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_zoo.dataset_generation import DatasetGenerationPipeline, PatternDatasetSampler
from model_zoo.classification_training import (
    load_dataset,
    create_dataloaders,
    compute_model_architecture,
    PatternClassifierMLP,
    ClassifierTrainer,
    load_checkpoint
)
from model_zoo.encoder_decoder_training import (
    load_dataset as load_dataset_encoder_decoder,
    create_dataloaders as create_dataloaders_encoder_decoder,
    create_encoder_decoder,
    EncoderDecoderTrainer,
    load_checkpoint as load_checkpoint_encoder_decoder
)

# ========== Logging Setup ==========
_thread_local = threading.local()
def set_example_id(example_id):
    _thread_local.example_id = example_id

def get_example_id():
    return getattr(_thread_local, 'example_id', None)

class ColoredExampleFormatter(logging.Formatter):
    COLORS = [
        '\033[94m',  # Blue
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[95m',  # Magenta
        '\033[96m',  # Cyan
        '\033[91m',  # Red
        '\033[97m',  # White
        '\033[90m',  # Gray
    ]
    RESET = '\033[0m'
    def format(self, record):
        example_id = get_example_id()
        formatted = super().format(record)
        if example_id is not None:
            color = self.COLORS[example_id % len(self.COLORS)]
            parts = formatted.split(' - ', 1)
            if len(parts) == 2:
                return f"{parts[0]} - {color}[Ex{example_id}] {parts[1]}{self.RESET}"
            return f"{color}[Ex{example_id}] {formatted}{self.RESET}"
        return formatted

def setup_data_gen_logging():
    formatter = ColoredExampleFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

def setup_experiment_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_run_directory(step_name: str, config_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{step_name}_{Path(config_path).stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

############## Batch Execution Helpers ##############
class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class BatchExecutionResult:
    config_path: Path
    status: ExecutionStatus
    error_message: Optional[str] = None
    run_dir: Optional[Path] = None
    execution_time: float = 0.0

def discover_yaml_configs(path: Path):
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    yaml_files = sorted(path.glob("*.yaml"), key=lambda p: p.name.lower())
    if not yaml_files:
        raise ValueError(f"No .yaml files found in directory: {path}")
    return yaml_files

def validate_config_file(config_path: Path):
    if not config_path.exists():
        return False, f"File not found: {config_path}"
    if not config_path.is_file():
        return False, f"Path is not a file: {config_path}"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            return False, f"Empty or invalid YAML: {config_path}"
    except yaml.YAMLError as e:
        return False, f"YAML parse error in {config_path.name}: {e}"
    except Exception as e:
        return False, f"Failed to read {config_path.name}: {e}"
    return True, None

def validate_all_configs(config_paths: List[Path]):
    errors = []
    for config_path in config_paths:
        is_valid, error_msg = validate_config_file(config_path)
        if not is_valid:
            errors.append(error_msg)
    return len(errors) == 0, errors

def execute_single_config(handler_func, args, config_path: Path, index: int, total: int):
    logging.info(f"\n\n\nProcessing config [{index}/{total}]: {config_path.name}")
    start_time = time.time()
    try:
        handler_func(args)
        execution_time = time.time() - start_time
        runs_dir = Path("runs")
        if runs_dir.exists():
            run_dirs = sorted(runs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            run_dir = run_dirs[0] if run_dirs else None
        else:
            run_dir = None
        return BatchExecutionResult(
            config_path=config_path,
            status=ExecutionStatus.SUCCESS,
            run_dir=run_dir,
            execution_time=execution_time
        )
    except KeyboardInterrupt:
        logging.warning(f"Interrupted by user during {config_path.name}")
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(f"Failed to process {config_path.name}: {e}", exc_info=True)
        return BatchExecutionResult(
            config_path=config_path,
            status=ExecutionStatus.FAILED,
            error_message=str(e),
            execution_time=execution_time
        )

def execute_batch_configs(handler_func, args_template, config_paths: List[Path], config_arg_name: str):
    results = []
    total = len(config_paths)
    logging.info(f"BATCH EXECUTION: {total} configs to process")
    for idx, config_path in enumerate(config_paths, start=1):
        args = argparse.Namespace(**vars(args_template))
        setattr(args, config_arg_name, str(config_path))
        setattr(args, 'is_batch_item', True)  # Flag to indicate this is part of batch execution
        result = execute_single_config(handler_func, args, config_path, idx, total)
        results.append(result)
        successes = sum(1 for r in results if r.status == ExecutionStatus.SUCCESS)
        logging.info(f"\nProgress: {idx}/{total} batches complete ({successes} successful)\n")
    return results

def cleanup_run_directory(run_dir: Path, config: dict):
    if not config.get('run_log_cleanup', False):
        return
    run_dir_abs = run_dir.absolute()
    runs_root = Path("runs").absolute()
    # verify it's in runs/
    if not str(run_dir_abs).startswith(str(runs_root)):
        logging.warning(f"Skipping cleanup: {run_dir} is not under runs/ directory")
        return
    if not run_dir.exists():
        logging.warning(f"Skipping cleanup: {run_dir} does not exist")
        return
    shutil.rmtree(run_dir)
    logging.info(f"✓ Cleaned up run directory: {run_dir}")

############## Dataset Generation Handlers ##############
def run_data_gen(args):
    setup_data_gen_logging()
    config_path = Path(args.config_path)

    # batch execution
    if config_path.is_dir():
        logging.info(f"Directory detected: {config_path}")
        try:
            # find configs
            config_files = discover_yaml_configs(config_path)
            logging.info(f"Found {len(config_files)} config files")
            # validate configs
            all_valid, errors = validate_all_configs(config_files)
            if not all_valid:
                logging.error("Config validation failed:")
                for error in errors:
                    logging.error(f"  - {error}")
                sys.exit(1)
            # execute batch
            results = execute_batch_configs(
                handler_func=run_data_gen,
                args_template=args,
                config_paths=config_files,
                config_arg_name='config_path'
            )
            failures = [r for r in results if r.status == ExecutionStatus.FAILED]
            if failures:
                sys.exit(1)
            return
        except Exception as e:
            logging.error(f"Batch execution failed: {e}")
            sys.exit(1)

    # single config execution
    if not config_path.exists():
        logging.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)

    # create run dir for logs and outputs
    run_dir = create_run_directory("data-generation", args.config_path)
    logging.info(f"Run directory: {run_dir.absolute()}")
    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # add run directory to config
    config['run_dir'] = str(run_dir.absolute())

    # save config to run dir
    run_config_path = run_dir / "config.yaml"
    with open(run_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # metrics config
    metrics_config = config.get('metrics', {})

    # start tb
    tensorboard_process = None
    tensorboard_config = metrics_config.get('tensorboard', {})
    tensorboard_enabled = tensorboard_config.get('enabled', False)
    tensorboard_auto_launch = tensorboard_config.get('auto_launch', False)
    if tensorboard_enabled and tensorboard_auto_launch:
        port = tensorboard_config.get('port', 6006)
        metrics_dir = run_dir / "metrics"
        try:
            tensorboard_process = subprocess.Popen(
                ['tensorboard', '--logdir', str(metrics_dir.absolute()), '--port', str(port), '--bind_all'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            def cleanup_tensorboard():
                if tensorboard_process and tensorboard_process.poll() is None:
                    logging.info("Shutting down TensorBoard...")
                    tensorboard_process.terminate()
                    try:
                        tensorboard_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        tensorboard_process.kill()
            atexit.register(cleanup_tensorboard)
            logging.info(f"TensorBoard: http://localhost:{port}")
        except Exception as e:
            logging.warning(f"Failed to start TensorBoard: {e}")

    logging.info("Initializing pipeline...")
    try:
        pipeline = DatasetGenerationPipeline(
            config=config,
            example_id_setter=set_example_id
        )
        logging.info("Starting dataset generation...")
        examples = pipeline.generate_training_examples()
        logging.info(f"Dataset generation completed! Generated {len(examples)} examples.")

        # only shutdown tb in batch mode
        if getattr(args, 'is_batch_item', False):
            if tensorboard_process and tensorboard_process.poll() is None:
                logging.info("Shutting down TensorBoard")
                tensorboard_process.terminate()
                tensorboard_process.wait(timeout=5)

        cleanup_run_directory(run_dir, config)

    except Exception as e:
        logging.error(f"Dataset generation failed: {e}")
        sys.exit(1)

def create_sig_dataset(args):
    setup_data_gen_logging()
    config_path = Path(args.config_path)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {args.config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vocab_size = config['model']['vocab_size']
    sequence_length = config['model']['sequence_length']
    sampler = PatternDatasetSampler(vocab_size=vocab_size, sequence_length=sequence_length)
    logging.info(f"Creating signature dataset: {args.filename} with {args.size} examples (vocab_size={vocab_size}, sequence_length={sequence_length})")
    output_path = sampler.create_signature_dataset_file(
        filename=args.filename,
        total_examples=args.size
    )
    logging.info(f"Signature dataset created successfully: {output_path}")

############## Experiments Handlers ##############
def train_classifier(args):
    setup_experiment_logging()
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)

    # batch execution
    if config_path.is_dir():
        logger.info(f"Directory detected: {config_path}")
        try:
            # find configs
            config_files = discover_yaml_configs(config_path)
            logger.info(f"Found {len(config_files)} config files")
            # validate configs
            all_valid, errors = validate_all_configs(config_files)
            if not all_valid:
                logger.error("Config validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                sys.exit(1)
            # execute batch
            results = execute_batch_configs(
                handler_func=train_classifier,
                args_template=args,
                config_paths=config_files,
                config_arg_name='config'
            )
            failures = [r for r in results if r.status == ExecutionStatus.FAILED]
            if failures:
                logging.error(f"{len(failures)} configurations failed during batch execution:")
                for failure in failures:
                    logging.error(f"  - {failure.config_path}")
            return
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            sys.exit(1)

    # single config execution
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    run_dir = create_run_directory("train-classifier", args.config)
    logger.info(f"\n{'='*70}")
    logger.info(f"📁 Run directory: {run_dir.absolute()}")
    logger.info(f"{'='*70}\n")

    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # add run dir to config
    config['run_dir'] = str(run_dir.absolute())

    # save config to run dir
    run_config_path = run_dir / "config.yaml"
    with open(run_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

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

    # only shutdown tb in batch mode
    if getattr(args, 'is_batch_item', False):
        trainer.stop_tensorboard()

    cleanup_run_directory(run_dir, config)


def train_encoder_decoder(args):
    setup_experiment_logging()
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)

    # batch execution
    if config_path.is_dir():
        logger.info(f"Directory detected: {config_path}")
        try:
            config_files = discover_yaml_configs(config_path)
            logger.info(f"Found {len(config_files)} config files")
            all_valid, errors = validate_all_configs(config_files)
            if not all_valid:
                logger.error("Config validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                sys.exit(1)
            results = execute_batch_configs(
                handler_func=train_encoder_decoder,
                args_template=args,
                config_paths=config_files,
                config_arg_name='config'
            )
            failures = [r for r in results if r.status == ExecutionStatus.FAILED]
            if failures:
                logging.error(f"{len(failures)} configurations failed during batch execution:")
                for failure in failures:
                    logging.error(f"  - {failure.config_path}")
            return
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            sys.exit(1)

    # single config execution
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    run_dir = create_run_directory("train-encoder-decoder", args.config)
    logger.info(f"\n{'='*70}")
    logger.info(f"📁 Run directory: {run_dir.absolute()}")
    logger.info(f"{'='*70}\n")

    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['run_dir'] = str(run_dir.absolute())

    # save config to run dir
    run_config_path = run_dir / "config.yaml"
    with open(run_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # load dataset and create tokenizer
    dataset_info = load_dataset_encoder_decoder(config)

    # create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_encoder_decoder(dataset_info, config)

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

    logger.info(f"Using device: {device}")

    # initialize model
    model = create_encoder_decoder(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created {config['architecture']['type']} encoder-decoder")
    logger.info(f"Model parameters: {total_params:,}")

    # checkpoint resumption
    if args.resume:
        checkpoint_info = load_checkpoint_encoder_decoder(args.resume, model)
        logger.info(f"Resuming from epoch {checkpoint_info['epoch'] + 1}")

    # initialize trainer
    logger.info("Starting training")
    trainer = EncoderDecoderTrainer(
        model, config, train_loader, val_loader, device,
        dataset_info['tokenizer'], test_loader
    )
    trainer.train()
    logger.info("Training complete")

    # only shutdown tb in batch mode
    if getattr(args, 'is_batch_item', False):
        trainer.stop_tensorboard()

    cleanup_run_directory(run_dir, config)

############## CLI ##############
def run_interactive_mode():
    print("\n" + "="*70)
    print("Model Zoo CLI")
    print("="*70 + "\n")

    try:
        while True:
            action = questionary.select("Select a step:", choices=[
                    "Dataset Generation",
                    "MLP Classifier Training",
                    "Encoder-Decoder Training",
                    "Exit"
                ]
            ).ask()

            if action is None or action == "Exit":
                print("\nExiting Model Zoo CLI...")
                break

            if action == "Dataset Generation":
                operation = questionary.select("Select an operation:", choices=[
                            "Run Generation Pipeline",
                            "Create Signature Dataset"
                            # add benchmark dataset gen later
                        ]
                    ).ask()

                if operation == "Run Generation Pipeline":
                    config_path = questionary.path("Config(s) path (filename for one, dir for multiple):", default="configs/dataset_gen/").ask()
                    if config_path:
                        args = argparse.Namespace(config_path=config_path)
                        run_data_gen(args)

                elif operation == "Create Signature Dataset":
                    config_path = questionary.path("Config file path:", default="configs/dataset_gen/").ask()
                    if not config_path:
                        continue
                    filename = questionary.text("Output filename:", default="signature_dataset.json").ask()
                    size = questionary.text("Number of examples:", default="200").ask()
                    if filename and size:
                        args = argparse.Namespace(config_path=config_path, filename=filename, size=int(size))
                        create_sig_dataset(args)

            elif action == "MLP Classifier Training":
                config_path = questionary.path(
                    "Config(s) path (filename for one, dir for multiple):",
                    default="configs/classification/"
                ).ask()

                if not config_path:
                    continue

                args = argparse.Namespace(
                    config=config_path,
                    resume=None
                )
                train_classifier(args)

            elif action == "Encoder-Decoder Training":
                config_path = questionary.path(
                    "Config(s) path (filename for one, dir for multiple):",
                    default="model_zoo/encoder_decoder_training/example_config.yaml"
                ).ask()
                if not config_path:
                    continue
                args = argparse.Namespace(config=config_path, resume=None)
                train_encoder_decoder(args)

    except KeyboardInterrupt:
        print("\n\nExiting Model Zoo CLI...")
        sys.exit(0)    

def run_traditional_cli():
    parser = argparse.ArgumentParser(
        prog='muat-cli',
        description='Unified CLI for MUAT dataset generation and experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='category', help='Command category')

    ###### dataset generation ######
    data_parser = subparsers.add_parser('data', help='Dataset generation commands')
    data_subparsers = data_parser.add_subparsers(dest='data_command', help='Data operations')

    # data run-data-gen
    run_parser = data_subparsers.add_parser(
        'run-data-gen',
        help='Run the full dataset generation pipeline'
    )
    run_parser.add_argument(
        '--config-path',
        type=str,
        default='configs/dataset_gen/example_config.yaml',
        help='Path to configuration YAML file or directory containing multiple configs'
    )

    # data create-sig-dataset
    sig_parser = data_subparsers.add_parser(
        'create-sig-dataset',
        help='Create a signature dataset file for activation extraction'
    )
    sig_parser.add_argument(
        '--config-path',
        type=str,
        default='configs/dataset_gen/example_config.yaml',
        help='Path to configuration YAML file or directory containing multiple configs'
    )
    sig_parser.add_argument(
        '--filename',
        type=str,
        default='signature_dataset.json',
        help='Output filename for signature dataset'
    )
    sig_parser.add_argument(
        '--size',
        type=int,
        default=200,
        help='Total number of examples in signature dataset'
    )

    ###### experiment ######
    experiment_parser = subparsers.add_parser('experiment', help='Experiment commands')
    experiment_subparsers = experiment_parser.add_subparsers(dest='experiment_type', help='Experiment type')

    # experiment classifier
    classifier_parser = experiment_subparsers.add_parser('classifier', help='Pattern classifier experiments')
    classifier_subparsers = classifier_parser.add_subparsers(dest='classifier_command', help='Classifier commands')

    # experiment classifier train
    classifier_train = classifier_subparsers.add_parser('train', help='Train the classifier')
    classifier_train.add_argument(
        '--config',
        required=True,
        help='Path to config YAML file or directory containing multiple configs'
    )
    classifier_train.add_argument(
        '--resume',
        help='Resume from checkpoint (path to .pt file)'
    )

    # experiment encoder_decoder
    encoder_decoder_parser = experiment_subparsers.add_parser('encoder-decoder', help='Weight-space encoder-decoder experiments')
    encoder_decoder_subparsers = encoder_decoder_parser.add_subparsers(dest='encoder_decoder_command', help='Encoder-decoder commands')

    # experiment encoder_decoder train
    encoder_decoder_train = encoder_decoder_subparsers.add_parser('train', help='Train the encoder-decoder')
    encoder_decoder_train.add_argument(
        '--config',
        required=True,
        help='Path to config YAML file or directory containing multiple configs'
    )
    encoder_decoder_train.add_argument(
        '--resume',
        help='Resume from checkpoint (path to .pt file)'
    )

    args = parser.parse_args()

    if not args.category:
        parser.print_help()
        sys.exit(1)

    try:
        if args.category == 'data':
            if not args.data_command:
                data_parser.print_help()
                sys.exit(1)
            if args.data_command == 'run-data-gen':
                run_data_gen(args)
            elif args.data_command == 'create-sig-dataset':
                create_sig_dataset(args)
        elif args.category == 'experiment':
            if not args.experiment_type:
                experiment_parser.print_help()
                sys.exit(1)
            if args.experiment_type == 'classifier':
                if not args.classifier_command:
                    classifier_parser.print_help()
                    sys.exit(1)
                if args.classifier_command == 'train':
                    train_classifier(args)
            elif args.experiment_type == 'encoder-decoder':
                if not args.encoder_decoder_command:
                    encoder_decoder_parser.print_help()
                    sys.exit(1)
                if args.encoder_decoder_command == 'train':
                    train_encoder_decoder(args)
    except KeyboardInterrupt:
        logging.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

def main():
    if len(sys.argv) == 1:
        run_interactive_mode()
    else:
        run_traditional_cli()

if __name__ == "__main__":
    main()
