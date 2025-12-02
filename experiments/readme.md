# Experiments

Experiments for LLM interpretability research.

## Setup

```bash
# Install dependencies for specific experiment
cd experiments/classification
pip install -r requirements.txt
```

## Classifier

Multi-label classifier that predicts which patterns a subject model was trained on.

**Input modes:**
- Signature only
- Weights only
- Both signature and weights

Handles variable-size architectures. Training includes validation, early stopping, TensorBoard logging, and checkpointing.

### Usage

```bash
# Train
python -m experiments.cli classifier train --config path/to/config.yaml

# Resume from checkpoint
python -m experiments.cli classifier train --config path/to/config.yaml --resume checkpoints/classifier/best_model.pt
```

## Adding Experiments

Create a new directory under `experiments/`, implement your code, then add command functions and routing to `experiments/cli.py`:

```python
# Add command function
def train_my_experiment(args):
    from .my_experiment.trainer import train
    # Implementation...

# Add subparser
my_exp_parser = subparsers.add_parser('my_experiment', help='...')
my_exp_subparsers = my_exp_parser.add_subparsers(dest='my_experiment_command')

# Add routing
if args.experiment_type == 'my_experiment':
    if args.my_experiment_command == 'train':
        train_my_experiment(args)
```
