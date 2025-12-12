import json
import logging
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

logger = logging.getLogger(__name__)


class RepresentationEngineeringDatasetBuilder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.examples = []

    def add_example(
        self,
        original_weights: Dict[str, torch.Tensor],
        original_metrics: Dict[str, Dict[str, float]],
        modified_weights: Dict[str, torch.Tensor],
        modified_metrics: Dict[str, Dict[str, float]],
    ):
        original_weights_serialized = self._serialize_weights(original_weights)
        modified_weights_serialized = self._serialize_weights(modified_weights)
        example = {
            "original_model_weights": json.dumps(original_weights_serialized),
            "original_metrics": json.dumps(original_metrics),
            "modified_model_weights": json.dumps(modified_weights_serialized),
            "modified_metrics": json.dumps(modified_metrics),
        }
        self.examples.append(example)
        logger.info(f"Added example {len(self.examples)} to dataset builder")

    def add_from_results(self, results: List[Dict[str, Any]], input_weights_dict: Dict[int, Dict[str, torch.Tensor]]):
        for result in results:
            model_id = result["model_id"]
            result_id = result["result_id"]
            if model_id not in input_weights_dict:
                logger.warning(
                    f"Skipping result {result_id}: missing original weights for model {model_id}"
                )
                continue
            weights_path = self.run_dir / "modified_weights" / f"model_{model_id}_result_{result_id}.pt"
            if not weights_path.exists():
                logger.warning(f"Skipping result {result_id}: modified weights not found at {weights_path}")
                continue
            modified_weights = torch.load(weights_path, map_location="cpu")
            evaluation = result.get("evaluation", {})
            original_metrics = evaluation.get("original_metrics", {})
            modified_metrics = evaluation.get("modified_metrics", {})
            self.add_example(
                original_weights=input_weights_dict[model_id],
                original_metrics=original_metrics,
                modified_weights=modified_weights,
                modified_metrics=modified_metrics,
            )

    def build_dataset(self) -> Dataset:
        if not self.examples:
            raise ValueError("No examples added to dataset builder")
        logger.info(f"Building HuggingFace dataset with {len(self.examples)} examples")
        dataset = Dataset.from_list(self.examples)
        return dataset

    def upload_to_hub(self, repo_id: str, private: bool = False, token: Optional[str] = None):
        dataset = self.build_dataset()
        logger.info(f"Uploading dataset to HuggingFace Hub: {repo_id}")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
            logger.info(f"Repository created/verified: {repo_id}")
        except Exception as e:
            logger.warning(f"Could not create repository: {e}")
        dataset.push_to_hub(repo_id=repo_id, token=token, private=private)
        logger.info(f"✅ Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
        self._create_dataset_card(repo_id, token)

    def _serialize_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, List]:
        serialized = {}
        for key, tensor in weights.items():
            if isinstance(tensor, torch.Tensor):
                serialized[key] = tensor.cpu().tolist()
            else:
                serialized[key] = tensor
        return serialized

    def _create_dataset_card(self, repo_id: str, token: Optional[str] = None):
        total_examples = len(self.examples)
        operations = {}
        target_patterns = set()
        success_count = 0
        for example in self.examples:
            op = example["operation"]
            operations[op] = operations.get(op, 0) + 1
            patterns = json.loads(example["target_patterns"])
            target_patterns.update(patterns)
            metrics = json.loads(example["metrics"])
            if metrics.get("overall_success", False):
                success_count += 1
        success_rate = success_count / total_examples if total_examples > 0 else 0

        readme = f"""# Representation Engineering Dataset

This dataset contains results from applying Linear Representation Engineering (LRE) to modify neural network weights in latent space.

## Dataset Description

This dataset was generated using the MUAT (Model Understanding and Analysis Toolkit) representation engineering pipeline. Each example contains:

- **input_model_weights**: Original model state_dict (JSON)
- **input_signature**: Latent representation of input model (256-dim vector, JSON)
- **output_model_weights**: Modified model state_dict (JSON)
- **operation**: Type of modification ('add' or 'remove')
- **target_patterns**: List of patterns being modified (JSON)
- **strength**: Steering vector strength used
- **original_patterns**: List of patterns in original model (JSON)
- **metrics**: Evaluation metrics including accuracy deltas (JSON)
- **metadata**: Additional metadata about the model (JSON)

## Dataset Statistics

- **Total Examples**: {total_examples}
- **Operations**:
"""

        for op, count in operations.items():
            readme += f"  - {op}: {count} ({count / total_examples * 100:.1f}%)\n"

        readme += f"- **Target Patterns**: {', '.join(sorted(target_patterns))}\n"
        readme += f"- **Overall Success Rate**: {success_rate:.1%}\n"

        readme += """
## Usage

```python
from datasets import load_dataset
import json
import torch

# Load dataset
dataset = load_dataset("{repo_id}")

# Access an example
example = dataset['train'][0]

# Parse weights
input_weights = json.loads(example['input_model_weights'])
output_weights = json.loads(example['output_model_weights'])

# Convert back to tensors
input_weights_tensors = {k: torch.tensor(v) for k, v in input_weights.items()}
output_weights_tensors = {k: torch.tensor(v) for k, v in output_weights.items()}

# Parse latent signature
input_signature = torch.tensor(json.loads(example['input_signature']))

# Parse metrics
metrics = json.loads(example['metrics'])
print(f"Accuracy delta: {{metrics['accuracy_delta']:.1%}}")
```

## Method

**Linear Representation Engineering (LRE)**:
1. Encode model weights to 256-dim latent space using pre-trained encoder
2. Compute steering vector: `mean(models_with_pattern) - mean(models_without_pattern)`
3. Apply steering vector in latent space: `latent_modified = latent + strength * steering_vector`
4. Decode back to weight space

## Citation

If you use this dataset, please cite:

```bibtex
@software{{muat_representation_engineering,
  title = {{MUAT Representation Engineering Dataset}},
  author = {{MUAT Team}},
  year = {{2025}},
  url = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

[Add your license here]
"""
        readme = readme.replace("{repo_id}", repo_id)

        try:
            readme_api = HfApi(token=token)
            readme_api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("Dataset card (README.md) uploaded successfully")
        except Exception as e:
            logger.warning(f"Failed to upload dataset card: {e}")

    def save_local(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self.build_dataset()
        dataset.save_to_disk(str(output_dir))
        logger.info(f"Dataset saved locally to: {output_dir}")
        json_path = output_dir / "dataset.json"
        with open(json_path, "w") as f:
            json.dump(self.examples, f, indent=2)
        logger.info(f"Dataset also saved as JSON: {json_path}")
