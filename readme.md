I'll write about the whole idea later


---

## Quick Start:

### Generate Dataset for Training Interpreter: 

**Initial Setup**

1. `cd dataset_generation/pipeline`
2. `pip install -r requirements.txt`

**Run**

1. Copy/modify the [example_config.yaml](dataset_generation/pipeline/config/example_config.yaml) file for your specific training plan (all dataset config done here).

2. Generate the signature dataset
    - This is the dataset that gets passed through the subject models (held constant across the examples), and what the interpreter model effectively learns to use as a key to understanding the activations of the subject models. 
    - You don't have to generate a new one for each dataset generation, but the vocab size and sequence length need to match the subject models of your dataset (this command uses the ones in the config).
    - `python cli.py create-sig-dataset --config-path path/to/config.yaml --filename output_filename.json --size 75`

3. *(IGNORE FOR NOW) Generate a benchmark dataset (to implement later):*
    - This will evenutally be used in the evaluation pipeline of the interpreter model (modification task) so that we can check how well it improves preformance on different patterns.
    - Also needs to match subject models sequence length and vocab size.
    - `python cli.py create-benchmark-dataset --config-path pipeline/config/example_config.yaml --samples-per-pattern 35 --filename benchmark_dataset.json`

4. Run the dataset generation pipeline: `python cli.py run-data-gen --config-path pipeline/config/example_config.yaml`
