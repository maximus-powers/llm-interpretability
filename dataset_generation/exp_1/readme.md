These are the configs for our first version of the experiment. Since training is such a pain and super expensive, we're going to start of with the simplest of tasks and work our way up.

Experiment 1 will be all classification tasks (focusing on the question of interpretability instead of modification). The variations in config will be the signature structure and subject model complexity.

**Simple pattern datasets**
- vocab size: 5
- sequence length: 5

**Small model size:**
- layers: 3-5
- neurons per layer: 5-8
- precision: float16

**Signature Complexities**

| config_path              | stat_methods    | interwoven |
|--------------------------|-----------------|------------|
| small_simple.yaml        | mean, stdev     | No         |
| small_medium.yaml        | mean, pca (3)   | No         |
| small_complex.yaml       | pattern wise    | No         |
| small_complex_inter.yaml | pattern wise    | Yes        |

**Patterns**
- One pattern included in each dataset, we're just training it to classify one pattern, not noisy models that have multiple patterns.