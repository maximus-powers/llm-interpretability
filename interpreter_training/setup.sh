# create venv
python3 -m venv arctic_venv
source arctic_venv/bin/activate
pip install --upgrade pip

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install transformers>=4.51.3
pip install deepspeed>=0.17.0
pip install liger-kernel
pip install datasets accelerate
pip install pyyaml wandb

# arctic trainer
git clone https://github.com/snowflakedb/ArcticTraining
cd ArcticTraining
pip install .
cd projects/sequence-parallelism

# env vars
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=

# training config file
cat > run-a100-sp8-llama-8b-custom.yml << 'EOF'
type: sft
micro_batch_size: 1
exit_iteration: 10
min_iterations: 5
activation_checkpoint_cpu_offload: true
tiled_mlp_compute: true
sequence_parallel_size: 8
deepspeed:
  zero_optimization:
    stage: 3
    offload_optimizer:
      device: cpu
  seq_parallel_communication_data_type: bf16
optimizer:
  type: cpu_adam
  learning_rate: 0.00001
model:
  type: "liger"
  name_or_path: meta-llama/Llama-3.1-8B-Instruct
  attn_implementation: sdpa # get this to be flash_attention_2 later
data:
  sources:
    - type: huggingface_instruct
      name_or_path: maximuspowers/llm-interpretability-v1-messages
      split: train
      role_mapping:
        user: prompt
        assistant: completion
  cache_dir: data-cache
  dl_num_workers: 1
  max_length: 500000
logger:
  level: WARNING
  output_dir: "logs"
  print_output_ranks: [0,1,2,3,4,5,6,7]
checkpoint:
  - type: huggingface
    save_end_of_training: true
    output_dir: ./fine-tuned-model
EOF

# RUN COMMAND: arctic_training run-a100-sp8-llama-8b-custom.yml
# monitor memory in another terminal: watch -n 1 nvidia-smi

# upload
pip install huggingface-hub
huggingface-cli login --token ""
huggingface-cli upload maximuspowers/llama-3.1-8b-interpreter ./fine-tuned-model
