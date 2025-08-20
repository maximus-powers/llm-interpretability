# Lambda Labs H100 Setup Commands

ssh -i lambdalabs.pem ubuntu@104.171.203.31



pip install --upgrade autotrain-advanced jinja2 tf-keras triton bitsandbytes


### RUN
autotrain llm \
  --train --model aws-prototyping/MegaBeam-Mistral-7B-512k \
  --project-name llm-interpretability-mistral7b-lora \
  --data-path maximuspowers/llm-interpretability-v1-messages \
  --chat-template chatml \
  --text-column messages \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --model_max_length 500000 \
  --block-size 500000 \
  --batch-size 1 \
  --epochs 3 \
  --lr 2e-5 \
  --gradient-accumulation 2 \
  --mixed_precision fp16 \
  --peft \
  --quantization int4 \
  --max-prompt-length 300000 \
  --push-to-hub \
  --username maximuspowers \
  --model-ref maximuspowers/MegaBeam-Mistral-7B-512k-LORA \
  --token 

