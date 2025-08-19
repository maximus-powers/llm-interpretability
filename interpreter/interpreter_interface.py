import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import json
import logging
from typing import Dict, Any, Optional
import ast

logger = logging.getLogger(__name__)


class InterpreterInterface:
    """
    Interface for running the trained StarCoder2 interpreter model.
    Handles prompt formatting, inference, and weight extraction from generated responses.
    """
    
    def __init__(self, model_name: str = "maximuspowers/starcoder2-7b-interpreter", device: str = "auto"):
        self.model_name = model_name
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"🤖 Initializing InterpreterInterface with model: {model_name}")
        logger.info(f"🖥️  Device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned interpreter model."""
        try:
            base_model_id = "bigcode/starcoder2-7b"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load LoRA adapter and merge
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
            self.model = self.model.merge_and_unload()
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_completion(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.1) -> str:
        """
        Generate completion for the given prompt.
        
        Args:
            prompt: Input prompt (should be formatted training prompt)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature (low for consistency)
            
        Returns:
            Generated completion text
        """
        try:
            # Format the full prompt with completion instruction
            full_prompt = prompt + "\n\n## Generate Improved Model Weights\n\nHere are the improved model weights that will correctly classify the target pattern:\n\n"
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=8192  # Leave room for generation
            ).to(self.model.device)
            
            logger.info(f"🔤 Input tokens: {inputs['input_ids'].shape[1]}")
            
            # Generate with low temperature for consistency
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            logger.info(f"✅ Generated {len(generated_tokens)} tokens")
            
            return completion
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def extract_weights_from_response(self, response: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract model weights from the generated response.
        
        Args:
            response: Generated completion text containing weights
            
        Returns:
            Dictionary of layer names to weight tensors, or None if extraction fails
        """
        try:
            # Look for weight dictionary patterns in the response
            # The training format should generate something like:
            # {'network.0.weight': [[...]], 'network.0.bias': [...], ...}
            
            # Find JSON-like dictionary patterns
            dict_pattern = r'\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}'
            matches = re.findall(dict_pattern, response, re.DOTALL)
            
            if not matches:
                logger.warning("⚠️  No weight dictionary found in response")
                return None
            
            # Try to parse the largest match (most likely to be complete)
            largest_match = max(matches, key=len)
            
            try:
                # Try JSON parsing first
                weights_dict = json.loads(largest_match)
            except json.JSONDecodeError:
                try:
                    # Try ast.literal_eval as fallback
                    weights_dict = ast.literal_eval(largest_match)
                except (ValueError, SyntaxError):
                    logger.warning("⚠️  Failed to parse weight dictionary")
                    return None
            
            # Convert lists to tensors
            tensor_dict = {}
            for key, value in weights_dict.items():
                if isinstance(value, list):
                    tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (int, float)):
                    tensor_dict[key] = torch.tensor([value], dtype=torch.float32)
                else:
                    logger.warning(f"⚠️  Unexpected type for {key}: {type(value)}")
                    continue
            
            logger.info(f"✅ Extracted weights for {len(tensor_dict)} layers")
            return tensor_dict
            
        except Exception as e:
            logger.error(f"❌ Weight extraction failed: {e}")
            return None
    
    def generate_and_extract_weights(self, prompt: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convenience method to generate completion and extract weights in one call.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dictionary of extracted weights, or None if failed
        """
        try:
            # Generate completion
            completion = self.generate_completion(prompt)
            
            # Extract weights
            weights = self.extract_weights_from_response(completion)
            
            if weights is None:
                logger.warning("⚠️  Could not extract weights from generated response")
                logger.debug(f"Response preview: {completion[:200]}...")
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ Generate and extract failed: {e}")
            return None