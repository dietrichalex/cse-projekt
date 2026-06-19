from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "meta-llama/Llama-3.2-1B"
adapter_dir = "./llama-fine-tuned"
output_dir = "./merged_model"

# 1. Load base model and tokenizer into RAM
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16)

# 2. Attach adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = model.merge_and_unload()

# 3. Save the unified model and tokenizer
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)