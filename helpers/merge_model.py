from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"  # ← changed
adapter_dir = "../llama-fine-tuned"
output_dir = "../merged_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16)

model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)