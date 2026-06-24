import os
import json
import gc
from safetensors.torch import load_file, save_file

model_dir = "../merged_model"
tensor_file_path = os.path.join(model_dir, "model.safetensors")
temp_file_path = os.path.join(model_dir, "model_temp.safetensors")

print("Loading safetensors...")
tensors = load_file(tensor_file_path)

# 1. Duplicate the tensor
if "lm_head.weight" not in tensors:
    print("lm_head.weight is missing. Duplicating model.embed_tokens.weight...")
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()

    # Write to a completely new file so Windows doesn't block us
    print("Saving modified tensors to a temporary file...")
    save_file(tensors, temp_file_path)

    print("Releasing memory map locks...")
    del tensors  # Destroy the variable holding the memory map
    gc.collect()  # Force Python to run garbage collection and release the file handle

    print("Swapping files...")
    os.remove(tensor_file_path)
    os.rename(temp_file_path, tensor_file_path)
    print("Safetensors updated successfully.")
else:
    print("lm_head.weight already exists. No tensor modification needed.")

# 2. Update the config.json
config_path = os.path.join(model_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

if config.get("tie_word_embeddings", False):
    print("Setting tie_word_embeddings to false in config.json...")
    config["tie_word_embeddings"] = False
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("config.json updated.")
else:
    print("tie_word_embeddings is already false.")

print("Process complete. You can now compile the GGUF.")