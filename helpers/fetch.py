from huggingface_hub import hf_hub_download

repo = "unsloth/Llama-3.2-1B-Instruct"
dest = "./merged_model"
files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

for f in files:
    hf_hub_download(
        repo_id=repo,
        filename=f,
        local_dir=dest,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded {f}")