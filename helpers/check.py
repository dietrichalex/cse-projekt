import json

with open("../../merged_model/tokenizer.json", "r", encoding="utf-8") as f:
    tok = json.load(f)

# Print just the model section minus the huge vocab/merges lists
model = {k: v for k, v in tok["model"].items() if k not in ("vocab", "merges")}
print(json.dumps(model, indent=2))