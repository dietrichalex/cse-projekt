from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../merged_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

inputs = tokenizer("Hello, who are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)

print("\n--- NATIVE OUTPUT ---")
print(tokenizer.decode(outputs[0]))