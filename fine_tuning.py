import torch, pandas as pd
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments
)

# ===============================
# Configuration
# ===============================

data = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
data.columns = data.columns.str.replace('Column1.', '', regex=False)

MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Change to the Hugging Face model repo ID
OUTPUT_DIR = "./llama-fine-tuned"  # Directory to save the fine-tuned model
DATASET_FILE = data  # Local dataset file
MAX_SEQ_LENGTH = 512  # Maximum sequence length
BATCH_SIZE = 4  # Batch size per device
EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 5e-5  # Learning rate
LOGGING_DIR = "./logs"  # Directory for logging

# ===============================
# Load Model and Tokenizer
# ===============================
print("Loading model and tokenizer from Hugging Face...")

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME)

# ===============================
# Load and Preprocess Dataset
# ===============================
print("Loading and preprocessing dataset...")
# Load dataset from local JSON file
dataset = load_dataset("csv", data_files=DATASET_FILE)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)

tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

# ===============================
# Training Configuration
# ===============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_steps=500,
    logging_dir=LOGGING_DIR,
    logging_steps=100,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,  # Ensure your private model stays local
)

# ===============================
# Initialize Trainer
# ===============================
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# ===============================
# Fine-Tuning
# ===============================
print("Starting fine-tuning...")
trainer.train()

# ===============================
# Save Fine-Tuned Model
# ===============================
print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete!")
