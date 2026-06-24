import torch, pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification
)
import numpy as np
import os
import sys
from peft import get_peft_model, LoraConfig, TaskType
from transformers.trainer_utils import get_last_checkpoint

print("MY EXACT PYTHON PATH:", sys.executable)
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

os.environ["WANDB_MODE"] = "offline"

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-fine-tuned"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
LOGGING_DIR = "./logs"

print("Loading model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 1. CRITICAL: Set padding side to right so pooling grabs the correct final token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# Load the BASE model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Pass the pad_token_id to the model explicitly
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# 2. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["score"]  # <-- CRITICAL: Explicitly mark the regression head as trainable
)

# Wrap the base model with the LoRA adapters
model = get_peft_model(model, peft_config)

# 3. CRITICAL FIXES: Cast parameters and Zero-Initialize the score head
for name, param in model.named_parameters():
    if param.requires_grad:
        # Cast LoRA adapters and the score head to float32 for optimizer stability
        param.data = param.data.to(torch.float32)

        # Zero out the regression head to prevent massive initial 7,000,000+ loss
        if "score" in name:
            param.data.zero_()

# Print out how many parameters you are actually training now
model.print_trainable_parameters()


print("Loading and preprocessing dataset...")
dataset = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
dataset.columns = dataset.columns.str.replace('Column1.', '', regex=False)

# 1. CRITICAL DATA CLEANING: Fix European decimals and force numeric types
if dataset["Rating"].dtype == object:
    # If ratings have commas instead of decimals (e.g. "8,5"), fix them
    dataset["Rating"] = dataset["Rating"].astype(str).str.replace(',', '.')

# Force the column to be numeric. Any weird text or missing data becomes a safe pd.NA / np.nan
dataset["Rating"] = pd.to_numeric(dataset["Rating"], errors="coerce")


# Clean and preprocess texts
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return str(text).strip()

texts = [clean_text(text) for text in dataset["Comment"].tolist()]
rating = dataset["Rating"].tolist()

# 2. CRITICAL FIX: Filter out BOTH empty texts AND missing ratings
# pd.notna(r) guarantees no NaNs sneak into our training batches
valid_data = [(t, r) for t, r in zip(texts, rating) if t and pd.notna(r)]
texts, rating = zip(*valid_data) if valid_data else ([], [])

if not texts:
    raise ValueError("No valid text data found after cleaning")

# 3. Safe Scaling
import numpy as np
rating_array = np.array(rating)
min_rating = rating_array.min()
max_rating = rating_array.max()

# Squeeze values between 0.0 and 1.0
if max_rating > min_rating:
    rating = ((rating_array - min_rating) / (max_rating - min_rating)).tolist()
else:
    rating = rating_array.tolist()

# Split
train_texts, val_texts, train_rating, val_rating = train_test_split(
    texts, rating, test_size=0.2, random_state=42
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False


def tokenize_texts(texts):
    # Ensure all texts are strings and handle batch processing
    cleaned_texts = [str(text).strip() for text in texts]

    # Process in batches to avoid potential memory issues
    batch_size = 32
    all_encodings = {
        'input_ids': [],
        'attention_mask': []
    }

    for i in range(0, len(cleaned_texts), batch_size):
        batch_texts = cleaned_texts[i:i + batch_size]
        batch_encodings = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None
        )

        all_encodings['input_ids'].extend(batch_encodings['input_ids'])
        all_encodings['attention_mask'].extend(batch_encodings['attention_mask'])

    return all_encodings


print("Tokenizing data...")
try:
    train_encodings = tokenize_texts(train_texts)
    val_encodings = tokenize_texts(val_texts)
except Exception as e:
    print(f"Error during tokenization: {str(e)}")
    print(f"Sample of texts: {train_texts[:5]}")
    raise


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}

        # CRITICAL FIX: The brackets [ ] around self.targets[idx] force it into a 2D shape,
        # preventing the catastrophic PyTorch broadcasting bug.
        item['labels'] = torch.tensor([self.targets[idx]], dtype=torch.float)

        return item


print("Creating datasets...")
train_dataset = RegressionDataset(train_encodings, train_rating)
val_dataset = RegressionDataset(val_encodings, val_rating)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - labels))
    return {"mse": mse, "rmse": rmse, "mae": mae}


print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",

# --- NEW SAFEGUARDS HERE ---
    logging_steps=10,        # Watch the loss closely from the very beginning
    warmup_steps=200,        # Slowly ease the learning rate in over 200 steps
    max_grad_norm=0.5,       # Stricter gradient clipping to stop sudden explosions
    # ---------------------------

    save_total_limit=2,
    logging_dir=LOGGING_DIR,
    load_best_model_at_end=True,
    report_to=[],
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Training model...")

# Check if the output directory exists and has checkpoints
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

# Tell the trainer to resume if a checkpoint is found
if last_checkpoint is not None:
    print(f"Found checkpoint at {last_checkpoint}. Resuming training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No existing checkpoints found. Starting training from scratch...")
    trainer.train()

print("Evaluating model...")
results = trainer.evaluate()
print("Evaluation results:", results)

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete!")