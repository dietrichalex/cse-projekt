import torch, pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
import os

os.environ["WANDB_MODE"] = "offline"

# Clear CUDA cache
torch.cuda.empty_cache()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-fine-tuned"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
LOGGING_DIR = "./logs"

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    bnb_4bit_compute_dtype=torch.float16
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    inference_mode=False,
)

print("Loading model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, lora_config)
print("\nTrainable parameters:")
model.print_trainable_parameters()

print("Loading and preprocessing dataset...")
dataset = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
dataset.columns = dataset.columns.str.replace('Column1.', '', regex=False)


def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return str(text).strip()


texts = [clean_text(text) for text in dataset["Comment"].tolist()]
rating = dataset["Rating"].tolist()

valid_data = [(t, r) for t, r in zip(texts, rating) if t]
texts, rating = zip(*valid_data) if valid_data else ([], [])

if not texts:
    raise ValueError("No valid text data found after cleaning")

train_texts, val_texts, train_rating, val_rating = train_test_split(
    texts, rating, test_size=0.2, random_state=42
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))


def tokenize_texts(texts):
    cleaned_texts = [str(text).strip() for text in texts]
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
train_encodings = tokenize_texts(train_texts)
val_encodings = tokenize_texts(val_texts)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
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
    eval_strategy="steps",
    eval_steps=100,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    load_best_model_at_end=True,
    report_to=[],
    remove_unused_columns=False,
    fp16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    warmup_steps=100,
    weight_decay=0.01,
)

# Optional: Set torch's multiprocessing start method
import torch.multiprocessing

torch.multiprocessing.set_start_method('spawn', force=True)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Training model...")
trainer.train()

print("Evaluating model...")
results = trainer.evaluate()
print("Evaluation results:", results)

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete!")