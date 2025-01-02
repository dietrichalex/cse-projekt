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

os.environ["WANDB_MODE"] = "offline"

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-fine-tuned"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 5e-5
LOGGING_DIR = "./logs"

print("Loading model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression"
)

print("Loading and preprocessing dataset...")
dataset = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
dataset.columns = dataset.columns.str.replace('Column1.', '', regex=False)


# Clean and preprocess texts
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return str(text).strip()


texts = [clean_text(text) for text in dataset["Comment"].tolist()]
rating = dataset["Rating"].tolist()

# Remove empty texts and corresponding ratings
valid_data = [(t, r) for t, r in zip(texts, rating) if t]  # Filter out empty strings
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
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=LOGGING_DIR,
    load_best_model_at_end=True,
    report_to=[],
    remove_unused_columns=False
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
trainer.train()

print("Evaluating model...")
results = trainer.evaluate()
print("Evaluation results:", results)

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete!")