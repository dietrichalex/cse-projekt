import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_MODEL_NAME  = "meta-llama/Llama-3.2-1B"
FINETUNED_DIR    = "./llama-fine-tuned"

OUTPUT_CSV       = r"N:/VAL/results_finetuned.csv"

RANDOM_STATE     = 42
TEST_SIZE        = 0.2
MIN_RATING       = 1.0
MAX_RATING       = 5.0
MAX_SEQ_LENGTH   = 256
BATCH_SIZE       = 16
MAX_SAMPLES      = None

# ─── Data ─────────────────────────────────────────────────────────────────────

def load_test_split() -> tuple[list[str], list[float]]:
    data = pd.read_csv("data/Scouting_Reports_FCA.csv", encoding="utf8", delimiter=";")
    data.columns = data.columns.str.replace("Column1.", "", regex=False)

    if data["Rating"].dtype == object:
        data["Rating"] = data["Rating"].astype(str).str.replace(",", ".")
    data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")

    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return str(text).strip()

    texts   = [clean_text(t) for t in data["Comment"].tolist()]
    ratings = data["Rating"].tolist()

    valid = [(t, r) for t, r in zip(texts, ratings) if t and pd.notna(r)]
    texts, ratings = zip(*valid) if valid else ([], [])

    scaled = ((np.array(ratings) - MIN_RATING) / (MAX_RATING - MIN_RATING)).tolist()
    _, val_texts, _, val_ratings = train_test_split(
        texts, scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return list(val_texts), list(val_ratings)

# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    print("  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token        = tokenizer.eos_token
    tokenizer.pad_token_id     = tokenizer.eos_token_id
    tokenizer.padding_side     = "right"

    print("  Loading base model ...")
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    print("  Loading LoRA adapters ...")
    model = PeftModel.from_pretrained(base, FINETUNED_DIR, device_map=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  Running on : {device} (CUDA available: {torch.cuda.is_available()})")
    return tokenizer, model

# ─── Inference ────────────────────────────────────────────────────────────────

def predict_batch(tokenizer, model, texts: list[str]) -> list[float]:
    """Run a batch of texts through the regression head, return [0,1] scaled floats."""
    device = next(model.parameters()).device
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    input_ids      = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # logits shape: (batch, 1) — squeeze and clamp to [0, 1]
    preds = logits.squeeze(-1).float().cpu().numpy()
    return np.clip(preds, 0.0, 1.0).tolist()

# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels):
    p, l = np.array(preds), np.array(labels)
    mse = float(np.mean((p - l) ** 2))
    return {"MAE": float(np.mean(np.abs(p - l))), "RMSE": float(np.sqrt(mse)), "MSE": mse}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Loading test split ...")
    val_texts, val_labels = load_test_split()
    print(f"  Test samples : {len(val_texts)}")

    if MAX_SAMPLES is not None:
        val_texts, val_labels = val_texts[:MAX_SAMPLES], val_labels[:MAX_SAMPLES]
        print(f"  Capped at    : {MAX_SAMPLES}")

    print("\nLoading fine-tuned model ...")
    tokenizer, model = load_model()
    print("  Model ready.\n")

    # Resume support
    try:
        existing   = pd.read_csv(OUTPUT_CSV)
        done_count = len(existing)
        rows       = existing.to_dict("records")
        print(f"  Resuming from sample {done_count + 1}")
    except FileNotFoundError:
        done_count, rows = 0, []

    n              = len(val_texts)
    remaining_texts  = val_texts[done_count:]
    remaining_labels = val_labels[done_count:]

    # Process in batches
    for batch_start in range(0, len(remaining_texts), BATCH_SIZE):
        abs_start   = done_count + batch_start
        batch_texts  = remaining_texts[batch_start : batch_start + BATCH_SIZE]
        batch_labels = remaining_labels[batch_start : batch_start + BATCH_SIZE]

        print(f"  [{abs_start + 1}–{abs_start + len(batch_texts)}/{n}] running batch ...",
              end=" ", flush=True)

        preds = predict_batch(tokenizer, model, list(batch_texts))
        print("OK")

        for comment, label, pred in zip(batch_texts, batch_labels, preds):
            rows.append({
                "model":            FINETUNED_DIR,
                "comment":          comment,
                "label_scaled":     round(label, 4),
                "raw_response":     round(float(pred), 6),   # raw regression output
                "pred_int":         None,                     # n/a for regression model
                "pred_scaled":      round(float(pred), 4),
                "abs_error_scaled": round(abs(float(pred) - label), 4),
            })

        # Save after every batch
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    valid = pd.DataFrame(rows).dropna(subset=["pred_scaled"])
    m = compute_metrics(valid["pred_scaled"].tolist(), valid["label_scaled"].tolist())
    print(f"\n  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MSE={m['MSE']:.4f}")
    print(f"\nDone. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()