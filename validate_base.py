import pandas as pd
import numpy as np
import requests
import json
import re
import time
import subprocess
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

FINE_TUNED_MODEL = "custom-model"
BASE_MODEL       = "llama-3.2-1b-instruct"

OUTPUT_CSV = r"N:\VAL\validation_results.csv"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MIN_RATING   = 1.0
MAX_RATING   = 5.0

MAX_SAMPLES = None

REQUEST_DELAY = 0.2

PAUSE_BETWEEN_MODELS = False

# ── RAM management ────────────────────────────────────────────────────────────
RESTART_EVERY_N = 1000

LM_STUDIO_EXE = r"C:\Users\alexd\AppData\Local\Programs\lm-studio\LM Studio.exe"

RESTART_WAIT_SECONDS = 15

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CATEGORIES = (
    "Stell dir vor, dass du ein Fußballscout bist und dem nachfolgenden Spielerbericht eines Spielers eine ganzzahlige Note zwischen 1 und 5 in den verschiedenen Kategorien 1.Physis, 2. Technik, 3. Spielintelligenz, 4. Charakter geben musst, wobei 5 die beste Note ist. Gebe nur die Noten pro Kategorie ohne weiteren Text aus. Dies ist der Spielerbericht: "
)

SYSTEM_PROMPT = (
    "Stell dir vor, dass du ein Fußballscout bist und dem nachfolgenden Spielerbericht eines Spielers eine ganzzahlige Note zwischen 1 und 5 geben musst, wobei 5 die beste Note ist. Gebe nur die Note ohne weiteren Text aus. Dies ist der Spielerbericht: "
)

def build_user_message(comment: str) -> str:
    return f"Scouting report:\n{comment}"

# ─── LM Studio restart ────────────────────────────────────────────────────────

def restart_lm_studio(model_id: str) -> None:
    """Kill LM Studio, wait, relaunch it, and wait for the API to come back up."""
    print(f"\n  [RAM FLUSH] Restarting LM Studio to clear KV cache ...")

    subprocess.run(
        ["taskkill", "/F", "/IM", "LM Studio.exe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    subprocess.Popen(
        [LM_STUDIO_EXE],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    print(f"  [RAM FLUSH] Waiting {RESTART_WAIT_SECONDS}s for LM Studio to come back up ...")
    time.sleep(RESTART_WAIT_SECONDS)

    for attempt in range(60):
        try:
            r = requests.get("http://127.0.0.1:1234/v1/models", timeout=3)
            if r.status_code == 200:
                print(f"  [RAM FLUSH] API is back up. Resuming ...\n")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    print("  [RAM FLUSH] WARNING: API did not respond after 60s — continuing anyway.")

# ─── Data Loading ──────────────────────────────────

def load_test_split() -> tuple[list[str], list[float]]:
    """Reproduce the exact train/test split from training and return the test set."""
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

    rating_array = np.array(ratings)
    scaled = ((rating_array - MIN_RATING) / (MAX_RATING - MIN_RATING)).tolist()

    _, val_texts, _, val_ratings = train_test_split(
        texts, scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return list(val_texts), list(val_ratings)

# ─── LM Studio API ────────────────────────────────────────────────────────────

def query_model(model_id: str, comment: str) -> str | None:
    """Send one scouting report to LM Studio and return the raw text response."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(comment)},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            LM_STUDIO_URL, headers=headers, data=json.dumps(payload), timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"    [REQUEST ERROR] {e}")
        return None

def parse_rating(raw: str | None) -> int | None:
    """Extract a single integer in 1–5 from the model's response."""
    if raw is None:
        return None
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        return int(match.group(1))
    match = re.search(r"([1-9])", raw)
    if match:
        return max(1, min(5, int(match.group(1))))
    return None

def scale_to_01(rating_int: int) -> float:
    return (rating_int - MIN_RATING) / (MAX_RATING - MIN_RATING)

# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds: list[float], labels: list[float]) -> dict:
    p = np.array(preds)
    l = np.array(labels)
    mse  = float(np.mean((p - l) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(p - l)))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse}

# ─── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate_model(
    model_id: str,
    texts: list[str],
    labels_scaled: list[float],
    partial_csv: str,
) -> pd.DataFrame:
    """
    Run all test samples through one model.
    - Saves progress after every sample (crash-safe resume).
    - Restarts LM Studio every RESTART_EVERY_N samples to flush RAM.
    """
    try:
        existing  = pd.read_csv(partial_csv)
        done_count = len(existing)
        rows      = existing.to_dict("records")
        print(f"  Resuming from sample {done_count + 1} (found {done_count} existing results)")
    except FileNotFoundError:
        done_count = 0
        rows      = []

    n = len(texts)

    for i, (comment, label) in enumerate(zip(texts, labels_scaled)):
        if i < done_count:
            continue

        # ── Periodic RAM flush ────────────────────────────────────────────────
        samples_done_this_run = i - done_count
        if samples_done_this_run > 0 and samples_done_this_run % RESTART_EVERY_N == 0:
            restart_lm_studio(model_id)

        print(f"  [{i+1}/{n}] querying {model_id} ...", end=" ", flush=True)

        raw      = query_model(model_id, comment)
        pred_int = parse_rating(raw)
        pred_01  = scale_to_01(pred_int) if pred_int is not None else None

        status = f"OK ({pred_int})" if pred_int is not None else f"PARSE FAIL (raw='{raw}')"
        print(status)

        rows.append({
            "model":            model_id,
            "comment":          comment,
            "label_scaled":     round(label, 4),
            "raw_response":     raw,
            "pred_int":         pred_int,
            "pred_scaled":      round(pred_01, 4) if pred_01 is not None else None,
            "abs_error_scaled": round(abs(pred_01 - label), 4) if pred_01 is not None else None,
        })

        pd.DataFrame(rows).to_csv(partial_csv, index=False, encoding="utf-8")
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(rows)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Loading and reproducing test split ...")
    val_texts, val_labels = load_test_split()
    print(f"  Test samples : {len(val_texts)}")
    print(f"  Rating range : {int(MIN_RATING)}–{int(MAX_RATING)}  (integers, scaled to [0, 1])")
    print(f"  RAM flush    : every {RESTART_EVERY_N} samples")

    if MAX_SAMPLES is not None:
        val_texts  = val_texts[:MAX_SAMPLES]
        val_labels = val_labels[:MAX_SAMPLES]
        print(f"  Capped at    : {MAX_SAMPLES} samples")

    all_results = []
    models = [BASE_MODEL, FINE_TUNED_MODEL]

    for idx, model_id in enumerate(models):
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_id}")
        print("=" * 60)

        if PAUSE_BETWEEN_MODELS and idx > 0:
            input(
                f"\n  >>> Please load '{model_id}' in LM Studio now, "
                "then press Enter to continue ..."
            )

        partial_csv = OUTPUT_CSV.replace(".csv", f"_{idx}_{model_id.replace('/', '_')}.csv")
        df = evaluate_model(model_id, val_texts, val_labels, partial_csv)
        all_results.append(df)

        valid = df.dropna(subset=["pred_scaled"])
        if len(valid) > 0:
            metrics    = compute_metrics(valid["pred_scaled"].tolist(), valid["label_scaled"].tolist())
            parse_rate = len(valid) / len(df) * 100
            print(f"\n  Parse success : {len(valid)}/{len(df)} ({parse_rate:.1f}%)")
            print(f"  MAE  (scaled) : {metrics['MAE']:.4f}")
            print(f"  RMSE (scaled) : {metrics['RMSE']:.4f}")
            print(f"  MSE  (scaled) : {metrics['MSE']:.4f}")
        else:
            print("  No valid predictions — check model name and LM Studio connection.")

    # ── Save combined CSV ──────────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {OUTPUT_CSV}")

    # ── side-by-side comparison ─────────────────────────────────────────
    print("\n── Summary comparison ──────────────────────────────────────")
    for model_id in models:
        subset = combined[combined["model"] == model_id].dropna(subset=["pred_scaled"])
        if len(subset) == 0:
            print(f"  {model_id}: no valid predictions")
            continue
        m = compute_metrics(subset["pred_scaled"].tolist(), subset["label_scaled"].tolist())
        print(f"  {model_id}")
        print(f"    MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MSE={m['MSE']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()