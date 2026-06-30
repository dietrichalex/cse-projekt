import pandas as pd
import numpy as np

BASE_CSV       = r"N:\VAL\validation_results_0_llama-3.2-1b-instruct.csv"
FINETUNED_CSV  = r"N:\VAL\results_finetuned.csv"
COMBINED_CSV   = r"N:\VAL\validation_results_combined.csv"

def compute_metrics(preds, labels):
    p, l = np.array(preds), np.array(labels)
    mse = float(np.mean((p - l) ** 2))
    return {"MAE": float(np.mean(np.abs(p - l))), "RMSE": float(np.sqrt(mse)), "MSE": mse}

def main():
    base      = pd.read_csv(BASE_CSV)
    finetuned = pd.read_csv(FINETUNED_CSV)

    combined = pd.concat([base, finetuned], ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False, encoding="utf-8")
    print(f"Combined CSV saved to {COMBINED_CSV}\n")

    print("── Results ─────────────────────────────────────────────────")
    for label, df in [("Base model (LM Studio)", base), ("Fine-tuned model (transformers)", finetuned)]:
        valid = df.dropna(subset=["pred_scaled"])
        if len(valid) == 0:
            print(f"  {label}: no valid predictions")
            continue
        m = compute_metrics(valid["pred_scaled"].tolist(), valid["label_scaled"].tolist())
        print(f"\n  {label}")
        print(f"    Samples : {len(valid)}/{len(df)}")
        print(f"    MAE     : {m['MAE']:.4f}")
        print(f"    RMSE    : {m['RMSE']:.4f}")
        print(f"    MSE     : {m['MSE']:.4f}")

    print("\n── Verdict ─────────────────────────────────────────────────")
    b = compute_metrics(
        base.dropna(subset=["pred_scaled"])["pred_scaled"].tolist(),
        base.dropna(subset=["pred_scaled"])["label_scaled"].tolist(),
    )
    f = compute_metrics(
        finetuned.dropna(subset=["pred_scaled"])["pred_scaled"].tolist(),
        finetuned.dropna(subset=["pred_scaled"])["label_scaled"].tolist(),
    )
    mae_diff = b["MAE"] - f["MAE"]
    if mae_diff > 0:
        print(f"  Fine-tuning IMPROVED MAE by {mae_diff:.4f} ({mae_diff/b['MAE']*100:.1f}%)")
    else:
        print(f"  Fine-tuning WORSENED MAE by {abs(mae_diff):.4f} ({abs(mae_diff)/b['MAE']*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()