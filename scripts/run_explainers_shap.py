#!/usr/bin/env python
"""
Compute SHAP and SHARP metrics for DistilBERT head-tail dev model.
"""
import argparse, pathlib, random, json, torch, shap, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from fairness.metrics_v2 import list_identity_columns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ------------------------------------------------------------
def load_model(path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt["config"] if "config" in ckpt else {}
    base = cfg.get("model_name", "distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model.eval()
    return model, tokenizer

# ------------------------------------------------------------
def get_sample(df, n=2000, seed=42):
    random.seed(seed)
    pos = df[df["target"] >= .5].sample(n//2, random_state=seed)
    neg = df[df["target"] <  .5].sample(n//2, random_state=seed)
    return pd.concat([pos, neg]).reset_index(drop=True)

# ------------------------------------------------------------
def build_explainer(model, tokenizer, max_len=192):
    def f(texts):
        enc = tokenizer(texts,
                        max_length=max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return model(**enc).logits
    # Use 100 background examples
    return shap.Explainer(f, shap.maskers.Text(tokenizer), output_names=["toxicity"])

# ------------------------------------------------------------
def sharp_scores(shap_vals, df):
    """
    shap_vals: (n, seq_len) absolute values aggregated per sample
    returns: pd.DataFrame[subgroup_name, sharp]
    """
    abs_vals = np.abs(shap_vals).mean(axis=1)      # (n,)
    global_mean = abs_vals.mean()
    recs = []
    for sg in list_identity_columns(df):
        mask = df[sg] >= 0.5
        if mask.sum() < 10:       # skip tiny groups
            continue
        sharp = abs(abs_vals[mask].mean() - global_mean)
        recs.append({"subgroup_name": sg, "sharp": sharp, "n": int(mask.sum())})
    return pd.DataFrame(recs).sort_values("sharp", ascending=False)

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/checkpoints/distilbert_headtail_fold0.pth")
    ap.add_argument("--valid", default="data/valid.csv")
    ap.add_argument("--out-dir", default="output/explainers")
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔹 loading model {args.ckpt}")
    model, tok = load_model(args.ckpt)

    df_valid = pd.read_csv(args.valid)
    df_s     = get_sample(df_valid, n=args.sample)
    texts    = df_s["comment_text"].astype(str).tolist()

    print("🔹 building explainer / computing SHAP values …")
    explainer = build_explainer(model, tok)
    shap_values = explainer(texts, silent=True)        # -> shap.Explanation

    # Save raw values
    np.savez_compressed(out_dir/"shap_distilbert_dev.npz",
                        values=shap_values.values, data=df_s.to_dict("list"))
    shap.plots.bar(shap_values, max_display=20, show=False)
    import matplotlib.pyplot as plt; plt.tight_layout()
    plt.savefig(out_dir/"shap_bar_distilbert_dev.png"); plt.close()

    # SHARP
    abs_token_mean = np.abs(shap_values.values).mean(axis=2)   # (n_samples,)
    sharp_df = sharp_scores(abs_token_mean, df_s)
    sharp_df.to_csv(out_dir/"sharp_scores_distilbert_dev.csv", index=False)
    print(sharp_df.head(10))
    print("✅ SHAP + SHARP artefacts saved in", out_dir)

if __name__ == "__main__":
    main() 