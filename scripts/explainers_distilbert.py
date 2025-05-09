#!/usr/bin/env python
"""
Compute SHAP + SHARP for distilbert_headtail_dev checkpoint
"""
import argparse, pathlib, numpy as np, pandas as pd, shap, torch, seaborn as sns, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from fairness.metrics_v2 import list_identity_columns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ---------------------------------------------------------------------
def load_checkpoint(ckpt_path: pathlib.Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    base = ckpt.get("config", {}).get("model_name", "distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(base)
    return model, tokenizer

# ---------------------------------------------------------------------
def get_sample(df: pd.DataFrame, sample: int, seed: int = 42) -> pd.DataFrame:
    # Stratified 50-50 toxic / non-toxic sample for SHAP stability
    pos = df[df["target"] >= .5].sample(n=sample//2, random_state=seed)
    neg = df[df["target"] <  .5].sample(n=sample//2, random_state=seed)
    return pd.concat([pos, neg]).reset_index(drop=True)

# ---------------------------------------------------------------------
def build_explainer(model, tokenizer, max_len=192):
    def predictor(texts):
        enc = tokenizer(texts,
                        max_length=max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt").to(DEVICE)
        return model(**enc).logits
    masker = shap.maskers.Text(tokenizer)
    return shap.Explainer(predictor, masker, output_names=["toxicity"])

# ---------------------------------------------------------------------
def sharp(df: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """Return SHARP scores per identity subgroup"""
    abs_token_mean = np.abs(shap_values).mean(axis=1)   # per-sample mean |SHAP|
    glob = abs_token_mean.mean()
    rows = []
    for sg in list_identity_columns(df):
        mask = df[sg] >= .5
        if mask.sum() < 8:          # ignore tiny groups
            continue
        score = abs(abs_token_mean[mask].mean() - glob)
        rows.append({"subgroup": sg, "sharp": score, "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("sharp", ascending=False)

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/checkpoints/distilbert_headtail_fold0.pth")
    ap.add_argument("--valid-csv", default="data/valid.csv")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--out-dir", default="output/explainers")
    args = ap.parse_args()

    out_d = pathlib.Path(args.out_dir); out_d.mkdir(parents=True, exist_ok=True)

    print("🔹 loading checkpoint …")
    model, tok = load_checkpoint(pathlib.Path(args.ckpt))

    print("🔹 sampling validation rows …")
    df_v = pd.read_csv(args.valid_csv)
    df_s = get_sample(df_v, args.sample)
    texts = df_s["comment_text"].astype(str).fillna("").tolist()

    print("🔹 building explainer …")
    expl = build_explainer(model, tok)

    print("🔹 computing SHAP values … (may take ~3-4 min)")
    shap_vals = expl(texts, silent=True)           # ⇒ shap.Explanation

    # save raw
    np.savez_compressed(out_d/"shap_values.npz",
                        values=shap_vals.values,
                        base_values=shap_vals.base_values)

    # 1️⃣  token bar plot
    shap.plots.bar(shap_vals, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_d/"shap_tokens_bar.png", dpi=300)
    plt.close()

    # 2️⃣  SHARP table + plots
    sharp_df = sharp(df_s, shap_vals.values)
    sharp_df.to_csv(out_d/"sharp_scores.csv", index=False)

    # bar plot
    plt.figure(figsize=(8,4))
    sns.barplot(data=sharp_df.head(15), x="sharp", y="subgroup", palette="viridis")
    plt.title("Top-15 SHARP scores – dev model")
    plt.xlabel("|mean(|SHAP|) subgroup – global|")
    plt.tight_layout()
    plt.savefig(out_d/"sharp_bar.png", dpi=300)
    plt.close()

    # heat-map (identities × |SHAP|)
    ids = list_identity_columns(df_s)
    heat = pd.DataFrame({sg: np.abs(shap_vals.values[df_s[sg]>=.5]).mean()
                         for sg in ids}, index=["mean_abs_shap"]).T
    plt.figure(figsize=(6,8))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="magma_r")
    plt.title("Mean |SHAP| per subgroup")
    plt.tight_layout()
    plt.savefig(out_d/"sharp_heatmap.png", dpi=300)

    print("✅  SHAP + SHARP artefacts written to", out_d)

if __name__ == "__main__":
    main() 