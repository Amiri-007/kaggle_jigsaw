#!/usr/bin/env python
"""
Exploratory analysis for Jigsaw toxicity dataset
 * Identity mention distributions
 * Toxicity distribution
 * Toxic vs non-toxic breakdown per identity
 * Correlation heat-map
Outputs: figs/eda/*  +  output/eda_summary.csv
"""
import argparse, pathlib, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from fairness.metrics_v2 import list_identity_columns   # already exists

sns.set_style("whitegrid")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--out-dir", default="figs/eda")
    args = ap.parse_args()

    out_d = pathlib.Path(args.out_dir); out_d.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    ids = list_identity_columns(df)
    tox_cols = ["target","severe_toxicity","obscene","threat","insult",
                "identity_attack","sexual_explicit"]
    # ------------- 1) Identity frequency --------------------------------------
    freq = (df[ids] >= .5).sum().sort_values(ascending=False)
    plt.figure(figsize=(8,6)); sns.barplot(y=freq.index, x=freq.values, color="#3182bd")
    plt.title("Identity mention counts (threshold ≥0.5)"); plt.xlabel("Count"); plt.ylabel("")
    for i,v in enumerate(freq.values): plt.text(v+200, i, f"{v:,}", va="center")
    plt.tight_layout(); plt.savefig(out_d/"identity_counts.png", dpi=300); plt.close()

    # ------------- 2) Toxicity distribution ------------------------------------
    plt.figure(figsize=(6,4)); sns.histplot(df["target"], bins=50, kde=True, color="#de2d26")
    plt.title("Distribution of target toxicity"); plt.tight_layout()
    plt.savefig(out_d/"toxicity_hist.png", dpi=300); plt.close()

    # ------------- 3) Stacked toxic vs non-toxic per identity ------------------
    bars = []
    for sg in ids:
        mask = df[sg] >= .5
        toks = (df.loc[mask,"target"]>=.5).sum()
        non  = mask.sum() - toks
        bars.append({"subgroup": sg, "toxic": toks, "nontoxic": non})
    bars_df = pd.DataFrame(bars).set_index("subgroup")
    bars_df[["toxic","nontoxic"]].div(bars_df.sum(axis=1), axis=0)\
           .plot(kind="barh", stacked=True, figsize=(8,6), color=["#cb181d","#9ecae1"])
    plt.legend(loc="lower right"); plt.title("Toxic vs non-toxic proportion per identity")
    plt.tight_layout(); plt.savefig(out_d/"toxic_stack.png", dpi=300); plt.close()

    # ------------- 4) Correlation heat-map -------------------------------------
    corr = df[tox_cols+ids].corr(method="pearson")
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="vlag", center=0, square=True, annot=False)
    plt.title("Pearson correlation – toxicity & identity columns")
    plt.tight_layout(); plt.savefig(out_d/"corr_heatmap.png", dpi=300); plt.close()

    # ------------- 5) Group-by summary CSV -------------------------------------
    summary = []
    bg_pos_rate = (df["target"]>=.5).mean()
    for sg in ids:
        mask = df[sg] >= .5
        pos_rate = (df.loc[mask,"target"]>=.5).mean()
        summary.append({"subgroup": sg,
                        "n_samples": int(mask.sum()),
                        "positives": int((df.loc[mask,'target']>=.5).sum()),
                        "pos_rate": pos_rate,
                        "bg_pos_rate": bg_pos_rate,
                        "odds_ratio": (pos_rate/(1-pos_rate+1e-9)) /
                                      (bg_pos_rate/(1-bg_pos_rate+1e-9))})
    pd.DataFrame(summary).to_csv("output/eda_summary.csv", index=False)
    print("✅  EDA plots written to", out_d, "and summary to output/eda_summary.csv")

if __name__ == "__main__":
    main() 