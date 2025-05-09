#!/usr/bin/env python
"""
Fairness Audit v2
=================
Outputs:
  • figs/fairness_v2/selection_rate.png
  • figs/fairness_v2/dp_diff_ratio.png
  • figs/fairness_v2/fpr_disparity.png
  • figs/fairness_v2/fnr_disparity.png
  • output/fairness_v2_summary.csv
"""
from pathlib import Path
import argparse, json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import os, sys
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
)

# Add project root to path to access fairness modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness.metrics_v2 import list_identity_columns  # already exists in repo

sns.set_style("whitegrid")


def compute_selection_rate(df, ids, thr):
    df["y_pred"] = (df["prediction"] >= thr).astype(int)
    sel_rates = df.groupby(ids)["y_pred"].mean()  # multi-index
    sel_rates = sel_rates.droplevel(list(range(len(ids) - 1)))  # keep subgroup level
    sel_rates.name = "sel_rate"
    return sel_rates


def barplot(series, title, ylabel, save_p, ref_line=None):
    plt.figure(figsize=(11, 6))
    sns.barplot(x=series.index, y=series.values, palette="viridis")
    plt.xticks(rotation=60, ha="right")
    if ref_line is not None:
        plt.axhline(ref_line, ls="--", c="red")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_p, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds_distilbert_dev.csv")
    ap.add_argument("--val", default="data/train.csv")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--majority", default="white")
    ap.add_argument("--out-dir", default="figs/fairness_v2")
    args = ap.parse_args()

    out_d = Path(args.out_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    print(f"🔹 Loading data...")
    preds = pd.read_csv(args.preds)
    
    # First load a sample to get identity columns
    sample_df = pd.read_csv(args.val, nrows=5)
    identity_cols = list_identity_columns(sample_df)
    
    # Now load only the columns we need
    val = pd.read_csv(
        args.val, usecols=["id", "target"] + identity_cols
    )
    
    print(f"  Predictions: {preds.shape[0]} rows")
    print(f"  Validation: {val.shape[0]} rows")
    
    df = val.merge(preds, on="id", how="inner", validate="one_to_one")
    print(f"  Merged dataset: {df.shape[0]} rows")
    
    df["y_true"] = (df["target"] >= 0.5).astype(int)

    ids = identity_cols
    print(f"  Identity columns: {len(ids)}")

    # ---------------------------------------------------- confusion matrix
    print(f"🔹 Computing metrics...")
    cm = confusion_matrix(df["y_true"], (df["prediction"] >= args.thr).astype(int))
    TN, FP, FN, TP = cm.ravel()

    # ---------------------------------------------------- MetricFrame
    print(f"🔹 Building MetricFrame for fairness analysis...")
    sensitive_features = pd.DataFrame({sg: df[sg] >= 0.5 for sg in ids})
    
    mf = MetricFrame(
        metrics={
            "sel_rate": selection_rate,
            "fpr": false_positive_rate,
            "fnr": false_negative_rate,
        },
        y_true=df["y_true"],
        y_pred=(df["prediction"] >= args.thr).astype(int),
        sensitive_features=sensitive_features
    )

    print(f"🔹 Extracting metrics by group...")
    sel_rate = mf.by_group["sel_rate"]
    fpr = mf.by_group["fpr"]
    fnr = mf.by_group["fnr"]

    print(f"🔹 Computing demographic parity...")
    dp_diff = demographic_parity_difference(
        df["y_true"], (df["prediction"] >= args.thr).astype(int), sensitive_features=sensitive_features
    )
    dp_ratio = demographic_parity_ratio(
        df["y_true"], (df["prediction"] >= args.thr).astype(int), sensitive_features=sensitive_features
    )

    # ---------------------------------------------------- disparities vs majority
    print(f"🔹 Computing disparities vs majority group ({args.majority})...")
    maj = args.majority
    
    # Ensure majority group exists
    if maj not in sel_rate.index:
        print(f"Warning: Majority group '{maj}' not in data. Using first available group.")
        maj = sel_rate.index[0]
    
    disp = pd.DataFrame(
        {
            "sel_rate": sel_rate,
            "fpr": fpr,
            "fnr": fnr,
            "dp_diff": dp_diff.reindex(sel_rate.index),
            "dp_ratio": dp_ratio.reindex(sel_rate.index),
        }
    )
    disp["fpr_disparity"] = disp["fpr"] / (disp.loc[maj, "fpr"] + 1e-9)
    disp["fnr_disparity"] = disp["fnr"] / (disp.loc[maj, "fnr"] + 1e-9)
    disp["within_0.8_1.2_fpr"] = disp["fpr_disparity"].between(0.8, 1.2)
    disp["within_0.8_1.2_fnr"] = disp["fnr_disparity"].between(0.8, 1.2)
    
    print(f"🔹 Saving results to output/fairness_v2_summary.csv")
    disp.to_csv("output/fairness_v2_summary.csv")

    # ---------------------------------------------------- plots
    print(f"🔹 Generating visualizations...")
    barplot(sel_rate.sort_values(), "Selection Rate (% toxic) per Group",
            "Positive Rate", out_d / "selection_rate.png")

    barplot(dp_diff.sort_values(),
            "Demographic Parity DIFFERENCE (selection rate – overall)",
            "DP Difference", out_d / "dp_difference.png", ref_line=0)

    barplot(dp_ratio.sort_values(),
            "Demographic Parity RATIO (selection / overall)",
            "DP Ratio", out_d / "dp_ratio.png", ref_line=1)

    barplot(disp["fpr_disparity"].sort_values(),
            "FPR Disparity (sub / maj)", "Ratio", out_d / "fpr_disparity.png",
            ref_line=1)
    barplot(disp["fnr_disparity"].sort_values(),
            "FNR Disparity (sub / maj)", "Ratio", out_d / "fnr_disparity.png",
            ref_line=1)

    # --------------- console summary
    print("\n==== Overall Confusion Matrix (thr={}) ====".format(args.thr))
    print(f"TN={TN}  FP={FP}  FN={FN}  TP={TP}")
    print(f"FPR={FP/(FP+TN+1e-9):.4f}  FNR={FN/(FN+TP+1e-9):.4f}")
    
    print("\n==== Demographic Parity ====")
    print(f"DP difference: {dp_diff.mean():.4f}   DP ratio: {dp_ratio.mean():.4f}")
    
    print("\n==== Disparities outside 0.8-1.2 ====")
    viol = disp[(~disp["within_0.8_1.2_fpr"]) | (~disp["within_0.8_1.2_fnr"])]
    if len(viol) > 0:
        print(viol[["fpr_disparity", "fnr_disparity"]])
    else:
        print("No disparities outside 0.8-1.2 range!")
    
    print(f"\n✅ Analysis complete. Visualizations saved to {args.out_dir}/")

if __name__ == "__main__":
    main() 