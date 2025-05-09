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
    df["y_pred"] = (df["prediction"] >= args.thr).astype(int)

    ids = identity_cols
    print(f"  Identity columns: {len(ids)}")

    # ---------------------------------------------------- confusion matrix
    print(f"🔹 Computing metrics...")
    cm = confusion_matrix(df["y_true"], df["y_pred"])
    TN, FP, FN, TP = cm.ravel()

    # ---------------------------------------------------- MetricFrame
    print(f"🔹 Building MetricFrame for fairness analysis...")
    # Create a dictionary of sensitive features for each identity column
    sensitive_features = {}
    for col in ids:
        # Check if this identity has enough samples to be meaningful
        if (df[col] >= 0.5).sum() >= 10:
            sensitive_features[col] = df[col] >= 0.5
    
    if not sensitive_features:
        print("Error: No identity columns with sufficient samples found")
        return
    
    # Create individual metric frames for each identity column
    metrics_by_group = {}
    dp_diff_by_group = {}
    dp_ratio_by_group = {}
    
    # Get overall selection rate for reference
    overall_sel_rate = df["y_pred"].mean()
    
    for col, feat in sensitive_features.items():
        try:
            # Create a MetricFrame for this specific identity column
            group_mf = MetricFrame(
                metrics={
                    "sel_rate": selection_rate,
                    "fpr": false_positive_rate,
                    "fnr": false_negative_rate,
                },
                y_true=df["y_true"],
                y_pred=df["y_pred"],
                sensitive_features={col: feat}
            )
            
            # Extract metrics
            metrics_by_group[col] = {
                "sel_rate": group_mf.by_group.loc[True, "sel_rate"],
                "fpr": group_mf.by_group.loc[True, "fpr"],
                "fnr": group_mf.by_group.loc[True, "fnr"],
            }
            
            # Calculate demographic parity metrics
            dp_diff_by_group[col] = metrics_by_group[col]["sel_rate"] - overall_sel_rate
            dp_ratio_by_group[col] = metrics_by_group[col]["sel_rate"] / overall_sel_rate
        except Exception as e:
            print(f"  Warning: Could not compute metrics for {col}: {e}")
    
    # Convert to dataframes
    sel_rate = pd.Series({k: v["sel_rate"] for k, v in metrics_by_group.items()})
    fpr = pd.Series({k: v["fpr"] for k, v in metrics_by_group.items()})
    fnr = pd.Series({k: v["fnr"] for k, v in metrics_by_group.items()})
    dp_diff = pd.Series(dp_diff_by_group)
    dp_ratio = pd.Series(dp_ratio_by_group)

    # ---------------------------------------------------- disparities vs majority
    print(f"🔹 Computing disparities vs majority group ({args.majority})...")
    maj = args.majority
    
    # Ensure majority group exists
    if maj not in sel_rate.index:
        print(f"  Warning: Majority group '{maj}' not in data. Using first available group.")
        maj = sel_rate.index[0]
    
    # Build the disparity dataframe
    disp = pd.DataFrame({
        "sel_rate": sel_rate,
        "fpr": fpr,
        "fnr": fnr,
        "dp_diff": dp_diff,
        "dp_ratio": dp_ratio,
    })
    
    # Calculate disparities relative to majority group
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

    # ---------- PER-SUBGROUP CONFUSION MATRICES + BPSN/BNSP -----------------
    from sklearn.metrics import roc_auc_score
    def subgroup_cm_plot(mask, name):
        cm_sg = confusion_matrix(df.loc[mask,"y_true"],
                                 (df.loc[mask,"prediction"]>=args.thr).astype(int))
        fig, ax = plt.subplots(figsize=(3,3))
        sns.heatmap(cm_sg, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["NT","T"], yticklabels=["NT","T"])
        ax.set_title(f"CM – {name}"); ax.set_xlabel("Pred"); ax.set_ylabel("True")
        fig.tight_layout(); fig.savefig(out_d/f"conf_matrix_{name}.png", dpi=250); plt.close()

    bpsn_vals, bnsp_vals = {}, {}
    for sg in ids:
        mask_sg  = df[sg] >= .5
        if mask_sg.sum() < 30:                 # skip tiny
            continue
        subgroup_cm_plot(mask_sg, sg)
        # BPSN / BNPS AUCs
        from fairness.metrics_v2 import bpsn_auc, bnsp_auc
        bpsn_vals[sg] = bpsn_auc(df["y_true"].values, df["prediction"].values, mask_sg)
        bnsp_vals[sg] = bnsp_auc(df["y_true"].values, df["prediction"].values, mask_sg)

    # bar-plots
    for name, d in [("bpsn", bpsn_vals), ("bnsp", bnsp_vals)]:
        s = pd.Series(d).sort_values()
        barplot(s, f"{name.upper()} AUC per subgroup", "AUC",
                out_d/f"{name}_auc_bar.png")

    # --------------- console summary
    print("\n==== Overall Confusion Matrix (thr={}) ====".format(args.thr))
    print(f"TN={TN}  FP={FP}  FN={FN}  TP={TP}")
    print(f"FPR={FP/(FP+TN+1e-9):.4f}  FNR={FN/(FN+TP+1e-9):.4f}")
    
    print("\n==== Demographic Parity ====")
    print(f"DP difference (mean): {dp_diff.mean():.4f}")
    print(f"DP ratio (mean): {dp_ratio.mean():.4f}")
    
    print("\n==== Disparities outside 0.8-1.2 ====")
    viol = disp[(~disp["within_0.8_1.2_fpr"]) | (~disp["within_0.8_1.2_fnr"])]
    if len(viol) > 0:
        print(viol[["fpr_disparity", "fnr_disparity"]])
    else:
        print("No disparities outside 0.8-1.2 range!")
    
    print(f"\n✅ Analysis complete. Visualizations saved to {args.out_dir}/")

if __name__ == "__main__":
    main() 