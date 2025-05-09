from pathlib import Path
import pandas as pd
import numpy as np
from fairness.metrics_v2 import (list_identity_columns, generalized_power_mean)

def main():
    # Set file paths
    metrics_csv = Path("output/preds/blend_dev.csv")
    valid_df = pd.read_csv("data/valid.csv")
    
    # Check if files exist
    if not metrics_csv.exists():
        print(f"Error: {metrics_csv} not found. Using the existing simplest_preds.csv file instead.")
        metrics_csv = Path("output/preds/simplest_preds.csv")
    
    # Load prediction data
    print(f"Loading predictions from {metrics_csv}")
    preds_df = pd.read_csv(metrics_csv)
    
    # Merge with validation data
    print("Merging with validation data...")
    df = valid_df.merge(preds_df, on="id", how="inner")
    print(f"Merged data shape: {df.shape}")
    
    # Get identity columns
    ids = list_identity_columns(df)
    print(f"Found {len(ids)} identity columns")
    
    # Compute metrics for each identity group
    print("Computing metrics for each identity group...")
    records = []
    for sg in ids:
        # Create subgroup mask
        sg_mask = df[sg] >= 0.5
        sg_size = sg_mask.sum()
        
        # Get binary labels and predictions
        y_true = df["target"].values
        y_pred = df["prediction"].values
        
        # Calculate AUC metrics
        from sklearn.metrics import roc_auc_score
        
        # Subgroup AUC - performance on just the subgroup
        subgroup_indices = sg_mask
        if subgroup_indices.sum() > 0:
            subgroup_auc_val = roc_auc_score(y_true[subgroup_indices], y_pred[subgroup_indices])
        else:
            subgroup_auc_val = np.nan
        
        # BPSN - background positive, subgroup negative
        bpsn_indices = ((~sg_mask) & (y_true == 1)) | ((sg_mask) & (y_true == 0))
        if bpsn_indices.sum() > 0:
            bpsn_auc_val = roc_auc_score(y_true[bpsn_indices], y_pred[bpsn_indices])
        else:
            bpsn_auc_val = np.nan
        
        # BNSP - background negative, subgroup positive
        bnsp_indices = ((~sg_mask) & (y_true == 0)) | ((sg_mask) & (y_true == 1))
        if bnsp_indices.sum() > 0:
            bnsp_auc_val = roc_auc_score(y_true[bnsp_indices], y_pred[bnsp_indices])
        else:
            bnsp_auc_val = np.nan
        
        # FPR/FNR at τ=0.5
        t = 0.5
        sg_pos = sg_mask
        y_binary = y_true >= 0.5
        yhat_binary = y_pred >= t
        
        # Calculate FPR (false positive rate)
        negatives_in_subgroup = (~y_binary) & sg_pos
        if negatives_in_subgroup.sum() > 0:
            false_positives_in_subgroup = (yhat_binary & ~y_binary) & sg_pos
            fpr = false_positives_in_subgroup.sum() / negatives_in_subgroup.sum()
        else:
            fpr = np.nan
        
        # Calculate FNR (false negative rate)
        positives_in_subgroup = (y_binary) & sg_pos
        if positives_in_subgroup.sum() > 0:
            false_negatives_in_subgroup = (~yhat_binary & y_binary) & sg_pos
            fnr = false_negatives_in_subgroup.sum() / positives_in_subgroup.sum()
        else:
            fnr = np.nan
        
        # Create record for this subgroup
        rec = {
            "subgroup_name": sg,
            "subgroup_size": sg_size,
            "subgroup_auc": subgroup_auc_val,
            "bpsn_auc": bpsn_auc_val,
            "bnsp_auc": bnsp_auc_val,
            "fpr": fpr,
            "fnr": fnr
        }
        records.append(rec)
        print(f"Subgroup: {sg}, Size: {sg_size}, AUC: {subgroup_auc_val:.4f}")
    
    # Create metrics dataframe
    m = pd.DataFrame(records)
    
    # Overall AUC
    overall_auc = roc_auc_score(df["target"], df["prediction"])
    print(f"Overall AUC: {overall_auc:.4f}")
    
    # Power mean of bias-AUCs (p = -5)
    bias_metrics = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    power_means = {}
    for metric in bias_metrics:
        power_means[metric] = generalized_power_mean(m[metric].values, p=-5)
        print(f"Power mean for {metric}: {power_means[metric]:.4f}")
    
    # Calculate final score using the power means
    final_score = 0.25 * overall_auc + 0.75 * np.mean([power_means[metric] for metric in bias_metrics])
    print(f"Final score: {final_score:.4f}")
    
    # Create summary dataframe
    summary = pd.DataFrame([{
        "model": "blend_dev",
        "overall_auc": overall_auc,
        "final_score": final_score
    }])
    
    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    m.to_parquet(out_dir/"audit_metrics.parquet", index=False)
    summary.to_csv(out_dir/"summary.tsv", sep="\t", index=False)
    print("✅ Fairness metrics saved to results/audit_metrics.parquet & summary.tsv")

if __name__ == "__main__":
    main() 