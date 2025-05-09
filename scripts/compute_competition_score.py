#!/usr/bin/env python
"""
Competition Score Calculator
===========================
Calculates the official Kaggle Jigsaw competition score for toxicity classification.
This implements the exact scoring function used in the competition:
- Overall AUC (25% weight)
- Subgroup AUC power mean (25% weight)
- BPSN AUC power mean (25% weight)
- BNSP AUC power mean (25% weight)

Usage:
    python scripts/compute_competition_score.py --csv results/preds_distilbert_dev.csv

Outputs:
    - JSON with all metrics
    - Optional output file with saved metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
import argparse
import os
import sys

# Add project root to path to access fairness modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness.metrics_v2 import list_identity_columns, bpsn_auc, bnsp_auc

# Standard identity columns used in the Jigsaw competition
COMPETITION_COLUMNS = [
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
    "black", "white", "psychiatric_or_mental_illness", "asian", "hindu", "buddhist",
    "atheist", "bisexual", "transgender", "latino"
]

def power_mean(series: pd.Series, p: float) -> float:
    """
    Generalised mean (power mean) with exponent p (p = -5 in competition).
    
    Args:
        series: Series of values to calculate power mean
        p: Exponent parameter (negative values emphasize lower values)
        
    Returns:
        Power mean value
    """
    values = np.clip(series.values, 1e-15, 1)   # avoid 0 ** negative power
    return (np.mean(values ** p)) ** (1 / p)

def bias_auc(df: pd.DataFrame, subgroup: str, pred_col: str, label_col: str):
    """
    Return Subgroup / BPSN / BNSP AUCs for a single identity column.
    
    Args:
        df: DataFrame with identity columns, predictions and labels
        subgroup: Name of the identity column
        pred_col: Column containing model predictions
        label_col: Column containing ground truth labels
        
    Returns:
        Tuple of (subgroup_auc, bpsn_auc, bnsp_auc)
    """
    subgroup_mask = df[subgroup] >= .5
    
    # Convert label to binary if needed
    y_true = (df[label_col] >= 0.5).astype(int).values
    y_pred = df[pred_col].values
    
    # Calculate subgroup AUC
    try:
        auc_subgroup = roc_auc_score(y_true[subgroup_mask], y_pred[subgroup_mask])
    except:
        auc_subgroup = np.nan
    
    # Calculate BPSN and BNSP AUCs using our fairness module's implementation
    auc_bpsn = bpsn_auc(y_true, y_pred, subgroup_mask.values)
    auc_bnsp = bnsp_auc(y_true, y_pred, subgroup_mask.values)
    
    return auc_subgroup, auc_bpsn, auc_bnsp

def compute_final_metric(df: pd.DataFrame,
                         pred_col: str = "prediction",
                         label_col: str = "target",
                         identity_cols: list = None) -> dict:
    """
    Calculate the full Kaggle competition score.
    
    Args:
        df: DataFrame with identity columns, predictions and labels
        pred_col: Column containing model predictions
        label_col: Column containing ground truth labels
        identity_cols: List of identity columns to use (defaults to competition columns)
        
    Returns:
        Dictionary with all metrics and final score
    """
    # Use competition columns by default, or provided identity columns
    if identity_cols is None:
        identity_cols = COMPETITION_COLUMNS
    
    # Ensure we have the necessary columns
    missing_cols = [col for col in identity_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing identity columns: {missing_cols}")
        print("Using only available identity columns.")
        identity_cols = [col for col in identity_cols if col in df.columns]
    
    # 1) Overall AUC
    # Make sure labels are binary for AUC computation
    y_true_binary = (df[label_col] >= 0.5).astype(int)
    overall_auc = roc_auc_score(y_true_binary, df[pred_col])

    # 2) Bias AUCs for each identity
    subgroup_aucs, bpsn_aucs, bnsp_aucs = [], [], []
    
    for sg in identity_cols:
        try:
            sg_auc, bp_auc, bn_auc = bias_auc(df, sg, pred_col, label_col)
            
            # Only include valid values
            if not np.isnan(sg_auc):
                subgroup_aucs.append(sg_auc)
            if not np.isnan(bp_auc):
                bpsn_aucs.append(bp_auc)
            if not np.isnan(bn_auc):
                bnsp_aucs.append(bn_auc)
                
        except Exception as e:
            print(f"Warning: Could not compute metrics for {sg}: {e}")

    # 3) Generalised mean with p = -5
    p = -5.0
    
    # Calculate power means
    subgroup_pm = power_mean(pd.Series(subgroup_aucs), p) if subgroup_aucs else np.nan
    bpsn_pm = power_mean(pd.Series(bpsn_aucs), p) if bpsn_aucs else np.nan
    bnsp_pm = power_mean(pd.Series(bnsp_aucs), p) if bnsp_aucs else np.nan
    
    # Combine bias components
    bias_components = [x for x in [subgroup_pm, bpsn_pm, bnsp_pm] if not np.isnan(x)]
    bias_score = np.mean(bias_components) if bias_components else np.nan

    # 4) Final metric (25% overall AUC, 75% bias score)
    final_score = 0.25 * overall_auc + 0.75 * bias_score

    # Return detailed metrics for analysis
    metrics = {
        "overall_auc": float(overall_auc),
        "bias_score": float(bias_score),
        "subgroup_auc_pm": float(subgroup_pm),
        "bpsn_auc_pm": float(bpsn_pm),
        "bnsp_auc_pm": float(bnsp_pm),
        "final_score": float(final_score),
        "identity_columns_used": identity_cols,
        "subgroup_metrics": {
            sg: {
                "subgroup_auc": float(sg_auc), 
                "bpsn_auc": float(bp_auc), 
                "bnsp_auc": float(bn_auc)
            }
            for sg, sg_auc, bp_auc, bn_auc in zip(
                identity_cols, 
                subgroup_aucs + [np.nan] * (len(identity_cols) - len(subgroup_aucs)),
                bpsn_aucs + [np.nan] * (len(identity_cols) - len(bpsn_aucs)),
                bnsp_aucs + [np.nan] * (len(identity_cols) - len(bnsp_aucs))
            )
        }
    }
    
    return metrics

def main():
    ap = argparse.ArgumentParser(description="Compute Kaggle Jigsaw final score")
    ap.add_argument("--csv", required=True,
                    help="CSV containing ground-truth & predictions")
    ap.add_argument("--pred-col", default="prediction",
                    help="Column name for model predictions")
    ap.add_argument("--label-col", default="target",
                    help="Column name for ground truth labels")
    ap.add_argument("--val-data", default=None,
                    help="Optional separate validation data with identity columns")
    ap.add_argument("--use-project-identities", action="store_true",
                    help="Use project's detected identity columns instead of competition columns")
    ap.add_argument("-o", "--out", default=None,
                    help="Optional JSON file to dump metrics")
    args = ap.parse_args()

    # Load prediction data
    print(f"🔹 Loading predictions from {args.csv}")
    preds_df = pd.read_csv(args.csv)
    
    # Check if we need to load separate validation data with identity columns
    if args.val_data:
        print(f"🔹 Loading validation data from {args.val_data}")
        val_df = pd.read_csv(args.val_data)
        
        # Merge predictions with validation data
        df = val_df.merge(preds_df, on="id", how="inner", validate="one_to_one")
        print(f"  Merged dataset: {df.shape[0]} rows")
    else:
        # Otherwise use prediction data directly (must contain identity columns)
        df = preds_df
        
    # Determine which identity columns to use
    if args.use_project_identities:
        print("🔹 Using project's detected identity columns")
        identity_cols = list_identity_columns(df)
    else:
        print("🔹 Using standard competition identity columns")
        identity_cols = COMPETITION_COLUMNS
    
    print(f"  Using {len(identity_cols)} identity columns: {identity_cols}")
    
    # Compute metrics
    print("🔹 Computing final competition score...")
    metrics = compute_final_metric(df, args.pred_col, args.label_col, identity_cols)
    
    # Print summary
    print("\n==== Kaggle Competition Score ====")
    print(f"Overall AUC:     {metrics['overall_auc']:.4f}")
    print(f"Subgroup AUC PM: {metrics['subgroup_auc_pm']:.4f}")
    print(f"BPSN AUC PM:     {metrics['bpsn_auc_pm']:.4f}")
    print(f"BNSP AUC PM:     {metrics['bnsp_auc_pm']:.4f}")
    print(f"Bias Score:      {metrics['bias_score']:.4f}")
    print(f"Final Score:     {metrics['final_score']:.4f}")
    
    # Save to file if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n✅ Metrics saved to {args.out}")
    
    return metrics

if __name__ == "__main__":
    main() 