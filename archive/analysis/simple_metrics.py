#!/usr/bin/env python
"""
Very simple metrics script - uses scikit-learn directly
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path

# Set constants
PREDICTIONS_FILE = "output/preds/simplest_preds.csv"
GROUND_TRUTH_FILE = "data/valid.csv"
MODEL_NAME = "distilbert_simplest"
RESULTS_DIR = Path("results")
FIGS_DIR = Path("figs")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# Set identity columns
IDENTITY_COLS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness',
    'asian', 'hindu', 'buddhist', 'atheist', 'bisexual', 'transgender'
]

def subgroup_auc(y_true, y_pred, subgroup_mask):
    """Calculate AUC for a specific demographic subgroup"""
    if subgroup_mask.sum() < 10:
        # Not enough examples
        return np.nan
    
    try:
        # Calculate AUC only for the subgroup
        return roc_auc_score(y_true[subgroup_mask], y_pred[subgroup_mask])
    except ValueError:
        # This happens if there's only one class in the subgroup
        return np.nan

def bpsn_auc(y_true, y_pred, subgroup_mask):
    """Calculate Background Positive, Subgroup Negative (BPSN) AUC"""
    bpsn_mask = ((subgroup_mask) & (y_true == 0)) | ((~subgroup_mask) & (y_true == 1))
    
    if np.sum(bpsn_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[bpsn_mask], y_pred[bpsn_mask])
    except ValueError:
        return np.nan

def bnsp_auc(y_true, y_pred, subgroup_mask):
    """Calculate Background Negative, Subgroup Positive (BNSP) AUC"""
    bnsp_mask = ((~subgroup_mask) & (y_true == 0)) | ((subgroup_mask) & (y_true == 1))
    
    if np.sum(bnsp_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[bnsp_mask], y_pred[bnsp_mask])
    except ValueError:
        return np.nan

def main():
    # Load predictions
    print(f"Loading predictions from {PREDICTIONS_FILE}")
    preds_df = pd.read_csv(PREDICTIONS_FILE)
    
    # Load ground truth
    print(f"Loading ground truth from {GROUND_TRUTH_FILE}")
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    
    # Print column info
    print(f"Predictions shape: {preds_df.shape}, columns: {preds_df.columns.tolist()}")
    print(f"Ground truth shape: {gt_df.shape}, columns (first 10): {gt_df.columns.tolist()[:10]}...")
    
    # Merge predictions with ground truth
    print("Merging datasets...")
    df = preds_df.merge(gt_df, on='id', how='inner')
    print(f"Merged data shape: {df.shape}")
    
    # Check target values
    print(f"Target min: {df['target'].min()}, max: {df['target'].max()}, unique values: {len(df['target'].unique())}")
    
    # Check for missing identity columns
    missing_identity_cols = [col for col in IDENTITY_COLS if col not in df.columns]
    if missing_identity_cols:
        print(f"Warning: Missing identity columns: {missing_identity_cols}")
        # Add missing columns with zero values
        for col in missing_identity_cols:
            df[col] = 0
    
    # Get target and prediction values
    # Binarize the target at 0.5 threshold if it's continuous
    y_true_raw = df['target'].values
    if len(np.unique(y_true_raw)) > 2:
        print("Target is continuous, binarizing at threshold 0.5")
        y_true = (y_true_raw >= 0.5).astype(int)
    else:
        y_true = y_true_raw.astype(int)
    
    y_pred = df['prediction'].values
    
    # Calculate overall AUC
    overall_auc = roc_auc_score(y_true, y_pred)
    print(f"Overall AUC: {overall_auc:.4f}")
    
    # Calculate subgroup metrics
    results = []
    
    # For each identity group
    for subgroup in IDENTITY_COLS:
        # Create boolean mask for this subgroup
        subgroup_mask = df[subgroup].astype(bool).values
        
        # Calculate metrics
        sub_auc = subgroup_auc(y_true, y_pred, subgroup_mask)
        bpsn = bpsn_auc(y_true, y_pred, subgroup_mask)
        bnsp = bnsp_auc(y_true, y_pred, subgroup_mask)
        
        # Calculate subgroup statistics
        subgroup_size = subgroup_mask.sum()
        subgroup_pos_rate = y_true[subgroup_mask].mean() if subgroup_size > 0 else np.nan
        background_pos_rate = y_true[~subgroup_mask].mean()
        
        # Add to results
        results.append({
            'subgroup': subgroup,
            'subgroup_size': int(subgroup_size),
            'subgroup_pos_rate': float(subgroup_pos_rate),
            'background_pos_rate': float(background_pos_rate),
            'subgroup_auc': float(sub_auc),
            'bpsn_auc': float(bpsn),
            'bnsp_auc': float(bnsp)
        })
        
        # Print results for this subgroup
        print(f"Subgroup: {subgroup}, Size: {subgroup_size}, AUC: {sub_auc:.4f}")
    
    # Add overall to results
    results.append({
        'subgroup': 'overall',
        'subgroup_size': len(y_true),
        'subgroup_pos_rate': float(y_true.mean()),
        'background_pos_rate': float(y_true.mean()),
        'subgroup_auc': float(overall_auc),
        'bpsn_auc': np.nan,
        'bnsp_auc': np.nan
    })
    
    # Create DataFrame with results
    metrics_df = pd.DataFrame(results)
    
    # Save metrics to CSV
    output_path = RESULTS_DIR / f"simple_metrics_{MODEL_NAME}.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")
    
    # Generate simple ROC curve
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve for overall
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'Overall (AUC = {overall_auc:.4f})')
    
    # Get worst and best subgroups
    subgroup_metrics = metrics_df[metrics_df['subgroup'] != 'overall']
    
    # Handle case where all subgroup AUCs might be NaN
    if subgroup_metrics['subgroup_auc'].notna().any():
        worst_subgroup = subgroup_metrics.loc[subgroup_metrics['subgroup_auc'].idxmin()]
        best_subgroup = subgroup_metrics.loc[subgroup_metrics['subgroup_auc'].idxmax()]
        
        # Print worst and best subgroups
        print(f"\nWorst performing subgroup: {worst_subgroup['subgroup']} (AUC: {worst_subgroup['subgroup_auc']:.4f})")
        print(f"Best performing subgroup: {best_subgroup['subgroup']} (AUC: {best_subgroup['subgroup_auc']:.4f})")
        
        # Plot ROC curves for worst and best subgroups
        for subgroup_type, subgroup_info in [("worst", worst_subgroup), ("best", best_subgroup)]:
            subgroup = subgroup_info['subgroup']
            subgroup_mask = df[subgroup].astype(bool).values
            
            if subgroup_mask.sum() >= 10:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true[subgroup_mask], y_pred[subgroup_mask])
                plt.plot(fpr, tpr, linestyle='--',
                         label=f'{subgroup} (AUC = {subgroup_info["subgroup_auc"]:.4f})')
    else:
        print("Warning: No valid subgroup AUC values found")
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {MODEL_NAME}')
    plt.legend()
    
    # Save figure
    fig_path = FIGS_DIR / f"roc_{MODEL_NAME}.png"
    plt.savefig(fig_path)
    print(f"ROC curve saved to {fig_path}")
    
    # Generate simple summary
    summary_path = RESULTS_DIR / "simple_summary.tsv"
    summary_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0
    
    with open(summary_path, 'a') as f:
        if not summary_exists:
            # Write header if file is empty
            f.write("model_name\toverall_auc\tworst_subgroup\tworst_auc\tbest_subgroup\tbest_auc\n")
        
        # Only write if we have valid subgroup metrics
        if subgroup_metrics['subgroup_auc'].notna().any():
            # Write entry
            f.write(f"{MODEL_NAME}\t{overall_auc:.6f}\t{worst_subgroup['subgroup']}\t"
                    f"{worst_subgroup['subgroup_auc']:.6f}\t{best_subgroup['subgroup']}\t"
                    f"{best_subgroup['subgroup_auc']:.6f}\n")
        else:
            # Write entry with just overall AUC
            f.write(f"{MODEL_NAME}\t{overall_auc:.6f}\tNA\tNA\tNA\tNA\n")
    
    print(f"Summary added to {summary_path}")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 