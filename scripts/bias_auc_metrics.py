#!/usr/bin/env python
"""
Calculate bias AUCs (AUC, BPSN AUC, BNSP AUC) for each identity subgroup
and create a side-by-side visualization.

Outputs: figs/bias_aucs/bias_aucs_comparison.png
"""
import argparse, pathlib, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root to path
from fairness.metrics_v2 import list_identity_columns, subgroup_auc, bpsn_auc, bnsp_auc
from sklearn.metrics import roc_auc_score

sns.set_style("whitegrid")
sns.set_palette("viridis")
plt.rcParams.update({'font.size': 10})

def compute_bias_metrics_for_model(df, identity_columns, target_column, pred_column):
    """
    Compute bias metrics for each identity subgroup.
    
    Args:
        df: DataFrame with identity columns, target, and predictions
        identity_columns: List of column names for identity subgroups
        target_column: Name of the column with ground truth labels
        pred_column: Name of the column with model predictions
        
    Returns:
        Dictionary with metrics for each subgroup and overall metrics
    """
    # Convert target to binary if needed
    y_true = (df[target_column].values >= 0.5).astype(float)
    y_pred = df[pred_column].values
    
    # Calculate overall AUC
    overall_auc = roc_auc_score(y_true, y_pred)
    
    # Calculate metrics for each subgroup
    results = {}
    worst_auc = 1.0
    worst_subgroup = None
    
    results['overall'] = {'auc': overall_auc}
    
    for subgroup in identity_columns:
        # Create mask for subgroup membership
        subgroup_mask = (df[subgroup].values >= 0.5)
        subgroup_size = subgroup_mask.sum()
        
        # Skip tiny subgroups
        if subgroup_size < 10:
            continue
        
        # Calculate metrics
        sg_auc = subgroup_auc(y_true, y_pred, subgroup_mask)
        sg_bpsn = bpsn_auc(y_true, y_pred, subgroup_mask)
        sg_bnsp = bnsp_auc(y_true, y_pred, subgroup_mask)
        
        # Store results
        results[subgroup] = {
            'subgroup_auc': sg_auc,
            'bpsn_auc': sg_bpsn,
            'bnsp_auc': sg_bnsp,
            'size': int(subgroup_size)
        }
        
        # Track worst subgroup AUC
        if not np.isnan(sg_auc) and sg_auc < worst_auc:
            worst_auc = sg_auc
            worst_subgroup = subgroup
    
    # Add worst subgroup information to overall results
    results['overall']['worst_auc'] = worst_auc
    results['overall']['worst_auc_identity'] = worst_subgroup
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-csv", default="data/valid.csv", 
                   help="Validation dataset with identity and target columns")
    ap.add_argument("--predictions-csv", default="output/preds/bert_headtail.csv",
                   help="Model predictions CSV file with prediction column")
    ap.add_argument("--model-name", default="bert_headtail",
                   help="Name of the model for plot title")
    ap.add_argument("--out-dir", default="figs/bias_aucs",
                   help="Output directory for visualizations")
    ap.add_argument("--pred-column", default="prediction",
                   help="Name of the prediction column in predictions CSV")
    args = ap.parse_args()

    # Create output directory
    out_d = pathlib.Path(args.out_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"🔹 Loading validation data and predictions from {args.predictions_csv}")
    df_valid = pd.read_csv(args.validation_csv)
    df_preds = pd.read_csv(args.predictions_csv)
    
    pred_col = args.pred_column
    
    # Ensure predictions are in the same order as validation data
    if 'id' in df_valid.columns and 'id' in df_preds.columns:
        # If we have IDs, merge on them
        df = df_valid.merge(df_preds[['id', pred_col]], on='id', how='inner')
        print(f"Merged {len(df)} rows using 'id' column")
    else:
        # Otherwise assume they're already aligned
        df = df_valid.copy()
        df[pred_col] = df_preds[pred_col].values
        print(f"Aligned {len(df)} rows (no merge)")
    
    # Get identity columns
    identity_columns = list_identity_columns(df)
    
    # Compute bias metrics
    print("🔹 Computing bias metrics for each subgroup")
    bias_metrics = compute_bias_metrics_for_model(df, identity_columns, 'target', pred_col)
    
    # Create a dataframe for plotting
    plot_data = []
    for subgroup in bias_metrics.keys():
        if subgroup == 'overall':
            continue
        
        # Extract metrics
        metrics = bias_metrics[subgroup]
        auc = metrics['subgroup_auc']
        bpsn_auc = metrics['bpsn_auc']
        bnsp_auc = metrics['bnsp_auc']
        
        # Add to plot data
        plot_data.extend([
            {'subgroup': subgroup, 'metric': 'Subgroup AUC', 'value': auc},
            {'subgroup': subgroup, 'metric': 'BPSN AUC', 'value': bpsn_auc},
            {'subgroup': subgroup, 'metric': 'BNSP AUC', 'value': bnsp_auc}
        ])
    
    # Convert to dataframe
    plot_df = pd.DataFrame(plot_data)
    
    # Calculate counts for each subgroup
    counts = {sg: (df[sg] >= 0.5).sum() for sg in identity_columns}
    
    # Sort subgroups by count (descending)
    sorted_subgroups = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    
    # Filter top N subgroups to keep plot readable
    top_n = 15
    top_subgroups = sorted_subgroups[:top_n]
    plot_df = plot_df[plot_df['subgroup'].isin(top_subgroups)]
    
    # Order subgroups by count
    plot_df['subgroup'] = pd.Categorical(plot_df['subgroup'], categories=top_subgroups, ordered=True)
    
    # Define colors for metrics
    colors = {"Subgroup AUC": "#3182bd", "BPSN AUC": "#31a354", "BNSP AUC": "#de2d26"}
    
    # Create side-by-side bar plot
    plt.figure(figsize=(12, 8))
    g = sns.barplot(
        data=plot_df,
        x='subgroup',
        y='value',
        hue='metric',
        palette=colors
    )
    
    # Add overall AUC reference line
    overall_auc = bias_metrics['overall']['auc']
    plt.axhline(y=overall_auc, color='black', linestyle='--', alpha=0.7, 
                label=f'Overall AUC: {overall_auc:.3f}')
    
    # Add counts to x-labels
    plt.xticks(rotation=45, ha='right')
    labels = [f"{sg}\n(n={counts[sg]:,})" for sg in top_subgroups]
    g.set_xticklabels(labels)
    
    plt.title(f"Bias AUCs by Identity Subgroup for {args.model_name}")
    plt.xlabel("")
    plt.ylabel("AUC Score")
    plt.ylim(0.5, 1.0)  # AUC is typically in range [0.5, 1.0]
    plt.legend(title="", loc="lower right")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    out_path = out_d / f"bias_aucs_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Bias AUCs plot saved to {out_path}")
    
    # Save data as CSV
    csv_path = out_d / f"bias_aucs_data.csv"
    plot_df.to_csv(csv_path, index=False)
    print(f"✅ Bias AUCs data saved to {csv_path}")
    
    # Print a summary of metrics
    print("\n===== BIAS METRICS SUMMARY =====")
    overall = bias_metrics['overall']
    print(f"Overall AUC: {overall['auc']:.4f}")
    print(f"Worst Subgroup AUC: {overall['worst_auc']:.4f} ({overall['worst_auc_identity']})")
    
    # Show min BPSN and BNSP metrics
    min_bpsn = min([bias_metrics[sg]['bpsn_auc'] for sg in top_subgroups 
                   if 'bpsn_auc' in bias_metrics[sg] and not np.isnan(bias_metrics[sg]['bpsn_auc'])])
    min_bpsn_sg = [sg for sg in top_subgroups 
                  if 'bpsn_auc' in bias_metrics[sg] and 
                  not np.isnan(bias_metrics[sg]['bpsn_auc']) and 
                  bias_metrics[sg]['bpsn_auc'] == min_bpsn][0]
    
    min_bnsp = min([bias_metrics[sg]['bnsp_auc'] for sg in top_subgroups 
                   if 'bnsp_auc' in bias_metrics[sg] and not np.isnan(bias_metrics[sg]['bnsp_auc'])])
    min_bnsp_sg = [sg for sg in top_subgroups 
                  if 'bnsp_auc' in bias_metrics[sg] and 
                  not np.isnan(bias_metrics[sg]['bnsp_auc']) and 
                  bias_metrics[sg]['bnsp_auc'] == min_bnsp][0]
    
    print(f"Worst BPSN AUC: {min_bpsn:.4f} ({min_bpsn_sg})")
    print(f"Worst BNSP AUC: {min_bnsp:.4f} ({min_bnsp_sg})")
    print("==============================")

if __name__ == "__main__":
    main() 