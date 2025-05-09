#!/usr/bin/env python
"""
Simplified script to run fairness metrics
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import fairness metrics
from fairness import BiasReport, final_score
from src.data.utils import list_identity_columns

def main():
    # Prediction file path
    pred_path = "output/preds/simplest_preds.csv"
    model_name = "distilbert_simplest"
    
    print(f"Loading predictions from {pred_path}")
    
    # Load predictions
    preds_df = pd.read_csv(pred_path)
    
    # Print column info for debugging
    print(f"Prediction columns: {preds_df.columns.tolist()}")
    print(f"Number of predictions: {len(preds_df)}")
    
    # Load ground truth
    gt_path = "data/valid.csv"
    if os.path.exists(gt_path):
        print(f"Loading ground truth from {gt_path}")
        truth_df = pd.read_csv(gt_path)
        print(f"Ground truth columns: {truth_df.columns.tolist()}")
        
        # Merge predictions with ground truth
        df = preds_df.merge(truth_df, on='id', how='inner')
        print(f"Merged data size: {len(df)}")
    else:
        print(f"Ground truth file {gt_path} not found")
        sys.exit(1)
    
    # Get identity columns
    identity_cols = list_identity_columns()
    
    # Check for missing identity columns
    missing_cols = [col for col in identity_cols if col not in df.columns]
    if missing_cols:
        print(f"Adding missing identity columns: {missing_cols}")
        for col in missing_cols:
            df[col] = 0
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create BiasReport
        print("Creating bias report...")
        bias_report = BiasReport(
            df=df,
            identity_cols=identity_cols,
            label_col='target',
            pred_col='prediction'
        )
        
        # Get metrics
        metrics_df = bias_report.get_metrics_df()
        
        # Calculate final score
        score = final_score(bias_report)
        print(f"Final fairness score: {score:.4f}")
        
        # Save metrics
        output_path = output_dir / f"metrics_{model_name}.csv"
        metrics_df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")
        
        # Get overall AUC
        overall_auc = metrics_df.loc[metrics_df['subgroup'] == 'overall', 'subgroup_auc'].iloc[0]
        print(f"Overall AUC: {overall_auc:.4f}")
        
        # Find worst performing subgroup
        subgroup_metrics = metrics_df[metrics_df['subgroup'] != 'overall']
        worst_subgroup = subgroup_metrics.loc[subgroup_metrics['subgroup_auc'].idxmin()]
        worst_sub_name = worst_subgroup['subgroup']
        worst_sub_auc = worst_subgroup['subgroup_auc']
        print(f"Worst performing subgroup: {worst_sub_name} (AUC: {worst_sub_auc:.4f})")
        
        # Append a summary line to summary.tsv
        summary_path = output_dir / 'summary.tsv'
        summary_exists = summary_path.exists()
        
        with open(summary_path, 'a') as f:
            # Write header if file doesn't exist
            if not summary_exists:
                f.write("model_name\toverall_auc\tfinal_score\tworst_subgroup\tworst_sub_auc\n")
            
            # Write summary line
            f.write(f"{model_name}\t{overall_auc:.6f}\t{score:.6f}\t{worst_sub_name}\t{worst_sub_auc:.6f}\n")
        
        print(f"Summary added to {summary_path}")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 