#!/usr/bin/env python
# coding: utf-8

"""
Standalone script to calculate bias metrics and run the Streamlit dashboard
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Create necessary directories
RESULTS_DIR = "results"
PREDS_DIR = "output/preds"
FIGURES_DIR = "output/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def subgroup_auc(y_true, y_pred, subgroup_mask):
    """Calculate AUC for a specific demographic subgroup in vectorized form."""
    if np.sum(subgroup_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[subgroup_mask], y_pred[subgroup_mask])
    except ValueError:
        return np.nan

def bpsn_auc(y_true, y_pred, subgroup_mask):
    """Calculate Background Positive, Subgroup Negative (BPSN) AUC."""
    bpsn_mask = ((subgroup_mask) & (y_true == 0)) | ((~subgroup_mask) & (y_true == 1))
    
    if np.sum(bpsn_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[bpsn_mask], y_pred[bpsn_mask])
    except ValueError:
        return np.nan

def bnsp_auc(y_true, y_pred, subgroup_mask):
    """Calculate Background Negative, Subgroup Positive (BNSP) AUC."""
    bnsp_mask = ((~subgroup_mask) & (y_true == 0)) | ((subgroup_mask) & (y_true == 1))
    
    if np.sum(bnsp_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[bnsp_mask], y_pred[bnsp_mask])
    except ValueError:
        return np.nan

def generalized_power_mean(auc_list, p=-5):
    """Calculate the generalized power mean of AUC values."""
    valid_aucs = np.array([auc for auc in auc_list if not np.isnan(auc)])
    
    if len(valid_aucs) == 0:
        return np.nan
    
    return np.power(np.mean(np.power(valid_aucs, p)), 1/p)

def compute_bias_metrics_for_subgroup(y_true, y_pred, subgroup_mask, subgroup_name):
    """Compute all bias metrics for a single subgroup."""
    sub_auc = subgroup_auc(y_true, y_pred, subgroup_mask)
    bpsn = bpsn_auc(y_true, y_pred, subgroup_mask)
    bnsp = bnsp_auc(y_true, y_pred, subgroup_mask)
    
    subgroup_size = np.sum(subgroup_mask)
    subgroup_pos_rate = np.mean(y_true[subgroup_mask]) if subgroup_size > 0 else np.nan
    
    return {
        "subgroup_name": subgroup_name,
        "subgroup_size": int(subgroup_size),
        "subgroup_positive_rate": float(subgroup_pos_rate),
        "subgroup_auc": float(sub_auc),
        "bpsn_auc": float(bpsn),
        "bnsp_auc": float(bnsp)
    }

def compute_all_metrics(y_true, y_pred, subgroup_masks, power=-5, weight_overall=0.25):
    """Calculate all bias metrics and final score."""
    overall_auc = roc_auc_score(y_true, y_pred)
    
    # Calculate individual subgroup metrics
    subgroup_metrics = []
    bias_auc_dict = {"subgroup_auc": [], "bpsn_auc": [], "bnsp_auc": []}
    
    for subgroup_name, mask in subgroup_masks.items():
        metrics = compute_bias_metrics_for_subgroup(y_true, y_pred, mask, subgroup_name)
        subgroup_metrics.append(metrics)
        
        # Collect AUC values for power mean calculation
        bias_auc_dict["subgroup_auc"].append(metrics["subgroup_auc"])
        bias_auc_dict["bpsn_auc"].append(metrics["bpsn_auc"])
        bias_auc_dict["bnsp_auc"].append(metrics["bnsp_auc"])
    
    # Calculate power means for each bias metric type
    means = {}
    for metric_name, auc_values in bias_auc_dict.items():
        means[f"power_mean_{metric_name}"] = generalized_power_mean(auc_values, p=power)
    
    # Calculate bias component (average of the three means)
    bias_metrics = [
        means["power_mean_subgroup_auc"],
        means["power_mean_bpsn_auc"], 
        means["power_mean_bnsp_auc"]
    ]
    bias_score = np.mean([m for m in bias_metrics if not np.isnan(m)])
    
    # Calculate final weighted score
    final_score = weight_overall * overall_auc + (1 - weight_overall) * bias_score
    
    # Return results
    return {
        "overall": {"auc": overall_auc, "final_score": final_score},
        "bias_metrics": means,
        "subgroup_metrics": subgroup_metrics
    }

def create_synthetic_data(n_samples=10):
    """Create synthetic data for testing."""
    np.random.seed(42)
    
    # Generate IDs
    ids = list(range(1, n_samples + 1))
    
    # Generate predictions (0-1 values)
    predictions = np.random.random(n_samples)
    
    # Create a predictions dataframe
    preds_df = pd.DataFrame({"id": ids, "prediction": predictions})
    
    # Create ground truth
    target = (np.random.random(n_samples) > 0.7).astype(int)
    
    # Create 3 identity attributes for testing
    identity1 = (np.random.random(n_samples) > 0.5).astype(int)
    identity2 = (np.random.random(n_samples) > 0.7).astype(int)
    identity3 = (np.random.random(n_samples) > 0.3).astype(int)
    
    # Create ground truth dataframe
    truth_df = pd.DataFrame({
        "id": ids,
        "target": target,
        "identity1": identity1,
        "identity2": identity2,
        "identity3": identity3
    })
    
    return truth_df, preds_df

def main():
    """Main function to run the metrics calculation."""
    print("Creating synthetic data for bias metrics evaluation...")
    
    # Create synthetic data
    ground_truth, predictions = create_synthetic_data(n_samples=100)
    
    # Save synthetic data
    model_name = "synthetic_model"
    preds_file = os.path.join(PREDS_DIR, f"{model_name}.csv")
    predictions.to_csv(preds_file, index=False)
    print(f"Saved synthetic predictions to {preds_file}")
    
    # Prepare for metrics calculation
    y_true = ground_truth['target'].values
    y_pred = predictions['prediction'].values
    
    # Create subgroup masks
    subgroup_masks = {}
    for col in ['identity1', 'identity2', 'identity3']:
        subgroup_masks[col] = ground_truth[col].values.astype(bool)
    
    # Calculate metrics
    print("Calculating bias metrics...")
    results = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        subgroup_masks=subgroup_masks,
        power=-5,
        weight_overall=0.25
    )
    
    # Create a DataFrame with subgroup metrics
    subgroup_metrics_df = pd.DataFrame(results["subgroup_metrics"])
    
    # Print results
    print("\nResults:")
    print(f"Overall AUC: {results['overall']['auc']:.4f}")
    print(f"Final Score: {results['overall']['final_score']:.4f}")
    
    # Display power means
    for key, value in results["bias_metrics"].items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics to CSV
    metrics_file = os.path.join(RESULTS_DIR, f"metrics_{model_name}.csv")
    subgroup_metrics_df.to_csv(metrics_file, index=False)
    print(f"\nSaved metrics to {metrics_file}")
    
    # Save predictions to results directory for dashboard
    results_pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
    predictions.to_csv(results_pred_file, index=False)
    print(f"Saved predictions to {results_pred_file}")
    
    print("\nMetrics calculation complete. You can now run the Streamlit dashboard with:")
    print("streamlit run app/app.py")

if __name__ == "__main__":
    main() 