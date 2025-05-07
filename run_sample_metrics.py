#!/usr/bin/env python
# coding: utf-8

"""
Script to calculate metrics for the sample_model data
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Create necessary directories
RESULTS_DIR = "results"
PREDS_DIR = "output/preds"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

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

def create_synthetic_data_for_sample_model():
    """Create synthetic ground truth data for the sample_model predictions."""
    # Load sample model predictions
    model_name = "larger_sample_model"
    pred_file = os.path.join(PREDS_DIR, f"{model_name}.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    predictions = pd.read_csv(pred_file)
    n_samples = len(predictions)
    
    # Generate synthetic ground truth
    np.random.seed(42)
    
    # Generate target values (0/1)
    target = (np.random.random(n_samples) > 0.7).astype(int)
    
    # Create 3 identity attributes for testing
    identity1 = (np.random.random(n_samples) > 0.5).astype(int)
    identity2 = (np.random.random(n_samples) > 0.7).astype(int)
    identity3 = (np.random.random(n_samples) > 0.3).astype(int)
    
    # Create ground truth dataframe with matching IDs
    truth_df = pd.DataFrame({
        "id": predictions['id'].values,
        "target": target,
        "identity1": identity1,
        "identity2": identity2,
        "identity3": identity3
    })
    
    return truth_df, predictions

def main():
    """Main function to run the metrics calculation."""
    print("Creating synthetic data for larger sample model evaluation...")
    
    # Create synthetic data for the sample model
    ground_truth, predictions = create_synthetic_data_for_sample_model()
    model_name = "larger_sample_model"
    
    print(f"Processing model: {model_name}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    
    # Prepare for metrics calculation
    y_true = ground_truth['target'].values
    y_pred = predictions['prediction'].values
    
    # Create subgroup masks
    subgroup_masks = {}
    for col in ['identity1', 'identity2', 'identity3']:
        subgroup_masks[col] = ground_truth[col].values.astype(bool)
    
    # Calculate metrics for each subgroup
    subgroup_metrics = []
    
    for subgroup_name, mask in subgroup_masks.items():
        metrics = compute_bias_metrics_for_subgroup(y_true, y_pred, mask, subgroup_name)
        subgroup_metrics.append(metrics)
        print(f"Subgroup: {subgroup_name}, AUC: {metrics['subgroup_auc']:.4f}")
    
    # Create a DataFrame with subgroup metrics
    subgroup_metrics_df = pd.DataFrame(subgroup_metrics)
    
    # Save metrics to CSV
    metrics_file = os.path.join(RESULTS_DIR, f"metrics_{model_name}.csv")
    subgroup_metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics to {metrics_file}")
    
    # Save predictions to results directory for dashboard
    results_pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
    predictions.to_csv(results_pred_file, index=False)
    print(f"Saved predictions to {results_pred_file}")
    
    print("\nMetrics calculation complete. You can now run the Streamlit dashboard with:")
    print("streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main() 