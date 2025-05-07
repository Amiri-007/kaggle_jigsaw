#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run bias metrics evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# Add current directory to path
sys.path.insert(0, os.path.abspath("."))

# Import our metrics module
import src.metrics_v2 as metrics_v2

# Set up directories
DATA_DIR = "data"
PREDS_DIR = "output/preds"
RESULTS_DIR = "results"
FIGURES_DIR = "output/figures"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Model name to evaluate
model_name = "sample_model"

# Load ground truth data
def load_ground_truth():
    """Load ground truth data with identity columns."""
    # First try test data (for real evaluation)
    test_file = os.path.join(DATA_DIR, "test_public_expanded.csv")
    if os.path.exists(test_file):
        print(f"Loading test data from {test_file}")
        return pd.read_csv(test_file)
    
    # Fall back to train data
    train_file = os.path.join(DATA_DIR, "train.csv")
    if os.path.exists(train_file):
        print(f"Loading train data from {train_file}")
        # For testing, let's just load a small subset
        return pd.read_csv(train_file, nrows=1000)
    
    raise FileNotFoundError("No ground truth data found. Please add train.csv or test_public_expanded.csv to the data directory.")

# Load predictions
def load_predictions(model_name):
    """Load model predictions."""
    pred_file = os.path.join(PREDS_DIR, f"{model_name}.csv")
    if not os.path.exists(pred_file):
        # Try looking in the results directory
        pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions file not found for model: {model_name}")
    
    print(f"Loading predictions from {pred_file}")
    return pd.read_csv(pred_file)

print(f"Running bias evaluation for model: {model_name}")

# Load data
ground_truth = load_ground_truth()

# Identify identity columns
# Exclude standard non-identity columns
non_identity_cols = [
    'id', 'comment_text', 'target', 'toxicity', 'severe_toxicity', 
    'obscene', 'threat', 'insult', 'sexual_explicit'
]
identity_cols = [col for col in ground_truth.columns if col not in non_identity_cols]

print(f"Identified {len(identity_cols)} identity columns: {', '.join(identity_cols[:5])}...")

# Load predictions for the specified model
predictions = load_predictions(model_name)
print(f"Prediction data shape: {predictions.shape}")

# Create a temporary subset of data to match our sample predictions
# Just keep the first N rows where N is the number of predictions
ground_truth_subset = ground_truth.iloc[:len(predictions)]
ground_truth_subset = ground_truth_subset.copy()

# Ensure predictions and ground truth have matching IDs
# For demonstration, we'll just use the first few rows and assign IDs
ground_truth_subset['id'] = predictions['id'].values

# Merge ground truth with predictions
merged_data = pd.merge(ground_truth_subset, predictions, on='id')
print(f"Merged data shape: {merged_data.shape}")

# Extract arrays
y_true = merged_data['target'].values
y_pred = merged_data['prediction'].values

# Create subgroup masks
subgroup_masks = {}
for col in identity_cols:
    if col in merged_data.columns:
        subgroup_masks[col] = merged_data[col].values.astype(bool)

# Calculate overall AUC
try:
    overall_auc = roc_auc_score(y_true, y_pred)
    print(f"Overall AUC: {overall_auc:.4f}")

    # Calculate comprehensive metrics
    print(f"Calculating bias metrics for {len(subgroup_masks)} identity subgroups...")
    results = metrics_v2.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        subgroup_masks=subgroup_masks,
        power=-5,            # Power parameter for generalized mean
        weight_overall=0.25  # Weight for overall AUC in final score
    )

    print(f"Bias calculation complete.")

    # Create a DataFrame with subgroup metrics
    subgroup_metrics_df = pd.DataFrame(results["subgroup_metrics"])

    # Add overall metrics
    print(f"Overall AUC: {results['overall']['auc']:.4f}")
    print(f"Final Score: {results['overall']['final_score']:.4f}")

    # Display power means
    for key, value in results["bias_metrics"].items():
        if key.startswith("power_mean"):
            print(f"{key}: {value:.4f}")

    # Save metrics to CSV
    metrics_file = os.path.join(RESULTS_DIR, f"metrics_{model_name}.csv")
    subgroup_metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics to {metrics_file}")

    # Save predictions to the results directory for the dashboard
    preds_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
    predictions.to_csv(preds_file, index=False)
    print(f"Saved predictions to {preds_file}")

except Exception as e:
    print(f"Error calculating metrics: {e}") 