#!/usr/bin/env python
# coding: utf-8

# # Jigsaw Unintended Bias Audit: Fairness Metrics Evaluation
# 
# This notebook evaluates bias metrics across different demographic subgroups for toxicity classification models.
# 
# **Papermill Parameters:**
# - `model_name`: Name of the model to evaluate (default: "tfidf_logreg")

# ## Setup

# In[1]:


# Parameters for papermill
model_name = "tfidf_logreg"  # Default model name, can be overridden by papermill


# In[2]:


import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# Add parent directory to path
sys.path.insert(0, os.path.abspath(".."))

# Import our metrics module
import src.metrics_v2 as metrics_v2

# Set up directories
DATA_DIR = "../data"
PREDS_DIR = "../output/preds"
RESULTS_DIR = "../results"
FIGURES_DIR = "../output/figures"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')


# ## Load Data

# In[3]:


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
        return pd.read_csv(train_file)
    
    raise FileNotFoundError("No ground truth data found. Please add train.csv or test_public_expanded.csv to the data directory.")


# In[4]:


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


# In[5]:


# Load ground truth data
ground_truth = load_ground_truth()

# Display basic info
print(f"Ground truth data shape: {ground_truth.shape}")
print(f"Columns: {', '.join(ground_truth.columns)}")


# In[6]:


# Identify identity columns
# Exclude standard non-identity columns
non_identity_cols = [
    'id', 'comment_text', 'target', 'toxicity', 'severe_toxicity', 
    'obscene', 'threat', 'insult', 'sexual_explicit'
]
identity_cols = [col for col in ground_truth.columns if col not in non_identity_cols]

print(f"Identified {len(identity_cols)} identity columns: {', '.join(identity_cols)}")


# In[7]:


# Load predictions for the specified model
predictions = load_predictions(model_name)
print(f"Prediction data shape: {predictions.shape}")


# ## Merge Data

# In[8]:


# Merge ground truth with predictions
merged_data = pd.merge(ground_truth, predictions, on='id')
print(f"Merged data shape: {merged_data.shape}")

# Check for missing values
missing_count = merged_data.isnull().sum().sum()
if missing_count > 0:
    print(f"Warning: {missing_count} missing values found in merged data")
    print(merged_data.isnull().sum()[merged_data.isnull().sum() > 0])


# ## Calculate Metrics

# In[9]:


# Extract arrays
y_true = merged_data['target'].values
y_pred = merged_data['prediction'].values

# Create subgroup masks
subgroup_masks = {}
for col in identity_cols:
    subgroup_masks[col] = merged_data[col].values.astype(bool)

# Calculate overall AUC
overall_auc = roc_auc_score(y_true, y_pred)
print(f"Overall AUC: {overall_auc:.4f}")


# In[10]:


# Calculate comprehensive metrics
print(f"Calculating bias metrics for {len(identity_cols)} identity subgroups...")
results = metrics_v2.compute_all_metrics(
    y_true=y_true,
    y_pred=y_pred,
    subgroup_masks=subgroup_masks,
    power=-5,            # Power parameter for generalized mean
    weight_overall=0.25  # Weight for overall AUC in final score
)

print(f"Bias calculation complete.")


# In[11]:


# Create a DataFrame with subgroup metrics
subgroup_metrics_df = pd.DataFrame(results["subgroup_metrics"])

# Add overall metrics
print(f"Overall AUC: {results['overall']['auc']:.4f}")
print(f"Final Score: {results['overall']['final_score']:.4f}")

# Display power means
for key, value in results["bias_metrics"].items():
    if key.startswith("power_mean"):
        print(f"{key}: {value:.4f}")


# ## Save Results

# In[12]:


# Save metrics to CSV
metrics_file = os.path.join(RESULTS_DIR, f"metrics_{model_name}.csv")
subgroup_metrics_df.to_csv(metrics_file, index=False)
print(f"Saved metrics to {metrics_file}")

# Save predictions to results directory for easier access by the dashboard
pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
if not os.path.exists(pred_file):
    predictions.to_csv(pred_file, index=False)
    print(f"Copied predictions to {pred_file}")


# ## Visualize Results

# In[13]:


# Import visualization utilities
from src.vis_utils import plot_auc_heatmap, plot_threshold_sweep

# Create heatmap
fig = plot_auc_heatmap(
    metrics_file,
    title=f"Bias Metrics for {model_name}",
    save_path=os.path.join(FIGURES_DIR, f"heatmap_{model_name}.svg")
)

# Display the figure
fig.show()


# In[14]:


# Create threshold sweep for a sample identity column
sample_identity = identity_cols[0]
sample_mask = subgroup_masks[sample_identity]

fig = plot_threshold_sweep(
    y_true,
    y_pred,
    sample_mask,
    sample_identity,
    save_path=os.path.join(FIGURES_DIR, f"threshold_sweep_{model_name}_{sample_identity}.svg")
)

# Display the figure
fig.show()


# ## Create Summary Report

# In[15]:


# Create a summary report with the key metrics
report = {
    "model_name": model_name,
    "overall_auc": results["overall"]["auc"],
    "final_score": results["overall"]["final_score"],
    "power_mean_subgroup_auc": results["bias_metrics"]["power_mean_subgroup_auc"],
    "power_mean_bpsn_auc": results["bias_metrics"]["power_mean_bpsn_auc"],
    "power_mean_bnsp_auc": results["bias_metrics"]["power_mean_bnsp_auc"],
    "subgroups_evaluated": len(identity_cols),
    "worst_subgroup_auc": subgroup_metrics_df["subgroup_auc"].min(),
    "worst_subgroup": subgroup_metrics_df.loc[subgroup_metrics_df["subgroup_auc"].idxmin(), "subgroup_name"]
}

# Convert to DataFrame for better display
report_df = pd.DataFrame([report])
report_df


# In[16]:


# Save summary report
report_file = os.path.join(RESULTS_DIR, f"summary_{model_name}.csv")
report_df.to_csv(report_file, index=False)
print(f"Saved summary report to {report_file}")

print("\nEvaluation complete!") 