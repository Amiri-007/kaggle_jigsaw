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

# Import metrics_v2 module directly (avoiding src package imports)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.metrics_v2 import (
    compute_all_metrics,
    list_identity_columns,
    BiasReport
)

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
results = compute_all_metrics(
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


# Define visualization utilities directly in this notebook
def plot_auc_heatmap(metrics_file, title="Bias Metrics Heatmap", save_path=None):
    """Create a heatmap visualization of bias metrics."""
    # Load metrics
    df = pd.read_csv(metrics_file)
    
    # Sort data by subgroup size
    df = df.sort_values(by='subgroup_size', ascending=False)
    
    # Prepare data for heatmap
    subgroups = df['subgroup_name'].tolist()
    metric_columns = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df[metric_columns].values,
        x=["Subgroup AUC", "BPSN AUC", "BNSP AUC"],
        y=subgroups,
        colorscale='RdBu',
        zmid=0.5,
        zmin=0.3,
        zmax=1.0,
        colorbar=dict(title="AUC Score"),
        hoverinfo="z+y",
        text=df[metric_columns].round(4).astype(str).values,
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Metric Type"),
        yaxis=dict(title="Identity Subgroup", autorange="reversed"),
        height=max(400, 30 * len(subgroups)),
        margin=dict(l=100, r=20, t=70, b=50),
    )
    
    # Save if path provided
    if save_path:
        fig.write_image(save_path)
    
    return fig

def plot_threshold_sweep(y_true, y_pred, subgroup_mask, subgroup_name, save_path=None):
    """Create a plot showing impact of different thresholds on a subgroup."""
    thresholds = np.linspace(0.1, 0.9, 9)
    
    # Calculate TPR and FPR at different thresholds
    background_tpr = []
    background_fpr = []
    subgroup_tpr = []
    subgroup_fpr = []
    
    for threshold in thresholds:
        # Overall
        y_pred_binary = (y_pred >= threshold).astype(int)
        overall_tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        overall_fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        overall_tn = np.sum((y_pred_binary == 0) & (y_true == 0))
        overall_fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        
        overall_tpr = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_fpr = overall_fp / (overall_fp + overall_tn) if (overall_fp + overall_tn) > 0 else 0
        
        # Background (not in subgroup)
        background_pred = y_pred[~subgroup_mask]
        background_true = y_true[~subgroup_mask]
        background_pred_binary = (background_pred >= threshold).astype(int)
        
        bg_tp = np.sum((background_pred_binary == 1) & (background_true == 1))
        bg_fp = np.sum((background_pred_binary == 1) & (background_true == 0))
        bg_tn = np.sum((background_pred_binary == 0) & (background_true == 0))
        bg_fn = np.sum((background_pred_binary == 0) & (background_true == 1))
        
        bg_tpr = bg_tp / (bg_tp + bg_fn) if (bg_tp + bg_fn) > 0 else 0
        bg_fpr = bg_fp / (bg_fp + bg_tn) if (bg_fp + bg_tn) > 0 else 0
        
        background_tpr.append(bg_tpr)
        background_fpr.append(bg_fpr)
        
        # Subgroup
        subgroup_pred = y_pred[subgroup_mask]
        subgroup_true = y_true[subgroup_mask]
        subgroup_pred_binary = (subgroup_pred >= threshold).astype(int)
        
        sg_tp = np.sum((subgroup_pred_binary == 1) & (subgroup_true == 1))
        sg_fp = np.sum((subgroup_pred_binary == 1) & (subgroup_true == 0))
        sg_tn = np.sum((subgroup_pred_binary == 0) & (subgroup_true == 0))
        sg_fn = np.sum((subgroup_pred_binary == 0) & (subgroup_true == 1))
        
        sg_tpr = sg_tp / (sg_tp + sg_fn) if (sg_tp + sg_fn) > 0 else 0
        sg_fpr = sg_fp / (sg_fp + sg_tn) if (sg_fp + sg_tn) > 0 else 0
        
        subgroup_tpr.append(sg_tpr)
        subgroup_fpr.append(sg_fpr)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plots for TPR and FPR
    fig.add_trace(go.Scatter(
        x=thresholds, y=background_tpr, mode='lines+markers',
        name='Background TPR', line=dict(color='blue', dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=background_fpr, mode='lines+markers',
        name='Background FPR', line=dict(color='blue', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=subgroup_tpr, mode='lines+markers',
        name=f'{subgroup_name} TPR', line=dict(color='red', dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=subgroup_fpr, mode='lines+markers',
        name=f'{subgroup_name} FPR', line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Threshold Impact: {subgroup_name} vs Background',
        xaxis_title='Threshold',
        yaxis_title='Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        width=800,
        height=500
    )
    
    # Save if path provided
    if save_path:
        fig.write_image(save_path)
    
    return fig


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