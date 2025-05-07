#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fairness Figures Generation
#
# This notebook generates a complete set of publication-ready fairness figures for each model whose metrics CSV lives in results/. All images are saved under the `figs/` directory.

# %% [markdown]
# ## Import Libraries

# %%
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import figure_utils, threshold_utils

# Create figs directory if it doesn't exist
os.makedirs('figs', exist_ok=True)

# %% [markdown]
# ## Parameters

# %%
# Parameters
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test_public_expanded.csv"
RESULTS_DIR = "results"
THRESHOLD = 0.5  # Decision threshold for binary classification

# %% [markdown]
# ## Utility Function to List Identity Columns

# %%
def list_identity_columns(df):
    """
    List all identity columns in the dataframe.
    
    Identity columns are typically those related to demographic groups.
    In the Jigsaw dataset, these are columns like 'male', 'female', 'black', etc.
    """
    # Common identity column patterns
    identity_patterns = [
        # Demographic groups
        'male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
        'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
        'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian', 'latino',
        'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness', 'other_disability',
        
        # Sometimes these are prefixed
        'identity_', 'demo_'
    ]
    
    identity_cols = []
    
    # Check each column in the dataframe
    for col in df.columns:
        # Check if the column matches any of the identity patterns
        if any(pattern in col.lower() for pattern in identity_patterns):
            identity_cols.append(col)
    
    return identity_cols

# %% [markdown]
# ## Generate Figure 1: Identity Prevalence

# %%
# Load the training data
train_df = pd.read_csv(TRAIN_CSV)

# Get identity columns
identity_cols = list_identity_columns(train_df)
print(f"Found {len(identity_cols)} identity columns: {identity_cols}")

# Generate the identity prevalence figure
prevalence_fig = figure_utils.identity_prevalence(train_df, identity_cols)
plt.close(prevalence_fig)  # Close the figure to free memory

print(f"Saved identity prevalence figure to figs/identity_prevalence.png")

# %% [markdown]
# ## Process Each Model's Results and Generate Figures

# %%
# Get all metrics files in the results directory
metrics_files = glob.glob(os.path.join(RESULTS_DIR, "metrics_*.csv"))
print(f"Found {len(metrics_files)} metrics files")

# Initialize a list to store figure metadata
figure_inventory = []

# %% [markdown]
# ## Loop Through Each Model's Metrics

# %%
for metrics_file in metrics_files:
    # Extract model name from the filename
    model_name = os.path.basename(metrics_file).replace("metrics_", "").replace(".csv", "")
    print(f"Processing model: {model_name}")
    
    # Load metrics
    try:
        metrics_df = pd.read_csv(metrics_file)
        print(f"Loaded metrics data with shape: {metrics_df.shape}")
    except Exception as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        continue
    
    # Check if this is a dry-run (all subgroup_auc values are NaN)
    if 'subgroup_auc' in metrics_df.columns and metrics_df['subgroup_auc'].isna().all():
        print(f"Dry-run detected: skipping figure generation for model {model_name}")
        continue
    
    # Check if the metrics file has the expected structure
    required_columns = ['identity_group', 'metric_name', 'value']
    if not all(col in metrics_df.columns for col in required_columns):
        # If not, try to reshape the data to match the expected format
        print(f"Metrics file doesn't have the expected columns: {required_columns}")
        print("Attempting to reshape data...")
        
        # Melt the DataFrame to convert it to the required format
        try:
            # Assuming first column is identity_group and other columns are metrics
            id_col = metrics_df.columns[0]
            metric_cols = metrics_df.columns[1:]
            
            metrics_df = pd.melt(
                metrics_df, 
                id_vars=[id_col], 
                value_vars=metric_cols,
                var_name='metric_name',
                value_name='value'
            )
            metrics_df = metrics_df.rename(columns={id_col: 'identity_group'})
            print(f"Reshaped metrics data to: {metrics_df.shape}")
        except Exception as e:
            print(f"Error reshaping metrics data: {e}")
            continue
    
    # Load predictions if available
    pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
    preds_df = None
    
    if os.path.exists(pred_file):
        try:
            preds_df = pd.read_csv(pred_file)
            print(f"Loaded predictions data with shape: {preds_df.shape}")
            
            # Check if the predictions file has the expected structure (id, prediction)
            if 'id' in preds_df.columns and 'prediction' in preds_df.columns:
                pass
            elif len(preds_df.columns) >= 2:
                # Rename the columns to match expected format
                preds_df = preds_df.iloc[:, :2]
                preds_df.columns = ['id', 'prediction']
            else:
                print(f"Predictions file doesn't have the expected structure")
                preds_df = None
                
            # If test_expanded file exists, merge with predictions
            if os.path.exists(TEST_CSV) and preds_df is not None:
                test_df = pd.read_csv(TEST_CSV)
                if 'id' in test_df.columns:
                    preds_df = pd.merge(preds_df, test_df, on='id', how='left')
                    print(f"Merged predictions with test data: {preds_df.shape}")
        except Exception as e:
            print(f"Error loading predictions file {pred_file}: {e}")
            preds_df = None
    
    # Generate Figure 2: ROC Curve
    if preds_df is not None and 'prediction' in preds_df.columns and 'target' in preds_df.columns:
        roc_fig = figure_utils.roc_curve_figure(
            preds_df['target'], 
            preds_df['prediction'], 
            model_name
        )
        plt.close(roc_fig)  # Close the figure to free memory
        figure_inventory.append(f"overall_roc_{model_name}.png")
        print(f"Generated ROC curve figure for {model_name}")
    else:
        print(f"Skipping ROC curve figure for {model_name} (missing predictions or target)")
    
    # Generate Figure 3: Fairness Heatmap (with updated settings)
    heatmap_fig = figure_utils.fairness_heatmap(metrics_df, model_name)
    plt.close(heatmap_fig)  # Close the figure to free memory
    figure_inventory.append(f"fairness_heatmap_{model_name}.png")
    print(f"Generated fairness heatmap figure for {model_name}")
    
    # Generate Figure 4: Power Mean Bar
    power_fig = figure_utils.power_mean_bar(metrics_df, model_name)
    plt.close(power_fig)  # Close the figure to free memory
    figure_inventory.append(f"power_mean_bar_{model_name}.png")
    print(f"Generated power mean bar figure for {model_name}")
    
    # Generate Figure 5: Grouped Bar by Identity (New)
    grouped_bar_fig = figure_utils.grouped_bar_by_identity(metrics_df, model_name)
    plt.close(grouped_bar_fig)  # Close the figure to free memory
    figure_inventory.append(f"grouped_bar_{model_name}.png")
    print(f"Generated grouped bar by identity figure for {model_name}")
    
    # Generate Figure 6: Worst K Bar (replaces worst k table)
    worst_k_fig = figure_utils.worst_k_bar(metrics_df, model_name, k=5)
    plt.close(worst_k_fig)  # Close the figure to free memory
    figure_inventory.append(f"worst_k_bar_{model_name}.png")
    print(f"Generated worst k bar figure for {model_name}")
    
    # Process error rate analysis if predictions are available
    if preds_df is not None and 'prediction' in preds_df.columns and 'target' in preds_df.columns:
        # Extract identity columns from the test data
        test_identity_cols = list_identity_columns(preds_df)
        
        if test_identity_cols:
            print(f"Found {len(test_identity_cols)} identity columns in test data")
            
            # Track worst performing subgroup
            worst_group = None
            worst_auc = 1.0
            
            # Calculate error rate gaps for each identity group
            gaps_dict = {}
            
            for identity_col in test_identity_cols:
                # Create identity mask
                identity_mask = preds_df[identity_col] == 1
                
                # Skip identities with too few samples
                if identity_mask.sum() < 50:
                    print(f"Skipping {identity_col} (insufficient samples: {identity_mask.sum()})")
                    continue
                
                # Calculate error rate gaps at threshold
                fpr_gap, fnr_gap = threshold_utils.error_rate_gaps(
                    preds_df['target'].values,
                    preds_df['prediction'].values, 
                    identity_mask.values,
                    thresh=THRESHOLD
                )
                
                # Store for heatmap
                gaps_dict[identity_col] = (fpr_gap, fnr_gap)
                
                # Find the worst performing group
                if 'subgroup_auc' in metrics_df['metric_name'].unique():
                    subgroup_metrics = metrics_df[
                        (metrics_df['metric_name'] == 'subgroup_auc') & 
                        (metrics_df['identity_group'] == identity_col)
                    ]
                    if not subgroup_metrics.empty:
                        auc_value = subgroup_metrics['value'].iloc[0]
                        if auc_value < worst_auc:
                            worst_auc = auc_value
                            worst_group = identity_col
                
                # Generate threshold sweep curve for current identity
                # First compute the sweep data
                df_sweep = threshold_utils.sweep_threshold_gaps(
                    preds_df['target'].values,
                    preds_df['prediction'].values, 
                    identity_mask.values,
                    n_steps=101
                )
                
                # Generate and save the curve plot
                curve_fig = figure_utils.threshold_gap_curve(df_sweep, identity_col, model_name)
                plt.close(curve_fig)  # Close the figure to free memory
                figure_inventory.append(f"threshold_gap_curve_{identity_col}_{model_name}.png")
                
            # Generate error gap heatmap with all identity groups
            if gaps_dict:
                error_heatmap_fig = figure_utils.error_gap_heatmap(
                    model_name,
                    gaps_dict,
                    threshold=THRESHOLD
                )
                plt.close(error_heatmap_fig)  # Close the figure to free memory
                figure_inventory.append(f"error_gap_heatmap_{model_name}.png")
                print(f"Generated error gap heatmap for {model_name}")
            
            # Generate confusion mosaic for worst performing identity group
            if worst_group:
                print(f"Worst performing identity group: {worst_group} (AUC: {worst_auc:.3f})")
                
                # Create binary predictions at threshold 0.5
                binary_preds = (preds_df['prediction'] >= THRESHOLD).astype(int)
                
                # Create confusion mosaic
                worst_mask = preds_df[worst_group] == 1
                
                mosaic_fig = figure_utils.add_confusion_mosaic(
                    preds_df['target'].values,
                    binary_preds.values,
                    worst_mask.values,
                    identity_name=worst_group,
                    model_name=model_name
                )
                plt.close(mosaic_fig)  # Close the figure to free memory
                figure_inventory.append(f"confusion_{worst_group}_{model_name}.png")
                print(f"Generated confusion mosaic for worst group {worst_group}")
            
    # Generate original threshold sweep if needed
    if preds_df is not None and 'prediction' in preds_df.columns and 'target' in preds_df.columns:
        test_identity_cols = list_identity_columns(preds_df)
        
        if test_identity_cols:
            # Create identity arrays for the threshold sweep
            identity_arrays = [preds_df[col].values for col in test_identity_cols]
            
            threshold_fig = figure_utils.threshold_sweep(
                preds_df['target'],
                preds_df['prediction'],
                identity_arrays,
                model_name
            )
            plt.close(threshold_fig)  # Close the figure to free memory
            figure_inventory.append(f"threshold_sweep_{model_name}.png")
            print(f"Generated threshold sweep figure for {model_name}")
        else:
            print(f"Skipping threshold sweep figure for {model_name} (no identity columns in test data)")
    else:
        print(f"Skipping threshold sweep figure for {model_name} (missing predictions or target)")

# %% [markdown]
# ## Generate Figure 7: Before vs After Scatter Plot

# %%
# Find baseline and improved model metrics
baseline_metrics_file = None
improved_metrics_file = None

# Look for baseline model metrics
for metrics_file in metrics_files:
    if "baseline" in metrics_file.lower():
        baseline_metrics_file = metrics_file
        break

# If no explicit baseline, use the first metrics file
if baseline_metrics_file is None and metrics_files:
    baseline_metrics_file = metrics_files[0]

# Look for improved model metrics (tfidf_lr_full is specified in the requirements)
for metrics_file in metrics_files:
    if "tfidf_lr_full" in metrics_file.lower():
        improved_metrics_file = metrics_file
        break

# If no explicit improved model, use the last metrics file (different from baseline)
if improved_metrics_file is None and len(metrics_files) > 1:
    improved_metrics_file = [f for f in metrics_files if f != baseline_metrics_file][0]

if baseline_metrics_file and improved_metrics_file and baseline_metrics_file != improved_metrics_file:
    print(f"Generating before vs after scatter plot:")
    print(f"  Baseline: {os.path.basename(baseline_metrics_file)}")
    print(f"  Improved: {os.path.basename(improved_metrics_file)}")
    
    # Load the metrics DataFrames
    baseline_df = pd.read_csv(baseline_metrics_file)
    improved_df = pd.read_csv(improved_metrics_file)
    
    # Check if the metrics files have the expected structure
    required_columns = ['identity_group', 'metric_name', 'value']
    
    for df, name in [(baseline_df, "baseline"), (improved_df, "improved")]:
        if not all(col in df.columns for col in required_columns):
            # If not, try to reshape the data to match the expected format
            print(f"{name} metrics file doesn't have the expected columns: {required_columns}")
            print("Attempting to reshape data...")
            
            # Melt the DataFrame to convert it to the required format
            try:
                # Assuming first column is identity_group and other columns are metrics
                id_col = df.columns[0]
                metric_cols = df.columns[1:]
                
                reshaped_df = pd.melt(
                    df, 
                    id_vars=[id_col], 
                    value_vars=metric_cols,
                    var_name='metric_name',
                    value_name='value'
                )
                reshaped_df = reshaped_df.rename(columns={id_col: 'identity_group'})
                
                if name == "baseline":
                    baseline_df = reshaped_df
                else:
                    improved_df = reshaped_df
                    
                print(f"Reshaped {name} metrics data successfully")
            except Exception as e:
                print(f"Error reshaping {name} metrics data: {e}")
                continue
    
    # Get common identity groups
    baseline_identity_groups = baseline_df['identity_group'].unique()
    improved_identity_groups = improved_df['identity_group'].unique()
    common_identity_groups = list(set(baseline_identity_groups) & set(improved_identity_groups))
    
    if common_identity_groups:
        scatter_fig = figure_utils.before_vs_after_scatter(
            baseline_df,
            improved_df,
            common_identity_groups
        )
        plt.close(scatter_fig)  # Close the figure to free memory
        figure_inventory.append("before_vs_after_scatter.png")
        print(f"Generated before vs after scatter plot")
    else:
        print("No common identity groups found between baseline and improved models")
else:
    print("Skipping before vs after scatter plot (couldn't find suitable baseline and improved models)")

# %% [markdown]
# ## Save Figure Inventory

# %%
# Save the figure inventory
inventory = {
    "figures": figure_inventory,
    "count": len(figure_inventory)
}

with open('figs/figure_inventory.json', 'w') as f:
    json.dump(inventory, f, indent=2)

print(f"Saved figure inventory to figs/figure_inventory.json")
print(f"Generated {len(figure_inventory)} figures in total")

# %% [markdown]
# ## Summary
#
# This notebook has generated the following figures:
#
# 1. `identity_prevalence.png` - Bar plot showing prevalence of identity groups
# 2. `overall_roc_{model}.png` - ROC curves for each model
# 3. `fairness_heatmap_{model}.png` - Heatmaps of fairness metrics for each model
# 4. `power_mean_bar_{model}.png` - Bar plots of power mean differences for each model
# 5. `grouped_bar_{model}.png` - Grouped bar plots of metrics by identity group
# 6. `worst_k_bar_{model}.png` - Bar charts of worst-performing identity groups
# 7. `threshold_sweep_{model}.png` - Threshold sweep analysis for each model
# 8. `threshold_gap_curve_{identity}_{model}.png` - FPR/FNR gap curves by threshold
# 9. `error_gap_heatmap_{model}.png` - Heatmap of FPR/FNR gaps across groups at τ=0.5
# 10. `confusion_{identity}_{model}.png` - Confusion matrix mosaic for worst subgroup
# 11. `before_vs_after_scatter.png` - Scatter plot comparing baseline and improved models
#
# All figures are saved in the `figs/` directory. 