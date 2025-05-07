#!/usr/bin/env python
"""
Script to generate fairness figures for model analysis.
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Import our own modules after setting path
from src.figure_utils import (
    identity_prevalence, 
    roc_curve_figure, 
    fairness_heatmap, 
    power_mean_bar,
    threshold_sweep, 
    worst_k_table, 
    before_vs_after_scatter
)

# Ensure directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

# Parameters
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test_public_expanded.csv"
RESULTS_DIR = "results"

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

def main():
    # Initialize figure inventory
    figure_inventory = []
    
    # Check if necessary files exist
    print(f"Checking for necessary files...")
    train_exists = os.path.exists(TRAIN_CSV)
    test_exists = os.path.exists(TEST_CSV)
    
    print(f"Train data: {'Found' if train_exists else 'Not found'}")
    print(f"Test data: {'Found' if test_exists else 'Not found'}")
    
    # Get metrics files
    metrics_files = glob.glob(os.path.join(RESULTS_DIR, "metrics_*.csv"))
    print(f"Found {len(metrics_files)} metrics files")
    
    # Get prediction files
    pred_files = glob.glob(os.path.join(RESULTS_DIR, "preds_*.csv"))
    print(f"Found {len(pred_files)} prediction files")
    
    # Generate identity prevalence figure if train data exists
    if train_exists:
        print("\nGenerating identity prevalence figure...")
        try:
            # Load training data
            train_df = pd.read_csv(TRAIN_CSV)
            
            # Get identity columns
            identity_cols = list_identity_columns(train_df)
            print(f"Found {len(identity_cols)} identity columns")
            
            # Generate figure
            fig = identity_prevalence(train_df, identity_cols)
            plt.close(fig)
            figure_inventory.append("identity_prevalence.png")
            print("✓ Generated identity prevalence figure")
        except Exception as e:
            print(f"Error generating identity prevalence figure: {e}")
    
    # Process each model's metrics and predictions
    for metrics_file in metrics_files:
        # Extract model name from filename
        model_name = os.path.basename(metrics_file).replace("metrics_", "").replace(".csv", "")
        print(f"\nProcessing model: {model_name}")
        
        try:
            # Load metrics
            metrics_df = pd.read_csv(metrics_file)
            print(f"Loaded metrics with shape: {metrics_df.shape}")
            
            # Check if metrics file has the expected structure
            required_columns = ['identity_group', 'metric_name', 'value']
            if not all(col in metrics_df.columns for col in required_columns):
                print(f"Metrics file doesn't have expected columns. Attempting to reshape...")
                
                # Reshape data to match expected format
                # Assuming first column is identity_group and others are metrics
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
                print(f"Reshaped metrics to: {metrics_df.shape}")
            
            # Generate fairness heatmap
            print("Generating fairness heatmap...")
            fig = fairness_heatmap(metrics_df, model_name)
            plt.close(fig)
            figure_inventory.append(f"fairness_heatmap_{model_name}.png")
            print("✓ Generated fairness heatmap")
            
            # Generate power mean bar plot
            print("Generating power mean bar plot...")
            fig = power_mean_bar(metrics_df, model_name)
            plt.close(fig)
            figure_inventory.append(f"power_mean_bar_{model_name}.png")
            print("✓ Generated power mean bar plot")
            
            # Generate worst K table
            print("Generating worst K table...")
            fig = worst_k_table(metrics_df, k=5, model_name=model_name)
            plt.close(fig)
            figure_inventory.append(f"worst_k_table_{model_name}.png")
            print("✓ Generated worst K table")
            
            # Look for corresponding predictions file
            pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
            if os.path.exists(pred_file):
                print(f"Found predictions file: {pred_file}")
                
                # Load predictions
                preds_df = pd.read_csv(pred_file)
                print(f"Loaded predictions with shape: {preds_df.shape}")
                
                # Check if predictions file has expected structure
                if 'id' in preds_df.columns and 'prediction' in preds_df.columns:
                    pass
                elif len(preds_df.columns) >= 2:
                    # Rename columns to match expected format
                    preds_df = preds_df.iloc[:, :2]
                    preds_df.columns = ['id', 'prediction']
                
                # If test_expanded file exists, merge with predictions
                if test_exists:
                    test_df = pd.read_csv(TEST_CSV)
                    if 'id' in test_df.columns:
                        preds_df = pd.merge(preds_df, test_df, on='id', how='left')
                        print(f"Merged predictions with test data: {preds_df.shape}")
                
                # Generate ROC curve figure if we have necessary data
                if 'target' in preds_df.columns:
                    print("Generating ROC curve...")
                    fig = roc_curve_figure(preds_df['target'], preds_df['prediction'], model_name)
                    plt.close(fig)
                    figure_inventory.append(f"overall_roc_{model_name}.png")
                    print("✓ Generated ROC curve")
                    
                    # Generate threshold sweep figure if we have identity columns
                    test_identity_cols = list_identity_columns(preds_df)
                    if test_identity_cols:
                        print("Generating threshold sweep...")
                        identity_arrays = [preds_df[col].values for col in test_identity_cols]
                        
                        fig = threshold_sweep(
                            preds_df['target'],
                            preds_df['prediction'],
                            identity_arrays,
                            model_name
                        )
                        plt.close(fig)
                        figure_inventory.append(f"threshold_sweep_{model_name}.png")
                        print("✓ Generated threshold sweep")
        
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
    
    # Generate before vs after scatter plot
    if len(metrics_files) >= 2:
        print("\nGenerating before vs after scatter plot...")
        
        # Find baseline and improved model metrics
        baseline_metrics_file = None
        improved_metrics_file = None
        
        # Look for baseline model
        for file in metrics_files:
            if "baseline" in file.lower():
                baseline_metrics_file = file
                break
        
        # If no explicit baseline, use first metrics file
        if baseline_metrics_file is None and metrics_files:
            baseline_metrics_file = metrics_files[0]
        
        # Look for improved model
        for file in metrics_files:
            if "tfidf_lr_full" in file.lower():
                improved_metrics_file = file
                break
        
        # If no explicit improved model, use another metrics file
        if improved_metrics_file is None and len(metrics_files) > 1:
            candidates = [f for f in metrics_files if f != baseline_metrics_file]
            if candidates:
                improved_metrics_file = candidates[0]
        
        if baseline_metrics_file and improved_metrics_file and baseline_metrics_file != improved_metrics_file:
            try:
                print(f"  Baseline: {os.path.basename(baseline_metrics_file)}")
                print(f"  Improved: {os.path.basename(improved_metrics_file)}")
                
                # Load metrics
                baseline_df = pd.read_csv(baseline_metrics_file)
                improved_df = pd.read_csv(improved_metrics_file)
                
                # Check and reshape data if needed
                required_columns = ['identity_group', 'metric_name', 'value']
                
                for df_name, df in [("baseline", baseline_df), ("improved", improved_df)]:
                    if not all(col in df.columns for col in required_columns):
                        print(f"{df_name} metrics don't have expected columns. Reshaping...")
                        
                        # Reshape
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
                        
                        if df_name == "baseline":
                            baseline_df = reshaped_df
                        else:
                            improved_df = reshaped_df
                
                # Get common identity groups
                baseline_identity_groups = baseline_df['identity_group'].unique()
                improved_identity_groups = improved_df['identity_group'].unique()
                common_identity_groups = list(set(baseline_identity_groups) & set(improved_identity_groups))
                
                if common_identity_groups:
                    fig = before_vs_after_scatter(
                        baseline_df,
                        improved_df,
                        common_identity_groups
                    )
                    plt.close(fig)
                    figure_inventory.append("before_vs_after_scatter.png")
                    print("✓ Generated before vs after scatter plot")
            except Exception as e:
                print(f"Error generating before vs after scatter plot: {e}")
    
    # Save figure inventory
    inventory = {
        "figures": figure_inventory,
        "count": len(figure_inventory)
    }
    
    with open('figs/figure_inventory.json', 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\nSaved figure inventory to figs/figure_inventory.json")
    print(f"Generated {len(figure_inventory)} figures in total:")
    for fig in figure_inventory:
        print(f"  - {fig}")

if __name__ == "__main__":
    main() 