#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Plotting Utilities

This module provides functions for visualizing model performance 
and bias metrics across different demographic subgroups.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import MaxNLocator

# Create figs directory if it doesn't exist
Path("figs").mkdir(exist_ok=True)

def set_style():
    """Set the plotting style for consistent visualizations."""
    # Handle different seaborn versions
    try:
        plt.style.use("seaborn-whitegrid")  # For newer versions
    except:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")  # For older versions
        except:
            plt.style.use("whitegrid")  # Fallback
    
    sns.set_context("talk")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

def set_plotting_style():
    """Set consistent matplotlib/seaborn style for plots."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def plot_subgroup_auc(metrics, identity_columns, model_names=None, save_path=None):
    """
    Plot AUC scores for each subgroup across different models.
    
    Args:
        metrics (dict): Dictionary containing model metrics from bias_metrics.compare_models()
        identity_columns (list): List of column names for identity subgroups
        model_names (list, optional): List of model names to include in plot
        save_path (str, optional): Path to save the figure
    """
    set_style()
    
    if model_names is None:
        model_names = list(metrics.keys())
    
    # Extract AUC values
    data = []
    for model in model_names:
        overall_auc = metrics[model]["overall"]["auc"]
        for subgroup in identity_columns:
            if subgroup in metrics[model]["subgroup"]:
                subgroup_auc = metrics[model]["subgroup"][subgroup]
                data.append({
                    "Model": model,
                    "Subgroup": subgroup,
                    "AUC": subgroup_auc,
                    "Overall AUC": overall_auc
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Set up bar positions
    n_models = len(model_names)
    n_groups = len(identity_columns)
    bar_width = 0.8 / n_models
    
    # Create the x-axis positions
    index = np.arange(n_groups)
    
    # Plot bars for each model
    for i, model in enumerate(model_names):
        # Get subset of data for this model
        model_data = df[df["Model"] == model]
        
        # Get overall AUC for reference line
        overall_auc = model_data["Overall AUC"].iloc[0] if len(model_data) > 0 else None
        
        # Extract AUC values for each subgroup, maintaining order
        auc_values = []
        for subgroup in identity_columns:
            subgroup_data = model_data[model_data["Subgroup"] == subgroup]
            if len(subgroup_data) > 0:
                auc_values.append(subgroup_data["AUC"].iloc[0])
            else:
                auc_values.append(np.nan)
        
        # Plot bars
        bars = plt.bar(index + i * bar_width - (bar_width * (n_models-1))/2, 
                        auc_values, bar_width, label=model)
        
        # Plot reference line for overall AUC
        if overall_auc is not None:
            plt.axhline(overall_auc, linestyle="--", color=bars[0].get_facecolor(), alpha=0.5)
    
    plt.xlabel("Demographic Subgroup")
    plt.ylabel("AUC Score")
    plt.title("Model Performance Across Demographic Subgroups")
    plt.xticks(index, identity_columns, rotation=45, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_bias_comparison(metrics, identity_columns, bias_type="bpsn", 
                        model_names=None, save_path=None):
    """
    Plot bias metrics (BPSN or BNSP) across subgroups for different models.
    
    Args:
        metrics (dict): Dictionary containing model metrics from bias_metrics.compare_models()
        identity_columns (list): List of column names for identity subgroups
        bias_type (str): Type of bias to plot ("bpsn" or "bnsp")
        model_names (list, optional): List of model names to include in plot
        save_path (str, optional): Path to save the figure
    """
    set_style()
    
    if model_names is None:
        model_names = list(metrics.keys())
    
    # Check bias_type
    if bias_type not in ["bpsn", "bnsp"]:
        raise ValueError("bias_type must be either 'bpsn' or 'bnsp'")
    
    # Extract bias values
    data = []
    for model in model_names:
        for subgroup in identity_columns:
            if subgroup in metrics[model][bias_type]:
                bias_value = metrics[model][bias_type][subgroup]
                overall_auc = metrics[model]["overall"]["auc"]
                data.append({
                    "Model": model,
                    "Subgroup": subgroup,
                    "Bias": bias_value,
                    "Overall AUC": overall_auc
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Set up bar positions
    n_models = len(model_names)
    n_groups = len(identity_columns)
    bar_width = 0.8 / n_models
    
    # Create the x-axis positions
    index = np.arange(n_groups)
    
    # Plot bars for each model
    for i, model in enumerate(model_names):
        # Get subset of data for this model
        model_data = df[df["Model"] == model]
        
        # Extract bias values for each subgroup, maintaining order
        bias_values = []
        for subgroup in identity_columns:
            subgroup_data = model_data[model_data["Subgroup"] == subgroup]
            if len(subgroup_data) > 0:
                bias_values.append(subgroup_data["Bias"].iloc[0])
            else:
                bias_values.append(np.nan)
        
        # Plot bars
        bars = plt.bar(index + i * bar_width - (bar_width * (n_models-1))/2, 
                        bias_values, bar_width, label=model)
    
    bias_name = "Background Positive, Subgroup Negative (BPSN)" if bias_type == "bpsn" else "Background Negative, Subgroup Positive (BNSP)"
    
    plt.xlabel("Demographic Subgroup")
    plt.ylabel("AUC Score")
    plt.title(f"{bias_name} Bias Across Demographic Subgroups")
    plt.xticks(index, identity_columns, rotation=45, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_power_diff(metrics, identity_columns, model_names=None, save_path=None):
    """
    Plot power difference across subgroups for different models.
    
    Args:
        metrics (dict): Dictionary containing model metrics from bias_metrics.compare_models()
        identity_columns (list): List of column names for identity subgroups
        model_names (list, optional): List of model names to include in plot
        save_path (str, optional): Path to save the figure
    """
    set_style()
    
    if model_names is None:
        model_names = list(metrics.keys())
    
    # Extract power difference values
    data = []
    for model in model_names:
        for subgroup in identity_columns:
            if subgroup in metrics[model]["power_diff"] and metrics[model]["power_diff"][subgroup] is not None:
                power_diff = metrics[model]["power_diff"][subgroup]
                data.append({
                    "Model": model,
                    "Subgroup": subgroup,
                    "Power Difference": power_diff
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Set up bar positions
    n_models = len(model_names)
    n_groups = len(identity_columns)
    bar_width = 0.8 / n_models
    
    # Create the x-axis positions
    index = np.arange(n_groups)
    
    # Plot bars for each model
    for i, model in enumerate(model_names):
        # Get subset of data for this model
        model_data = df[df["Model"] == model]
        
        # Extract power difference values for each subgroup, maintaining order
        power_diff_values = []
        for subgroup in identity_columns:
            subgroup_data = model_data[model_data["Subgroup"] == subgroup]
            if len(subgroup_data) > 0:
                power_diff_values.append(subgroup_data["Power Difference"].iloc[0])
            else:
                power_diff_values.append(np.nan)
        
        # Plot bars
        bars = plt.bar(index + i * bar_width - (bar_width * (n_models-1))/2, 
                        power_diff_values, bar_width, label=model)
    
    # Add a horizontal line at y=1 (no power difference)
    plt.axhline(1, linestyle="--", color="black", alpha=0.5)
    
    plt.xlabel("Demographic Subgroup")
    plt.ylabel("Power Difference (Toxic Rate / Non-Toxic Rate)")
    plt.title("Power Difference Across Demographic Subgroups")
    plt.xticks(index, identity_columns, rotation=45, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_confusion_heatmap(df, pred_col, label_col="target", threshold=0.5, save_path=None):
    """
    Plot confusion matrices as heatmaps for each subgroup.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        pred_col (str): Column name of the predicted probabilities
        label_col (str): Column name of the true label
        threshold (float): Classification threshold
        save_path (str, optional): Path to save the figure
    """
    set_style()
    
    # Convert predictions to binary
    df[f"{pred_col}_binary"] = (df[pred_col] >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = pd.crosstab(df[label_col], df[f"{pred_col}_binary"], 
                    rownames=["True"], colnames=["Predicted"], normalize="index")
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", cbar=False)
    plt.title(f"Confusion Matrix - {pred_col}")
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_prediction_distribution(df, pred_cols, identity_col=None, bins=50, save_path=None):
    """
    Plot distribution of predictions, optionally split by an identity group.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions
        pred_cols (list): List of column names with model predictions
        identity_col (str, optional): Column name of identity to split by
        bins (int): Number of histogram bins
        save_path (str, optional): Path to save the figure
    """
    set_style()
    
    if identity_col is not None:
        # Plot distribution split by identity group
        plt.figure(figsize=(15, 5 * len(pred_cols)))
        
        for i, pred_col in enumerate(pred_cols):
            plt.subplot(len(pred_cols), 1, i+1)
            
            # Plot for rows where identity_col is 0
            sns.histplot(df[df[identity_col] <= 0][pred_col], 
                        bins=bins, alpha=0.5, label=f"Not {identity_col}")
            
            # Plot for rows where identity_col is > 0
            sns.histplot(df[df[identity_col] > 0][pred_col], 
                        bins=bins, alpha=0.5, label=f"{identity_col}")
            
            plt.title(f"Prediction Distribution - {pred_col} by {identity_col}")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Count")
            plt.legend()
    
    else:
        # Plot distribution for all models
        plt.figure(figsize=(12, 8))
        
        for pred_col in pred_cols:
            sns.histplot(df[pred_col], bins=bins, alpha=0.5, label=pred_col)
        
        plt.title("Prediction Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_subgroup_auc_comparison(model_metrics, identity_cols, 
                                model_names=None, sort_by=None,
                                title="AUC by Demographic Subgroup"):
    """
    Create a bar plot comparing model performance across demographic subgroups.
    
    Args:
        model_metrics: Dictionary with model metrics (output from bias_metrics.compare_models_bias)
        identity_cols: List of identity columns to include
        model_names: Optional dictionary mapping model column names to display names
        sort_by: Optional model name to sort subgroups by its performance
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    # Prepare data for plotting
    plot_data = []
    
    for model_col, metrics in model_metrics.items():
        display_name = model_names[model_col] if model_names else model_col
        
        # Get overall AUC
        overall_auc = metrics['auc']['overall']
        
        # Get subgroup AUCs
        for subgroup in identity_cols:
            if subgroup in metrics['auc']['subgroup']:
                subgroup_auc = metrics['auc']['subgroup'][subgroup]
                if not np.isnan(subgroup_auc):  # Skip NaN values
                    plot_data.append({
                        'Model': display_name,
                        'Subgroup': subgroup,
                        'AUC': subgroup_auc,
                        'Overall AUC': overall_auc,
                        'Difference': subgroup_auc - overall_auc
                    })
    
    # If no valid data, create a simple plot with overall AUC
    if not plot_data:
        print("No valid subgroup data found. Creating overall AUC plot only.")
        
        # Create simple overall AUC plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names_list = []
        auc_values = []
        
        for model_col, metrics in model_metrics.items():
            display_name = model_names[model_col] if model_names else model_col
            overall_auc = metrics['auc']['overall']
            model_names_list.append(display_name)
            auc_values.append(overall_auc)
        
        ax.bar(model_names_list, auc_values)
        ax.set_xlabel('Model')
        ax.set_ylabel('Overall AUC')
        ax.set_title('Overall AUC by Model')
        
        # Set y-axis to start at 0.5 to better show differences
        ax.set_ylim(0.5, 1.0)
        
        for i, v in enumerate(auc_values):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        return fig
    
    df_plot = pd.DataFrame(plot_data)
    
    # Sort subgroups if requested
    if sort_by and sort_by in df_plot['Model'].unique():
        # Get mean AUC per subgroup for the specified model
        sort_model = df_plot[df_plot['Model'] == sort_by]
        sort_order = sort_model.groupby('Subgroup')['AUC'].mean().sort_values().index.tolist()
        df_plot['Subgroup'] = pd.Categorical(df_plot['Subgroup'], categories=sort_order, ordered=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars for each model
    unique_models = df_plot['Model'].unique()
    n_models = len(unique_models)
    width = 0.8 / n_models
    
    for i, model in enumerate(unique_models):
        model_data = df_plot[df_plot['Model'] == model]
        x = np.arange(len(model_data['Subgroup'].unique()))
        offset = (i - n_models / 2 + 0.5) * width
        
        bars = ax.bar(x + offset, model_data['AUC'], width, label=model)
        
        # Add horizontal lines for overall AUC
        overall = model_data['Overall AUC'].iloc[0]
        ax.axhline(y=overall, color=bars[0].get_facecolor(), linestyle='--', alpha=0.5)
    
    # Add labels and legend
    ax.set_xlabel('Demographic Subgroup')
    ax.set_ylabel('AUC')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(df_plot['Subgroup'].unique())))
    ax.set_xticklabels(df_plot['Subgroup'].unique(), rotation=45, ha='right')
    ax.legend()
    
    # Adjust y-axis to highlight differences
    y_min = max(0, df_plot['AUC'].min() - 0.05)
    y_max = min(1, df_plot['AUC'].max() + 0.05)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

def plot_threshold_optimization(threshold_results, model_name=None):
    """
    Plot the results of threshold optimization.
    
    Args:
        threshold_results: Dictionary with threshold optimization results
        model_name: Optional model name for the title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    results_df = threshold_results['all_results']
    best_threshold = threshold_results['best_threshold']
    
    # Create plot with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot FPR and FNR
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate', color='tab:blue')
    ax1.plot(results_df['threshold'], results_df['overall_fpr'], 'o-', label='False Positive Rate', color='tab:blue')
    ax1.plot(results_df['threshold'], results_df['overall_fnr'], 's-', label='False Negative Rate', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Create second y-axis for disparities
    ax2 = ax1.twinx()
    ax2.set_ylabel('Disparity', color='tab:red')
    ax2.plot(results_df['threshold'], results_df['fpr_disparity'], '^-', label='FPR Disparity', color='tab:red')
    ax2.plot(results_df['threshold'], results_df['fnr_disparity'], 'v-', label='FNR Disparity', color='tab:orange')
    ax2.plot(results_df['threshold'], results_df['disparity_sum'], 'D-', label='Total Disparity', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Highlight the best threshold
    ax1.axvline(x=best_threshold, linestyle='--', color='black', alpha=0.5, 
                label=f'Best Threshold = {best_threshold:.2f}')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Set title
    title = "Threshold Optimization"
    if model_name:
        title += f" for {model_name}"
    plt.title(title)
    
    plt.tight_layout()
    return fig

def plot_bias_metrics_heatmap(model_metrics, identity_cols, model_names=None):
    """
    Create a heatmap of bias metrics across models and subgroups.
    
    Args:
        model_metrics: Dictionary with model metrics (output from bias_metrics.compare_models_bias)
        identity_cols: List of identity columns to include
        model_names: Optional dictionary mapping model column names to display names
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    # Prepare data for heatmap
    if not model_metrics:
        # No model metrics available, create a simple placeholder plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No bias metrics available for heatmap", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Extract subgroup AUCs
    heatmap_data = []
    
    for model_col, metrics in model_metrics.items():
        display_name = model_names[model_col] if model_names else model_col.replace('pred_', '')
        
        # Skip models with no subgroup data
        if not metrics['auc']['subgroup']:
            continue
            
        for subgroup, auc in metrics['auc']['subgroup'].items():
            if subgroup in identity_cols:
                # Calculate differences to overall AUC
                overall_auc = metrics['auc']['overall']
                diff = auc - overall_auc
                
                heatmap_data.append({
                    'Model': display_name,
                    'Subgroup': subgroup,
                    'AUC': auc,
                    'Difference': diff
                })
    
    # If no heatmap data (no subgroup metrics), create a simple plot with overall AUC
    if not heatmap_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get overall AUC for each model
        model_names_list = []
        auc_values = []
        
        for model_col, metrics in model_metrics.items():
            display_name = model_names[model_col] if model_names else model_col.replace('pred_', '')
            model_names_list.append(display_name)
            auc_values.append(metrics['auc']['overall'])
        
        # Create a table instead of a heatmap
        data = {'Model': model_names_list, 'Overall AUC': auc_values}
        tab = ax.table(cellText=[[model, f"{auc:.4f}"] for model, auc in zip(model_names_list, auc_values)],
                        colLabels=['Model', 'Overall AUC'],
                        loc='center', cellLoc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(12)
        tab.scale(1.2, 1.5)
        ax.set_title('Model Performance (Overall AUC)', fontsize=14)
        ax.set_axis_off()
        
        return fig
    
    # Convert to DataFrame
    df_heatmap = pd.DataFrame(heatmap_data)
    df_heatmap = df_heatmap.pivot(index='Subgroup', columns='Model', values='Difference')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap=cmap, center=0,
                linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('AUC Difference from Overall (by Subgroup)', fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_roc_curves_by_subgroup(df, pred_col, label_col, identity_cols, model_name=None):
    """
    Plot ROC curves for each demographic subgroup for a single model.
    
    Args:
        df: DataFrame with predictions
        pred_col: Name of prediction column
        label_col: Name of label column
        identity_cols: List of identity column names
        model_name: Optional model name for the title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    from sklearn.metrics import roc_curve, auc
    
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot ROC curve for overall dataset
    fpr, tpr, _ = roc_curve(df[label_col], df[pred_col])
    overall_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, 'k-', lw=2, label=f'Overall (AUC = {overall_auc:.3f})')
    
    # Plot ROC curves for each subgroup
    for subgroup in identity_cols:
        if subgroup in df.columns:
            subgroup_mask = df[subgroup] > 0.5
            if subgroup_mask.sum() > 0:
                fpr, tpr, _ = roc_curve(
                    df[subgroup_mask][label_col], 
                    df[subgroup_mask][pred_col]
                )
                subgroup_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.5, alpha=0.8, 
                        label=f'{subgroup} (AUC = {subgroup_auc:.3f})')
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Set axis labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    title = "ROC Curves by Demographic Subgroup"
    if model_name:
        title += f" for {model_name}"
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    return fig

def save_all_plots(model_metrics, df, identity_cols, output_dir="figures"):
    """
    Generate and save all plots for model comparison.
    
    Args:
        model_metrics: Dictionary with model metrics
        df: DataFrame with predictions
        identity_cols: List of identity column names
        output_dir: Directory to save plots
        
    Returns:
        list: Paths to saved plot files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # 1. Plot subgroup AUC comparison
    model_names = {k: k.replace('pred_', '') for k in model_metrics.keys()}
    fig = plot_subgroup_auc_comparison(model_metrics, identity_cols, model_names)
    filename = os.path.join(output_dir, "subgroup_auc_comparison.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    saved_files.append(filename)
    plt.close(fig)
    
    # 2. Plot bias metrics heatmap
    fig = plot_bias_metrics_heatmap(model_metrics, identity_cols, model_names)
    filename = os.path.join(output_dir, "bias_metrics_heatmap.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    saved_files.append(filename)
    plt.close(fig)
    
    # 3. Plot threshold optimization for each model
    for model_col, metrics in model_metrics.items():
        model_display = model_names[model_col]
        fig = plot_threshold_optimization(
            metrics['threshold_optimization'], model_display)
        filename = os.path.join(output_dir, f"{model_display}_threshold_optimization.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        # 4. Plot ROC curves for each model
        fig = plot_roc_curves_by_subgroup(
            df, model_col, 'target', identity_cols, model_display)
        filename = os.path.join(output_dir, f"{model_display}_roc_curves.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
    
    return saved_files 