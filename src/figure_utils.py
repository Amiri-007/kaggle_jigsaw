#!/usr/bin/env python
"""
Fairness figure utilities for the RDS project.

This module provides functions to generate publication-ready fairness figures
for model evaluation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import roc_curve, auc

# Ensure the figs directory exists
os.makedirs('figs', exist_ok=True)


def identity_prevalence(df_train: pd.DataFrame, identity_cols: list) -> plt.Figure:
    """
    Create a bar plot showing the prevalence of each identity group in the dataset.
    
    Args:
        df_train: Training dataframe with identity columns
        identity_cols: List of identity column names
        
    Returns:
        matplotlib Figure object
    """
    # Calculate prevalence of each identity
    prevalence = {col: df_train[col].mean() for col in identity_cols}
    
    # Sort by prevalence
    prevalence = {k: v for k, v in sorted(prevalence.items(), key=lambda item: item[1], reverse=True)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = ax.bar(list(prevalence.keys()), list(prevalence.values()))
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax.set_xlabel('Identity Group')
    ax.set_ylabel('Prevalence')
    ax.set_title('Prevalence of Identity Groups in Training Data')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figs/identity_prevalence.png', dpi=300, bbox_inches='tight')
    
    return fig


def roc_curve_figure(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    """
    Create a ROC curve plot for the model.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic - {model_name}')
    ax.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/overall_roc_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def fairness_heatmap(metrics_df: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Create a heatmap visualizing fairness metrics across demographic groups.
    
    Args:
        metrics_df: DataFrame containing fairness metrics for each identity group
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Pivot the data to create a heatmap
    # Assuming metrics_df has columns: identity_group, metric_name, value
    pivot_df = metrics_df.pivot_table(index='identity_group', columns='metric_name', values='value')
    
    # Sort identities by subgroup_auc if available (to match grouped bar chart order)
    if 'subgroup_auc' in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by='subgroup_auc', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with adjusted parameters
    heatmap = sns.heatmap(pivot_df, annot=True, cmap='RdYlGn_r', 
                  vmin=0.5, vmax=1.0, fmt='.3f', ax=ax)
    
    # Add colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('AUC')
    
    # Set title
    ax.set_title(f'Fairness Metrics Across Demographic Groups - {model_name}')
    
    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=30, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/fairness_heatmap_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def power_mean_bar(metrics_df: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Create a bar plot showing the power mean differences across demographic groups.
    
    Args:
        metrics_df: DataFrame containing fairness metrics for each identity group
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Filter for power_diff metric if available, otherwise use a suitable alternative
    if 'power_diff' in metrics_df['metric_name'].unique():
        power_df = metrics_df[metrics_df['metric_name'] == 'power_diff']
    else:
        # Use another fairness metric as fallback
        available_metrics = metrics_df['metric_name'].unique()
        fallback_metric = [m for m in ['bias_score', 'disparate_impact', 'bnsp_auc'] if m in available_metrics][0]
        power_df = metrics_df[metrics_df['metric_name'] == fallback_metric]
    
    # Sort by value
    power_df = power_df.sort_values(by='value', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = ax.bar(power_df['identity_group'], power_df['value'])
    
    # Add horizontal line at ideal value (1.0 for power_diff)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    metric_name = power_df['metric_name'].iloc[0]
    ax.set_xlabel('Identity Group')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name.replace("_", " ").title()} by Identity Group - {model_name}')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/power_mean_bar_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def threshold_sweep(y_true: np.ndarray, y_pred: np.ndarray, identity_cols: List[np.ndarray], 
                    model_name: str) -> plt.Figure:
    """
    Create a plot showing how fairness metrics change with different thresholds.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        identity_cols: List of binary arrays indicating membership in each identity group
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    from src.threshold_utils import sweep_thresholds
    
    # Create a DataFrame for the identity masks
    identity_masks = {}
    
    # For each identity column, create a mask
    for i, identity_array in enumerate(identity_cols):
        identity_masks[f"identity_{i}"] = identity_array
    
    # Run threshold sweep for each identity group
    results = []
    
    for identity_name, identity_mask in identity_masks.items():
        result_df = sweep_thresholds(y_true, y_pred, identity_mask)
        result_df['identity_group'] = identity_name
        results.append(result_df)
    
    # Combine results
    combined_results = pd.concat(results)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot FPR gap by threshold
    sns.lineplot(data=combined_results, x='threshold', y='fpr_gap', 
                 hue='identity_group', ax=axes[0])
    axes[0].set_title('FPR Gap by Threshold')
    axes[0].set_xlabel('Classification Threshold')
    axes[0].set_ylabel('False Positive Rate Gap')
    
    # Plot FNR gap by threshold
    sns.lineplot(data=combined_results, x='threshold', y='fnr_gap', 
                 hue='identity_group', ax=axes[1])
    axes[1].set_title('FNR Gap by Threshold')
    axes[1].set_xlabel('Classification Threshold')
    axes[1].set_ylabel('False Negative Rate Gap')
    
    # Add overall title
    plt.suptitle(f'Threshold Sweep Analysis - {model_name}')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(f'figs/threshold_sweep_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def worst_k_table(metrics_df: pd.DataFrame, k: int = 5, model_name: str) -> plt.Figure:
    """
    Create a table figure showing the k worst-performing identity groups for each metric.
    
    Args:
        metrics_df: DataFrame containing fairness metrics for each identity group
        k: Number of worst-performing groups to display
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Get unique metrics
    metrics = metrics_df['metric_name'].unique()
    
    # Create a dictionary to store worst performers
    worst_performers = {}
    
    # Find worst performers for each metric
    for metric in metrics:
        metric_df = metrics_df[metrics_df['metric_name'] == metric]
        
        # Sort based on whether higher or lower is better
        # For AUC-based metrics, higher is better, so sort ascending
        # For gap/difference metrics, lower is better, so sort descending
        if any(substr in metric for substr in ['auc', 'precision', 'recall', 'f1']):
            sorted_df = metric_df.sort_values(by='value', ascending=True)
        else:
            sorted_df = metric_df.sort_values(by='value', ascending=False)
        
        # Get k worst performers
        worst_performers[metric] = sorted_df.head(k)
    
    # Create a figure large enough to hold all tables
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, n_metrics * 3))
    
    # If only one metric, axes won't be an array
    if n_metrics == 1:
        axes = [axes]
    
    # Add each table to the figure
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create table data
        worst_df = worst_performers[metric]
        table_data = [
            worst_df['identity_group'].values,
            worst_df['value'].values
        ]
        
        # Create table
        table = ax.table(
            cellText=[[f"{val:.4f}" for val in table_data[1]]],
            rowLabels=[metric],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center'
        )
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Remove axis ticks
        ax.axis('off')
    
    # Add title
    plt.suptitle(f'Top {k} Worst Performing Identity Groups by Metric - {model_name}')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the figure
    plt.savefig(f'figs/worst_k_table_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def before_vs_after_scatter(baseline_df: pd.DataFrame, improved_df: pd.DataFrame, 
                           identity_cols: list) -> plt.Figure:
    """
    Create a scatter plot comparing fairness metrics before and after model improvements.
    
    Args:
        baseline_df: DataFrame containing baseline model metrics
        improved_df: DataFrame containing improved model metrics
        identity_cols: List of identity column names
        
    Returns:
        matplotlib Figure object
    """
    # Create a DataFrame to store the comparison
    comparison_df = pd.DataFrame(index=identity_cols)
    
    # Extract metrics for baseline and improved models
    # Assuming both DataFrames have columns 'identity_group', 'metric_name', 'value'
    
    # Get a metric that exists in both DataFrames
    common_metrics = set(baseline_df['metric_name'].unique()) & set(improved_df['metric_name'].unique())
    selected_metric = list(common_metrics)[0]  # Choose the first common metric
    
    # Filter both DataFrames for the selected metric and identity groups
    baseline_filtered = baseline_df[
        (baseline_df['metric_name'] == selected_metric) & 
        (baseline_df['identity_group'].isin(identity_cols))
    ]
    
    improved_filtered = improved_df[
        (improved_df['metric_name'] == selected_metric) & 
        (improved_df['identity_group'].isin(identity_cols))
    ]
    
    # Set up the comparison DataFrame
    comparison_df['baseline'] = baseline_filtered.set_index('identity_group')['value']
    comparison_df['improved'] = improved_filtered.set_index('identity_group')['value']
    comparison_df['difference'] = comparison_df['improved'] - comparison_df['baseline']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create scatter plot
    scatter = ax.scatter(
        comparison_df['baseline'], 
        comparison_df['improved'],
        s=100,
        alpha=0.7
    )
    
    # Add diagonal line (no change)
    min_val = min(comparison_df['baseline'].min(), comparison_df['improved'].min())
    max_val = max(comparison_df['baseline'].max(), comparison_df['improved'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add labels for each point
    for i, identity in enumerate(comparison_df.index):
        ax.annotate(
            identity,
            (comparison_df['baseline'].iloc[i], comparison_df['improved'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Set axis labels and title
    ax.set_xlabel('Baseline Model')
    ax.set_ylabel('Improved Model')
    ax.set_title(f'Comparison of {selected_metric} Between Baseline and Improved Models')
    
    # Add label indicating which quadrant is better
    # For AUC-type metrics, top-left is better
    # For gap/difference metrics, bottom-left is better
    if any(substr in selected_metric for substr in ['auc', 'precision', 'recall', 'f1']):
        better_quadrant = "Top-left: Improved model better"
    else:
        better_quadrant = "Bottom-left: Improved model better"
    
    ax.text(0.05, 0.95, better_quadrant, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('figs/before_vs_after_scatter.png', dpi=300, bbox_inches='tight')
    
    return fig


def grouped_bar_by_identity(metrics_df: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Create a grouped bar chart comparing different metrics across identity groups.
    
    Args:
        metrics_df: DataFrame containing fairness metrics for each identity group
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Check if subgroup_auc is available for sorting
    if 'subgroup_auc' in metrics_df['metric_name'].unique():
        # Calculate mean subgroup AUC for each identity group for sorting
        subgroup_auc = metrics_df[metrics_df['metric_name'] == 'subgroup_auc']
        sorted_identities = subgroup_auc.sort_values('value', ascending=True)['identity_group'].tolist()
        
        # Create pivot table for plotting
        pivot_df = metrics_df.pivot_table(index='identity_group', columns='metric_name', values='value')
        
        # Sort by the previously determined order
        pivot_df = pivot_df.reindex(sorted_identities)
    else:
        # Create pivot table without sorting
        pivot_df = metrics_df.pivot_table(index='identity_group', columns='metric_name', values='value')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot grouped bars
    pivot_df.plot(kind='bar', ax=ax)
    
    # Set title and labels
    ax.set_title(f'Fairness Metrics Across Demographic Groups - {model_name}')
    ax.set_xlabel('Identity Group')
    ax.set_ylabel('Metric Value')
    
    # Rotate x-axis labels to 60 degrees as requested
    plt.xticks(rotation=60, ha='right')
    
    # Add legend
    ax.legend(title='Metric')
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/grouped_bar_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def worst_k_bar(metrics_df: pd.DataFrame, model_name: str, k: int = 5) -> plt.Figure:
    """
    Create a horizontal bar chart showing the k worst-performing identity groups based on subgroup AUC.
    
    Args:
        metrics_df: DataFrame containing fairness metrics for each identity group
        model_name: Name of the model
        k: Number of worst-performing groups to display
        
    Returns:
        matplotlib Figure object
    """
    # Filter for subgroup_auc metric if available
    if 'subgroup_auc' in metrics_df['metric_name'].unique():
        auc_df = metrics_df[metrics_df['metric_name'] == 'subgroup_auc']
    else:
        # Use another AUC metric as fallback
        available_metrics = metrics_df['metric_name'].unique()
        fallback_metrics = [m for m in ['bpsn_auc', 'bnsp_auc', 'overall_auc'] if m in available_metrics]
        if not fallback_metrics:
            raise ValueError("No AUC metric found in metrics dataframe")
        auc_df = metrics_df[metrics_df['metric_name'] == fallback_metrics[0]]
    
    # Sort by value ascending (lower AUC is worse) and take the k worst performers
    sorted_df = auc_df.sort_values(by='value', ascending=True).head(k)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(5, k * 0.6)))
    
    # Create horizontal bar chart
    bars = ax.barh(sorted_df['identity_group'], sorted_df['value'])
    
    # Add a vertical line at AUC = 0.8 for reference
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center')
    
    # Set axis limits, labels, and title
    ax.set_xlim(0.5, 1.0)  # AUC range from 0.5 to 1.0
    ax.set_xlabel('Subgroup AUC')
    ax.set_ylabel('Identity Group')
    ax.set_title(f'Worst {k} Performing Demographic Groups - {model_name}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/worst_k_bar_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def add_confusion_mosaic(y_true: np.ndarray, y_pred: np.ndarray, 
                        identity_mask: np.ndarray, identity_name: str, 
                        model_name: str) -> plt.Figure:
    """
    Create a mosaic plot of the confusion matrix for a specific identity group.
    
    Args:
        y_true: True binary labels
        y_pred: Binary predictions (already thresholded)
        identity_mask: Binary mask indicating membership in the identity group
        identity_name: Name of the identity group
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Calculate confusion matrix for the specified identity group
    y_true_subgroup = y_true[identity_mask]
    y_pred_subgroup = y_pred[identity_mask]
    
    # Create confusion matrix
    cm = pd.crosstab(
        y_true_subgroup, 
        y_pred_subgroup, 
        rownames=['True'], 
        colnames=['Predicted'],
        normalize='all'  # Normalize to show proportions
    )
    
    # Try to use statsmodels for mosaic plot if available
    try:
        from statsmodels.graphics.mosaicplot import mosaic
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Convert confusion matrix to dict for mosaic
        conf_dict = {
            ('True 0', 'Predicted 0'): cm.iloc[0, 0],  # True Negative
            ('True 0', 'Predicted 1'): cm.iloc[0, 1],  # False Positive
            ('True 1', 'Predicted 0'): cm.iloc[1, 0],  # False Negative
            ('True 1', 'Predicted 1'): cm.iloc[1, 1],  # True Positive
        }
        
        # Create mosaic plot
        mosaic(conf_dict, ax=ax, title=f'Confusion Matrix Mosaic - {identity_name} - {model_name}',
               gap=0.01, properties=lambda key: {
                   'color': 'green' if ('True 1' in key and 'Predicted 1' in key) or 
                                      ('True 0' in key and 'Predicted 0' in key) else 'red'
               })
        
    except ImportError:
        # Fallback to seaborn heatmap if statsmodels is not available
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create heatmap of confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn',
            ax=ax,
            cbar=False
        )
        
        # Add labels
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {identity_name} - {model_name}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/confusion_{identity_name}_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def threshold_gap_curve(df_sweep: pd.DataFrame, identity_name: str, model_name: str) -> plt.Figure:
    """
    Create a plot showing how FPR and FNR gaps change with different classification thresholds.
    
    Args:
        df_sweep: DataFrame from sweep_threshold_gaps containing threshold, fpr_gap, and fnr_gap
        identity_name: Name of the identity group
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot absolute FPR gap
    ax.plot(df_sweep['threshold'], df_sweep['abs_fpr_gap'], 
           label='|FPR Gap|', color='blue', linewidth=2)
    
    # Plot absolute FNR gap
    ax.plot(df_sweep['threshold'], df_sweep['abs_fnr_gap'], 
           label='|FNR Gap|', color='red', linewidth=2)
    
    # Plot mean gap
    ax.plot(df_sweep['threshold'], df_sweep['mean_gap'], 
           label='Mean Gap', color='purple', linestyle='--', linewidth=1.5)
    
    # Add vertical line at threshold = 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Absolute Rate Gap')
    ax.set_title(f'Error Rate Gaps vs. Threshold - {identity_name} - {model_name}')
    
    # Add legend
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/threshold_gap_curve_{identity_name}_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def error_gap_heatmap(model_name: str, gaps_dict: Dict[str, Tuple[float, float]], 
                     threshold: float = 0.5) -> plt.Figure:
    """
    Create a heatmap visualizing FPR and FNR gaps across demographic groups.
    
    Args:
        model_name: Name of the model
        gaps_dict: Dictionary mapping identity names to (fpr_gap, fnr_gap) tuples
        threshold: Classification threshold used to compute the gaps
        
    Returns:
        matplotlib Figure object
    """
    # Create dataframe from gaps dictionary
    data = []
    for identity, (fpr_gap, fnr_gap) in gaps_dict.items():
        data.append({
            'identity_group': identity,
            'fpr_gap': fpr_gap,
            'fnr_gap': fnr_gap,
            'max_gap': max(abs(fpr_gap), abs(fnr_gap))
        })
    
    gap_df = pd.DataFrame(data)
    
    # Sort by maximum gap (descending) to highlight the most problematic groups
    gap_df = gap_df.sort_values('max_gap', ascending=False)
    
    # Create pivot data for heatmap
    pivot_df = gap_df.set_index('identity_group')[['fpr_gap', 'fnr_gap']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(gaps_dict) * 0.5)))
    
    # Create heatmap
    heatmap = sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt='.3f', 
        cmap='RdBu_r',  # Red-Blue diverging colormap
        center=0,       # Center colormap at 0
        vmin=-0.3,      # Min gap value
        vmax=0.3,       # Max gap value
        cbar_kws={'label': 'Gap (Subgroup Rate - Non-subgroup Rate)'}
    )
    
    # Set title
    plt.title(f'Error Rate Gaps at Threshold {threshold} - {model_name}')
    
    # Add x-axis label explaining what positive/negative gaps mean
    ax.set_xlabel('Positive gap = Higher error rate for subgroup')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'figs/error_gap_heatmap_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig 