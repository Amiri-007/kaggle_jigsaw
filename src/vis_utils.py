#!/usr/bin/env python3
"""
Visualization utilities for bias metrics.

This module provides functions for creating interactive visualizations
of bias metrics across demographic subgroups.
"""

from typing import Dict, List, Tuple, Union, Optional
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_auc_heatmap(
    metrics_csv_path: str, 
    title: str = "Bias Metrics Heatmap",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive heatmap of AUC metrics for different identities.
    
    Args:
        metrics_csv_path: Path to the CSV file with metrics data
                         (should have columns for subgroup, bias metrics, and sample sizes)
        title: Title for the heatmap
        save_path: Optional path to save the figure (SVG format)
        
    Returns:
        Plotly figure object with the heatmap
    """
    # Load metrics data
    df = pd.read_csv(metrics_csv_path)
    
    # Validate required columns exist
    required_cols = ['subgroup_name', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc', 'subgroup_size']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metrics CSV: {missing_cols}")
    
    # Create a pivot table for the heatmap
    metrics = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    
    # Format AUC values as 2-decimal percentages
    for metric in metrics:
        df[f"{metric}_fmt"] = df[metric].apply(lambda x: f"{x:.2%}")
    
    # Create the heatmap data
    z_data = df[metrics].values
    
    # Create text for hover with AUC and sample size
    hover_text = []
    for i in range(len(df)):
        row_text = []
        for metric in metrics:
            row_text.append(
                f"Subgroup: {df.iloc[i]['subgroup_name']}<br>"
                f"Metric: {metric}<br>"
                f"AUC: {df.iloc[i][metric]:.4f}<br>"
                f"Sample size: {int(df.iloc[i]['subgroup_size'])}"
            )
        hover_text.append(row_text)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=["Subgroup AUC", "BPSN AUC", "BNSP AUC"],
        y=df['subgroup_name'],
        hoverinfo="text",
        text=hover_text,
        colorscale="RdBu",  # Red-Blue diverging colorscale
        zmin=0.5,           # AUC range from 0.5 (random) to 1.0 (perfect)
        zmax=1.0,
        colorbar=dict(
            title="AUC",
            titleside="right"
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Metric Type",
        yaxis_title="Identity Subgroup",
        margin=dict(l=120, r=20, t=70, b=50),
        height=500 + 20 * len(df),  # Adjust height based on number of subgroups
        width=800
    )
    
    # Save figure if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Saved heatmap to {save_path}")
    
    return fig


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_mask: np.ndarray,
    identity_name: str,
    thresholds: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive plot showing FPR and FNR gaps across threshold values.
    
    This visualization helps identify optimal thresholds that minimize
    the maximum disparity between subgroup and background.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities
        subgroup_mask: Boolean mask identifying members of the subgroup
        identity_name: Name of the identity subgroup (for display)
        thresholds: Optional array of thresholds to use (default: 0 to 1 by 0.01)
        save_path: Optional path to save the figure (SVG format)
        
    Returns:
        Plotly figure with FPR and FNR gap curves
    """
    if thresholds is None:
        thresholds = np.arange(0, 1.01, 0.01)
    
    background_mask = ~subgroup_mask
    
    # Initialize arrays to store FPR and FNR for both subgroup and background
    subgroup_fpr = np.zeros_like(thresholds)
    background_fpr = np.zeros_like(thresholds)
    subgroup_fnr = np.zeros_like(thresholds)
    background_fnr = np.zeros_like(thresholds)
    
    # For each threshold, calculate confusion matrix metrics
    for i, threshold in enumerate(thresholds):
        # Subgroup predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Subgroup metrics
        if np.sum(subgroup_mask) > 0:
            subgroup_y_true = y_true[subgroup_mask]
            subgroup_y_pred = y_pred_binary[subgroup_mask]
            
            # True positives, false positives, true negatives, false negatives
            subgroup_tp = np.sum((subgroup_y_true == 1) & (subgroup_y_pred == 1))
            subgroup_fp = np.sum((subgroup_y_true == 0) & (subgroup_y_pred == 1))
            subgroup_tn = np.sum((subgroup_y_true == 0) & (subgroup_y_pred == 0))
            subgroup_fn = np.sum((subgroup_y_true == 1) & (subgroup_y_pred == 0))
            
            # Calculate rates (with handling for division by zero)
            if (subgroup_fp + subgroup_tn) > 0:
                subgroup_fpr[i] = subgroup_fp / (subgroup_fp + subgroup_tn)
            else:
                subgroup_fpr[i] = 0
                
            if (subgroup_fn + subgroup_tp) > 0:
                subgroup_fnr[i] = subgroup_fn / (subgroup_fn + subgroup_tp)
            else:
                subgroup_fnr[i] = 0
        
        # Background metrics
        if np.sum(background_mask) > 0:
            background_y_true = y_true[background_mask]
            background_y_pred = y_pred_binary[background_mask]
            
            # True positives, false positives, true negatives, false negatives
            background_tp = np.sum((background_y_true == 1) & (background_y_pred == 1))
            background_fp = np.sum((background_y_true == 0) & (background_y_pred == 1))
            background_tn = np.sum((background_y_true == 0) & (background_y_pred == 0))
            background_fn = np.sum((background_y_true == 1) & (background_y_pred == 0))
            
            # Calculate rates (with handling for division by zero)
            if (background_fp + background_tn) > 0:
                background_fpr[i] = background_fp / (background_fp + background_tn)
            else:
                background_fpr[i] = 0
                
            if (background_fn + background_tp) > 0:
                background_fnr[i] = background_fn / (background_fn + background_tp)
            else:
                background_fnr[i] = 0
    
    # Calculate gaps (absolute differences)
    fpr_gap = np.abs(subgroup_fpr - background_fpr)
    fnr_gap = np.abs(subgroup_fnr - background_fnr)
    
    # Find index of minimum max gap
    max_gap = np.maximum(fpr_gap, fnr_gap)
    min_max_gap_idx = np.argmin(max_gap)
    optimal_threshold = thresholds[min_max_gap_idx]
    min_max_gap = max_gap[min_max_gap_idx]
    
    # Create figure with two subplots (FPR and FNR)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"False Positive Rate (Subgroup vs Background)",
            f"False Negative Rate (Subgroup vs Background)"
        ),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Add FPR traces
    fig.add_trace(
        go.Scatter(
            x=thresholds, 
            y=subgroup_fpr,
            mode='lines',
            name=f"{identity_name} FPR",
            line=dict(color='rgb(31, 119, 180)'),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds, 
            y=background_fpr,
            mode='lines',
            name="Background FPR",
            line=dict(color='rgb(255, 127, 14)'),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds, 
            y=fpr_gap,
            mode='lines',
            name="FPR Gap (abs)",
            line=dict(color='rgb(44, 160, 44)', dash='dash'),
        ),
        row=1, col=1
    )
    
    # Add FNR traces
    fig.add_trace(
        go.Scatter(
            x=thresholds, 
            y=subgroup_fnr,
            mode='lines',
            name=f"{identity_name} FNR",
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds, 
            y=background_fnr,
            mode='lines',
            name="Background FNR",
            line=dict(color='rgb(255, 127, 14)'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=thresholds, 
            y=fnr_gap,
            mode='lines',
            name="FNR Gap (abs)",
            line=dict(color='rgb(214, 39, 40)', dash='dash'),
        ),
        row=2, col=1
    )
    
    # Add vertical line at optimal threshold
    fig.add_vline(
        x=optimal_threshold,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal threshold: {optimal_threshold:.2f}<br>Max gap: {min_max_gap:.3f}",
        annotation_position="top right",
    )
    
    # Update layout
    fig.update_layout(
        title=f"Threshold Analysis for {identity_name}",
        xaxis_title="Threshold",
        yaxis_title="False Positive Rate",
        xaxis2_title="Threshold",
        yaxis2_title="False Negative Rate",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        width=1000
    )
    
    # Save figure if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Saved threshold sweep plot to {save_path}")
    
    return fig


def plot_fairness_radar(
    metrics_df: pd.DataFrame,
    model_names: List[str],
    metric_columns: List[str] = None,
    title: str = "Fairness Metrics Comparison",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create a radar chart comparing fairness metrics across multiple models.
    
    Args:
        metrics_df: DataFrame with fairness metrics for different models
        model_names: List of model names to include in the comparison
        metric_columns: List of metric columns to display (default: use power means)
        title: Title for the radar chart
        save_path: Optional path to save the figure
        
    Returns:
        Plotly figure with the radar chart
    """
    if metric_columns is None:
        metric_columns = [
            'power_mean_subgroup_auc', 
            'power_mean_bpsn_auc', 
            'power_mean_bnsp_auc',
            'overall_auc',
            'final_score'
        ]
    
    # Filter dataframe to only include specified models and metrics
    plot_df = metrics_df[metrics_df['model_name'].isin(model_names)]
    
    # Create radar chart
    fig = go.Figure()
    
    for model in model_names:
        if model not in plot_df['model_name'].values:
            continue
            
        model_data = plot_df[plot_df['model_name'] == model]
        
        fig.add_trace(go.Scatterpolar(
            r=[model_data[col].values[0] for col in metric_columns],
            theta=metric_columns,
            fill='toself',
            name=model
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.5, 1]  # AUC values range from 0.5 to 1
            )
        ),
        title=title,
        showlegend=True
    )
    
    # Save figure if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Saved radar chart to {save_path}")
    
    return fig


def plot_power_mean_bars(
    metrics_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create a bar chart showing power mean metrics for a single model.
    
    Args:
        metrics_df: DataFrame with fairness metrics
        model_name: Name of model to visualize
        save_path: Optional path to save the figure
        
    Returns:
        Plotly figure with bar chart
    """
    # Filter dataframe for the specified model
    model_data = metrics_df[metrics_df['model_name'] == model_name]
    
    if len(model_data) == 0:
        raise ValueError(f"Model '{model_name}' not found in metrics dataframe")
    
    # Get power mean metrics
    metrics = [
        'power_mean_subgroup_auc', 
        'power_mean_bpsn_auc', 
        'power_mean_bnsp_auc',
        'overall_auc'
    ]
    
    metric_names = [
        'Subgroup AUC (Power Mean)', 
        'BPSN AUC (Power Mean)', 
        'BNSP AUC (Power Mean)',
        'Overall AUC'
    ]
    
    values = [model_data[metric].values[0] for metric in metrics]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=values,
            marker_color=['rgba(31, 119, 180, 0.8)', 
                         'rgba(44, 160, 44, 0.8)', 
                         'rgba(214, 39, 40, 0.8)',
                         'rgba(148, 103, 189, 0.8)'],
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        )
    ])
    
    # Add final score as a line
    final_score = model_data['final_score'].values[0]
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=final_score,
        x1=3.5,
        y1=final_score,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=3.5,
        y=final_score,
        text=f"Final Score: {final_score:.4f}",
        showarrow=False,
        yshift=10,
        xshift=5,
        align="left"
    )
    
    # Update layout
    fig.update_layout(
        title=f"Fairness Metrics for {model_name}",
        xaxis_title="Metric",
        yaxis_title="AUC Score",
        yaxis=dict(range=[0.5, 1.0]),  # AUC values range from 0.5 to 1
        template="plotly_white"
    )
    
    # Save figure if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Saved bar chart to {save_path}")
    
    return fig 