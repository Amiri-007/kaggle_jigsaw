#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Bias Metrics Module

This module provides functions for calculating various bias metrics
for evaluating model performance across different demographic subgroups.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_subgroup_auc(df, subgroup, label, pred):
    """
    Calculate the AUC score for a specific demographic subgroup.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        subgroup (str): Column name of the subgroup identity
        label (str): Column name of the true label
        pred (str): Column name of the predicted probabilities
        
    Returns:
        float: AUC score for the subgroup
    """
    subgroup_mask = df[subgroup] > 0
    if sum(subgroup_mask) < 10:  # Skip small subgroups
        return None
    return roc_auc_score(df[subgroup_mask][label], df[subgroup_mask][pred])

def get_background_auc(df, subgroup, label, pred):
    """
    Calculate the AUC score for the background (non-subgroup).
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        subgroup (str): Column name of the subgroup identity
        label (str): Column name of the true label
        pred (str): Column name of the predicted probabilities
        
    Returns:
        float: AUC score for the background
    """
    background_mask = df[subgroup] <= 0
    if sum(background_mask) < 10:  # Skip small backgrounds
        return None
    return roc_auc_score(df[background_mask][label], df[background_mask][pred])

def get_bpsn_auc(df, subgroup, label, pred):
    """
    Calculate the Background Positive, Subgroup Negative (BPSN) AUC.
    
    This measures whether models more frequently produce false positives 
    for the subgroup than the background.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        subgroup (str): Column name of the subgroup identity
        label (str): Column name of the true label
        pred (str): Column name of the predicted probabilities
        
    Returns:
        float: BPSN AUC score
    """
    mask = ((df[subgroup] > 0) & (df[label] == 0)) | ((df[subgroup] <= 0) & (df[label] == 1))
    if sum(mask) < 10:  # Skip if insufficient data
        return None
    return roc_auc_score(df[mask][label], df[mask][pred])

def get_bnsp_auc(df, subgroup, label, pred):
    """
    Calculate the Background Negative, Subgroup Positive (BNSP) AUC.
    
    This measures whether models more frequently produce false negatives 
    for the subgroup than the background.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        subgroup (str): Column name of the subgroup identity
        label (str): Column name of the true label
        pred (str): Column name of the predicted probabilities
        
    Returns:
        float: BNSP AUC score
    """
    mask = ((df[subgroup] <= 0) & (df[label] == 0)) | ((df[subgroup] > 0) & (df[label] == 1))
    if sum(mask) < 10:  # Skip if insufficient data
        return None
    return roc_auc_score(df[mask][label], df[mask][pred])

def compute_bias_metrics(df, identity_columns, label_column="target", 
                        pred_column="pred", threshold=0.5):
    """
    Compute comprehensive bias metrics for all subgroups.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        identity_columns (list): List of column names for identity subgroups
        label_column (str): Column name of the true label
        pred_column (str): Column name of the predicted probabilities
        threshold (float): Classification threshold for binary metrics
        
    Returns:
        dict: Dictionary with bias metrics for each subgroup
    """
    # Convert probability predictions to binary for threshold-based metrics
    df[f"{pred_column}_binary"] = (df[pred_column] >= threshold).astype(int)
    
    # Calculate overall AUC
    overall_auc = roc_auc_score(df[label_column], df[pred_column])
    
    # Calculate overall threshold-based metrics
    overall_accuracy = accuracy_score(df[label_column], df[f"{pred_column}_binary"])
    overall_precision = precision_score(df[label_column], df[f"{pred_column}_binary"])
    overall_recall = recall_score(df[label_column], df[f"{pred_column}_binary"])
    overall_f1 = f1_score(df[label_column], df[f"{pred_column}_binary"])
    
    # Initialize results dictionary
    results = {
        "overall": {
            "auc": overall_auc,
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1
        },
        "subgroup": {},
        "bpsn": {},
        "bnsp": {}
    }
    
    # Calculate subgroup metrics
    for subgroup in identity_columns:
        # Skip if column doesn't exist
        if subgroup not in df.columns:
            continue
            
        # Skip if no positive examples
        if df[subgroup].sum() == 0:
            continue
        
        # Calculate subgroup AUC
        subgroup_auc = get_subgroup_auc(df, subgroup, label_column, pred_column)
        if subgroup_auc is not None:
            results["subgroup"][subgroup] = subgroup_auc
        
        # Calculate BPSN AUC
        bpsn_auc = get_bpsn_auc(df, subgroup, label_column, pred_column)
        if bpsn_auc is not None:
            results["bpsn"][subgroup] = bpsn_auc
            
        # Calculate BNSP AUC
        bnsp_auc = get_bnsp_auc(df, subgroup, label_column, pred_column)
        if bnsp_auc is not None:
            results["bnsp"][subgroup] = bnsp_auc
    
    # Calculate power difference
    identity_df = df[identity_columns]
    non_toxic_subgroups = identity_df[df[label_column] == 0]
    toxic_subgroups = identity_df[df[label_column] == 1]
    non_toxic_subgroup_base_rates = non_toxic_subgroups.mean()
    toxic_subgroup_base_rates = toxic_subgroups.mean()
    power_diff = {}
    for subgroup in identity_columns:
        # Skip if column doesn't exist
        if subgroup not in toxic_subgroup_base_rates or subgroup not in non_toxic_subgroup_base_rates:
            continue
        toxic_rate = toxic_subgroup_base_rates[subgroup]
        non_toxic_rate = non_toxic_subgroup_base_rates[subgroup]
        if non_toxic_rate > 0:
            power_diff[subgroup] = toxic_rate / non_toxic_rate
        else:
            power_diff[subgroup] = None
    
    results["power_diff"] = power_diff
    
    return results

def compare_models(df, identity_columns, model_columns, label_column="target"):
    """
    Compare multiple model predictions across subgroups.
    
    Args:
        df (pandas.DataFrame): DataFrame with predictions and labels
        identity_columns (list): List of column names for identity subgroups
        model_columns (list): List of column names with model predictions
        label_column (str): Column name of the true label
        
    Returns:
        dict: Dictionary with comparative metrics for each model
    """
    results = {}
    
    for model_col in model_columns:
        results[model_col] = compute_bias_metrics(
            df, identity_columns, label_column, model_col
        )
    
    return results

def subgroup_auc(df, subgroup, label, pred):
    """
    Calculate AUC for a specific subgroup.
    
    Args:
        df: DataFrame containing the data
        subgroup: Name of the column identifying the subgroup
        label: Name of the column containing true labels
        pred: Name of the column containing model predictions
        
    Returns:
        float: AUC score for the subgroup
    """
    subgroup_mask = df[subgroup] > 0.5  # Convert to boolean
    if subgroup_mask.sum() == 0:
        return np.nan
    
    return roc_auc_score(df[subgroup_mask][label], df[subgroup_mask][pred])

def calculate_bias_metrics_for_model(df, identity_columns, label_col, pred_col):
    """
    Calculate bias metrics for a model across all identity subgroups.
    
    Args:
        df: DataFrame with predictions and identity columns
        identity_columns: List of column names for identity groups
        label_col: Name of the column with true labels
        pred_col: Name of the column with model predictions
        
    Returns:
        dict: Dictionary with overall and subgroup metrics
    """
    # Calculate overall AUC
    metrics = {
        'overall': roc_auc_score(df[label_col], df[pred_col]),
        'subgroup': {}
    }
    
    # Check if we have any actual identity data
    has_identity_data = False
    for subgroup in identity_columns:
        if subgroup in df.columns and df[subgroup].sum() > 0:
            has_identity_data = True
            break
    
    # Early return with default values if no identity data
    if not has_identity_data:
        print(f"Warning: No demographic data found in the dataset for {pred_col}.")
        print("Using only overall metrics.")
        metrics['bias_metrics'] = {
            'min_subgroup_auc': metrics['overall'],
            'max_subgroup_auc': metrics['overall'],
            'auc_variance': 0.0,
            'auc_difference': 0.0
        }
        return metrics
    
    # Calculate AUC for each identity subgroup
    valid_aucs = []
    for subgroup in identity_columns:
        if subgroup in df.columns:
            # Only calculate if we have enough samples
            subgroup_mask = df[subgroup] > 0.5
            if subgroup_mask.sum() >= 100:  # Require at least 100 samples
                subgroup_auc_value = subgroup_auc(df, subgroup, label_col, pred_col)
                metrics['subgroup'][subgroup] = subgroup_auc_value
                if not np.isnan(subgroup_auc_value):
                    valid_aucs.append(subgroup_auc_value)
    
    # Calculate bias metrics from valid values
    if valid_aucs:
        auc_values = np.array(valid_aucs)
        metrics['bias_metrics'] = {
            'min_subgroup_auc': np.min(auc_values),
            'max_subgroup_auc': np.max(auc_values),
            'auc_variance': np.var(auc_values) if len(auc_values) > 1 else 0.0,
            'auc_difference': np.max(auc_values) - np.min(auc_values)
        }
    else:
        # If no valid AUCs, use overall AUC as fallback
        metrics['bias_metrics'] = {
            'min_subgroup_auc': metrics['overall'],
            'max_subgroup_auc': metrics['overall'],
            'auc_variance': 0.0,
            'auc_difference': 0.0
        }
    
    return metrics

def threshold_optimization(df, pred_col, label_col, identity_cols, 
                          start=0.1, end=0.9, step=0.05):
    """
    Find optimal thresholds that minimize bias across subgroups.
    
    Args:
        df: DataFrame with predictions
        pred_col: Name of prediction column
        label_col: Name of label column
        identity_cols: List of identity column names
        start, end, step: Range for threshold search
        
    Returns:
        dict: Best threshold and corresponding metrics
    """
    thresholds = np.arange(start, end, step)
    results = []
    
    for threshold in thresholds:
        # Create binary predictions using this threshold
        binary_preds = (df[pred_col] >= threshold).astype(int)
        
        # Calculate overall metrics
        tn, fp, fn, tp = confusion_matrix(df[label_col], binary_preds).ravel()
        overall_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        overall_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Calculate subgroup metrics
        subgroup_fprs = {}
        subgroup_fnrs = {}
        
        for subgroup in identity_cols:
            if subgroup in df.columns:
                subgroup_mask = df[subgroup] > 0.5
                if subgroup_mask.sum() > 0:
                    sub_tn, sub_fp, sub_fn, sub_tp = confusion_matrix(
                        df[subgroup_mask][label_col], 
                        binary_preds[subgroup_mask.values]
                    ).ravel()
                    
                    subgroup_fpr = sub_fp / (sub_fp + sub_tn) if (sub_fp + sub_tn) > 0 else 0
                    subgroup_fnr = sub_fn / (sub_fn + sub_tp) if (sub_fn + sub_tp) > 0 else 0
                    
                    subgroup_fprs[subgroup] = subgroup_fpr
                    subgroup_fnrs[subgroup] = subgroup_fnr
        
        # Calculate max disparities
        if subgroup_fprs:
            fpr_disparity = max(subgroup_fprs.values()) - min(subgroup_fprs.values())
        else:
            fpr_disparity = 0
            
        if subgroup_fnrs:
            fnr_disparity = max(subgroup_fnrs.values()) - min(subgroup_fnrs.values())
        else:
            fnr_disparity = 0
        
        # Calculate combined metric (lower is better)
        disparity_sum = fpr_disparity + fnr_disparity
        
        results.append({
            'threshold': threshold,
            'overall_fpr': overall_fpr,
            'overall_fnr': overall_fnr,
            'fpr_disparity': fpr_disparity,
            'fnr_disparity': fnr_disparity,
            'disparity_sum': disparity_sum
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find threshold with minimum disparity sum
    best_idx = results_df['disparity_sum'].idxmin()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_metrics = results_df.loc[best_idx].to_dict()
    
    return {
        'best_threshold': best_threshold,
        'metrics': best_metrics,
        'all_results': results_df
    }

def compare_models_bias(df, model_pred_cols, label_col, identity_cols):
    """
    Compare multiple models based on their bias metrics.
    
    Args:
        df: DataFrame with predictions from multiple models
        model_pred_cols: List of column names with model predictions
        label_col: Name of label column
        identity_cols: List of identity column names
        
    Returns:
        dict: Comparative metrics for all models
    """
    comparison = {}
    
    for model_col in model_pred_cols:
        # Calculate bias metrics for this model
        model_metrics = calculate_bias_metrics_for_model(
            df, identity_cols, label_col, model_col
        )
        
        # Find optimal threshold
        threshold_results = threshold_optimization(
            df, model_col, label_col, identity_cols
        )
        
        # Store results
        comparison[model_col] = {
            'auc': model_metrics,
            'threshold_optimization': threshold_results
        }
    
    return comparison 