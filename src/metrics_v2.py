#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Enhanced Bias Metrics Module (v2)

This module provides vectorized implementations of bias metrics for evaluating
model performance across different demographic subgroups.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from sklearn.metrics import roc_auc_score


def subgroup_auc(y_true: np.ndarray, y_pred: np.ndarray, 
                 subgroup_mask: np.ndarray) -> float:
    """
    Calculate AUC for a specific demographic subgroup in vectorized form.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities
        subgroup_mask: Boolean mask identifying members of the subgroup
        
    Returns:
        AUC score for the subgroup or np.nan if insufficient samples
    """
    # Skip computation for small subgroups to avoid statistical issues
    if np.sum(subgroup_mask) < 10:
        return np.nan
    
    # Calculate AUC only on the subgroup
    try:
        return roc_auc_score(y_true[subgroup_mask], y_pred[subgroup_mask])
    except ValueError:  # Handles case with only one class
        return np.nan


def bpsn_auc(y_true: np.ndarray, y_pred: np.ndarray, 
             subgroup_mask: np.ndarray) -> float:
    """
    Calculate Background Positive, Subgroup Negative (BPSN) AUC in vectorized form.
    
    This identifies disparities where models more frequently produce false positives
    for the subgroup compared to the background population.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities
        subgroup_mask: Boolean mask identifying members of the subgroup
        
    Returns:
        BPSN AUC score or np.nan if insufficient samples
    """
    # BPSN: Background positive (y=1), Subgroup negative (y=0)
    bpsn_mask = ((subgroup_mask) & (y_true == 0)) | ((~subgroup_mask) & (y_true == 1))
    
    # Skip computation for small samples
    if np.sum(bpsn_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[bpsn_mask], y_pred[bpsn_mask])
    except ValueError:  # Handles case with only one class
        return np.nan


def bnsp_auc(y_true: np.ndarray, y_pred: np.ndarray, 
             subgroup_mask: np.ndarray) -> float:
    """
    Calculate Background Negative, Subgroup Positive (BNSP) AUC in vectorized form.
    
    This identifies disparities where models more frequently produce false negatives 
    for the subgroup compared to the background population.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities
        subgroup_mask: Boolean mask identifying members of the subgroup
        
    Returns:
        BNSP AUC score or np.nan if insufficient samples
    """
    # BNSP: Background negative (y=0), Subgroup positive (y=1)
    bnsp_mask = ((~subgroup_mask) & (y_true == 0)) | ((subgroup_mask) & (y_true == 1))
    
    # Skip computation for small samples
    if np.sum(bnsp_mask) < 10:
        return np.nan
    
    try:
        return roc_auc_score(y_true[bnsp_mask], y_pred[bnsp_mask])
    except ValueError:  # Handles case with only one class
        return np.nan


def generalised_power_mean(auc_list: List[float], p: float = -5) -> float:
    """
    Calculate the generalized power mean of AUC values.
    
    The power mean (or generalized mean) with exponent p is:
    M_p(x) = (1/n * sum(x_i^p))^(1/p)
    
    As p approaches negative infinity, this focuses more on the minimum values,
    which helps prioritize the worst-performing subgroups.
    
    Args:
        auc_list: List of AUC values to aggregate
        p: Power parameter (default -5, more negative values focus more on minimums)
        
    Returns:
        Generalized power mean of the AUC values
    """
    # Filter out missing values
    valid_aucs = np.array([auc for auc in auc_list if not np.isnan(auc)])
    
    if len(valid_aucs) == 0:
        return np.nan
    
    # Calculate power mean
    # Note: For negative p, this emphasizes lower values (worst performer)
    return np.power(np.mean(np.power(valid_aucs, p)), 1/p)


def final_bias_score(overall_auc: float, 
                     bias_auc_dict: Dict[str, List[float]],
                     power: float = -5,
                     weight_overall: float = 0.25) -> Tuple[float, Dict]:
    """
    Calculate the final bias score as per the Jigsaw competition formula.
    
    The formula combines overall AUC with three bias metrics:
    score = w0 * overall_auc + (1-w0) * (mean of power means of bias metrics)
    
    Args:
        overall_auc: Overall ROC AUC score for the model
        bias_auc_dict: Dictionary with keys 'subgroup_auc', 'bpsn_auc', 'bnsp_auc'
                     and values as lists of corresponding AUC scores
        power: Power parameter for generalized mean calculation
        weight_overall: Weight for overall AUC (w0) in [0,1]
        
    Returns:
        Tuple of (final_score, detailed_metrics_dict)
    """
    # Calculate power means for each bias metric type
    means = {}
    for metric_name, auc_values in bias_auc_dict.items():
        means[f"power_mean_{metric_name}"] = generalised_power_mean(auc_values, p=power)
    
    # Calculate bias component (average of the three means)
    bias_metrics = [
        means["power_mean_subgroup_auc"],
        means["power_mean_bpsn_auc"], 
        means["power_mean_bnsp_auc"]
    ]
    bias_score = np.mean([m for m in bias_metrics if not np.isnan(m)])
    
    # Calculate final weighted score
    final_score = weight_overall * overall_auc + (1 - weight_overall) * bias_score
    
    # Return both the final score and detailed metrics
    metrics_dict = {
        "overall_auc": overall_auc,
        "bias_score": bias_score,
        "final_score": final_score,
        **means
    }
    
    return final_score, metrics_dict


def compute_bias_metrics_for_subgroup(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    subgroup_mask: np.ndarray,
    subgroup_name: str
) -> Dict[str, float]:
    """
    Compute all bias metrics for a single subgroup.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities
        subgroup_mask: Boolean mask identifying members of the subgroup
        subgroup_name: Name of the subgroup (for reporting)
        
    Returns:
        Dictionary with bias metrics for the subgroup
    """
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


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_masks: Dict[str, np.ndarray],
    power: float = -5,
    weight_overall: float = 0.25
) -> Dict:
    """
    Compute comprehensive bias metrics across all subgroups.
    
    Args:
        y_true: Array of true binary labels
        y_pred: Array of predicted probabilities  
        subgroup_masks: Dictionary mapping subgroup names to boolean masks
        power: Power parameter for generalized mean
        weight_overall: Weight for overall AUC in final score
        
    Returns:
        Dictionary with detailed metrics for model evaluation
    """
    # Calculate overall AUC
    overall_auc = roc_auc_score(y_true, y_pred)
    
    # Compute metrics for each subgroup
    subgroup_metrics = []
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []
    
    for subgroup_name, mask in subgroup_masks.items():
        metrics = compute_bias_metrics_for_subgroup(
            y_true, y_pred, mask, subgroup_name
        )
        subgroup_metrics.append(metrics)
        
        # Collect AUCs for power mean calculation
        subgroup_aucs.append(metrics["subgroup_auc"])
        bpsn_aucs.append(metrics["bpsn_auc"])
        bnsp_aucs.append(metrics["bnsp_auc"])
    
    # Calculate final bias score
    bias_auc_dict = {
        "subgroup_auc": subgroup_aucs,
        "bpsn_auc": bpsn_aucs,
        "bnsp_auc": bnsp_aucs
    }
    
    final_score, detailed_metrics = final_bias_score(
        overall_auc, bias_auc_dict, power, weight_overall
    )
    
    # Combine everything into a complete results dictionary
    results = {
        "overall": {
            "auc": overall_auc,
            "final_score": final_score,
        },
        "subgroup_metrics": subgroup_metrics,
        "bias_metrics": detailed_metrics
    }
    
    return results 