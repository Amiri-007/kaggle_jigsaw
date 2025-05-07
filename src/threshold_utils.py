#!/usr/bin/env python
"""
Threshold sweep utilities for the RDS project.

This module provides functions to analyze how fairness metrics vary
with different classification thresholds.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix


def sweep_thresholds(y_true: np.ndarray, y_pred: np.ndarray, identity_mask: np.ndarray, 
                     n_steps: int = 101) -> pd.DataFrame:
    """
    Perform a threshold sweep analysis to find optimal classification thresholds
    for minimizing fairness disparities between demographic groups.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        identity_mask: Binary array indicating membership in an identity group
        n_steps: Number of threshold steps to evaluate between 0 and 1
        
    Returns:
        DataFrame with threshold values and corresponding fairness metrics
    """
    # Generate threshold values
    thresholds = np.linspace(0, 1, n_steps)
    
    # Initialize results
    results = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Convert probabilities to binary predictions using the threshold
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix for the subgroup
        subgroup_cm = confusion_matrix(
            y_true[identity_mask], 
            y_pred_binary[identity_mask], 
            labels=[0, 1]
        )
        
        # Calculate confusion matrix for the non-subgroup
        non_subgroup_cm = confusion_matrix(
            y_true[~identity_mask], 
            y_pred_binary[~identity_mask], 
            labels=[0, 1]
        )
        
        # Extract confusion matrix elements for subgroup
        tn_subgroup, fp_subgroup, fn_subgroup, tp_subgroup = subgroup_cm.ravel()
        
        # Extract confusion matrix elements for non-subgroup
        tn_non_subgroup, fp_non_subgroup, fn_non_subgroup, tp_non_subgroup = non_subgroup_cm.ravel()
        
        # Calculate FPR and FNR for subgroup
        fpr_subgroup = fp_subgroup / (fp_subgroup + tn_subgroup) if (fp_subgroup + tn_subgroup) > 0 else 0
        fnr_subgroup = fn_subgroup / (fn_subgroup + tp_subgroup) if (fn_subgroup + tp_subgroup) > 0 else 0
        
        # Calculate FPR and FNR for non-subgroup
        fpr_non_subgroup = fp_non_subgroup / (fp_non_subgroup + tn_non_subgroup) if (fp_non_subgroup + tn_non_subgroup) > 0 else 0
        fnr_non_subgroup = fn_non_subgroup / (fn_non_subgroup + tp_non_subgroup) if (fn_non_subgroup + tp_non_subgroup) > 0 else 0
        
        # Calculate gaps
        fpr_gap = abs(fpr_subgroup - fpr_non_subgroup)
        fnr_gap = abs(fnr_subgroup - fnr_non_subgroup)
        
        # Calculate a combined score (weighted average of the gaps)
        # Lower is better
        final_score = (fpr_gap + fnr_gap) / 2
        
        # Store results
        results.append({
            'threshold': threshold,
            'fpr_subgroup': fpr_subgroup,
            'fnr_subgroup': fnr_subgroup,
            'fpr_non_subgroup': fpr_non_subgroup,
            'fnr_non_subgroup': fnr_non_subgroup,
            'fpr_gap': fpr_gap,
            'fnr_gap': fnr_gap,
            'final_score': final_score
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray, 
                          identity_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Find the optimal threshold for each identity group that minimizes fairness disparities.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        identity_masks: Dictionary mapping identity group names to binary masks
        
    Returns:
        Dictionary mapping identity group names to their optimal thresholds
    """
    optimal_thresholds = {}
    
    for group_name, mask in identity_masks.items():
        # Perform threshold sweep
        sweep_results = sweep_thresholds(y_true, y_pred, mask)
        
        # Find threshold with minimum final score
        optimal_idx = sweep_results['final_score'].idxmin()
        optimal_threshold = sweep_results.loc[optimal_idx, 'threshold']
        
        optimal_thresholds[group_name] = optimal_threshold
    
    return optimal_thresholds


def calculate_fairness_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, 
                                  identity_mask: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Calculate fairness metrics at a specific threshold for a given identity group.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        identity_mask: Binary array indicating membership in an identity group
        threshold: Classification threshold to use
        
    Returns:
        Dictionary containing fairness metrics
    """
    # Convert probabilities to binary predictions using the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate confusion matrix for the subgroup
    subgroup_cm = confusion_matrix(
        y_true[identity_mask], 
        y_pred_binary[identity_mask], 
        labels=[0, 1]
    )
    
    # Calculate confusion matrix for the non-subgroup
    non_subgroup_cm = confusion_matrix(
        y_true[~identity_mask], 
        y_pred_binary[~identity_mask], 
        labels=[0, 1]
    )
    
    # Extract confusion matrix elements for subgroup
    tn_subgroup, fp_subgroup, fn_subgroup, tp_subgroup = subgroup_cm.ravel()
    
    # Extract confusion matrix elements for non-subgroup
    tn_non_subgroup, fp_non_subgroup, fn_non_subgroup, tp_non_subgroup = non_subgroup_cm.ravel()
    
    # Calculate metrics for subgroup
    fpr_subgroup = fp_subgroup / (fp_subgroup + tn_subgroup) if (fp_subgroup + tn_subgroup) > 0 else 0
    fnr_subgroup = fn_subgroup / (fn_subgroup + tp_subgroup) if (fn_subgroup + tp_subgroup) > 0 else 0
    tpr_subgroup = tp_subgroup / (tp_subgroup + fn_subgroup) if (tp_subgroup + fn_subgroup) > 0 else 0
    tnr_subgroup = tn_subgroup / (tn_subgroup + fp_subgroup) if (tn_subgroup + fp_subgroup) > 0 else 0
    
    # Calculate metrics for non-subgroup
    fpr_non_subgroup = fp_non_subgroup / (fp_non_subgroup + tn_non_subgroup) if (fp_non_subgroup + tn_non_subgroup) > 0 else 0
    fnr_non_subgroup = fn_non_subgroup / (fn_non_subgroup + tp_non_subgroup) if (fn_non_subgroup + tp_non_subgroup) > 0 else 0
    tpr_non_subgroup = tp_non_subgroup / (tp_non_subgroup + fn_non_subgroup) if (tp_non_subgroup + fn_non_subgroup) > 0 else 0
    tnr_non_subgroup = tn_non_subgroup / (tn_non_subgroup + fp_non_subgroup) if (tn_non_subgroup + fp_non_subgroup) > 0 else 0
    
    # Calculate gaps
    fpr_gap = fpr_subgroup - fpr_non_subgroup
    fnr_gap = fnr_subgroup - fnr_non_subgroup
    tpr_gap = tpr_subgroup - tpr_non_subgroup
    tnr_gap = tnr_subgroup - tnr_non_subgroup
    
    # Calculate disparate impact
    p_subgroup = (tp_subgroup + fp_subgroup) / (tp_subgroup + fp_subgroup + tn_subgroup + fn_subgroup) if (tp_subgroup + fp_subgroup + tn_subgroup + fn_subgroup) > 0 else 0
    p_non_subgroup = (tp_non_subgroup + fp_non_subgroup) / (tp_non_subgroup + fp_non_subgroup + tn_non_subgroup + fn_non_subgroup) if (tp_non_subgroup + fp_non_subgroup + tn_non_subgroup + fn_non_subgroup) > 0 else 0
    disparate_impact = p_subgroup / p_non_subgroup if p_non_subgroup > 0 else float('inf')
    
    # Return all metrics
    return {
        'threshold': threshold,
        'fpr_subgroup': fpr_subgroup,
        'fnr_subgroup': fnr_subgroup,
        'tpr_subgroup': tpr_subgroup,
        'tnr_subgroup': tnr_subgroup,
        'fpr_non_subgroup': fpr_non_subgroup,
        'fnr_non_subgroup': fnr_non_subgroup,
        'tpr_non_subgroup': tpr_non_subgroup,
        'tnr_non_subgroup': tnr_non_subgroup,
        'fpr_gap': fpr_gap,
        'fnr_gap': fnr_gap,
        'tpr_gap': tpr_gap,
        'tnr_gap': tnr_gap,
        'disparate_impact': disparate_impact
    }


def error_rate_gaps(y_true: np.ndarray, y_prob: np.ndarray, 
                   subgroup_mask: np.ndarray, thresh: float = 0.5) -> Tuple[float, float]:
    """
    Calculate false positive rate (FPR) and false negative rate (FNR) gaps between a subgroup
    and the rest of the population at a specific classification threshold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        subgroup_mask: Binary mask indicating membership in a demographic subgroup
        thresh: Classification threshold to use
        
    Returns:
        Tuple containing (fpr_gap, fnr_gap)
    """
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_prob >= thresh).astype(int)
    
    # Calculate confusion matrix for the subgroup
    subgroup_cm = confusion_matrix(
        y_true[subgroup_mask], 
        y_pred[subgroup_mask], 
        labels=[0, 1]
    )
    
    # Calculate confusion matrix for the non-subgroup
    non_subgroup_cm = confusion_matrix(
        y_true[~subgroup_mask], 
        y_pred[~subgroup_mask], 
        labels=[0, 1]
    )
    
    # Extract confusion matrix elements for subgroup
    tn_subgroup, fp_subgroup, fn_subgroup, tp_subgroup = subgroup_cm.ravel()
    
    # Extract confusion matrix elements for non-subgroup
    tn_non_subgroup, fp_non_subgroup, fn_non_subgroup, tp_non_subgroup = non_subgroup_cm.ravel()
    
    # Calculate FPR and FNR for subgroup
    fpr_subgroup = fp_subgroup / (fp_subgroup + tn_subgroup) if (fp_subgroup + tn_subgroup) > 0 else 0
    fnr_subgroup = fn_subgroup / (fn_subgroup + tp_subgroup) if (fn_subgroup + tp_subgroup) > 0 else 0
    
    # Calculate FPR and FNR for non-subgroup
    fpr_non_subgroup = fp_non_subgroup / (fp_non_subgroup + tn_non_subgroup) if (fp_non_subgroup + tn_non_subgroup) > 0 else 0
    fnr_non_subgroup = fn_non_subgroup / (fn_non_subgroup + tp_non_subgroup) if (fn_non_subgroup + tp_non_subgroup) > 0 else 0
    
    # Calculate gaps (subgroup rate - non-subgroup rate)
    fpr_gap = fpr_subgroup - fpr_non_subgroup
    fnr_gap = fnr_subgroup - fnr_non_subgroup
    
    return fpr_gap, fnr_gap


def sweep_threshold_gaps(y_true: np.ndarray, y_prob: np.ndarray, 
                        subgroup_mask: np.ndarray, n_steps: int = 101) -> pd.DataFrame:
    """
    Perform a threshold sweep to analyze how FPR and FNR gaps change with different thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        subgroup_mask: Binary mask indicating membership in a demographic subgroup
        n_steps: Number of threshold steps to evaluate between 0 and 1
        
    Returns:
        DataFrame with threshold values and corresponding FPR and FNR gaps
    """
    # Generate threshold values
    thresholds = np.linspace(0, 1, n_steps)
    
    # Initialize results
    results = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Calculate error rate gaps at this threshold
        fpr_gap, fnr_gap = error_rate_gaps(y_true, y_prob, subgroup_mask, thresh=threshold)
        
        # Store results
        results.append({
            'threshold': threshold,
            'fpr_gap': fpr_gap,
            'fnr_gap': fnr_gap,
            'abs_fpr_gap': abs(fpr_gap),
            'abs_fnr_gap': abs(fnr_gap),
            'max_gap': max(abs(fpr_gap), abs(fnr_gap)),
            'mean_gap': (abs(fpr_gap) + abs(fnr_gap)) / 2
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df 