#!/usr/bin/env python
"""
Script to create dummy metrics files for CI testing.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_dummy_metrics():
    """
    Create a dummy metrics CSV file for CI testing.
    """
    # Define identity columns
    identity_cols = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
    
    # Create a DataFrame with metrics for each identity column
    rows = []
    
    # Overall metrics
    row = {
        'identity': 'overall',
        'subgroup': 'all',
        'subgroup_size': 1000,
        'true_prevalence': 0.1,
        'predicted_prevalence': 0.12,
        'threshold': 0.5,
        'accuracy': 0.9,
        'auc': 0.85,
        'tn': 850,
        'fp': 50,
        'fn': 50,
        'tp': 50,
        'fpr': 0.05,
        'fnr': 0.5,
        'tpr': 0.5,
        'tnr': 0.95,
        'mcc': 0.5
    }
    rows.append(row)
    
    # Metrics for each identity column
    for identity in identity_cols:
        # Positive subgroup
        row = row.copy()
        row['identity'] = identity
        row['subgroup'] = 'pos'
        row['subgroup_size'] = 100
        row['true_prevalence'] = 0.2
        row['predicted_prevalence'] = 0.22
        row['accuracy'] = 0.85
        row['auc'] = 0.8
        row['tn'] = 75
        row['fp'] = 5
        row['fn'] = 10
        row['tp'] = 10
        row['fpr'] = 0.06
        row['fnr'] = 0.5
        row['tpr'] = 0.5
        row['tnr'] = 0.94
        row['mcc'] = 0.45
        rows.append(row)
        
        # Negative subgroup
        row = row.copy()
        row['subgroup'] = 'neg'
        row['subgroup_size'] = 900
        row['true_prevalence'] = 0.08
        row['predicted_prevalence'] = 0.1
        row['accuracy'] = 0.92
        row['auc'] = 0.87
        row['tn'] = 800
        row['fp'] = 28
        row['fn'] = 40
        row['tp'] = 32
        row['fpr'] = 0.035
        row['fnr'] = 0.55
        row['tpr'] = 0.45
        row['tnr'] = 0.965
        row['mcc'] = 0.52
        rows.append(row)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(rows)
    
    # Set datatypes
    int_cols = ['subgroup_size', 'tn', 'fp', 'fn', 'tp']
    float_cols = [col for col in metrics_df.columns if col not in int_cols + ['identity', 'subgroup']]
    
    for col in int_cols:
        metrics_df[col] = metrics_df[col].astype(int)
    
    for col in float_cols:
        metrics_df[col] = metrics_df[col].astype(float)
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics files for multiple models
    model_names = ['bert_headtail', 'gpt2_headtail', 'lstm_caps']
    
    for model_name in model_names:
        # Add some variation for each model
        df = metrics_df.copy()
        df['auc'] = df['auc'] + np.random.uniform(-0.05, 0.05, len(df))
        df['accuracy'] = df['accuracy'] + np.random.uniform(-0.03, 0.03, len(df))
        
        # Ensure values are in valid ranges
        df['auc'] = df['auc'].clip(0, 1)
        df['accuracy'] = df['accuracy'].clip(0, 1)
        
        # Save to file
        output_path = output_dir / f"metrics_{model_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Created dummy metrics file: {output_path}")
    
    # Also create a few dummy prediction CSV files for testing
    output_dir = Path('output/preds')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create 1000 dummy predictions
    ids = list(range(1, 1001))
    
    for model_name in model_names:
        # Generate slightly different predictions for each model
        predictions = np.random.beta(1.5, 8.5, 1000) + np.random.uniform(-0.05, 0.05, 1000)
        predictions = np.clip(predictions, 0, 1)
        
        # Create DataFrame
        preds_df = pd.DataFrame({
            'id': ids,
            'prediction': predictions
        })
        
        # Save to file
        output_path = output_dir / f"{model_name}.csv"
        preds_df.to_csv(output_path, index=False)
        print(f"Created dummy predictions file: {output_path}")

if __name__ == '__main__':
    create_dummy_metrics() 