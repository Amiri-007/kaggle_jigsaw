#!/usr/bin/env python
"""
Script to generate fairness metrics from prediction files and write them to CSV files.
This integrates with the existing fairness dashboard visualization.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Import fairness metrics
from fairness import BiasReport, final_score

def load_data(predictions_path, ground_truth_path=None):
    """
    Load prediction data and optionally ground truth data.
    
    Args:
        predictions_path: Path to prediction CSV file
        ground_truth_path: Path to ground truth CSV file
        
    Returns:
        DataFrame with predictions and ground truth (if available)
    """
    # Load predictions
    preds_df = pd.read_csv(predictions_path)
    
    # Check required columns
    if 'id' not in preds_df.columns or 'prediction' not in preds_df.columns:
        print(f"Error: Prediction file must contain 'id' and 'prediction' columns")
        sys.exit(1)
    
    # If ground truth provided, merge with predictions
    if ground_truth_path:
        if not os.path.exists(ground_truth_path):
            print(f"Error: Ground truth file not found: {ground_truth_path}")
            sys.exit(1)
            
        truth_df = pd.read_csv(ground_truth_path)
        
        # Check required columns
        required_cols = ['id', 'target']
        for col in required_cols:
            if col not in truth_df.columns:
                print(f"Error: Required column '{col}' not found in ground truth data")
                sys.exit(1)
                
        # Merge with predictions
        df = preds_df.merge(truth_df, on='id', how='inner')
        
    else:
        # Try to find ground truth file in standard locations
        for potential_path in ['data/valid.csv', 'data/test_public_expanded.csv']:
            if os.path.exists(potential_path):
                try:
                    truth_df = pd.read_csv(potential_path)
                    if 'id' in truth_df.columns and 'target' in truth_df.columns:
                        df = preds_df.merge(truth_df, on='id', how='inner')
                        print(f"Found and merged ground truth data from {potential_path}")
                        break
                except:
                    continue
        else:
            print("Warning: No ground truth data found. Using predictions file only.")
            df = preds_df
            
            # Add dummy target column if needed for BiasReport
            if 'target' not in df.columns:
                df['target'] = np.nan
    
    return df

def list_identity_columns():
    """Return the list of standard identity columns used in the dataset."""
    return [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness',
        'asian', 'hindu', 'buddhist', 'atheist', 'bisexual', 'transgender'
    ]

def write_metrics(df, model_name):
    """
    Calculate fairness metrics and write to CSV file.
    
    Args:
        df: DataFrame with predictions and ground truth
        model_name: Name of the model for output file
        
    Returns:
        Path to the output metrics file
    """
    # Get identity columns
    identity_cols = list_identity_columns()
        
    # Ensure identity columns exist
    missing_cols = [col for col in identity_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following identity columns are missing: {missing_cols}")
        print("Adding them with default value of 0")
        for col in missing_cols:
            df[col] = 0
    
    # Create BiasReport
    try:
        bias_report = BiasReport(
            df=df,
            identity_cols=identity_cols,
            label_col='target',
            pred_col='prediction'
        )
        
        # Get metrics
        metrics_df = bias_report.get_metrics_df()
        
        # Calculate final score
        score = final_score(bias_report)
        print(f"Final fairness score: {score:.4f}")
        
        # Create output directory
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        output_path = output_dir / f"metrics_{model_name}.csv"
        metrics_df.to_csv(output_path, index=False)
        
        print(f"Metrics written to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Calculate fairness metrics from prediction file")
    parser.add_argument('--predictions', type=str, required=True,
                      help="Path to predictions CSV file")
    parser.add_argument('--ground_truth', type=str, default=None,
                      help="Path to ground truth CSV file")
    parser.add_argument('--model_name', type=str, default=None,
                      help="Model name for output file")
    parser.add_argument('--output_dir', type=str, default='results',
                      help="Directory to save metrics CSV file")
    args = parser.parse_args()
    
    # Determine model name from predictions file if not provided
    if args.model_name is None:
        args.model_name = Path(args.predictions).stem
    
    # Load data
    df = load_data(args.predictions, args.ground_truth)
    
    # Calculate and write metrics
    try:
        metrics_path = write_metrics(df, args.model_name)
        print(f"Fairness metrics calculation complete. Results saved to {metrics_path}")
        
        # Check if the figure generation script exists and run it
        figures_script = Path('notebooks') / '04_generate_figures.py'
        if figures_script.exists():
            try:
                print(f"Running figure generation script...")
                import subprocess
                cmd = [sys.executable, str(figures_script)]
                subprocess.run(cmd, check=True)
                print("Figure generation complete")
            except Exception as e:
                print(f"Error running figure generation script: {str(e)}")
        
    except Exception as e:
        print(f"Error calculating fairness metrics: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 