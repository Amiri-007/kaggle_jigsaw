#!/usr/bin/env python
"""
Merge predictions with labels for fairness analysis.
This script joins model predictions with the validation data containing identity labels,
which is needed for demographic group analysis.
"""
import argparse
import pandas as pd
import os
from pathlib import Path

def merge_preds_with_labels(preds_path, labels_path, output_path):
    """
    Merge predictions with labels from the validation set.
    
    Args:
        preds_path: Path to predictions CSV with 'id' and 'prediction' columns
        labels_path: Path to validation CSV with 'id', 'target', and identity columns
        output_path: Path to save the merged CSV
    """
    print(f"Loading predictions from {preds_path}")
    preds_df = pd.read_csv(preds_path)
    
    print(f"Loading validation data from {labels_path}")
    labels_df = pd.read_csv(labels_path)
    
    # Check for required columns
    if 'id' not in preds_df.columns or 'prediction' not in preds_df.columns:
        raise ValueError("Predictions CSV must have 'id' and 'prediction' columns")
    
    if 'id' not in labels_df.columns or 'target' not in labels_df.columns:
        raise ValueError("Labels CSV must have 'id' and 'target' columns")
    
    # Merge on id
    print("Merging datasets...")
    merged_df = pd.merge(preds_df, labels_df, on='id', how='inner')
    
    # Check if we have any rows
    if len(merged_df) == 0:
        raise ValueError("No matching IDs found between predictions and labels")
    
    print(f"Successfully merged {len(merged_df)} rows")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save merged dataset
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")
    
    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Merge predictions with validation data for fairness analysis")
    parser.add_argument("--preds", required=True, help="Path to predictions CSV")
    parser.add_argument("--labels", required=True, help="Path to validation CSV with labels")
    parser.add_argument("--out", required=True, help="Path to save merged CSV")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    preds_path = Path(args.preds)
    labels_path = Path(args.labels)
    output_path = Path(args.out)
    
    # Check if files exist
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Merge datasets
    merge_preds_with_labels(preds_path, labels_path, output_path)

if __name__ == "__main__":
    main() 
