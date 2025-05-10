#!/usr/bin/env python
"""
Merge prediction files with ground truth and identity columns
Usage:
    python scripts/merge_preds_with_labels.py \
           --preds output/preds/distilbert_dev.csv \
           --labels data/valid.csv \
           --out output/data/merged_val.csv
"""
import argparse
import pandas as pd
import os

def merge_predictions_with_labels(preds_file, labels_file, output_file):
    """
    Merge predictions CSV with ground truth labels and identity columns
    
    Args:
        preds_file: Path to predictions CSV (must have "id" and "prediction" columns)
        labels_file: Path to labels CSV (must have "id", "target", and identity columns)
        output_file: Path to save merged data
    """
    print(f"Loading predictions from {preds_file}...")
    
    # Handle different prediction file formats
    try:
        preds_df = pd.read_csv(preds_file)
        
        # Check if we have id and prediction columns
        if "id" not in preds_df.columns:
            # Try to use the first column as id
            preds_df = pd.read_csv(preds_file, index_col=0)
            preds_df.reset_index(inplace=True)
            preds_df.rename(columns={"index": "id"}, inplace=True)
            
        # If we don't have a prediction column, use the first column that's not id
        if "prediction" not in preds_df.columns:
            non_id_cols = [col for col in preds_df.columns if col != "id"]
            if non_id_cols:
                preds_df.rename(columns={non_id_cols[0]: "prediction"}, inplace=True)
            else:
                raise ValueError("No prediction column found in predictions file")
    except Exception as e:
        # If the file has a simple format, try to load it directly
        print(f"Error loading predictions: {e}")
        print("Trying alternative format...")
        preds_df = pd.read_csv(preds_file, header=None, names=["id", "prediction"])
    
    print(f"Loaded {len(preds_df)} predictions")
    
    print(f"Loading labels from {labels_file}...")
    # Load labels
    labels_df = pd.read_csv(labels_file)
    print(f"Loaded {len(labels_df)} labeled examples")
    
    # Get identity columns
    identity_cols = [col for col in labels_df.columns if col in [
        "male", "female", "transgender", "heterosexual", "homosexual_gay_or_lesbian", 
        "bisexual", "christian", "jewish", "muslim", "hindu", "buddhist", "atheist",
        "black", "white", "asian", "latino", "other_race_or_ethnicity",
        "physical_disability", "intellectual_or_learning_disability", 
        "psychiatric_or_mental_illness", "other_disability"
    ]]
    
    if not identity_cols:
        print("WARNING: No identity columns found in labels file")
    else:
        print(f"Found {len(identity_cols)} identity columns")
    
    # Columns to keep from labels
    keep_cols = ["id", "comment_text", "target"] + identity_cols
    
    # Merge predictions with labels
    print("Merging predictions with labels...")
    merged_df = pd.merge(
        preds_df[["id", "prediction"]], 
        labels_df[keep_cols],
        on="id",
        how="inner"
    )
    
    print(f"Merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    
    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Merge predictions with labels and identity columns")
    parser.add_argument("--preds", required=True, help="Path to predictions CSV")
    parser.add_argument("--labels", required=True, help="Path to labels CSV with identity columns")
    parser.add_argument("--out", required=True, help="Path to save merged data")
    
    args = parser.parse_args()
    merge_predictions_with_labels(args.preds, args.labels, args.out)

if __name__ == "__main__":
    main() 