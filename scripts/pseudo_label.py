#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from a trained model")
    parser.add_argument("--base-model", required=True, help="Path to base model checkpoint or model name")
    parser.add_argument("--unlabeled-csv", required=True, help="Path to CSV with unlabeled data")
    parser.add_argument("--out-csv", required=True, help="Path to output pseudo-labeled CSV")
    parser.add_argument("--pred-thresh", type=float, default=0.9, help="Prediction threshold for positive labels")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (CI testing)")
    return parser.parse_args()


def generate_pseudo_labels(base_model, unlabeled_df, pred_thresh=0.9):
    """
    Generate pseudo-labels from model predictions
    
    Args:
        base_model: Trained model
        unlabeled_df: DataFrame with unlabeled data
        pred_thresh: Threshold for positive pseudo-labels
        
    Returns:
        DataFrame with pseudo-labels
    """
    # Get predictions from model
    text_column = "comment_text"
    ids = unlabeled_df["id"].values
    texts = unlabeled_df[text_column].values
    
    # Predict on texts
    preds = base_model.predict(texts)
    
    # Filter based on confidence thresholds
    high_conf_mask = (preds > pred_thresh) | (preds < (1 - pred_thresh))
    high_conf_ids = ids[high_conf_mask]
    high_conf_texts = texts[high_conf_mask]
    high_conf_preds = preds[high_conf_mask]
    
    # Create DataFrame with pseudo-labels
    pseudo_df = pd.DataFrame({
        "id": high_conf_ids,
        "comment_text": high_conf_texts,
        "pseudo_target": high_conf_preds
    })
    
    print(f"Generated {len(pseudo_df)} pseudo-labels from {len(unlabeled_df)} unlabeled examples")
    print(f"Positive examples: {(pseudo_df['pseudo_target'] > 0.5).sum()}")
    print(f"Negative examples: {(pseudo_df['pseudo_target'] <= 0.5).sum()}")
    
    return pseudo_df


def main():
    args = parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    # In dry-run mode, just create an empty CSV with header
    if args.dry_run:
        print("Dry run mode: creating empty pseudo-label CSV with header only")
        # Create empty DataFrame with just the header
        pseudo_df = pd.DataFrame(columns=["id", "comment_text", "pseudo_target"])
        pseudo_df.to_csv(args.out_csv, index=False)
        return
    
    # Load unlabeled data
    print(f"Loading unlabeled data from {args.unlabeled_csv}")
    unlabeled_df = pd.read_csv(args.unlabeled_csv)
    
    # Load base model
    print(f"Loading base model from {args.base_model}")
    base_model = load_model(args.base_model)
    
    # Generate pseudo-labels
    pseudo_df = generate_pseudo_labels(
        base_model, 
        unlabeled_df,
        pred_thresh=args.pred_thresh
    )
    
    # Save pseudo-labeled data
    pseudo_df.to_csv(args.out_csv, index=False)
    print(f"Saved pseudo-labels to {args.out_csv}")


if __name__ == "__main__":
    main() 