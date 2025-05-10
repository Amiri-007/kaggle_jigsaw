#!/usr/bin/env python
"""
Script to generate model explanations using SHAP and LIME.
This creates explainer visualizations to help understand model predictions.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import random
import shap
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models import load_model
from src.data.utils import list_identity_columns


def parse_args():
    parser = argparse.ArgumentParser(description="Generate model explanations using SHAP and LIME")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint or model name")
    parser.add_argument("--data-file", default="data/valid.csv", help="Path to data file for explanations")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to explain")
    parser.add_argument("--out-dir", default="output/explainers", help="Output directory for explanations")
    parser.add_argument("--identity-explain", action="store_true", help="Generate explanations for each identity group")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (CI testing)")
    return parser.parse_args()


def run_shap_explainer(model, tokenizer, texts, out_dir, prefix=""):
    """
    Run SHAP explainer on transformer model
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer for the model
        texts: List of text samples to explain
        out_dir: Output directory
        prefix: Prefix for output files
    """
    # Create the explainer
    explainer = shap.Explainer(model, tokenizer)
    
    # Sample a subset of texts to explain (SHAP can be slow)
    if len(texts) > 50:
        explain_texts = random.sample(texts, 50)
    else:
        explain_texts = texts
    
    # Run explanations
    print(f"Running SHAP explanations for {len(explain_texts)} samples...")
    shap_values = explainer(explain_texts)
    
    # Save the explainer and values
    with open(os.path.join(out_dir, f"{prefix}shap_values.pkl"), "wb") as f:
        pickle.dump(shap_values, f)
    
    # Generate waterfall plots for a few examples
    for i in range(min(5, len(explain_texts))):
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[i], max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}shap_waterfall_{i}.png"))
        plt.close()
    
    # Generate a summary plot
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}shap_summary.png"))
    plt.close()
    
    print(f"SHAP explanations saved to {out_dir}")


def run_lime_explainer(model, texts, class_names, out_dir, prefix=""):
    """
    Run LIME explainer on any model
    
    Args:
        model: Model with predict method
        texts: List of text samples to explain
        class_names: List of class names
        out_dir: Output directory
        prefix: Prefix for output files
    """
    # Create the explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Sample a subset of texts to explain
    if len(texts) > 20:
        explain_texts = random.sample(texts, 20)
    else:
        explain_texts = texts
    
    # Define prediction function for LIME
    def predict_proba(texts):
        preds = model.predict(texts)
        # Convert single probability to [1-p, p] for binary classification
        return np.column_stack([1 - preds, preds])
    
    # Run explanations
    print(f"Running LIME explanations for {len(explain_texts)} samples...")
    explanations = []
    
    for i, text in enumerate(tqdm(explain_texts)):
        try:
            exp = explainer.explain_instance(text, predict_proba, num_features=20)
            explanations.append(exp)
            
            # Save HTML visualization
            html_path = os.path.join(out_dir, f"{prefix}lime_explanation_{i}.html")
            exp.save_to_file(html_path)
            
            # Generate and save figure
            fig = exp.as_pyplot_figure(label=1)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}lime_exp_{i}.png"))
            plt.close()
        except Exception as e:
            print(f"Error explaining instance {i}: {str(e)}")
    
    # Save the explainer and explanations
    with open(os.path.join(out_dir, f"{prefix}lime_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)
    
    with open(os.path.join(out_dir, f"{prefix}lime_explanations.pkl"), "wb") as f:
        pickle.dump(explanations, f)
    
    print(f"LIME explanations saved to {out_dir}")


def main():
    args = parse_args()
    
    # In dry-run mode, just create output directory and return
    if args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Dry run mode: created output directory {args.out_dir}")
        return
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_file}")
    df = pd.read_csv(args.data_file)
    if len(df) > args.n_samples:
        df = df.sample(args.n_samples, random_state=42)
    
    # Get texts
    text_col = "comment_text"
    texts = df[text_col].values
    
    # Get model type to determine explainer
    if hasattr(model, "tokenizer"):
        print("Using SHAP explainer for transformer model")
        tokenizer = model.tokenizer
        run_shap_explainer(model, tokenizer, texts, args.out_dir)
    else:
        print("Using LIME explainer")
        run_lime_explainer(model, texts, ["Non-toxic", "Toxic"], args.out_dir)
    
    # Generate identity-specific explanations if requested
    if args.identity_explain:
        identity_cols = list_identity_columns()
        for col in identity_cols:
            if col in df.columns:
                # Get examples with this identity
                identity_df = df[df[col] == 1]
                if len(identity_df) > 0:
                    print(f"Generating explanations for {col} group ({len(identity_df)} examples)")
                    identity_dir = os.path.join(args.out_dir, col)
                    os.makedirs(identity_dir, exist_ok=True)
                    
                    identity_texts = identity_df[text_col].values
                    
                    if hasattr(model, "tokenizer"):
                        run_shap_explainer(model, tokenizer, identity_texts, identity_dir, f"{col}_")
                    else:
                        run_lime_explainer(model, identity_texts, ["Non-toxic", "Toxic"], identity_dir, f"{col}_")


if __name__ == "__main__":
    main() 