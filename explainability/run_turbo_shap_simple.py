#!/usr/bin/env python
"""
Simple SHAP analyzer for turbo model - Focuses on token importance visualization
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import sys
sys.path.append(".")
from fairness_analysis.metrics_v2 import list_identity_columns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
torch.set_grad_enabled(False)  # Disable gradients for SHAP analysis

def load_turbo_model(ckpt_path):
    """Load the turbo model checkpoint and tokenizer"""
    print(f"Loading turbo model from {ckpt_path}...")
    
    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
    # Get model name from checkpoint
    if "config" in ckpt and "model_name" in ckpt["config"]:
        base_model = ckpt["config"]["model_name"]
    else:
        # Default for turbo model
        base_model = "distilbert-base-uncased"
    
    print(f"Base model: {base_model}")
    
    # Load model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load state dict
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            print("Warning: Could not find state_dict in checkpoint, using base model without weights")
        
        model.eval().to(DEVICE)
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {num_params:,} parameters")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error setting up model: {e}")
        return None, None

def calculate_token_importances(model, tokenizer, text, max_len=128):
    """Calculate token importances using occlusion-based approach"""
    print(f"Analyzing text: {text[:50]}...")
    
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_len - 2:  # Account for special tokens
        tokens = tokens[:max_len - 2]
    
    # Create input tensors with special tokens
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_tensor = torch.tensor([input_ids]).to(DEVICE)
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(input_tensor).logits[0].item()
    
    # Calculate importance for each token by masking
    token_importances = []
    for i in range(1, len(input_ids) - 1):  # Skip [CLS] and [SEP]
        # Create a copy with token masked (using [PAD] token)
        masked_ids = input_ids.copy()
        masked_ids[i] = tokenizer.pad_token_id
        masked_tensor = torch.tensor([masked_ids]).to(DEVICE)
        
        # Get prediction with token masked
        with torch.no_grad():
            masked_output = model(masked_tensor).logits[0].item()
        
        # Calculate importance
        importance = baseline_output - masked_output
        token_importances.append(importance)
    
    # Return tokens and importances
    return tokens, token_importances

def analyze_example(model, tokenizer, text, output_dir, prefix, max_len=128):
    """Analyze a single example and create visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate token importances
    try:
        tokens, importances = calculate_token_importances(model, tokenizer, text, max_len)
        
        # Create token importance bar chart
        plt.figure(figsize=(12, 8))
        
        # Sort tokens by importance
        token_imp_pairs = list(zip(tokens, importances))
        sorted_pairs = sorted(token_imp_pairs, key=lambda x: abs(x[1]), reverse=True)
        
        # Get top tokens
        top_n = min(20, len(sorted_pairs))
        top_tokens = [pair[0] for pair in sorted_pairs[:top_n]]
        top_importances = [pair[1] for pair in sorted_pairs[:top_n]]
        
        # Plot bar chart
        plt.barh(range(top_n), top_importances)
        plt.yticks(range(top_n), top_tokens)
        plt.title(f"Top {top_n} Tokens by Importance")
        plt.xlabel("Token Importance (Effect on Prediction)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_token_importance.png"), dpi=300)
        plt.close()
        
        # Save token importances to text file
        with open(os.path.join(output_dir, f"{prefix}_token_importances.txt"), "w", encoding="utf-8") as f:
            f.write(f"Text: {text}\n\n")
            f.write("Token Importances:\n")
            for token, importance in sorted_pairs[:top_n]:
                f.write(f"{token}: {importance:.6f}\n")
        
        # Create a simple heatmap
        plt.figure(figsize=(14, 4))
        
        # Calculate colormap normalization
        vmax = max(abs(min(importances)), abs(max(importances)))
        vmin = -vmax
        
        # Create heatmap rectangle for each token
        for i, (token, importance) in enumerate(zip(tokens, importances)):
            color = 'red' if importance > 0 else 'blue'
            alpha = min(1.0, abs(importance) / (vmax + 1e-10))
            plt.text(i, 0, token, ha='center', va='center', 
                     bbox=dict(facecolor=color, alpha=alpha, edgecolor='none'))
        
        plt.xlim(-0.5, len(tokens) - 0.5)
        plt.ylim(-0.5, 0.5)
        plt.axis('off')
        plt.title(f"Token Importance Visualization")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_heatmap.png"), dpi=300)
        plt.close()
        
        print(f"Analysis for example complete. Visualizations saved to {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error analyzing example: {e}")
        return False

def main():
    """Main entry point for simple SHAP analysis"""
    parser = argparse.ArgumentParser(description="Run simple SHAP analysis on turbo model")
    parser.add_argument("--ckpt", default="output/checkpoints/distilbert_headtail_fold0.pth",
                      help="Path to turbo model checkpoint")
    parser.add_argument("--valid-csv", default="data/valid.csv",
                      help="Path to validation or test CSV file")
    parser.add_argument("--sample", type=int, default=5,
                      help="Number of examples to sample for analysis")
    parser.add_argument("--out-dir", default="output/turbo_shap_simple",
                      help="Output directory for visualizations")
    parser.add_argument("--max-len", type=int, default=128,
                      help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for sample selection")
    
    args = parser.parse_args()
    
    # Load turbo model
    model, tokenizer = load_turbo_model(args.ckpt)
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load validation data
    try:
        print(f"Loading data from {args.valid_csv}...")
        valid_df = pd.read_csv(args.valid_csv)
        print(f"Loaded {len(valid_df)} examples")
        
        # Get balanced sample of examples
        if "target" in valid_df.columns:
            print("Creating balanced sample...")
            
            # Get equal numbers of toxic and non-toxic examples
            toxic = valid_df[valid_df["target"] >= 0.5].sample(
                min(args.sample // 2, len(valid_df[valid_df["target"] >= 0.5])), 
                random_state=args.seed
            )
            non_toxic = valid_df[valid_df["target"] < 0.5].sample(
                min(args.sample // 2, len(valid_df[valid_df["target"] < 0.5])), 
                random_state=args.seed
            )
            sample_df = pd.concat([toxic, non_toxic]).sample(frac=1, random_state=args.seed)
        else:
            print("Target column not found, using random sample...")
            sample_df = valid_df.sample(min(args.sample, len(valid_df)), random_state=args.seed)
        
        print(f"Selected {len(sample_df)} examples for analysis")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Analyze each example
    successful = 0
    for i, row in enumerate(sample_df.iterrows()):
        idx, row_data = row
        
        # Get text and toxicity label
        text = row_data["comment_text"]
        is_toxic = "unknown"
        if "target" in row_data:
            is_toxic = "toxic" if row_data["target"] >= 0.5 else "non_toxic"
        
        # Create prefix for files
        prefix = f"example_{i+1}_{is_toxic}"
        
        # Analyze example
        print(f"\nAnalyzing example {i+1}/{len(sample_df)}...")
        if analyze_example(model, tokenizer, text, args.out_dir, prefix, args.max_len):
            successful += 1
    
    print(f"\nAnalysis complete! Successfully analyzed {successful}/{len(sample_df)} examples.")
    print(f"Results saved to {args.out_dir}")

if __name__ == "__main__":
    main() 