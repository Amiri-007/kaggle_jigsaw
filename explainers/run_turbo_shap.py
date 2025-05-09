#!/usr/bin/env python
"""
SHAP explainer for turbo model - Optimized for distilbert_headtail_fold0.pth
"""
import argparse
import os
import pathlib
import json
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

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

def build_turbo_explainer(model, tokenizer, max_len=128):
    """Build SHAP explainer optimized for turbo model"""
    def turbo_predictor(texts):
        """Prediction function for SHAP"""
        # Handle different types of input
        if isinstance(texts, str):
            batch_texts = [texts]
        elif isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            batch_texts = texts
        else:
            batch_texts = [str(x) for x in texts]
        
        # Process in smaller batches to avoid OOM errors
        batch_size = 16
        all_preds = []
        
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i:i+batch_size]
            
            # Tokenize and get model output
            enc = tokenizer(batch,
                          max_length=max_len,
                          truncation=True,
                          padding="max_length",
                          return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**enc).logits
                all_preds.append(outputs.cpu())
        
        # Combine predictions
        return torch.cat(all_preds, dim=0)
    
    # Create text masker for the tokenizer
    masker = shap.maskers.Text(tokenizer)
    
    # Create explainer with output name
    return shap.Explainer(turbo_predictor, masker, output_names=["toxicity"])

def run_shap_analysis(model, tokenizer, samples_df, out_dir, max_len=128, 
                      n_examples=5, background_size=50):
    """Run SHAP analysis on sample data"""
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Build explainer
    print("Building SHAP explainer...")
    explainer = build_turbo_explainer(model, tokenizer, max_len)
    
    # Get texts for analysis
    texts = samples_df["comment_text"].tolist()[:n_examples]
    print(f"Analyzing {len(texts)} examples")
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer(texts)
    
    # Save each example's analysis
    all_results = []
    for i, text in enumerate(texts):
        print(f"Analyzing example {i+1}...")
        
        # Is this a toxic example?
        is_toxic = "unknown"
        if "target" in samples_df.columns:
            target = samples_df["target"].iloc[i]
            is_toxic = "toxic" if target >= 0.5 else "non_toxic"
        
        # Create visualizations
        result = save_example_analysis(
            shap_values, i, pathlib.Path(out_dir),
            f"example_{i+1}_{is_toxic}", 
            f"SHAP Analysis for Example {i+1} ({is_toxic.replace('_', ' ').title()})"
        )
        
        # Store target if available
        if "target" in samples_df.columns:
            result["true_target"] = float(samples_df["target"].iloc[i])
        
        all_results.append(result)
    
    # Create summary visualization for all examples
    try:
        print("Creating summary visualizations...")
        plt.figure(figsize=(12, 8))
        
        # Safe approach to create bar plot of most important features
        feature_importance = np.abs(shap_values.values).mean(0)
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance[:, 0]  # Get first output
        
        # Create bar plot directly
        sorted_idx = np.argsort(-feature_importance)
        plt.barh(range(min(20, len(sorted_idx))), 
                feature_importance[sorted_idx[:20]])
        
        # Create y-tick labels
        features = []
        for i in sorted_idx[:20]:
            if i < len(shap_values.feature_names):
                features.append(shap_values.feature_names[i])
            else:
                features.append(f"Feature_{i}")
        
        plt.yticks(range(min(20, len(sorted_idx))), features)
        plt.title("Top 20 Features by Mean |SHAP|")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top_features_bar.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating summary bar plot: {e}")
    
    # Save summary to JSON
    with open(os.path.join(out_dir, "shap_analysis_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Analysis complete! Results saved to {out_dir}")
    return all_results

def save_example_analysis(shap_values, idx, out_dir, prefix, title):
    """Save visualizations and data for a single example"""
    # Get example text
    text = shap_values.data[idx]
    
    # Get base value and SHAP values
    base_value = shap_values.base_values[idx]
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]  # Get first output for multi-output models
    
    # Get SHAP values for this example
    shap_value = shap_values.values[idx]
    if len(shap_value.shape) > 1:
        shap_value = shap_value[:, 0]  # Get first output for multi-output models
    
    shap_sum = np.sum(shap_value)
    
    # Calculate final prediction
    prediction = float(base_value + shap_sum)
    
    # Create token importance bar chart (simpler alternative to waterfall)
    plt.figure(figsize=(12, 8))
    
    # Get token importances
    token_importances = np.abs(shap_value)
    tokens = []
    
    # Safe way to match tokens with importances
    min_len = min(len(token_importances), len(shap_values.feature_names))
    for i in range(min_len):
        tokens.append((shap_values.feature_names[i], token_importances[i]))
    
    # Sort by importance
    sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)[:20]
    
    token_names = [t[0] for t in sorted_tokens]
    token_values = [t[1] for t in sorted_tokens]
    
    plt.barh(range(len(token_names)), token_values)
    plt.yticks(range(len(token_names)), token_names)
    plt.title(f"Top Token Importances for {prefix.replace('_', ' ').title()}")
    plt.xlabel("Token Importance (|SHAP|)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_token_importance.png"), dpi=300)
    plt.close()
    
    # Create manual waterfall-like visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Sort SHAP values by magnitude
        sorted_idx = np.argsort(-np.abs(shap_value))
        
        # Take top 10 values
        top_n = 10
        values = []
        names = []
        
        for i in range(min(top_n, len(sorted_idx))):
            if sorted_idx[i] < len(shap_values.feature_names):
                values.append(shap_value[sorted_idx[i]])
                names.append(shap_values.feature_names[sorted_idx[i]])
        
        # Add base value and total
        values = [base_value] + values + [prediction]
        names = ["Base value"] + names + ["Prediction"]
        
        # Create waterfall plot manually
        cumulative = np.cumsum(values)
        plt.bar(range(len(names)), values, bottom=np.concatenate(([0], cumulative[:-1])))
        plt.xticks(range(len(names)), names, rotation=90)
        plt.title(f"Contribution to Prediction for {prefix.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_waterfall.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating waterfall plot: {e}")
    
    # Create text heatmap visualization
    try:
        plt.figure(figsize=(16, 4))
        
        # Create a simple text heatmap
        words = text.split()
        word_importance = np.zeros(len(words))
        
        # Try to match words with feature importances
        for i, word in enumerate(words):
            for j, feature in enumerate(shap_values.feature_names):
                if word == feature and j < len(shap_value):
                    word_importance[i] = shap_value[j]
                    break
        
        # Plot as a heatmap
        plt.imshow([word_importance], cmap="coolwarm", aspect="auto")
        plt.xticks(range(len(words)), words, rotation=45, ha="right")
        plt.yticks([])
        plt.colorbar(label="SHAP value")
        plt.title(f"Token Importance Heatmap for {prefix.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_text_heatmap.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating text heatmap: {e}")
    
    # Save example details to text file
    with open(os.path.join(out_dir, f"{prefix}_details.txt"), "w", encoding="utf-8") as f:
        f.write(f"Text: {text}\n\n")
        f.write(f"Base value: {base_value:.4f}\n")
        f.write(f"SHAP sum: {shap_sum:.4f}\n")
        f.write(f"Prediction (base + SHAP): {prediction:.4f}\n")
        
        f.write("\nTop 20 Tokens by Importance:\n")
        for i, (token, importance) in enumerate(sorted_tokens):
            f.write(f"{i+1}. {token}: {importance:.4f}\n")
    
    # Return analysis results
    return {
        "text": text,
        "base_value": float(base_value),
        "shap_sum": float(shap_sum),
        "prediction": prediction,
        "top_tokens": [{
            "token": token,
            "importance": float(importance)
        } for token, importance in sorted_tokens[:10]]
    }

def main():
    """Main entry point for SHAP analysis"""
    parser = argparse.ArgumentParser(description="Run SHAP analysis on turbo model")
    parser.add_argument("--ckpt", default="output/checkpoints/distilbert_headtail_fold0.pth",
                      help="Path to turbo model checkpoint")
    parser.add_argument("--valid-csv", default="data/valid.csv",
                      help="Path to validation or test CSV file")
    parser.add_argument("--sample", type=int, default=10,
                      help="Number of examples to sample for analysis")
    parser.add_argument("--out-dir", default="output/turbo_shap_analysis",
                      help="Output directory for SHAP visualizations")
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
    
    # Run SHAP analysis
    run_shap_analysis(
        model=model,
        tokenizer=tokenizer,
        samples_df=sample_df,
        out_dir=args.out_dir,
        max_len=args.max_len,
        n_examples=min(5, len(sample_df))  # Limit to 5 examples for detailed analysis
    )

if __name__ == "__main__":
    main() 