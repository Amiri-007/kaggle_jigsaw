#!/usr/bin/env python
"""
Generate simple token attributions for toxicity models
This is a simplified approach that doesn't use the full SHAP library
"""
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(ckpt_path: pathlib.Path):
    """Load model checkpoint and tokenizer"""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Get model name from checkpoint
    if "config" in ckpt and "model_name" in ckpt["config"]:
        base = ckpt["config"]["model_name"]
    else:
        base = "distilbert-base-uncased"
    
    print(f"Using base model: {base}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1)
    
    # Load state dict
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        print("Warning: Could not find state_dict in checkpoint")
    
    model.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(base)
    return model, tokenizer

def get_sample(df: pd.DataFrame, sample: int, seed: int = 42) -> pd.DataFrame:
    """Get a balanced sample of texts for analysis"""
    try:
        pos = df[df["target"] >= .5].sample(n=sample//2, random_state=seed)
        neg = df[df["target"] < .5].sample(n=sample//2, random_state=seed)
    except:
        print("Warning: Could not create balanced sample, using random sample instead")
        return df.sample(n=min(sample, len(df)), random_state=seed)
    
    return pd.concat([pos, neg]).reset_index(drop=True)

def predict_text(model, tokenizer, text, max_len=192):
    """Get model prediction for a text"""
    # Tokenize text
    inputs = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
    
    return probs.cpu().numpy()[0][0]

def get_token_contributions(model, tokenizer, text, max_len=128):
    """Measure token contributions by occlusion method"""
    # Get baseline prediction
    baseline_pred = predict_text(model, tokenizer, text, max_len)
    
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_len - 2:  # Account for special tokens
        tokens = tokens[:max_len - 2]
    
    # Only analyze up to 50 tokens for efficiency
    if len(tokens) > 50:
        print(f"Limiting analysis to first 50 tokens out of {len(tokens)} total tokens")
        tokens = tokens[:50]
    
    # Process in batches for efficiency
    batch_size = 5  # Process 5 tokens at a time
    contributions = []
    
    for batch_start in range(0, len(tokens), batch_size):
        batch_end = min(batch_start + batch_size, len(tokens))
        batch_tokens = tokens[batch_start:batch_end]
        batch_contributions = []
        
        for i, token in enumerate(batch_tokens):
            # Copy tokens and mask one
            masked_tokens = tokens.copy()
            masked_tokens[batch_start + i] = tokenizer.mask_token
            
            # Convert back to text
            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
            
            # Get prediction
            masked_pred = predict_text(model, tokenizer, masked_text, max_len)
            
            # Calculate contribution
            contribution = baseline_pred - masked_pred
            batch_contributions.append(contribution)
        
        contributions.extend(batch_contributions)
        
        # Print progress
        print(f"Processed {batch_end}/{len(tokens)} tokens", end="\r")
    
    print(f"Completed token analysis for {len(tokens)} tokens       ")
    
    return tokens, contributions, baseline_pred

def visualize_contributions(tokens, contributions, baseline_pred, out_path, title):
    """Create visualization of token contributions"""
    # Sort tokens by contribution
    sorted_idx = np.argsort(contributions)
    
    # Get top positive and negative contributions
    n_top = min(10, len(tokens))
    top_positive_idx = sorted_idx[-n_top:][::-1]
    top_negative_idx = sorted_idx[:n_top]
    
    # Create figure for positive contributions
    plt.figure(figsize=(10, 6))
    
    # Plot positive contributions
    pos_tokens = [tokens[i] for i in top_positive_idx]
    pos_contributions = [contributions[i] for i in top_positive_idx]
    
    plt.barh(range(len(pos_tokens)), pos_contributions, color='#cc0000')
    plt.yticks(range(len(pos_tokens)), pos_tokens)
    plt.title(f"Top Tokens Increasing Toxicity\n{title}\nBase prediction: {baseline_pred:.4f}")
    plt.xlabel("Contribution to Toxicity Score")
    plt.tight_layout()
    plt.savefig(f"{out_path}_positive.png", dpi=300)
    plt.close()
    
    # Create figure for negative contributions
    plt.figure(figsize=(10, 6))
    
    # Plot negative contributions
    neg_tokens = [tokens[i] for i in top_negative_idx]
    neg_contributions = [contributions[i] for i in top_negative_idx]
    
    plt.barh(range(len(neg_tokens)), neg_contributions, color='#006600')
    plt.yticks(range(len(neg_tokens)), neg_tokens)
    plt.title(f"Top Tokens Decreasing Toxicity\n{title}\nBase prediction: {baseline_pred:.4f}")
    plt.xlabel("Contribution to Toxicity Score")
    plt.tight_layout()
    plt.savefig(f"{out_path}_negative.png", dpi=300)
    plt.close()
    
    # Create word importance overlay for complete text
    plt.figure(figsize=(12, 4))
    
    # Normalize contributions to [-1, 1]
    max_abs_contrib = max(abs(min(contributions)), abs(max(contributions)))
    if max_abs_contrib > 0:
        norm_contributions = [c / max_abs_contrib for c in contributions]
    else:
        norm_contributions = contributions
    
    # Create heatmap-like visualization
    sns.heatmap(
        np.array([norm_contributions]), 
        cmap='coolwarm', 
        center=0,
        cbar=True,
        xticklabels=tokens,
        yticklabels=False
    )
    plt.title(f"Token Importance Heatmap\n{title}\nBase prediction: {baseline_pred:.4f}")
    plt.tight_layout()
    plt.savefig(f"{out_path}_heatmap.png", dpi=300)
    plt.close()
    
    # Save token contributions to CSV
    df = pd.DataFrame({
        'token': tokens,
        'contribution': contributions
    }).sort_values('contribution', ascending=False)
    
    df.to_csv(f"{out_path}_contributions.csv", index=False)
    
    # Save results as text file
    with open(f"{out_path}_summary.txt", "w", encoding="utf-8") as f:
        # Just join tokens with spaces instead of using tokenizer
        f.write(f"Text: {' '.join(tokens)}\n")
        f.write(f"Prediction: {baseline_pred:.4f}\n\n")
        
        f.write("Top tokens increasing toxicity:\n")
        for i, idx in enumerate(top_positive_idx):
            f.write(f"{i+1}. {tokens[idx]}: {contributions[idx]:.4f}\n")
        
        f.write("\nTop tokens decreasing toxicity:\n")
        for i, idx in enumerate(top_negative_idx):
            f.write(f"{i+1}. {tokens[idx]}: {contributions[idx]:.4f}\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate token attributions for toxicity models")
    parser.add_argument("--ckpt", default="output/checkpoints/distilbert_headtail_fold0.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--data", default="data/valid.csv",
                       help="Path to data CSV file")
    parser.add_argument("--out-dir", default="output/attributions",
                       help="Output directory")
    parser.add_argument("--sample", type=int, default=10,
                       help="Number of examples to sample")
    parser.add_argument("--max-len", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--text-col", default="comment_text",
                       help="Column name for text data")
    parser.add_argument("--target-col", default="target",
                       help="Column name for target data")
    args = parser.parse_args()

    # Create output directory
    out_d = pathlib.Path(args.out_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {out_d}")

    # Load model and tokenizer
    model, tokenizer = load_checkpoint(pathlib.Path(args.ckpt))
    model.eval()
    print(f"Model loaded successfully. Using device: {DEVICE}")

    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} examples")

    # Get sample
    df_sample = get_sample(df, args.sample)
    print(f"Sampled {len(df_sample)} examples")

    # Analyze each example
    results = []
    for i, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Analyzing examples"):
        text = row[args.text_col]
        target = row[args.target_col] if args.target_col in row else None
        
        # Create output dir for this example
        example_dir = out_d / f"example_{i+1}"
        example_dir.mkdir(exist_ok=True)
        
        # Get token contributions
        tokens, contributions, prediction = get_token_contributions(
            model, tokenizer, text, args.max_len
        )
        
        # Create visualizations
        example_title = f"Example {i+1}" + (f" (True Target: {target:.4f})" if target is not None else "")
        visualize_contributions(
            tokens, contributions, prediction, 
            str(example_dir / f"example_{i+1}"), 
            example_title
        )
        
        # Store results
        results.append({
            "id": i,
            "text": text,
            "prediction": float(prediction),
            "target": float(target) if target is not None else None
        })
    
    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_d / "results.csv", index=False)
    
    # Create summary markdown file
    with open(out_d / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Toxicity Attribution Analysis\n\n")
        f.write(f"Model: {args.ckpt}\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Sample size: {len(results)}\n\n")
        
        f.write("## Examples\n\n")
        for i, res in enumerate(results):
            text = res["text"]
            if len(text) > 100:
                text = text[:100] + "..."
                
            pred = res["prediction"]
            target = res["target"]
            
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Text**: {text}\n\n")
            f.write(f"**Prediction**: {pred:.4f}")
            if target is not None:
                f.write(f" (True: {target:.4f})")
            f.write("\n\n")
            
            f.write("**Visualizations**:\n")
            f.write(f"- [Positive Contributions](example_{i+1}/example_{i+1}_positive.png)\n")
            f.write(f"- [Negative Contributions](example_{i+1}/example_{i+1}_negative.png)\n")
            f.write(f"- [Token Heatmap](example_{i+1}/example_{i+1}_heatmap.png)\n")
            f.write(f"- [Detailed Analysis](example_{i+1}/example_{i+1}_summary.txt)\n\n")
    
    print(f"✅ Attribution analysis complete. Results saved to {out_d}")

if __name__ == "__main__":
    main() 