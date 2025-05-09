#!/usr/bin/env python
"""
Simplified SHAP explainer for distilbert toxicity model
"""
import argparse
import pathlib
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
def get_sample(df: pd.DataFrame, sample: int, seed: int = 42) -> pd.DataFrame:
    """Get a balanced sample of texts for SHAP analysis"""
    # Stratified 50-50 toxic / non-toxic sample for SHAP stability
    try:
        pos = df[df["target"] >= .5].sample(n=sample//2, random_state=seed)
        neg = df[df["target"] < .5].sample(n=sample//2, random_state=seed)
    except:
        print("Warning: Could not create balanced sample, using random sample instead")
        return df.sample(n=min(sample, len(df)), random_state=seed)
    
    return pd.concat([pos, neg]).reset_index(drop=True)

# ---------------------------------------------------------------------
def build_explainer(model, tokenizer, max_len=192):
    """Build SHAP explainer for transformer model"""
    def predictor(texts):
        # Handle different types of input that SHAP might pass
        if isinstance(texts, str):
            # Single text input
            batch_texts = [texts]
        elif isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            # List of text inputs
            batch_texts = texts
        else:
            # Convert to string if needed
            if hasattr(texts, 'tolist'):  # For numpy arrays
                batch_texts = [str(x) for x in texts.tolist()]
            else:
                batch_texts = [str(x) for x in texts]
        
        # Tokenize and get model output
        enc = tokenizer(batch_texts,
                        max_length=max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt").to(DEVICE)
        return model(**enc).logits
    
    masker = shap.maskers.Text(tokenizer)
    return shap.Explainer(predictor, masker, output_names=["toxicity"])

# ---------------------------------------------------------------------
def save_example_analysis(shap_values, idx, out_dir, prefix, title):
    """Save analysis for a single example"""
    # Get the example text
    text = shap_values.data[idx]
    
    # Calculate the prediction (base value + sum of SHAP values)
    base_value = shap_values.base_values[idx]
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]  # Get the first (and only) output
    
    # Get SHAP values for this example
    shap_value = shap_values.values[idx]
    shap_sum = np.sum(shap_value)
    
    # Calculate final prediction
    prediction = base_value + shap_sum
    
    # Create waterfall plot - handle the case where we need to index correctly
    plt.figure(figsize=(12, 8))
    try:
        # Try regular way first
        shap.plots.waterfall(shap_values[idx], max_display=20, show=False)
    except ValueError as e:
        print(f"Adjusting waterfall plot indexing: {str(e)}")
        # If we got a matrix, try to get first element
        if len(shap_values.values[idx].shape) > 1:
            # Handle multi-output models
            waterfall_values = shap.Explanation(
                values=shap_values.values[idx, 0] if len(shap_values.values[idx].shape) > 1 else shap_values.values[idx],
                base_values=base_value,
                data=text,
                feature_names=shap_values.feature_names
            )
            shap.plots.waterfall(waterfall_values, max_display=20, show=False)
        else:
            # Create a simplified explanation
            waterfall_values = shap.Explanation(
                values=shap_values.values[idx],
                base_values=base_value,
                data=text,
                feature_names=shap_values.feature_names
            )
            shap.plots.waterfall(waterfall_values, max_display=20, show=False)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_waterfall.png", dpi=300)
    plt.close()
    
    # Create text importance visualization (fallback if waterfall fails)
    plt.figure(figsize=(12, 8))
    
    # Get absolute importance for each token
    token_importances = np.abs(shap_value)
    
    # Get tokens from text
    tokens = []
    for i, name in enumerate(shap_values.feature_names):
        if i < len(token_importances):
            tokens.append((name, token_importances[i]))
    
    # Sort tokens by importance
    sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)[:20]
    
    # Plot bar chart of token importances
    token_names = [t[0] for t in sorted_tokens]
    token_values = [t[1] for t in sorted_tokens]
    
    plt.barh(range(len(token_names)), token_values)
    plt.yticks(range(len(token_names)), token_names)
    plt.title(f"Top Token Importances for {prefix.capitalize()} Example")
    plt.xlabel("Token Importance (|SHAP|)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_token_importance.png", dpi=300)
    plt.close()
    
    # Save example text and prediction
    with open(out_dir / f"{prefix}_example.txt", "w", encoding="utf-8") as f:
        f.write(f"Text: {text}\n")
        f.write(f"Base value: {base_value:.4f}\n")
        f.write(f"SHAP sum: {shap_sum:.4f}\n")
        f.write(f"Prediction: {prediction:.4f}\n")
        f.write("\nTop 20 Tokens by Importance:\n")
        for i, (token, importance) in enumerate(sorted_tokens):
            f.write(f"{i+1}. {token}: {importance:.4f}\n")
    
    return {
        "text": text,
        "base_value": float(base_value),
        "shap_sum": float(shap_sum),
        "prediction": float(prediction),
        "top_tokens": [{
            "token": token,
            "importance": float(importance)
        } for token, importance in sorted_tokens[:10]]
    }

# ---------------------------------------------------------------------
def main():
    """Main function to run SHAP analysis"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run SHAP analysis on toxicity model")
    parser.add_argument("--ckpt", default="output/checkpoints/distilbert_headtail_fold0.pth",
                      help="Path to model checkpoint")
    parser.add_argument("--valid-csv", default="data/valid.csv",
                      help="Path to validation CSV file")
    parser.add_argument("--sample", type=int, default=20,
                      help="Number of examples to sample for SHAP analysis")
    parser.add_argument("--out-dir", default="output/explainers",
                      help="Output directory for SHAP visualizations")
    parser.add_argument("--max-len", type=int, default=192,
                      help="Maximum sequence length for tokenizer")
    parser.add_argument("--text-col", default="comment_text",
                      help="Column name containing the text")
    parser.add_argument("--target-col", default="target",
                      help="Column name containing the target")
    parser.add_argument("--num-examples", type=int, default=3,
                      help="Number of examples to generate visualizations for")
    args = parser.parse_args()

    # Create output directory
    out_d = pathlib.Path(args.out_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to {out_d}")

    # Load model and tokenizer
    model, tokenizer = load_checkpoint(pathlib.Path(args.ckpt))
    print(f"Model loaded successfully. Using device: {DEVICE}")

    # Load and sample validation data
    print(f"Loading validation data from {args.valid_csv}")
    df_v = pd.read_csv(args.valid_csv)
    print(f"Loaded {len(df_v)} examples")
    
    df_s = get_sample(df_v, args.sample)
    print(f"Sampled {len(df_s)} examples for SHAP analysis")
    
    texts = df_s[args.text_col].astype(str).fillna("").tolist()
    print(f"Using text column: {args.text_col}")

    # Build explainer
    print("Building SHAP explainer...")
    explainer = build_explainer(model, tokenizer, max_len=args.max_len)

    # Run SHAP analysis
    print(f"Computing SHAP values for {len(texts)} examples (this may take several minutes)...")
    try:
        shap_values = explainer(texts, silent=True)
        print("SHAP analysis completed successfully")
    except Exception as e:
        print(f"Error during SHAP computation: {str(e)}")
        print("Trying with a smaller batch...")
        small_sample = texts[:min(10, len(texts))]
        shap_values = explainer(small_sample, silent=True)
        print(f"SHAP analysis completed with reduced sample size ({len(small_sample)} examples)")

    # Save raw SHAP values
    print("Saving SHAP values...")
    np.savez_compressed(out_d/"shap_values.npz",
                      values=shap_values.values,
                      base_values=shap_values.base_values)
    
    # Calculate predictions for each example
    predictions = []
    for i in range(len(shap_values)):
        base_value = shap_values.base_values[i]
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        shap_value = shap_values.values[i]
        shap_sum = np.sum(shap_value)
        pred = base_value + shap_sum
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Create analysis directories
    examples_dir = out_d / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Find the most toxic and least toxic examples
    num_examples = min(args.num_examples, len(predictions))
    most_toxic_indices = np.argsort(predictions)[-num_examples:][::-1]
    least_toxic_indices = np.argsort(predictions)[:num_examples]
    
    # Analyze and save results
    results = {
        "model": str(args.ckpt),
        "sample_size": len(shap_values),
        "most_toxic": [],
        "least_toxic": []
    }
    
    # Analyze most toxic examples
    print(f"Analyzing top {num_examples} most toxic examples...")
    for i, idx in enumerate(most_toxic_indices):
        toxic_dir = examples_dir / f"toxic_{i+1}"
        toxic_dir.mkdir(exist_ok=True)
        example_data = save_example_analysis(
            shap_values, 
            idx, 
            toxic_dir, 
            "toxic", 
            f"Token Contributions for Toxic Example {i+1} (Prediction: {predictions[idx]:.4f})"
        )
        results["most_toxic"].append(example_data)
    
    # Analyze least toxic examples
    print(f"Analyzing top {num_examples} least toxic examples...")
    for i, idx in enumerate(least_toxic_indices):
        nontoxic_dir = examples_dir / f"nontoxic_{i+1}"
        nontoxic_dir.mkdir(exist_ok=True)
        example_data = save_example_analysis(
            shap_values, 
            idx, 
            nontoxic_dir, 
            "nontoxic", 
            f"Token Contributions for Non-Toxic Example {i+1} (Prediction: {predictions[idx]:.4f})"
        )
        results["least_toxic"].append(example_data)
    
    # Save results as JSON
    import json
    with open(out_d / "shap_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Create a summary report
    print("Creating summary report...")
    with open(out_d / "shap_summary.md", "w", encoding="utf-8") as f:
        f.write("# SHAP Analysis Summary\n\n")
        f.write(f"Model: {args.ckpt}\n")
        f.write(f"Sample size: {len(shap_values)}\n\n")
        
        f.write("## Most Influential Tokens\n\n")
        # Get top tokens by mean absolute SHAP value
        token_importance = np.abs(shap_values.values).mean(axis=0)
        top_tokens = np.argsort(token_importance)[-20:][::-1]
        
        f.write("| Token | Mean |SHAP| |\n")
        f.write("|-------|-------------|\n")
        
        for token_idx in top_tokens:
            try:
                if token_idx < len(shap_values.feature_names):
                    token = shap_values.feature_names[token_idx]
                    importance = token_importance[token_idx]
                    f.write(f"| {token} | {importance:.4f} |\n")
            except:
                pass
        
        f.write("\n## Most Toxic Examples\n\n")
        for i, idx in enumerate(most_toxic_indices):
            pred = predictions[idx]
            text = shap_values.data[idx]
            if len(text) > 100:
                text = text[:100] + "..."
            f.write(f"{i+1}. **Prediction: {pred:.4f}** - {text}\n")
            f.write(f"   - [Detailed Analysis](examples/toxic_{i+1}/toxic_waterfall.png)\n\n")
        
        f.write("\n## Least Toxic Examples\n\n")
        for i, idx in enumerate(least_toxic_indices):
            pred = predictions[idx]
            text = shap_values.data[idx]
            if len(text) > 100:
                text = text[:100] + "..."
            f.write(f"{i+1}. **Prediction: {pred:.4f}** - {text}\n")
            f.write(f"   - [Detailed Analysis](examples/nontoxic_{i+1}/nontoxic_waterfall.png)\n\n")

    print(f"✅ SHAP analysis complete. Results saved to {out_d}")

if __name__ == "__main__":
    main() 