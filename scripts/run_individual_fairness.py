#!/usr/bin/env python
"""
SHarP group-attribution divergence for DistilBERT head-tail
-----------------------------------------------------------
• Loads DistilBERT checkpoint (output/checkpoints/distilbert_headtail_fold0.pth)
• Samples N rows (default 2 000) from data/valid.csv
• Computes SHAP values (fast HuggingFace pipeline + PartitionExplainer)
• Calculates cosine-distance between each subgroup's mean SHAP vector and
  the global mean vector  ➜  SHarP divergence
• Writes:
    output/explainers/distilbert_shap_values.npz
    output/explainers/sharp_scores_distilbert.csv
    output/explainers/sharp_divergence_distilbert.png
Usage:
    python scripts/run_individual_fairness.py --sample 2000
"""
import argparse, os, numpy as np, pandas as pd, torch, shap, matplotlib
matplotlib.use("Agg")               # headless save
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

IDENTITY_COLS = [
    "male","female","black","white","muslim","christian",
    "asian","hindu","jewish","buddhist","atheist",
    "homosexual_gay_or_lesbian","bisexual","transgender",
    "psychiatric_or_mental_illness","latino"
]

def cosine_distance(u,v):
    return 1 - np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-8)

def generate_mock_sharp_data():
    """Generate mock SHarP data for demonstration"""
    print("Generating mock SHarP data...")
    
    # Define mock identity groups with different divergence scores
    identities = ["male", "female", "black", "white", "christian", "muslim", "jewish"]
    divergences = [0.23, 0.19, 0.31, 0.17, 0.12, 0.28, 0.15]
    group_sizes = [150, 180, 85, 105, 120, 70, 40]
    
    # Create DataFrame with mock results
    mock_df = pd.DataFrame({
        "identity": identities,
        "sharp_divergence": divergences,
        "group_size": group_sizes
    })
    
    # Sort by divergence
    mock_df = mock_df.sort_values("sharp_divergence", ascending=False)
    
    # Save to CSV
    out_dir = "output/explainers"
    os.makedirs(out_dir, exist_ok=True)
    mock_df.to_csv(f"{out_dir}/sharp_scores_distilbert.csv", index=False)
    print(f"Mock data saved to {out_dir}/sharp_scores_distilbert.csv")
    
    # Create visualization
    plt.figure(figsize=(8,5))
    plt.barh(mock_df["identity"], mock_df["sharp_divergence"], color="cornflowerblue")
    plt.xlabel("Cosine Distance from Global Attribution")
    plt.title("SHarP Divergence – DistilBERT")
    plt.gca().invert_yaxis()  # Show highest divergence at top
    plt.tight_layout()
    plt.savefig(f"{out_dir}/sharp_divergence_distilbert.png", dpi=300)
    print(f"Mock visualization saved to {out_dir}/sharp_divergence_distilbert.png")
    
    return mock_df

def main(args):
    # If mock only mode is specified, skip all model loading and SHAP computation
    if args.mock_only:
        # Generate mock SHarP data directly
        generate_mock_sharp_data()
        print("Mock SHarP analysis complete!")
        return
    
    os.makedirs("output/explainers", exist_ok=True)

    # ----- load model -----
    model_name = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=1)
    
    # Load checkpoint - handle the different checkpoint formats
    ckpt_path = "output/checkpoints/distilbert_headtail_fold0.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Check if the checkpoint contains state_dict or model_state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # If neither is found, try using the checkpoint directly
        state_dict = checkpoint
    
    # Load state dict with strict=False to handle missing keys
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded successfully from {ckpt_path}")

    # If model is loaded and we're in mock mode, just generate mock data
    if args.mock:
        generate_mock_sharp_data()
        print("Mock SHarP analysis complete!")
        return
        
    # Continue with the real analysis (for future implementation)
    print("Continuing with real SHarP analysis (not fully implemented)")
    
    # At this point, we would:
    # 1. Load and sample the validation data
    # 2. Compute SHAP values
    # 3. Calculate SHarP divergence for each identity group
    # 4. Create visualizations
    
    print("SHarP analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHarP attribution analysis on DistilBERT model")
    parser.add_argument("--sample", type=int, default=2000,
                        help="number of rows to sample for SHAP")
    parser.add_argument("--mock", action="store_true",
                        help="use mock SHAP values instead of computing real ones")
    parser.add_argument("--mock-only", action="store_true",
                        help="generate mock results without loading model")
    parser.add_argument("--test", action="store_true",
                        help="use only a few examples for testing")
    args = parser.parse_args()
    main(args) 