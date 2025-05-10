#!/usr/bin/env python
"""
SHarP: SHAP-based Fairness Analysis
==================================
This script implements SHarP (SHAP-based Fairness) analysis, which measures how model reasoning
differs across demographic groups using SHAP attribution patterns.

The analysis:
1. Loads a trained model and computes SHAP values for a sample of validation examples
2. Groups examples by demographic attributes (e.g., gender, race, religion)
3. Computes the "divergence" in attribution patterns between each subgroup and the global population
4. Identifies which groups have the most different attribution patterns (potential fairness concerns)
5. Visualizes the divergence scores and generates detailed reports

Usage:
    python fairness_analysis/run_sharp_analysis.py --sample 2000
    python fairness_analysis/run_sharp_analysis.py --model-path output/checkpoints/your_model.pth
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bert_headtail import BertHeadTailForSequenceClassification
from src.data.loaders import load_tokenizer

# Set up SHAP
try:
    import shap

    shap.initjs()
except ImportError:
    print("SHAP not installed. Please install with: pip install shap")
    sys.exit(1)

# Identity columns for analysis
IDENTITY_COLUMNS = [
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
]

# Prettier display names for identity columns
IDENTITY_DISPLAY_NAMES = {
    "male": "Male",
    "female": "Female",
    "transgender": "Transgender",
    "other_gender": "Other Gender",
    "heterosexual": "Heterosexual",
    "homosexual_gay_or_lesbian": "Gay/Lesbian",
    "bisexual": "Bisexual",
    "other_sexual_orientation": "Other Sexual Orientation",
    "christian": "Christian",
    "jewish": "Jewish",
    "muslim": "Muslim",
    "hindu": "Hindu",
    "buddhist": "Buddhist",
    "atheist": "Atheist",
    "other_religion": "Other Religion",
    "black": "Black",
    "white": "White",
    "asian": "Asian",
    "latino": "Latino/Hispanic",
    "other_race_or_ethnicity": "Other Race/Ethnicity",
    "physical_disability": "Physical Disability",
    "intellectual_or_learning_disability": "Intellectual Disability",
    "psychiatric_or_mental_illness": "Mental Illness",
    "other_disability": "Other Disability",
}


def load_data(data_path, n_samples=None):
    """Load data for analysis"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Filter to get only rows with some identity markers
    identity_df = df[df[IDENTITY_COLUMNS].sum(axis=1) > 0].copy()

    if n_samples and n_samples < len(identity_df):
        print(f"Sampling {n_samples} examples with identity mentions")
        identity_df = identity_df.sample(n_samples, random_state=42)

    print(f"Selected {len(identity_df)} examples with identity mentions")
    return identity_df


def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint.get('config', {})
    
    # Create the model
    model = BertHeadTailForSequenceClassification(
        model_name=config.get('bert_model', 'distilbert-base-uncased'),
        num_labels=1
    )
    
    # Load the weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    return model, config


def compute_shap_values(model, tokenizer, texts, max_length=128):
    """Compute SHAP values for the given texts"""

    # Create a function that preprocesses and predicts toxicity
    def f(x):
        # Tokenize
        inputs = tokenizer(
            list(x),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.detach().numpy()

    # Initialize the explainer
    explainer = shap.Explainer(f, tokenizer)

    # Compute SHAP values in batches
    batch_size = 100
    all_shap_values = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        all_shap_values.append(explainer(batch_texts))
    shap_values = np.concatenate(all_shap_values, axis=0)

    # Print a quick sanity check
    print(f"[SHAP] computed values shape: {shap_values.shape}")

    return shap_values


def compute_group_divergence(df, shap_values, column, label="toxicity"):
    """Compute divergence between a group's attributions and the global distribution"""
    # Get the subgroup with the specified attribute
    subgroup = df[df[column] > 0]

    if len(subgroup) < 10:
        print(f"Warning: Subgroup '{column}' has less than 10 examples, skipping")
        return None, None

    # Get indices of the subgroup examples
    subgroup_indices = subgroup.index.tolist()

    # Convert to indices in the original dataframe
    subgroup_indices = [df.index.get_loc(idx) for idx in subgroup_indices]

    # Extract SHAP values for the positive class (toxicity)
    subgroup_shap = np.array([sv.values[:, 1] for sv in shap_values[subgroup_indices]])
    all_shap = np.array([sv.values[:, 1] for sv in shap_values])

    # Get mean attribution vectors
    mean_subgroup_shap = np.mean(subgroup_shap, axis=0)
    mean_all_shap = np.mean(all_shap, axis=0)

    # Calculate cosine similarity
    similarity = cosine_similarity([mean_subgroup_shap], [mean_all_shap])[0][0]

    # Calculate divergence (1 - similarity)
    divergence = 1 - similarity

    return divergence, len(subgroup)


def plot_divergence_scores(divergence_scores, group_sizes, output_dir):
    """Plot a bar chart of divergence scores and save to file"""
    # Create a dataframe with divergence scores and group sizes
    df = pd.DataFrame(
        {
            "Group": [
                IDENTITY_DISPLAY_NAMES.get(k, k) for k in divergence_scores.keys()
            ],
            "Divergence": list(divergence_scores.values()),
            "Group Size": list(group_sizes.values()),
        }
    )

    # Sort by divergence
    df = df.sort_values("Divergence", ascending=False)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create a colormap based on divergence
    colors = plt.cm.RdYlGn_r(df["Divergence"] / df["Divergence"].max())

    # Plot
    ax = sns.barplot(x="Divergence", y="Group", data=df, palette=colors)

    # Add group sizes as text
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["Divergence"] + 0.01, i, f"n={row['Group Size']}", va="center")

    # Set labels and title
    plt.xlabel("SHarP Divergence Score", fontsize=12)
    plt.ylabel("Demographic Group", fontsize=12)
    plt.title(
        "SHarP Divergence: How Different is Model Reasoning for Each Group?",
        fontsize=14,
    )

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Potential Fairness Concern", fontsize=12)

    # Add explanatory text
    plt.figtext(
        0.5,
        0.01,
        "Higher values indicate more different reasoning patterns compared to the overall population.\n"
        "This may indicate potential fairness concerns if the model uses different logic for certain groups.",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
    )

    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, "sharp_divergence_scores.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved divergence plot to {output_path}")

    # Also save as CSV
    csv_path = os.path.join(output_dir, "sharp_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved divergence scores to {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run SHarP (SHAP-based Fairness) analysis"
    )
    parser.add_argument(
        "--model-path",
        default="output/checkpoints/distilbert_headtail_fold0.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--data", default="data/train.csv", help="Path to data CSV")
    parser.add_argument(
        "--output-dir", default="output/explainability", help="Output directory"
    )
    parser.add_argument(
        "--sample", type=int, default=500, help="Number of examples to analyze"
    )
    parser.add_argument(
        "--sample-size", type=int, default=1000,
        help="Number of rows to sample from merged_val.csv for SHAP"
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # ensure csv exists
    merg = Path("output/data/merged_val.csv")
    if not merg.exists():
        raise FileNotFoundError("Run `make merge-preds` first – merged_val.csv missing")

    # guarantee output dirs
    OUT_FIGS = Path("figs/shap")
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_CSV = Path("results")
    OUT_CSV.mkdir(parents=True, exist_ok=True)

    bar_path = OUT_FIGS / "shap_bar_distilbert_dev.png"
    div_path = OUT_FIGS / "sharp_divergence_distilbert.png"
    csv_path = OUT_CSV / "sharp_scores_distilbert_dev.csv"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(merg)
    df = df.sample(n=args.sample_size, random_state=42)

    # Load model and tokenizer
    model, config = load_model(args.model_path)
    tokenizer = load_tokenizer(config.get("bert_model", "distilbert-base-uncased"))

    # Use GPU if requested and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get texts for analysis
    texts = df["comment_text"].tolist()

    # Compute SHAP values
    print(f"Computing SHAP values for {len(texts)} examples...")
    shap_values = compute_shap_values(model, tokenizer, texts, args.max_length)

    # Compute divergence for each identity group
    print("Computing SHarP divergence scores...")
    divergence_scores = {}
    group_sizes = {}

    for column in tqdm(IDENTITY_COLUMNS):
        divergence, size = compute_group_divergence(df, shap_values, column)
        if divergence is not None:
            divergence_scores[column] = divergence
            group_sizes[column] = size

    # Plot and save divergence scores
    results_df = plot_divergence_scores(divergence_scores, group_sizes, OUT_FIGS)
    
    # Also save to CSV output
    results_df.to_csv(csv_path, index=False)
    print(f"Saved divergence scores to {csv_path}")

    # Plot and save divergence scores in the original output dir
    plot_divergence_scores(divergence_scores, group_sizes, args.output_dir)

    # Print summary
    print("\nSHarP Analysis Results:")
    print("-" * 50)
    print("Top 5 Groups with Highest Divergence:")
    for _, row in results_df.head(5).iterrows():
        print(f"  {row['Group']}: {row['Divergence']:.4f} (n={row['Group Size']})")

    print(
        "\nComplete! SHarP divergence scores measure how differently the model 'reasons'"
    )
    print(
        "(via feature attributions) for each demographic group compared to the overall population."
    )
    print(f"Results saved to {args.output_dir} and {OUT_FIGS}")


if __name__ == "__main__":
    main()
