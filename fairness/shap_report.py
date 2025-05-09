#!/usr/bin/env python
"""
SHAP Analysis Report for Fairness Integration
This module analyzes SHAP explanations to identify potential fairness issues
"""
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Local imports - adjust paths as needed
import sys
sys.path.append(".")
from fairness.metrics_v2 import list_identity_columns

class SHAPFairnessAnalyzer:
    """Analyze SHAP explanations from a fairness perspective"""
    
    def __init__(self, shap_dir, data_path, output_dir=None):
        """
        Initialize the analyzer
        
        Args:
            shap_dir: Directory containing SHAP analysis results
            data_path: Path to the data file with identity attributes
            output_dir: Directory to save reports (defaults to shap_dir/fairness)
        """
        self.shap_dir = pathlib.Path(shap_dir)
        self.data_path = pathlib.Path(data_path)
        
        if output_dir is None:
            self.output_dir = self.shap_dir / "fairness"
        else:
            self.output_dir = pathlib.Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data and SHAP results
        self._load_data()
        self._load_shap_results()
    
    def _load_data(self):
        """Load data with identity attributes"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} examples")
        
        # Get identity columns
        self.identity_cols = list_identity_columns(self.data)
        print(f"Found {len(self.identity_cols)} identity columns")
    
    def _load_shap_results(self):
        """Load SHAP analysis results"""
        # Try to load JSON results first
        json_path = self.shap_dir / "shap_results.json"
        if json_path.exists():
            print(f"Loading SHAP results from {json_path}...")
            with open(json_path, "r") as f:
                self.shap_results = json.load(f)
        else:
            print(f"SHAP results JSON not found at {json_path}")
            self.shap_results = None
        
        # Try to load NPZ file with raw SHAP values
        npz_path = self.shap_dir / "shap_values.npz"
        if npz_path.exists():
            print(f"Loading raw SHAP values from {npz_path}...")
            self.shap_values = np.load(npz_path)
            print(f"Loaded SHAP values with shapes: {[k + ': ' + str(v.shape) for k, v in self.shap_values.items()]}")
        else:
            print(f"Raw SHAP values not found at {npz_path}")
            self.shap_values = None
        
        # Load token attribution CSV files from example directories
        self.attributions = []
        for example_dir in self.shap_dir.glob("example_*"):
            if example_dir.is_dir():
                contrib_files = list(example_dir.glob("*_contributions.csv"))
                if contrib_files:
                    for contrib_file in contrib_files:
                        try:
                            contrib_df = pd.read_csv(contrib_file)
                            example_id = int(example_dir.name.split("_")[1])
                            self.attributions.append({
                                "example_id": example_id,
                                "file": contrib_file,
                                "attributions": contrib_df
                            })
                        except Exception as e:
                            print(f"Error loading {contrib_file}: {e}")
        
        print(f"Loaded {len(self.attributions)} attribution files")
    
    def analyze_identity_terms(self):
        """Analyze how identity-related terms affect predictions"""
        if not self.attributions:
            print("No attribution data available for analysis")
            return
        
        # Create a mapping of identity group to related terms
        identity_terms = {
            "gender": ["man", "woman", "male", "female", "he", "she", "his", "her", "him", "men", "women"],
            "sexual_orientation": ["gay", "lesbian", "homosexual", "bisexual", "lgbt", "queer"],
            "race": ["black", "white", "african", "caucasian", "asian", "hispanic", "latino", "latina"],
            "religion": ["muslim", "islam", "christian", "christianity", "jewish", "jew", "hindu", "buddhist", "catholic"],
            "disability": ["disabled", "disability", "handicap", "mental", "psychiatric", "illness", "disorder"]
        }
        
        # Analyze each attribution file
        all_term_impacts = []
        
        for attribution in self.attributions:
            df = attribution["attributions"]
            
            # Convert tokens to lowercase for matching
            df["token_lower"] = df["token"].str.lower()
            
            # Check each identity category
            for category, terms in identity_terms.items():
                # Find terms in this attribution
                for term in terms:
                    matches = df[df["token_lower"] == term]
                    if not matches.empty:
                        for _, row in matches.iterrows():
                            all_term_impacts.append({
                                "example_id": attribution["example_id"],
                                "category": category,
                                "term": row["token"],
                                "contribution": row["contribution"],
                                "normalized_contribution": row["contribution"] / abs(df["contribution"]).max()
                            })
        
        # Create a DataFrame with the results
        if all_term_impacts:
            impact_df = pd.DataFrame(all_term_impacts)
            
            # Save to CSV
            impact_df.to_csv(self.output_dir / "identity_term_impacts.csv", index=False)
            
            # Create summary by category
            category_summary = impact_df.groupby("category").agg({
                "contribution": ["mean", "std", "count"],
                "normalized_contribution": ["mean", "std"]
            }).reset_index()
            
            category_summary.columns = ["category", "mean_contribution", "std_contribution", 
                                        "count", "mean_norm_contribution", "std_norm_contribution"]
            
            category_summary.to_csv(self.output_dir / "identity_category_summary.csv", index=False)
            
            # Create visualizations
            self._visualize_identity_impacts(impact_df, category_summary)
            
            return impact_df, category_summary
        else:
            print("No identity terms found in attributions")
            return None, None
    
    def _visualize_identity_impacts(self, impact_df, category_summary):
        """Create visualizations of identity term impacts"""
        # Bar plot of average contribution by category
        plt.figure(figsize=(10, 6))
        sns.barplot(data=category_summary, x="category", y="mean_contribution",
                   yerr=category_summary["std_contribution"], capsize=0.2)
        plt.title("Average Impact of Identity Terms by Category")
        plt.xlabel("Identity Category")
        plt.ylabel("Average Contribution to Toxicity Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "identity_category_impact.png", dpi=300)
        plt.close()
        
        # Distribution of contributions by term
        plt.figure(figsize=(12, 8))
        # Only plot terms with sufficient occurrences
        term_counts = impact_df["term"].value_counts()
        common_terms = term_counts[term_counts >= 3].index.tolist()
        
        if common_terms:
            term_data = impact_df[impact_df["term"].isin(common_terms)]
            sns.boxplot(data=term_data, x="term", y="contribution")
            plt.title("Distribution of Contributions by Identity Term")
            plt.xlabel("Identity Term")
            plt.ylabel("Contribution to Toxicity Score")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(self.output_dir / "identity_term_distribution.png", dpi=300)
        plt.close()
        
        # Heatmap of average normalized contribution by term and category
        pivot_data = impact_df.pivot_table(
            values="normalized_contribution", 
            index="category", 
            columns="term",
            aggfunc="mean"
        )
        
        if not pivot_data.empty and not pivot_data.dropna().empty:
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot_data, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                        linewidths=0.5, cbar_kws={"label": "Avg. Normalized Contribution"})
            plt.title("Average Impact of Identity Terms on Toxicity Prediction")
            plt.tight_layout()
            plt.savefig(self.output_dir / "identity_term_heatmap.png", dpi=300)
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive SHAP fairness report"""
        # Analyze identity terms
        impact_df, category_summary = self.analyze_identity_terms()
        
        # Create markdown report
        with open(self.output_dir / "shap_fairness_report.md", "w", encoding="utf-8") as f:
            f.write("# SHAP Fairness Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"This report analyzes the impact of identity terms on model predictions based on SHAP explanations.\n")
            f.write(f"Data source: {self.data_path}\n")
            f.write(f"SHAP analysis source: {self.shap_dir}\n\n")
            
            f.write("## Identity Term Impact\n\n")
            
            if impact_df is not None and not impact_df.empty:
                f.write("### Summary by Category\n\n")
                f.write("| Category | Mean Contribution | Std Deviation | Count | Normalized Mean |\n")
                f.write("|----------|------------------|---------------|-------|----------------|\n")
                
                for _, row in category_summary.iterrows():
                    f.write(f"| {row['category']} | {row['mean_contribution']:.4f} | ")
                    f.write(f"{row['std_contribution']:.4f} | {row['count']} | ")
                    f.write(f"{row['mean_norm_contribution']:.4f} |\n")
                
                f.write("\n### Visualizations\n\n")
                f.write("![Identity Category Impact](identity_category_impact.png)\n\n")
                f.write("![Identity Term Distribution](identity_term_distribution.png)\n\n")
                f.write("![Identity Term Heatmap](identity_term_heatmap.png)\n\n")
                
                # Get most biasing terms
                f.write("### Most Biasing Identity Terms\n\n")
                most_positive = impact_df.sort_values("contribution", ascending=False).head(10)
                most_negative = impact_df.sort_values("contribution", ascending=True).head(10)
                
                f.write("**Terms most increasing toxicity prediction:**\n\n")
                f.write("| Term | Category | Contribution |\n")
                f.write("|------|----------|-------------|\n")
                for _, row in most_positive.iterrows():
                    f.write(f"| {row['term']} | {row['category']} | {row['contribution']:.4f} |\n")
                
                f.write("\n**Terms most decreasing toxicity prediction:**\n\n")
                f.write("| Term | Category | Contribution |\n")
                f.write("|------|----------|-------------|\n")
                for _, row in most_negative.iterrows():
                    f.write(f"| {row['term']} | {row['category']} | {row['contribution']:.4f} |\n")
            else:
                f.write("No identity terms found in the analyzed examples.\n")
            
            f.write("\n## Conclusions\n\n")
            f.write("Based on the SHAP analysis, we observe the following patterns in how identity terms affect toxicity prediction:\n\n")
            
            if impact_df is not None and not impact_df.empty:
                # Add some conclusions based on the data
                categories_by_impact = category_summary.sort_values("mean_contribution", ascending=False)
                
                if not categories_by_impact.empty:
                    most_biased_category = categories_by_impact.iloc[0]["category"]
                    least_biased_category = categories_by_impact.iloc[-1]["category"]
                    
                    f.write(f"1. Terms related to **{most_biased_category}** have the strongest effect on increasing toxicity predictions\n")
                    f.write(f"2. Terms related to **{least_biased_category}** have the least effect on toxicity predictions\n")
                    
                    # Check if there are any categories with negative mean
                    neg_categories = categories_by_impact[categories_by_impact["mean_contribution"] < 0]
                    if not neg_categories.empty:
                        neg_category = neg_categories.iloc[0]["category"]
                        f.write(f"3. Terms related to **{neg_category}** tend to decrease toxicity predictions on average\n")
            else:
                f.write("Insufficient data to draw conclusions. More examples with identity terms are needed.\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on this analysis, the following actions are recommended:\n\n")
            f.write("1. Review training data to ensure balanced representation of identity groups\n")
            f.write("2. Consider data augmentation or balancing techniques for underrepresented groups\n")
            f.write("3. Implement specific bias mitigation techniques during model training\n")
            f.write("4. Conduct more comprehensive intersectional analysis with larger samples\n")
        
        print(f"Fairness report generated: {self.output_dir}/shap_fairness_report.md")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate SHAP fairness report")
    parser.add_argument("--shap-dir", default="output/attributions",
                      help="Directory containing SHAP analysis results")
    parser.add_argument("--data", default="data/valid.csv",
                      help="Path to data file with identity columns")
    parser.add_argument("--out-dir", default=None,
                      help="Directory to save fairness report (default: shap_dir/fairness)")
    args = parser.parse_args()
    
    analyzer = SHAPFairnessAnalyzer(args.shap_dir, args.data, args.out_dir)
    analyzer.generate_report()

if __name__ == "__main__":
    main() 