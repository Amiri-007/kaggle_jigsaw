#!/usr/bin/env python
"""
Simplified SHAP Analysis Report for Fairness Integration
This module analyzes token attribution data to identify potential fairness issues
"""
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob

# Local imports - adjust paths as needed
import sys
sys.path.append(".")
from fairness.metrics_v2 import list_identity_columns

class SimplifiedFairnessAnalyzer:
    """Analyze token attributions from a fairness perspective"""
    
    def __init__(self, attribution_dir, data_path, output_dir=None):
        """
        Initialize the analyzer
        
        Args:
            attribution_dir: Directory containing attribution analysis results
            data_path: Path to the data file with identity attributes
            output_dir: Directory to save reports (defaults to attribution_dir/fairness)
        """
        self.attribution_dir = pathlib.Path(attribution_dir)
        self.data_path = pathlib.Path(data_path)
        
        if output_dir is None:
            self.output_dir = self.attribution_dir / "fairness"
        else:
            self.output_dir = pathlib.Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data and attributions
        self._load_data()
        self._load_attributions()
    
    def _load_data(self):
        """Load data with identity attributes"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} examples")
        
        # Get identity columns
        self.identity_cols = list_identity_columns(self.data)
        print(f"Found {len(self.identity_cols)} identity columns")
    
    def _load_attributions(self):
        """Load attribution analysis results"""
        self.attribution_files = []
        
        # Find all contribution CSV files in example directories
        contribution_files = glob.glob(str(self.attribution_dir / "example_*/*_contributions.csv"))
        print(f"Found {len(contribution_files)} attribution files")
        
        # Load each contribution file
        for contrib_file in contribution_files:
            try:
                contrib_df = pd.read_csv(contrib_file)
                example_id = int(pathlib.Path(contrib_file).parent.name.split("_")[1])
                self.attribution_files.append({
                    "example_id": example_id,
                    "file": contrib_file,
                    "attributions": contrib_df
                })
            except Exception as e:
                print(f"Error loading {contrib_file}: {e}")
        
        print(f"Successfully loaded {len(self.attribution_files)} attribution files")
    
    def analyze_identity_terms(self):
        """Analyze how identity-related terms affect predictions"""
        if not self.attribution_files:
            print("No attribution data available for analysis")
            return None, None
        
        # Create a mapping of identity group to related terms
        identity_terms = {
            "gender": ["man", "woman", "male", "female", "he", "she", "his", "her", "him", "men", "women", "gender"],
            "sexual_orientation": ["gay", "lesbian", "homosexual", "bisexual", "lgbt", "queer", "straight"],
            "race": ["black", "white", "african", "caucasian", "asian", "hispanic", "latino", "latina", "race"],
            "religion": ["muslim", "islam", "christian", "christianity", "jewish", "jew", "hindu", "buddhist", "catholic", "religion"],
            "disability": ["disabled", "disability", "handicap", "mental", "psychiatric", "illness", "disorder"]
        }
        
        # Analyze each attribution file
        all_term_impacts = []
        
        for attribution in self.attribution_files:
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
                                "normalized_contribution": row["contribution"] / (abs(df["contribution"]).max() or 1.0)
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
            })
            
            # Flatten the multi-index columns
            category_summary.columns = ["_".join(col).strip() for col in category_summary.columns.values]
            category_summary = category_summary.reset_index()
            
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
        sns.barplot(data=category_summary, x="category", y="contribution_mean")
        plt.title("Average Impact of Identity Terms by Category")
        plt.xlabel("Identity Category")
        plt.ylabel("Average Contribution to Toxicity Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "identity_category_impact.png", dpi=300)
        plt.close()
        
        # Distribution of contributions by term
        term_counts = impact_df["term"].value_counts()
        common_terms = term_counts[term_counts >= 2].index.tolist()
        
        if common_terms:
            plt.figure(figsize=(12, 8))
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
        term_counts = impact_df["term"].value_counts()
        if len(term_counts) > 0:
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
        """Generate a comprehensive fairness report"""
        # Analyze identity terms
        impact_df, category_summary = self.analyze_identity_terms()
        
        # Create markdown report
        with open(self.output_dir / "fairness_report.md", "w", encoding="utf-8") as f:
            f.write("# Token Attribution Fairness Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"This report analyzes the impact of identity terms on model predictions based on token attribution.\n")
            f.write(f"Data source: {self.data_path}\n")
            f.write(f"Attribution analysis source: {self.attribution_dir}\n\n")
            
            f.write("## Identity Term Impact\n\n")
            
            if impact_df is not None and not impact_df.empty:
                f.write("### Summary by Category\n\n")
                f.write("| Category | Mean Contribution | Std Deviation | Count | Normalized Mean |\n")
                f.write("|----------|------------------|---------------|-------|----------------|\n")
                
                for _, row in category_summary.iterrows():
                    f.write(f"| {row['category']} | {row['contribution_mean']:.4f} | ")
                    f.write(f"{row['contribution_std']:.4f} | {row['contribution_count']} | ")
                    f.write(f"{row['normalized_contribution_mean']:.4f} |\n")
                
                f.write("\n### Visualizations\n\n")
                f.write("![Identity Category Impact](identity_category_impact.png)\n\n")
                
                # Only include these if they were generated
                if pathlib.Path(self.output_dir / "identity_term_distribution.png").exists():
                    f.write("![Identity Term Distribution](identity_term_distribution.png)\n\n")
                
                if pathlib.Path(self.output_dir / "identity_term_heatmap.png").exists():
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
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on this analysis, the following actions are recommended:\n\n")
            f.write("1. Review training data to ensure balanced representation of identity groups\n")
            f.write("2. Consider data augmentation techniques for underrepresented groups\n")
            f.write("3. Implement specific bias mitigation strategies during training\n")
            f.write("4. Conduct more comprehensive analysis with larger samples\n")
        
        print(f"Fairness report generated: {self.output_dir}/fairness_report.md")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate simplified fairness report from token attributions")
    parser.add_argument("--attribution-dir", default="output/analysis_large",
                      help="Directory containing attribution analysis results")
    parser.add_argument("--data", default="data/valid.csv",
                      help="Path to data file with identity columns")
    parser.add_argument("--out-dir", default=None,
                      help="Directory to save fairness report")
    args = parser.parse_args()
    
    analyzer = SimplifiedFairnessAnalyzer(args.attribution_dir, args.data, args.out_dir)
    analyzer.generate_report()

if __name__ == "__main__":
    main() 