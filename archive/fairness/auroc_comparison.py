#!/usr/bin/env python
"""
Generate AUROC comparison visualization for all categories
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Set constants
METRICS_FILE = "results/simple_metrics_distilbert_simplest.csv"
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

def plot_all_auroc_comparison():
    """Plot a comparison of all AUROC scores including overall"""
    # Load data
    df = pd.read_csv(METRICS_FILE)
    
    # Create a new dataframe for plotting
    plot_df = df[['subgroup', 'subgroup_auc']].copy()
    
    # Sort by AUC for better visualization
    plot_df = plot_df.sort_values('subgroup_auc')
    
    # Identify the overall category for special formatting
    plot_df['is_overall'] = plot_df['subgroup'] == 'overall'
    
    # Set up colors - highlight the overall category
    colors = ['#ff7f0e' if x else '#1f77b4' for x in plot_df['is_overall']]
    
    # Create the horizontal bar chart
    plt.figure(figsize=(12, 10))
    ax = plt.barh(y=plot_df['subgroup'], width=plot_df['subgroup_auc'], color=colors)
    
    # Add value labels to each bar
    for i, (auc, name) in enumerate(zip(plot_df['subgroup_auc'], plot_df['subgroup'])):
        if name == 'overall':
            plt.text(auc - 0.01, i, f"{auc:.4f}", ha='right', va='center', 
                     color='white', fontweight='bold', fontsize=12)
        else:
            plt.text(auc + 0.0005, i, f"{auc:.4f}", ha='left', va='center', fontsize=10)
    
    # Add reference line for overall AUC
    overall_auc = plot_df.loc[plot_df['subgroup'] == 'overall', 'subgroup_auc'].values[0]
    plt.axvline(x=overall_auc, color='#ff7f0e', linestyle='--', alpha=0.7)
    
    # Set axis limits to focus on the relevant range
    min_auc = plot_df['subgroup_auc'].min() - 0.005
    max_auc = plot_df['subgroup_auc'].max() + 0.005
    plt.xlim(min_auc, max_auc)
    
    # Customize the plot
    plt.title('AUROC Comparison Across All Categories', fontsize=16)
    plt.xlabel('Area Under ROC Curve (AUROC)', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Demographic Groups'),
        Patch(facecolor='#ff7f0e', label='Overall Performance')
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.05), frameon=True, ncol=2)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = FIGS_DIR / "all_auroc_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"AUROC comparison saved to {output_file}")
    return output_file

if __name__ == "__main__":
    plot_all_auroc_comparison() 