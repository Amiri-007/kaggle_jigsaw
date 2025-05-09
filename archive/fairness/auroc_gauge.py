#!/usr/bin/env python
"""
Generate AUROC gauge visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set constants
METRICS_FILE = "results/simple_metrics_distilbert_simplest.csv"
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

def plot_auroc_gauge():
    """Create a gauge-like visualization of all AUROC scores"""
    # Load data
    df = pd.read_csv(METRICS_FILE)
    
    # Extract AUC values
    overall_auc = df.loc[df['subgroup'] == 'overall', 'subgroup_auc'].values[0]
    group_df = df[df['subgroup'] != 'overall'].copy()
    
    # Sort demographic groups by AUC
    group_df = group_df.sort_values('subgroup_auc')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Gauge range parameters
    min_auc = min(0.9, group_df['subgroup_auc'].min() - 0.05)  # Lower bound
    max_auc = min(1.0, group_df['subgroup_auc'].max() + 0.05)  # Upper bound
    
    # Create color gradient
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(min_auc, max_auc)
    
    # Plot colored background gradient
    for val in np.linspace(min_auc, max_auc, 100):
        ax.axvline(val, color=cmap(norm(val)), alpha=0.7, linewidth=8)
    
    # Plot all demographic group AUC values as vertical lines
    for i, row in group_df.iterrows():
        ax.axvline(row['subgroup_auc'], color='black', linewidth=1, alpha=0.5)
        
    # Highlight overall AUC
    ax.axvline(overall_auc, color='blue', linewidth=3, linestyle='-', 
               label=f'Overall AUC: {overall_auc:.4f}')
    
    # Add range labels
    ax.text(min_auc, 0.5, f"{min_auc:.2f}", ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(max_auc, 0.5, f"{max_auc:.2f}", ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Add annotations for min and max demographic AUCs
    min_group = group_df.iloc[0]
    max_group = group_df.iloc[-1]
    
    ax.annotate(f"Min: {min_group['subgroup']} ({min_group['subgroup_auc']:.4f})",
                xy=(min_group['subgroup_auc'], 0.3), xytext=(min_group['subgroup_auc'] - 0.02, 0.1),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    ax.annotate(f"Max: {max_group['subgroup']} ({max_group['subgroup_auc']:.4f})",
                xy=(max_group['subgroup_auc'], 0.3), xytext=(max_group['subgroup_auc'] + 0.02, 0.1),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    # Configure plot
    ax.set_ylim(0, 1)
    ax.set_xlim(min_auc, max_auc)
    ax.set_yticks([])
    ax.set_title("AUROC Performance Gauge", fontsize=16)
    ax.set_xlabel("Area Under ROC Curve (AUROC)", fontsize=12)
    ax.legend(loc='upper center')
    
    # Add text table at the bottom
    table_str = "Category\tAUROC\n"
    table_str += "Overall\t{:.4f}\n".format(overall_auc)
    
    # Add top 5 and bottom 5 groups
    table_str += "\nTop 5 groups:\n"
    for i in range(1, 6):
        if i <= len(group_df):
            row = group_df.iloc[-(i)]
            table_str += "{}\t{:.4f}\n".format(row['subgroup'], row['subgroup_auc'])
    
    table_str += "\nBottom 5 groups:\n"
    for i in range(5):
        if i < len(group_df):
            row = group_df.iloc[i]
            table_str += "{}\t{:.4f}\n".format(row['subgroup'], row['subgroup_auc'])
    
    plt.figtext(0.5, -0.15, table_str, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save the figure
    output_file = FIGS_DIR / "auroc_gauge.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"AUROC gauge visualization saved to {output_file}")
    return output_file

if __name__ == "__main__":
    plot_auroc_gauge() 