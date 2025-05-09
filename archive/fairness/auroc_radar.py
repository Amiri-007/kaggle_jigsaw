#!/usr/bin/env python
"""
Generate radar plot of AUROC metrics across demographic groups
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set constants
METRICS_FILE = "results/simple_metrics_distilbert_simplest.csv"
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

def plot_auroc_radar():
    """Create a radar plot comparing all AUC metrics across demographic groups"""
    # Load data
    df = pd.read_csv(METRICS_FILE)
    
    # Filter out 'overall' row and select relevant columns
    plot_df = df[df['subgroup'] != 'overall'].copy()
    metrics = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    
    # Get overall AUC for reference
    overall_auc = df.loc[df['subgroup'] == 'overall', 'subgroup_auc'].values[0]
    
    # Number of variables
    categories = plot_df['subgroup'].tolist()
    N = len(categories)
    
    # Create a figure with a polar projection
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up the plot
    ax.set_theta_offset(np.pi / 2)  # Start at top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Add axis labels
    plt.xticks(angles[:-1], categories, fontsize=8)
    
    # Draw y-axis lines from center to outer edge
    ax.set_rlabel_position(0)
    
    # Set y-axis limits to highlight differences
    min_value = min(plot_df[metrics].min().min(), overall_auc) - 0.02
    max_value = 1.0
    plt.ylim(min_value, max_value)
    
    # Plot data for each metric
    for metric_name in metrics:
        values = plot_df[metric_name].tolist()
        values += values[:1]  # Close the loop
        
        # Plot the metric line
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
                label=f"{metric_name} (avg: {plot_df[metric_name].mean():.4f})")
        ax.fill(angles, values, alpha=0.1)
    
    # Add overall AUC as a reference circle
    overall_circle = [overall_auc] * (N + 1)
    ax.plot(angles, overall_circle, linewidth=2, linestyle='--', color='red', 
            label=f"Overall AUC: {overall_auc:.4f}")
    
    # Add metric descriptions
    metric_descriptions = {
        'subgroup_auc': 'Performance on the specific demographic subgroup',
        'bpsn_auc': 'Background Positive, Subgroup Negative - measures performance on non-toxic content mentioning the identity',
        'bnsp_auc': 'Background Negative, Subgroup Positive - measures performance on toxic content mentioning the identity'
    }
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('AUROC Metrics Across Demographic Groups', size=15, y=1.1)
    
    # Add metric descriptions in a text box
    desc_text = "\n".join([f"{k}: {v}" for k, v in metric_descriptions.items()])
    plt.figtext(0.5, -0.05, desc_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
    
    # Save figure
    output_file = FIGS_DIR / "auroc_radar.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"AUROC radar plot saved to {output_file}")
    return output_file

if __name__ == "__main__":
    plot_auroc_radar() 