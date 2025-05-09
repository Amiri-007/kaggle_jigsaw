#!/usr/bin/env python
"""
Generate additional fairness metrics visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Set constants
METRICS_FILE = "results/simple_metrics_distilbert_simplest.csv"
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load metrics data"""
    df = pd.read_csv(METRICS_FILE)
    # Move 'overall' to the end for better visualization
    overall_row = df[df['subgroup'] == 'overall']
    df = pd.concat([df[df['subgroup'] != 'overall'], overall_row])
    return df

def plot_subgroup_auc_comparison(df):
    """Plot subgroup AUC comparison"""
    # Filter out 'overall' for this plot
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='subgroup', y='subgroup_auc', data=plot_df)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Customize plot
    plt.axhline(y=df[df['subgroup'] == 'overall']['subgroup_auc'].values[0], 
                color='red', linestyle='--', alpha=0.7, 
                label=f"Overall AUC: {df[df['subgroup'] == 'overall']['subgroup_auc'].values[0]:.4f}")
    
    plt.ylim(0.945, 0.965)  # Zoom in to better see the differences
    plt.title('Subgroup AUC Comparison')
    plt.xlabel('Demographic Group')
    plt.ylabel('AUC')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    
    # Save figure
    plt.savefig(FIGS_DIR / "subgroup_auc_comparison.png", dpi=300)
    plt.close()

def plot_metric_comparison(df):
    """Plot comparison of all metrics"""
    # Filter out 'overall' since it doesn't have BPSN/BNSP
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    # Melt the dataframe for seaborn
    metrics = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    melted_df = pd.melt(plot_df, id_vars=['subgroup'], value_vars=metrics, 
                       var_name='Metric', value_name='AUC')
    
    plt.figure(figsize=(14, 10))
    chart = sns.barplot(x='subgroup', y='AUC', hue='Metric', data=melted_df)
    
    # Customize plot
    plt.title('Fairness Metrics Comparison Across Demographic Groups')
    plt.xlabel('Demographic Group')
    plt.ylabel('AUC')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric Type')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(FIGS_DIR / "fairness_metrics_comparison.png", dpi=300)
    plt.close()

def plot_bias_gap(df):
    """Plot the difference between subgroup_auc and bnsp_auc to visualize bias gap"""
    # Filter out 'overall'
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    # Calculate bias gap
    plot_df['bias_gap'] = plot_df['subgroup_auc'] - plot_df['bnsp_auc']
    
    # Sort by bias gap for better visualization
    plot_df = plot_df.sort_values('bias_gap', ascending=False)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='subgroup', y='bias_gap', data=plot_df)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    plt.title('Bias Gap (subgroup_auc - bnsp_auc) by Demographic Group')
    plt.xlabel('Demographic Group')
    plt.ylabel('Bias Gap')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(FIGS_DIR / "bias_gap.png", dpi=300)
    plt.close()

def plot_demographic_size(df):
    """Plot demographic group sizes"""
    # Filter out 'overall'
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='subgroup', y='subgroup_size', data=plot_df)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    plt.title('Size of Demographic Groups in Validation Data')
    plt.xlabel('Demographic Group')
    plt.ylabel('Number of Examples')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(FIGS_DIR / "demographic_group_sizes.png", dpi=300)
    plt.close()

def plot_metrics_heatmap(df):
    """Create a heatmap of all metrics"""
    # Filter out 'overall'
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    # Create a matrix for the heatmap
    subgroups = plot_df['subgroup'].values
    metrics = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    
    # Create heatmap data
    heatmap_data = plot_df[metrics].values
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", 
                linewidths=.5, cbar_kws={"shrink": .8},
                yticklabels=subgroups, xticklabels=metrics)
    
    # Customize plot
    plt.title('Fairness Metrics Heatmap')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(FIGS_DIR / "fairness_metrics_heatmap.png", dpi=300)
    plt.close()

def plot_positive_rate_comparison(df):
    """Plot comparison of positive rates in subgroups vs background"""
    # Filter out 'overall'
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    # Melt the dataframe for seaborn
    rate_cols = ['subgroup_pos_rate', 'background_pos_rate']
    melted_df = pd.melt(plot_df, id_vars=['subgroup'], value_vars=rate_cols, 
                       var_name='Rate Type', value_name='Positive Rate')
    
    # Rename for better labels
    melted_df['Rate Type'] = melted_df['Rate Type'].replace({
        'subgroup_pos_rate': 'Subgroup',
        'background_pos_rate': 'Background'
    })
    
    plt.figure(figsize=(14, 10))
    chart = sns.barplot(x='subgroup', y='Positive Rate', hue='Rate Type', data=melted_df)
    
    # Customize plot
    plt.title('Positive Rate Comparison: Subgroup vs Background')
    plt.xlabel('Demographic Group')
    plt.ylabel('Positive Rate (% Toxic)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Population')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(FIGS_DIR / "positive_rate_comparison.png", dpi=300)
    plt.close()

def create_summary_figure(df):
    """Create a summary figure with performance and bias metrics"""
    # Filter out 'overall'
    plot_df = df[df['subgroup'] != 'overall'].copy()
    
    # Calculate metrics for viz
    plot_df['bias_gap'] = plot_df['subgroup_auc'] - plot_df['bnsp_auc']
    overall_auc = df[df['subgroup'] == 'overall']['subgroup_auc'].values[0]
    plot_df['diff_from_overall'] = plot_df['subgroup_auc'] - overall_auc
    
    # Sort by AUC for better visualization
    plot_df = plot_df.sort_values('subgroup_auc')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot AUC with difference from overall
    sns.barplot(x='subgroup', y='subgroup_auc', data=plot_df, ax=ax1)
    ax1.axhline(y=overall_auc, color='red', linestyle='--', 
                label=f"Overall AUC: {overall_auc:.4f}")
    ax1.set_ylim(min(plot_df['subgroup_auc'])-0.005, max(plot_df['subgroup_auc'])+0.005)
    
    # Add annotations for subgroup AUC
    for i, v in enumerate(plot_df['subgroup_auc']):
        diff = plot_df['diff_from_overall'].iloc[i]
        color = 'green' if diff >= 0 else 'red'
        ax1.text(i, v + 0.0005, f"{v:.4f}\n({diff:+.4f})", 
                ha='center', va='bottom', fontsize=8, color=color)
    
    ax1.set_title('Subgroup Performance with Difference from Overall AUC')
    ax1.set_ylabel('AUC')
    ax1.legend()
    
    # Plot bias gap
    sns.barplot(x='subgroup', y='bias_gap', data=plot_df, ax=ax2)
    
    # Add annotations for bias gap
    for i, v in enumerate(plot_df['bias_gap']):
        ax2.text(i, v + 0.0005, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    
    ax2.set_title('Bias Gap (subgroup_auc - bnsp_auc)')
    ax2.set_xlabel('Demographic Group')
    ax2.set_ylabel('Bias Gap')
    
    # Rotate x-labels
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fairness_summary.png", dpi=300)
    plt.close()

def main():
    """Main function to generate all plots"""
    print("Loading metrics data...")
    df = load_data()
    
    print("Generating subgroup AUC comparison...")
    plot_subgroup_auc_comparison(df)
    
    print("Generating metrics comparison...")
    plot_metric_comparison(df)
    
    print("Generating bias gap visualization...")
    plot_bias_gap(df)
    
    print("Generating demographic size visualization...")
    plot_demographic_size(df)
    
    print("Generating metrics heatmap...")
    plot_metrics_heatmap(df)
    
    print("Generating positive rate comparison...")
    plot_positive_rate_comparison(df)
    
    print("Generating summary figure...")
    create_summary_figure(df)
    
    print(f"All figures saved to {FIGS_DIR}")

if __name__ == "__main__":
    main() 