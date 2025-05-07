#!/usr/bin/env python
"""
Simplified script to generate fairness visualizations.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Create output directory
os.makedirs('figs', exist_ok=True)

# Set parameters
RESULTS_DIR = "results"
OUTPUT_DIR = "figs"
THRESHOLD = 0.5  # Decision threshold for binary classification

def generate_fairness_heatmap(metrics_df, model_name):
    """Generate a heatmap of fairness metrics by demographic group."""
    # Ensure the DataFrame has the correct structure
    if not all(col in metrics_df.columns for col in ['subgroup_name', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc']):
        print(f"Metrics dataframe missing required columns for {model_name}")
        return
    
    # Sort by subgroup AUC
    if 'subgroup_auc' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('subgroup_auc', ascending=True)
    
    # Create a pivot table for the heatmap
    metrics_to_plot = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    df_melted = pd.melt(
        metrics_df, 
        id_vars=['subgroup_name'], 
        value_vars=metrics_to_plot,
        var_name='metric',
        value_name='value'
    )
    
    # Convert to heatmap format
    pivot_df = df_melted.pivot_table(index='subgroup_name', columns='metric', values='value')
    
    # Create figure
    plt.figure(figsize=(12, len(metrics_df) * 0.5 + 2))
    
    # Generate heatmap
    heatmap = sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap='RdYlGn', 
        vmin=0.5,    # Min value for AUC
        vmax=1.0,    # Max value for AUC
        fmt='.3f'
    )
    
    # Add colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('AUC')
    
    plt.title(f'Fairness Metrics by Demographic Group - {model_name}')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'fairness_heatmap_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fairness heatmap to {output_path}")
    return output_path

def generate_auc_gap_chart(metrics_df, model_name):
    """Generate a bar chart showing the AUC gap by demographic group."""
    # Ensure the DataFrame has the correct structure
    if not all(col in metrics_df.columns for col in ['subgroup_name', 'subgroup_auc']):
        print(f"Metrics dataframe missing required columns for {model_name}")
        return
    
    # Sort by subgroup AUC
    sorted_df = metrics_df.sort_values('subgroup_auc').dropna(subset=['subgroup_auc'])
    
    # Create figure
    plt.figure(figsize=(12, len(sorted_df) * 0.4 + 2))
    
    # Generate bar chart
    bars = plt.barh(sorted_df['subgroup_name'], sorted_df['subgroup_auc'])
    
    # Add a vertical line at AUC = 0.8 for reference
    plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}', 
            ha='left', 
            va='center'
        )
    
    plt.xlim(0.5, 1.0)  # AUC range from 0.5 to 1.0
    plt.xlabel('AUC')
    plt.ylabel('Demographic Group')
    plt.title(f'Subgroup AUC by Demographic Group - {model_name}')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'subgroup_auc_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved AUC gap chart to {output_path}")
    return output_path

def compare_models_chart(metrics_files):
    """Generate a comparison chart for multiple models."""
    if len(metrics_files) < 2:
        print("Need at least 2 models to compare")
        return
    
    model_data = {}
    
    # Load metrics for each model
    for metrics_file in metrics_files:
        model_name = os.path.basename(metrics_file).replace("metrics_", "").replace(".csv", "")
        try:
            metrics_df = pd.read_csv(metrics_file)
            if 'subgroup_name' in metrics_df.columns and 'subgroup_auc' in metrics_df.columns:
                model_data[model_name] = metrics_df[['subgroup_name', 'subgroup_auc']].copy()
        except Exception as e:
            print(f"Error loading metrics for {model_name}: {e}")
    
    if len(model_data) < 2:
        print("Could not load enough models for comparison")
        return
    
    # Find common subgroups across models
    common_subgroups = set()
    for model_name, df in model_data.items():
        if common_subgroups:
            common_subgroups &= set(df['subgroup_name'])
        else:
            common_subgroups = set(df['subgroup_name'])
    
    # Create comparison dataframe
    comparison_data = []
    for subgroup in common_subgroups:
        row = {'subgroup': subgroup}
        for model_name, df in model_data.items():
            subgroup_df = df[df['subgroup_name'] == subgroup]
            if not subgroup_df.empty and not pd.isna(subgroup_df['subgroup_auc'].iloc[0]):
                row[model_name] = subgroup_df['subgroup_auc'].iloc[0]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(
        comparison_df, 
        id_vars=['subgroup'], 
        var_name='model', 
        value_name='auc'
    )
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Generate grouped bar chart
    sns.barplot(x='subgroup', y='auc', hue='model', data=melted_df)
    
    plt.ylim(0.5, 1.0)  # AUC range from 0.5 to 1.0
    plt.xlabel('Demographic Group')
    plt.ylabel('AUC')
    plt.title('Model Comparison by Demographic Group')
    plt.xticks(rotation=60, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison chart to {output_path}")
    return output_path

def worst_performers_bar(metrics_df, model_name, k=5):
    """Generate a bar chart of the k worst-performing demographic groups."""
    if not all(col in metrics_df.columns for col in ['subgroup_name', 'subgroup_auc']):
        print(f"Metrics dataframe missing required columns for {model_name}")
        return
    
    # Filter out NaN values and sort by subgroup AUC
    filtered_df = metrics_df.dropna(subset=['subgroup_auc'])
    sorted_df = filtered_df.sort_values('subgroup_auc').head(k)
    
    # Create figure
    plt.figure(figsize=(10, max(5, k * 0.6)))
    
    # Create horizontal bar chart
    bars = plt.barh(sorted_df['subgroup_name'], sorted_df['subgroup_auc'])
    
    # Add a vertical line at AUC = 0.8 for reference
    plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}', 
            va='center'
        )
    
    # Set axis limits, labels, and title
    plt.xlim(0.5, 1.0)  # AUC range from 0.5 to 1.0
    plt.xlabel('Subgroup AUC')
    plt.ylabel('Identity Group')
    plt.title(f'Worst {k} Performing Demographic Groups - {model_name}')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'worst_k_bar_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved worst performers bar chart to {output_path}")
    return output_path

def calculate_error_rates(y_true, y_pred, subgroup_mask):
    """Calculate error rates for a subgroup vs non-subgroup."""
    # Calculate confusion matrix for the subgroup
    subgroup_cm = confusion_matrix(
        y_true[subgroup_mask], 
        y_pred[subgroup_mask], 
        labels=[0, 1]
    )
    
    # Calculate confusion matrix for the non-subgroup
    non_subgroup_cm = confusion_matrix(
        y_true[~subgroup_mask], 
        y_pred[~subgroup_mask], 
        labels=[0, 1]
    )
    
    # Extract confusion matrix elements for subgroup
    tn_subgroup, fp_subgroup, fn_subgroup, tp_subgroup = subgroup_cm.ravel()
    
    # Extract confusion matrix elements for non-subgroup
    tn_non_subgroup, fp_non_subgroup, fn_non_subgroup, tp_non_subgroup = non_subgroup_cm.ravel()
    
    # Calculate FPR and FNR for subgroup
    fpr_subgroup = fp_subgroup / (fp_subgroup + tn_subgroup) if (fp_subgroup + tn_subgroup) > 0 else 0
    fnr_subgroup = fn_subgroup / (fn_subgroup + tp_subgroup) if (fn_subgroup + tp_subgroup) > 0 else 0
    
    # Calculate FPR and FNR for non-subgroup
    fpr_non_subgroup = fp_non_subgroup / (fp_non_subgroup + tn_non_subgroup) if (fp_non_subgroup + tn_non_subgroup) > 0 else 0
    fnr_non_subgroup = fn_non_subgroup / (fn_non_subgroup + tp_non_subgroup) if (fn_non_subgroup + tp_non_subgroup) > 0 else 0
    
    # Calculate gaps (subgroup rate - non-subgroup rate)
    fpr_gap = fpr_subgroup - fpr_non_subgroup
    fnr_gap = fnr_subgroup - fnr_non_subgroup
    
    return fpr_gap, fnr_gap

def generate_error_gap_heatmap(gaps_dict, model_name, threshold=0.5):
    """Generate a heatmap of FPR and FNR gaps across demographic groups."""
    # Create dataframe from gaps dictionary
    data = []
    for identity, (fpr_gap, fnr_gap) in gaps_dict.items():
        data.append({
            'identity_group': identity,
            'fpr_gap': fpr_gap,
            'fnr_gap': fnr_gap,
            'max_gap': max(abs(fpr_gap), abs(fnr_gap))
        })
    
    gap_df = pd.DataFrame(data)
    
    # Sort by maximum gap (descending) to highlight the most problematic groups
    gap_df = gap_df.sort_values('max_gap', ascending=False)
    
    # Create pivot data for heatmap
    pivot_df = gap_df.set_index('identity_group')[['fpr_gap', 'fnr_gap']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(gaps_dict) * 0.5)))
    
    # Create heatmap
    heatmap = sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt='.3f', 
        cmap='RdBu_r',  # Red-Blue diverging colormap
        center=0,       # Center colormap at 0
        vmin=-0.3,      # Min gap value
        vmax=0.3,       # Max gap value
        cbar_kws={'label': 'Gap (Subgroup Rate - Non-subgroup Rate)'}
    )
    
    # Set title
    plt.title(f'Error Rate Gaps at Threshold {threshold} - {model_name}')
    
    # Add x-axis label explaining what positive/negative gaps mean
    ax.set_xlabel('Positive gap = Higher error rate for subgroup')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'error_gap_heatmap_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error gap heatmap to {output_path}")
    return output_path

def generate_confusion_mosaic(y_true, y_pred, subgroup_mask, subgroup_name, model_name):
    """Generate a confusion matrix mosaic for a specific subgroup."""
    # Calculate confusion matrix for the specified subgroup
    y_true_subgroup = y_true[subgroup_mask]
    y_pred_subgroup = y_pred[subgroup_mask]
    
    # Create confusion matrix
    cm = pd.crosstab(
        y_true_subgroup, 
        y_pred_subgroup, 
        rownames=['True'], 
        colnames=['Predicted'],
        normalize='all'  # Normalize to show proportions
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create heatmap of confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f', 
        cmap='RdYlGn',
        ax=ax,
        cbar=False
    )
    
    # Add labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {subgroup_name} - {model_name}')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'confusion_{subgroup_name}_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion mosaic for {subgroup_name} to {output_path}")
    return output_path

def sweep_thresholds_gaps(y_true, y_prob, subgroup_mask, n_steps=101):
    """Perform a threshold sweep to analyze how FPR and FNR gaps change with different thresholds."""
    # Generate threshold values
    thresholds = np.linspace(0, 1, n_steps)
    
    # Initialize results
    results = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Create binary predictions at this threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate error rate gaps
        fpr_gap, fnr_gap = calculate_error_rates(y_true, y_pred, subgroup_mask)
        
        # Store results
        results.append({
            'threshold': threshold,
            'fpr_gap': fpr_gap,
            'fnr_gap': fnr_gap,
            'abs_fpr_gap': abs(fpr_gap),
            'abs_fnr_gap': abs(fnr_gap),
            'max_gap': max(abs(fpr_gap), abs(fnr_gap)),
            'mean_gap': (abs(fpr_gap) + abs(fnr_gap)) / 2
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def generate_threshold_gap_curve(df_sweep, identity_name, model_name):
    """Create a plot showing how FPR and FNR gaps change with different thresholds."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot absolute FPR gap
    ax.plot(df_sweep['threshold'], df_sweep['abs_fpr_gap'], 
           label='|FPR Gap|', color='blue', linewidth=2)
    
    # Plot absolute FNR gap
    ax.plot(df_sweep['threshold'], df_sweep['abs_fnr_gap'], 
           label='|FNR Gap|', color='red', linewidth=2)
    
    # Plot mean gap
    ax.plot(df_sweep['threshold'], df_sweep['mean_gap'], 
           label='Mean Gap', color='purple', linestyle='--', linewidth=1.5)
    
    # Add vertical line at threshold = 0.5
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set axis labels and title
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Absolute Rate Gap')
    ax.set_title(f'Error Rate Gaps vs. Threshold - {identity_name} - {model_name}')
    
    # Add legend
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'threshold_gap_curve_{identity_name}_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved threshold gap curve for {identity_name} to {output_path}")
    return output_path

def main():
    # Get metrics files
    metrics_files = glob.glob(os.path.join(RESULTS_DIR, "metrics_*.csv"))
    
    if not metrics_files:
        print("No metrics files found in results directory")
        return
    
    print(f"Found {len(metrics_files)} metrics files")
    
    # Track generated figures
    generated_figures = []
    
    # Process each model
    for metrics_file in metrics_files:
        model_name = os.path.basename(metrics_file).replace("metrics_", "").replace(".csv", "")
        print(f"\nProcessing model: {model_name}")
        
        try:
            # Load metrics
            metrics_df = pd.read_csv(metrics_file)
            print(f"Loaded metrics with {len(metrics_df)} rows and {len(metrics_df.columns)} columns")
            
            # Check if we need to rename columns to match expected format
            if 'subgroup_name' not in metrics_df.columns and metrics_df.columns[0] in ['identity_group', 'subgroup']:
                metrics_df = metrics_df.rename(columns={metrics_df.columns[0]: 'subgroup_name'})
            
            # Generate fairness heatmap
            heatmap_path = generate_fairness_heatmap(metrics_df, model_name)
            if heatmap_path:
                generated_figures.append(heatmap_path)
            
            # Generate AUC gap chart
            auc_gap_path = generate_auc_gap_chart(metrics_df, model_name)
            if auc_gap_path:
                generated_figures.append(auc_gap_path)
            
            # Generate worst performers bar chart
            worst_bar_path = worst_performers_bar(metrics_df, model_name)
            if worst_bar_path:
                generated_figures.append(worst_bar_path)
            
            # Load predictions if available
            pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
            if os.path.exists(pred_file):
                # Load predictions and test data
                preds_df = pd.read_csv(pred_file)
                test_file = os.path.join("data", "test_public_expanded.csv")
                
                if os.path.exists(test_file):
                    test_df = pd.read_csv(test_file)
                    
                    # Merge predictions with test data
                    if 'id' in preds_df.columns and 'id' in test_df.columns:
                        merged_df = pd.merge(preds_df, test_df, on='id', how='inner')
                        
                        if 'prediction' in merged_df.columns and 'target' in merged_df.columns:
                            # Find identity columns
                            identity_cols = [col for col in merged_df.columns 
                                            if any(pattern in col.lower() for pattern in 
                                                  ['male', 'female', 'religion', 'race', 'orientation', 'gender', 'identity_'])]
                            
                            if identity_cols:
                                print(f"Found {len(identity_cols)} identity columns in test data")
                                
                                # Track worst performing group
                                worst_group = None
                                worst_auc = 1.0
                                
                                # Store gaps for heatmap
                                gaps_dict = {}
                                
                                for identity_col in identity_cols:
                                    # Create mask for this identity group
                                    identity_mask = merged_df[identity_col] == 1
                                    
                                    # Skip if too few samples
                                    if sum(identity_mask) < 50:
                                        continue
                                    
                                    # If we have subgroup metrics, find the worst group
                                    if 'subgroup_name' in metrics_df.columns and 'subgroup_auc' in metrics_df.columns:
                                        subgroup_metrics = metrics_df[
                                            (metrics_df['subgroup_name'] == identity_col) & 
                                            (~metrics_df['subgroup_auc'].isna())
                                        ]
                                        if not subgroup_metrics.empty:
                                            auc_value = subgroup_metrics['subgroup_auc'].iloc[0]
                                            if auc_value < worst_auc:
                                                worst_auc = auc_value
                                                worst_group = identity_col
                                    
                                    # Convert predictions to binary at threshold 0.5
                                    binary_preds = (merged_df['prediction'] >= THRESHOLD).astype(int)
                                    
                                    # Calculate error rate gaps
                                    fpr_gap, fnr_gap = calculate_error_rates(
                                        merged_df['target'].values,
                                        binary_preds.values,
                                        identity_mask.values
                                    )
                                    
                                    # Store for heatmap
                                    gaps_dict[identity_col] = (fpr_gap, fnr_gap)
                                    
                                    # Generate threshold gap curve
                                    df_sweep = sweep_thresholds_gaps(
                                        merged_df['target'].values,
                                        merged_df['prediction'].values,
                                        identity_mask.values
                                    )
                                    
                                    gap_curve_path = generate_threshold_gap_curve(
                                        df_sweep, 
                                        identity_col, 
                                        model_name
                                    )
                                    generated_figures.append(gap_curve_path)
                                
                                # Generate error gap heatmap
                                if gaps_dict:
                                    error_heatmap_path = generate_error_gap_heatmap(
                                        gaps_dict,
                                        model_name
                                    )
                                    generated_figures.append(error_heatmap_path)
                                
                                # Generate confusion mosaic for worst group
                                if worst_group:
                                    print(f"Worst performing group: {worst_group} with AUC {worst_auc:.3f}")
                                    worst_mask = merged_df[worst_group] == 1
                                    binary_preds = (merged_df['prediction'] >= THRESHOLD).astype(int)
                                    
                                    confusion_path = generate_confusion_mosaic(
                                        merged_df['target'].values,
                                        binary_preds.values,
                                        worst_mask.values,
                                        worst_group,
                                        model_name
                                    )
                                    generated_figures.append(confusion_path)
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Generate model comparison chart if we have multiple models
    if len(metrics_files) >= 2:
        comparison_path = compare_models_chart(metrics_files)
        if comparison_path:
            generated_figures.append(comparison_path)
    
    # Create a summary of generated figures
    summary = {
        "figures": generated_figures,
        "count": len(generated_figures)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'figure_inventory.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGenerated {len(generated_figures)} figures:")
    for fig in generated_figures:
        print(f"  - {fig}")

if __name__ == "__main__":
    main() 