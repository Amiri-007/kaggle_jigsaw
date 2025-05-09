#!/usr/bin/env python
"""
Intersectional Fairness Analysis
================================
This script analyzes fairness metrics across intersectional demographic groups,
considering combinations of identity attributes rather than single attributes in isolation.

Outputs:
  • figs/intersectional/heatmap.png - Heatmap of disparities across intersectional groups
  • figs/intersectional/worst_intersections.png - Bar chart of worst performing intersections
  • output/intersectional_metrics.csv - CSV with metrics for all analyzed intersections
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from sklearn.metrics import confusion_matrix, roc_auc_score
from fairlearn.metrics import (
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
)

# Add project root to path to access fairness modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness.metrics_v2 import list_identity_columns

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

def create_intersection_column(df, cols):
    """Create a new column for the intersection of multiple identity columns"""
    intersection_name = " & ".join(cols)
    df[intersection_name] = (df[list(cols)] >= 0.5).all(axis=1).astype(int)
    return intersection_name

def compute_metrics_for_group(df, group_col):
    """Compute fairness metrics for a specific group column"""
    # Filter to only include rows where the group value is 1 (True)
    group_df = df[df[group_col] >= 0.5]
    
    if len(group_df) < 10:  # Skip if too few samples
        return None
    
    # Compute basic metrics
    y_true = group_df["y_true"]
    y_pred = group_df["y_pred"]
    
    if len(set(y_true)) == 1 or len(set(y_pred)) == 1:
        # Skip if all predictions or all true values are the same
        return None
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        sel_rate = selection_rate(y_true, y_pred)
        fpr = false_positive_rate(y_true, y_pred)
        fnr = false_negative_rate(y_true, y_pred)
        
        # Calculate AUC if possible
        try:
            auc = roc_auc_score(y_true, group_df["prediction"])
        except:
            auc = np.nan
            
        return {
            "group": group_col,
            "count": len(group_df),
            "selection_rate": sel_rate,
            "fpr": fpr,
            "fnr": fnr, 
            "auc": auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    except Exception as e:
        print(f"Error computing metrics for {group_col}: {str(e)}")
        return None

def compute_disparities(metrics_df, overall_metrics, majority_group=None):
    """Compute disparities relative to overall metrics or majority group"""
    # If majority group specified, use it as reference
    if majority_group and majority_group in metrics_df["group"].values:
        ref_metrics = metrics_df[metrics_df["group"] == majority_group].iloc[0]
        ref_type = "majority"
    else:
        # Otherwise use overall metrics
        ref_metrics = overall_metrics
        ref_type = "overall"
    
    # Calculate disparities
    metrics_df["sel_rate_disparity"] = metrics_df["selection_rate"] / ref_metrics["selection_rate"]
    metrics_df["fpr_disparity"] = metrics_df["fpr"] / (ref_metrics["fpr"] + 1e-9)  # Avoid div by 0
    metrics_df["fnr_disparity"] = metrics_df["fnr"] / (ref_metrics["fnr"] + 1e-9)
    metrics_df["auc_disparity"] = metrics_df["auc"] / (ref_metrics["auc"] + 1e-9)
    
    # Flag violations
    metrics_df["within_0.8_1.2_fpr"] = metrics_df["fpr_disparity"].between(0.8, 1.2)
    metrics_df["within_0.8_1.2_fnr"] = metrics_df["fnr_disparity"].between(0.8, 1.2)
    metrics_df["within_0.8_1.2_sel"] = metrics_df["sel_rate_disparity"].between(0.8, 1.2)
    
    # Add reference information
    metrics_df["reference_type"] = ref_type
    metrics_df["reference_group"] = majority_group if ref_type == "majority" else "overall"
    
    return metrics_df

def plot_heatmap(pivot_df, metric, output_path):
    """Create a heatmap visualization for disparities across intersectional groups"""
    plt.figure(figsize=(12, 10))
    mask = pivot_df.isnull()
    
    # Determine colormap based on metric
    if "disparity" in metric:
        # Center at 1.0 for disparity metrics
        cmap = "RdBu_r"
        center = 1.0
        vmin = 0.5
        vmax = 1.5
    else:
        # Use standard colormap for other metrics
        cmap = "viridis"
        center = None
        vmin = None
        vmax = None
    
    ax = sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap=cmap,
        mask=mask,
        fmt=".2f",
        linewidths=0.5,
        center=center,
        vmin=vmin,
        vmax=vmax
    )
    
    plt.title(f"Intersectional Analysis: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_worst_intersections(metrics_df, metric, output_path, ascending=True, n=10):
    """Plot the worst performing intersectional groups for a given metric"""
    # Sort the data
    sorted_df = metrics_df.sort_values(metric, ascending=ascending).head(n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_df["group"], sorted_df[metric])
    
    # Add reference line for parity if it's a disparity metric
    if "disparity" in metric:
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=0.8, color='red', linestyle=':', alpha=0.5)
        plt.axvline(x=1.2, color='red', linestyle=':', alpha=0.5)
    
    plt.title(f"Top {n} Intersectional Groups: {metric}")
    plt.xlabel(metric)
    plt.ylabel("Intersectional Group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze intersectional fairness metrics")
    parser.add_argument("--preds", default="results/preds_distilbert_dev.csv", help="Predictions file")
    parser.add_argument("--val", default="data/train.csv", help="Validation data file")
    parser.add_argument("--thr", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--max-degree", type=int, default=2, 
                        help="Maximum number of identity attributes to combine (default: 2)")
    parser.add_argument("--min-samples", type=int, default=50, 
                        help="Minimum number of samples required in an intersectional group")
    parser.add_argument("--majority", default=None, 
                        help="Majority group for disparity calculation, or None for overall reference")
    parser.add_argument("--out-dir", default="figs/intersectional", help="Output directory for figures")
    parser.add_argument("--primary-groups", nargs="+", default=[], 
                        help="Primary identity groups to analyze (empty for all)")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"🔹 Loading data...")
    preds_df = pd.read_csv(args.preds)
    
    # First load a sample to get identity columns
    sample_df = pd.read_csv(args.val, nrows=5)
    identity_cols = list_identity_columns(sample_df)
    
    # Filter to primary groups if specified
    if args.primary_groups:
        identity_cols = [col for col in identity_cols if col in args.primary_groups]
        if not identity_cols:
            print(f"Error: None of the specified primary groups found in data.")
            return
    
    print(f"  Analyzing {len(identity_cols)} identity columns: {identity_cols}")
    
    # Now load only the columns we need
    val_df = pd.read_csv(args.val, usecols=["id", "target"] + identity_cols)
    
    # Merge predictions with validation data
    df = val_df.merge(preds_df, on="id", how="inner", validate="one_to_one")
    print(f"  Merged dataset: {df.shape[0]} rows")
    
    # Create binary labels
    df["y_true"] = (df["target"] >= 0.5).astype(int)
    df["y_pred"] = (df["prediction"] >= args.thr).astype(int)
    
    # Compute baseline overall metrics
    print(f"🔹 Computing overall metrics...")
    overall_metrics = compute_metrics_for_group(df, "y_true")  # Dummy column that exists for all rows
    if not overall_metrics:
        print("Error: Could not compute overall metrics.")
        return
    
    # Create intersectional groups
    print(f"🔹 Creating intersectional groups (max degree: {args.max_degree})...")
    
    # Generate all combinations up to max_degree
    all_intersections = []
    for i in range(1, args.max_degree + 1):
        for cols in combinations(identity_cols, i):
            intersection_name = create_intersection_column(df, cols)
            all_intersections.append(intersection_name)
    
    print(f"  Created {len(all_intersections)} intersectional groups")
    
    # Compute metrics for each intersectional group
    print(f"🔹 Computing metrics for intersectional groups...")
    metrics_list = []
    
    for group_col in all_intersections:
        metrics = compute_metrics_for_group(df, group_col)
        if metrics and metrics["count"] >= args.min_samples:
            metrics_list.append(metrics)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics_list)
    print(f"  Computed metrics for {len(metrics_df)} groups with sufficient samples")
    
    # Compute disparities
    print(f"🔹 Computing disparities...")
    metrics_df = compute_disparities(metrics_df, overall_metrics, args.majority)
    
    # Sort by count (descending)
    metrics_df = metrics_df.sort_values("count", ascending=False)
    
    # Extract the main attribute for each intersection
    metrics_df["primary_attribute"] = metrics_df["group"].apply(lambda x: x.split(" & ")[0])
    metrics_df["num_attributes"] = metrics_df["group"].apply(lambda x: len(x.split(" & ")))
    
    # Create pivot tables for heatmaps
    if args.max_degree >= 2:
        print(f"🔹 Creating visualizations...")
        
        # Filter to only 2-way intersections for the heatmap
        two_way_df = metrics_df[metrics_df["num_attributes"] == 2].copy()
        
        if len(two_way_df) > 0:
            # Extract the two attributes
            two_way_df[["attr1", "attr2"]] = two_way_df["group"].str.split(" & ", expand=True)
            
            # Create pivot tables
            for metric in ["fpr_disparity", "fnr_disparity", "sel_rate_disparity", "auc"]:
                pivot_df = two_way_df.pivot(index="attr1", columns="attr2", values=metric)
                
                # Plot heatmap
                plot_heatmap(pivot_df, metric, out_dir / f"{metric}_heatmap.png")
    
    # Plot worst performing intersections
    print(f"🔹 Generating worst-case analysis...")
    
    # For disparity metrics, worst is highest (for FPR, FNR) 
    plot_worst_intersections(metrics_df, "fpr_disparity", out_dir / "worst_fpr_disparity.png", 
                            ascending=False, n=10)
    plot_worst_intersections(metrics_df, "fnr_disparity", out_dir / "worst_fnr_disparity.png", 
                            ascending=False, n=10)
    
    # For AUC, worst is lowest
    plot_worst_intersections(metrics_df, "auc", out_dir / "worst_auc.png", 
                            ascending=True, n=10)
    
    # Save metrics to CSV
    print(f"🔹 Saving metrics to CSV...")
    metrics_df.to_csv("output/intersectional_metrics.csv", index=False)
    
    # Print summary of violations
    print("\n==== Intersectional Fairness Summary ====")
    print(f"Total groups analyzed: {len(metrics_df)}")
    
    # Count violations
    fpr_violations = (~metrics_df["within_0.8_1.2_fpr"]).sum()
    fnr_violations = (~metrics_df["within_0.8_1.2_fnr"]).sum()
    sel_violations = (~metrics_df["within_0.8_1.2_sel"]).sum()
    
    print(f"Groups violating 0.8-1.2 FPR disparity: {fpr_violations} ({fpr_violations/len(metrics_df)*100:.1f}%)")
    print(f"Groups violating 0.8-1.2 FNR disparity: {fnr_violations} ({fnr_violations/len(metrics_df)*100:.1f}%)")
    print(f"Groups violating 0.8-1.2 selection rate disparity: {sel_violations} ({sel_violations/len(metrics_df)*100:.1f}%)")
    
    print(f"\n✅ Analysis complete. Visualizations saved to {args.out_dir}/")
    print(f"✅ Metrics saved to output/intersectional_metrics.csv")

if __name__ == "__main__":
    main() 