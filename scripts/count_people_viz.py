#!/usr/bin/env python
"""
Count people and generate visualizations:
  (1) Total comment rows predicted on
  (2) Unique comment IDs in train and prediction
  (3) Number of unique annotators
  (4) Distribution of comments and annotators per identity subgroup
Outputs: 
  - stdout log
  - output/people_counts.csv
  - figs/people/overall_counts.png
  - figs/people/subgroup_counts.png
  - figs/people/annotators_per_subgroup.png
"""
import argparse, pathlib, pandas as pd, numpy as np, json, os, sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path to access fairness modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness.metrics_v2 import list_identity_columns

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds_distilbert_dev.csv")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--valid", default="data/valid.csv")
    ap.add_argument("--tox-anno", default="data/toxicity_individual_annotations.csv")
    ap.add_argument("--id-anno",  default="data/identity_individual_annotations.csv")
    ap.add_argument("--out-csv", default="output/people_counts.csv")
    ap.add_argument("--out-dir", default="figs/people")
    args = ap.parse_args()

    # Create output directory
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"🔹 Loading data...")
    preds_df = pd.read_csv(args.preds, usecols=["id"])
    train_df = pd.read_csv(args.train)
    
    try:
        valid_df = pd.read_csv(args.valid)
        has_valid = True
    except:
        print(f"Warning: Validation file not found at {args.valid}")
        has_valid = False

    # Overall counts
    counts = {
        "n_pred_rows": len(preds_df),
        "unique_ids_pred": preds_df["id"].nunique(),
        "unique_ids_train": train_df["id"].nunique(),
    }
    
    if has_valid:
        counts["unique_ids_valid"] = valid_df["id"].nunique()

    # Annotator counts
    annotator_counts = {}
    for fkey, path in [("tox", args.tox_anno), ("id", args.id_anno)]:
        if pathlib.Path(path).exists():
            anno = pd.read_csv(path)
            n_annotators = anno["worker"].nunique()
            counts[f"unique_annotators_{fkey}"] = n_annotators
            annotator_counts[fkey] = anno
        else:
            counts[f"unique_annotators_{fkey}"] = "NA"
            print(f"Warning: Annotation file not found at {path}")

    # Save overall counts to CSV
    out_p = pathlib.Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([counts]).to_csv(out_p, index=False)
    
    # Print overall counts
    print("\n===== OVERALL COUNTS =====")
    print(json.dumps(counts, indent=2))
    print("===========================\n")

    # Get identity columns
    identity_cols = list_identity_columns(train_df)
    
    # Count per subgroup
    print(f"🔹 Counting per subgroup...")
    subgroup_counts = {}
    for col in identity_cols:
        # Count comments in each subgroup
        mask = train_df[col] >= 0.5
        subgroup_counts[col] = {
            "n_comments": int(mask.sum()),
            "percent_of_dataset": round(100 * mask.sum() / len(train_df), 2)
        }
        
        # Count annotators for this subgroup if we have annotation data
        if "id" in annotator_counts and "attribute" in annotator_counts["id"].columns:
            id_anno_df = annotator_counts["id"]
            # Match attribute field to subgroup column (may need adjustment based on actual data format)
            subgroup_anno_mask = id_anno_df["attribute"].str.lower() == col.lower()
            if subgroup_anno_mask.sum() > 0:
                n_annotators = id_anno_df.loc[subgroup_anno_mask, "worker"].nunique()
                subgroup_counts[col]["n_annotators_id"] = n_annotators
    
    # Print subgroup counts
    print("===== SUBGROUP COUNTS =====")
    subgroup_df = pd.DataFrame.from_dict(subgroup_counts, orient='index')
    print(subgroup_df.sort_values("n_comments", ascending=False).to_string())
    print("===========================\n")
    
    # Save subgroup counts to CSV
    subgroup_df.reset_index().rename(columns={"index": "subgroup"}).to_csv(
        out_dir / "subgroup_counts.csv", index=False
    )
    
    # Generate visualizations
    print(f"🔹 Generating visualizations...")
    
    # 1. Overall counts bar chart
    plt.figure(figsize=(10, 6))
    visual_counts = {k: v for k, v in counts.items() if not isinstance(v, str)}
    bars = plt.bar(visual_counts.keys(), visual_counts.values(), color=sns.color_palette("viridis", len(visual_counts)))
    plt.title("Overall People Counts")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=0)
    
    plt.savefig(out_dir / "overall_counts.png", dpi=300)
    plt.close()
    
    # 2. Subgroup counts (top N subgroups)
    top_n = 15
    top_subgroups = subgroup_df.sort_values("n_comments", ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_subgroups.index, top_subgroups["n_comments"], color=sns.color_palette("viridis", top_n))
    plt.title(f"Number of Comments per Identity Subgroup (Top {top_n})")
    plt.xlabel("Number of Comments")
    plt.tight_layout()
    
    # Add count labels inside bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width*0.98, bar.get_y() + bar.get_height()/2.,
                f'{int(width):,}',
                ha='right', va='center', color='white', fontweight='bold')
    
    plt.savefig(out_dir / "subgroup_counts.png", dpi=300)
    plt.close()
    
    # 3. If we have annotator data per subgroup, visualize that too
    if "n_annotators_id" in subgroup_df.columns:
        plt.figure(figsize=(12, 8))
        top_annotated = subgroup_df.sort_values("n_annotators_id", ascending=False).head(top_n)
        
        bars = plt.barh(top_annotated.index, top_annotated["n_annotators_id"], 
                        color=sns.color_palette("magma", top_n))
        plt.title(f"Number of Unique Annotators per Identity Subgroup (Top {top_n})")
        plt.xlabel("Number of Annotators")
        plt.tight_layout()
        
        # Add count labels inside bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width*0.98, bar.get_y() + bar.get_height()/2.,
                    f'{int(width):,}',
                    ha='right', va='center', color='white', fontweight='bold')
        
        plt.savefig(out_dir / "annotators_per_subgroup.png", dpi=300)
        plt.close()
    
    # 4. Create a pie chart showing the distribution of top subgroups
    plt.figure(figsize=(10, 10))
    plt.pie(top_subgroups["n_comments"], labels=top_subgroups.index, 
            autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", top_n))
    plt.axis('equal')
    plt.title("Distribution of Comments Across Top Identity Subgroups")
    plt.tight_layout()
    plt.savefig(out_dir / "subgroup_distribution_pie.png", dpi=300)
    plt.close()
    
    print(f"✅ Visualizations saved to {out_dir}")

if __name__ == "__main__":
    main() 
