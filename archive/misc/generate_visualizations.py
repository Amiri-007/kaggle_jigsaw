import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

def main():
    # Set theme
    sns.set_theme(style="whitegrid")
    
    # Create output directory
    out = pathlib.Path("figs")
    out.mkdir(exist_ok=True)
    
    # Load metrics data
    try:
        m = pd.read_parquet("results/audit_metrics.parquet")
        print("Loaded audit_metrics.parquet")
    except:
        # Fallback to CSV if parquet file doesn't exist
        print("Could not load parquet file, trying CSV...")
        m = pd.read_csv("results/simple_metrics_distilbert_simplest.csv")
        print("Loaded simple_metrics_distilbert_simplest.csv")
    
    # 3.1 Heatmap
    print("Generating heatmap...")
    try:
        h = m.set_index("subgroup_name")[["subgroup_auc", "bpsn_auc", "bnsp_auc"]]
    except:
        h = m.set_index("subgroup")[["subgroup_auc", "bpsn_auc", "bnsp_auc"]]
    
    plt.figure(figsize=(6, 8))
    sns.heatmap(h, annot=True, vmin=0.5, vmax=1.0, cmap="RdYlGn")
    plt.title("Fairness metrics heatmap – dev run")
    plt.tight_layout()
    plt.savefig(out/"heatmap_dev.png")
    plt.close()
    
    # 3.2 Error-rate gaps bar
    print("Generating error-rate gap plots...")
    try:
        # Check if the fpr and fnr columns exist
        if "fpr" in m.columns and "fnr" in m.columns:
            gaps = m.assign(
                fpr_gap=abs(m.fpr - m.fpr.mean()),
                fnr_gap=abs(m.fnr - m.fnr.mean())
            )
            
            # FPR gap
            plt.figure(figsize=(8, 4))
            gaps.sort_values("fpr_gap").plot.bar(x="subgroup_name", y="fpr_gap")
            plt.ylabel("|FPR – mean(FPR)|")
            plt.title("FPR gap by subgroup")
            plt.tight_layout()
            plt.savefig(out/"fpr_gap_dev.png")
            plt.close()
            
            # FNR gap
            plt.figure(figsize=(8, 4))
            gaps.sort_values("fnr_gap").plot.bar(x="subgroup_name", y="fnr_gap")
            plt.ylabel("|FNR – mean(FNR)|")
            plt.title("FNR gap by subgroup")
            plt.tight_layout()
            plt.savefig(out/"fnr_gap_dev.png")
            plt.close()
        else:
            print("Warning: FPR/FNR columns not found in metrics file")
    except Exception as e:
        print(f"Error generating gap plots: {e}")
    
    print(f"✅ Generated visualizations: figs/heatmap_dev.png, figs/fpr_gap_dev.png, figs/fnr_gap_dev.png")

if __name__ == "__main__":
    main() 