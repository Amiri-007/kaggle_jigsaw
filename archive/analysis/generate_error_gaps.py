import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Set up directories
    out_dir = Path("figs")
    out_dir.mkdir(exist_ok=True)
    
    # Use existing metrics file
    metrics_file = "results/simple_metrics_distilbert_simplest.csv"
    print(f"Loading metrics from {metrics_file}")
    m = pd.read_csv(metrics_file)
    
    # Filter out overall row
    m_groups = m[m['subgroup'] != 'overall']
    
    # Calculate synthetic error rates based on AUC differences
    # We'll use the difference between subgroup_auc and overall_auc as a proxy for error rate gaps
    overall_auc = m.loc[m['subgroup'] == 'overall', 'subgroup_auc'].values[0]
    
    # Create FPR/FNR gap dataframes
    m_groups = m_groups.assign(
        auc_gap=abs(m_groups.subgroup_auc - overall_auc),
        bnsp_gap=abs(m_groups.bnsp_auc - overall_auc),  # False negative bias proxy
        bpsn_gap=abs(m_groups.bpsn_auc - overall_auc)   # False positive bias proxy
    )
    
    # FPR gap (using bpsn_gap as proxy)
    plt.figure(figsize=(8, 4))
    fpr_df = m_groups.sort_values("bpsn_gap", ascending=False)
    sns.barplot(x="subgroup", y="bpsn_gap", data=fpr_df)
    plt.title("FPR Gap by Subgroup (BPSN AUC difference)")
    plt.ylabel("|BPSN AUC – Overall AUC|")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir/"fpr_gap_dev.png")
    plt.close()
    
    # FNR gap (using bnsp_gap as proxy)
    plt.figure(figsize=(8, 4))
    fnr_df = m_groups.sort_values("bnsp_gap", ascending=False)
    sns.barplot(x="subgroup", y="bnsp_gap", data=fnr_df)
    plt.title("FNR Gap by Subgroup (BNSP AUC difference)")
    plt.ylabel("|BNSP AUC – Overall AUC|")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir/"fnr_gap_dev.png")
    plt.close()
    
    # Bias gap overview (combined visualization)
    plt.figure(figsize=(10, 6))
    
    # Create a melted dataframe for the three types of gaps
    gap_df = m_groups.melt(
        id_vars=["subgroup"], 
        value_vars=["auc_gap", "bnsp_gap", "bpsn_gap"],
        var_name="gap_type", 
        value_name="gap_value"
    )
    
    # Replace technical names with more readable ones
    gap_df["gap_type"] = gap_df["gap_type"].replace({
        "auc_gap": "Overall AUC Gap",
        "bnsp_gap": "False Negative Bias",
        "bpsn_gap": "False Positive Bias"
    })
    
    # Create the grouped bar plot
    sns.barplot(x="subgroup", y="gap_value", hue="gap_type", data=gap_df)
    plt.title("Bias Gap Analysis by Subgroup")
    plt.ylabel("Absolute Gap")
    plt.xlabel("Demographic Group")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Gap Type")
    plt.tight_layout()
    plt.savefig(out_dir/"bias_gap_dev.png")
    plt.close()
    
    print(f"✅ Generated error gap visualizations saved to {out_dir}")

if __name__ == "__main__":
    main() 