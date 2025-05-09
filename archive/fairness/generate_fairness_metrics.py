import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def generalized_power_mean(values, p=-5):
    valid_values = [v for v in values if not np.isnan(v)]
    if not valid_values:
        return np.nan
    return np.power(np.mean(np.power(valid_values, p)), 1/p)

def main():
    # Set up directories
    out_dir = Path("figs")
    out_dir.mkdir(exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Use existing metrics file
    metrics_file = "results/simple_metrics_distilbert_simplest.csv"
    print(f"Loading metrics from {metrics_file}")
    m = pd.read_csv(metrics_file)
    
    # 3.1 Heatmap
    print("Generating heatmap...")
    h = m.set_index("subgroup")[["subgroup_auc", "bpsn_auc", "bnsp_auc"]]
    plt.figure(figsize=(6, 8))
    sns.heatmap(h, annot=True, vmin=0.5, vmax=1.0, cmap="RdYlGn")
    plt.title("Fairness metrics heatmap – dev run")
    plt.tight_layout()
    plt.savefig(out_dir/"heatmap_dev.png")
    plt.close()
    
    # Calculate error-rate gaps
    print("Generating error-rate gap plots...")
    m_groups = m[m['subgroup'] != 'overall']
    
    # Calculate positive rate gap
    if 'subgroup_pos_rate' in m.columns and 'background_pos_rate' in m.columns:
        m_groups = m_groups.assign(
            pos_rate_gap=abs(m_groups.subgroup_pos_rate - m_groups.background_pos_rate)
        )
        
        # Create positive rate gap plot
        plt.figure(figsize=(8, 4))
        m_groups.sort_values("pos_rate_gap", ascending=False).plot.bar(x="subgroup", y="pos_rate_gap")
        plt.ylabel("Positive rate gap")
        plt.title("Gap between subgroup and background positive rates")
        plt.tight_layout()
        plt.savefig(out_dir/"pos_rate_gap_dev.png")
        plt.close()
    
    # Extract overall AUC
    overall_auc = m.loc[m['subgroup'] == 'overall', 'subgroup_auc'].values[0]
    print(f"Overall AUC: {overall_auc:.4f}")
    
    # Calculate power means
    bias_metrics = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    power_means = {}
    for metric in bias_metrics:
        power_means[metric] = generalized_power_mean(m_groups[metric].dropna().values)
        print(f"Power mean for {metric}: {power_means[metric]:.4f}")
    
    # Calculate final score
    final_score = 0.25 * overall_auc + 0.75 * np.mean([power_means[metric] for metric in bias_metrics])
    print(f"Final score: {final_score:.4f}")
    
    # Create and save summary
    summary = pd.DataFrame([{
        "model": "blend_dev",
        "overall_auc": overall_auc,
        "final_score": final_score
    }])
    summary.to_csv(results_dir/"summary.tsv", sep="\t", index=False)
    print("✅ Summary saved to results/summary.tsv")
    
    # Also save metrics as CSV for compatibility
    m.to_csv(results_dir/"audit_metrics.csv", index=False)
    print("✅ Metrics saved to results/audit_metrics.csv")
    
    # Generate demographic size visualization
    print("Generating demographic size visualization...")
    plt.figure(figsize=(10, 6))
    m_groups = m_groups.sort_values("subgroup_size", ascending=False)
    m_groups.plot.bar(x="subgroup", y="subgroup_size")
    plt.title("Size of demographic groups in validation set")
    plt.ylabel("Number of examples")
    plt.tight_layout()
    plt.savefig(out_dir/"demographic_group_sizes.png")
    plt.close()
    
    # Generate metrics comparison bar plot
    print("Generating metrics comparison visualization...")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=m_groups.melt(id_vars=['subgroup'], 
                                   value_vars=['subgroup_auc', 'bpsn_auc', 'bnsp_auc']),
               x='subgroup', y='value', hue='variable')
    plt.xticks(rotation=45, ha='right')
    plt.title("Fairness metrics comparison across demographic groups")
    plt.ylabel("AUC score")
    plt.legend(title="Metric type")
    plt.tight_layout()
    plt.savefig(out_dir/"fairness_metrics_comparison.png")
    plt.close()
    
    print(f"✅ Generated visualizations saved to {out_dir}")

if __name__ == "__main__":
    main() 