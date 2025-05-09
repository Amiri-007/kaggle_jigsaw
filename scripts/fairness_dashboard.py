#!/usr/bin/env python
"""
Fairness Dashboard
=================
An interactive dashboard for exploring fairness metrics across demographic groups.

Run with:
    streamlit run scripts/fairness_dashboard.py

This dashboard loads data from:
- output/fairness_v2_summary.csv (from audit_fairness_v2.py)
- results/metrics_*.csv (from write_metrics.py)
- Any confusion matrix data in figs/fairness/

Features:
- Interactive selection of demographic groups
- Visualization of key fairness metrics
- Comparison of multiple models
- Filtering by disparity thresholds
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
import sys
import altair as alt

# Add project root to path to access fairness modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness.metrics_v2 import list_identity_columns

# Set page config
st.set_page_config(
    page_title="Fairness Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper functions
def load_fairness_summary():
    """Load fairness summary data"""
    summary_path = Path("output/fairness_v2_summary.csv")
    if not summary_path.exists():
        st.error(f"File not found: {summary_path}")
        return None
    
    return pd.read_csv(summary_path)

def load_metrics_files():
    """Load all metrics files from results directory"""
    metrics_files = list(Path("results").glob("metrics_*.csv"))
    
    if not metrics_files:
        st.warning("No metrics files found in results directory.")
        return {}
    
    metrics_data = {}
    for file_path in metrics_files:
        model_name = file_path.stem.replace("metrics_", "")
        try:
            metrics_data[model_name] = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    
    return metrics_data

def load_images():
    """Load fairness visualization images"""
    image_paths = {
        "Selection Rate": "figs/fairness_v2/selection_rate.png",
        "DP Difference": "figs/fairness_v2/dp_difference.png",
        "DP Ratio": "figs/fairness_v2/dp_ratio.png",
        "FPR Disparity": "figs/fairness_v2/fpr_disparity.png",
        "FNR Disparity": "figs/fairness_v2/fnr_disparity.png",
    }
    
    # Check which images exist
    valid_images = {}
    for name, path in image_paths.items():
        if Path(path).exists():
            valid_images[name] = path
    
    return valid_images

def create_disparity_chart(df, metric_col, threshold=0.2):
    """Create an interactive disparity chart"""
    if df is None:
        return None
    
    # Calculate whether each group is within threshold
    df['within_threshold'] = df[metric_col].between(1 - threshold, 1 + threshold)
    
    # Create chart
    chart = alt.Chart(df.reset_index()).mark_bar().encode(
        x=alt.X('index:N', title='Demographic Group', sort='-y'),
        y=alt.Y(f'{metric_col}:Q', title='Disparity Ratio'),
        color=alt.condition(
            alt.datum.within_threshold,
            alt.value('steelblue'),
            alt.value('firebrick')
        ),
        tooltip=['index', metric_col]
    ).properties(
        width=600,
        height=400,
        title=f'{metric_col} by Demographic Group'
    )
    
    # Add reference line at 1.0 (parity)
    reference_line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color='black', strokeDash=[5, 5]).encode(y='y')
    
    # Add threshold lines
    upper_threshold = alt.Chart(pd.DataFrame({'y': [1 + threshold]})).mark_rule(color='red', strokeDash=[2, 2]).encode(y='y')
    lower_threshold = alt.Chart(pd.DataFrame({'y': [1 - threshold]})).mark_rule(color='red', strokeDash=[2, 2]).encode(y='y')
    
    return chart + reference_line + upper_threshold + lower_threshold

def create_model_comparison_chart(metrics_data):
    """Create a chart comparing AUC across models and demographic groups"""
    if not metrics_data:
        return None
    
    # Combine data from all models
    comparison_data = []
    for model_name, df in metrics_data.items():
        # Only use rows with subgroup data (not overall)
        model_df = df[df['subgroup'] != 'overall'].copy()
        model_df['model'] = model_name
        comparison_data.append(model_df)
    
    if not comparison_data:
        return None
    
    # Combine into single dataframe
    combined_df = pd.concat(comparison_data)
    
    # Create chart
    chart = alt.Chart(combined_df).mark_bar().encode(
        x=alt.X('subgroup:N', title='Demographic Group'),
        y=alt.Y('subgroup_auc:Q', title='AUC Score'),
        color='model:N',
        column=alt.Column('model:N', title='Model'),
        tooltip=['model', 'subgroup', 'subgroup_auc']
    ).properties(
        width=150,
        height=300,
        title='AUC by Demographic Group Across Models'
    )
    
    return chart

def main():
    # Set up sidebar
    st.sidebar.title("Fairness Dashboard")
    st.sidebar.markdown("Explore fairness metrics across demographic groups.")
    
    # Load data
    fairness_summary = load_fairness_summary()
    metrics_data = load_metrics_files()
    images = load_images()
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Key Metrics", "Model Comparison", "Fairness Requirements"])
    
    with tab1:
        st.header("Key Fairness Metrics")
        
        if fairness_summary is None:
            st.warning("No fairness summary data available. Run audit_fairness_v2.py first.")
        else:
            # Show interactive metrics
            threshold = st.slider("Disparity Threshold", 0.1, 0.5, 0.2, 0.05, 
                                 help="Sets the acceptable range for disparity ratios (1±threshold)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("FPR Disparity")
                fpr_chart = create_disparity_chart(fairness_summary, 'fpr_disparity', threshold)
                if fpr_chart:
                    st.altair_chart(fpr_chart, use_container_width=True)
                
                # Display groups outside threshold
                st.subheader("Groups Outside FPR Threshold")
                outside_threshold = fairness_summary[
                    (fairness_summary['fpr_disparity'] < 1 - threshold) | 
                    (fairness_summary['fpr_disparity'] > 1 + threshold)
                ][['fpr_disparity']].sort_values('fpr_disparity')
                
                if not outside_threshold.empty:
                    st.dataframe(outside_threshold)
                else:
                    st.success("All groups within threshold!")
            
            with col2:
                st.subheader("FNR Disparity")
                fnr_chart = create_disparity_chart(fairness_summary, 'fnr_disparity', threshold)
                if fnr_chart:
                    st.altair_chart(fnr_chart, use_container_width=True)
                
                # Display groups outside threshold
                st.subheader("Groups Outside FNR Threshold")
                outside_threshold = fairness_summary[
                    (fairness_summary['fnr_disparity'] < 1 - threshold) | 
                    (fairness_summary['fnr_disparity'] > 1 + threshold)
                ][['fnr_disparity']].sort_values('fnr_disparity')
                
                if not outside_threshold.empty:
                    st.dataframe(outside_threshold)
                else:
                    st.success("All groups within threshold!")
            
            # Selection rate and demographic parity
            st.subheader("Selection Rate and Demographic Parity")
            col1, col2 = st.columns(2)
            
            with col1:
                # Selection rate bar chart
                if 'sel_rate' in fairness_summary.columns:
                    sel_rate_data = fairness_summary['sel_rate'].sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=sel_rate_data.index, y=sel_rate_data.values, palette='viridis', ax=ax)
                    plt.xticks(rotation=60, ha='right')
                    plt.title('Selection Rate by Group')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with col2:
                # DP ratio chart
                dp_chart = create_disparity_chart(fairness_summary, 'dp_ratio', threshold)
                if dp_chart:
                    st.altair_chart(dp_chart, use_container_width=True)
    
    with tab2:
        st.header("Model Comparison")
        
        if not metrics_data:
            st.warning("No model metrics data available. Run write_metrics.py first.")
        else:
            comparison_chart = create_model_comparison_chart(metrics_data)
            if comparison_chart:
                st.altair_chart(comparison_chart, use_container_width=True)
            
            # Show summary table
            st.subheader("Model Summary")
            summary_rows = []
            
            for model_name, df in metrics_data.items():
                overall_auc = df.loc[df['subgroup'] == 'overall', 'subgroup_auc'].iloc[0] if 'subgroup_auc' in df.columns else np.nan
                min_subgroup_auc = df.loc[df['subgroup'] != 'overall', 'subgroup_auc'].min() if 'subgroup_auc' in df.columns else np.nan
                max_subgroup_auc = df.loc[df['subgroup'] != 'overall', 'subgroup_auc'].max() if 'subgroup_auc' in df.columns else np.nan
                worst_subgroup = df.loc[df['subgroup_auc'].idxmin(), 'subgroup'] if 'subgroup_auc' in df.columns else "N/A"
                
                summary_rows.append({
                    'Model': model_name,
                    'Overall AUC': overall_auc,
                    'Min Subgroup AUC': min_subgroup_auc,
                    'Max Subgroup AUC': max_subgroup_auc,
                    'AUC Range': max_subgroup_auc - min_subgroup_auc,
                    'Worst Subgroup': worst_subgroup
                })
            
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df.set_index('Model'), use_container_width=True)
    
    with tab3:
        st.header("Fairness Requirements Compliance")
        
        # Load compliance report if it exists
        compliance_path = Path("output/compliance_report.md")
        if compliance_path.exists():
            with open(compliance_path, 'r') as f:
                compliance_report = f.read()
            st.markdown(compliance_report)
        else:
            st.warning("No compliance report found. Run check_compliance.py first.")
            
            # Show compliance check button
            if st.button("Run Compliance Check"):
                try:
                    st.info("Running compliance check...")
                    import subprocess
                    result = subprocess.run(["python", "scripts/check_compliance.py"], 
                                           capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("Compliance check completed successfully!")
                        # Reload the page to show the new report
                        st.experimental_rerun()
                    else:
                        st.error(f"Compliance check failed: {result.stderr}")
                except Exception as e:
                    st.error(f"Error running compliance check: {e}")

if __name__ == "__main__":
    main() 