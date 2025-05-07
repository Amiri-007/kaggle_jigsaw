#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Fairness Dashboard

A Streamlit app for exploring bias metrics and threshold analysis
across demographic subgroups.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics_v2 import compute_all_metrics
from src.vis_utils import (
    plot_auc_heatmap, 
    plot_threshold_sweep, 
    plot_fairness_radar,
    plot_power_mean_bars
)

# Set page configuration
st.set_page_config(
    page_title="Jigsaw Bias Audit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_available_models():
    """Get a list of available model metrics files."""
    metrics_files = glob.glob(os.path.join(RESULTS_DIR, "metrics_*.csv"))
    model_names = [os.path.basename(f).replace("metrics_", "").replace(".csv", "") 
                  for f in metrics_files]
    return sorted(model_names)


def load_model_metrics(model_name):
    """Load metrics data for a specific model."""
    metrics_file = os.path.join(RESULTS_DIR, f"metrics_{model_name}.csv")
    if not os.path.exists(metrics_file):
        st.error(f"Metrics file not found: {metrics_file}")
        return None
    
    return pd.read_csv(metrics_file)


def load_predictions(model_name):
    """Load prediction data for a specific model."""
    pred_file = os.path.join(RESULTS_DIR, f"preds_{model_name}.csv")
    if not os.path.exists(pred_file):
        st.error(f"Predictions file not found: {pred_file}")
        return None
    
    return pd.read_csv(pred_file)


def load_ground_truth():
    """Load ground truth data with identity columns."""
    # First try test data (for real evaluation)
    test_file = os.path.join(DATA_DIR, "test_public_expanded.csv")
    if os.path.exists(test_file):
        return pd.read_csv(test_file)
    
    # Fall back to train data
    train_file = os.path.join(DATA_DIR, "train.csv")
    if os.path.exists(train_file):
        return pd.read_csv(train_file)
    
    st.error("No ground truth data found. Please add train.csv or test_public_expanded.csv to the data directory.")
    return None


def metrics_explorer_page():
    """Render the Metrics Explorer page."""
    st.title("Metrics Explorer")
    
    # Sidebar for model selection
    model_names = get_available_models()
    if not model_names:
        st.error("No model metrics files found. Please run the bias evaluation notebook first.")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_names,
        index=0
    )
    
    # Load metrics for the selected model
    metrics_df = load_model_metrics(selected_model)
    if metrics_df is None:
        return
    
    # Calculate summary metrics if needed
    if 'final_score' not in metrics_df.columns:
        st.warning("Summary metrics not found in the metrics file. Some visualizations may be unavailable.")
    
    # Add model name to metrics dataframe for plotting functions
    metrics_df['model_name'] = selected_model
    
    # Identity filter in sidebar
    all_identities = metrics_df['subgroup_name'].unique().tolist()
    selected_identities = st.sidebar.multiselect(
        "Filter Identity Subgroups",
        all_identities,
        default=all_identities
    )
    
    if not selected_identities:
        st.warning("Please select at least one identity subgroup.")
        return
    
    # Filter metrics by selected identities
    filtered_metrics = metrics_df[metrics_df['subgroup_name'].isin(selected_identities)]
    
    # Create heatmap
    st.subheader("Bias Metrics Heatmap")
    
    # Create a temporary CSV for the heatmap function
    temp_file = os.path.join(RESULTS_DIR, "temp_metrics.csv")
    filtered_metrics.to_csv(temp_file, index=False)
    
    fig = plot_auc_heatmap(
        temp_file,
        title=f"Bias Metrics for {selected_model}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Power Mean Bar Chart
    st.subheader("Power Mean Performance")
    
    # Summary metrics dataframe for the selected model
    summary_metrics = pd.DataFrame({
        'model_name': [selected_model],
        'power_mean_subgroup_auc': [metrics_df['subgroup_auc'].mean()],
        'power_mean_bpsn_auc': [metrics_df['bpsn_auc'].mean()],
        'power_mean_bnsp_auc': [metrics_df['bnsp_auc'].mean()],
        'overall_auc': [metrics_df.get('overall_auc', [0.75])[0]],
        'final_score': [metrics_df.get('final_score', [0.75])[0]]
    })
    
    fig = plot_power_mean_bars(summary_metrics, selected_model)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed metrics table
    with st.expander("View Detailed Metrics Table"):
        st.dataframe(filtered_metrics)


def threshold_playground_page():
    """Render the Threshold Playground page."""
    st.title("Threshold Playground")
    
    # Load data
    model_names = get_available_models()
    if not model_names:
        st.error("No model predictions found. Please run the bias evaluation notebook first.")
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_names,
        index=0
    )
    
    # Load ground truth and predictions
    ground_truth = load_ground_truth()
    predictions = load_predictions(selected_model)
    
    if ground_truth is None or predictions is None:
        return
    
    # Merge data
    merged_data = pd.merge(ground_truth, predictions, on='id')
    
    # Get identity columns
    identity_columns = [col for col in ground_truth.columns 
                        if col not in ['id', 'comment_text', 'target', 'toxicity', 'severe_toxicity', 
                                      'obscene', 'threat', 'insult', 'sexual_explicit']]
    
    if not identity_columns:
        st.error("No identity columns found in the ground truth data.")
        return
    
    # Identity selection
    selected_identity = st.sidebar.selectbox(
        "Select Identity for Analysis",
        identity_columns
    )
    
    # Calculate prevalence
    identity_prevalence = ground_truth[selected_identity].mean() * 100
    st.sidebar.metric(
        f"{selected_identity} Prevalence", 
        f"{identity_prevalence:.2f}%",
        help="Percentage of examples with this identity attribute"
    )
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    
    # Variables for analysis
    y_true = merged_data['target'].values
    y_pred = merged_data['prediction'].values
    subgroup_mask = merged_data[selected_identity].values.astype(bool)
    
    # Display threshold analysis plot
    st.subheader(f"Threshold Analysis for {selected_identity}")
    
    fig = plot_threshold_sweep(
        y_true, 
        y_pred, 
        subgroup_mask, 
        selected_identity
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate metrics at the selected threshold
    col1, col2 = st.columns(2)
    
    # Apply threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Subgroup metrics
    subgroup_y_true = y_true[subgroup_mask]
    subgroup_y_pred = y_pred_binary[subgroup_mask]
    
    # Background metrics
    background_mask = ~subgroup_mask
    background_y_true = y_true[background_mask]
    background_y_pred = y_pred_binary[background_mask]
    
    # Calculate confusion matrix metrics
    def calculate_confusion_metrics(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Handle division by zero
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1 Score": f1,
            "FPR": fpr,
            "FNR": fnr
        }
    
    # Calculate metrics
    subgroup_metrics = calculate_confusion_metrics(subgroup_y_true, subgroup_y_pred)
    background_metrics = calculate_confusion_metrics(background_y_true, background_y_pred)
    
    # Display metrics
    with col1:
        st.subheader(f"{selected_identity} Group Metrics")
        for metric, value in subgroup_metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.subheader("Background Group Metrics")
        for metric, value in background_metrics.items():
            bg_value = background_metrics[metric]
            delta = value - bg_value
            st.metric(metric, f"{bg_value:.4f}", f"{delta:+.4f}", delta_color="off")
    
    # Display explanation of threshold effects
    st.subheader("Understanding Threshold Effects")
    st.markdown("""
    The threshold value determines how conservative or liberal the model is in making positive predictions. 
    A lower threshold results in more positive predictions (higher recall, lower precision), while a higher 
    threshold leads to fewer positive predictions (lower recall, higher precision).
    
    **Key fairness considerations:**
    
    * **FPR Gap**: The difference in False Positive Rate between the subgroup and background. 
      A large gap indicates the model falsely flags one group more than others.
    
    * **FNR Gap**: The difference in False Negative Rate between the subgroup and background.
      A large gap indicates the model misses harmful content for one group more than others.
    
    * **Optimal Threshold**: The threshold that minimizes the maximum of the FPR and FNR gaps,
      representing a fairness-optimal operating point.
    
    Adjusting the threshold is a post-processing technique for fairness that doesn't require retraining
    the model, but it represents a trade-off between different fairness metrics and overall performance.
    """)


def main():
    """Main function to run the Streamlit dashboard."""
    # Create sidebar navigation
    st.sidebar.title("Jigsaw Bias Audit")
    st.sidebar.image("https://jigsaw.google.com/static/images/jigsaw-logo.svg", width=200)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Metrics Explorer", "Threshold Playground"]
    )
    
    # Create directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Display the selected page
    if page == "Metrics Explorer":
        metrics_explorer_page()
    else:
        threshold_playground_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Jigsaw Unintended Bias Audit Dashboard")
    st.sidebar.caption("© 2025 Fairness Metrics Team")


if __name__ == "__main__":
    main() 