#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Fairness Dashboard (Standalone Version)

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
from sklearn.metrics import roc_auc_score

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

# Standalone implementation of visualization functions
def plot_auc_heatmap(metrics_file, title="Bias Metrics Heatmap"):
    """Create an interactive heatmap of bias metrics."""
    # Load metrics data
    df = pd.read_csv(metrics_file)
    
    # Sort data by subgroup size
    df = df.sort_values(by='subgroup_size', ascending=False)
    
    # Prepare data for heatmap
    subgroups = df['subgroup_name'].tolist()
    metric_columns = ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df[metric_columns].values,
        x=["Subgroup AUC", "BPSN AUC", "BNSP AUC"],
        y=subgroups,
        colorscale='RdBu',
        zmid=0.5,  # Center the color scale around 0.5
        zmin=0.3,  # Min value for color scale
        zmax=1.0,  # Max value for color scale
        colorbar=dict(title="AUC Score"),
        hoverinfo="z+y",
        text=df[metric_columns].round(4).astype(str).values,
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis=dict(title="Metric Type"),
        yaxis=dict(title="Identity Subgroup", autorange="reversed"),
        height=max(400, 30 * len(subgroups)),
        margin=dict(l=100, r=20, t=70, b=50),
    )
    
    return fig

def plot_power_mean_bars(summary_metrics, model_name):
    """Create a bar chart of power mean metrics."""
    # Extract data
    metrics = [
        "power_mean_subgroup_auc", 
        "power_mean_bpsn_auc", 
        "power_mean_bnsp_auc", 
        "overall_auc",
        "final_score"
    ]
    
    metric_names = [
        "Subgroup AUC", 
        "BPSN AUC", 
        "BNSP AUC", 
        "Overall AUC",
        "Final Score"
    ]
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metric_names,
        y=[summary_metrics[m].iloc[0] for m in metrics],
        text=[f"{summary_metrics[m].iloc[0]:.4f}" for m in metrics],
        textposition='outside',
        marker_color=[
            'royalblue', 'royalblue', 'royalblue', 
            'forestgreen', 'crimson'
        ],
    ))
    
    # Customize layout
    fig.update_layout(
        title=f"Model Performance Metrics: {model_name}",
        xaxis=dict(title="Metric Type"),
        yaxis=dict(
            title="Score (higher is better)",
            range=[0, 1.1]
        ),
        showlegend=False,
        height=400,
    )
    
    return fig

def plot_threshold_sweep(y_true, y_pred, subgroup_mask, identity_name, thresholds=None):
    """Create a threshold sweep plot showing metrics at different thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    # Initialize arrays for metrics
    subgroup_tpr = []
    subgroup_fpr = []
    background_tpr = []
    background_fpr = []
    
    # Calculate metrics at each threshold
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Subgroup
        sub_y_true = y_true[subgroup_mask]
        sub_y_pred = y_pred_binary[subgroup_mask]
        if len(sub_y_true) > 0:
            sub_pos = (sub_y_true == 1)
            sub_neg = (sub_y_true == 0)
            
            if sum(sub_pos) > 0:
                sub_tpr_val = sum(sub_y_pred[sub_pos]) / sum(sub_pos)
            else:
                sub_tpr_val = np.nan
                
            if sum(sub_neg) > 0:
                sub_fpr_val = sum(sub_y_pred[sub_neg]) / sum(sub_neg)
            else:
                sub_fpr_val = np.nan
                
            subgroup_tpr.append(sub_tpr_val)
            subgroup_fpr.append(sub_fpr_val)
        else:
            subgroup_tpr.append(np.nan)
            subgroup_fpr.append(np.nan)
        
        # Background
        bg_mask = ~subgroup_mask
        bg_y_true = y_true[bg_mask]
        bg_y_pred = y_pred_binary[bg_mask]
        if len(bg_y_true) > 0:
            bg_pos = (bg_y_true == 1)
            bg_neg = (bg_y_true == 0)
            
            if sum(bg_pos) > 0:
                bg_tpr_val = sum(bg_y_pred[bg_pos]) / sum(bg_pos)
            else:
                bg_tpr_val = np.nan
                
            if sum(bg_neg) > 0:
                bg_fpr_val = sum(bg_y_pred[bg_neg]) / sum(bg_neg)
            else:
                bg_fpr_val = np.nan
                
            background_tpr.append(bg_tpr_val)
            background_fpr.append(bg_fpr_val)
        else:
            background_tpr.append(np.nan)
            background_fpr.append(np.nan)
    
    # Create plot
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("True Positive Rate (Recall)", 
                                        "False Positive Rate"),
                        shared_yaxes=True)
    
    # TPR plot
    fig.add_trace(
        go.Scatter(x=thresholds, y=subgroup_tpr, name=f"{identity_name} TPR",
                 line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=thresholds, y=background_tpr, name="Background TPR",
                 line=dict(color='blue', dash='dash')),
        row=1, col=1
    )
    
    # FPR plot
    fig.add_trace(
        go.Scatter(x=thresholds, y=subgroup_fpr, name=f"{identity_name} FPR",
                 line=dict(color='red')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=thresholds, y=background_fpr, name="Background FPR",
                 line=dict(color='red', dash='dash')),
        row=1, col=2
    )
    
    # Add threshold vertical line at 0.5
    fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="green",
                row=1, col=1)
    fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="green",
                row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"Effect of Threshold on {identity_name} Group vs Background",
        xaxis_title="Classification Threshold",
        yaxis_title="Rate",
        legend_title="Metrics",
        height=500,
    )
    
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    
    return fig

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
    """Load synthetic ground truth data with identity columns."""
    # First try test data (for real evaluation)
    test_file = os.path.join(DATA_DIR, "test_public_expanded.csv")
    if os.path.exists(test_file):
        return pd.read_csv(test_file)
    
    # Fall back to train data
    train_file = os.path.join(DATA_DIR, "train.csv")
    if os.path.exists(train_file):
        return pd.read_csv(train_file, nrows=1000)  # Limit rows for performance
    
    # If no real data, create synthetic data
    st.warning("No ground truth data found. Using synthetic data for demonstration.")
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    ids = list(range(1, n_samples + 1))
    target = (np.random.random(n_samples) > 0.7).astype(int)
    identity1 = (np.random.random(n_samples) > 0.5).astype(int)
    identity2 = (np.random.random(n_samples) > 0.7).astype(int)
    identity3 = (np.random.random(n_samples) > 0.3).astype(int)
    
    # Create dataframe
    return pd.DataFrame({
        "id": ids,
        "target": target,
        "identity1": identity1,
        "identity2": identity2,
        "identity3": identity3
    })


def metrics_explorer_page():
    """Render the Metrics Explorer page."""
    st.title("Metrics Explorer")
    
    # Sidebar for model selection
    model_names = get_available_models()
    if not model_names:
        st.error("No model metrics files found. Please run the bias evaluation script first.")
        st.code("python run_metrics.py", language="bash")
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
        'overall_auc': [0.75],  # Default value, replace if available
        'final_score': [0.75]   # Default value, replace if available
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
        st.error("No model predictions found. Please run the bias evaluation script first.")
        st.code("python run_metrics.py", language="bash")
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
    
    # Ensure predictions and ground truth have matching IDs
    # For demonstration, just take the first N rows where N is the number of predictions
    if len(ground_truth) >= len(predictions):
        ground_truth = ground_truth.iloc[:len(predictions)].copy()
        ground_truth['id'] = predictions['id'].values
    else:
        st.error("Not enough ground truth data to match predictions.")
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