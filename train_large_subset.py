#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Large Subset Training Script

This script trains a TF-IDF + Logistic Regression model on 20% of the dataset
and evaluates it using the metrics_v2 module, optimized for RTX 3070Ti.
"""

import os
import sys
import argparse
import json
import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import pickle
import time
import plotly.graph_objects as go

# Add the src directory to the path to import metrics_v2 directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import metrics_v2
from metrics_v2 import (
    compute_all_metrics,
    list_identity_columns,
    BiasReport
)

# Create necessary directories
os.makedirs("results", exist_ok=True)
os.makedirs("output/models", exist_ok=True)
os.makedirs("output/preds", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/artifacts", exist_ok=True)

def preprocess_text(text):
    """Basic text preprocessing"""
    if isinstance(text, str):
        # Convert to lowercase and strip
        text = text.lower().strip()
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    return ""

# Custom transformer to extract text features
class TextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = pd.DataFrame()
        # More memory-efficient feature extraction
        features['text_length'] = X.apply(len)
        features['word_count'] = X.apply(lambda x: len(x.split()))
        features['unique_word_ratio'] = X.apply(lambda x: len(set(x.split())) / (len(x.split()) + 1))
        features['caps_ratio'] = X.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
        return features.values

def plot_auc_heatmap(metrics_df, title="Bias Metrics Heatmap", model_name="model"):
    """Create an interactive heatmap of bias metrics."""
    # Sort data by subgroup size
    df = metrics_df.sort_values(by='subgroup_size', ascending=False)
    
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

def create_analysis_report(bias_report, model_name):
    """Create a JSON analysis report with key metrics and findings."""
    # Get the worst and best performing subgroups
    metrics_df = bias_report.metrics.copy()
    worst_subgroups = metrics_df.sort_values("subgroup_auc").head(3)[["subgroup_name", "subgroup_auc", "subgroup_size"]]
    best_subgroups = metrics_df.sort_values("subgroup_auc", ascending=False).head(3)[["subgroup_name", "subgroup_auc", "subgroup_size"]]
    
    # Create report dictionary
    report = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance": {
            "overall_auc": float(bias_report.overall_auc),
            "final_score": float(bias_report.final_score),
        },
        "bias_analysis": {
            "identity_groups_count": len(metrics_df),
            "worst_performing_subgroups": [
                {
                    "name": row["subgroup_name"],
                    "auc": float(row["subgroup_auc"]),
                    "size": int(row["subgroup_size"])
                } for _, row in worst_subgroups.iterrows()
            ],
            "best_performing_subgroups": [
                {
                    "name": row["subgroup_name"],
                    "auc": float(row["subgroup_auc"]),
                    "size": int(row["subgroup_size"])
                } for _, row in best_subgroups.iterrows()
            ],
            "auc_range": {
                "min": float(metrics_df["subgroup_auc"].min()),
                "max": float(metrics_df["subgroup_auc"].max()),
                "range": float(metrics_df["subgroup_auc"].max() - metrics_df["subgroup_auc"].min())
            }
        },
        "recommendations": {
            "areas_of_concern": [sg["name"] for sg in report["bias_analysis"]["worst_performing_subgroups"]],
            "suggested_actions": [
                "Review training data distribution for underrepresented groups",
                "Consider data augmentation for worst performing subgroups",
                "Implement fairness constraints during model training"
            ]
        }
    }
    
    return report

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a bias-aware model on a large subset")
    parser.add_argument("--model-name", type=str, default="tfidf_lr_large",
                        help="Name for the model and output files")
    parser.add_argument("--sample-size", type=float, default=0.2,
                        help="Fraction of dataset to use (default: 0.2 = 20%)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for prediction to manage memory usage")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    model_name = args.model_name
    sample_size = args.sample_size
    threshold = args.threshold
    batch_size = args.batch_size
    
    start_time = time.time()
    print(f"Starting training for model: {model_name}")
    print(f"Using {sample_size*100:.1f}% of the dataset")
    
    print("Loading training data...")
    # Set sample fraction
    sample_frac = sample_size
    
    # Load train data with sampling
    train_df = pd.read_csv("data/train.csv")
    original_size = len(train_df)
    
    if sample_frac < 1.0:
        print(f"Sampling {sample_frac*100:.1f}% of the data ({int(original_size * sample_frac):,} rows)")
        train_df = train_df.sample(frac=sample_frac, random_state=42)
    
    print(f"Loaded training data with {len(train_df):,} rows")
    
    # Preprocess text
    print("Preprocessing text...")
    train_df["comment_text"] = train_df["comment_text"].fillna("").apply(preprocess_text)
    
    # Set target column
    target_col = "target"
    
    # Check the target distribution
    print("Checking target distribution...")
    print(train_df[target_col].describe())
    
    # Create binary classification target for training the model
    print(f"Binarizing target values with threshold {threshold}")
    train_df["binary_target"] = (train_df[target_col] >= threshold).astype(int)
    
    # Check the binary target distribution
    binary_counts = train_df["binary_target"].value_counts()
    print("Binary target distribution:")
    print(binary_counts)
    print(f"Positive class percentage: {100 * binary_counts[1] / len(train_df):.2f}%")
    
    # Split data into train and validation sets
    print("Splitting data into train and validation sets...")
    train_data, valid_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["binary_target"]
    )
    
    print(f"Training set size: {len(train_data):,}")
    print(f"Validation set size: {len(valid_data):,}")
    
    # Extract features and targets
    X_train = train_data["comment_text"]
    y_train = train_data["binary_target"]
    X_valid = valid_data["comment_text"]
    y_valid = valid_data["binary_target"]
    
    # Free up memory
    del train_df
    gc.collect()
    
    # Create and train the model
    print("Training model...")
    
    # Create a feature union with TF-IDF and custom text features
    features = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            max_features=70000,  # Reduced for memory efficiency
            min_df=5,
            max_df=0.9,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )),
        ('text_features', TextFeatures())
    ])
    
    # Create the full pipeline with memory-optimized settings
    model = Pipeline([
        ('features', features),
        ('classifier', LogisticRegression(
            C=5,
            max_iter=200,  # Reduced for faster convergence
            class_weight='balanced',
            random_state=42,
            solver='saga',  # Memory-efficient solver
            n_jobs=-1,     # Use all cores
            verbose=1      # Show progress
        ))
    ])
    
    # Start training
    train_start = time.time()
    print(f"Started training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model.fit(X_train, y_train)
    train_end = time.time()
    print(f"Training completed in {(train_end - train_start)/60:.2f} minutes")
    
    # Save the model
    model_path = os.path.join("output/models", f"{model_name}.pkl")
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Make predictions on validation set
    print("Making validation predictions...")
    valid_preds = model.predict_proba(X_valid)[:, 1]
    
    # Add predictions to validation dataframe
    valid_data["prediction"] = valid_preds
    
    # Save validation predictions
    valid_preds_path = os.path.join("output/preds", f"{model_name}_valid.csv")
    valid_data[["id", "prediction"]].to_csv(valid_preds_path, index=False)
    
    # Load test data and make predictions in batches
    print("Loading test data and making predictions in batches...")
    test_df = pd.read_csv("data/test_public_expanded.csv")
    print(f"Loaded test data with {len(test_df):,} rows")
    
    # Create predictions dataframe
    test_preds_df = pd.DataFrame({"id": test_df["id"]})
    test_preds_df["prediction"] = np.nan
    
    # Process in batches to manage memory
    n_batches = (len(test_df) + batch_size - 1) // batch_size
    print(f"Processing test data in {n_batches} batches of {batch_size} rows")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(test_df))
        print(f"Processing batch {i+1}/{n_batches} (rows {start_idx}-{end_idx})")
        
        # Get batch
        batch = test_df.iloc[start_idx:end_idx].copy()
        
        # Preprocess
        batch["comment_text"] = batch["comment_text"].fillna("").apply(preprocess_text)
        
        # Make predictions
        batch_preds = model.predict_proba(batch["comment_text"])[:, 1]
        
        # Store predictions
        test_preds_df.loc[start_idx:end_idx-1, "prediction"] = batch_preds
        
        # Free memory
        del batch, batch_preds
        gc.collect()
    
    # Save test predictions
    test_preds_path = os.path.join("output/preds", f"{model_name}.csv")
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions to {test_preds_path}")
    
    # Evaluate using metrics_v2 on validation data
    print("Evaluating model using metrics_v2...")
    
    # Get identity columns from validation data
    identity_cols = list_identity_columns(valid_data)
    print(f"Found {len(identity_cols)} identity columns: {', '.join(identity_cols)}")
    
    # Create subgroup masks for validation data
    subgroup_masks = {}
    for col in identity_cols:
        subgroup_masks[col] = (valid_data[col] > 0.5).values
    
    # Calculate metrics on validation data
    y_true = (valid_data[target_col] >= threshold).astype(int).values
    y_pred = valid_data["prediction"].values
    
    bias_report = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        subgroup_masks=subgroup_masks,
        model_name=model_name
    )
    
    # Display metrics
    print("\n=== MODEL EVALUATION ===")
    print(f"Model: {model_name}")
    print(f"Overall AUC: {bias_report.overall_auc:.4f}")
    print(f"Final Score: {bias_report.final_score:.4f}")
    
    # Display worst performing subgroups
    print("\n=== WORST PERFORMING SUBGROUPS ===")
    print(bias_report.metrics.sort_values("subgroup_auc").head(5)[
        ["subgroup_name", "subgroup_size", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    ])
    
    # Display top performing subgroups
    print("\n=== TOP PERFORMING SUBGROUPS ===")
    print(bias_report.metrics.sort_values("subgroup_auc", ascending=False).head(5)[
        ["subgroup_name", "subgroup_size", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    ])
    
    # Save metrics to CSV
    metrics_path = os.path.join("results", f"metrics_{model_name}.csv")
    bias_report.metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save overall metrics for dashboard
    overall_metrics_path = os.path.join("results", "overall_metrics.csv")
    if os.path.exists(overall_metrics_path):
        overall_df = pd.read_csv(overall_metrics_path)
        # Remove if model exists
        overall_df = overall_df[overall_df['model_name'] != model_name]
    else:
        overall_df = pd.DataFrame(columns=['model_name', 'overall_auc', 'final_score'])
    
    # Add new model metrics
    overall_df = pd.concat([
        overall_df,
        pd.DataFrame({
            'model_name': [model_name],
            'overall_auc': [bias_report.overall_auc],
            'final_score': [bias_report.final_score]
        })
    ], ignore_index=True)
    
    # Save overall metrics
    overall_df.to_csv(overall_metrics_path, index=False)
    print(f"Saved overall metrics to {overall_metrics_path}")
    
    # Save results for dashboard
    results_pred_file = os.path.join("results", f"preds_{model_name}.csv")
    valid_data[["id", "prediction"]].to_csv(results_pred_file, index=False)
    print(f"Saved dashboard predictions to {results_pred_file}")
    
    # Generate and save heatmap
    print("Generating heatmap visualization...")
    heatmap_fig = plot_auc_heatmap(
        bias_report.metrics, 
        title=f"Bias Metrics Heatmap - {model_name}", 
        model_name=model_name
    )
    heatmap_path = os.path.join("output/figures", f"heatmap_{model_name}.svg")
    heatmap_fig.write_image(heatmap_path, scale=2)
    print(f"Saved heatmap to {heatmap_path}")
    
    # Create analysis report
    print("Creating analysis report...")
    analysis_report = create_analysis_report(bias_report, model_name)
    report_path = os.path.join("output/artifacts", f"analysis_report_{model_name}.json")
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    print(f"Saved analysis report to {report_path}")
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\nTotal execution time: {total_time:.2f} minutes")
    print("\nTraining, evaluation, and analysis complete.")

if __name__ == "__main__":
    main() 