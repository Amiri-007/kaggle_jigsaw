#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Baseline Model Training Script (Sample Version)

This script trains a TF-IDF + Logistic Regression model on a sample of the dataset
and evaluates it using the metrics_v2 module.
"""

import os
import sys
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
        features['text_length'] = X.apply(len)
        features['word_count'] = X.apply(lambda x: len(x.split()))
        features['unique_word_count'] = X.apply(lambda x: len(set(x.split())))
        features['caps_count'] = X.apply(lambda x: sum(1 for c in x if c.isupper()))
        features['exclamation_count'] = X.apply(lambda x: x.count('!'))
        features['question_count'] = X.apply(lambda x: x.count('?'))
        return features.values

def main():
    start_time = time.time()
    print("Loading data...")
    # Use a smaller sample for demonstration
    sample_frac = 0.2
    
    # Load and sample data
    df = pd.read_csv("data/train.csv")
    
    if sample_frac < 1.0:
        print(f"Using {sample_frac*100:.1f}% of the data for faster demonstration")
        df = df.sample(frac=sample_frac, random_state=42)
    
    print(f"Loaded data with {len(df):,} rows")
    
    # Preprocess text
    print("Preprocessing text...")
    df["comment_text"] = df["comment_text"].fillna("").apply(preprocess_text)
    
    # Set target column
    target_col = "target"
    
    # Check the target distribution
    print("Checking target distribution...")
    print(df[target_col].describe())
    
    # Create binary classification target for training the model
    # Using a threshold of 0.5 to match the metrics evaluation
    threshold = 0.5
    print(f"Binarizing target values with threshold {threshold}")
    df["binary_target"] = (df[target_col] >= threshold).astype(int)
    
    # Check the binary target distribution
    binary_counts = df["binary_target"].value_counts()
    print("Binary target distribution:")
    print(binary_counts)
    print(f"Positive class percentage: {100 * binary_counts[1] / len(df):.2f}%")
    
    # Split data into train and test sets
    print("Splitting data into train and validation sets...")
    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["binary_target"]
    )
    
    print(f"Training set size: {len(train_df):,}")
    print(f"Validation set size: {len(valid_df):,}")
    
    # Extract features and targets
    X_train = train_df["comment_text"]
    y_train = train_df["binary_target"]
    X_valid = valid_df["comment_text"]
    y_valid = valid_df["binary_target"]
    
    # Create and train the baseline model
    print("Training model...")
    
    # Create a feature union with TF-IDF and custom text features
    features = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            max_features=50000,  # Reduced for faster training
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
    
    # Create the full pipeline
    model = Pipeline([
        ('features', features),
        ('classifier', LogisticRegression(
            C=5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Start training
    train_start = time.time()
    print(f"Started training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model.fit(X_train, y_train)
    train_end = time.time()
    print(f"Training completed in {(train_end - train_start)/60:.2f} minutes")
    
    # Save the model
    model_name = f"sample_tfidf_logreg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join("output/models", f"{model_name}.pkl")
    
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Make predictions on validation set
    print("Making predictions on validation set...")
    valid_preds = model.predict_proba(X_valid)[:, 1]
    
    # Add predictions to validation dataframe
    valid_df["prediction"] = valid_preds
    
    # Save predictions
    preds_path = os.path.join("output/preds", f"{model_name}.csv")
    valid_df[["id", "prediction"]].to_csv(preds_path, index=False)
    
    # Evaluate using metrics_v2
    print("Evaluating model using metrics_v2...")
    
    # Get identity columns
    identity_cols = list_identity_columns(valid_df)
    print(f"Found {len(identity_cols)} identity columns: {', '.join(identity_cols)}")
    
    # Create subgroup masks
    subgroup_masks = {}
    for col in identity_cols:
        subgroup_masks[col] = (valid_df[col] > 0.5).values
    
    # For metrics evaluation, use binary targets
    y_true = (valid_df[target_col] >= threshold).astype(int).values
    y_pred = valid_df["prediction"].values
    
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
    
    # Display subgroup metrics
    print("\n=== WORST PERFORMING SUBGROUPS ===")
    print(bias_report.metrics.sort_values("subgroup_auc").head(10)[
        ["subgroup_name", "subgroup_size", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    ])
    
    # Display top 5 performing subgroups
    print("\n=== TOP PERFORMING SUBGROUPS ===")
    print(bias_report.metrics.sort_values("subgroup_auc", ascending=False).head(5)[
        ["subgroup_name", "subgroup_size", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    ])
    
    # Save detailed metrics
    metrics_path = os.path.join("results", f"metrics_{model_name}.csv")
    bias_report.metrics.to_csv(metrics_path, index=False)
    
    # Save results for dashboard
    results_pred_file = os.path.join("results", f"preds_{model_name}.csv")
    valid_df[["id", "prediction"]].to_csv(results_pred_file, index=False)
    
    print(f"\nDetailed metrics saved to {metrics_path}")
    print(f"Predictions saved to {results_pred_file}")
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\nTotal execution time: {total_time:.2f} minutes")
    print("\nTraining and evaluation complete.")

if __name__ == "__main__":
    main() 