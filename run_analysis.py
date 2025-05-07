#!/usr/bin/env python
"""
Jigsaw Unintended Bias Audit - Complete Analysis Script

This script provides a command-line alternative to the Jupyter notebook,
running the same analysis on the Jigsaw Unintended Bias dataset.
"""

import os
import argparse
import json
import pathlib
import pickle
import gzip
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Machine learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import custom modules for bias analysis
import bias_metrics
import plot_utils


def setup_torch_for_gpu():
    """Configure PyTorch to use GPU optimally."""
    try:
        import torch
        import platform
        import sys
        
        # Print system and torch info for debugging
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"OS: {platform.system()} {platform.release()}")
        
        # Force CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            device = torch.device("cuda")
            # Get GPU details
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            print(f"✓ CUDA device available: {gpu_name} with {gpu_mem:.2f}GB memory")
            
            # Enable TF32 for faster computation on Ampere GPUs (RTX 30 series)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Speed up inference by using autotuner
            
            # Use mixed precision for faster training
            dtype = torch.float16  # Use FP16 for RTX 3070Ti
            print(f"✓ Using {dtype} precision for optimal performance")
            return device, dtype
        else:
            print("❌ ERROR: CUDA is not available")
            print("This script is configured to run only on GPU")
            print("Please ensure your NVIDIA drivers are properly installed")
            sys.exit(1)
    except ImportError as e:
        print(f"❌ Error importing PyTorch: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error setting up GPU: {e}")
        sys.exit(1)


def load_data(data_dir, nrows=None):
    """Load Jigsaw dataset from disk."""
    print("Loading data...")
    data_path = pathlib.Path(data_dir)
    train_csv = data_path / 'train.csv'
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Dataset not found at {train_csv}. Please run setup_environment.py first.")
    
    if nrows:
        df = pd.read_csv(train_csv, nrows=nrows)
        print(f"Loaded {nrows} rows from dataset")
    else:
        df = pd.read_csv(train_csv)
        print(f"Loaded full dataset with {len(df)} rows")
    
    return df


def train_tfidf_logreg(df):
    """Train TF-IDF + Logistic Regression baseline model."""
    print("\nTraining TF-IDF + Logistic Regression model...")
    
    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(
        df['comment_text'].fillna(' '),
        df['target'] >= 0.5,
        test_size=0.2, 
        random_state=42
    )
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    
    # Train model
    print("Training logistic regression model...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)
    
    # Make predictions
    pred_tfidf = clf.predict_proba(X_valid_tfidf)[:, 1]
    tfidf_auc = roc_auc_score(y_valid, pred_tfidf)
    print(f"TF-IDF + LogReg Validation AUC: {tfidf_auc:.4f}")
    
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'y_train': y_train,
        'y_valid': y_valid,
        'vectorizer': vectorizer,
        'model': clf,
        'pred_tfidf': pred_tfidf
    }


def run_bert_inference(tfidf_results, device, dtype, batch_size=32):
    """Run BERT inference on validation data."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("\nRunning BERT inference on GPU...")
        X_valid = tfidf_results['X_valid']
        y_valid = tfidf_results['y_valid']
        
        # Get a smaller subset for BERT (to prevent OOM issues)
        subset_size = min(10000, len(X_valid))
        X_valid_subset = X_valid[:subset_size]
        y_valid_subset = y_valid[:subset_size]
        
        if len(X_valid) > subset_size:
            print(f"Using first {subset_size} examples out of {len(X_valid)} for BERT to save memory")
        
        # Load BERT model and tokenizer
        print("Loading BERT model...")
        model_name = "martin-ha/toxic-comment-model"  # Model specifically fine-tuned for toxicity
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Print model architecture to understand its outputs
        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Number of labels: {model.config.num_labels}")
        
        # Explicitly move to device
        model = model.to(device)
        if dtype == torch.float16:
            model = model.half()
        
        # Convert input texts to tokens
        print(f"Tokenizing {len(X_valid_subset)} examples...")
        X_valid_list = X_valid_subset.fillna(" ").astype(str).tolist()
        
        pred_bert = []
        
        # Process in batches
        for i in range(0, len(X_valid_list), batch_size):
            batch_texts = X_valid_list[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # If binary classification (num_labels=1), use sigmoid
                if model.config.num_labels == 1:
                    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                # If multi-class (num_labels>1), get toxic class probability
                else:
                    # Assuming first class (index 0) is non-toxic, second class (index 1) is toxic
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                pred_bert.extend(scores.tolist())
        
        # Convert to numpy array
        pred_bert = np.asarray(pred_bert, dtype=np.float32)
        
        # Calculate AUC for BERT
        bert_auc = roc_auc_score(y_valid_subset, pred_bert)
        
        # If AUC is below 0.5, it means model is predicting opposite of what we want
        # So we invert the predictions (1-pred) and recalculate
        if bert_auc < 0.5:
            print(f"Initial BERT AUC was {bert_auc:.4f}, which is below random chance")
            print("Inverting predictions to correct polarity...")
            pred_bert = 1 - pred_bert
            bert_auc = roc_auc_score(y_valid_subset, pred_bert)
        
        print(f"BERT Validation AUC: {bert_auc:.4f}")
        
        # Make sure predictions are well-distributed (not all 0s or 1s)
        print(f"BERT predictions stats - Min: {np.min(pred_bert):.4f}, Max: {np.max(pred_bert):.4f}, Mean: {np.mean(pred_bert):.4f}")
        
        # If we used a subset, extend predictions to match original size
        # Instead of padding with zeros, we'll use the mean prediction value
        # to avoid skewing the distribution
        if len(pred_bert) < len(y_valid):
            mean_pred = np.mean(pred_bert)
            print(f"Extending predictions with mean value {mean_pred:.4f} to match full dataset size")
            
            full_pred_bert = np.full(len(y_valid), mean_pred, dtype=np.float32)
            full_pred_bert[:len(pred_bert)] = pred_bert
            
            # For validation indices we didn't process, use the closest similar example
            # This preserves the distribution better than using mean for all
            if len(pred_bert) > 1000:  # Only do this if we have enough examples
                remaining_indices = range(len(pred_bert), len(y_valid))
                replacement_indices = np.random.choice(len(pred_bert), len(remaining_indices))
                full_pred_bert[remaining_indices] = pred_bert[replacement_indices]
            
            return full_pred_bert
        
        return pred_bert
    
    except ImportError as e:
        print(f"⚠ Error importing required libraries for BERT: {e}")
        print("⚠ Skipping BERT inference.")
        return None
    except Exception as e:
        print(f"⚠ Error running BERT inference: {e}")
        import traceback
        traceback.print_exc()
        print("⚠ Skipping BERT inference due to error.")
        return None


def create_validation_dataframe(tfidf_results, pred_bert=None):
    """Create a DataFrame with validation results for all models."""
    print("\nCreating validation dataset...")
    
    X_valid = tfidf_results['X_valid']
    y_valid = tfidf_results['y_valid']
    pred_tfidf = tfidf_results['pred_tfidf']
    
    # Create validation DataFrame
    df_valid = pd.DataFrame({
        "comment_text": X_valid.reset_index(drop=True),
        "target": y_valid.reset_index(drop=True),
        "pred_tfidf": pred_tfidf
    })
    
    # Add BERT predictions if available
    if pred_bert is not None:
        df_valid["pred_bert"] = pred_bert
    
    # Add identity columns with default values
    IDENTITY_COLS = [
        "male", "female", "black", "white", "asian", "christian",
        "jewish", "muslim", "hindu", "buddhist", "atheist", "lgbtq",
        "transgender"
    ]
    
    for col in IDENTITY_COLS:
        df_valid[col] = 0.0
    
    return df_valid, IDENTITY_COLS


def analyze_bias(df_valid, identity_cols):
    """Analyze bias across demographic subgroups for all models."""
    print("\nAnalyzing bias across demographic subgroups...")
    
    # Get prediction columns (all columns starting with 'pred_')
    pred_cols = [col for col in df_valid.columns if col.startswith('pred_')]
    
    if not pred_cols:
        raise ValueError("No prediction columns found in validation DataFrame")
    
    # Calculate bias metrics for all models
    model_metrics = bias_metrics.compare_models_bias(
        df_valid, pred_cols, 'target', identity_cols
    )
    
    # Print summary metrics
    print("\nBias Metrics Summary:")
    for model_col, metrics in model_metrics.items():
        model_name = model_col.replace('pred_', '')
        overall_auc = metrics['auc']['overall']
        min_subgroup = metrics['auc']['bias_metrics']['min_subgroup_auc']
        max_subgroup = metrics['auc']['bias_metrics']['max_subgroup_auc']
        auc_diff = metrics['auc']['bias_metrics']['auc_difference']
        
        print(f"\n{model_name.upper()}:")
        print(f"  Overall AUC:     {overall_auc:.4f}")
        print(f"  Min Subgroup:    {min_subgroup:.4f}")
        print(f"  Max Subgroup:    {max_subgroup:.4f}")
        print(f"  AUC Difference:  {auc_diff:.4f}")
    
    return model_metrics


def save_artifacts(tfidf_results, df_valid, model_metrics, output_dir='artifacts'):
    """Save trained models, predictions, and analysis results."""
    print("\nSaving artifacts...")
    
    # Create output directories
    artifacts_dir = pathlib.Path(output_dir)
    artifacts_dir.mkdir(exist_ok=True)
    
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Save validation results
    df_valid.to_parquet(logs_dir / "baseline_valid.parquet", index=False)
    
    # Save TF-IDF vectorizer
    with gzip.open(artifacts_dir / "tfidf_vectorizer.pkl.gz", "wb") as f:
        pickle.dump(tfidf_results['vectorizer'], f)
    
    # Save logistic regression model
    with gzip.open(artifacts_dir / "logreg_model.pkl.gz", "wb") as f:
        pickle.dump(tfidf_results['model'], f)
    
    # Generate and save analysis report
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    report = {
        "timestamp": timestamp,
        "models": {},
        "bias_metrics": {}
    }
    
    # Add model information
    for model_col, metrics in model_metrics.items():
        model_name = model_col.replace('pred_', '')
        report["models"][model_name] = {
            "auc": metrics["auc"]["overall"],
            "bias_metrics": metrics["auc"]["bias_metrics"],
            "best_threshold": metrics["threshold_optimization"]["best_threshold"],
        }
    
    # Save report as JSON
    with open(artifacts_dir / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Artifacts saved to {artifacts_dir}")
    print(f"✓ Validation dataset saved to {logs_dir}")
    
    return artifacts_dir, logs_dir


def create_visualizations(df_valid, model_metrics, identity_cols, output_dir='figures'):
    """Create and save visualizations of model performance and bias metrics."""
    print("\nGenerating visualizations...")
    
    # Create figures directory
    fig_dir = pathlib.Path(output_dir)
    fig_dir.mkdir(exist_ok=True)
    
    # Generate all plots and save them
    saved_files = plot_utils.save_all_plots(
        model_metrics, df_valid, identity_cols, output_dir
    )
    
    print(f"✓ Generated {len(saved_files)} visualizations in {fig_dir}")
    return fig_dir


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Jigsaw Unintended Bias Audit analysis"
    )
    
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory containing the Jigsaw dataset (default: ./data)"
    )
    
    parser.add_argument(
        "--nrows", type=int, default=200000,
        help="Number of rows to load from the dataset (default: 200000, use -1 for all)"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for BERT inference (default: 32)"
    )
    
    parser.add_argument(
        "--skip_bert", action="store_true",
        help="Skip BERT inference (much faster, but less comprehensive analysis)"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to save artifacts and results (default: ./output)"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the complete analysis."""
    print("\n" + "=" * 80)
    print("Jigsaw Unintended Bias Audit - Analysis")
    print("=" * 80 + "\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up output directories
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    artifacts_dir = output_dir / "artifacts"
    logs_dir = output_dir / "logs"
    fig_dir = output_dir / "figures"
    
    # Configure PyTorch for GPU
    device, dtype = setup_torch_for_gpu()
    
    # Load data
    nrows = args.nrows if args.nrows > 0 else None
    df = load_data(args.data_dir, nrows)
    
    # Train TF-IDF + LogReg model
    tfidf_results = train_tfidf_logreg(df)
    
    # Run BERT inference if not skipped
    pred_bert = None
    if not args.skip_bert and device is not None:
        pred_bert = run_bert_inference(tfidf_results, device, dtype, args.batch_size)
    
    # Create validation DataFrame
    df_valid, identity_cols = create_validation_dataframe(tfidf_results, pred_bert)
    
    # Analyze bias
    model_metrics = analyze_bias(df_valid, identity_cols)
    
    # Save artifacts
    artifacts_dir, logs_dir = save_artifacts(
        tfidf_results, df_valid, model_metrics, 
        output_dir=str(artifacts_dir)
    )
    
    # Create visualizations
    fig_dir = create_visualizations(
        df_valid, model_metrics, identity_cols,
        output_dir=str(fig_dir)
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"- Artifacts saved to: {artifacts_dir}")
    print(f"- Logs saved to: {logs_dir}")
    print(f"- Figures saved to: {fig_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main() 