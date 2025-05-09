#!/usr/bin/env python
"""
Simple script to calculate metrics on toxicity predictions
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error

def main():
    # Load predictions
    pred_path = "output/large_predictions/predictions.csv"
    print(f"Loading predictions from {pred_path}")
    df = pd.read_csv(pred_path)
    
    # Check if target column exists
    if "target" not in df.columns:
        print("Error: No target column found in predictions file")
        return
    
    # Calculate metrics
    y_true = df["target"].values
    y_pred = df["prediction"].values
    
    # Check if the targets are binary or continuous
    unique_targets = np.unique(y_true)
    print(f"Unique target values: {unique_targets}")
    
    # Convert targets to binary for ROC AUC calculation
    # Using 0.5 as a threshold for both true and predicted values
    y_true_binary = (y_true >= 0.5).astype(int)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Calculate AUC-ROC with binary targets
    auc = roc_auc_score(y_true_binary, y_pred)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Calculate other metrics with binary values
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate regression metrics for continuous values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save metrics to file
    metrics = {
        "model": "large_simplified",
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("output/large_predictions/metrics.csv", index=False)
    print(f"Saved metrics to output/large_predictions/metrics.csv")
    
    # Create a pickle file copy
    import pickle
    import shutil
    
    # Copy the model checkpoint as a reference
    model_path = "output/checkpoints/distilbert_headtail_fold0.pth"
    target_path = "output/large_predictions/distilbert_headtail_large.pth"
    try:
        shutil.copy(model_path, target_path)
        print(f"Copied model checkpoint to {target_path}")
    except Exception as e:
        print(f"Error copying model: {e}")
    
    # Save predictions to pickle
    pickle_path = "output/large_predictions/predictions.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "y_true": y_true,
            "y_pred": y_pred,
            "y_true_binary": y_true_binary,
            "y_pred_binary": y_pred_binary,
            "metrics": metrics
        }, f)
    print(f"Saved pickle file to {pickle_path}")

if __name__ == "__main__":
    main() 