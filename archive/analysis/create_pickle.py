#!/usr/bin/env python
"""
Simple script to save predictions to pickle format for later analysis
"""
import pandas as pd
import numpy as np
import pickle
import shutil
import os

def main():
    # Load predictions
    pred_path = "output/large_predictions/predictions.csv"
    print(f"Loading predictions from {pred_path}")
    df = pd.read_csv(pred_path)
    
    # Create directories if they don't exist
    os.makedirs("output/large_predictions", exist_ok=True)
    
    # Save basic information about the predictions
    print(f"Loaded {len(df)} examples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic stats on predictions
    pred_mean = df["prediction"].mean()
    pred_std = df["prediction"].std()
    pred_min = df["prediction"].min()
    pred_max = df["prediction"].max()
    
    print(f"Prediction Mean: {pred_mean:.4f}")
    print(f"Prediction Std Dev: {pred_std:.4f}")
    print(f"Prediction Min: {pred_min:.4f}")
    print(f"Prediction Max: {pred_max:.4f}")
    
    # If target exists, get correlation
    corr = None
    if "target" in df.columns:
        corr = df[["target", "prediction"]].corr().iloc[0, 1]
        print(f"Correlation between target and prediction: {corr:.4f}")
    
    # Create results dataframe
    results = {
        "model": "large_simplified",
        "num_examples": len(df),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "correlation": corr
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv("output/large_predictions/results.csv", index=False)
    print(f"Saved results to output/large_predictions/results.csv")
    
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
    
    # Create a dictionary of model data
    model_data = {
        "predictions": df["prediction"].values,
        "results": results
    }
    
    # Add target if it exists
    if "target" in df.columns:
        model_data["targets"] = df["target"].values
    
    # Add text if it exists
    if "comment_text" in df.columns:
        model_data["texts"] = df["comment_text"].values
    
    # Save to pickle
    with open(pickle_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Saved pickle file to {pickle_path}")
    print(f"Pickle file size: {os.path.getsize(pickle_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main() 