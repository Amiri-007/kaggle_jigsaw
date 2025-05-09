#!/usr/bin/env python
"""
Custom prediction script for toxicity classification
Uses the existing DistilBERT checkpoint to make predictions on a larger dataset
"""
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add project root to path for imports
sys.path.append(".")
from src.models.bert_headtail import BertHeadTailClassifier

class ToxicityDataset(Dataset):
    """Dataset for toxicity prediction"""
    def __init__(self, df, tokenizer, text_column="comment_text", max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_column]
        text = str(text) if not pd.isna(text) else ""
        
        # Tokenize with explicit padding to max_len
        encoding = self.tokenizer(
            text, 
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "index": idx
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions using existing model")
    parser.add_argument("--checkpoint", default="output/checkpoints/distilbert_headtail_fold0.pth",
                      help="Path to model checkpoint")
    parser.add_argument("--data-file", default="data/valid.csv",
                      help="Path to data file for predictions")
    parser.add_argument("--output-csv", default="output/large_predictions/large_predictions.csv",
                      help="Path to save predictions")
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Batch size for predictions")
    parser.add_argument("--text-col", default="comment_text",
                      help="Column name containing the text")
    parser.add_argument("--target-col", default="target",
                      help="Column name containing the target (if available)")
    parser.add_argument("--max-len", type=int, default=128,
                      help="Maximum sequence length")
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "distilbert-base-uncased")
    max_len = config.get("max_len", 128)
    max_head_len = config.get("max_head_len", 64)
    
    # Create model
    model = BertHeadTailClassifier(
        model_name=model_name,
        max_len=max_len,
        max_head_len=max_head_len
    )
    
    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: No state_dict found in checkpoint")
        
    model.to(device)
    model.eval()
    
    return model, model.tokenizer, device

def predict(model, tokenizer, df, text_col, device, batch_size=16, max_len=128):
    """Generate predictions for a dataframe"""
    # Create dataset
    dataset = ToxicityDataset(df, tokenizer, text_column=text_col, max_len=max_len)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Generate predictions
    all_preds = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            # Convert to probabilities
            logits = outputs
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            
            # Store indices for ordering
            all_indices.extend(batch["index"].cpu().numpy())
    
    # Create predictions dataframe
    df_preds = pd.DataFrame({
        "index": all_indices,
        "prediction": all_preds
    })
    df_preds = df_preds.sort_values("index").reset_index(drop=True)
    predictions = df_preds["prediction"].values
    
    return predictions

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer, device = load_model(args.checkpoint)
    
    # Load data
    print(f"Loading data from {args.data_file}")
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} examples")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predict(
        model, 
        tokenizer, 
        df, 
        args.text_col, 
        device, 
        args.batch_size, 
        args.max_len
    )
    
    # Save predictions
    print(f"Saving predictions to {args.output_csv}")
    df_out = df.copy()
    df_out["prediction"] = predictions
    
    # Keep only necessary columns
    if args.target_col in df_out.columns:
        output_columns = ["id", args.text_col, args.target_col, "prediction"]
    else:
        output_columns = ["id", args.text_col, "prediction"]
    
    # Filter columns that exist
    output_columns = [col for col in output_columns if col in df_out.columns]
    
    # Add id column if it doesn't exist
    if "id" not in df_out.columns:
        df_out["id"] = np.arange(len(df_out))
    
    df_out[output_columns].to_csv(args.output_csv, index=False)
    print(f"Saved predictions for {len(df_out)} examples")

if __name__ == "__main__":
    main() 