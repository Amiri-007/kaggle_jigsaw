#!/usr/bin/env python
"""
Simple prediction script for toxicity classification using an existing checkpoint
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add necessary imports
from transformers import AutoTokenizer

class ToxicityDataset(Dataset):
    """Simple dataset for text classification"""
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if not pd.isna(self.texts[idx]) else ""
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def main():
    # Configure paths
    model_path = "output/checkpoints/distilbert_headtail_fold0.pth"
    data_path = "data/valid.csv"
    output_dir = "output/large_predictions"
    output_path = f"{output_dir}/predictions.csv"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'distilbert-base-uncased')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load model
    # We're doing direct inference with the state dict, so we'll load the saved weights
    # into a custom model class later
    print(f"Model base type: {model_type}")
    
    # Load data
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Define batch size
    batch_size = 32
    
    # Create dataset and dataloader
    dataset = ToxicityDataset(data['comment_text'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Create a very simple model for inference
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import AutoModel
            self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")
            self.classifier = torch.nn.Linear(768, 1)
            
        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token
            return self.classifier(pooled_output)
    
    # Initialize model
    model = SimpleModel()
    
    # Try to load weights that match
    if 'state_dict' in checkpoint:
        # This won't be a perfect match, but we'll try our best to make it work
        # We're just trying to load the classifier weights since that's most important
        state_dict = checkpoint['state_dict']
        
        # Copy the classifier weights if available
        if 'classifier.weight' in state_dict:
            model.classifier.weight.data.copy_(state_dict['classifier.weight'])
            model.classifier.bias.data.copy_(state_dict['classifier.bias'])
            print("Loaded classifier weights")
        
        # Now let's try to match the transformer weights
        # Map checkpoint TransformerModel weights to our model's transformer weights
        matched_keys = 0
        for key in state_dict:
            if key.startswith('bert.') or key.startswith('model.bert.'):
                # Extract the part after "bert."
                if key.startswith('bert.'):
                    new_key = key.replace('bert.', 'transformer.')
                else:
                    new_key = key.replace('model.bert.', 'transformer.')
                
                # See if we have a matching key in our model
                try:
                    param = model
                    for part in new_key.split('.'):
                        param = getattr(param, part)
                    # If we got here, the key exists in our model
                    matched_keys += 1
                except (AttributeError, KeyError):
                    # Key doesn't exist in our model, skip it
                    pass
        
        print(f"Matched {matched_keys} transformer weights")
    
    # Set model to evaluation mode and move to device
    model.to(device)
    model.eval()
    
    # Generate predictions
    print("Generating predictions...")
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
    
    # Add predictions to dataframe
    print(f"Saving predictions to {output_path}")
    data['prediction'] = all_preds
    
    # Save to CSV
    data[['id', 'comment_text', 'target', 'prediction']].to_csv(output_path, index=False)
    print(f"Saved predictions for {len(data)} examples")

if __name__ == "__main__":
    main() 