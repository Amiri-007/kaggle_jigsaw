#!/usr/bin/env python
"""
Pre-fine-tune a language model on the 2018 Jigsaw Toxic Comment Classification Challenge dataset.
This is an optional step that can improve model performance.

Note: This script is meant to be run offline, as it downloads a large dataset and
      performs heavy computation (2h+ on an A100 GPU).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, AdamW
)
from sklearn.metrics import roc_auc_score
import requests
import zipfile
import io
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# URL for the 2018 Jigsaw Toxic Comment Classification Challenge dataset
DATASET_URL = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/7948/868316/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1642425629&Signature=lXXBbzuRMO134YTqRY9W3kzWQDj3XaYZuDxKK4EYXaRVujS3OrDQz3KQhpIxTIGUQmqDyZxqVdcUNhVmzjV93fCSrJjyGZCq29qpv1NpKilXNzlvgN5hbKZG%2B%2Bc%2FV3fOVJwCc32GR4%2FNUTmcJ1rANnVy0Zka2oNJgVH9edvn1f9CKTNGEfTsJJoDV%2BvZW2RcHPNOJ5nOYbRGDs1oSpqmJDvQiZL8kS7ufMeSYU9N7Zh%2FfNa%2FcIa88pu4NfS3QMzKDCpskTRxdoY9USVxuFdEfPd3FEsOYWmITe7wa9e9VZbHVWcC7n5VXJGZcHOA0KGQQrpLhVBE2%2BJGPB61hDx1ww%3D%3D"
DATASET_DIR = "data/old_toxic"

class ToxicDataset(Dataset):
    """Dataset for toxic comment classification"""
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Remove batch dimension added by tokenizer
            for k, v in encoding.items():
                encoding[k] = v.squeeze(0)
            
            if self.labels is not None:
                encoding['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                
            return encoding
        else:
            item = {'text': text}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

def download_dataset():
    """Download the 2018 Jigsaw dataset if not already present"""
    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Check if dataset files already exist
    if os.path.exists(os.path.join(DATASET_DIR, 'train.csv')):
        logger.info(f"Dataset already exists in {DATASET_DIR}")
        return
    
    # Download and extract dataset
    logger.info(f"Downloading dataset from {DATASET_URL}")
    response = requests.get(DATASET_URL, stream=True)
    
    if response.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(DATASET_DIR)
        logger.info(f"Dataset downloaded and extracted to {DATASET_DIR}")
    else:
        logger.error(f"Failed to download dataset: {response.status_code}")
        raise Exception("Dataset download failed")

def preprocess_data():
    """Load and preprocess the dataset"""
    # Load dataset
    train_path = os.path.join(DATASET_DIR, 'train.csv')
    df = pd.read_csv(train_path)
    
    # Check for required columns
    if 'comment_text' not in df.columns or 'toxic' not in df.columns:
        logger.error(f"Dataset missing required columns")
        raise Exception("Invalid dataset format")
    
    # Create a single binary label (any toxic class)
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = df[toxic_columns].max(axis=1)
    
    # Split into train/valid
    valid_size = 0.1
    valid_indices = np.random.choice(df.index, size=int(len(df) * valid_size), replace=False)
    valid_df = df.loc[valid_indices]
    train_df = df.drop(valid_indices)
    
    logger.info(f"Preprocessed dataset: {len(train_df)} train, {len(valid_df)} validation examples")
    
    return train_df, valid_df

def mlm_pre_finetune(args):
    """Pre-finetune a language model using masked language modeling objective"""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    
    # Load and preprocess data
    train_df, valid_df = preprocess_data()
    
    # Create datasets
    train_dataset = ToxicDataset(
        texts=train_df['comment_text'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    valid_dataset = ToxicDataset(
        texts=valid_df['comment_text'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Starting MLM pre-finetuning on {device}")
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create masked input for MLM
            masked_input_ids, labels = mask_tokens(input_ids, tokenizer)
            
            # Forward pass
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}")
    
    # Save the model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{args.model_name.split('/')[-1]}_mlm_pretrained")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    logger.info(f"Saved pre-finetuned model to {model_path}")
    
    return model_path

def classification_pre_finetune(args):
    """Pre-finetune a model for classification on the old dataset"""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1
    )
    
    # Load and preprocess data
    train_df, valid_df = preprocess_data()
    
    # Create datasets
    train_dataset = ToxicDataset(
        texts=train_df['comment_text'].values,
        labels=train_df['is_toxic'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    valid_dataset = ToxicDataset(
        texts=valid_df['comment_text'].values,
        labels=valid_df['is_toxic'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Starting classification pre-finetuning on {device}")
    
    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, labels)
                
                valid_loss += loss.item()
                
                # Collect predictions and labels for AUC
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_auc = roc_auc_score(all_labels, all_preds)
        
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                   f"Valid Loss: {avg_valid_loss:.4f}, Valid AUC: {valid_auc:.4f}")
    
    # Save the model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{args.model_name.split('/')[-1]}_cls_pretrained")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    logger.info(f"Saved pre-finetuned model to {model_path}")
    
    return model_path

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling"""
    device = inputs.device
    labels = inputs.clone()
    
    # Sample tokens to mask
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # 80% of the time, replace masked tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10% of the time, replace masked tokens with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    inputs[indices_random] = random_words[indices_random]
    
    # The rest of the 10% of the time, keep the masked tokens as is
    return inputs, labels

def main():
    parser = argparse.ArgumentParser(description="Pre-finetune language models on the 2018 Jigsaw dataset")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model name or path")
    parser.add_argument("--mode", type=str, choices=['mlm', 'classification'], default='classification',
                        help="Pre-finetuning mode (masked language modeling or classification)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--output_dir", type=str, default="output/pretrained",
                        help="Output directory for saving model")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for dataloading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Download dataset
    download_dataset()
    
    # Run pre-finetuning
    if args.mode == 'mlm':
        model_path = mlm_pre_finetune(args)
    else:
        model_path = classification_pre_finetune(args)
    
    logger.info(f"Pre-finetuning complete! Model saved to {model_path}")

if __name__ == "__main__":
    main() 