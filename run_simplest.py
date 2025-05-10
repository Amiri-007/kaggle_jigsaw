#!/usr/bin/env python
"""
Ultra-fast minimal test script for RTX 3070Ti
"""
import os
import time
import logging
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fairness_analysis import BiasReport, final_score
from src.data.utils import list_identity_columns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make directories
os.makedirs("output/simplest", exist_ok=True)
os.makedirs("output/preds", exist_ok=True)

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 1
SAMPLE_FRACTION = 0.05
MODEL_NAME = "distilbert-base-uncased"

def main():
    start_time = time.time()
    logger.info("Starting simplest turbo run...")
    
    # Load a small subset of data
    logger.info("Loading data...")
    df = pd.read_csv('data/train.csv')
    
    # Clean the data - handle NaN values in comment_text
    df['comment_text'] = df['comment_text'].fillna("").astype(str)
    
    # Sample a fraction of the data
    df = df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    
    # Create a simple train/val split
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            # Clean the texts to ensure they're all strings
            texts = texts.fillna("").astype(str)
            
            self.encodings = tokenizer(texts.tolist(), 
                                     truncation=True, 
                                     padding='max_length',
                                     max_length=max_length,
                                     return_tensors="pt")
            self.labels = torch.tensor(labels.values, dtype=torch.float32)
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx].unsqueeze(0)  # Add batch dimension
            return item
        
        def __len__(self):
            return len(self.labels)
    
    # Initialize tokenizer and model
    logger.info("Initializing model...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1,
        problem_type="regression"
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = SimpleDataset(train_df['comment_text'], train_df['target'], tokenizer, MAX_LENGTH)
    valid_dataset = SimpleDataset(valid_df['comment_text'], valid_df['target'], tokenizer, MAX_LENGTH)
    
    # Training arguments - simplified to avoid compatibility issues
    training_args = TrainingArguments(
        output_dir="output/simplest",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        logging_dir="output/simplest/logs",
        logging_steps=10,
        save_total_limit=1,
        save_steps=500,
        eval_steps=500,
        fp16=torch.cuda.is_available(),  # Use FP16 if available
        dataloader_num_workers=0,  # Simplest approach to avoid collation issues
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        trainer.train()
        
        # Save model
        model_path = "output/simplest/final"
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Generate predictions for validation set
        logger.info("Generating predictions for validation set...")
        predictions = trainer.predict(valid_dataset)
        logits = predictions.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # Create predictions CSV
        preds_df = pd.DataFrame({
            'id': valid_df['id'],
            'prediction': probs.flatten()
        })
        
        # Save predictions
        preds_path = "output/preds/simplest_preds.csv"
        preds_df.to_csv(preds_path, index=False)
        logger.info(f"Predictions saved to {preds_path}")
        
        # Simple prediction
        logger.info("Running sample prediction...")
        text = "I hate this movie, it was terrible."
        encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            prediction = torch.sigmoid(logits).item()
        
        logger.info(f"Sample text: {text}")
        logger.info(f"Prediction (toxicity): {prediction:.4f}")
        
        # Report training time
        duration = time.time() - start_time
        logger.info(f"Training completed in {duration:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    main() 