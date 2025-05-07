import os
import sys
import time
import json
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup,
    BertTokenizer, GPT2Tokenizer
)
from typing import Dict, List, Optional, Tuple, Any, Union
import random

from data import create_dataloaders, apply_negative_downsampling, get_sample_weights
from models.lstm_caps import create_lstm_capsule_model
from models.bert_headtail import BertHeadTailForSequenceClassification
from models.gpt2_headtail import GPT2HeadTailForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(model_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = os.path.join('configs', f'{model_name}.yaml')
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data"""
    train_path = os.path.join(config['data_dir'], config['train_file'])
    valid_path = os.path.join(config['data_dir'], config['valid_file'])
    
    if not os.path.exists(train_path):
        logger.error(f"Training file not found: {train_path}")
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    if not os.path.exists(valid_path):
        logger.error(f"Validation file not found: {valid_path}")
        raise FileNotFoundError(f"Validation file not found: {valid_path}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    
    logger.info(f"Loaded training data: {train_df.shape}")
    logger.info(f"Loaded validation data: {valid_df.shape}")
    
    return train_df, valid_df

def create_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[nn.Module, Any]:
    """Create model and tokenizer based on config"""
    model_type = config['model_type']
    
    if model_type == 'lstm_caps':
        # Create vocabulary and tokenizer for LSTM-Capsule model
        from src.utils.text import build_vocab, BasicTokenizer
        
        vocab = build_vocab(config.get('vocab_file'))
        tokenizer = BasicTokenizer(vocab)
        
        # Create model
        model = create_lstm_capsule_model(config)
        
    elif model_type == 'bert_headtail':
        # Load pre-trained tokenizer
        tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        
        # Create model
        model = BertHeadTailForSequenceClassification(
            model_name=config['model_name'],
            num_labels=config.get('num_labels', 1)
        )
        
    elif model_type == 'gpt2_headtail':
        # Load pre-trained tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
        
        # GPT-2 tokenizer doesn't have a padding token by default
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create model
        model = GPT2HeadTailForSequenceClassification(
            model_name=config['model_name'],
            num_labels=config.get('num_labels', 1)
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Created model of type: {model_type}")
    return model, tokenizer

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    sample_weights: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0,
    model_type: str = 'lstm_caps',
    dry_run: bool = False,
    dry_run_batches: int = 5
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    # Track metrics
    total_loss = 0
    total_examples = 0
    step = 0
    
    # Loss function 
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Stop early if dry run
        if dry_run and batch_idx >= dry_run_batches:
            break
            
        # Move tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        # Get sample weights for this batch
        if sample_weights is not None:
            batch_indices = batch['idx'].to(device)
            batch_weights = sample_weights[batch_indices].to(device)
        else:
            batch_weights = None
        
        # Forward pass
        if model_type == 'lstm_caps':
            outputs = model(input_ids, lengths=attention_mask.sum(dim=1))
            logits = outputs
            
        elif model_type == 'bert_headtail':
            # For BERT head-tail, prepare head and tail inputs
            head_inputs = {
                'head_input_ids': input_ids,
                'head_attention_mask': attention_mask,
                'tail_input_ids': input_ids,  # Placeholder, will be truncated in the model
                'tail_attention_mask': attention_mask,  # Placeholder
            }
            outputs = model(**head_inputs)
            logits = outputs['logits'].squeeze(-1)
            
        elif model_type == 'gpt2_headtail':
            # For GPT-2 head-tail, prepare head and tail inputs
            head_inputs = {
                'head_input_ids': input_ids,
                'head_attention_mask': attention_mask,
                'tail_input_ids': input_ids,  # Placeholder, will be truncated in the model
                'tail_attention_mask': attention_mask,  # Placeholder
            }
            outputs = model(**head_inputs)
            logits = outputs['logits'].squeeze(-1)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate loss
        losses = loss_fn(logits, targets)
        
        # Apply sample weights if provided
        if batch_weights is not None:
            losses = losses * batch_weights
            
        loss = losses.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update weights
        optimizer.step()
        
        # Update EMA weights if using LSTM-Capsule model
        if model_type == 'lstm_caps' and hasattr(model, 'update_ema_weights'):
            model.update_ema_weights()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item() * len(targets)
        total_examples += len(targets)
        step += 1
        
        # Log progress
        if step % 50 == 0 or step == len(train_loader):
            logger.info(f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                       f"Loss: {total_loss/total_examples:.4f} | "
                       f"Time: {time.time() - start_time:.2f}s")
    
    # Calculate average loss
    avg_loss = total_loss / total_examples
    
    metrics = {
        'loss': avg_loss,
        'epoch': epoch,
        'steps': step
    }
    
    return metrics

def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    model_type: str = 'lstm_caps',
    dry_run: bool = False,
    dry_run_batches: int = 5
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    
    # Track metrics
    total_loss = 0
    total_examples = 0
    all_predictions = []
    all_targets = []
    
    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            # Stop early if dry run
            if dry_run and batch_idx >= dry_run_batches:
                break
                
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            if model_type == 'lstm_caps':
                outputs = model(input_ids, lengths=attention_mask.sum(dim=1))
                logits = outputs
                
            elif model_type == 'bert_headtail':
                # For BERT head-tail, prepare head and tail inputs
                head_inputs = {
                    'head_input_ids': input_ids,
                    'head_attention_mask': attention_mask,
                    'tail_input_ids': input_ids,  # Placeholder, will be truncated in the model
                    'tail_attention_mask': attention_mask,  # Placeholder
                }
                outputs = model(**head_inputs)
                logits = outputs['logits'].squeeze(-1)
                
            elif model_type == 'gpt2_headtail':
                # For GPT-2 head-tail, prepare head and tail inputs
                head_inputs = {
                    'head_input_ids': input_ids,
                    'head_attention_mask': attention_mask,
                    'tail_input_ids': input_ids,  # Placeholder, will be truncated in the model
                    'tail_attention_mask': attention_mask,  # Placeholder
                }
                outputs = model(**head_inputs)
                logits = outputs['logits'].squeeze(-1)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Calculate loss
            loss = loss_fn(logits, targets)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            
            # Track metrics
            total_loss += loss.item() * len(targets)
            total_examples += len(targets)
            all_predictions.extend(probs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    
    # Calculate average loss and AUC
    avg_loss = total_loss / total_examples
    
    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_targets, all_predictions)
    except:
        auc = 0.0
    
    metrics = {
        'val_loss': avg_loss,
        'val_auc': auc
    }
    
    return metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    fold: int = 0
) -> str:
    """Save model checkpoint"""
    # Skip saving if save_checkpoint is False
    if not config.get('save_checkpoint', True):
        logger.info("Skipping checkpoint saving (save_checkpoint=False)")
        # Return a placeholder path
        return "no_checkpoint_saved"
    
    # Create output directory if it doesn't exist
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint path
    model_type = config['model_type']
    checkpoint_path = os.path.join(output_dir, f"{model_type}_fold{fold}.pth")
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'metrics': metrics,
        'config': config
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return checkpoint_path

def train(config: Dict[str, Any]) -> str:
    """Main training function"""
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df, valid_df = load_data(config)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(config)
    model = model.to(device)
    
    # Create dataloaders
    train_loader, valid_loader, sample_weights = create_dataloaders(
        train_df=train_df,
        valid_df=valid_df,
        tokenizer=tokenizer,
        batch_size=config.get('batch_size', 32),
        max_length=config.get('max_length', 256),
        text_col=config.get('text_col', 'comment_text'),
        target_col=config.get('target_col', 'target'),
        identity_cols=config.get('identity_cols'),
        apply_downsampling=config.get('apply_downsampling', True),
        apply_weights=config.get('apply_weights', True),
        num_workers=config.get('num_workers', 4),
        first_epoch=True,
        random_state=config.get('seed', 42)
    )
    
    if sample_weights is not None:
        sample_weights = sample_weights.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * config.get('num_epochs', 3)
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_auc = 0.0
    best_checkpoint_path = None
    
    for epoch in range(1, config.get('num_epochs', 3) + 1):
        logger.info(f"Starting epoch {epoch}/{config.get('num_epochs', 3)}")
        
        # Apply downsampling for this epoch
        first_epoch = (epoch == 1)
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            sample_weights=sample_weights,
            max_grad_norm=config.get('max_grad_norm', 1.0),
            model_type=config.get('model_type', 'lstm_caps'),
            dry_run=config.get('dry_run', False),
            dry_run_batches=config.get('dry_run_batches', 5)
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            valid_loader=valid_loader,
            device=device,
            model_type=config.get('model_type', 'lstm_caps'),
            dry_run=config.get('dry_run', False),
            dry_run_batches=config.get('dry_run_batches', 5)
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Log metrics
        logger.info(f"Epoch {epoch} results: {metrics}")
        
        # Save checkpoint if validation AUC improved
        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                fold=config.get('fold', 0)
            )
            
            best_checkpoint_path = checkpoint_path
            logger.info(f"New best model saved to {checkpoint_path} with val_auc={best_val_auc:.4f}")
        
        # Early stopping
        if config.get('dry_run', False):
            logger.info("Dry run complete, stopping training")
            break
    
    logger.info(f"Training complete. Best val_auc={best_val_auc:.4f}")
    return best_checkpoint_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train a toxicity classification model")
    parser.add_argument('--model', type=str, default='bert_headtail',
                        choices=['lstm_caps', 'bert_headtail', 'gpt2_headtail'],
                        help="Model type to train")
    parser.add_argument('--config_dir', type=str, default='configs',
                        help="Directory containing model configurations")
    parser.add_argument('--fold', type=int, default=0,
                        help="Fold number for cross-validation")
    parser.add_argument('--dry_run', action='store_true',
                        help="Run a small training loop (2 batches) for testing, skip saving checkpoint")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(args.config_dir, f"{args.model}.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command-line arguments
    config['model_type'] = args.model
    config['fold'] = args.fold
    config['seed'] = args.seed
    
    # For dry-run, set specific parameters
    if args.dry_run:
        config['dry_run'] = True
        config['dry_run_batches'] = 2
        config['save_checkpoint'] = False
    else:
        config['dry_run'] = False
        config['save_checkpoint'] = True
    
    # Train model
    try:
        best_checkpoint_path = train(config)
        logger.info(f"Training complete. Best checkpoint: {best_checkpoint_path}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main() 