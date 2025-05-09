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
from pathlib import Path
from tqdm.auto import tqdm

from src.data import create_dataloaders, apply_negative_downsampling, get_sample_weights, load_train_valid
from src.models.lstm_caps import create_lstm_capsule_model
from src.models.bert_headtail import BertHeadTailForSequenceClassification
from src.models.gpt2_headtail import GPT2HeadTailForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define data path constant
DATA_PATH = Path("data")

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

def load_pseudo_labels(path: str) -> pd.DataFrame:
    """Load pseudo-labeled data from CSV file"""
    if not os.path.exists(path):
        logger.error(f"Pseudo-label file not found: {path}")
        raise FileNotFoundError(f"Pseudo-label file not found: {path}")
    
    pseudo_df = pd.read_csv(path)
    logger.info(f"Loaded pseudo-labeled data: {pseudo_df.shape}")
    
    # Ensure required columns exist
    required_cols = ["id", "comment_text", "pseudo_target"]
    for col in required_cols:
        if col not in pseudo_df.columns:
            raise ValueError(f"Pseudo-label file missing required column: {col}")
    
    # Rename pseudo_target to target for consistency
    pseudo_df = pseudo_df.rename(columns={"pseudo_target": "target"})
    
    return pseudo_df

def load_data(config: Dict[str, Any], pseudo_label_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data, optionally adding pseudo-labeled data"""
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
    
    # Add pseudo-labeled data if provided
    if pseudo_label_path is not None:
        try:
            pseudo_df = load_pseudo_labels(pseudo_label_path)
            
            # Ensure no ID collisions by adding prefix to pseudo-label IDs
            if "id" in pseudo_df.columns and "id" in train_df.columns:
                pseudo_df["id"] = "pseudo_" + pseudo_df["id"].astype(str)
            
            # Concatenate with original training data
            original_count = len(train_df)
            train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
            
            logger.info(f"Added {len(pseudo_df)} pseudo-labeled examples to training data")
            logger.info(f"New training data size: {len(train_df)} (original: {original_count})")
        except Exception as e:
            logger.warning(f"Failed to load pseudo-labeled data: {e}")
    
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
    dry_run_batches: int = 5,
    scaler: Optional[Any] = None,  # GradScaler for mixed precision
    turbo_mode: bool = False
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
    
    # Add progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=False)
    
    for batch_idx, batch in enumerate(pbar):
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
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            # Use torch.cuda.amp.autocast
            with torch.cuda.amp.autocast():
                # Forward pass depends on model type
                if model_type == 'lstm_caps':
                    outputs = model(input_ids)
                    logits = outputs
                elif model_type in ['bert_headtail', 'gpt2_headtail']:
                    # For transformer models
                    tail_input_ids = batch.get('tail_input_ids', None)
                    tail_attention_mask = batch.get('tail_attention_mask', None)
                    
                    if tail_input_ids is not None:
                        tail_input_ids = tail_input_ids.to(device)
                        tail_attention_mask = tail_attention_mask.to(device)
                        
                        # Use head_ prefix for parameters with BERT head-tail model
                        if model_type == 'bert_headtail':
                            outputs = model(
                                head_input_ids=input_ids,
                                head_attention_mask=attention_mask,
                                tail_input_ids=tail_input_ids,
                                tail_attention_mask=tail_attention_mask
                            )
                            # BERT model returns a dict with 'logits' key
                            logits = outputs['logits']
                        else:
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                tail_input_ids=tail_input_ids,
                                tail_attention_mask=tail_attention_mask
                            )
                            # GPT2 model returns an object with logits attribute
                            logits = outputs.logits
                    else:
                        # If no tail inputs, use input as both head and tail
                        if model_type == 'bert_headtail':
                            outputs = model(
                                head_input_ids=input_ids,
                                head_attention_mask=attention_mask,
                                tail_input_ids=input_ids,
                                tail_attention_mask=attention_mask
                            )
                            # BERT model returns a dict with 'logits' key
                            logits = outputs['logits']
                        else:
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            # GPT2 model returns an object with logits attribute
                            logits = outputs.logits
                    
                    # Calculate loss (weighted if sample_weights provided)
                    loss = loss_fn(logits.view(-1), targets.view(-1).float())
                    
                    if sample_weights is not None:
                        # Apply sample weights
                        loss = loss * batch_weights
                    
                    # Calculate mean loss
                    loss = loss.mean()
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # Clip gradients
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Step optimizer and scaler
                scaler.step(optimizer)
                scaler.update()
        else:
            # Regular forward pass without mixed precision
            # Forward pass depends on model type
            if model_type == 'lstm_caps':
                outputs = model(input_ids)
                logits = outputs
            elif model_type in ['bert_headtail', 'gpt2_headtail']:
                # For transformer models
                tail_input_ids = batch.get('tail_input_ids', None)
                tail_attention_mask = batch.get('tail_attention_mask', None)
                
                if tail_input_ids is not None:
                    tail_input_ids = tail_input_ids.to(device)
                    tail_attention_mask = tail_attention_mask.to(device)
                    
                    # Use head_ prefix for parameters with BERT head-tail model
                    if model_type == 'bert_headtail':
                        outputs = model(
                            head_input_ids=input_ids,
                            head_attention_mask=attention_mask,
                            tail_input_ids=tail_input_ids,
                            tail_attention_mask=tail_attention_mask
                        )
                        # BERT model returns a dict with 'logits' key
                        logits = outputs['logits']
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            tail_input_ids=tail_input_ids,
                            tail_attention_mask=tail_attention_mask
                        )
                        # GPT2 model returns an object with logits attribute
                        logits = outputs.logits
                else:
                    # If no tail inputs, use input as both head and tail
                    if model_type == 'bert_headtail':
                        outputs = model(
                            head_input_ids=input_ids,
                            head_attention_mask=attention_mask,
                            tail_input_ids=input_ids,
                            tail_attention_mask=attention_mask
                        )
                        # BERT model returns a dict with 'logits' key
                        logits = outputs['logits']
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        # GPT2 model returns an object with logits attribute
                        logits = outputs.logits
                
                # Calculate loss (weighted if sample_weights provided)
                loss = loss_fn(logits.view(-1), targets.view(-1).float())
                
                if sample_weights is not None:
                    # Apply sample weights
                    loss = loss * batch_weights
                
                # Calculate mean loss
                loss = loss.mean()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Step optimizer
                optimizer.step()
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * len(targets)
        total_examples += len(targets)
        step += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / total_examples})
        
        # In turbo mode, break early (train on fewer batches)
        if turbo_mode and batch_idx >= (100 if model_type == 'lstm_caps' else 50):
            logger.info(f"Turbo mode enabled: stopping after {batch_idx+1} batches")
            break
    
    # Calculate epoch metrics
    epoch_loss = total_loss / total_examples
    metrics = {
        'loss': epoch_loss,
        'epoch': epoch,
        'steps': step
    }
    
    duration = time.time() - start_time
    logger.debug(f"Epoch {epoch} training completed in {duration:.2f}s. Metrics: {metrics}")
    
    return metrics

def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    model_type: str = 'lstm_caps',
    dry_run: bool = False,
    dry_run_batches: int = 5,
    turbo_mode: bool = False
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
    
    # Add progress bar
    pbar = tqdm(valid_loader, desc="Validation", disable=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Stop early if dry run
            if dry_run and batch_idx >= dry_run_batches:
                break
            
            # In turbo mode, only evaluate on a small subset
            if turbo_mode and batch_idx >= 50:
                logger.info(f"Turbo mode enabled: stopping validation after {batch_idx+1} batches")
                break
                
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            if model_type == 'lstm_caps':
                outputs = model(input_ids)
                logits = outputs
                
            elif model_type == 'bert_headtail':
                # For BERT head-tail
                outputs = model(
                    head_input_ids=input_ids,
                    head_attention_mask=attention_mask,
                    tail_input_ids=input_ids,  # Using same input for tail in validation
                    tail_attention_mask=attention_mask
                )
                logits = outputs['logits']
                
            elif model_type == 'gpt2_headtail':
                # For GPT-2 head-tail
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
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
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / total_examples})
    
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
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Create checkpoint path
    model_type = config['model_type']
    checkpoint_path = os.path.join(output_dir, "checkpoints", f"{model_type}_fold{fold}.pth")
    
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

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load saved model checkpoint for resuming training"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 0)})")
    return checkpoint

def train(config: Dict[str, Any], pseudo_label_path: Optional[str] = None, resume_checkpoint_path: Optional[str] = None) -> str:
    """Main training function"""
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Check if turbo mode is enabled
    turbo_mode = config.get('turbo_mode', False)
    if turbo_mode:
        logger.info("🚀 TURBO MODE ENABLED - Ultra-fast training with reduced steps")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize mixed precision training if specified
    fp16 = config.get('fp16', False)
    scaler = None
    if fp16 and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Mixed precision (FP16) training enabled with GradScaler")
    
    # Load data
    train_df, valid_df = load_data(config, pseudo_label_path)
    
    # Resume from checkpoint or create a new model
    start_epoch = 1
    best_val_auc = 0.0
    best_checkpoint_path = None
    
    if resume_checkpoint_path:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = load_checkpoint(resume_checkpoint_path)
        
        # Extract checkpoint data
        saved_config = checkpoint.get('config', {})
        start_epoch = checkpoint.get('epoch', 0) + 1
        metrics = checkpoint.get('metrics', {})
        best_val_auc = metrics.get('val_auc', 0.0)
        
        # Create model and tokenizer from the original config to ensure architecture match
        model, tokenizer = create_model_and_tokenizer(config)
        model = model.to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from checkpoint (epoch {start_epoch-1})")
        
        # Create optimizer and scheduler (will be overwritten by checkpoint states)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state from checkpoint")
        
        # Create learning rate scheduler
        total_steps = len(train_df) // config.get('batch_size', 32) * config.get('num_epochs', 3)
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state from checkpoint")
    else:
        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer(config)
        model = model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Create learning rate scheduler
        total_steps = len(train_df) // config.get('batch_size', 32) * config.get('num_epochs', 3)
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    # Apply annotator weight if specified
    apply_annotator_weight = config.get('annotator_weight', False)
    logger.info(f"Annotator weight enabled: {apply_annotator_weight}")
    
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
        annotator_weight=apply_annotator_weight,
        num_workers=config.get('num_workers', 4),
        first_epoch=(start_epoch == 1),  # Only apply first-epoch downsampling if this is epoch 1
        random_state=config.get('seed', 42)
    )
    
    if sample_weights is not None:
        sample_weights = sample_weights.to(device)
    
    # Training loop
    num_epochs = config.get('num_epochs', 3)
    logger.info(f"Starting training from epoch {start_epoch}/{num_epochs}")
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")
        
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
            dry_run_batches=config.get('dry_run_batches', 5),
            scaler=scaler,
            turbo_mode=turbo_mode
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            valid_loader=valid_loader,
            device=device,
            model_type=config.get('model_type', 'lstm_caps'),
            dry_run=config.get('dry_run', False),
            dry_run_batches=config.get('dry_run_batches', 5),
            turbo_mode=turbo_mode
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
    parser.add_argument('--pseudo-label-csv', type=str, default=None,
                        help="Path to CSV file containing pseudo-labeled data")
    parser.add_argument('--fp16', action='store_true', 
                        help="Use mixed precision (FP16) training to speed up and reduce memory usage")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to specific config file (overrides config_dir and model)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override number of epochs in config")
    parser.add_argument('--save-checkpoint', action='store_true',
                        help="Force saving checkpoint even in dry-run mode")
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument('--cache-dir', type=str, default=None,
                        help="Directory with cached tokenized tensors for faster loading")
    parser.add_argument('--token-cache', type=str, default=None,
                        help="Alias for --cache-dir for backward compatibility")
    parser.add_argument('--valid-frac', type=float, default=0.05)
    parser.add_argument('--sample-frac', type=float, default=None,
                      help="Train on random subset (e.g. 0.1 = 10 %) for fast dev.")
    parser.add_argument('--turbo', action='store_true',
                      help="Enable turbo mode for ultra-fast training with reduced steps")
    
    args = parser.parse_args()
    
    # Set random seed
    seed = args.seed
    set_seed(seed)
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(args.config_dir, f"{args.model}.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    train_df, valid_df = load_train_valid(
        DATA_PATH / "train.csv",
        valid_frac=args.valid_frac,
        random_state=seed,
        sample_frac=args.sample_frac,
    )
    
    # Update config with command-line arguments
    config['model_type'] = args.model
    config['fold'] = args.fold
    config['seed'] = args.seed
    
    # Override number of epochs if specified
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    
    # For dry-run, set specific parameters
    if args.dry_run:
        config['dry_run'] = True
        config['dry_run_batches'] = 2
        config['save_checkpoint'] = args.save_checkpoint
    else:
        config['dry_run'] = False
        config['save_checkpoint'] = True
    
    # Set FP16 flag for mixed precision training
    config['fp16'] = args.fp16
    if args.fp16:
        logger.info("Using mixed precision (FP16) training")
    
    # Set turbo mode flag
    config['turbo_mode'] = args.turbo
    
    # Check for resume checkpoint in config
    resume_checkpoint_path = args.resume_checkpoint
    if resume_checkpoint_path is None and 'resume_checkpoint' in config:
        resume_checkpoint_path = config.get('resume_checkpoint')
    
    # Train model
    try:
        best_checkpoint_path = train(config, args.pseudo_label_csv, resume_checkpoint_path)
        logger.info(f"Training complete. Best checkpoint: {best_checkpoint_path}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main() 