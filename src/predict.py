import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import BertTokenizer, GPT2Tokenizer

# Import model classes
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

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load saved model checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def create_model_and_tokenizer(checkpoint: Dict[str, Any]) -> Tuple[torch.nn.Module, Any]:
    """Create model and tokenizer from checkpoint"""
    config = checkpoint['config']
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model of type: {model_type}")
    return model, tokenizer

def batch_predict(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
    model_type: str = 'lstm_caps'
) -> np.ndarray:
    """Generate predictions in batches"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        if model_type == 'lstm_caps':
            # Basic tokenization for LSTM model
            encodings = []
            for text in batch_texts:
                encoding = tokenizer.encode(text, max_length=max_length, padding=True, truncation=True)
                encodings.append(encoding)
            
            # Convert to tensors
            input_ids = torch.tensor(encodings).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).float().to(device)
            
            # Get lengths for packing
            lengths = attention_mask.sum(dim=1)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids, lengths=lengths)
                probs = outputs  # Already sigmoid activated
                
        elif model_type in ['bert_headtail', 'gpt2_headtail']:
            # HuggingFace tokenization
            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # For head-tail models, prepare head and tail inputs
            if model_type == 'bert_headtail':
                # Prepare head-tail inputs
                model_inputs = model.prepare_head_tail_inputs(batch_texts, tokenizer, max_length=max_length)
                
                # Move tensors to device
                for k, v in model_inputs.items():
                    model_inputs[k] = v.to(device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(**model_inputs)
                    logits = outputs['logits']
                    probs = torch.sigmoid(logits)
            
            elif model_type == 'gpt2_headtail':
                # Prepare head-tail inputs
                model_inputs = model.prepare_head_tail_inputs(batch_texts, tokenizer, max_length=max_length)
                
                # Move tensors to device
                for k, v in model_inputs.items():
                    model_inputs[k] = v.to(device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(**model_inputs)
                    logits = outputs['logits']
                    probs = torch.sigmoid(logits)
        
        # Add batch predictions to results
        all_predictions.extend(probs.squeeze().cpu().numpy().tolist())
        
        # Log progress
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {i+len(batch_texts)}/{len(texts)} examples")
    
    return np.array(all_predictions)

def predict(
    checkpoint_path: Optional[str] = None,
    test_file: str = 'data/test_public_expanded.csv',
    output_file: Optional[str] = None,
    text_col: str = 'comment_text',
    id_col: str = 'id',
    batch_size: int = 32,
    model_type: Optional[str] = None,
    dry_run: bool = False
) -> pd.DataFrame:
    """Generate predictions for test data"""
    # Handle dry-run mode when no checkpoint is available
    if checkpoint_path is None and dry_run:
        if model_type is None:
            logger.error("Model type must be provided for dry-run mode")
            raise ValueError("Model type must be provided for dry-run mode")
        
        logger.info("Running in dry-run mode with random predictions")
        
        # Load test data or create dummy data
        try:
            logger.info(f"Loading test data from {test_file}")
            test_df = pd.read_csv(test_file)
        except:
            logger.warning(f"Could not load test data from {test_file}, creating dummy data")
            # Create dummy data with 100 examples
            test_df = pd.DataFrame({
                id_col: [f"id_{i}" for i in range(100)],
                text_col: [f"Text example {i}" for i in range(100)]
            })
        
        # Use only the first 100 examples
        if len(test_df) > 100:
            test_df = test_df.head(100)
        
        logger.info(f"Generating random predictions for {len(test_df)} examples")
        
        # Generate random predictions
        np.random.seed(42)  # For reproducibility
        predictions = np.random.uniform(0, 1, len(test_df))
        
        # Create output dataframe
        results_df = pd.DataFrame({
            id_col: test_df[id_col],
            'prediction': predictions
        })
        
        # Determine output file
        if output_file is None:
            output_dir = os.path.join('output', 'preds')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{model_type}.csv")
        
        # Save predictions
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved random predictions to {output_file}")
        
        return results_df
    
    # Regular prediction with checkpoint
    if checkpoint_path is None:
        logger.error("Checkpoint path must be provided for regular prediction mode")
        raise ValueError("Checkpoint path must be provided for regular prediction mode")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint['config']
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(checkpoint)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_df = pd.read_csv(test_file)
    logger.info(f"Loaded {len(test_df)} test examples")
    
    # Get text data
    texts = test_df[text_col].tolist()
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = batch_predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=config.get('max_length', 256),
        model_type=config['model_type']
    )
    
    # Create output dataframe
    results_df = pd.DataFrame({
        id_col: test_df[id_col],
        'prediction': predictions
    })
    
    # Save predictions
    if output_file is None:
        model_name = os.path.basename(checkpoint_path).split('.')[0]
        output_dir = os.path.join('output', 'preds')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}.csv")
    
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")
    
    # Call metrics script if available
    metrics_script = os.path.join('scripts', 'write_metrics.py')
    if os.path.exists(metrics_script):
        try:
            logger.info("Calculating fairness metrics...")
            import subprocess
            cmd = [sys.executable, metrics_script, '--predictions', output_file]
            subprocess.run(cmd, check=True)
            logger.info("Fairness metrics calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating fairness metrics: {e}")
    
    return results_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate predictions using a trained model")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument('--model', type=str, default='bert_headtail',
                        choices=['lstm_caps', 'bert_headtail', 'gpt2_headtail'],
                        help="Model type (required if no checkpoint path is provided)")
    parser.add_argument('--test_file', type=str, default='data/test_public_expanded.csv',
                        help="Path to test data CSV file")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Path to save predictions CSV file")
    parser.add_argument('--text_col', type=str, default='comment_text',
                        help="Column name for text data")
    parser.add_argument('--id_col', type=str, default='id',
                        help="Column name for ID")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for prediction")
    parser.add_argument('--dry_run', action='store_true',
                        help="Run in dry-run mode (generate random predictions)")
    args = parser.parse_args()
    
    # Check if either checkpoint_path or (model and dry_run) is provided
    if args.checkpoint_path is None and (args.model is None or not args.dry_run):
        logger.error("Either --checkpoint_path or both --model and --dry_run must be provided")
        sys.exit(1)
    
    # Generate predictions
    try:
        predict(
            checkpoint_path=args.checkpoint_path,
            test_file=args.test_file,
            output_file=args.output_file,
            text_col=args.text_col,
            id_col=args.id_col,
            batch_size=args.batch_size,
            model_type=args.model,
            dry_run=args.dry_run
        )
        logger.info("Prediction complete")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == '__main__':
    main() 