#!/usr/bin/env python
"""
Test script for verifying model checkpoint loading in SHarP analysis.

This script tests the enhanced model loading function from run_sharp_analysis.py
with different checkpoint formats to ensure cross-platform compatibility.

Usage:
    python scripts/test_sharp_loading.py --model-path output/checkpoints/your_model.pth
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the loader function from run_sharp_analysis.py
from fairness_analysis.run_sharp_analysis import load_model
from src.data.loaders import load_tokenizer

def test_model_loading(model_path):
    """Test loading a model checkpoint and verify it's working"""
    print(f"Testing model loading from: {model_path}")
    
    try:
        # Load the model
        model, config = load_model(model_path)
        print(f"✅ Successfully loaded model")
        print(f"Model config: {config}")
        
        # Check model structure
        for name, param in model.named_parameters():
            print(f"Parameter: {name}, Shape: {param.shape}")
            # Only show a few parameters
            if name.startswith("model.bert.embeddings"):
                print(f"Sample values: {param.flatten()[:5]}")
            # Limit output to just a few parameters
            if len(name.split('.')) > 4:
                break
        
        # Try to load tokenizer
        model_name = config.get('bert_model', 'distilbert-base-uncased')
        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = load_tokenizer(model_name)
        
        # Test a simple prediction
        test_text = "This is a test input to verify the model is working properly."
        
        # Use prepare_head_tail_inputs if available
        if hasattr(model, 'prepare_head_tail_inputs'):
            try:
                # Try to use the model's prepare_head_tail_inputs method
                inputs = model.prepare_head_tail_inputs([test_text], tokenizer, max_length=128)
                print("Using model's prepare_head_tail_inputs method")
            except Exception as e:
                print(f"Error using prepare_head_tail_inputs: {e}")
                # Fall back to manual head/tail preparation
                print("Falling back to manual head/tail preparation")
                
                # Standard tokenization
                standard_inputs = tokenizer(
                    [test_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                # For DistilBERT compatibility (no token_type_ids)
                if 'token_type_ids' not in standard_inputs:
                    print("Adding token_type_ids for DistilBERT compatibility")
                    standard_inputs['token_type_ids'] = torch.zeros_like(
                        standard_inputs['input_ids'], 
                        dtype=torch.long
                    )
                
                # Convert to head/tail format
                inputs = {
                    'head_input_ids': standard_inputs['input_ids'],
                    'head_attention_mask': standard_inputs['attention_mask'],
                    'head_token_type_ids': standard_inputs['token_type_ids'],
                    'tail_input_ids': standard_inputs['input_ids'],
                    'tail_attention_mask': standard_inputs['attention_mask'],
                    'tail_token_type_ids': standard_inputs['token_type_ids']
                }
        else:
            # Fallback to standard tokenization
            inputs = tokenizer(
                [test_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            print("Using standard tokenization")
        
        # Set model to eval mode and run inference
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                print(f"Model output shape: {outputs.shape if hasattr(outputs, 'shape') else 'dict'}")
                print(f"Model outputs: {outputs}")
            
            print("\n✅ Model checkpoint loading test passed successfully!")
            return True
        except Exception as e:
            print(f"\n❌ Error during model inference: {e}")
            
            # Try a different approach if the first one fails
            if 'input_ids' in inputs and hasattr(model, 'model') and hasattr(model.model, 'bert'):
                print("Attempting alternative input format...")
                try:
                    # Extract standard inputs and convert to head/tail format
                    input_ids = inputs['input_ids']
                    attention_mask = inputs.get('attention_mask', None)
                    token_type_ids = inputs.get('token_type_ids', None)
                    
                    # Create head/tail inputs manually
                    alt_inputs = {
                        'head_input_ids': input_ids,
                        'head_attention_mask': attention_mask,
                        'head_token_type_ids': token_type_ids,
                        'tail_input_ids': input_ids,
                        'tail_attention_mask': attention_mask,
                        'tail_token_type_ids': token_type_ids
                    }
                    
                    with torch.no_grad():
                        outputs = model(**alt_inputs)
                        print(f"Alternative model output: {outputs}")
                    
                    print("\n✅ Model checkpoint loading test passed with alternative inputs!")
                    return True
                except Exception as alt_e:
                    print(f"Alternative approach also failed: {alt_e}")
            
            return False
    
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test model checkpoint loading")
    parser.add_argument(
        "--model-path",
        default="output/checkpoints/distilbert_headtail_fold0.pth",
        help="Path to model checkpoint to test"
    )
    args = parser.parse_args()
    
    success = test_model_loading(args.model_path)
    
    # Exit with appropriate status code for CI/CD pipelines
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 