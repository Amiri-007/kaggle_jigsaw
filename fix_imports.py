#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick script to run BERT training in dry-run mode
by fixing the import issues in src/train.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to Python path
sys.path.append(os.getcwd())

# Prepare data directory
data_dir = "data"
if not os.path.exists(os.path.join(data_dir, "valid.csv")):
    print("Creating validation file from train.csv...")
    if os.path.exists(os.path.join(data_dir, "train.csv")):
        # Create a small validation set from train.csv
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        # Set random seed for reproducibility
        np.random.seed(42)
        # Split 95/5
        valid_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        valid_df = df.loc[valid_indices]
        valid_df.to_csv(os.path.join(data_dir, "valid.csv"), index=False)
        print(f"Created validation file with {len(valid_df)} samples")

# Update config to use train.csv instead of train_folds.csv
import yaml
bert_config_path = os.path.join("configs", "bert_headtail.yaml")
with open(bert_config_path, 'r') as f:
    config = yaml.safe_load(f)
    
# Modify config to use existing files
config["train_file"] = "train.csv"
config["valid_file"] = "valid.csv"

# Save modified config
with open(bert_config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Updated {bert_config_path} to use train.csv and valid.csv")

# Import modules directly from src
from src.data import create_dataloaders, apply_negative_downsampling, get_sample_weights
from src.models.lstm_caps import create_lstm_capsule_model
from src.models.bert_headtail import BertHeadTailForSequenceClassification
from src.models.gpt2_headtail import GPT2HeadTailForSequenceClassification

# Monkeypatch modules in sys.modules to make train.py import them correctly
import types

# Fix data module imports
data_module = types.ModuleType('data')
data_module.create_dataloaders = create_dataloaders
data_module.apply_negative_downsampling = apply_negative_downsampling
data_module.get_sample_weights = get_sample_weights
sys.modules['data'] = data_module

# Fix models imports
models_module = types.ModuleType('models')
sys.modules['models'] = models_module

# Add submodules
lstm_caps_module = types.ModuleType('models.lstm_caps')
lstm_caps_module.create_lstm_capsule_model = create_lstm_capsule_model
sys.modules['models.lstm_caps'] = lstm_caps_module

bert_module = types.ModuleType('models.bert_headtail')
bert_module.BertHeadTailForSequenceClassification = BertHeadTailForSequenceClassification
sys.modules['models.bert_headtail'] = bert_module

gpt2_module = types.ModuleType('models.gpt2_headtail')
gpt2_module.GPT2HeadTailForSequenceClassification = GPT2HeadTailForSequenceClassification
sys.modules['models.gpt2_headtail'] = gpt2_module

# Now run the training
from src.train import main

if __name__ == "__main__":
    # Set dry run argument
    sys.argv = ["train.py", "--model", "bert_headtail", "--dry_run"]
    main() 