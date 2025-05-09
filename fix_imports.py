#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the full RDS fairness audit pipeline
by fixing the import issues in src/train.py and other modules
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import time

# Add current directory to Python path
sys.path.append(os.getcwd())

# Parse arguments
parser = argparse.ArgumentParser(description='Run the full RDS pipeline')
parser.add_argument('--skip-bert', action='store_true', help='Skip BERT training')
parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM training')
parser.add_argument('--skip-gpt2', action='store_true', help='Skip GPT-2 training')
parser.add_argument('--skip-blend', action='store_true', help='Skip model blending')
parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
parser.add_argument('--fp16', action='store_true', help='Use mixed precision (FP16)')
args = parser.parse_args()

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

# Fix model imports
models_module = types.ModuleType('models')
lstm_caps_module = types.ModuleType('models.lstm_caps')
lstm_caps_module.create_lstm_capsule_model = create_lstm_capsule_model
sys.modules['models'] = models_module
sys.modules['models.lstm_caps'] = lstm_caps_module

# Prepare data directory
data_dir = "data"

# Fix config files to use train.csv instead of train_folds.csv
for config_file in ["configs/bert_headtail.yaml", "configs/lstm_caps.yaml", "configs/gpt2_headtail.yaml"]:
    with open(config_file, 'r') as f:
        content = f.read()
    
    if "train_folds.csv" in content:
        content = content.replace("train_folds.csv", "train.csv")
        with open(config_file, 'w') as f:
            f.write(content)
        print(f"Updated {config_file} to use train.csv and valid.csv")

# Initialize directories
os.makedirs("output/checkpoints", exist_ok=True)
os.makedirs("output/preds", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# Set common args
common_args = []
if args.dry_run:
    common_args.append("--dry_run")
if args.fp16:
    common_args.append("--fp16")

def run_command(cmd):
    """Run a command and print output in real-time"""
    print(f"\n{'='*80}\nRunning command: {cmd}\n{'='*80}\n")
    import subprocess
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return process.poll()

# 1. Train BERT head-tail model
if not args.skip_bert:
    cmd = f"python -m src.train --model bert_headtail --config configs/bert_headtail.yaml --epochs 2 --save-checkpoint {' '.join(common_args)}"
    run_command(cmd)
    
    # Generate pseudo-labels
    cmd = f"python scripts/pseudo_label.py --base-model output/checkpoints/bert_headtail_fold0.pth --unlabeled-csv data/train.csv --out-csv output/pseudo_bert.csv"
    run_command(cmd)

# 2. Train LSTM-Capsule model
if not args.skip_lstm:
    cmd = f"python -m src.train --model lstm_caps --config configs/lstm_caps.yaml --epochs 6 {' '.join(common_args)}"
    run_command(cmd)

# 3. Train GPT-2 head-tail model
if not args.skip_gpt2:
    cmd = f"python -m src.train --model gpt2_headtail --config configs/gpt2_headtail.yaml --epochs 2 {' '.join(common_args)}"
    run_command(cmd)

# 4. Blend models with Optuna
if not args.skip_blend:
    cmd = f"python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 200 --out-csv output/preds/blend_ensemble.csv"
    run_command(cmd)
    
    # Generate metrics
    cmd = f"python scripts/write_metrics.py --predictions output/preds/blend_ensemble.csv --model-name blend_ensemble"
    run_command(cmd)
    
    # Generate figures
    cmd = f"python notebooks/04_generate_figures.py"
    run_command(cmd)
    
    # Run explainers
    cmd = f"python scripts/run_explainers.py --model-path output/checkpoints/bert_headtail_fold0.pth --n-samples 500"
    run_command(cmd)

# Print summary
if os.path.exists("results/summary.tsv"):
    print("\n\nFinal metrics summary:\n")
    with open("results/summary.tsv", "r") as f:
        print(f.read())
else:
    print("\n\nNo summary.tsv found - make sure the pipeline completed successfully")

if not args.dry_run and not args.skip_blend:
    print("\n\nFull pipeline completed successfully!")
    print(f"Checkpoints saved to: output/checkpoints/")
    print(f"Predictions saved to: output/preds/")
    print(f"Metrics saved to: results/")
    print(f"Figures saved to: figs/")
else:
    print("\n\nPartial pipeline completed.")

if __name__ == "__main__":
    if len(sys.argv) == 1:  # No arguments provided
        # Run in FP16 mode by default
        sys.argv.append("--fp16") 