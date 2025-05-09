#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Targeted script to fix imports and run BERT head-tail training with FP16
"""

import os
import sys
import subprocess

# Ensure output directories exist
os.makedirs("output/checkpoints", exist_ok=True)
os.makedirs("output/preds", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# Set Python path to include current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Function to run a command and capture output
def run_command(cmd):
    print(f"\n{'='*80}\nRunning: {cmd}\n{'='*80}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")
    print(f"Exit code: {result.returncode}")
    return result.returncode == 0

# Fix train.py imports - using sed for Windows PowerShell
sed_cmd = r'(Get-Content src\train.py) -replace "from data import", "from src.data import" -replace "from models\.", "from src.models." | Set-Content src\train.py'
run_command(sed_cmd)

# Fix any config files that use train_folds.csv
for config_file in ["configs/bert_headtail.yaml", "configs/lstm_caps.yaml", "configs/gpt2_headtail.yaml"]:
    sed_cmd = f'(Get-Content {config_file}) -replace "train_folds.csv", "train.csv" | Set-Content {config_file}'
    run_command(sed_cmd)

# Run BERT head-tail training with FP16
cmd = "python -m src.train --model bert_headtail --config configs/bert_headtail.yaml --epochs 2 --save-checkpoint --fp16"
success = run_command(cmd)

if success:
    print("\n✅ BERT head-tail training completed successfully!")
    
    # Generate pseudo-labels
    cmd = "python scripts/pseudo_label.py --base-model output/checkpoints/bert_headtail_fold0.pth --unlabeled-csv data/train.csv --out-csv output/pseudo_bert.csv"
    run_command(cmd)
    
    # Train LSTM-Capsule
    cmd = "python -m src.train --model lstm_caps --config configs/lstm_caps.yaml --epochs 6 --fp16"
    run_command(cmd)
    
    # Train GPT-2 head-tail
    cmd = "python -m src.train --model gpt2_headtail --config configs/gpt2_headtail.yaml --epochs 2 --fp16"
    run_command(cmd)
    
    # Blend models
    cmd = "python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 200 --out-csv output/preds/blend_ensemble.csv"
    run_command(cmd)
    
    # Generate metrics
    cmd = "python scripts/write_metrics.py --predictions output/preds/blend_ensemble.csv --model-name blend_ensemble"
    run_command(cmd)
    
    # Print summary if it exists
    if os.path.exists("results/summary.tsv"):
        print("\nFinal metrics summary:")
        with open("results/summary.tsv", "r") as f:
            print(f.read())
    else:
        print("\nNo metrics summary found.")
else:
    print("\n❌ BERT head-tail training failed.") 