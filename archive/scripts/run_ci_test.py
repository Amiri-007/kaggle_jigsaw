#!/usr/bin/env python
"""
Run a complete CI-friendly trial with dry-run flags.
This script runs the entire pipeline with the --dry-run flag to test everything works.
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n### {description} ###")
    print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"Return code: {process.returncode}")
    print(f"Output:")
    print(process.stdout)
    
    if process.returncode != 0:
        print(f"Error:")
        print(process.stderr)
        sys.exit(process.returncode)
    
    return process

def main():
    # Create required directories
    os.makedirs("output/preds", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    
    # Step 1: Train model in dry-run mode
    run_command(
        ["python", "-m", "src.train", "--model", "bert_headtail", "--dry_run"],
        "Training model in dry-run mode"
    )
    
    # Step 2: Run pseudo-labeling in dry-run mode
    temp_csv = os.path.join(os.path.dirname(__file__), "../output/pseudo_temp.csv")
    run_command(
        [
            "python", 
            "scripts/pseudo_label.py", 
            "--base-model", "bert_headtail", 
            "--unlabeled-csv", "data/train.csv", 
            "--out-csv", temp_csv, 
            "--dry-run"
        ],
        "Running pseudo-labeling in dry-run mode"
    )
    
    # Step 3: Train with pseudo-labels in dry-run mode
    run_command(
        [
            "python", 
            "-m", "src.train", 
            "--model", "bert_headtail", 
            "--pseudo-label-csv", temp_csv, 
            "--dry_run"
        ],
        "Training with pseudo-labels in dry-run mode"
    )
    
    # Step 4: Run prediction in dry-run mode
    run_command(
        ["python", "-m", "src.predict", "--model", "bert_headtail", "--dry_run"],
        "Running prediction in dry-run mode"
    )
    
    # Step 5: Generate metrics
    run_command(
        [
            "python", 
            "scripts/write_metrics.py", 
            "--predictions", "output/preds/bert_headtail.csv", 
            "--model_name", "bert_ci"
        ],
        "Generating metrics"
    )
    
    # Count the number of files generated
    result_files = list(Path("results").glob("*.csv"))
    figure_files = list(Path("figs").glob("*.png"))
    
    print(f"\n### CI Test Summary ###")
    print(f"Results files: {len(result_files)}")
    print(f"Figure files: {len(figure_files)}")
    print("All CI tests completed successfully!")

if __name__ == "__main__":
    main() 