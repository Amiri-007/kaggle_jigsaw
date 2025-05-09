#!/usr/bin/env python
"""
Run Fairness Analysis Pipeline
=============================
This script runs the complete fairness analysis pipeline:
1. Count and visualize demographic distribution
2. Run fairness auditing with metrics calculation
3. Analyze intersectional fairness
4. Check compliance with fairness requirements
5. Launch the fairness dashboard

Usage:
    python run_fairness_analysis.py --model your_model_name
"""
import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

def run_command(command, desc, show_output=False):
    """Run a command with progress indication"""
    print(f"\n{'='*80}")
    print(f"🔹 {desc}")
    print(f"{'='*80}")
    print(f"$ {command}")
    
    try:
        if show_output:
            # Run with output shown
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                
            process.wait()
            return process.returncode == 0
        else:
            # Run with a progress bar
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT
            )
            
            with tqdm(total=100, desc="Running", bar_format="{desc}: {bar}| {percentage:3.0f}%") as pbar:
                while process.poll() is None:
                    pbar.update(1)
                    if pbar.n >= 100:
                        pbar.n = 0
                    time.sleep(0.1)
                
                # Ensure we reach 100%
                pbar.n = 100
                pbar.refresh()
                
            return process.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the fairness analysis pipeline")
    parser.add_argument("--model", default="distilbert_dev", help="Model name for predictions")
    parser.add_argument("--preds", help="Path to predictions file (overrides model name)")
    parser.add_argument("--data", default="data/train.csv", help="Path to validation data")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show command output")
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip launching the dashboard")
    parser.add_argument("--skip-counts", action="store_true", help="Skip demographic counting")
    parser.add_argument("--skip-intersectional", action="store_true", help="Skip intersectional analysis")
    args = parser.parse_args()
    
    # Determine predictions file path
    if args.preds:
        preds_file = args.preds
    else:
        preds_file = f"results/preds_{args.model}.csv"
    
    # Check if predictions file exists
    if not Path(preds_file).exists():
        print(f"Error: Predictions file not found: {preds_file}")
        print("Please specify a valid model with --model or provide a file path with --preds")
        return False
    
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        return False
    
    # Create required directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("figs/fairness", exist_ok=True)
    os.makedirs("figs/fairness_v2", exist_ok=True)
    os.makedirs("figs/intersectional", exist_ok=True)
    
    # 1. Count and visualize demographic distribution
    if not args.skip_counts:
        success = run_command(
            f"python scripts/count_people.py --data {args.data}",
            "Counting data distribution across demographic groups",
            args.verbose
        )
        if not success:
            print("⚠️ Warning: Demographic counting failed")
        
        success = run_command(
            f"python scripts/count_people_viz.py",
            "Visualizing demographic distribution",
            args.verbose
        )
        if not success:
            print("⚠️ Warning: Demographic visualization failed")
    
    # 2. Run fairness audit
    success = run_command(
        f"python scripts/audit_fairness_v2.py --preds {preds_file} --val {args.data} --thr {args.threshold}",
        "Running fairness audit",
        args.verbose
    )
    if not success:
        print("❌ Error: Fairness audit failed")
        return False
    
    # 3. Intersectional fairness analysis
    if not args.skip_intersectional:
        success = run_command(
            f"python scripts/intersectional_fairness.py --preds {preds_file} --val {args.data} --thr {args.threshold}",
            "Analyzing intersectional fairness",
            args.verbose
        )
        if not success:
            print("⚠️ Warning: Intersectional fairness analysis failed")
    
    # 4. Check compliance
    success = run_command(
        "python scripts/check_compliance.py",
        "Checking compliance with fairness requirements",
        args.verbose
    )
    if not success:
        print("⚠️ Warning: Compliance check failed")
    
    # 5. Launch dashboard (if not skipped)
    if not args.skip_dashboard:
        print("\n✅ Analysis pipeline completed successfully!")
        print(f"\n{'='*80}")
        print("🔹 Launching Fairness Dashboard")
        print(f"{'='*80}")
        print("Press Ctrl+C to exit the dashboard when done")
        
        try:
            subprocess.run(["streamlit", "run", "scripts/fairness_dashboard.py"])
        except KeyboardInterrupt:
            print("\nDashboard closed.")
        except Exception as e:
            print(f"\n⚠️ Warning: Could not launch dashboard: {e}")
            print("Try running manually with: streamlit run scripts/fairness_dashboard.py")
    else:
        print("\n✅ Analysis pipeline completed successfully!")
        print("\nTo view the interactive dashboard, run:")
        print("  streamlit run scripts/fairness_dashboard.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 