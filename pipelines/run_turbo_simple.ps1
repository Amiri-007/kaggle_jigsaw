# PowerShell script to run the turbo mode pipeline (simplified version)
# Usage: .\run_turbo_simple.ps1

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Set environment variables
$env:HF_HOME = "$HOME\.cache\huggingface"

# Show current pipeline mode
Write-Host "RUNNING TURBO MODE PIPELINE: Ultra-fast optimization for development"
Write-Host "Using 5% of data with progress bars, estimated time ~5 mins"

# Run BERT training
Write-Host "Training DistilBERT model..."
python -m src.train --model bert_headtail --config configs/bert_headtail_turbo.yaml --fp16 --turbo

# Run LSTM training
Write-Host "Training LSTM-Capsule model..."
python -m src.train --model lstm_caps --config configs/lstm_caps_turbo.yaml --turbo

# Run GPT-2 training
Write-Host "Training GPT-2 model..."
python -m src.train --model gpt2_headtail --config configs/gpt2_headtail_turbo.yaml --fp16 --turbo

# Run model blending
Write-Host "Blending models with Optuna (10 trials)..."
python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 10 --out-csv output/preds/blend_turbo.csv

# Run metrics calculation
Write-Host "Calculating fairness metrics..."
python scripts/write_metrics.py --pred output/preds/blend_turbo.csv --model-name blend_turbo

# Generate figures
Write-Host "Generating figures..."
jupytext --to notebook notebooks/04_generate_figures.py -o notebooks/tmp_figs.ipynb
jupyter nbconvert --execute notebooks/tmp_figs.ipynb --to html --output figs_run.html --ExecutePreprocessor.timeout=60

# Display summary
Write-Host "TURBO RUN COMPLETE! Results:"
if (Test-Path "results\summary.tsv") {
    Get-Content "results\summary.tsv"
} else {
    Write-Host "Error: summary.tsv not found"
}

Write-Host "Figures generated in 'figs' directory."
Get-ChildItem "figs" | Select-Object -First 5 