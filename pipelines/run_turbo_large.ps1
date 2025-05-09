# PowerShell script to run the turbo mode pipeline with larger data sample
# Usage: .\run_turbo_large.ps1

# Activate virtual environment
Write-Host "🔵 Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Set environment variables
$env:HF_HOME = "$HOME\.cache\huggingface"

# Show current pipeline mode
Write-Host "🚀 RUNNING ENHANCED TURBO MODE PIPELINE: Larger sample for improved model quality"
Write-Host "⚡ Using 8% of data with progress bars, estimated time ~10 mins"

# Create output directories
$outputDir = "output/large_run"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}
$predsDir = "$outputDir/preds"
if (-not (Test-Path $predsDir)) {
    New-Item -ItemType Directory -Path $predsDir -Force
}
$checkpointsDir = "$outputDir/checkpoints"
if (-not (Test-Path $checkpointsDir)) {
    New-Item -ItemType Directory -Path $checkpointsDir -Force
}

# Track if steps succeed
$distilbertSuccess = $false
$lstmSuccess = $false

# Run BERT training
Write-Host "`n🔵 [1/3] Training DistilBERT model with larger sample..."
try {
    python -m src.train --model bert_headtail --config configs/bert_headtail_turbo_large.yaml --fp16 --turbo
    
    # Check if checkpoint was created
    if (Test-Path "$checkpointsDir/distilbert_headtail_fold0.pth") {
        $distilbertSuccess = $true
        Write-Host "✅ DistilBERT model trained successfully."
    } else {
        Write-Host "⚠️ DistilBERT training completed but checkpoint not found."
    }
} catch {
    Write-Host "❌ Error during DistilBERT training: $_"
}

# Run LSTM training
Write-Host "`n🔵 [2/3] Training LSTM-Capsule model with larger sample..."
try {
    python -m src.train --model lstm_caps --config configs/lstm_caps_turbo_large.yaml --turbo
    
    # Check if checkpoint was created
    if (Test-Path "$checkpointsDir/lstm_caps_fold0.pth") {
        $lstmSuccess = $true
        Write-Host "✅ LSTM model trained successfully."
    } else {
        Write-Host "⚠️ LSTM training completed but checkpoint not found."
    }
} catch {
    Write-Host "❌ Error during LSTM training: $_"
}

# Run model blending if at least one model trained successfully
Write-Host "`n🔵 [3/3] Blending models..."
if ($distilbertSuccess -or $lstmSuccess) {
    try {
        python -m src.blend_optuna --pred-dir $predsDir --ground-truth data/valid.csv --n-trials 10 --out-csv "$predsDir/blend_large.csv"
        Write-Host "✅ Model blending completed successfully."
    } catch {
        Write-Host "❌ Error during model blending: $_"
    }
} else {
    Write-Host "⚠️ Skipping model blending because no models were successfully trained."
}

# Display summary
Write-Host "`n✅ ENHANCED TURBO RUN COMPLETE!"
Write-Host "Results and model files stored in: $outputDir"

# List available files in the output directory
Write-Host "`nAvailable output files:"
Get-ChildItem -Recurse $outputDir -File | ForEach-Object {
    Write-Host "- $_"
} 