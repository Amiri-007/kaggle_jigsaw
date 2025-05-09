# PowerShell script to run simplified SHAP analysis on the turbo model
# Usage: .\pipelines\run_turbo_shap_simple.ps1

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Define paths
$MODEL_CHECKPOINT = "output/checkpoints/distilbert_headtail_fold0.pth"
$VALID_DATA = "data/valid.csv"
$OUTPUT_DIR = "output/turbo_shap_simple"

# Check if the model checkpoint exists
if (-not (Test-Path $MODEL_CHECKPOINT)) {
    Write-Host "Model checkpoint not found at $MODEL_CHECKPOINT"
    Write-Host "Please run the turbo model pipeline first: .\pipelines\run_turbo.ps1"
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_DIR)) {
    Write-Host "Creating output directory $OUTPUT_DIR"
    New-Item -Path $OUTPUT_DIR -ItemType Directory -Force | Out-Null
}

# Run simplified SHAP analysis
Write-Host "Running simplified token importance analysis on turbo model..."
Write-Host "Model: $MODEL_CHECKPOINT"
Write-Host "Data: $VALID_DATA"
Write-Host "Output: $OUTPUT_DIR"

python explainers/run_turbo_shap_simple.py --ckpt $MODEL_CHECKPOINT --valid-csv $VALID_DATA --out-dir $OUTPUT_DIR --sample 5

# Check if analysis completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host "Analysis complete!"
    Write-Host "Results saved to $OUTPUT_DIR"
    
    # Display generated files
    Write-Host "Generated files:"
    Get-ChildItem $OUTPUT_DIR | Select-Object -First 10 | ForEach-Object {
        Write-Host "  - $($_.Name)"
    }
    
    # Open output directory
    Write-Host "Opening output directory..."
    Invoke-Item $OUTPUT_DIR
} else {
    Write-Host "Analysis failed with exit code $LASTEXITCODE"
} 