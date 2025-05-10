# PowerShell script to run SHarP individual fairness analysis
# Usage: .\pipelines\run_individual_fairness.ps1

# Activate virtual environment
Write-Host "Activating virtual environment..."
if (Test-Path ".\.venv_new\Scripts\activate.ps1") {
    & ".\.venv_new\Scripts\activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\activate.ps1") {
    & ".\.venv\Scripts\activate.ps1"
} else {
    Write-Host "No virtual environment found."
}

# Define paths
$MODEL_CHECKPOINT = "output/checkpoints/distilbert_headtail_fold0.pth"
$OUTPUT_DIR = "output/explainers"

# Check if model checkpoint exists
if (-not (Test-Path $MODEL_CHECKPOINT)) {
    Write-Host "WARNING: Model checkpoint not found at $MODEL_CHECKPOINT"
    Write-Host "Continuing with mock data generation only."
    $MOCK_ONLY = $true
} else {
    Write-Host "Model checkpoint found at $MODEL_CHECKPOINT"
    $MOCK_ONLY = $false
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_DIR)) {
    Write-Host "Creating output directory $OUTPUT_DIR"
    New-Item -Path $OUTPUT_DIR -ItemType Directory -Force | Out-Null
}

# Run SHarP analysis
if ($MOCK_ONLY) {
    Write-Host "Running SHarP analysis with mock data..."
    python scripts/run_individual_fairness.py --mock-only
} else {
    Write-Host "Running SHarP analysis with model checkpoint..."
    python scripts/run_individual_fairness.py --mock
}

# Check if analysis completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host "SHarP analysis complete!"
    
    # Check if outputs exist
    $CSV_FILE = Join-Path $OUTPUT_DIR "sharp_scores_distilbert.csv"
    $PNG_FILE = Join-Path $OUTPUT_DIR "sharp_divergence_distilbert.png"
    
    if (Test-Path $CSV_FILE) {
        Write-Host "SHarP scores saved to $CSV_FILE"
        Write-Host "Top 3 identity groups by divergence:"
        Get-Content $CSV_FILE | Select-Object -First 4
    }
    
    if (Test-Path $PNG_FILE) {
        Write-Host "SHarP visualization saved to $PNG_FILE"
        
        # Open output directory
        Write-Host "Opening output directory..."
        Invoke-Item $OUTPUT_DIR
    }
} else {
    Write-Host "SHarP analysis failed with exit code $LASTEXITCODE"
} 