# PowerShell script to run SHarP individual fairness analysis
# Usage: .\pipelines\run_individual_fairness.ps1 [--sample 500]

# Get command line arguments
param (
    [int]$Sample = 2000,
    [switch]$NoSaveShap = $false
)

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
$SAMPLE_DATA = "output/data/sample_val.csv"
$OUTPUT_DIR = "output/explainers"

# Check if model checkpoint exists
if (-not (Test-Path $MODEL_CHECKPOINT)) {
    Write-Host "ERROR: Model checkpoint not found at $MODEL_CHECKPOINT"
    Write-Host "Please run the turbo pipeline first to generate the model checkpoint."
    exit 1
}

# Check if sample data exists
if (-not (Test-Path $SAMPLE_DATA)) {
    Write-Host "WARNING: Sample validation data not found at $SAMPLE_DATA"
    Write-Host "Trying to create it from valid.csv..."
    
    # Create the merged data
    if (-not (Test-Path "output/data/merged_val.csv")) {
        Write-Host "Creating merged validation data..."
        
        # Use simplest_preds.csv if available, otherwise fail
        if (Test-Path "output/preds/simplest_preds.csv") {
            python scripts/merge_preds_with_labels.py --preds output/preds/simplest_preds.csv --labels data/valid.csv --out output/data/merged_val.csv
        } else {
            Write-Host "ERROR: No prediction file found. Please run the turbo pipeline first."
            exit 1
        }
    }
    
    # Create the sample data
    Write-Host "Creating sample validation data..."
    python -c "import pandas as pd; df = pd.read_csv('output/data/merged_val.csv'); df.sample(2000, random_state=42).to_csv('output/data/sample_val.csv', index=False)"
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_DIR)) {
    Write-Host "Creating output directory $OUTPUT_DIR"
    New-Item -Path $OUTPUT_DIR -ItemType Directory -Force | Out-Null
}

# Run SHarP analysis
Write-Host "Running SHarP analysis..."
Write-Host "Model checkpoint: $MODEL_CHECKPOINT"
Write-Host "Sample data: $SAMPLE_DATA"
Write-Host "Sample size: $Sample"
Write-Host "Output directory: $OUTPUT_DIR"

# Build command arguments
$CMD_ARGS = "--ckpt `"$MODEL_CHECKPOINT`" --data `"$SAMPLE_DATA`" --output-dir `"$OUTPUT_DIR`" --sample $Sample"
if ($NoSaveShap) {
    $CMD_ARGS += " --no-save-shap"
}

# Run the command
Invoke-Expression "python scripts/run_individual_fairness.py $CMD_ARGS"

# Check if analysis completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host "SHarP analysis complete!"
    
    # Check if outputs exist
    $CSV_FILE = Join-Path $OUTPUT_DIR "sharp_scores_distilbert.csv"
    $PNG_FILE = Join-Path $OUTPUT_DIR "sharp_divergence_distilbert.png"
    
    if (Test-Path $CSV_FILE) {
        Write-Host "SHarP scores saved to $CSV_FILE"
        Write-Host "Top identity groups by divergence:"
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