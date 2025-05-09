# PowerShell script to generate predictions using existing model checkpoint
# Usage: .\run_predictions_only.ps1

# Activate virtual environment
Write-Host "🔵 Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Set environment variables
$env:HF_HOME = "$HOME\.cache\huggingface"

# Define paths
$modelPath = "output/checkpoints/distilbert_headtail_fold0.pth"
$outputDir = "output/large_predictions"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}

# Display info
Write-Host "🚀 GENERATING PREDICTIONS: Using existing model checkpoint"
Write-Host "⚡ Model path: $modelPath"
Write-Host "⚡ Output directory: $outputDir"

# Run prediction
Write-Host "`n🔵 Generating predictions with existing model..."
python -m src.predict --model-checkpoint $modelPath --test-file data/valid.csv --output-csv "$outputDir/large_predictions.csv"

# Check if predictions were created
if (Test-Path "$outputDir/large_predictions.csv") {
    Write-Host "✅ Predictions generated successfully."
    
    # Calculate metrics on the predictions
    Write-Host "`n🔵 Calculating metrics on predictions..."
    python scripts/write_metrics.py --pred "$outputDir/large_predictions.csv" --model-name "large_model"
    
    # Copy model checkpoint for reference
    Write-Host "`n🔵 Copying model checkpoint for reference..."
    Copy-Item $modelPath "$outputDir/distilbert_headtail_large.pth"
    
    # Display summary
    Write-Host "`n✅ PREDICTION GENERATION COMPLETE!"
    Write-Host "Results and files stored in: $outputDir"
    
    # List available files
    Write-Host "`nGenerated files:"
    Get-ChildItem -Recurse $outputDir | ForEach-Object {
        Write-Host "- $_"
    }
} else {
    Write-Host "❌ Failed to generate predictions."
} 