# PowerShell script to run custom prediction with the existing model
# Usage: .\run_large_predict.ps1

# Activate virtual environment
Write-Host "🔵 Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Define paths
$modelPath = "output/checkpoints/distilbert_headtail_fold0.pth"
$outputDir = "output/large_predictions"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}
$outputCsv = "$outputDir/large_predictions.csv"

# Display info
Write-Host "🚀 GENERATING LARGE DATASET PREDICTIONS: Using existing model"
Write-Host "⚡ Model checkpoint: $modelPath"
Write-Host "⚡ Output CSV: $outputCsv"

# Run custom prediction script
Write-Host "`n🔵 Generating predictions..."
python run_custom_predict.py --checkpoint $modelPath --data-file data/valid.csv --output-csv $outputCsv --batch-size 16

# Check if predictions were created
if (Test-Path $outputCsv) {
    Write-Host "✅ Predictions generated successfully!"
    
    # Calculate metrics
    Write-Host "`n🔵 Calculating metrics..."
    python scripts/write_metrics.py --pred "$outputCsv" --model-name "large_model"
    
    # Copy checkpoint for reference
    Write-Host "`n🔵 Copying model checkpoint for reference..."
    Copy-Item $modelPath "$outputDir/distilbert_headtail_large.pth"
    
    # Display summary
    Write-Host "`n✅ LARGE PREDICTION GENERATION COMPLETE!"
    Write-Host "Results stored in: $outputDir"
    
    # List available files
    Write-Host "`nGenerated files:"
    Get-ChildItem -Path $outputDir -Recurse | ForEach-Object {
        Write-Host "- $_"
    }
} else {
    Write-Host "❌ Failed to generate predictions. Check for errors above."
} 