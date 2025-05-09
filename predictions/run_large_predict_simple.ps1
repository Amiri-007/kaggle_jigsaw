# PowerShell script to run simplified prediction with the existing model
# Usage: .\run_large_predict_simple.ps1

# Activate virtual environment
Write-Host "🔵 Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Run prediction
Write-Host "🚀 GENERATING PREDICTIONS: Using existing model with simplified script"
python large_predict.py

# Check if predictions were created
$outputPath = "output/large_predictions/predictions.csv"
if (Test-Path $outputPath) {
    Write-Host "✅ Predictions generated successfully!"
    
    # Calculate metrics
    Write-Host "`n🔵 Calculating metrics..."
    python scripts/write_metrics.py --pred "$outputPath" --model-name "large_simplified"
    
    # Display summary
    Write-Host "`n✅ PREDICTION GENERATION COMPLETE!"
    Write-Host "Results stored in: output/large_predictions"
    
    # List available files
    Write-Host "`nGenerated files:"
    Get-ChildItem -Path "output/large_predictions" -Recurse | ForEach-Object {
        Write-Host "- $_"
    }
} else {
    Write-Host "❌ Failed to generate predictions. Check for errors above."
} 