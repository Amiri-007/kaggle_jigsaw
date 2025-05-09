# PowerShell script to run attribution analysis
# Usage: .\run_attribution_analysis.ps1

# Activate virtual environment
Write-Host "🔵 Activating virtual environment..."
& ".\.venv_new\Scripts\activate.ps1"

# Show current mode
Write-Host "🔍 RUNNING ATTRIBUTION ANALYSIS: Analyzing model predictions with token attribution"

# Create output directories
$outputDir = "output/analysis_large"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}

# Run attribution generation with a larger sample
Write-Host "`n🔵 [1/2] Generating token attributions with a larger sample..."
python generate_mock_shap.py --ckpt "output/checkpoints/distilbert_headtail_fold0.pth" --out-dir "$outputDir" --sample 50

# Run SHAP fairness analysis
Write-Host "`n🔵 [2/2] Running SHAP fairness analysis on generated attributions..."
python fairness/shap_report.py --shap-dir "$outputDir" --data data/valid.csv --out-dir "$outputDir/fairness"

# Display summary
Write-Host "`n✅ ATTRIBUTION ANALYSIS COMPLETE!"
Write-Host "Results stored in: $outputDir"

# List available files
Write-Host "`nGenerated files:"
Get-ChildItem -Recurse $outputDir -File | ForEach-Object {
    Write-Host "- $_"
} 